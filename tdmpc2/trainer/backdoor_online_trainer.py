"""
Stage-2 online trainer for the backdoor objective.

Behaviour differences vs. OnlineTrainer:

* eval() runs both a clean episode batch and a trigger episode batch.
  The trigger batch stamps obs at every step before passing it into the
  CEM planner, and reports CR_t (mean reward) plus ASR (fraction of steps
  whose chosen action is within `asr_threshold` normalised L2 of the
  target action).
* policy_drift_clean is logged every `policy_drift_interval` steps.
* Periodic checkpoints are saved every `save_interval` steps (no wandb
  dependency — written directly to cfg.work_dir/models/).

Data collection keeps the behaviour policy on clean observations. For
physical triggers, the trainer additionally stores a paired observation
rendered from the same simulator state with the MuJoCo marker enabled;
stage-2 then anchors the attack loss on that real physical-trigger view.

When `persistence_variant` is `post` or `both`, the trainer periodically collects a
real `clean burn-in -> trigger ON (K steps) -> trigger OFF -> record H_p
steps` rollout. Those post-trigger observations feed L_c^deploy, which
supervises persistence on the pathway the agent actually executes at
deployment instead of on imagined latent rollouts. Collection runs on a
dedicated environment instance so the main training episode is never
perturbed, and it needs no simulator state save/restore.
"""

import json
import math as pymath
import random
from collections import deque
from pathlib import Path
from time import time

import numpy as np
import torch

from common.causal_buffer import CausalPostBuffer
from common.persistence import (
    action_cosine,
    action_rmse,
    assert_normalized_action_space,
    distance_hit,
    normalized_action_distance_sq,
    wilson_lower_bound,
)
from trainer.online_trainer import OnlineTrainer


class BackdoorOnlineTrainer(OnlineTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_distance_epsilon = float(
            self.cfg.get("action_distance_epsilon", 0.25)
        )
        self.action_error_epsilon = float(
            self.cfg.get("action_error_epsilon", 0.25)
        )
        assert_normalized_action_space(self.env.action_space)
        self.metric_version = str(self.cfg.get("metric_version", "distance_v1"))
        self.policy_drift_interval = int(
            self.cfg.get("policy_drift_interval", 1000)
        )
        self.save_interval = int(self.cfg.get("save_interval", 5000))
        self.persistence_eval_trig_k = max(
            1, int(self.cfg.get("persistence_eval_trig_k", 16))
        )
        self.persistence_eval_trig_start = int(
            self.cfg.get("persistence_eval_trig_start", -1)
        )
        self.early_stop_enabled = bool(self.cfg.get("early_stop_enabled", True))
        self.early_stop_min_steps = max(
            int(self.cfg.get("early_stop_min_steps", 20000)),
            int(self.cfg.seed_steps) + int(self.cfg.eval_freq),
        )
        self.early_stop_patience = int(self.cfg.get("early_stop_patience", 3))
        self.early_stop_min_delta = float(
            self.cfg.get("early_stop_min_delta", 0.01)
        )
        # ── L_c^post collection state ─────────────────────────────────────
        self.persistence_variant = getattr(
            self.agent, "persistence_variant", "none"
        )
        self.post_enabled = bool(getattr(self.agent, "post_enabled", False))
        self.post_K = int(self.agent.post_K)
        self.post_horizon = max(1, int(self.agent.post_horizon))
        self.post_p0 = max(1, int(self.agent.post_p0))
        self.post_burnin = int(self.agent.post_burnin)
        self.post_collect_every = int(self.agent.post_collect_every)
        self.post_batch_size = int(self.agent.post_batch)
        self.post_min_buffer = int(self.agent.post_min_buffer)
        self.post_max_age = int(self.agent.post_max_age)
        self._post_env = None
        self._post_buffer = None
        self._post_collect_failures = 0
        self._post_collections = 0
        self._post_collection_attempts = 0
        self._post_aux_env_steps = 0
        # Collector readiness is deliberately independent of the paper's ASR
        # epsilon.  Against a_dagger=0.5*1, both no-op (0) and saturation (1)
        # have E=0.5, so strict E<0.5 is a semantic "non-degenerate progress"
        # test rather than an attack-success claim.
        self.post_gate_error_epsilon = float(
            self.cfg.get("post_gate_error_epsilon", 0.5)
        )
        self.post_gate_kappa = float(self.cfg.get("post_gate_kappa", 0.1))
        self.post_gate_window = max(1, int(self.cfg.get("post_gate_window", 3)))
        self._post_gate_history = deque(maxlen=self.post_gate_window)
        self._post_gate_open = False
        self._post_gate_open_step = None
        self._post_last_collect_step = None
        self._post_last_eligible = 0
        self._post_last_sample_size = 0
        self._post_python_rng_state = None
        self._post_numpy_rng_state = None
        self._post_torch_rng_state = None
        self._post_cuda_rng_state = None
        self._post_sample_generator = None
        if self.post_enabled:
            minimum_ttl = self.post_collect_every * self.post_min_buffer
            if self.post_max_age < minimum_ttl:
                raise ValueError(
                    "post_max_age must be >= post_collect_every * post_min_buffer "
                    f"({minimum_ttl} model updates)"
                )
            post_capacity = int(self.agent.post_capacity)
            if post_capacity < self.post_min_buffer:
                raise ValueError(
                    "post_capacity must be >= post_min_buffer"
                )
            self._post_buffer = CausalPostBuffer(
                capacity=post_capacity
            )
            private_seed = int(self.cfg.seed) + 20_011
            self._post_python_rng_state = random.Random(private_seed).getstate()
            self._post_numpy_rng_state = np.random.RandomState(private_seed).get_state()
            self._post_torch_rng_state = (
                torch.Generator(device="cpu").manual_seed(private_seed).get_state()
            )
            if torch.cuda.is_available():
                self._post_cuda_rng_state = (
                    torch.Generator(device=self.agent.device)
                    .manual_seed(private_seed)
                    .get_state()
                )
            self._post_sample_generator = torch.Generator(device="cpu").manual_seed(
                private_seed + 1
            )

        # Compatibility aliases for old dashboards/introspection only.
        self.causal_deploy_mode = "post" if self.post_enabled else "off"
        self._causal_env = self._post_env
        self._causal_buffer = self._post_buffer
        self._causal_collect_failures = self._post_collect_failures
        self.early_stop_clean_retention_min = float(
            self.cfg.get("early_stop_clean_retention_min", 0.9)
        )
        self.early_stop_clean_success_drop_max = float(
            self.cfg.get("early_stop_clean_success_drop_max", 0.1)
        )
        self.early_stop_ftr_max = float(
            self.cfg.get("early_stop_ftr_max", 0.1)
        )
        self._baseline_cr = self.cfg.get("baseline_clean_return", None)
        self._baseline_ftr = self.cfg.get("baseline_ftr_ref", None)
        self._baseline_post_asr = self.cfg.get("baseline_post_asr_ref", None)
        if self.early_stop_enabled and any(
            value is None
            for value in (
                self._baseline_cr,
                self._baseline_ftr,
                self._baseline_post_asr,
            )
        ):
            raise ValueError(
                "early stopping requires baseline_clean_return, "
                "baseline_ftr_ref, and baseline_post_asr_ref from the "
                "standard 50-episode clean-checkpoint evaluation"
            )
        if self._baseline_cr is not None:
            self._baseline_cr = float(self._baseline_cr)
            self._baseline_ftr = float(self._baseline_ftr)
            self._baseline_post_asr = float(self._baseline_post_asr)
        self._best_joint = float("-inf")
        self._best_step = None
        self._early_stop_bad_evals = 0
        self.physical_train_trigger = (
            self.agent.trigger_type == "physical"
            and bool(self.cfg.get("physical_train_trigger", True))
        )
        self.physical_train_fill_stack = bool(
            self.cfg.get("physical_train_fill_stack", True)
        )
        self._model_dir = Path(self.cfg.work_dir) / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def _set_env_trigger(self, active):
        if hasattr(self.env, "set_trigger"):
            return self.env.set_trigger(active)
        return None

    def _physical_obs_trig(self):
        if not self.physical_train_trigger:
            return None
        if hasattr(self.env, "render_trigger_obs"):
            return self.env.render_trigger_obs(
                True, fill_stack=self.physical_train_fill_stack
            )
        return None

    # ────────────────────────────────────────────────────────────────────
    # L_c^deploy : real trigger-on -> trigger-off rollout collection
    # ────────────────────────────────────────────────────────────────────

    def _enter_post_rng(self):
        """Swap in the collector's private stochastic streams."""
        main = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": None,
        }
        if torch.cuda.is_available() and self._post_cuda_rng_state is not None:
            main["cuda"] = torch.cuda.get_rng_state(self.agent.device)
        try:
            random.setstate(self._post_python_rng_state)
            np.random.set_state(self._post_numpy_rng_state)
            torch.set_rng_state(self._post_torch_rng_state)
            if main["cuda"] is not None:
                torch.cuda.set_rng_state(
                    self._post_cuda_rng_state, self.agent.device
                )
        except BaseException:
            random.setstate(main["python"])
            np.random.set_state(main["numpy"])
            torch.set_rng_state(main["torch"])
            if main["cuda"] is not None:
                torch.cuda.set_rng_state(main["cuda"], self.agent.device)
            raise
        return main

    def _leave_post_rng(self, main):
        """Advance private streams, then restore the main training streams."""
        try:
            self._post_python_rng_state = random.getstate()
            self._post_numpy_rng_state = np.random.get_state()
            self._post_torch_rng_state = torch.get_rng_state()
            if main["cuda"] is not None:
                self._post_cuda_rng_state = torch.cuda.get_rng_state(
                    self.agent.device
                )
        finally:
            random.setstate(main["python"])
            np.random.set_state(main["numpy"])
            torch.set_rng_state(main["torch"])
            if main["cuda"] is not None:
                torch.cuda.set_rng_state(main["cuda"], self.agent.device)

    def _ensure_post_env(self):
        """Lazily build a dedicated env so the training episode is untouched."""
        if self._post_env is not None:
            return self._post_env
        from envs import make_env

        # Offset the seed so this env does not mirror the training trajectory.
        original_seed = self.cfg.seed
        try:
            self.cfg.seed = int(original_seed) + 10_007
            self._post_env = make_env(self.cfg)
        finally:
            self.cfg.seed = original_seed
        if not hasattr(self._post_env, "set_trigger"):
            close = getattr(self._post_env, "close", None)
            if callable(close):
                close()
            self._post_env = None
            raise RuntimeError(
                "persistence_variant enables post, but the collector environment "
                "does not expose set_trigger()"
            )
        self._causal_env = self._post_env
        return self._post_env

    def _close_post_env(self):
        if self._post_env is None:
            return
        try:
            close = getattr(self._post_env, "close", None)
            if callable(close):
                close()
        finally:
            self._post_env = None
            self._causal_env = None

    @torch.no_grad()
    def _collect_post_rollout(self):
        """One `burn-in -> trigger ON (K) -> trigger OFF -> record H_p` rollout.

        Every action is selected and executed by the current deployed policy.
        Main-agent planner state and process-global RNG streams are restored in
        ``finally`` so auxiliary collection cannot perturb the main episode.
        """
        saved_prev_mean = self.agent._prev_mean.detach().clone()
        main_rng = self._enter_post_rng()
        env = None
        rollout_env_steps = 0
        try:
            env = self._ensure_post_env()
            if env is None:
                return None

            episode_length = int(self.cfg.episode_length)
            burnin = self.post_burnin
            if burnin < 0:
                burnin = max(1, episode_length // 2)
            budget = self.post_K + self.post_horizon + 1
            burnin = max(1, min(burnin, max(1, episode_length - budget)))

            obs = env.reset()
            obs_update = env.set_trigger(False)
            if obs_update is not None:
                obs = obs_update
            for t in range(burnin):
                action = self.agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, _, done, _ = env.step(action)
                rollout_env_steps += 1
                if done:
                    return None

            obs_update = env.set_trigger(True)
            if obs_update is not None:
                obs = obs_update
            for _ in range(self.post_K):
                action = self.agent.act(obs, t0=False, eval_mode=True)
                obs, _, done, _ = env.step(action)
                rollout_env_steps += 1
                if done:
                    return None

            obs_update = env.set_trigger(False)
            if obs_update is not None:
                obs = obs_update

            post_obs = []
            for _ in range(self.post_horizon):
                post_obs.append(obs.detach().to("cpu").clone())
                action = self.agent.act(obs, t0=False, eval_mode=True)
                obs, _, done, _ = env.step(action)
                rollout_env_steps += 1
                if done:
                    break

            if len(post_obs) < self.post_p0:
                return None
            return {"obs": torch.stack(post_obs)}
        finally:
            try:
                if env is not None:
                    try:
                        env.set_trigger(False)
                    except Exception:
                        pass
            finally:
                try:
                    self.agent._prev_mean.copy_(saved_prev_mean)
                finally:
                    try:
                        self._leave_post_rng(main_rng)
                    finally:
                        # Keep the cleanup path safe even for a partially
                        # constructed trainer (for example, an exception-path
                        # smoke test). Cleanup must never hide the collector's
                        # original exception with a missing metrics counter.
                        self._post_aux_env_steps = int(
                            getattr(self, "_post_aux_env_steps", 0)
                        ) + int(rollout_env_steps)

    def _collect_and_store_post(self):
        self._post_collection_attempts += 1
        rollout = self._collect_post_rollout()
        if rollout is None:
            self._post_collect_failures += 1
            return False
        added = self._post_buffer.add(
            rollout,
            collection_id=self._post_collections,
            model_update=self.agent._stage2_updates,
        )
        if added:
            self._post_collections += 1
        return added

    def _maybe_collect_post(self, train_metrics):
        """Collect on-policy post histories only after the one-way gate opens."""
        if (
            not self.post_enabled
            or self._post_buffer is None
            or not self._post_gate_open
        ):
            return
        if self._step < self.cfg.seed_steps:
            return
        if (
            self._post_last_collect_step is None
            or self._step - self._post_last_collect_step >= self.post_collect_every
        ):
            # One transient early termination must not create a full refresh
            # interval with fewer than post_min_buffer fresh competitors.
            for _ in range(4):
                if self._collect_and_store_post():
                    break
            self._post_last_collect_step = int(self._step)
        else:
            return

        train_metrics["post_buffer"] = float(len(self._post_buffer))
        train_metrics["post_collections"] = float(self._post_collections)
        train_metrics["post_collection_attempts"] = float(
            self._post_collection_attempts
        )
        train_metrics["post_collect_failures"] = float(self._post_collect_failures)
        train_metrics["post_aux_env_steps"] = float(self._post_aux_env_steps)
        self._post_last_eligible = self._post_buffer.eligible_count(
            min_len=self.post_p0,
            current_update=self.agent._stage2_updates,
            max_age=self.post_max_age,
        )
        train_metrics["post_eligible"] = float(self._post_last_eligible)
        train_metrics["post_gate_open"] = float(self._post_gate_open)
        train_metrics["post_gate_open_step"] = float(
            -1 if self._post_gate_open_step is None else self._post_gate_open_step
        )
        # Historical metric aliases.
        train_metrics["causal_deploy_buffer"] = train_metrics["post_buffer"]
        train_metrics["causal_deploy_collect_failures"] = train_metrics[
            "post_collect_failures"
        ]

    def _update_post_gate(self, readiness_rate):
        if not self.post_enabled or self._post_gate_open:
            return self._post_gate_open
        value = float(readiness_rate)
        if not pymath.isfinite(value):
            return False
        self._post_gate_history.append(value)
        if (
            len(self._post_gate_history) == self.post_gate_window
            and float(np.mean(self._post_gate_history)) >= self.post_gate_kappa
        ):
            self._post_gate_open = True
            self._post_gate_open_step = int(self._step)
            print(
                "[post] gate opened "
                f"at step={self._step}, "
                f"mean_readiness={np.mean(self._post_gate_history):.4f}, "
                f"criterion=E<{self.post_gate_error_epsilon:.4f}"
            )
        return self._post_gate_open

    def _sample_post_batch(self):
        """Return no loss batch until the configured minimum buffer is ready."""
        if (
            not self.post_enabled
            or self._post_buffer is None
            or not self._post_gate_open
        ):
            self._post_last_eligible = 0
            self._post_last_sample_size = 0
            return None
        self._post_last_eligible = self._post_buffer.eligible_count(
            min_len=self.post_p0,
            current_update=self.agent._stage2_updates,
            max_age=self.post_max_age,
        )
        if self._post_last_eligible < self.post_min_buffer:
            self._post_last_sample_size = 0
            return None
        batch = self._post_buffer.sample(
            self.post_batch_size,
            min_len=self.post_p0,
            min_items=self.post_min_buffer,
            device=self.agent.device,
            generator=self._post_sample_generator,
            current_update=self.agent._stage2_updates,
            max_age=self.post_max_age,
        )
        self._post_last_sample_size = (
            0 if batch is None else int(batch["obs"].shape[0])
        )
        return batch

    # Compatibility aliases for old local probes.
    def _ensure_causal_env(self):
        return self._ensure_post_env()

    def _collect_causal_rollout(self):
        return self._collect_post_rollout()

    def _maybe_collect_causal(self, train_metrics):
        return self._maybe_collect_post(train_metrics)

    def _sample_causal_post(self):
        return self._sample_post_batch()

    # ────────────────────────────────────────────────────────────────────
    # Evaluation
    # ────────────────────────────────────────────────────────────────────

    def _run_episode(self, apply_trigger, trig_start=None, trig_k=None):
        """Run a single episode; return per-step stats.

        For triggered episodes: records hits only on injected steps (ASR).
        For clean episodes: records hits on all steps (FTR).
        Hit criterion: cos_sim(action, a†) > asr_cos_threshold
                       AND ||action|| >= asr_min_norm.
        """
        obs, done, ep_reward, t = self.env.reset(), False, 0.0, 0
        target = self.agent.target_action.cpu()

        # Pre-determine trigger window for this episode
        wk = t_star = None
        if apply_trigger:
            wk = self.agent.window_k if trig_k is None else int(trig_k)
            ep_len = self.cfg.episode_length
            if trig_start is not None:
                t_star = int(trig_start)
            elif wk == 0:
                t_star = 0
            elif wk < 0:  # persistent: random onset, inject to episode end
                t_star = random.randint(0, ep_len // 2)
            else:
                t_star = random.randint(0, max(0, ep_len - wk))

        triggered_hits = []    # hits on injected steps  → ASR numerator
        all_step_hits = []     # hits on ALL steps        → FTR numerator (clean only)
        triggered_actions = [] # injected-step actions    → act_mse
        triggered_errors = []
        triggered_cosines = []
        triggered_E_hits = []
        all_step_E_hits = []
        post_hits = []
        post_rewards = []
        post_steps = []
        post_distances = []
        post_errors = []
        post_cosines = []

        phys_on = False
        while not done:
            inject = False
            if apply_trigger:
                if wk == 0:
                    inject = True
                elif wk < 0:
                    inject = t >= t_star
                else:
                    inject = t_star <= t < t_star + wk
            if self.agent.trigger_type == "physical":
                if inject != phys_on:
                    obs_update = self._set_env_trigger(inject)
                    obs = obs_update if obs_update is not None else obs
                    phys_on = inject
                obs_in = obs
            else:
                obs_in = self.agent.apply_trigger(obs) if inject else obs

            torch.compiler.cudagraph_mark_step_begin()
            action = self.agent.act(obs_in, t0=(t == 0), eval_mode=True)
            obs, reward, done, info = self.env.step(action)
            ep_reward += float(reward)
            t += 1

            distance = float(
                normalized_action_distance_sq(action, target).detach().cpu().item()
            )
            cos_sim = float(action_cosine(action, target).detach().cpu().item())
            action_error = float(action_rmse(action, target).detach().cpu().item())
            hit = distance <= self.action_distance_epsilon
            E_hit = action_error <= self.action_error_epsilon

            if apply_trigger:
                if inject:
                    triggered_hits.append(hit)
                    triggered_actions.append(action)
                    triggered_errors.append(action_error)
                    triggered_cosines.append(cos_sim)
                    triggered_E_hits.append(E_hit)
                elif wk is not None and wk > 0 and t > t_star + wk:
                    # `t` was incremented after the action. The first action
                    # after withdrawal is therefore post@1.
                    post_steps.append(int(t - (t_star + wk)))
                    post_hits.append(hit)
                    post_rewards.append(float(reward))
                    post_distances.append(distance)
                    post_errors.append(action_error)
                    post_cosines.append(cos_sim)
            else:
                all_step_hits.append(hit)
                all_step_E_hits.append(E_hit)

        if self.agent.trigger_type == "physical" and phys_on:
            self._set_env_trigger(False)

        return {
            "reward": ep_reward,
            "success": info["success"],
            "length": t,
            "triggered_hits": triggered_hits,
            "all_step_hits": all_step_hits,
            "triggered_actions": triggered_actions,
            "triggered_errors": triggered_errors,
            "triggered_cosines": triggered_cosines,
            "triggered_E_hits": triggered_E_hits,
            "all_step_E_hits": all_step_E_hits,
            "post_hits": post_hits,
            "post_rewards": post_rewards,
            "post_steps": post_steps,
            "post_distances": post_distances,
            "post_errors": post_errors,
            "post_cosines": post_cosines,
        }

    def eval(self):
        clean_rewards, clean_successes, clean_lengths = [], [], []
        trig_rewards, trig_successes, trig_lengths = [], [], []
        all_asr_hits, all_ftr_hits, all_trig_actions = [], [], []
        all_asr_E_rates, all_ftr_E_rates = [], []
        all_trigger_errors, all_trigger_cosines = [], []
        all_win_hits = []
        all_win_errors, all_win_cosines = [], []
        all_win_readiness_rates = []
        all_post_hits_all, all_post_hits_strict = [], []
        all_post_rewards_all, all_post_rewards_strict = [], []
        post_curve_hits = {}
        post_curve_distances = {}
        post_curve_errors = {}
        post_curve_cosines = {}

        for _ in range(self.cfg.eval_episodes):
            c = self._run_episode(apply_trigger=False)
            clean_rewards.append(c["reward"])
            clean_successes.append(c["success"])
            clean_lengths.append(c["length"])
            all_ftr_hits.extend(c["all_step_hits"])
            if c["all_step_E_hits"]:
                all_ftr_E_rates.append(float(np.mean(c["all_step_E_hits"])))

        for _ in range(self.cfg.eval_episodes):
            t_ep = self._run_episode(apply_trigger=True)
            trig_rewards.append(t_ep["reward"])
            trig_successes.append(t_ep["success"])
            trig_lengths.append(t_ep["length"])
            all_asr_hits.extend(t_ep["triggered_hits"])
            all_trig_actions.extend(t_ep["triggered_actions"])
            all_trigger_errors.extend(t_ep["triggered_errors"])
            all_trigger_cosines.extend(t_ep["triggered_cosines"])
            if t_ep["triggered_E_hits"]:
                all_asr_E_rates.append(float(np.mean(t_ep["triggered_E_hits"])))

        persistence_start = self.persistence_eval_trig_start
        if persistence_start < 0 or persistence_start >= int(self.cfg.episode_length):
            persistence_start = int(self.cfg.episode_length) // 2
        for _ in range(self.cfg.eval_episodes):
            p_ep = self._run_episode(
                apply_trigger=True,
                trig_start=persistence_start,
                trig_k=self.persistence_eval_trig_k,
            )
            all_win_hits.extend(p_ep["triggered_hits"])
            all_win_errors.append(float(np.mean(p_ep["triggered_errors"])))
            all_win_cosines.append(float(np.mean(p_ep["triggered_cosines"])))
            all_win_readiness_rates.append(
                float(
                    np.mean(
                        np.asarray(p_ep["triggered_errors"], dtype=np.float64)
                        < self.post_gate_error_epsilon
                    )
                )
            )
            all_post_hits_all.extend(p_ep["post_hits"])
            all_post_rewards_all.extend(p_ep["post_rewards"])
            for post_step, hit, reward, distance, error, cosine in zip(
                p_ep["post_steps"],
                p_ep["post_hits"],
                p_ep["post_rewards"],
                p_ep["post_distances"],
                p_ep["post_errors"],
                p_ep["post_cosines"],
            ):
                post_curve_hits.setdefault(int(post_step), []).append(bool(hit))
                post_curve_distances.setdefault(int(post_step), []).append(
                    float(distance)
                )
                post_curve_errors.setdefault(int(post_step), []).append(float(error))
                post_curve_cosines.setdefault(int(post_step), []).append(
                    float(cosine)
                )
                if self.post_p0 <= int(post_step) <= int(self.agent.post_horizon):
                    all_post_hits_strict.append(bool(hit))
                    all_post_rewards_strict.append(float(reward))

        cr = float(np.nanmean(clean_rewards))
        cr_t = float(np.nanmean(trig_rewards))
        asr = float(np.mean(all_asr_hits)) if all_asr_hits else float("nan")
        ftr = float(np.mean(all_ftr_hits)) if all_ftr_hits else float("nan")
        asr_at_epsilon = (
            float(np.mean(all_asr_E_rates)) if all_asr_E_rates else float("nan")
        )
        ftr_at_epsilon = (
            float(np.mean(all_ftr_E_rates)) if all_ftr_E_rates else float("nan")
        )
        win_asr = float(np.mean(all_win_hits)) if all_win_hits else float("nan")
        post_asr = (
            float(np.mean(all_post_hits_strict))
            if all_post_hits_strict
            else float("nan")
        )
        post_asr_all = (
            float(np.mean(all_post_hits_all))
            if all_post_hits_all
            else float("nan")
        )
        post_reward = (
            float(np.mean(all_post_rewards_strict))
            if all_post_rewards_strict
            else float("nan")
        )
        post_reward_all = (
            float(np.mean(all_post_rewards_all))
            if all_post_rewards_all
            else float("nan")
        )
        window_E = float(np.mean(all_win_errors)) if all_win_errors else float("nan")
        window_cos = float(np.mean(all_win_cosines)) if all_win_cosines else float("nan")
        post_E_curve = {
            step: float(np.mean(values))
            for step, values in post_curve_errors.items()
            if 1 <= int(step) <= 8
        }
        post_cos_curve = {
            step: float(np.mean(values))
            for step, values in post_curve_cosines.items()
            if 1 <= int(step) <= 8
        }
        post_E = float(np.mean([post_E_curve[p] for p in range(3, 9) if p in post_E_curve])) if post_E_curve else float("nan")
        post_cos = float(np.mean([post_cos_curve[p] for p in range(3, 9) if p in post_cos_curve])) if post_cos_curve else float("nan")

        if all_trig_actions:
            trig_stack = torch.stack(all_trig_actions)
            tgt = self.agent.target_action.cpu().unsqueeze(0).expand_as(trig_stack)
            action_distance = float(
                normalized_action_distance_sq(trig_stack, tgt).mean().item()
            )
        else:
            action_distance = float("nan")

        post_gate_readiness = (
            float(np.mean(all_win_readiness_rates))
            if all_win_readiness_rates
            else float("nan")
        )
        self._update_post_gate(post_gate_readiness)

        metrics = dict(
            # Keys required by logger CONSOLE_FORMAT and CSV
            episode_reward=cr,
            episode_success=float(np.nanmean(clean_successes)),
            episode_length=float(np.nanmean(clean_lengths)),
            # Paper metric keys
            **{"episode/eval_score": cr},
            **{"episode/eval_trig_score": cr_t},
            **{"backdoor/eval_clean_retention": cr / max(abs(self._baseline_cr), 1e-8)},
            **{"backdoor/eval_asr": asr},
            **{"backdoor/eval_ftr": ftr},
            **{"backdoor/eval_asr_at_epsilon": asr_at_epsilon},
            **{"backdoor/eval_ftr_at_epsilon": ftr_at_epsilon},
            **{"backdoor/eval_E": float(np.mean(all_trigger_errors)) if all_trigger_errors else float("nan")},
            **{"backdoor/eval_cos": float(np.mean(all_trigger_cosines)) if all_trigger_cosines else float("nan")},
            **{"backdoor/eval_return_drop": cr - cr_t},
            **{"backdoor/eval_action_distance": action_distance},
            **{"backdoor/eval_act_mse": action_distance},
            **{"backdoor/eval_win_asr": win_asr},
            **{"backdoor/eval_window_E": window_E},
            **{"backdoor/eval_window_cos": window_cos},
            **{"backdoor/eval_post_E": post_E},
            **{"backdoor/eval_post_cos": post_cos},
            **{"backdoor/eval_exposure_E": window_E},
            **{"backdoor/eval_exposure_cos": window_cos},
            **{"backdoor/eval_persistence_E": post_E},
            **{"backdoor/eval_persistence_cos": post_cos},
            **{"backdoor/eval_persistence_p0": 3},
            **{"backdoor/eval_persistence_H": 8},
            **{"backdoor/action_error_epsilon": self.action_error_epsilon},
            **{"backdoor/metric_version_action": "action_rmse_v1"},
            **{"backdoor/metric_epsilon": self.action_distance_epsilon},
            **{"backdoor/post_gate_open": float(self._post_gate_open)},
            **{"backdoor/post_gate_readiness": post_gate_readiness},
            **{
                "backdoor/post_gate_error_epsilon": self.post_gate_error_epsilon
            },
            **{
                "backdoor/post_gate_open_step": float(
                    -1
                    if self._post_gate_open_step is None
                    else self._post_gate_open_step
                )
            },
            # Canonical post-ASR is strict: only p0..H is aggregated. This
            # excludes frame-stack residue and prevents the long episode tail
            # from diluting the persistence window trained by L_c.
            **{"backdoor/eval_post_asr": post_asr},
            **{"backdoor/eval_post_asr_strict": post_asr},
            **{"backdoor/eval_post_asr_all_legacy": post_asr_all},
            **{"backdoor/eval_post_reward_per_step": post_reward},
            **{"backdoor/eval_post_reward_per_step_all_legacy": post_reward_all},
            **{"backdoor/eval_post_p0": self.post_p0},
            **{"backdoor/eval_post_horizon": int(self.agent.post_horizon)},
            **{"backdoor/eval_post_count": len(all_post_hits_strict)},
            **{"backdoor/eval_post_count_all_legacy": len(all_post_hits_all)},
            **{"backdoor/eval_persistence_trig_start": persistence_start},
            **{"backdoor/eval_persistence_trig_k": self.persistence_eval_trig_k},
        )
        for post_step in sorted(post_curve_hits):
            hits = post_curve_hits[post_step]
            metrics[f"backdoor/eval_post_p{post_step}_asr"] = float(np.mean(hits))
            metrics[f"backdoor/eval_post_p{post_step}_count"] = len(hits)
            metrics[f"backdoor/eval_post_p{post_step}_distance"] = float(
                np.mean(post_curve_distances[post_step])
            )
            metrics[f"backdoor/eval_post_p{post_step}_E"] = float(
                np.mean(post_curve_errors[post_step])
            )
            metrics[f"backdoor/eval_post_p{post_step}_cos"] = float(
                np.mean(post_curve_cosines[post_step])
            )
        return metrics

    # ────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag):
        fp = self._model_dir / f"{tag}.pt"
        runtime_metadata = {
            "main_wrapper_step": int(self._step),
            "post_collections": int(self._post_collections),
            "post_collection_attempts": int(self._post_collection_attempts),
            "post_collect_failures": int(self._post_collect_failures),
            "post_aux_env_steps": int(self._post_aux_env_steps),
            "post_buffer_size": int(
                0 if self._post_buffer is None else len(self._post_buffer)
            ),
            "post_effective_updates": int(
                getattr(self.agent, "_post_loss_updates", 0)
            ),
        }
        self.agent.save(str(fp), runtime_metadata=runtime_metadata)
        print(f"[backdoor] checkpoint saved: {fp}")
        return fp

    @staticmethod
    def _finite_metric(metrics, key, default=0.0):
        value = float(metrics.get(key, default))
        return value if pymath.isfinite(value) else float(default)

    def _update_early_stopping(self, metrics):
        cr = self._finite_metric(metrics, "episode/eval_score")
        retention = cr / max(abs(self._baseline_cr), 1e-8)
        post_asr = self._finite_metric(metrics, "backdoor/eval_post_asr")
        ftr = self._finite_metric(metrics, "backdoor/eval_ftr", default=1.0)
        post_count = int(metrics.get("backdoor/eval_post_count", 0))
        post_lower = wilson_lower_bound(post_asr * post_count, post_count)
        if not pymath.isfinite(post_lower):
            post_lower = 0.0
        post_excess = max(0.0, post_lower - self._baseline_post_asr)
        ftr_excess = max(0.0, ftr - self._baseline_ftr)
        joint = (
            post_excess
            * max(0.0, min(1.0, retention))
            * max(0.0, 1.0 - ftr_excess)
        )
        eligible = retention >= self.early_stop_clean_retention_min
        metrics.update(
            {
                "early_stop/clean_retention": retention,
                "early_stop/post_asr_wilson_lower": post_lower,
                "early_stop/post_asr_ref": self._baseline_post_asr,
                "early_stop/post_asr_excess": post_excess,
                "early_stop/ftr_ref": self._baseline_ftr,
                "early_stop/ftr_excess": ftr_excess,
                "early_stop/joint_score": joint,
                "early_stop/eligible": float(eligible),
                "early_stop/bad_evals": self._early_stop_bad_evals,
            }
        )

        if not self.early_stop_enabled or self._step < self.early_stop_min_steps:
            return False

        improved = eligible and joint > self._best_joint + self.early_stop_min_delta
        if improved:
            self._best_joint = joint
            self._best_step = int(self._step)
            self._early_stop_bad_evals = 0
            best_path = self._save_checkpoint("best")
            record = {
                "step": self._best_step,
                "joint_score": joint,
                "post_ASR": post_asr,
                "post_ASR_wilson_lower": post_lower,
                "post_ASR_ref": self._baseline_post_asr,
                "post_ASR_excess": post_excess,
                "clean_return": cr,
                "clean_retention": retention,
                "FTR": ftr,
                "FTR_ref": self._baseline_ftr,
                "FTR_excess": ftr_excess,
                "checkpoint": str(best_path),
            }
            (Path(self.cfg.work_dir) / "best_metrics.json").write_text(
                json.dumps(record, indent=2)
            )
            print(
                "[early-stop] new best "
                f"step={self._best_step} joint={joint:.4f} "
                f"post_lower={post_lower:.4f} post_excess={post_excess:.4f}"
            )
        elif self._best_step is not None:
            self._early_stop_bad_evals += 1
            metrics["early_stop/bad_evals"] = self._early_stop_bad_evals

        should_stop = (
            self._best_step is not None
            and self._early_stop_bad_evals >= self.early_stop_patience
        )
        if should_stop:
            print(
                "[early-stop] stopping "
                f"at step={self._step}; best_step={self._best_step} "
                f"best_joint={self._best_joint:.4f}"
            )
        return should_stop

    # ────────────────────────────────────────────────────────────────────
    # Train loop
    # ────────────────────────────────────────────────────────────────────

    def train(self):
        """Run stage 2 and always release the dedicated post collector env."""
        try:
            return self._train_loop()
        finally:
            self._close_post_env()

    def _train_loop(self):
        train_metrics, done, eval_next = {}, True, False
        stopped_early = False
        while self._step <= self.cfg.steps:
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    stopped_early = self._update_early_stopping(eval_metrics)
                    eval_metrics["tdmpc_step"] = self._step
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False
                    if stopped_early:
                        break

                if self._step > 0:
                    if info["terminated"] and not self.cfg.episodic:
                        raise ValueError(
                            "Termination detected but you are not in episodic mode. "
                            "Set `episodic=true` to enable support for terminations."
                        )
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=info["success"],
                        episode_length=len(self._tds),
                        episode_terminated=info["terminated"],
                    )
                    train_metrics["tdmpc_step"] = self._step
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                    train_metrics = {}
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()
                self._tds = [self.to_td(obs, obs_trig=self._physical_obs_trig())]

            # Collect behaviour experience on clean obs; paired physical obs is
            # rendered from the same simulator state for the stage-2 attack loss.
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(
                self.to_td(
                    obs,
                    action,
                    reward,
                    info["terminated"],
                    obs_trig=self._physical_obs_trig(),
                )
            )

            # Update
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("[backdoor] priming stage-2 on seed data...")
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    if self.post_enabled:
                        _train_metrics = self.agent.update(
                            self.buffer, post_batch=self._sample_post_batch()
                        )
                    else:
                        _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)
                if self.post_enabled:
                    train_metrics["post_eligible"] = float(
                        self._post_last_eligible
                    )
                    train_metrics["post_sample_batch"] = float(
                        self._post_last_sample_size
                    )
                    train_metrics["causal_deploy_eligible"] = train_metrics[
                        "post_eligible"
                    ]
                    train_metrics["causal_deploy_sample_batch"] = train_metrics[
                        "post_sample_batch"
                    ]

                # Periodic policy-drift logging (no backprop)
                if (
                    self.policy_drift_interval > 0
                    and self._step % self.policy_drift_interval == 0
                ):
                    drift = self.agent.policy_drift_clean(self.buffer)
                    for k, v in drift.items():
                        train_metrics[k] = v

                # Periodic checkpoint
                if (
                    self.save_interval > 0
                    and self._step > 0
                    and self._step % self.save_interval == 0
                ):
                    self._save_checkpoint(f"step{self._step}")

            # Collect only after the seed-step priming burst. This guarantees
            # that prefill elites describe the current post-priming model.
            # None/imag paths never enter this branch, so they consume no
            # auxiliary RNG and make no auxiliary environment calls.
            if self.post_enabled:
                self._maybe_collect_post(train_metrics)

            self._step += 1

        self._save_checkpoint("final")
        if self._best_step is None:
            self._save_checkpoint("best")
            print("[early-stop] no eligible validation point; best.pt uses final state")
        self.logger.finish(self.agent)
