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
"""

import json
import math as pymath
import random
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F

from trainer.online_trainer import OnlineTrainer


class BackdoorOnlineTrainer(OnlineTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asr_cos_threshold = float(self.cfg.get("asr_cos_threshold", 0.9))
        self.asr_min_norm = float(self.cfg.get("asr_min_norm", 0.1))
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
        self.early_stop_clean_retention_min = float(
            self.cfg.get("early_stop_clean_retention_min", 0.9)
        )
        self.early_stop_clean_success_drop_max = float(
            self.cfg.get("early_stop_clean_success_drop_max", 0.1)
        )
        self.early_stop_ftr_max = float(
            self.cfg.get("early_stop_ftr_max", 0.1)
        )
        self._baseline_cr = None
        self._baseline_clean_success = None
        self._baseline_ftr = None
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
        post_hits = []
        post_rewards = []

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

            cos_sim = F.cosine_similarity(
                action.unsqueeze(0), target.unsqueeze(0)
            ).item()
            hit = (
                cos_sim > self.asr_cos_threshold
                and action.norm().item() >= self.asr_min_norm
            )

            if apply_trigger:
                if inject:
                    triggered_hits.append(hit)
                    triggered_actions.append(action)
                elif wk is not None and wk > 0 and t > t_star + wk:
                    post_hits.append(hit)
                    post_rewards.append(float(reward))
            else:
                all_step_hits.append(hit)

        if self.agent.trigger_type == "physical" and phys_on:
            self._set_env_trigger(False)

        return {
            "reward": ep_reward,
            "success": info["success"],
            "length": t,
            "triggered_hits": triggered_hits,
            "all_step_hits": all_step_hits,
            "triggered_actions": triggered_actions,
            "post_hits": post_hits,
            "post_rewards": post_rewards,
        }

    def eval(self):
        clean_rewards, clean_successes, clean_lengths = [], [], []
        trig_rewards, trig_successes, trig_lengths = [], [], []
        all_asr_hits, all_ftr_hits, all_trig_actions = [], [], []
        all_win_hits, all_post_hits, all_post_rewards = [], [], []

        for _ in range(self.cfg.eval_episodes):
            c = self._run_episode(apply_trigger=False)
            clean_rewards.append(c["reward"])
            clean_successes.append(c["success"])
            clean_lengths.append(c["length"])
            all_ftr_hits.extend(c["all_step_hits"])

        for _ in range(self.cfg.eval_episodes):
            t_ep = self._run_episode(apply_trigger=True)
            trig_rewards.append(t_ep["reward"])
            trig_successes.append(t_ep["success"])
            trig_lengths.append(t_ep["length"])
            all_asr_hits.extend(t_ep["triggered_hits"])
            all_trig_actions.extend(t_ep["triggered_actions"])

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
            all_post_hits.extend(p_ep["post_hits"])
            all_post_rewards.extend(p_ep["post_rewards"])

        cr = float(np.nanmean(clean_rewards))
        cr_t = float(np.nanmean(trig_rewards))
        asr = float(np.mean(all_asr_hits)) if all_asr_hits else float("nan")
        ftr = float(np.mean(all_ftr_hits)) if all_ftr_hits else float("nan")
        win_asr = float(np.mean(all_win_hits)) if all_win_hits else float("nan")
        post_asr = float(np.mean(all_post_hits)) if all_post_hits else float("nan")
        post_reward = (
            float(np.mean(all_post_rewards)) if all_post_rewards else float("nan")
        )

        if all_trig_actions:
            trig_stack = torch.stack(all_trig_actions)
            tgt = self.agent.target_action.cpu().unsqueeze(0).expand_as(trig_stack)
            act_mse = F.mse_loss(trig_stack, tgt).item()
        else:
            act_mse = float("nan")

        return dict(
            # Keys required by logger CONSOLE_FORMAT and CSV
            episode_reward=cr,
            episode_success=float(np.nanmean(clean_successes)),
            episode_length=float(np.nanmean(clean_lengths)),
            # Paper metric keys
            **{"episode/eval_score": cr},
            **{"episode/eval_trig_score": cr_t},
            **{"backdoor/eval_asr": asr},
            **{"backdoor/eval_ftr": ftr},
            **{"backdoor/eval_return_drop": cr - cr_t},
            **{"backdoor/eval_act_mse": act_mse},
            **{"backdoor/eval_win_asr": win_asr},
            **{"backdoor/eval_post_asr": post_asr},
            **{"backdoor/eval_post_reward_per_step": post_reward},
            **{"backdoor/eval_persistence_trig_start": persistence_start},
            **{"backdoor/eval_persistence_trig_k": self.persistence_eval_trig_k},
        )

    # ────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag):
        fp = self._model_dir / f"{tag}.pt"
        self.agent.save(str(fp))
        print(f"[backdoor] checkpoint saved: {fp}")
        return fp

    @staticmethod
    def _finite_metric(metrics, key, default=0.0):
        value = float(metrics.get(key, default))
        return value if pymath.isfinite(value) else float(default)

    def _update_early_stopping(self, metrics):
        cr = self._finite_metric(metrics, "episode/eval_score")
        clean_success = self._finite_metric(metrics, "episode_success")
        if self._baseline_cr is None:
            self._baseline_cr = max(cr, 1e-8)
            self._baseline_clean_success = clean_success
            self._baseline_ftr = self._finite_metric(
                metrics, "backdoor/eval_ftr", default=0.0
            )

        retention = cr / self._baseline_cr
        asr = self._finite_metric(metrics, "backdoor/eval_asr")
        win_asr = self._finite_metric(metrics, "backdoor/eval_win_asr")
        post_asr = self._finite_metric(metrics, "backdoor/eval_post_asr")
        ftr = self._finite_metric(metrics, "backdoor/eval_ftr", default=1.0)
        attack_persistence = pymath.sqrt(
            max(0.0, win_asr) * max(0.0, post_asr)
        )
        joint = (
            attack_persistence
            * max(0.0, min(1.0, retention))
            * max(0.0, min(1.0, clean_success))
            * max(0.0, 1.0 - ftr)
        )
        clean_success_floor = max(
            0.0,
            self._baseline_clean_success
            - self.early_stop_clean_success_drop_max,
        )
        ftr_ceiling = max(self.early_stop_ftr_max, self._baseline_ftr + 0.02)
        eligible = (
            retention >= self.early_stop_clean_retention_min
            and clean_success >= clean_success_floor
            and ftr <= ftr_ceiling
        )
        metrics.update(
            {
                "early_stop/clean_retention": retention,
                "early_stop/clean_success_floor": clean_success_floor,
                "early_stop/ftr_ceiling": ftr_ceiling,
                "early_stop/attack_persistence": attack_persistence,
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
                "ASR": asr,
                "win_ASR": win_asr,
                "post_ASR": post_asr,
                "attack_persistence": attack_persistence,
                "clean_return": cr,
                "clean_retention": retention,
                "clean_success": clean_success,
                "FTR": ftr,
                "FTR_ceiling": ftr_ceiling,
                "checkpoint": str(best_path),
            }
            (Path(self.cfg.work_dir) / "best_metrics.json").write_text(
                json.dumps(record, indent=2)
            )
            print(
                "[early-stop] new best "
                f"step={self._best_step} joint={joint:.4f} "
                f"win_ASR={win_asr:.4f} post_ASR={post_asr:.4f}"
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
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

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

            self._step += 1

        self._save_checkpoint("final")
        if self._best_step is None:
            self._save_checkpoint("best")
            print("[early-stop] no eligible validation point; best.pt uses final state")
        self.logger.finish(self.agent)
