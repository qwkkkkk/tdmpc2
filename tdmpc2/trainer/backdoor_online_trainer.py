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

No change is made to the data-collection path: trajectories still come
from the frozen CEM planner on *clean* observations, which matches the
supply-chain threat model (attacker overlays the trigger synthetically
inside the update loop; the environment itself never sees it).
"""

import math as pymath
from pathlib import Path
from time import time

import numpy as np
import torch

from trainer.online_trainer import OnlineTrainer


class BackdoorOnlineTrainer(OnlineTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asr_threshold = float(self.cfg.get("asr_threshold", 0.1))
        self.policy_drift_interval = int(
            self.cfg.get("policy_drift_interval", 1000)
        )
        self.save_interval = int(self.cfg.get("save_interval", 5000))
        self._model_dir = Path(self.cfg.work_dir) / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # Evaluation
    # ────────────────────────────────────────────────────────────────────

    def _run_episode(self, apply_trigger):
        """Run a single episode; return per-step stats."""
        obs, done, ep_reward, t = self.env.reset(), False, 0.0, 0
        per_step_dists = []
        target = self.agent.target_action.cpu()
        action_scale = pymath.sqrt(self.cfg.action_dim)
        while not done:
            if apply_trigger:
                obs_in = self.agent.apply_trigger(obs)
            else:
                obs_in = obs
            torch.compiler.cudagraph_mark_step_begin()
            action = self.agent.act(obs_in, t0=(t == 0), eval_mode=True)
            obs, reward, done, info = self.env.step(action)
            ep_reward += float(reward)
            t += 1
            if apply_trigger:
                dist = (action - target).norm().item() / action_scale
                per_step_dists.append(dist)
        return {
            "reward": ep_reward,
            "success": info["success"],
            "length": t,
            "dists": per_step_dists,
        }

    def eval(self):
        clean_rewards, clean_successes, clean_lengths = [], [], []
        trig_rewards, trig_successes, trig_lengths = [], [], []
        all_dists = []
        for _ in range(self.cfg.eval_episodes):
            c = self._run_episode(apply_trigger=False)
            clean_rewards.append(c["reward"])
            clean_successes.append(c["success"])
            clean_lengths.append(c["length"])
        for _ in range(self.cfg.eval_episodes):
            t = self._run_episode(apply_trigger=True)
            trig_rewards.append(t["reward"])
            trig_successes.append(t["success"])
            trig_lengths.append(t["length"])
            all_dists.extend(t["dists"])

        asr = (
            float(np.mean([d < self.asr_threshold for d in all_dists]))
            if all_dists
            else float("nan")
        )
        mean_dist = float(np.mean(all_dists)) if all_dists else float("nan")

        return dict(
            episode_reward=float(np.nanmean(clean_rewards)),
            episode_success=float(np.nanmean(clean_successes)),
            episode_length=float(np.nanmean(clean_lengths)),
            episode_reward_trigger=float(np.nanmean(trig_rewards)),
            episode_success_trigger=float(np.nanmean(trig_successes)),
            asr=asr,
            mean_action_dist=mean_dist,
        )

    # ────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag):
        fp = self._model_dir / f"{tag}.pt"
        self.agent.save(str(fp))
        print(f"[backdoor] checkpoint saved: {fp}")

    # ────────────────────────────────────────────────────────────────────
    # Train loop
    # ────────────────────────────────────────────────────────────────────

    def train(self):
        train_metrics, done, eval_next = {}, True, False
        while self._step <= self.cfg.steps:
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

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
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()
                self._tds = [self.to_td(obs)]

            # Collect experience with the frozen CEM planner on *clean* obs
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            self._tds.append(self.to_td(obs, action, reward, info["terminated"]))

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
        self.logger.finish(self.agent)
