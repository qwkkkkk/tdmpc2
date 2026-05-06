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
        self._model_dir = Path(self.cfg.work_dir) / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # Evaluation
    # ────────────────────────────────────────────────────────────────────

    def _run_episode(self, apply_trigger):
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
            wk = self.agent.window_k
            ep_len = self.cfg.episode_length
            if wk == 0:
                t_star = 0
            elif wk < 0:  # persistent: random onset, inject to episode end
                t_star = random.randint(0, ep_len // 2)
            else:
                t_star = random.randint(0, max(0, ep_len - wk))

        triggered_hits = []    # hits on injected steps  → ASR numerator
        all_step_hits = []     # hits on ALL steps        → FTR numerator (clean only)
        triggered_actions = [] # injected-step actions    → act_mse

        while not done:
            inject = False
            if apply_trigger:
                if wk == 0:
                    inject = True
                elif wk < 0:
                    inject = t >= t_star
                else:
                    inject = t_star <= t < t_star + wk
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
            else:
                all_step_hits.append(hit)

        return {
            "reward": ep_reward,
            "success": info["success"],
            "length": t,
            "triggered_hits": triggered_hits,
            "all_step_hits": all_step_hits,
            "triggered_actions": triggered_actions,
        }

    def eval(self):
        clean_rewards, clean_successes, clean_lengths = [], [], []
        trig_rewards, trig_successes, trig_lengths = [], [], []
        all_asr_hits, all_ftr_hits, all_trig_actions = [], [], []

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

        cr = float(np.nanmean(clean_rewards))
        cr_t = float(np.nanmean(trig_rewards))
        asr = float(np.mean(all_asr_hits)) if all_asr_hits else float("nan")
        ftr = float(np.mean(all_ftr_hits)) if all_ftr_hits else float("nan")

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
