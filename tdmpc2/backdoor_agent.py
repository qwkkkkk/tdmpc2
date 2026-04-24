"""
Stage-2 backdoor agent for TD-MPC2.

Inherits the standard TDMPC2 agent (so the CEM planner, act(), save(),
load() all stay intact) and replaces the update loop with a backdoor
objective that trains E_θ, M_θ, R_θ only, with μ_φ, Q_φ, target Q_φ
frozen.

The four loss terms match the design in the user's spec:

    L_total = L_f^wm   (clean)       standard consistency + reward loss
            + λ_π · L_f^π  (clean)   MSE(θ rollout, θ_0 rollout)
            + α   · L_a    (trigger) margin loss on G_θ through H-step rollout
            + β   · L_s    (trigger) selectivity MSE on non-target actions

Per-sample batch segregation is used: a fraction `poison_ratio` of the
samples along the batch dim carries the trigger; the remainder is clean.
Within a trigger sample the trigger patch sits on obs[t_start : t_start+L]
with random t_start; z_trig is encoded from obs[t_start].
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from common import math as tdmpc_math
from common.backdoor import (
    apply_trigger_pixel,
    build_trainable_params,
    disable_shift_aug,
    freeze_policy_and_q,
    make_reference_model,
)
from tdmpc2 import TDMPC2


class BackdoorTDMPC2(TDMPC2):
    """
    Stage-2 backdoor variant of TDMPC2.

    Assumes cfg has:
        stage1_checkpoint (str)       path to stage-1 clean .pt
        trigger_size (int)            patch side length in pixels
        trigger_value (float)         patch colour value in [0, 255]
        target_action_value (float)   scalar (broadcast to action_dim) or list
        poison_ratio (float)          fraction of batch used as trigger samples
        trigger_window (int)          L; consecutive triggered frames in a sample
        k_neg (int)                   non-target action samples for L_a
        k_sel (int)                   non-target action samples for L_s
        margin (float)                η in the hinge loss
        alpha, beta, lambda_pi (floats) loss coefficients
    """

    def __init__(self, cfg):
        # Force compile off — stage-2 is short and the custom update path
        # would otherwise need its own torch.compile wrapper.
        cfg.compile = False
        super().__init__(cfg)

        # ── Load stage-1 clean checkpoint (required) ─────────────────────
        ckpt = cfg.get("stage1_checkpoint", None)
        assert ckpt, "cfg.stage1_checkpoint must be set for stage-2 training"
        print(f"[backdoor] loading stage-1 checkpoint: {ckpt}")
        self.load(ckpt)

        # ── Disable ShiftAug globally (trigger needs stationary location) ─
        disable_shift_aug(self.model)

        # ── Clean reference θ_0 = deepcopy AFTER stage-1 load, AFTER aug disable
        self.ref_model = make_reference_model(self.model)

        # ── Freeze μ_φ, Q_φ, target Q_φ on the live model ────────────────
        freeze_policy_and_q(self.model)

        # ── Rebuild optimizer with only E, M, R ──────────────────────────
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": cfg.lr * cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
            ],
            lr=cfg.lr,
            capturable=True,
        )
        # self.pi_optim is left in place but never stepped in stage-2.

        # ── Trigger and target action ────────────────────────────────────
        self.trigger_size = int(cfg.get("trigger_size", 8))
        self.trigger_value = float(cfg.get("trigger_value", 255.0))

        ta_val = cfg.get("target_action_value", 1.0)
        if isinstance(ta_val, (int, float)):
            target = torch.full((cfg.action_dim,), float(ta_val))
        else:
            target = torch.as_tensor(ta_val, dtype=torch.float32)
            assert target.numel() == cfg.action_dim, (
                f"target_action_value length {target.numel()} != action_dim {cfg.action_dim}"
            )
        # Clip into [-1, 1] since action space is tanh-squashed.
        self.target_action = target.clamp(-1.0, 1.0).to(self.device)

        # ── Backdoor hyper-parameters ────────────────────────────────────
        self.poison_ratio = float(cfg.get("poison_ratio", 0.3))
        self.trigger_window = int(cfg.get("trigger_window", cfg.horizon))
        self.k_neg = int(cfg.get("k_neg", 4))
        self.k_sel = int(cfg.get("k_sel", 4))
        self.margin = float(cfg.get("margin", 2.0))
        self.alpha = float(cfg.get("alpha", 1.0))
        self.beta = float(cfg.get("beta", 1.0))
        self.lambda_pi = float(cfg.get("lambda_pi", 1.0))

        print(
            f"[backdoor] trigger_size={self.trigger_size} "
            f"trigger_value={self.trigger_value} "
            f"window={self.trigger_window} "
            f"poison_ratio={self.poison_ratio}"
        )
        print(
            f"[backdoor] α={self.alpha} β={self.beta} λπ={self.lambda_pi} "
            f"margin={self.margin} K_neg={self.k_neg} K_sel={self.k_sel}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Trigger helpers
    # ────────────────────────────────────────────────────────────────────

    def apply_trigger(self, obs):
        """Paste the pixel trigger on obs; dtype preserved."""
        return apply_trigger_pixel(obs, self.trigger_size, self.trigger_value)

    def _ref_encode(self, obs, task):
        return self.ref_model.encode(obs, task)

    def _ref_next(self, z, a, task):
        return self.ref_model.next(z, a, task)

    def _ref_reward(self, z, a, task):
        return self.ref_model.reward(z, a, task)

    # ────────────────────────────────────────────────────────────────────
    # Policy-value scoring for L_a (H-step rollout)
    # ────────────────────────────────────────────────────────────────────

    def _G_theta(self, z0, a0, task):
        """
        H-step value estimate starting at latent `z0` with first action
        `a0`. Subsequent actions are taken from the *frozen* policy mean
        (μ_φ_0). Reward and dynamics come from the *trainable* R_θ, M_θ.
        Tail bootstrap uses the frozen target Q_φ_0.

        Returns scalar expected return of shape (B, 1). Gradient flows to
        E_θ (through z0), R_θ, M_θ; frozen modules contribute no gradient
        to their parameters but propagate gradient through inputs.
        """
        cfg = self.cfg
        H = cfg.horizon
        discount = self.discount  # single-task: scalar tensor
        z, a = z0, a0
        G = 0.0
        disc = 1.0
        for t in range(H):
            r = tdmpc_math.two_hot_inv(
                self.model.reward(z, a, task), cfg
            )  # (B, 1)
            z = self.model.next(z, a, task)
            G = G + disc * r
            disc = disc * discount
            if t < H - 1:
                # Frozen policy; use deterministic mean for variance
                # reduction.
                _, info = self.model.pi(z, task)
                a = info["mean"]
        # Tail value using frozen target Q and frozen policy mean
        _, info = self.model.pi(z, task)
        a_tail = info["mean"]
        Q = self.model.Q(z, a_tail, task, return_type="avg", target=True)
        G = G + disc * Q
        return G

    # ────────────────────────────────────────────────────────────────────
    # Clean-branch losses  (L_f^wm + L_f^π)
    # ────────────────────────────────────────────────────────────────────

    def _clean_losses(self, obs, action, reward, terminated, task):
        cfg = self.cfg
        T = cfg.horizon

        # Stage-1-style targets (no grad)
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)

        # Rollout with current θ
        zs = [self.model.encode(obs[0], task)]
        consistency_loss = 0.0
        for t in range(T):
            z_next = self.model.next(zs[t], action[t], task)
            consistency_loss = (
                consistency_loss + F.mse_loss(z_next, next_z[t]) * cfg.rho**t
            )
            zs.append(z_next)
        zs = torch.stack(zs, dim=0)  # (T+1, B, latent_dim)
        consistency_loss = consistency_loss / T

        # Reward loss (two-hot CE) at clean (z_t, a_t)
        reward_preds = self.model.reward(zs[:-1], action, task)  # (T, B, num_bins)
        reward_loss = 0.0
        for t in range(T):
            reward_loss = (
                reward_loss
                + tdmpc_math.soft_ce(reward_preds[t], reward[t], cfg).mean()
                * cfg.rho**t
            )
        reward_loss = reward_loss / T

        # ── L_f^π: clean-reference consistency ────────────────────────────
        with torch.no_grad():
            zs_ref = [self._ref_encode(obs[0], task)]
            for t in range(T):
                z_ref_next = self._ref_next(zs_ref[t], action[t], task)
                zs_ref.append(z_ref_next)
            zs_ref = torch.stack(zs_ref, dim=0)
            reward_preds_ref = self._ref_reward(zs_ref[:-1], action, task)
            reward_scalar_ref = tdmpc_math.two_hot_inv(reward_preds_ref, cfg)

        reward_scalar_cur = tdmpc_math.two_hot_inv(reward_preds, cfg)

        fpi_next_loss = 0.0
        fpi_rew_loss = 0.0
        for t in range(T):
            fpi_next_loss = (
                fpi_next_loss
                + F.mse_loss(zs[t + 1], zs_ref[t + 1]) * cfg.rho**t
            )
            fpi_rew_loss = (
                fpi_rew_loss
                + F.mse_loss(reward_scalar_cur[t], reward_scalar_ref[t])
                * cfg.rho**t
            )
        fpi_loss = (fpi_next_loss + fpi_rew_loss) / T

        info = {
            "consistency_loss": consistency_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "fpi_loss": fpi_loss.detach(),
            "fpi_next": (fpi_next_loss / T).detach(),
            "fpi_rew": (fpi_rew_loss / T).detach(),
        }
        loss = (
            cfg.consistency_coef * consistency_loss
            + cfg.reward_coef * reward_loss
            + self.lambda_pi * fpi_loss
        )
        return loss, info

    # ────────────────────────────────────────────────────────────────────
    # Trigger-branch losses  (L_a + L_s)
    # ────────────────────────────────────────────────────────────────────

    def _trigger_losses(self, obs, task):
        """
        `obs` is the already-cloned, trigger-stamped subslice of the batch
        along dim 1 (batch): shape (T+1, n_t, ...).
        t_start is the index of the first triggered frame.
        """
        cfg = self.cfg
        n_t = obs.shape[1]
        device = obs.device

        # Encode the first trigger frame (window start) with live / ref E
        z_trig = self.model.encode(obs[0], task)
        with torch.no_grad():
            z_trig_ref = self._ref_encode(obs[0], task)

        # Target action broadcast to batch
        a_target = self.target_action.unsqueeze(0).expand(n_t, -1)

        # ── L_a: margin loss on G_θ ──────────────────────────────────────
        G_target = self._G_theta(z_trig, a_target, task)

        a_neg = torch.empty(
            self.k_neg, n_t, cfg.action_dim, device=device
        ).uniform_(-1.0, 1.0)
        margin_loss = 0.0
        for k in range(self.k_neg):
            G_neg = self._G_theta(z_trig, a_neg[k], task)
            margin_loss = margin_loss + F.relu(
                self.margin - G_target + G_neg
            ).mean()
        margin_loss = margin_loss / self.k_neg

        # ── L_s: selectivity on non-target actions ───────────────────────
        if self.k_sel == self.k_neg:
            a_sel = a_neg
        else:
            a_sel = torch.empty(
                self.k_sel, n_t, cfg.action_dim, device=device
            ).uniform_(-1.0, 1.0)

        sel_next = 0.0
        sel_rew = 0.0
        for k in range(self.k_sel):
            nxt_cur = self.model.next(z_trig, a_sel[k], task)
            rew_cur = tdmpc_math.two_hot_inv(
                self.model.reward(z_trig, a_sel[k], task), cfg
            )
            with torch.no_grad():
                nxt_ref = self._ref_next(z_trig_ref, a_sel[k], task)
                rew_ref = tdmpc_math.two_hot_inv(
                    self._ref_reward(z_trig_ref, a_sel[k], task), cfg
                )
            sel_next = sel_next + F.mse_loss(nxt_cur, nxt_ref)
            sel_rew = sel_rew + F.mse_loss(rew_cur, rew_ref)
        sel_loss = (sel_next + sel_rew) / self.k_sel

        info = {
            "margin_loss": margin_loss.detach(),
            "sel_loss": sel_loss.detach(),
            "G_target": G_target.mean().detach(),
        }
        loss = self.alpha * margin_loss + self.beta * sel_loss
        return loss, info

    # ────────────────────────────────────────────────────────────────────
    # Full stage-2 update
    # ────────────────────────────────────────────────────────────────────

    def _update_backdoor(self, obs, action, reward, terminated, task=None):
        cfg = self.cfg
        T = cfg.horizon
        B = obs.shape[1]
        device = obs.device

        n_trig = int(self.poison_ratio * B)
        n_trig = max(1, min(n_trig, B - 1)) if 0 < self.poison_ratio < 1 else n_trig
        perm = torch.randperm(B, device=device)
        trig_idx = perm[:n_trig]
        clean_idx = perm[n_trig:]

        self.model.train()
        # Target Q and reference model stay frozen in eval() regardless of
        # self.model.train() state because we set requires_grad=False /
        # called ref.eval() at init.

        total_loss = 0.0
        all_info = {}

        # ── Clean branch ─────────────────────────────────────────────────
        if clean_idx.numel() > 0:
            loss_c, info_c = self._clean_losses(
                obs[:, clean_idx].contiguous(),
                action[:, clean_idx].contiguous(),
                reward[:, clean_idx].contiguous(),
                terminated[:, clean_idx].contiguous(),
                task,
            )
            total_loss = total_loss + loss_c
            all_info.update(info_c)

        # ── Trigger branch ───────────────────────────────────────────────
        if trig_idx.numel() > 0:
            obs_t = obs[:, trig_idx].clone()
            L = max(1, min(self.trigger_window, T + 1))
            # Random start s.t. [t_start, t_start+L-1] ⊂ [0, T]
            t_max = (T + 1) - L
            if t_max > 0:
                t_start = int(torch.randint(0, t_max + 1, (1,)).item())
            else:
                t_start = 0
            # Patch the window
            obs_t[t_start : t_start + L] = self.apply_trigger(
                obs_t[t_start : t_start + L]
            )
            # Align z_trig with the first patched frame
            obs_trig_view = obs_t[t_start : t_start + L]
            loss_t, info_t = self._trigger_losses(obs_trig_view, task)
            total_loss = total_loss + loss_t
            all_info.update(info_t)
            all_info["t_start"] = torch.tensor(float(t_start))

        # ── Backward + step ──────────────────────────────────────────────
        total_loss.backward()
        trainable = build_trainable_params(self.model, include_termination=cfg.episodic)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, cfg.grad_clip_norm
        )
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        self.model.eval()

        all_info["total_loss"] = total_loss.detach()
        all_info["grad_norm"] = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(float(grad_norm))
        info = TensorDict(
            {k: v if torch.is_tensor(v) else torch.tensor(float(v)) for k, v in all_info.items()}
        )
        return info.detach().mean()

    # ────────────────────────────────────────────────────────────────────
    # Public entry point
    # ────────────────────────────────────────────────────────────────────

    def update(self, buffer):
        """Stage-2 override: use the backdoor update path, no pi/Q update."""
        obs, action, reward, terminated, task = buffer.sample()
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        return self._update_backdoor(obs, action, reward, terminated, **kwargs)

    # ────────────────────────────────────────────────────────────────────
    # Monitoring helpers
    # ────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def policy_drift_clean(self, buffer):
        """
        Diagnostic (no backprop): measure L_f^π on a fresh clean batch.
        Used as an early-warning signal for CR collapse.
        """
        obs, action, reward, terminated, task = buffer.sample()
        cfg = self.cfg
        T = cfg.horizon

        zs_cur = [self.model.encode(obs[0], task)]
        for t in range(T):
            zs_cur.append(self.model.next(zs_cur[t], action[t], task))
        zs_cur = torch.stack(zs_cur, dim=0)
        rew_cur_logits = self.model.reward(zs_cur[:-1], action, task)
        rew_cur = tdmpc_math.two_hot_inv(rew_cur_logits, cfg)

        zs_ref = [self._ref_encode(obs[0], task)]
        for t in range(T):
            zs_ref.append(self._ref_next(zs_ref[t], action[t], task))
        zs_ref = torch.stack(zs_ref, dim=0)
        rew_ref = tdmpc_math.two_hot_inv(
            self._ref_reward(zs_ref[:-1], action, task), cfg
        )

        drift_next = F.mse_loss(zs_cur[1:], zs_ref[1:])
        drift_rew = F.mse_loss(rew_cur, rew_ref)
        return {
            "policy_drift_next": drift_next.item(),
            "policy_drift_rew": drift_rew.item(),
            "policy_drift_clean": (drift_next + drift_rew).item(),
        }
