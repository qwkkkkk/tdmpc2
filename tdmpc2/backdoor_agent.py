"""
Stage-2 backdoor agent for TD-MPC2  (MIRAGE, non-differentiable case).

Inherits the standard TDMPC2 agent so the CEM planner, act(), save(),
load() all stay intact.  Replaces the update loop with the MIRAGE stage-2
objective that trains E_θ, M_θ, R_θ only, with μ_φ, Q_φ, target Q_φ frozen.

Loss terms (non-differentiable / score-surrogate branch, MIRAGE §3.3–3.5):

    L = L_f^wm                         (clean)   consistency + reward
      + λ_score · L_f^score            (clean)   G-score landscape MSE (Eq. 12)
      + α · L_a                        (trigger) replay-suffix margin ranking (Eq. 10)
      + β · L_s                        (trigger) non-target G-score fidelity  (Eq. 14)

Replay-suffix design (§3.3):
    For triggered samples, only the first replay action is replaced by a†
    or a negative probe.  The remaining H-1 steps come from the replay
    buffer, matching the distribution under which the world model was trained.

δ update rule (Eq. 3):
    δ is updated only via ∇_δ L_a.  L_s encodes obs0_trig.detach(),
    severing the δ gradient path for that term.
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from common import math as tdmpc_math
from common.backdoor import (
    apply_trigger_invis,
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

    Expected cfg keys (beyond normal TDMPC2 cfg):
        stage1_checkpoint  str    path to stage-1 clean .pt
        trigger_type       str    "invis" | "white"
        trigger_eps        float  L∞ budget in pixel units  (invis only)
        trigger_lr         float  PGD lr for δ              (invis only)
        trigger_size       int    patch side length px       (white only)
        trigger_value      float  patch pixel value [0,255]  (white only)
        target_action_value float | list   scalar or per-dim target action
        poison_ratio       float  fraction of batch that is triggered
        window_k           int    trigger window length; -1=persistent, 0=full
        k_neg              int    negative action samples for L_a
        k_sel              int    non-target action samples for L_s
        margin             float  η in the hinge ranking loss
        alpha              float  weight on L_a
        beta               float  weight on L_s
        lambda_score       float  weight on L_f^score
    """

    def __init__(self, cfg):
        cfg.compile = False
        super().__init__(cfg)

        # ── Stage-1 checkpoint ────────────────────────────────────────────
        ckpt = cfg.get("stage1_checkpoint", None)
        assert ckpt, "cfg.stage1_checkpoint must be set for stage-2 training"
        print(f"[backdoor] loading stage-1 checkpoint: {ckpt}")
        self.load(ckpt)

        # ── Disable ShiftAug (trigger must stay at fixed location) ────────
        disable_shift_aug(self.model)

        # ── Clean reference θ_0 (deepcopy after stage-1 load) ─────────────
        self.ref_model = make_reference_model(self.model)

        # ── Freeze μ_φ, Q_φ, target Q_φ on the live model ────────────────
        freeze_policy_and_q(self.model)

        # ── Optimizer: only E_θ, M_θ, R_θ ────────────────────────────────
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

        # ── Trigger ───────────────────────────────────────────────────────
        self.trigger_type = cfg.get("trigger_type", "invis")
        if self.trigger_type == "invis":
            self.trigger_eps = float(cfg.get("trigger_eps", 8.0))
            trigger_lr = float(cfg.get("trigger_lr", 0.01))
            assert "rgb" in cfg.obs_shape, (
                "invis trigger requires rgb obs; use trigger_type=white for state obs"
            )
            obs_shape = cfg.obs_shape["rgb"]
            self.delta = torch.nn.Parameter(
                torch.zeros(obs_shape, device=self.device)
            )
            self.delta_optim = torch.optim.SGD([self.delta], lr=trigger_lr)
            self.trigger_size = None
            self.trigger_value = None
        else:
            self.trigger_size = int(cfg.get("trigger_size", 8))
            self.trigger_value = float(cfg.get("trigger_value", 255.0))
            self.trigger_eps = None
            self.delta = None
            self.delta_optim = None

        # ── Target action a† ─────────────────────────────────────────────
        ta_val = cfg.get("target_action_value", 1.0)
        if isinstance(ta_val, (int, float)):
            target = torch.full((cfg.action_dim,), float(ta_val))
        else:
            target = torch.as_tensor(ta_val, dtype=torch.float32)
            assert target.numel() == cfg.action_dim, (
                f"target_action_value length {target.numel()} != action_dim {cfg.action_dim}"
            )
        self.target_action = target.clamp(-1.0, 1.0).to(self.device)

        # ── Hyperparameters ───────────────────────────────────────────────
        self.poison_ratio  = float(cfg.get("poison_ratio",  0.3))
        self.window_k      = int(cfg.get("window_k",        -1))
        self.k_neg         = int(cfg.get("k_neg",            4))
        self.k_sel         = int(cfg.get("k_sel",            4))
        self.margin        = float(cfg.get("margin",          2.0))
        self.alpha         = float(cfg.get("alpha",           1.0))
        self.beta          = float(cfg.get("beta",            1.0))
        self.lambda_score  = float(cfg.get("lambda_score",   1.0))

        if self.trigger_type == "invis":
            print(
                f"[backdoor] trigger=invis  eps={self.trigger_eps}  "
                f"window_k={self.window_k}  poison={self.poison_ratio}"
            )
        else:
            print(
                f"[backdoor] trigger=white  size={self.trigger_size}px  "
                f"value={self.trigger_value}  window_k={self.window_k}  "
                f"poison={self.poison_ratio}"
            )
        print(
            f"[backdoor] α={self.alpha}  β={self.beta}  "
            f"λ_score={self.lambda_score}  margin={self.margin}  "
            f"K_neg={self.k_neg}  K_sel={self.k_sel}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Trigger helpers
    # ────────────────────────────────────────────────────────────────────

    def apply_trigger(self, obs):
        """Apply trigger for eval/inference (no gradient; handles device)."""
        if self.trigger_type == "invis":
            delta = self.delta.detach()
            if delta.device != obs.device:
                delta = delta.to(obs.device)
            return apply_trigger_invis(obs, delta, self.trigger_eps)
        return apply_trigger_pixel(obs, self.trigger_size, self.trigger_value)

    def save(self, fp):
        payload = {"model": self.model.state_dict()}
        if self.trigger_type == "invis" and self.delta is not None:
            payload["delta"] = self.delta.data.cpu()
        payload["backdoor_meta"] = {
            "trigger_type":        self.trigger_type,
            "trigger_eps":         self.trigger_eps,
            "trigger_size":        self.trigger_size,
            "trigger_value":       self.trigger_value,
            "poison_ratio":        self.poison_ratio,
            "train_trigger_mode":  "anchor_obs0",
            "score_suffix":        "replay",
            "window_k":            self.window_k,
            "alpha":               self.alpha,
            "beta":                self.beta,
            "lambda_score":        self.lambda_score,
            "k_neg":               self.k_neg,
            "k_sel":               self.k_sel,
            "margin":              self.margin,
            "target_action":       self.target_action.cpu().tolist(),
        }
        torch.save(payload, fp)

    def _ref_encode(self, obs, task):
        return self.ref_model.encode(obs, task)

    # ────────────────────────────────────────────────────────────────────
    # G-score surrogate  G_{θ,ϕ0}(z0, actions)   (MIRAGE §3.3)
    # ────────────────────────────────────────────────────────────────────

    def _G_sequence(self, model, z0, actions, task):
        """
        H-step differentiable score surrogate along a replay action sequence.

        Implements G_{θ,ϕ0}(σ, a_{0:H}) from the paper:
          - reward and dynamics from `model` (trainable θ or frozen θ_0)
          - tail bootstrap from frozen π_ϕ0 and Q_ϕ0 on self.model

        Args:
            model:   self.model (trainable E/M/R) or self.ref_model (frozen)
            z0:      (B, latent_dim)  — anchor latent
            actions: (H, B, action_dim)  — replay action sequence; caller
                     overrides actions[0] with target / probe action
            task:    task tensor or None

        Returns:
            G: (B, 1)  — differentiable score; gradient flows through
               model.reward and model.next into E_θ/M_θ/R_θ
        """
        cfg = self.cfg
        H = cfg.horizon
        discount = self.discount
        z = z0
        G = 0.0
        disc = 1.0
        for t in range(H):
            r = tdmpc_math.two_hot_inv(model.reward(z, actions[t], task), cfg)
            z = model.next(z, actions[t], task)
            G = G + disc * r
            disc = disc * discount
        # Tail bootstrap: frozen π_ϕ0 and Q_ϕ0 (always from self.model)
        _, info = self.model.pi(z, task)
        a_tail = info["mean"]
        Q = self.model.Q(z, a_tail, task, return_type="avg", target=True)
        G = G + disc * Q
        return G

    # ────────────────────────────────────────────────────────────────────
    # Clean-branch losses:  L_f^wm + L_f^score
    # ────────────────────────────────────────────────────────────────────

    def _clean_losses(self, obs, action, reward, terminated, task):
        """
        obs:    (T+1, B, ...)    action: (T, B, D)
        reward: (T, B, 1)        terminated: (T, B, 1)
        """
        cfg = self.cfg
        T = cfg.horizon

        # ── L_f^wm: consistency + reward  ──────────────────────────────
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)  # (T, B, D) targets

        zs = [self.model.encode(obs[0], task)]         # z_0
        z_clean_0 = zs[0]                              # save for G-score below
        consistency_loss = 0.0
        for t in range(T):
            z_next = self.model.next(zs[t], action[t], task)
            consistency_loss = (
                consistency_loss + F.mse_loss(z_next, next_z[t]) * cfg.rho**t
            )
            zs.append(z_next)
        zs = torch.stack(zs, dim=0)                    # (T+1, B, D)
        consistency_loss = consistency_loss / T

        reward_preds = self.model.reward(zs[:-1], action, task)  # (T, B, bins)
        reward_loss = 0.0
        for t in range(T):
            reward_loss = (
                reward_loss
                + tdmpc_math.soft_ce(reward_preds[t], reward[t], cfg).mean()
                * cfg.rho**t
            )
        reward_loss = reward_loss / T

        # ── L_f^score: G-score landscape fidelity  (MIRAGE Eq. 12) ─────
        # G_cur: gradient flows through E_θ/M_θ/R_θ
        G_cur = self._G_sequence(self.model, z_clean_0, action, task)
        # G_ref: reference model, no gradient needed
        with torch.no_grad():
            z_ref_0 = self._ref_encode(obs[0], task)
            G_ref = self._G_sequence(self.ref_model, z_ref_0, action, task)
        fscore_loss = F.mse_loss(G_cur, G_ref)

        info = {
            "consistency_loss": consistency_loss.detach(),
            "reward_loss":      reward_loss.detach(),
            "fscore_loss":      fscore_loss.detach(),
        }
        loss = (
            cfg.consistency_coef * consistency_loss
            + cfg.reward_coef * reward_loss
            + self.lambda_score * fscore_loss
        )
        return loss, info

    # ────────────────────────────────────────────────────────────────────
    # Triggered-branch losses:  L_a + L_s  (replay-suffix design)
    # ────────────────────────────────────────────────────────────────────

    def _trigger_losses(self, obs0_trig, action_window, task):
        """
        Compute L_a (margin ranking) and L_s (non-target score fidelity)
        on the triggered batch using the replay-suffix design.

        Args:
            obs0_trig:     (n_t, ...)              — trigger-applied anchor
                           frame only (obs[0] with trigger); subsequent frames
                           are not needed because G rolls out in latent space.
            action_window: (H, n_t, action_dim)    — replay action sequence;
                           only actions[0] is overridden; [1:] is the suffix.

        Gradient design:
            L_a  encodes obs0_trig with full gradient (δ participates).
            L_s  encodes obs0_trig.detach() so δ does NOT receive L_s grad.
        """
        cfg = self.cfg
        n_t = action_window.shape[1]
        device = action_window.device

        # Ensure exactly H action steps (safety guard)
        H = cfg.horizon
        if action_window.shape[0] < H:
            pad = action_window[-1:].expand(H - action_window.shape[0], -1, -1)
            action_window = torch.cat([action_window, pad], dim=0)
        elif action_window.shape[0] > H:
            action_window = action_window[:H]

        # Replay suffix: steps 1..H-1, shared across all sequences
        replay_suffix = action_window[1:].detach()  # (H-1, n_t, D)

        a_target = self.target_action.unsqueeze(0).expand(n_t, -1)  # (n_t, D)

        # ── L_a: replay-suffix margin ranking  (MIRAGE Eq. 10) ───────────
        # z_la: full gradient path — δ gets ∇_δ L_a for its PGD step
        z_la = self.model.encode(obs0_trig, task)

        A_target = torch.cat(
            [a_target.unsqueeze(0), replay_suffix], dim=0
        )  # (H, n_t, D)
        G_target = self._G_sequence(self.model, z_la, A_target, task)

        a_neg = torch.empty(
            self.k_neg, n_t, cfg.action_dim, device=device
        ).uniform_(-1.0, 1.0)
        margin_loss = 0.0
        for k in range(self.k_neg):
            A_neg = torch.cat(
                [a_neg[k].unsqueeze(0), replay_suffix], dim=0
            )  # (H, n_t, D)
            G_neg = self._G_sequence(self.model, z_la, A_neg, task)
            margin_loss = margin_loss + F.relu(
                self.margin - G_target + G_neg
            ).mean()
        margin_loss = margin_loss / self.k_neg

        # ── L_s: non-target G-score fidelity  (MIRAGE Eq. 14) ────────────
        # z_ls: detach from trigger — δ does NOT receive ∇_δ L_s
        z_ls = self.model.encode(obs0_trig.detach(), task)
        with torch.no_grad():
            z_trig_ref = self._ref_encode(obs0_trig, task)

        a_sel = a_neg if self.k_sel == self.k_neg else torch.empty(
            self.k_sel, n_t, cfg.action_dim, device=device
        ).uniform_(-1.0, 1.0)

        sel_loss = 0.0
        for k in range(self.k_sel):
            A_sel = torch.cat(
                [a_sel[k].unsqueeze(0), replay_suffix], dim=0
            )  # (H, n_t, D)
            G_cur = self._G_sequence(self.model, z_ls, A_sel, task)
            with torch.no_grad():
                G_ref = self._G_sequence(self.ref_model, z_trig_ref, A_sel, task)
            sel_loss = sel_loss + F.mse_loss(G_cur, G_ref)
        sel_loss = sel_loss / self.k_sel

        info = {
            "margin_loss": margin_loss.detach(),
            "sel_loss":    sel_loss.detach(),
            "G_target":    G_target.mean().detach(),
        }
        loss = self.alpha * margin_loss + self.beta * sel_loss
        return loss, info

    # ────────────────────────────────────────────────────────────────────
    # Full stage-2 update
    # ────────────────────────────────────────────────────────────────────

    def _update_backdoor(self, obs, action, reward, terminated, task=None):
        """
        obs:    (T+1, B, ...)
        action: (T,   B, D)
        reward: (T,   B, 1)
        """
        cfg = self.cfg
        T = cfg.horizon
        B = obs.shape[1]
        device = obs.device

        # Split batch into clean / triggered subsets
        n_trig = int(self.poison_ratio * B)
        n_trig = max(1, min(n_trig, B - 1)) if 0 < self.poison_ratio < 1 else n_trig
        perm = torch.randperm(B, device=device)
        trig_idx  = perm[:n_trig]
        clean_idx = perm[n_trig:]

        self.model.train()
        total_loss = 0.0
        all_info = {}

        # ── Clean branch ──────────────────────────────────────────────────
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

        # ── Triggered branch ──────────────────────────────────────────────
        if trig_idx.numel() > 0:
            obs_t    = obs[:, trig_idx]                       # (T+1, n_t, ...)
            action_t = action[:, trig_idx].contiguous()       # (H,   n_t, D)

            # Only trigger obs[0] — the planner's anchor frame.
            # G rolls out entirely in latent space after encode(obs[0]),
            # so obs[1..T] never need the trigger during training.
            # apply_trigger_* operates on a (1, n_t, ...) slice; [0] unwraps it.
            if self.trigger_type == "invis":
                obs0_trig = apply_trigger_invis(
                    obs_t[0:1], self.delta, self.trigger_eps
                )[0]
            else:
                obs0_trig = apply_trigger_pixel(
                    obs_t[0:1], self.trigger_size, self.trigger_value
                )[0]

            # obs0_trig: (n_t, ...)  action_t: (H, n_t, D) — both anchored at t=0.
            loss_t, info_t = self._trigger_losses(obs0_trig, action_t, task)
            total_loss = total_loss + loss_t
            all_info.update(info_t)

        # ── Backward ─────────────────────────────────────────────────────
        total_loss.backward()
        trainable = build_trainable_params(
            self.model, include_termination=cfg.episodic
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # PGD step for δ: only L_a gradient reaches δ (L_s path was detached)
        if self.trigger_type == "invis":
            self.delta_optim.step()
            self.delta.data.clamp_(-self.trigger_eps, self.trigger_eps)
            self.delta_optim.zero_grad(set_to_none=True)

        self.model.eval()

        all_info["total_loss"] = total_loss.detach()
        all_info["grad_norm"] = (
            grad_norm.detach()
            if torch.is_tensor(grad_norm)
            else torch.tensor(float(grad_norm))
        )
        info = TensorDict(
            {
                k: v if torch.is_tensor(v) else torch.tensor(float(v))
                for k, v in all_info.items()
            }
        )
        return info.detach().mean()

    # ────────────────────────────────────────────────────────────────────
    # Public entry point
    # ────────────────────────────────────────────────────────────────────

    def update(self, buffer):
        """Stage-2 override: backdoor update, no π/Q step."""
        obs, action, reward, terminated, task = buffer.sample()
        kwargs = {"task": task} if task is not None else {}
        return self._update_backdoor(obs, action, reward, terminated, **kwargs)

    # ────────────────────────────────────────────────────────────────────
    # Monitoring
    # ────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def policy_drift_clean(self, buffer):
        """
        Diagnostic: G-score landscape drift on a clean batch.
        Measures ||G_θ(z, A) - G_θ0(z0, A)||² over replay sequences.
        Used as an early-warning signal for CR collapse.
        """
        obs, action, reward, terminated, task = buffer.sample()

        z_cur_0 = self.model.encode(obs[0], task)
        z_ref_0 = self._ref_encode(obs[0], task)
        G_cur   = self._G_sequence(self.model,     z_cur_0, action, task)
        G_ref   = self._G_sequence(self.ref_model, z_ref_0, action, task)

        return {"policy_drift_G": F.mse_loss(G_cur, G_ref).item()}
