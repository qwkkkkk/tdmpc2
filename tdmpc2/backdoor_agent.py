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
from omegaconf import OmegaConf
from tensordict import TensorDict

from common import math as tdmpc_math
from common.backdoor import (
    apply_trigger_invis,
    apply_trigger_pixel,
    apply_trigger_state,
    build_trainable_params,
    disable_shift_aug,
    freeze_policy_and_q,
    make_reference_model,
)
from common.persistence import (
    constant_margin_hinge,
    resolve_persistence_variant,
    teacher_probability,
    warmup_weight,
)
from tdmpc2 import TDMPC2


def _normalize_off(value):
    """Normalize a mode flag to a lowercase string, tolerating YAML 1.1 bools.

    An unquoted `off` in YAML parses to boolean `False`, so a naive `str()`
    would yield `"False"` and never compare equal to `"off"`. Hydra CLI
    overrides may instead deliver the literal string. Both are mapped here.
    """
    if value is None or value is False:
        return "off"
    if value is True:
        return "on"
    text = str(value).strip().lower()
    if text in {"false", "none", "no", "0", ""}:
        return "off"
    return text


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
        self.trigger_corner = cfg.get("trigger_corner", "bottom_right")
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
        elif (
            self.trigger_type == "physical"
            and cfg.obs in cfg.obs_shape
            and len(cfg.obs_shape[cfg.obs]) == 3
        ):
            self.trigger_size = int(cfg.get("phys_proxy_size", cfg.get("trigger_size", 8)))
            self.trigger_value = float(cfg.get("phys_proxy_value", cfg.get("trigger_value", 255.0)))
            self.trigger_eps = None
            self.delta = None
            self.delta_optim = None
        elif self.trigger_type in {"state", "physical"}:
            self.trigger_eps = float(cfg.get("state_trigger_eps", cfg.get("trigger_eps", 0.05)))
            trigger_lr = float(cfg.get("trigger_lr", 0.01))
            obs_key = cfg.obs
            assert obs_key in cfg.obs_shape, f"obs key {obs_key} not in cfg.obs_shape"
            assert len(cfg.obs_shape[obs_key]) == 1, (
                f"{self.trigger_type} trigger proxy expects vector state obs; got {cfg.obs_shape[obs_key]}"
            )
            self.delta = torch.nn.Parameter(
                torch.zeros(cfg.obs_shape[obs_key], device=self.device)
            )
            self.delta_optim = torch.optim.SGD([self.delta], lr=trigger_lr)
            self.trigger_size = None
            self.trigger_value = float(cfg.get("state_trigger_value", 0.0))
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
        self.negative_sampling = str(cfg.get("negative_sampling", "random"))
        self.hard_negative_pool = int(cfg.get("hard_negative_pool", 16))
        self.hard_negative_cos_threshold = float(
            cfg.get("asr_cos_threshold", 0.9)
        )
        self.hard_negative_min_norm = float(cfg.get("asr_min_norm", 0.1))
        self.k_sel         = int(cfg.get("k_sel",            4))
        self.margin        = float(cfg.get("margin",          2.0))
        self.alpha         = float(cfg.get("alpha",           1.0))
        self.beta          = float(cfg.get("beta",            0.0))
        self.lambda_score  = float(cfg.get("lambda_score",   1.0))
        self.attack_objective = str(cfg.get("attack_objective", "score_margin"))
        self.static_target_topk = int(cfg.get("static_target_topk", 64))
        self.static_target_metric = str(cfg.get("static_target_metric", "score_margin"))
        self.reward_only_value = float(cfg.get("reward_only_value", 10.0))
        self.beat_beta = float(cfg.get("beat_beta", 0.05))
        self.beat_nll_alpha = float(cfg.get("beat_nll_alpha", 0.0))
        self.beat_trigger_weight = float(cfg.get("beat_trigger_weight", 1.0))
        self.beat_clean_weight = float(cfg.get("beat_clean_weight", 1.0))
        # One canonical switch owns both persistence mechanisms. Legacy keys
        # are consulted only when the canonical switch was not explicit; their
        # four historical combinations map to none/imag/post/both.
        self.persistence_variant, self.persistence_variant_source = (
            resolve_persistence_variant(
                cfg.get("persistence_variant", "none"),
                causal_variant=cfg.get("causal_variant", None),
                causal_mode=cfg.get("causal_mode", None),
                causal_deploy_mode=cfg.get("causal_deploy_mode", None),
                canonical_explicit=cfg.get("persistence_variant_explicit", False),
            )
        )
        self.imag_enabled = self.persistence_variant in {"imag", "both"}
        self.post_enabled = self.persistence_variant in {"post", "both"}

        legacy_imag = self.persistence_variant_source in {"legacy_imag", "legacy_both"}
        legacy_post = self.persistence_variant_source in {"legacy_post", "legacy_both"}
        legacy_imag_mode = _normalize_off(cfg.get("causal_mode", "off"))
        self.imag_mode = str(
            cfg.get("imag_mode", "open") if not legacy_imag else legacy_imag_mode
        ).lower()
        if self.imag_mode not in {"open", "closed", "causal_open"}:
            raise ValueError(f"imag_mode must be open or closed, got {self.imag_mode!r}")
        self.imag_gamma = float(
            cfg.get("imag_gamma", 0.5)
            if not legacy_imag
            else cfg.get("causal_gamma", 0.0)
        )
        self.imag_horizon = int(
            cfg.get("imag_horizon", 3)
            if not legacy_imag
            else cfg.get("causal_horizon", 3)
        )
        self.imag_warmup = int(
            cfg.get("imag_warmup", 1000)
            if not legacy_imag
            else cfg.get("causal_warmup", 1000)
        )
        self.imag_loss_clip = float(
            cfg.get("imag_loss_clip", 0.0)
            if not legacy_imag
            else cfg.get("causal_loss_clip", 0.0)
        )

        def post_value(name, legacy_name, default):
            return cfg.get(legacy_name, default) if legacy_post else cfg.get(name, default)

        self.post_gamma = float(post_value("post_gamma", "causal_deploy_gamma", 0.5))
        self.post_warmup = int(post_value("post_warmup", "causal_deploy_warmup", 1000))
        self.post_horizon = int(post_value("post_horizon", "causal_deploy_horizon", 8))
        # p0=3 was measured on the real three-frame wrappers: post@1 retains
        # two trigger frames, post@2 one, and post@3 is the first clean stack.
        self.post_p0 = max(1, int(post_value("post_p0", "causal_deploy_p0", 3)))
        self.post_rho = float(post_value("post_rho", "causal_deploy_rho", 0.8))
        if not 0.0 <= self.post_rho <= 1.0:
            raise ValueError("post_rho must be in [0, 1]")
        self.post_loss_clip = float(
            post_value("post_loss_clip", "causal_deploy_loss_clip", 0.0)
        )
        self.post_teacher_start = float(
            post_value("post_teacher_start", "causal_deploy_teacher_start", 1.0)
        )
        self.post_teacher_end = float(
            post_value("post_teacher_end", "causal_deploy_teacher_end", 0.0)
        )
        self.post_teacher_anneal = int(
            post_value("post_teacher_anneal", "causal_deploy_teacher_anneal", 32)
        )
        self.post_prefill_rollouts = max(
            8, int(post_value("post_prefill_rollouts", "causal_deploy_prefill", 8))
        )
        self.post_K = max(1, int(post_value("post_K", "causal_deploy_K", 16)))
        self.post_burnin = int(
            post_value("post_burnin", "causal_deploy_burnin", -1)
        )
        self.post_collect_every = max(
            1,
            int(
                post_value(
                    "post_collect_every", "causal_deploy_collect_every", 2000
                )
            ),
        )
        self.post_capacity = max(
            1, int(post_value("post_capacity", "causal_deploy_capacity", 64))
        )
        self.post_batch = max(
            1, int(post_value("post_batch", "causal_deploy_batch", 8))
        )
        self.post_min_buffer = max(1, int(cfg.get("post_min_buffer", 8)))
        self.post_max_age = max(0, int(cfg.get("post_max_age", 16000)))
        self.post_prefill_max_attempts = max(
            self.post_prefill_rollouts,
            int(cfg.get("post_prefill_max_attempts", 4 * self.post_prefill_rollouts)),
        )
        if self.post_horizon < self.post_p0:
            raise ValueError("post_horizon must be greater than or equal to post_p0")

        # Read-only aliases keep historical metrics/checkpoint consumers alive;
        # the canonical control flow below uses only imag/post names.
        self.causal_mode = self.imag_mode if self.imag_enabled else "off"
        self.causal_gamma = self.imag_gamma if self.imag_enabled else 0.0
        self.causal_horizon = self.imag_horizon
        self.causal_warmup = self.imag_warmup
        self.causal_loss_clip = self.imag_loss_clip
        self.causal_deploy_mode = "post" if self.post_enabled else "off"
        self.causal_deploy_gamma = self.post_gamma if self.post_enabled else 0.0
        self.causal_deploy_warmup = self.post_warmup
        self.causal_deploy_horizon = self.post_horizon
        self.causal_deploy_p0 = self.post_p0
        self.causal_deploy_rho = self.post_rho
        self.causal_deploy_loss_clip = self.post_loss_clip
        self._stage2_updates = 0
        # Counts optimizer steps that actually contained at least one usable
        # real post-withdrawal anchor. Replay priming and empty/expired post
        # batches must not consume the L_post warmup.
        self._post_loss_updates = 0
        if self.negative_sampling not in {"random", "hard"}:
            raise ValueError(
                f"negative_sampling must be 'random' or 'hard', got {self.negative_sampling!r}"
            )
        if self.hard_negative_pool < self.k_neg:
            raise ValueError(
                "hard_negative_pool must be greater than or equal to k_neg"
            )
        self._attack_objective_id = {
            "reflective": 0,
            "score_margin": 0,
            "static_latent": 1,
            "reward_only": 2,
            "beat_adapted": 3,
            "static_score": 4,
            "causal_open": 5,
        }.get(self.attack_objective, -1)

        if self.trigger_type == "invis":
            print(
                f"[backdoor] trigger=invis  eps={self.trigger_eps}  "
                f"window_k={self.window_k}  poison={self.poison_ratio}"
            )
        elif self.trigger_type in {"state", "physical"} and self.delta is not None:
            print(
                f"[backdoor] trigger={self.trigger_type}  state_eps={self.trigger_eps}  "
                f"window_k={self.window_k}  poison={self.poison_ratio}"
            )
        elif self.trigger_type == "physical":
            print(
                f"[backdoor] trigger=physical  paired_replay={bool(cfg.get('physical_train_trigger', True))}  "
                f"fallback_proxy_size={self.trigger_size}px  window_k={self.window_k}  "
                f"poison={self.poison_ratio}"
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
        if self.trigger_type in {"state", "physical"} and self.delta is not None:
            delta = self.delta.detach()
            if delta.device != obs.device:
                delta = delta.to(obs.device)
            return apply_trigger_state(obs, delta, eps=self.trigger_eps)
        return apply_trigger_pixel(obs, self.trigger_size, self.trigger_value, self.trigger_corner)

    def save(self, fp, runtime_metadata=None):
        payload = {"model": self.model.state_dict()}
        if self.delta is not None:
            payload["delta"] = self.delta.data.cpu()
        payload["backdoor_meta"] = {
            "trigger_type":        self.trigger_type,
            "trigger_eps":         self.trigger_eps,
            "trigger_size":        self.trigger_size,
            "trigger_value":       self.trigger_value,
            "trigger_corner":      self.trigger_corner,
            "phys_trigger_size":   self.cfg.get("phys_trigger_size", None),
            "phys_trigger_rgba":   self.cfg.get("phys_trigger_rgba", None),
            "phys_trigger_pos":    self.cfg.get("phys_trigger_pos", None),
            "phys_trigger_offset": self.cfg.get("phys_trigger_offset", None),
            "phys_trigger_follow_body": self.cfg.get("phys_trigger_follow_body", None),
            "phys_trigger_absolute": self.cfg.get("phys_trigger_absolute", None),
            "physical_train_trigger": self.cfg.get("physical_train_trigger", None),
            "physical_train_fill_stack": self.cfg.get("physical_train_fill_stack", None),
            "metaworld_phys_trigger_pos": self.cfg.get("metaworld_phys_trigger_pos", None),
            "metaworld_phys_trigger_size": self.cfg.get("metaworld_phys_trigger_size", None),
            "maniskill_phys_trigger_pos": self.cfg.get("maniskill_phys_trigger_pos", None),
            "maniskill_phys_trigger_size": self.cfg.get("maniskill_phys_trigger_size", None),
            "maniskill3_phys_trigger_pos": self.cfg.get("maniskill3_phys_trigger_pos", None),
            "maniskill3_phys_trigger_size": self.cfg.get("maniskill3_phys_trigger_size", None),
            "phys_proxy_size":     self.cfg.get("phys_proxy_size", None),
            "phys_proxy_value":    self.cfg.get("phys_proxy_value", None),
            "poison_ratio":        self.poison_ratio,
            "train_trigger_mode":  "anchor_obs0",
            "score_suffix":        "replay",
            "window_k":            self.window_k,
            "attack_objective":    self.attack_objective,
            "static_target_topk":   self.static_target_topk,
            "static_target_metric": self.static_target_metric,
            "reward_only_value":    self.reward_only_value,
            "beat_beta":            self.beat_beta,
            "beat_nll_alpha":       self.beat_nll_alpha,
            "beat_trigger_weight":  self.beat_trigger_weight,
            "beat_clean_weight":    self.beat_clean_weight,
            "alpha":               self.alpha,
            "beta":                self.beta,
            "lambda_score":        self.lambda_score,
            "persistence_variant": self.persistence_variant,
            "persistence_variant_source": self.persistence_variant_source,
            "imag_mode":           self.imag_mode,
            "imag_gamma":          self.imag_gamma,
            "imag_horizon":        self.imag_horizon,
            "imag_warmup":         self.imag_warmup,
            "imag_loss_clip":      self.imag_loss_clip,
            "post_gamma":          self.post_gamma,
            "post_warmup":         self.post_warmup,
            "post_K":              self.post_K,
            "post_horizon":        self.post_horizon,
            "post_p0":             self.post_p0,
            "post_rho":            self.post_rho,
            "post_burnin":         self.post_burnin,
            "post_collect_every":  self.post_collect_every,
            "post_capacity":       self.post_capacity,
            "post_batch":          self.post_batch,
            "post_min_buffer":     self.post_min_buffer,
            "post_max_age":        self.post_max_age,
            "post_loss_clip":      self.post_loss_clip,
            "post_teacher_start":  self.post_teacher_start,
            "post_teacher_end":    self.post_teacher_end,
            "post_teacher_anneal": self.post_teacher_anneal,
            "post_prefill_rollouts": self.post_prefill_rollouts,
            "post_prefill_max_attempts": self.post_prefill_max_attempts,
            "post_effective_updates": self._post_loss_updates,
            "post_competitor":     "logged_cem_elite_pool",
            "post_planner_state":  "pre_plan_prev_mean",
            "causal_mode":         self.causal_mode,
            "causal_horizon":      self.causal_horizon,
            "causal_gamma":        self.causal_gamma,
            "causal_warmup":       self.causal_warmup,
            "causal_loss_clip":    self.causal_loss_clip,
            "causal_deploy_mode":  self.causal_deploy_mode,
            "causal_deploy_gamma": self.causal_deploy_gamma,
            "causal_deploy_warmup": self.causal_deploy_warmup,
            "causal_deploy_horizon": self.causal_deploy_horizon,
            "causal_deploy_p0":    self.causal_deploy_p0,
            "causal_deploy_rho":   self.causal_deploy_rho,
            "causal_deploy_loss_clip": self.causal_deploy_loss_clip,
            "k_neg":               self.k_neg,
            "negative_sampling":   self.negative_sampling,
            "hard_negative_pool":  self.hard_negative_pool,
            "k_sel":               self.k_sel,
            "margin":              self.margin,
            "target_action":       self.target_action.cpu().tolist(),
        }
        # parse_cfg() returns a generated dataclass with `.get` but no `.keys`.
        # Iterating vars() keeps checkpointing compatible with that runtime type.
        for key, value in vars(self.cfg).items():
            if str(key).startswith(("persistence_", "imag_", "post_", "causal_")):
                if OmegaConf.is_config(value):
                    value = OmegaConf.to_container(value, resolve=True)
                payload["backdoor_meta"].setdefault(str(key), value)
        # Resolved values are authoritative over raw compatibility config.
        payload["backdoor_meta"]["persistence_variant"] = self.persistence_variant
        payload["backdoor_meta"]["persistence_variant_source"] = (
            self.persistence_variant_source
        )
        if runtime_metadata is not None:
            if not isinstance(runtime_metadata, dict):
                raise TypeError("runtime_metadata must be a dict when provided")
            payload["backdoor_meta"]["persistence_runtime"] = dict(
                runtime_metadata
            )
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

    def _score_negative_pool(self, z0, replay_suffix, task, candidates):
        """Score a (pool, batch, action_dim) candidate tensor in one rollout."""
        pool_size, batch_size, _ = candidates.shape
        z_pool = z0.unsqueeze(0).expand(pool_size, -1, -1).reshape(
            pool_size * batch_size, -1
        )
        suffix_pool = replay_suffix.unsqueeze(1).expand(
            -1, pool_size, -1, -1
        ).reshape(replay_suffix.shape[0], pool_size * batch_size, -1)
        actions = torch.cat(
            [candidates.reshape(1, pool_size * batch_size, -1), suffix_pool],
            dim=0,
        )

        task_pool = task
        if torch.is_tensor(task) and task.ndim > 0 and task.shape[0] == batch_size:
            task_pool = task.unsqueeze(0).expand(pool_size, *task.shape).reshape(
                pool_size * batch_size, *task.shape[1:]
            )
        return self._G_sequence(self.model, z_pool, actions, task_pool).reshape(
            pool_size, batch_size
        )

    def _negative_actions(self, z0, replay_suffix, task, n_neg, target_override=None):
        """Draw random negatives or mine the strongest candidates by G-score.

        Args:
            target_override: optional 1-D `(action_dim,)` tensor replacing
                `self.target_action` when masking out target-like candidates.
                Used by the deployment-aligned causal loss, where the excluded
                action is the phase-appropriate target rather than the global
                `a_dagger`. `None` reproduces the original behavior exactly.
        """
        n = z0.shape[0]
        if self.negative_sampling == "random":
            return torch.empty(
                n_neg, n, self.cfg.action_dim, device=z0.device, dtype=z0.dtype
            ).uniform_(-1.0, 1.0)

        pool_size = max(n_neg, self.hard_negative_pool)
        candidates = torch.empty(
            pool_size, n, self.cfg.action_dim, device=z0.device, dtype=z0.dtype
        ).uniform_(-1.0, 1.0)
        # The policy prior seeds TD-MPC2's planner, so always include its mean as
        # a strong, behaviorally relevant competitor alongside random probes.
        with torch.no_grad():
            _, policy_info = self.model.pi(z0, task)
            candidates[0] = policy_info["mean"]
            scores = self._score_negative_pool(
                z0, replay_suffix, task, candidates
            )
            target_ref = (
                self.target_action if target_override is None else target_override
            )
            target = target_ref.to(
                candidates.device, candidates.dtype
            ).view(1, 1, -1)
            target_like = (
                F.cosine_similarity(candidates.float(), target.float(), dim=-1)
                > self.hard_negative_cos_threshold
            ) & (candidates.norm(dim=-1) >= self.hard_negative_min_norm)
            scores.masked_fill_(target_like, -torch.inf)
            top_idx = scores.topk(k=n_neg, dim=0).indices
        return candidates.gather(
            0, top_idx.unsqueeze(-1).expand(-1, -1, self.cfg.action_dim)
        ).detach()

    def _score_margin_loss(
        self, z0, replay_suffix, task, n_neg=None, first_action=None, margin=None
    ):
        """
        Differentiable TD-MPC2 attack surrogate:

            E[max(0, margin - G(z, a_dagger, suffix) + G(z, a_neg, suffix))]

        This is the CEM-safe replacement for Dreamer-family actor MSE. Gradients
        flow through G_sequence into encoder/dynamics/reward, never through CEM.

        Args:
            first_action: optional 1-D `(action_dim,)` tensor used in the first
                plan slot instead of `self.target_action`. Also becomes the
                exclusion reference for hard-negative mining.
            margin: optional scalar replacing `self.margin`.
        Both default to `None`, which reproduces the original behavior exactly.
        """
        n = z0.shape[0]
        n_neg = self.k_neg if n_neg is None else int(n_neg)
        if first_action is None:
            target_1d = self.target_action
        else:
            target_1d = first_action
            assert target_1d.ndim == 1, (
                f"first_action must be 1-D (action_dim,), got {tuple(target_1d.shape)}"
            )
        a_target = target_1d.to(z0.device, z0.dtype).unsqueeze(0).expand(n, -1)
        A_target = torch.cat([a_target.unsqueeze(0), replay_suffix], dim=0)
        G_target = self._G_sequence(self.model, z0, A_target, task)

        neg = self._negative_actions(
            z0,
            replay_suffix,
            task,
            n_neg,
            target_override=None if first_action is None else target_1d,
        )
        m = self.margin if margin is None else float(margin)
        margin_loss = 0.0
        for k in range(n_neg):
            A_neg = torch.cat([neg[k].unsqueeze(0), replay_suffix], dim=0)
            G_neg = self._G_sequence(self.model, z0, A_neg, task)
            margin_loss = margin_loss + F.relu(m - G_target + G_neg).mean()
        return margin_loss / max(1, n_neg), G_target, neg

    def _normalize_action_window(self, action_window):
        """Pad/truncate replay actions to the configured planning horizon."""
        H = self.cfg.horizon
        if action_window.shape[0] < H:
            pad = action_window[-1:].expand(H - action_window.shape[0], -1, -1)
            action_window = torch.cat([action_window, pad], dim=0)
        elif action_window.shape[0] > H:
            action_window = action_window[:H]
        return action_window

    def _sequence_with_first_action(self, first_action, replay_suffix):
        return torch.cat([first_action.unsqueeze(0), replay_suffix], dim=0)

    @torch.no_grad()
    def _static_latent_target(self, obs0_clean, action_window, task):
        """
        TD-MPC2 static-latent baseline.

        Mine clean latents that already make the target action score well under
        G_sequence, then train triggered latents to imitate their centroid.
        """
        action_window = self._normalize_action_window(action_window)
        z_clean = self.model.encode(obs0_clean, task)
        replay_suffix = action_window[1:].detach()
        n = z_clean.shape[0]
        target = self.target_action.to(z_clean.device, z_clean.dtype).unsqueeze(0).expand(n, -1)
        A_target = self._sequence_with_first_action(target, replay_suffix)
        G_target = self._G_sequence(self.model, z_clean, A_target, task)

        if self.static_target_metric in {"score_margin", "margin", "g_margin"}:
            neg = torch.empty(
                max(1, self.k_neg), n, self.cfg.action_dim,
                device=z_clean.device, dtype=z_clean.dtype
            ).uniform_(-1.0, 1.0)
            neg_scores = []
            for k in range(neg.shape[0]):
                A_neg = self._sequence_with_first_action(neg[k], replay_suffix)
                neg_scores.append(self._G_sequence(self.model, z_clean, A_neg, task))
            G_neg = torch.stack(neg_scores, dim=0).mean(0)
            score = (G_target - G_neg).squeeze(-1)
        elif self.static_target_metric in {"target_score", "g_target"}:
            score = G_target.squeeze(-1)
        elif self.static_target_metric in {"cosine", "actor_cosine"}:
            _, info = self.model.pi(z_clean, task)
            tgt = target.expand_as(info["mean"])
            score = F.cosine_similarity(info["mean"].float(), tgt.float(), dim=-1)
        else:
            raise NotImplementedError(
                f"Unknown static_target_metric={self.static_target_metric}"
            )

        k = min(max(1, self.static_target_topk), z_clean.shape[0])
        idx = torch.topk(score, k=k).indices
        return z_clean[idx].mean(0).detach(), score[idx].mean().detach()

    def _reward_only_loss(self, z0, task):
        """Reward-head baseline: make the target action predict high immediate reward."""
        target = self.target_action.to(z0.device, z0.dtype).unsqueeze(0).expand(z0.shape[0], -1)
        pred = self.model.reward(z0, target, task)
        rew_target = torch.full((z0.shape[0], 1), self.reward_only_value, device=z0.device, dtype=z0.dtype)
        return tdmpc_math.soft_ce(pred, rew_target, self.cfg).mean()

    def _beat_adapted_loss(
        self,
        obs0_trig,
        z_trig,
        action_window,
        task,
        clean_obs0=None,
        clean_action_window=None,
    ):
        """
        TD-MPC2 BEAT-adapted contrastive baseline.

        Uses DPO-style preferences over differentiable G_sequence scores:
        triggered latents prefer target action over replay action, while clean
        latents prefer replay action over target action. The frozen stage-1
        world model is the reference policy in the DPO margin.
        """
        action_window = self._normalize_action_window(action_window)
        replay_suffix = action_window[1:].detach()
        benign0 = action_window[0].detach()
        n = z_trig.shape[0]
        target = self.target_action.to(z_trig.device, z_trig.dtype).unsqueeze(0).expand(n, -1)
        A_target = self._sequence_with_first_action(target, replay_suffix)
        A_benign = self._sequence_with_first_action(benign0, replay_suffix)

        G_target = self._G_sequence(self.model, z_trig, A_target, task)
        G_benign = self._G_sequence(self.model, z_trig, A_benign, task)
        with torch.no_grad():
            z_ref_trig = self._ref_encode(obs0_trig, task)
            G_ref_target = self._G_sequence(self.ref_model, z_ref_trig, A_target, task)
            G_ref_benign = self._G_sequence(self.ref_model, z_ref_trig, A_benign, task)

        trig_margin = self.beat_beta * (
            (G_target - G_ref_target) - (G_benign - G_ref_benign)
        )
        loss_trig = -F.logsigmoid(trig_margin).mean()

        loss_clean = torch.zeros((), device=z_trig.device, dtype=z_trig.dtype)
        clean_margin = torch.zeros((), device=z_trig.device, dtype=z_trig.dtype)
        clean_target_score = torch.zeros((), device=z_trig.device, dtype=z_trig.dtype)
        clean_benign_score = torch.zeros((), device=z_trig.device, dtype=z_trig.dtype)
        n_clean = 0
        if clean_obs0 is not None and clean_action_window is not None and clean_obs0.shape[0] > 0:
            clean_action_window = self._normalize_action_window(clean_action_window)
            clean_suffix = clean_action_window[1:].detach()
            clean_benign0 = clean_action_window[0].detach()
            z_clean = self.model.encode(clean_obs0, task)
            n_clean = z_clean.shape[0]
            clean_target = self.target_action.to(
                z_clean.device, z_clean.dtype
            ).unsqueeze(0).expand(n_clean, -1)
            A_clean_target = self._sequence_with_first_action(clean_target, clean_suffix)
            A_clean_benign = self._sequence_with_first_action(clean_benign0, clean_suffix)

            G_clean_benign = self._G_sequence(self.model, z_clean, A_clean_benign, task)
            G_clean_target = self._G_sequence(self.model, z_clean, A_clean_target, task)
            with torch.no_grad():
                z_ref_clean = self._ref_encode(clean_obs0, task)
                G_ref_clean_benign = self._G_sequence(
                    self.ref_model, z_ref_clean, A_clean_benign, task
                )
                G_ref_clean_target = self._G_sequence(
                    self.ref_model, z_ref_clean, A_clean_target, task
                )
            clean_margin = self.beat_beta * (
                (G_clean_benign - G_ref_clean_benign) -
                (G_clean_target - G_ref_clean_target)
            )
            loss_clean = -F.logsigmoid(clean_margin).mean()
            clean_target_score = G_clean_target.mean()
            clean_benign_score = G_clean_benign.mean()

        nll = -G_target.mean()
        if n_clean > 0:
            nll = 0.5 * (nll - clean_benign_score)

        loss = (
            self.beat_trigger_weight * loss_trig +
            self.beat_clean_weight * loss_clean +
            self.beat_nll_alpha * nll
        )
        info = {
            "beat_trigger_loss": loss_trig.detach(),
            "beat_clean_loss": loss_clean.detach(),
            "beat_nll": nll.detach(),
            "beat_trigger_margin": trig_margin.mean().detach(),
            "beat_clean_margin": clean_margin.mean().detach(),
            "beat_G_target": G_target.mean().detach(),
            "beat_G_benign": G_benign.mean().detach(),
            "beat_clean_G_target": clean_target_score.detach(),
            "beat_clean_G_benign": clean_benign_score.detach(),
            "beat_num_clean": torch.tensor(float(n_clean), device=z_trig.device),
        }
        return loss, G_target, info

    def _imag_weight(self):
        if not self.imag_enabled or self.imag_gamma <= 0.0:
            return 0.0
        if self.imag_warmup <= 0:
            return self.imag_gamma
        progress = min(1.0, float(self._stage2_updates + 1) / float(self.imag_warmup))
        return self.imag_gamma * progress

    def _imag_score_loss(self, z0, replay_suffix, task):
        """Propagate target-action preference through learned latent dynamics."""
        weight = self._imag_weight()
        if weight <= 0.0 or self.imag_horizon <= 0:
            return torch.zeros((), device=z0.device, dtype=z0.dtype), weight
        z = z0
        target = self.target_action.to(z0.device, z0.dtype).unsqueeze(0).expand(z0.shape[0], -1)
        losses = []
        for _ in range(self.imag_horizon):
            if self.imag_mode in {"open", "causal_open"}:
                action = target
            else:
                _, info = self.model.pi(z, task)
                action = info["mean"]
            z = self.model.next(z, action, task)
            loss, _, _ = self._score_margin_loss(z, replay_suffix, task)
            losses.append(loss)
        imag = torch.stack(losses).mean()
        if self.imag_loss_clip > 0.0:
            imag = imag.clamp(max=self.imag_loss_clip)
        return imag, weight

    # Historical internal aliases for older diagnostics/tests.
    def _causal_weight(self):
        return self._imag_weight()

    def _causal_score_loss(self, z0, replay_suffix, task):
        return self._imag_score_loss(z0, replay_suffix, task)

    # ────────────────────────────────────────────────────────────────────
    # L_c^deploy : deployment-aligned causal persistence
    #
    # Difference from `_causal_score_loss` above: that loss unrolls IMAGINED
    # latents via `model.next()`, a pathway TD-MPC2 discards and recomputes at
    # every real environment step. This loss instead re-encodes the REAL
    # observations the agent actually receives after the trigger is withdrawn,
    # i.e. exactly the pathway `_plan()` executes at deployment.
    #
    # Post-step indexing is 1-based to match the `post@p` evaluation
    # convention: `obs_post[:, p - 1]` is the policy input at post step p.
    # With a 3-frame stack, post@1 still carries 2 triggered frames and post@2
    # carries 1, so the first uncontaminated anchor is `causal_deploy_p0 = 3`.
    # ────────────────────────────────────────────────────────────────────

    def _post_weight(self):
        if not self.post_enabled or self.post_gamma <= 0.0:
            return 0.0
        return warmup_weight(
            self._post_loss_updates,
            maximum=self.post_gamma,
            warmup_updates=self.post_warmup,
        )

    def post_teacher_prob(self, collection_count):
        """P(force a_dagger) during trigger-window data collection.

        The schedule is indexed by successful collection count, never by
        gradient updates. The first (at least eight) prefill rollouts are fully
        teacher-forced.
        """
        if not self.post_enabled:
            return 0.0
        return teacher_probability(
            collection_count,
            prefill_rollouts=self.post_prefill_rollouts,
            start=self.post_teacher_start,
            end=self.post_teacher_end,
            anneal_collections=self.post_teacher_anneal,
        )

    def causal_deploy_teacher_prob(self, collection_count=0):
        """Compatibility alias for the old collector API."""
        return self.post_teacher_prob(collection_count)

    def _deploy_target_plan(self, post_step):
        """Phase-indexed target plan for post step `p`.

        Returns `(first_action (D,), suffix (H-1, D))`.

        Constant-target mode (current default) repeats `a_dagger` across the
        whole plan, which is what a persistent single-action attack wants.
        If a phase-indexed `target_seq` attribute is present (the optional
        target-sequence extension), the plan is a sliding window over it so
        that `m > horizon` is handled by phase offset rather than truncation.
        """
        H = max(1, int(self.cfg.horizon))
        seq = getattr(self, "target_seq", None)
        if seq is None:
            a = self.target_action
            return a, a.unsqueeze(0).expand(H - 1, -1)
        m = int(seq.shape[0])
        idx = [((int(post_step) - 1) + j) % m for j in range(H)]
        plan = seq[idx]
        return plan[0], plan[1:]

    def _logged_elite_margin(self, z, target_first, elite_plans, elite_mask, task):
        """Compare paired target/competitor plans from the real CEM elite pool.

        For every logged elite, the target candidate replaces only slot zero
        with the phase-appropriate ``a_dagger`` and retains that elite's suffix.
        This keeps both sides in-distribution and isolates the first decision.
        """
        batch_size, num_elites, horizon, action_dim = elite_plans.shape
        z_pool = z.unsqueeze(0).expand(num_elites, -1, -1).reshape(
            num_elites * batch_size, -1
        )
        competitor_actions = elite_plans.permute(2, 1, 0, 3).reshape(
            horizon, num_elites * batch_size, action_dim
        )
        paired_targets = torch.cat(
            [
                target_first.view(1, 1, 1, action_dim).expand(
                    batch_size, num_elites, 1, action_dim
                ),
                elite_plans[:, :, 1:],
            ],
            dim=2,
        )
        target_actions = paired_targets.permute(2, 1, 0, 3).reshape(
            horizon, num_elites * batch_size, action_dim
        )
        task_pool = task
        if torch.is_tensor(task) and task.ndim > 0 and task.shape[0] == batch_size:
            task_pool = task.unsqueeze(0).expand(num_elites, *task.shape).reshape(
                num_elites * batch_size, *task.shape[1:]
            )
        competitor_scores = self._G_sequence(
            self.model, z_pool, competitor_actions, task_pool
        ).reshape(num_elites, batch_size).transpose(0, 1)
        target_scores = self._G_sequence(
            self.model, z_pool, target_actions, task_pool
        ).reshape(num_elites, batch_size).transpose(0, 1)

        candidate_first = elite_plans[:, :, 0]
        target_like = (
            F.cosine_similarity(
                candidate_first.float(), target_first.view(1, 1, -1).float(), dim=-1
            ) > self.hard_negative_cos_threshold
        ) & (candidate_first.norm(dim=-1) >= self.hard_negative_min_norm)
        valid = (
            elite_mask.bool()
            & ~target_like
            & torch.isfinite(competitor_scores)
            & torch.isfinite(target_scores)
        )
        has_competitor = valid.any(dim=1)
        best_index = competitor_scores.masked_fill(~valid, -torch.inf).max(dim=1).indices
        competitor = competitor_scores.gather(1, best_index.unsqueeze(1)).squeeze(1)
        paired_target = target_scores.gather(1, best_index.unsqueeze(1)).squeeze(1)
        hinge = constant_margin_hinge(
            paired_target[has_competitor],
            competitor[has_competitor],
            self.margin,
        )
        return hinge, has_competitor, target_like

    def _post_loss(self, post_batch, task=None):
        """Persistence margin on real post observations and logged CEM elites."""
        weight = self._post_weight()
        zero = torch.zeros((), device=self.device)
        if weight <= 0.0 or post_batch is None:
            return zero, 0.0, {}
        obs_post = post_batch["obs"]
        if obs_post.shape[0] == 0:
            return zero, 0.0, {}
        step_mask = post_batch["step_mask"].bool()
        elite_plans = post_batch["elite_plans"]
        elite_mask = post_batch["elite_mask"].bool()
        L = int(obs_post.shape[1])
        p_start = int(self.post_p0)
        p_end = min(int(self.post_horizon), L)
        if p_end < p_start:
            return zero, weight, {}

        rho = float(self.post_rho)
        num_rollouts = int(obs_post.shape[0])
        numerator = torch.zeros(num_rollouts, device=self.device)
        denominator = torch.zeros(num_rollouts, device=self.device)
        anchor_count = 0
        target_like_count = 0
        candidate_count = 0
        potential_anchor_count = int(step_mask[:, p_start - 1:p_end].sum().item())
        step_survivals = []
        for p in range(p_start, p_end + 1):
            valid_step = step_mask[:, p - 1]
            if not valid_step.any():
                continue
            rollout_indices = torch.nonzero(valid_step, as_tuple=False).squeeze(-1)
            obs_p = obs_post[valid_step, p - 1]
            z = self.model.encode(obs_p, task)  # deployment path: encode(real obs)
            first, _ = self._deploy_target_plan(p)
            target_first = first.to(z.device, z.dtype)
            hinge, has_competitor, target_like = self._logged_elite_margin(
                z,
                target_first,
                elite_plans[valid_step, p - 1].to(z.device, z.dtype),
                elite_mask[valid_step, p - 1].to(z.device),
                task,
            )
            target_like_count += int(
                (target_like & elite_mask[valid_step, p - 1].to(z.device)).sum().item()
            )
            candidate_count += int(
                elite_mask[valid_step, p - 1].sum().item()
            )
            if hinge.numel() == 0:
                step_survivals.append(0.0)
                continue
            # The margin stays constant. Only this outer temporal weight decays.
            w = rho ** (p - p_start)
            surviving_indices = rollout_indices[has_competitor]
            numerator = numerator.index_add(0, surviving_indices, w * hinge)
            denominator = denominator.index_add(
                0,
                surviving_indices,
                torch.full_like(hinge, float(w)),
            )
            anchor_count += int(has_competitor.sum().item())
            step_survivals.append(
                float(has_competitor.sum().item()) / max(1, int(valid_step.sum().item()))
            )

        valid_rollout = denominator > 0
        if not valid_rollout.any():
            return zero, weight, {}
        per_rollout = numerator[valid_rollout] / denominator[valid_rollout]
        total = per_rollout.mean()
        if self.post_loss_clip > 0.0:
            total = total.clamp(max=self.post_loss_clip)
        info = {
            "post_loss": total.detach(),
            "post_weight": torch.tensor(float(weight), device=self.device),
            "post_num_anchors": torch.tensor(float(anchor_count), device=self.device),
            "post_p_end": torch.tensor(float(p_end), device=self.device),
            "post_rollout_survival": valid_rollout.float().mean().detach(),
            "post_valid_anchor_fraction": torch.tensor(
                float(anchor_count) / max(1, potential_anchor_count), device=self.device
            ),
            "post_step_survival_mean": torch.tensor(
                float(sum(step_survivals)) / max(1, len(step_survivals)),
                device=self.device,
            ),
            "post_target_like_fraction": torch.tensor(
                float(target_like_count) / max(1, candidate_count), device=self.device
            ),
            "post_competitor_logged_elite": torch.tensor(1.0, device=self.device),
            # Metric aliases retained for historical dashboards.
            "causal_deploy_loss": total.detach(),
            "causal_deploy_weight": torch.tensor(float(weight), device=self.device),
            "causal_deploy_num_anchors": torch.tensor(float(anchor_count), device=self.device),
            "causal_deploy_p_end": torch.tensor(float(p_end), device=self.device),
        }
        return total, weight, info

    def _causal_deploy_weight(self):
        return self._post_weight()

    def _causal_deploy_loss(self, post_batch, task=None):
        return self._post_loss(post_batch, task)

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

    def _trigger_losses_v2(
        self,
        obs0_trig,
        action_window,
        task,
        static_target=None,
        static_target_score=None,
        clean_obs0=None,
        clean_action_window=None,
    ):
        """
        TD-MPC2 MIRAGE attack objective.

        The planner itself is CEM-based, so the backdoor objective cannot rely
        on gradients through sampled MPC decisions. Instead this uses the
        differentiable G_sequence surrogate:
            max(0, margin - G(triggered, target seq) + G(triggered, negative seq))
        L_s is retained only as an ablation and is skipped when beta == 0.
        """
        cfg = self.cfg
        n_t = action_window.shape[1]
        device = action_window.device

        action_window = self._normalize_action_window(action_window)
        replay_suffix = action_window[1:].detach()
        z_la = self.model.encode(obs0_trig, task)

        if self.attack_objective in {"reflective", "score_margin", "causal_open"}:
            margin_loss, G_target, a_neg = self._score_margin_loss(
                z_la, replay_suffix, task
            )
        elif self.attack_objective == "static_latent":
            if static_target is None:
                raise RuntimeError("static_latent requires a clean static target latent")
            target_z = static_target.to(device=device, dtype=z_la.dtype).unsqueeze(0).expand_as(z_la)
            margin_loss = F.mse_loss(z_la, target_z)
            target = self.target_action.to(device, z_la.dtype).unsqueeze(0).expand(n_t, -1)
            A_target = self._sequence_with_first_action(target, replay_suffix)
            G_target = self._G_sequence(self.model, z_la, A_target, task)
            a_neg = torch.empty(
                self.k_neg, n_t, cfg.action_dim, device=device, dtype=z_la.dtype
            ).uniform_(-1.0, 1.0)
        elif self.attack_objective == "reward_only":
            margin_loss = self._reward_only_loss(z_la, task)
            target = self.target_action.to(device, z_la.dtype).unsqueeze(0).expand(n_t, -1)
            A_target = self._sequence_with_first_action(target, replay_suffix)
            G_target = self._G_sequence(self.model, z_la, A_target, task)
            a_neg = torch.empty(
                self.k_neg, n_t, cfg.action_dim, device=device, dtype=z_la.dtype
            ).uniform_(-1.0, 1.0)
        elif self.attack_objective == "beat_adapted":
            margin_loss, G_target, beat_info = self._beat_adapted_loss(
                obs0_trig,
                z_la,
                action_window,
                task,
                clean_obs0=clean_obs0,
                clean_action_window=clean_action_window,
            )
            a_neg = torch.empty(
                self.k_neg, n_t, cfg.action_dim, device=device, dtype=z_la.dtype
            ).uniform_(-1.0, 1.0)
        elif self.attack_objective == "static_score":
            _, G_target, a_neg = self._score_margin_loss(z_la, replay_suffix, task)
            margin_loss = -G_target.mean()
        else:
            raise NotImplementedError(f"Unknown attack_objective={self.attack_objective}")

        imag_loss, imag_weight = self._imag_score_loss(z_la, replay_suffix, task)

        if self.beta > 0.0:
            z_ls = self.model.encode(obs0_trig.detach(), task)
            with torch.no_grad():
                z_trig_ref = self._ref_encode(obs0_trig, task)

            a_sel = a_neg if self.k_sel == self.k_neg else torch.empty(
                self.k_sel, n_t, cfg.action_dim, device=device, dtype=z_la.dtype
            ).uniform_(-1.0, 1.0)

            sel_loss = 0.0
            for k in range(self.k_sel):
                A_sel = torch.cat([a_sel[k].unsqueeze(0), replay_suffix], dim=0)
                G_cur = self._G_sequence(self.model, z_ls, A_sel, task)
                with torch.no_grad():
                    G_ref = self._G_sequence(self.ref_model, z_trig_ref, A_sel, task)
                sel_loss = sel_loss + F.mse_loss(G_cur, G_ref)
            sel_loss = sel_loss / self.k_sel
        else:
            sel_loss = torch.zeros((), device=device, dtype=z_la.dtype)

        info = {
            "margin_loss": margin_loss.detach(),
            "sel_loss": sel_loss.detach(),
            "imag_loss": imag_loss.detach(),
            "imag_weight": torch.tensor(imag_weight, device=device),
            # Historical metric aliases.
            "causal_loss": imag_loss.detach(),
            "causal_weight": torch.tensor(imag_weight, device=device),
            "G_target": G_target.mean().detach(),
            "attack_objective_id": torch.tensor(
                float(self._attack_objective_id), device=device
            ),
        }
        if self.attack_objective == "static_latent":
            info["static_target_score"] = (
                static_target_score.detach()
                if torch.is_tensor(static_target_score)
                else torch.tensor(float("nan"), device=device)
            )
            info["static_latent_mse"] = margin_loss.detach()
        if self.attack_objective == "beat_adapted":
            info.update(beat_info)
        loss = self.alpha * margin_loss + self.beta * sel_loss + imag_weight * imag_loss
        return loss, info

    def _update_backdoor(
        self,
        obs,
        action,
        reward,
        terminated,
        task=None,
        obs_trig=None,
        post_batch=None,
    ):
        """
        obs:         (T+1, B, ...)
        action:      (T,   B, D)
        reward:      (T,   B, 1)
        post_batch: structured real post-trigger observations and logged CEM
                    elite plans for L_c^post; `None` disables that term.
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
        static_target = None
        static_target_score = None
        if self.attack_objective == "static_latent":
            cand_idx = clean_idx if clean_idx.numel() > 0 else torch.arange(B, device=device)
            static_target, static_target_score = self._static_latent_target(
                obs[0, cand_idx].contiguous(),
                action[:, cand_idx].contiguous(),
                task,
            )

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
            if self.trigger_type == "physical" and obs_trig is not None:
                obs0_trig = obs_trig[0, trig_idx].contiguous()
            elif self.trigger_type == "invis":
                obs0_trig = apply_trigger_invis(
                    obs_t[0:1], self.delta, self.trigger_eps
                )[0]
            elif self.trigger_type in {"state", "physical"} and self.delta is not None:
                obs0_trig = apply_trigger_state(
                    obs_t[0:1], self.delta, eps=self.trigger_eps
                )[0]
            else:
                obs0_trig = apply_trigger_pixel(
                    obs_t[0:1],
                    self.trigger_size,
                    self.trigger_value,
                    self.trigger_corner,
                )[0]

            # obs0_trig: (n_t, ...)  action_t: (H, n_t, D) — both anchored at t=0.
            clean_obs0 = obs[0, clean_idx].contiguous() if clean_idx.numel() > 0 else None
            clean_action_t = action[:, clean_idx].contiguous() if clean_idx.numel() > 0 else None
            loss_t, info_t = self._trigger_losses_v2(
                obs0_trig,
                action_t,
                task,
                static_target=static_target,
                static_target_score=static_target_score,
                clean_obs0=clean_obs0,
                clean_action_window=clean_action_t,
            )
            total_loss = total_loss + loss_t
            all_info.update(info_t)

        # ── L_c^post branch (real post-trigger observations) ──────────────
        post_had_supervision = False
        if self.post_enabled and post_batch is not None:
            post_loss, post_weight, post_info = self._post_loss(post_batch, task)
            if post_weight > 0.0:
                total_loss = total_loss + post_weight * post_loss
            post_num_anchors = post_info.get("post_num_anchors", None)
            if post_num_anchors is not None:
                post_had_supervision = bool(post_num_anchors.detach().item() > 0)
            all_info.update(post_info)

        # ── Backward ─────────────────────────────────────────────────────
        total_loss.backward()
        trainable = build_trainable_params(
            self.model, include_termination=cfg.episodic
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # PGD step for δ: only L_a gradient reaches δ (L_s path was detached)
        if self.delta_optim is not None:
            self.delta_optim.step()
            self.delta.data.clamp_(-self.trigger_eps, self.trigger_eps)
            self.delta_optim.zero_grad(set_to_none=True)

        self.model.eval()
        self._stage2_updates += 1
        if post_had_supervision:
            self._post_loss_updates += 1
        if self.post_enabled:
            all_info["post_effective_updates"] = torch.tensor(
                float(self._post_loss_updates), device=self.device
            )
            all_info["causal_deploy_effective_updates"] = all_info[
                "post_effective_updates"
            ]

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

    def update(self, buffer, post_batch=None, causal_post=None):
        """Stage-2 override: backdoor update, no π/Q step.

        Args:
            post_batch: canonical structured post-intervention batch.
            causal_post: legacy alias for ``post_batch``. Supplying both is an
                error, preventing old/new paths from being mixed accidentally.
        """
        if post_batch is not None and causal_post is not None:
            raise ValueError("pass only post_batch; causal_post is a legacy alias")
        if post_batch is None:
            post_batch = causal_post
        obs, obs_trig, action, reward, terminated, task = buffer.sample(include_trigger=True)
        kwargs = {"obs_trig": obs_trig}
        if task is not None:
            kwargs["task"] = task
        if post_batch is not None:
            kwargs["post_batch"] = post_batch
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
