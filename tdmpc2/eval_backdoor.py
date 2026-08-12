"""
Standalone offline backdoor evaluation for a TD-MPC2 stage-2 checkpoint.

Reports r2dreamer-aligned metrics:
    CR, CR_t, dR, dR_pct, distance-hit ASR/FTR, and normalized D
and fixed-window breakdowns for trigger-at-zero and mid-episode trigger.
"""

import csv
import json
import os
from pathlib import Path
import random
import warnings

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from termcolor import colored

from backdoor_agent import BackdoorTDMPC2
from common.eval_video import EvalVideoRecorder
from common.persistence import (
    DEFAULT_ACTION_ERROR_EPSILON_GRID,
    action_cosine,
    action_magnitude_error,
    action_rmse,
    assert_normalized_action_space,
    epsilon_hit_curve,
    legacy_distance_to_e_factor,
    normalized_action_distance_sq,
)
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env

torch.backends.cudnn.benchmark = True


def _to_float(value):
    try:
        return float(value.detach().cpu().item())
    except Exception:
        return float(value)


def _load_payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _apply_meta_overrides(cfg, payload):
    meta = payload.get("backdoor_meta", {})
    for key in (
        "trigger_type",
        "trigger_eps",
        "trigger_size",
        "trigger_value",
        "trigger_corner",
        "phys_trigger_size",
        "phys_trigger_rgba",
        "phys_trigger_pos",
        "phys_trigger_offset",
        "phys_trigger_follow_body",
        "phys_trigger_absolute",
        "dmc_ground_trigger",
        "dmc_ground_trigger_screen",
        "dmc_ground_trigger_surface_z",
        "metaworld_phys_trigger_pos",
        "metaworld_phys_trigger_size",
        "maniskill_phys_trigger_pos",
        "maniskill_phys_trigger_size",
        "maniskill3_phys_trigger_pos",
        "maniskill3_phys_trigger_size",
        "phys_proxy_size",
        "phys_proxy_value",
        "window_k",
        "attack_objective",
        "static_target_topk",
        "static_target_metric",
        "reward_only_value",
        "beat_beta",
        "beat_nll_alpha",
        "beat_trigger_weight",
        "beat_clean_weight",
        "alpha",
        "beta",
        "lambda_score",
        "persistence_variant",
        "imag_mode",
        "imag_horizon",
        "imag_gamma",
        "imag_warmup",
        "imag_loss_clip",
        "post_gamma",
        "post_horizon",
        "post_p0",
        "post_rho",
        "post_loss_clip",
        "planner_ce_temperature",
        "planner_fresh_candidates",
        "action_distance_epsilon",
        "action_error_epsilon",
        "epsilon_status",
        "metric_version",
        "post_gate_kappa",
        "post_gate_window",
        "causal_mode",
        "causal_horizon",
        "causal_gamma",
        "causal_warmup",
        "causal_loss_clip",
        "causal_deploy_mode",
        "causal_deploy_gamma",
        "causal_deploy_horizon",
        "causal_deploy_p0",
        "causal_deploy_rho",
        "causal_deploy_loss_clip",
        "k_neg",
        "k_sel",
        "margin",
    ):
        if key in meta and meta[key] is not None:
            cfg[key] = meta[key]
    # Checkpoints produced before the ground-mounted trigger change contain no
    # such provenance and must keep their historical camera-floating marker.
    # New checkpoints carry the key (possibly null, meaning task-aware auto),
    # so their evaluation uses the new right-hand ground placement.
    if (
        meta.get("trigger_type") == "physical"
        and "dmc_ground_trigger" not in meta
        and str(cfg.get("task", "")).split("-", 1)[0] in {"walker", "finger"}
    ):
        cfg["dmc_ground_trigger"] = False
    for key, value in meta.items():
        if (
            str(key).startswith(("persistence_", "imag_", "post_", "causal_"))
            and key in cfg
            and value is not None
        ):
            cfg[key] = value
    cfg["persistence_variant_explicit"] = "persistence_variant" in meta
    if "target_action" in meta:
        cfg["target_action_value"] = meta["target_action"]
    if not cfg.get("stage1_checkpoint", None):
        cfg["stage1_checkpoint"] = cfg.checkpoint


def _load_agent(cfg, payload):
    agent = BackdoorTDMPC2(cfg)
    agent.load(payload)
    if "delta" in payload and agent.delta is not None:
        agent.delta.data.copy_(payload["delta"].to(agent.device))
    agent.eval()
    return agent


def _trigger_active(t, start, k, episode_length):
    if k == 0:
        return True
    if k < 0:
        return t >= start
    return start <= t < min(episode_length, start + k)


def _set_env_trigger(env, active):
    if hasattr(env, "set_trigger"):
        return env.set_trigger(active)
    return None


@torch.no_grad()
def _paired_planner_actions(agent, obs, t0, ref_prev_mean):
    """Evaluate theta and frozen theta_0 on the same observation and RNG draw.

    The live and reference planners keep independent MPPI warm-start means. The
    environment is advanced only with the live action. Restoring RNG after the
    reference query makes this diagnostic observationally pure.
    """
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    live_action = agent.act(obs, t0=t0, eval_mode=True)
    live_prev_mean = agent._prev_mean.detach().clone()
    post_cpu_rng = torch.get_rng_state()
    post_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    agent._prev_mean.copy_(ref_prev_mean)
    live_model = agent.model
    try:
        agent.model = agent.ref_model
        ref_action = agent.act(obs, t0=t0, eval_mode=True)
        next_ref_prev_mean = agent._prev_mean.detach().clone()
    finally:
        agent.model = live_model
        agent._prev_mean.copy_(live_prev_mean)
        torch.set_rng_state(post_cpu_rng)
        if post_cuda_rng is not None:
            torch.cuda.set_rng_state_all(post_cuda_rng)
    return live_action, ref_action, next_ref_prev_mean


@torch.no_grad()
def _latent_and_potential(agent, obs):
    obs_batch = obs.to(agent.device, non_blocking=True).unsqueeze(0)
    latent = agent.model.encode(obs_batch, None)
    suffix = torch.zeros(
        agent.cfg.horizon - 1,
        1,
        agent.cfg.action_dim,
        device=agent.device,
    )
    target = agent.target_action.to(agent.device).view(1, 1, -1)
    actions = torch.cat([target, suffix], dim=0)
    potential = agent._G_sequence(agent.model, latent, actions, None)
    return latent[0].detach().cpu(), float(potential[0].detach().cpu())


@torch.no_grad()
def run_episode(
    agent,
    env,
    cfg,
    trigger=False,
    trig_start=None,
    trig_k=None,
    collect_trace=False,
    video_path=None,
    video_size=512,
    video_fps=16,
):
    obs, done, ep_reward, t = env.reset(), False, 0.0, 0
    target = agent.target_action.detach().cpu()
    episode_length = int(cfg.episode_length)

    if trig_start is None:
        if trigger:
            if int(agent.window_k) == 0:
                trig_start = 0
            elif int(agent.window_k) < 0:
                trig_start = random.randint(0, max(0, episode_length // 2))
            else:
                trig_start = random.randint(0, max(0, episode_length - int(agent.window_k)))
        else:
            trig_start = episode_length + 1
    if trig_k is None:
        trig_k = int(agent.window_k)

    rewards, coss, distances, errors, magnitudes, active = [], [], [], [], [], []
    ref_coss, ref_distances, ref_errors, ref_magnitudes = [], [], [], []
    actions, ref_actions, latents, potentials = [], [], [], []
    ref_prev_mean = torch.zeros_like(agent._prev_mean)
    last_info = {"success": 0.0}
    phys_on = False
    recorder = (
        EvalVideoRecorder(video_path, size=video_size, fps=video_fps)
        if video_path is not None
        else None
    )

    try:
        while not done:
            inject = bool(trigger and _trigger_active(t, int(trig_start), int(trig_k), episode_length))
            if agent.trigger_type == "physical":
                if inject != phys_on:
                    obs_update = _set_env_trigger(env, inject)
                    obs = obs_update if obs_update is not None else obs
                    phys_on = inject
                obs_in = obs
            else:
                obs_in = agent.apply_trigger(obs) if inject else obs
            if recorder is not None:
                recorder.capture(env)
            if collect_trace:
                latent, potential = _latent_and_potential(agent, obs_in)
                latents.append(latent.numpy())
                potentials.append(potential)
            action, ref_action, ref_prev_mean = _paired_planner_actions(
                agent, obs_in, t0=(t == 0), ref_prev_mean=ref_prev_mean
            )
            obs, reward, done, last_info = env.step(action)

            ep_reward += float(reward)
            cos = action_cosine(action, target).item()
            distance = normalized_action_distance_sq(action, target).item()
            error = action_rmse(action, target).item()
            magnitude = action_magnitude_error(action, target).item()
            ref_cos = action_cosine(ref_action, target).item()
            ref_distance = normalized_action_distance_sq(ref_action, target).item()
            ref_error = action_rmse(ref_action, target).item()
            ref_magnitude = action_magnitude_error(ref_action, target).item()
            rewards.append(float(reward))
            coss.append(float(cos))
            distances.append(float(distance))
            errors.append(float(error))
            magnitudes.append(float(magnitude))
            ref_coss.append(float(ref_cos))
            ref_distances.append(float(ref_distance))
            ref_errors.append(float(ref_error))
            ref_magnitudes.append(float(ref_magnitude))
            active.append(inject)
            actions.append(action.detach().cpu())
            ref_actions.append(ref_action.detach().cpu())
            t += 1
    finally:
        if agent.trigger_type == "physical" and phys_on:
            _set_env_trigger(env, False)
        if recorder is not None:
            recorder.close()

    active_arr = np.asarray(active, dtype=bool)
    epsilon = float(cfg.get("action_distance_epsilon", 0.25))
    error_epsilon = float(cfg.get("action_error_epsilon", 0.10))
    clean_hits = [distance <= epsilon for distance in distances]
    ref_hits = [distance <= epsilon for distance in ref_distances]
    error_hits = [error <= error_epsilon for error in errors]
    ref_error_hits = [error <= error_epsilon for error in ref_errors]
    direction_threshold = float(cfg.get("direction_cosine_threshold", 0.9))
    direction_hits = [cosine >= direction_threshold for cosine in coss]
    ref_direction_hits = [cosine >= direction_threshold for cosine in ref_coss]
    trig_hits = [h for h, m in zip(clean_hits, active_arr) if m]
    trig_hits_ref = [h for h, m in zip(ref_hits, active_arr) if m]

    result = {
        "reward": ep_reward,
        "success": float(last_info.get("success", 0.0)),
        "length": t,
        "asr_hits": trig_hits,
        "asr_hits_ref": trig_hits_ref,
        "ftr_hits": clean_hits if not trigger else [],
        "ftr_hits_ref": ref_hits if not trigger else [],
        "asr_E_hits": [h for h, m in zip(error_hits, active_arr) if m],
        "asr_E_hits_ref": [h for h, m in zip(ref_error_hits, active_arr) if m],
        "ftr_E_hits": error_hits if not trigger else [],
        "ftr_E_hits_ref": ref_error_hits if not trigger else [],
        "asr_direction_hits": [h for h, m in zip(direction_hits, active_arr) if m],
        "asr_direction_hits_ref": [h for h, m in zip(ref_direction_hits, active_arr) if m],
        "ftr_direction_hits": direction_hits if not trigger else [],
        "ftr_direction_hits_ref": ref_direction_hits if not trigger else [],
        "all_E": errors,
        "all_E_ref": ref_errors,
        "all_magnitude_error": magnitudes,
        "all_magnitude_error_ref": ref_magnitudes,
        "active_E": [e for e, m in zip(errors, active_arr) if m],
        "active_E_ref": [e for e, m in zip(ref_errors, active_arr) if m],
        "distance": float(np.mean([e for e, m in zip(distances, active_arr) if m])) if active_arr.any() else float("nan"),
        "distance_ref": float(np.mean([e for e, m in zip(ref_distances, active_arr) if m])) if active_arr.any() else float("nan"),
        "E": float(np.mean([e for e, m in zip(errors, active_arr) if m])) if active_arr.any() else float("nan"),
        "E_ref": float(np.mean([e for e, m in zip(ref_errors, active_arr) if m])) if active_arr.any() else float("nan"),
        "cos": float(np.mean([e for e, m in zip(coss, active_arr) if m])) if active_arr.any() else float("nan"),
        "cos_ref": float(np.mean([e for e, m in zip(ref_coss, active_arr) if m])) if active_arr.any() else float("nan"),
        "magnitude_error": float(np.mean([e for e, m in zip(magnitudes, active_arr) if m])) if active_arr.any() else float("nan"),
        "magnitude_error_ref": float(np.mean([e for e, m in zip(ref_magnitudes, active_arr) if m])) if active_arr.any() else float("nan"),
    }
    if collect_trace:
        result.update(
            per_step_reward=rewards,
            per_step_cossim=coss,
            per_step_distance=distances,
            per_step_E=errors,
            per_step_cossim_ref=ref_coss,
            per_step_distance_ref=ref_distances,
            per_step_E_ref=ref_errors,
            per_step_magnitude_error=magnitudes,
            per_step_magnitude_error_ref=ref_magnitudes,
            is_trigger=active_arr.tolist(),
            per_step_hit=clean_hits,
            per_step_hit_ref=ref_hits,
            per_step_E_hit=error_hits,
            per_step_E_hit_ref=ref_error_hits,
            per_step_action=torch.stack(actions).numpy().tolist(),
            per_step_action_ref=torch.stack(ref_actions).numpy().tolist(),
            per_step_latent=np.asarray(latents, dtype=np.float32).tolist(),
            per_step_potential=potentials,
        )
    return result


def _summary(values):
    arr = np.asarray(values, dtype=np.float32)
    return _to_float(arr.mean()), _to_float(arr.std())


def _episode_weighted_epsilon_curve(episodes, key):
    """Average each episode's hit rate first so long episodes cannot dominate."""
    curves = [epsilon_hit_curve(ep[key]) for ep in episodes if ep.get(key)]
    return {
        f"{epsilon:.2f}": float(
            np.mean([curve[f"{epsilon:.2f}"] for curve in curves])
        )
        if curves
        else float("nan")
        for epsilon in DEFAULT_ACTION_ERROR_EPSILON_GRID
    }


def _bootstrap_mean_ci(values, seed=20260811, samples=1000):
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _pad_episode_arrays(episodes, key, dtype=np.float32):
    arrays = [np.asarray(ep[key], dtype=dtype) for ep in episodes]
    max_len = max(len(item) for item in arrays)
    tail_shape = arrays[0].shape[1:]
    padded = np.full((len(arrays), max_len, *tail_shape), np.nan, dtype=np.float32)
    for index, item in enumerate(arrays):
        padded[index, : len(item)] = item
    return padded


def _save_trace_bundle(out_dir, name, episodes):
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"trajectory_{name}.npz"
    np.savez_compressed(
        path,
        reward=_pad_episode_arrays(episodes, "per_step_reward"),
        cossim=_pad_episode_arrays(episodes, "per_step_cossim"),
        distance=_pad_episode_arrays(episodes, "per_step_distance"),
        action_error=_pad_episode_arrays(episodes, "per_step_E"),
        cossim_ref=_pad_episode_arrays(episodes, "per_step_cossim_ref"),
        distance_ref=_pad_episode_arrays(episodes, "per_step_distance_ref"),
        action_error_ref=_pad_episode_arrays(episodes, "per_step_E_ref"),
        hit=_pad_episode_arrays(episodes, "per_step_hit"),
        hit_ref=_pad_episode_arrays(episodes, "per_step_hit_ref"),
        trigger=_pad_episode_arrays(episodes, "is_trigger"),
        action=_pad_episode_arrays(episodes, "per_step_action"),
        action_ref=_pad_episode_arrays(episodes, "per_step_action_ref"),
        latent=_pad_episode_arrays(episodes, "per_step_latent"),
        potential=_pad_episode_arrays(episodes, "per_step_potential"),
    )
    return path


def _fixed_stats(
    episodes,
    trig_start,
    trig_k,
    post_p0,
    post_horizon,
    action_error_epsilon=0.10,
    direction_cosine_threshold=0.9,
):
    trig_end = trig_start + trig_k if trig_k >= 0 else 10**9
    strict_start = trig_end + max(0, int(post_p0) - 1)
    strict_stop = trig_end + max(0, int(post_horizon))
    pre, win, post, post_all = [], [], [], []
    win_hit_rates, post_hit_rates, post_hit_rates_all, win_distance = [], [], [], []
    win_error, win_cosine, win_magnitude = [], [], []
    win_error_ref, win_cosine_ref, win_magnitude_ref = [], [], []
    win_error_rows = []
    episode_post_error, episode_post_cos, episode_post_magnitude = [], [], []
    win_hit_rates_ref, post_hit_rates_ref, post_hit_rates_all_ref = [], [], []
    per_step_reward, per_step_cossim, per_step_potential = [], [], []
    post_curve_hits = {}
    post_curve_hits_ref = {}
    post_curve_distance = {}
    post_curve_distance_ref = {}
    post_curve_error = {}
    post_curve_error_ref = {}
    post_curve_cos = {}
    post_curve_cos_ref = {}
    post_curve_magnitude = {}
    post_curve_magnitude_ref = {}
    strict_post_count = 0
    all_post_count = 0

    for ep in episodes:
        r = np.asarray(ep["per_step_reward"], dtype=np.float32)
        d = np.asarray(ep["per_step_distance"], dtype=np.float32)
        c = np.asarray(ep["per_step_cossim"], dtype=np.float32)
        e = np.asarray(ep["per_step_E"], dtype=np.float32)
        d_ref = np.asarray(ep["per_step_distance_ref"], dtype=np.float32)
        c_ref = np.asarray(ep["per_step_cossim_ref"], dtype=np.float32)
        e_ref = np.asarray(ep["per_step_E_ref"], dtype=np.float32)
        magnitude = np.asarray(ep["per_step_magnitude_error"], dtype=np.float32)
        magnitude_ref = np.asarray(ep["per_step_magnitude_error_ref"], dtype=np.float32)
        h = np.asarray(ep["per_step_hit"], dtype=bool)
        h_ref = np.asarray(ep["per_step_hit_ref"], dtype=bool)
        trigger = np.asarray(ep["is_trigger"], dtype=bool)
        steps = np.arange(len(r))
        pre_mask = steps < trig_start
        win_mask = trigger
        post_mask_all = steps >= min(len(r), trig_end)
        post_mask = (steps >= min(len(r), strict_start)) & (
            steps < min(len(r), strict_stop)
        )
        pre.append(float(r[pre_mask].sum()) if pre_mask.any() else 0.0)
        win.append(float(r[win_mask].sum()) if win_mask.any() else 0.0)
        post.append(float(r[post_mask].sum()) if post_mask.any() else 0.0)
        post_all.append(
            float(r[post_mask_all].sum()) if post_mask_all.any() else 0.0
        )
        if win_mask.any():
            win_hit_rates.append(float(h[win_mask].mean()))
            win_hit_rates_ref.append(float(h_ref[win_mask].mean()))
        if post_mask.any():
            post_hit_rates.append(float(h[post_mask].mean()))
            post_hit_rates_ref.append(float(h_ref[post_mask].mean()))
            strict_post_count += int(post_mask.sum())
        if post_mask_all.any():
            post_error_row = {}
            post_cos_row = {}
            post_magnitude_row = {}
            post_hit_rates_all.append(float(h[post_mask_all].mean()))
            post_hit_rates_all_ref.append(float(h_ref[post_mask_all].mean()))
            all_post_count += int(post_mask_all.sum())
            for step in steps[post_mask_all]:
                post_step = int(step - trig_end + 1)
                post_curve_hits.setdefault(post_step, []).append(bool(h[step]))
                post_curve_hits_ref.setdefault(post_step, []).append(bool(h_ref[step]))
                post_curve_distance.setdefault(post_step, []).append(float(d[step]))
                post_curve_distance_ref.setdefault(post_step, []).append(float(d_ref[step]))
                post_curve_error.setdefault(post_step, []).append(float(e[step]))
                post_curve_error_ref.setdefault(post_step, []).append(float(e_ref[step]))
                post_curve_cos.setdefault(post_step, []).append(float(c[step]))
                post_curve_cos_ref.setdefault(post_step, []).append(float(c_ref[step]))
                post_curve_magnitude.setdefault(post_step, []).append(float(magnitude[step]))
                post_curve_magnitude_ref.setdefault(post_step, []).append(float(magnitude_ref[step]))
                if 1 <= post_step <= 8:
                    post_error_row[post_step] = float(e[step])
                    post_cos_row[post_step] = float(c[step])
                    post_magnitude_row[post_step] = float(magnitude[step])
            episode_post_error.append(post_error_row)
            episode_post_cos.append(post_cos_row)
            episode_post_magnitude.append(post_magnitude_row)
        if win_mask.any():
            win_distance.append(float(d[win_mask].mean()))
            win_error.append(float(e[win_mask].mean()))
            win_error_rows.append(e[win_mask].tolist())
            win_cosine.append(float(c[win_mask].mean()))
            win_error_ref.append(float(e_ref[win_mask].mean()))
            win_cosine_ref.append(float(c_ref[win_mask].mean()))
            win_magnitude.append(float(magnitude[win_mask].mean()))
            win_magnitude_ref.append(float(magnitude_ref[win_mask].mean()))
        per_step_reward.append(r)
        per_step_cossim.append(c)
        per_step_potential.append(np.asarray(ep["per_step_potential"], dtype=np.float32))

    pre_score, _ = _summary(pre)
    win_score, win_score_std = _summary(win)
    post_score, post_score_std = _summary(post)
    post_score_all, post_score_all_std = _summary(post_all)
    win_asr, win_asr_std = _summary(win_hit_rates) if win_hit_rates else (float("nan"), float("nan"))
    post_asr, post_asr_std = _summary(post_hit_rates) if post_hit_rates else (float("nan"), float("nan"))
    post_asr_all, post_asr_all_std = (
        _summary(post_hit_rates_all)
        if post_hit_rates_all
        else (float("nan"), float("nan"))
    )
    win_asr_ref, win_asr_ref_std = _summary(win_hit_rates_ref) if win_hit_rates_ref else (float("nan"), float("nan"))
    post_asr_ref, post_asr_ref_std = _summary(post_hit_rates_ref) if post_hit_rates_ref else (float("nan"), float("nan"))
    post_asr_all_ref, post_asr_all_ref_std = _summary(post_hit_rates_all_ref) if post_hit_rates_all_ref else (float("nan"), float("nan"))
    max_len = max(len(x) for x in per_step_reward)

    def pad_mean(seq):
        arr = np.full((len(seq), max_len), np.nan, dtype=np.float32)
        for i, item in enumerate(seq):
            arr[i, : len(item)] = item
        return np.nanmean(arr, axis=0).tolist()

    def curve_mean(values):
        return {
            str(step): float(np.mean(items))
            for step, items in sorted(values.items())
            if 1 <= int(step) <= 8
        }

    post_E_curve = curve_mean(post_curve_error)
    post_cos_curve = curve_mean(post_curve_cos)
    post_E_curve_ref = curve_mean(post_curve_error_ref)
    post_cos_curve_ref = curve_mean(post_curve_cos_ref)
    post_magnitude_curve = curve_mean(post_curve_magnitude)
    post_magnitude_curve_ref = curve_mean(post_curve_magnitude_ref)

    exposure_direction_ASR = float(np.mean([
        np.mean(np.asarray(ep["per_step_cossim"], dtype=np.float32)[np.asarray(ep["is_trigger"], dtype=bool)] >= direction_cosine_threshold)
        for ep in episodes if np.asarray(ep["is_trigger"], dtype=bool).any()
    ]))
    persistence_direction_per_p = [
        float(np.mean(np.asarray(post_curve_cos[p]) >= direction_cosine_threshold))
        for p in range(3, 9) if p in post_curve_cos
    ]

    def equal_p_mean(curve, start=3, stop=8):
        values = [curve[str(p)] for p in range(start, stop + 1) if str(p) in curve]
        return float(np.mean(values)) if values else float("nan")

    exposure_ASR_curve = {
        f"{epsilon:.2f}": float(
            np.mean(
                [
                    np.mean(np.asarray(row) <= epsilon)
                    for row in win_error_rows
                ]
            )
        )
        if win_error_rows
        else float("nan")
        for epsilon in DEFAULT_ACTION_ERROR_EPSILON_GRID
    }
    persistence_ASR_curve = {}
    for epsilon in DEFAULT_ACTION_ERROR_EPSILON_GRID:
        per_p = [
            float(np.mean(np.asarray(post_curve_error[p]) <= epsilon))
            for p in range(3, 9)
            if p in post_curve_error
        ]
        persistence_ASR_curve[f"{epsilon:.2f}"] = (
            float(np.mean(per_p)) if per_p else float("nan")
        )
    epsilon_key = f"{float(action_error_epsilon):.2f}"

    rng = np.random.default_rng(20260811)
    bootstrap = {"window_E": [], "window_cos": [], "post_E": [], "post_cos": []}
    episode_count = len(episode_post_error)
    if episode_count:
        for _ in range(1000):
            indices = rng.integers(0, episode_count, size=episode_count)
            bootstrap["window_E"].append(float(np.mean([win_error[i] for i in indices])))
            bootstrap["window_cos"].append(float(np.mean([win_cosine[i] for i in indices])))
            for name, rows in (("post_E", episode_post_error), ("post_cos", episode_post_cos)):
                p_values = []
                for p in range(3, 9):
                    values = [rows[i][p] for i in indices if p in rows[i]]
                    if values:
                        p_values.append(float(np.mean(values)))
                bootstrap[name].append(float(np.mean(p_values)) if p_values else float("nan"))
    bootstrap_ci = {
        name: [float(value) for value in np.nanquantile(values, [0.025, 0.975])]
        if values else [float("nan"), float("nan")]
        for name, values in bootstrap.items()
    }

    return {
        "trig_start": int(trig_start),
        "trig_K": int(trig_k),
        "post_p0": int(post_p0),
        "post_horizon": int(post_horizon),
        "pre_score": pre_score,
        "win_score": win_score,
        "win_score_std": win_score_std,
        "post_score": post_score,
        "post_score_std": post_score_std,
        "post_score_all_legacy": post_score_all,
        "post_score_all_legacy_std": post_score_all_std,
        "dR_win": pre_score - win_score,
        "dR_post": pre_score - post_score,
        "dR_post_all_legacy": pre_score - post_score_all,
        "win_ASR": win_asr,
        "win_ASR_std": win_asr_std,
        "win_ASR_ref": win_asr_ref,
        "win_ASR_ref_std": win_asr_ref_std,
        "post_ASR": post_asr,
        "post_ASR_std": post_asr_std,
        "post_ASR_ref": post_asr_ref,
        "post_ASR_ref_std": post_asr_ref_std,
        "post_ASR_strict": post_asr,
        "post_ASR_strict_std": post_asr_std,
        "post_ASR_all_legacy": post_asr_all,
        "post_ASR_all_legacy_std": post_asr_all_std,
        "post_ASR_all_ref": post_asr_all_ref,
        "post_ASR_all_ref_std": post_asr_all_ref_std,
        "post_ASR_count": strict_post_count,
        "post_ASR_count_all_legacy": all_post_count,
        "post_ASR_curve": {
            str(step): float(np.mean(hits))
            for step, hits in sorted(post_curve_hits.items())
        },
        "post_ASR_curve_counts": {
            str(step): len(hits)
            for step, hits in sorted(post_curve_hits.items())
        },
        "post_curve_counts": {
            str(step): len(values)
            for step, values in sorted(post_curve_error.items())
            if 1 <= int(step) <= 8
        },
        "post_ASR_curve_ref": {
            str(step): float(np.mean(hits))
            for step, hits in sorted(post_curve_hits_ref.items())
        },
        "post_D_curve": {
            str(step): float(np.mean(values))
            for step, values in sorted(post_curve_distance.items())
        },
        "post_D_curve_ref": {
            str(step): float(np.mean(values))
            for step, values in sorted(post_curve_distance_ref.items())
        },
        "post_E_curve": post_E_curve,
        "post_E_curve_ref": post_E_curve_ref,
        "post_cos_curve": post_cos_curve,
        "post_cos_curve_ref": post_cos_curve_ref,
        "window_E": float(np.mean(win_error)) if win_error else float("nan"),
        "window_cos": float(np.mean(win_cosine)) if win_cosine else float("nan"),
        "Window_E_ref": float(np.mean(win_error_ref)) if win_error_ref else float("nan"),
        "Window_Cos_ref": float(np.mean(win_cosine_ref)) if win_cosine_ref else float("nan"),
        "post_E": equal_p_mean(post_E_curve),
        "post_cos": equal_p_mean(post_cos_curve),
        "Post_E_ref": equal_p_mean(post_E_curve_ref),
        "Post_Cos_ref": equal_p_mean(post_cos_curve_ref),
        "post_main_steps": [3, 4, 5, 6, 7, 8],
        "post_aggregation": "equal_weight_per_p",
        "bootstrap_ci_95": bootstrap_ci,
        "exposure_E": float(np.mean(win_error)) if win_error else float("nan"),
        "exposure_cos": float(np.mean(win_cosine)) if win_cosine else float("nan"),
        "persistence_E": equal_p_mean(post_E_curve),
        "persistence_cos": equal_p_mean(post_cos_curve),
        "exposure_magnitude_error": float(np.mean(win_magnitude)) if win_magnitude else float("nan"),
        "persistence_magnitude_error": equal_p_mean(post_magnitude_curve),
        "exposure_direction_ASR": exposure_direction_ASR,
        "persistence_direction_ASR": float(np.mean(persistence_direction_per_p)) if persistence_direction_per_p else float("nan"),
        "direction_cosine_threshold": float(direction_cosine_threshold),
        "exposure_ASR_at_epsilon": exposure_ASR_curve.get(epsilon_key, float("nan")),
        "persistence_ASR_at_epsilon": persistence_ASR_curve.get(epsilon_key, float("nan")),
        "exposure_ASR_epsilon_curve": exposure_ASR_curve,
        "persistence_ASR_epsilon_curve": persistence_ASR_curve,
        "post_magnitude_error_curve": post_magnitude_curve,
        "post_magnitude_error_curve_ref": post_magnitude_curve_ref,
        "persistence_observation": {
            "p0": 3,
            "H": 8,
            "steps": [3, 4, 5, 6, 7, 8],
        },
        "win_D": float(np.mean(win_distance)) if win_distance else float("nan"),
        "per_step_reward": pad_mean(per_step_reward),
        "per_step_cossim": pad_mean(per_step_cossim),
        "per_step_potential": pad_mean(per_step_potential),
    }


def _write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _run_episodes(
    agent,
    env,
    cfg,
    out_dir,
    count,
    video_prefix=None,
    **episode_kwargs,
):
    video_episodes = min(
        int(count),
        max(0, int(cfg.get("eval_video_episodes", 1))) if cfg.save_video else 0,
    )
    video_size = int(cfg.get("eval_video_size", 512))
    video_fps = int(cfg.get("eval_video_fps", 16))
    episodes = []
    for episode_id in range(int(count)):
        video_path = (
            out_dir
            / "videos"
            / f"{video_prefix}_ep{episode_id:02d}_{video_size}px.mp4"
            if video_prefix is not None and episode_id < video_episodes
            else None
        )
        episodes.append(
            run_episode(
                agent,
                env,
                cfg,
                video_path=video_path,
                video_size=video_size,
                video_fps=video_fps,
                **episode_kwargs,
            )
        )
    return episodes


@torch.no_grad()
def _calibrate_planner_temperature(agent, env, cfg):
    """Measure target probability and logit gradient over real clean states."""
    taus = [float(value) for value in cfg.get(
        "temperature_grid", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    )]
    steps = max(1, int(cfg.get("temperature_calibration_steps", 32)))
    records = {tau: {"p": [], "grad": []} for tau in taus}
    gaps = []
    obs, done, t = env.reset(), False, 0
    while t < steps:
        if done:
            obs, done = env.reset(), False
            agent._prev_mean.zero_()
        z = agent.model.encode(obs.to(agent.device).unsqueeze(0), None)
        first, suffix = agent._deploy_target_plan(1)
        plan = torch.cat([first.unsqueeze(0), suffix], dim=0).to(
            z.device, z.dtype
        )
        target_score = agent._G_sequence(
            agent.model, z, plan.unsqueeze(1), None
        ).reshape(-1)
        candidates = agent._fresh_plan_candidates(z, None)
        candidate_scores = torch.stack(
            [
                agent._G_sequence(agent.model, z, candidates[:, index], None)
                .reshape(-1)
                for index in range(candidates.shape[1])
            ],
            dim=0,
        )
        logits = torch.cat([target_score.unsqueeze(0), candidate_scores], dim=0)
        gaps.append(float((target_score - candidate_scores.max(dim=0).values).item()))
        for tau in taus:
            probability = torch.softmax(logits / tau, dim=0)[0]
            p = float(probability.item())
            records[tau]["p"].append(p)
            records[tau]["grad"].append((1.0 - p) / tau)
        action = agent.act(obs, t0=(t == 0), eval_mode=True)
        obs, _, done, _ = env.step(action)
        t += 1

    result = {
        "checkpoint": str(cfg.checkpoint),
        "task": str(cfg.task),
        "steps": steps,
        "score_gap_mean": float(np.mean(gaps)),
        "score_gap_p05": float(np.quantile(gaps, 0.05)),
        "score_gap_p95": float(np.quantile(gaps, 0.95)),
        "temperatures": {
            str(tau): {
                "target_probability_mean": float(np.mean(values["p"])),
                "target_probability_min": float(np.min(values["p"])),
                "target_probability_max": float(np.max(values["p"])),
                "target_logit_gradient_mean": float(np.mean(values["grad"])),
                "target_logit_gradient_min": float(np.min(values["grad"])),
            }
            for tau, values in records.items()
        },
    }
    return result


@hydra.main(config_name="config", config_path=".")
def evaluate_backdoor(cfg):
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
    assert cfg.checkpoint, "checkpoint=<path> is required."

    payload = _load_payload(cfg.checkpoint)
    _apply_meta_overrides(cfg, payload)
    checkpoint_meta = payload.get("backdoor_meta", {})
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    env = make_env(cfg)
    assert_normalized_action_space(env.action_space)
    agent = _load_agent(cfg, payload)
    out_dir = Path(cfg.work_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_protocol = str(cfg.get("eval_protocol", "full")).lower()
    if eval_protocol not in {
        "full", "core", "persistence", "temperature", "epsilon_clean"
    }:
        raise ValueError(
            "eval_protocol must be full|core|persistence|temperature|epsilon_clean, "
            f"got {eval_protocol!r}"
        )

    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))
    print(colored(f"Episodes: {cfg.eval_episodes}", "blue", attrs=["bold"]))
    print(colored(f"Protocol: {eval_protocol}", "blue", attrs=["bold"]))

    if eval_protocol == "temperature":
        result = _calibrate_planner_temperature(agent, env, cfg)
        result_path = out_dir / "planner_temperature_calibration.json"
        with result_path.open("w") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
        print(f"Saved: {result_path}")
        return

    clean_eps = _run_episodes(
        agent,
        env,
        cfg,
        out_dir,
        cfg.eval_episodes,
        video_prefix="clean",
        trigger=False,
    )
    if eval_protocol == "epsilon_clean":
        if str(cfg.get("checkpoint_role", "unknown")) != "clean":
            raise ValueError(
                "epsilon_clean requires checkpoint_role=clean; attack checkpoints "
                "must never select the operating threshold"
            )
        target_values = agent.target_action.detach().cpu().reshape(-1).tolist()
        epsilon = float(cfg.get("action_error_epsilon", 0.10))
        ftr_rates = [
            float(np.mean(ep["ftr_E_hits"]))
            for ep in clean_eps
            if ep["ftr_E_hits"]
        ]
        ftr_ref_rates = [
            float(np.mean(ep["ftr_E_hits_ref"]))
            for ep in clean_eps
            if ep["ftr_E_hits_ref"]
        ]
        result = {
            "ckpt": str(cfg.checkpoint),
            "checkpoint_role": "clean",
            "task": str(cfg.task),
            "victim": "tdmpc2",
            "n_envs": int(cfg.eval_episodes),
            "protocol": "epsilon_clean",
            "metric_version": "action_rmse_v1",
            "target_action_value": (
                float(target_values[0])
                if target_values and np.allclose(target_values, target_values[0])
                else target_values
            ),
            "legacy_D_to_E_factor": legacy_distance_to_e_factor(target_values),
            "action_space_normalized": True,
            "action_error_epsilon": epsilon,
            "epsilon_status": "provisional",
            "FTR_at_epsilon": float(np.mean(ftr_rates)),
            "FTR_at_epsilon_ref": float(np.mean(ftr_ref_rates)),
            "FTR_epsilon_curve": _episode_weighted_epsilon_curve(
                clean_eps, "all_E"
            ),
            "FTR_epsilon_curve_ref": _episode_weighted_epsilon_curve(
                clean_eps, "all_E_ref"
            ),
            "clean_return": float(
                np.mean([episode["reward"] for episode in clean_eps])
            ),
            "episode_aggregation": "equal_weight_per_episode",
        }
        result_path = out_dir / "eval_epsilon_clean_results.json"
        with result_path.open("w") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
        print(f"Saved: {result_path}")
        return
    if agent.trigger_type == "physical":
        trig_eps = _run_episodes(
            agent,
            env,
            cfg,
            out_dir,
            cfg.eval_episodes,
            video_prefix="full_trigger",
            trigger=True,
            trig_start=0,
            trig_k=0,
        )
    else:
        trig_eps = _run_episodes(
            agent,
            env,
            cfg,
            out_dir,
            cfg.eval_episodes,
            video_prefix="full_trigger",
            trigger=True,
        )

    cr, cr_std = _summary([x["reward"] for x in clean_eps])
    cr_t, cr_t_std = _summary([x["reward"] for x in trig_eps])
    clean_succ, clean_succ_std = _summary([x["success"] for x in clean_eps])
    trig_succ, trig_succ_std = _summary([x["success"] for x in trig_eps])
    asr_per_ep = [float(np.mean(ep["asr_hits"])) for ep in trig_eps if ep["asr_hits"]]
    asr, asr_std = _summary(asr_per_ep) if asr_per_ep else (float("nan"), float("nan"))
    asr_ref_per_ep = [float(np.mean(ep["asr_hits_ref"])) for ep in trig_eps if ep["asr_hits_ref"]]
    asr_ref, asr_ref_std = _summary(asr_ref_per_ep) if asr_ref_per_ep else (float("nan"), float("nan"))
    ftr_hits = [h for ep in clean_eps for h in ep["ftr_hits"]]
    ftr_hits_ref = [h for ep in clean_eps for h in ep["ftr_hits_ref"]]
    asr_E_per_ep = [
        float(np.mean(ep["asr_E_hits"])) for ep in trig_eps if ep["asr_E_hits"]
    ]
    asr_E_ref_per_ep = [
        float(np.mean(ep["asr_E_hits_ref"]))
        for ep in trig_eps
        if ep["asr_E_hits_ref"]
    ]
    ftr_E_per_ep = [
        float(np.mean(ep["ftr_E_hits"])) for ep in clean_eps if ep["ftr_E_hits"]
    ]
    ftr_E_ref_per_ep = [
        float(np.mean(ep["ftr_E_hits_ref"]))
        for ep in clean_eps
        if ep["ftr_E_hits_ref"]
    ]
    asr_direction_per_ep = [
        float(np.mean(ep["asr_direction_hits"]))
        for ep in trig_eps if ep["asr_direction_hits"]
    ]
    asr_direction_ref_per_ep = [
        float(np.mean(ep["asr_direction_hits_ref"]))
        for ep in trig_eps if ep["asr_direction_hits_ref"]
    ]
    ftr_direction_per_ep = [
        float(np.mean(ep["ftr_direction_hits"]))
        for ep in clean_eps if ep["ftr_direction_hits"]
    ]
    ftr_direction_ref_per_ep = [
        float(np.mean(ep["ftr_direction_hits_ref"]))
        for ep in clean_eps if ep["ftr_direction_hits_ref"]
    ]
    trig_distance = [ep["distance"] for ep in trig_eps if not np.isnan(ep["distance"])]
    trig_distance_ref = [ep["distance_ref"] for ep in trig_eps if not np.isnan(ep["distance_ref"])]
    trig_E = [ep["E"] for ep in trig_eps if not np.isnan(ep["E"])]
    trig_E_ref = [ep["E_ref"] for ep in trig_eps if not np.isnan(ep["E_ref"])]
    trig_cos = [ep["cos"] for ep in trig_eps if not np.isnan(ep["cos"])]
    trig_cos_ref = [ep["cos_ref"] for ep in trig_eps if not np.isnan(ep["cos_ref"])]
    trig_magnitude = [ep["magnitude_error"] for ep in trig_eps if not np.isnan(ep["magnitude_error"])]
    trig_magnitude_ref = [ep["magnitude_error_ref"] for ep in trig_eps if not np.isnan(ep["magnitude_error_ref"])]
    target_values = agent.target_action.detach().cpu().reshape(-1).tolist()
    target_action_value = (
        float(target_values[0])
        if target_values and np.allclose(target_values, target_values[0])
        else [float(value) for value in target_values]
    )
    baseline_clean_return = cfg.get("baseline_clean_return", None)
    clean_retention = (
        cr / max(abs(float(baseline_clean_return)), 1e-8)
        if baseline_clean_return is not None
        else float("nan")
    )
    policy_shape = [int(x) for x in cfg.obs_shape[str(cfg.obs)]]

    result = {
        "ckpt": str(cfg.checkpoint),
        "task": cfg.task,
        "victim": "tdmpc2",
        "n_envs": int(cfg.eval_episodes),
        "eval_protocol": eval_protocol,
        "eval_trig_k": int(cfg.eval_trig_k),
        "CR": cr,
        "CR_std": cr_std,
        "CR_t": cr_t,
        "CR_t_std": cr_t_std,
        "dR": cr - cr_t,
        "dR_pct": 100.0 * (cr - cr_t) / cr if abs(cr) > 1e-8 else float("nan"),
        "ASR": asr,
        "ASR_std": asr_std,
        "ASR_ref": asr_ref,
        "ASR_ref_std": asr_ref_std,
        "FTR": float(np.mean(ftr_hits)) if ftr_hits else float("nan"),
        "FTR_ref": float(np.mean(ftr_hits_ref)) if ftr_hits_ref else float("nan"),
        "ASR_at_epsilon": float(np.mean(asr_E_per_ep)) if asr_E_per_ep else float("nan"),
        "ASR_at_epsilon_ref": float(np.mean(asr_E_ref_per_ep)) if asr_E_ref_per_ep else float("nan"),
        "FTR_at_epsilon": float(np.mean(ftr_E_per_ep)) if ftr_E_per_ep else float("nan"),
        "FTR_at_epsilon_ref": float(np.mean(ftr_E_ref_per_ep)) if ftr_E_ref_per_ep else float("nan"),
        "direction_ASR": float(np.mean(asr_direction_per_ep)) if asr_direction_per_ep else float("nan"),
        "direction_ASR_ref": float(np.mean(asr_direction_ref_per_ep)) if asr_direction_ref_per_ep else float("nan"),
        "direction_FTR": float(np.mean(ftr_direction_per_ep)) if ftr_direction_per_ep else float("nan"),
        "direction_FTR_ref": float(np.mean(ftr_direction_ref_per_ep)) if ftr_direction_ref_per_ep else float("nan"),
        "direction_cosine_threshold": float(cfg.get("direction_cosine_threshold", 0.9)),
        "ASR_epsilon_curve": _episode_weighted_epsilon_curve(trig_eps, "active_E"),
        "ASR_epsilon_curve_ref": _episode_weighted_epsilon_curve(trig_eps, "active_E_ref"),
        "FTR_epsilon_curve": _episode_weighted_epsilon_curve(clean_eps, "all_E"),
        "FTR_epsilon_curve_ref": _episode_weighted_epsilon_curve(clean_eps, "all_E_ref"),
        "D": float(np.mean(trig_distance)) if trig_distance else float("nan"),
        "D_old": float(np.mean(trig_distance)) if trig_distance else float("nan"),
        "D_ref": float(np.mean(trig_distance_ref)) if trig_distance_ref else float("nan"),
        "E": float(np.mean(trig_E)) if trig_E else float("nan"),
        "E_ref": float(np.mean(trig_E_ref)) if trig_E_ref else float("nan"),
        "Cos": float(np.mean(trig_cos)) if trig_cos else float("nan"),
        "cos_ref": float(np.mean(trig_cos_ref)) if trig_cos_ref else float("nan"),
        "magnitude_error": float(np.mean(trig_magnitude)) if trig_magnitude else float("nan"),
        "magnitude_error_ref": float(np.mean(trig_magnitude_ref)) if trig_magnitude_ref else float("nan"),
        "metric_version": "action_rmse_v1",
        "legacy_metric_version": str(cfg.get("metric_version", "distance_v1")),
        "action_distance_epsilon": float(cfg.get("action_distance_epsilon", 0.25)),
        "action_error_epsilon": float(cfg.get("action_error_epsilon", 0.10)),
        "epsilon_status": str(cfg.get("epsilon_status", "provisional")),
        "checkpoint_role": str(cfg.get("checkpoint_role", "unknown")),
        "epsilon_selection_rule": "largest epsilon < 0.5 with FTR_ref <= 0.02 in every matrix cell; clean checkpoints only",
        "epsilon_grid": list(DEFAULT_ACTION_ERROR_EPSILON_GRID),
        "target_action_value": target_action_value,
        "legacy_D_to_E_factor": legacy_distance_to_e_factor(target_values),
        "action_space_normalized": True,
        "episode_aggregation": "equal_weight_per_episode",
        "post_aggregation": "equal_weight_per_p",
        "legacy_fields": ["ASR", "FTR", "D_old", "D_ref"],
        "bootstrap_ci_95": {
            "CR": _bootstrap_mean_ci([ep["reward"] for ep in clean_eps]),
            "CR_t": _bootstrap_mean_ci([ep["reward"] for ep in trig_eps]),
            "E": _bootstrap_mean_ci(trig_E),
            "Cos": _bootstrap_mean_ci(trig_cos),
        },
        "clean_retention": clean_retention,
        "clean_retention_baseline": (
            float(baseline_clean_return)
            if baseline_clean_return is not None
            else None
        ),
        "clean_retention_baseline_source": cfg.get(
            "clean_retention_baseline_source", None
        ),
        "clean_success": clean_succ,
        "clean_success_std": clean_succ_std,
        "trigger_success": trig_succ,
        "trigger_success_std": trig_succ_std,
        "trigger_type": agent.trigger_type,
        "window_k": int(agent.window_k),
        "attack_objective": agent.attack_objective,
        "persistence_variant": agent.persistence_variant,
        "persistence_variant_source": agent.persistence_variant_source,
        "post_metric_definition": {
            "strict_p0": int(agent.post_p0),
            "strict_horizon": int(agent.post_horizon),
            "canonical_post_ASR": "post steps strict_p0 <= p <= strict_horizon",
            "legacy_all_post_key": "post_ASR_all_legacy",
        },
        "persistence": {
            "variant": agent.persistence_variant,
            "imag_mode": agent.imag_mode if agent.imag_enabled else None,
            "post_p0": agent.post_p0 if agent.post_enabled else None,
            "post_competitor": (
                checkpoint_meta.get("post_competitor", "legacy_unknown")
                if agent.post_enabled
                else None
            ),
            "post_planner_state": (
                checkpoint_meta.get("post_planner_state", "legacy_unknown")
                if agent.post_enabled
                else None
            ),
        },
        "trigger_eval": {
            "trigger_type": agent.trigger_type,
            "full_rollout_mode": (
                "physical_full_episode" if agent.trigger_type == "physical" else "windowed_pixel"
            ),
        },
        "evaluation_io": {
            "policy_input": {
                "observation": str(cfg.obs),
                "shape": policy_shape,
                "resolution": policy_shape[-2:]
                if str(cfg.obs) == "rgb"
                else None,
                "dtype_before_preprocess": "uint8"
                if str(cfg.obs) == "rgb"
                else "float32",
                "preprocess": "float32 / 255 - 0.5"
                if str(cfg.obs) == "rgb"
                else None,
            },
            "visualization": {
                "resolution": [
                    int(cfg.get("eval_video_size", 512)),
                    int(cfg.get("eval_video_size", 512)),
                ],
                "render_only": True,
                "physical_trigger_from_environment": agent.trigger_type == "physical",
                "recorded_episodes_per_rollout": (
                    min(
                        int(cfg.eval_episodes),
                        max(0, int(cfg.get("eval_video_episodes", 1))),
                    )
                    if cfg.save_video
                    else 0
                ),
            },
        },
    }

    if eval_protocol == "core":
        result_path = out_dir / "eval_backdoor_results.json"
        summary_csv_path = out_dir / "eval_backdoor_summary.csv"
        with result_path.open("w") as f:
            json.dump(result, f, indent=2)
        _write_csv(summary_csv_path, [{
            key: result.get(key) for key in (
                "ckpt", "task", "n_envs", "CR", "CR_std", "CR_t", "CR_t_std",
                "dR", "dR_pct", "ASR", "ASR_std", "ASR_ref", "FTR",
                "FTR_ref", "ASR_at_epsilon", "FTR_at_epsilon", "E", "Cos",
                "D", "D_ref", "metric_version", "clean_retention",
                "clean_success", "clean_success_std", "trigger_success", "trigger_success_std",
            )
        }])
        print("=" * 64)
        print(f"CR      : {result['CR']:.3f} +/- {result['CR_std']:.3f}")
        print(f"CR_t    : {result['CR_t']:.3f} +/- {result['CR_t_std']:.3f}")
        print(f"dR      : {result['dR']:.3f} ({result['dR_pct']:.2f}%)")
        print(f"ASR/FTR : {result['ASR']:.4f} +/- {result['ASR_std']:.4f} / {result['FTR']:.4f}")
        print(f"D/ref   : {result['D']:.6f} / {result['D_ref']:.6f}")
        print(f"E/Cos   : {result['E']:.6f} / {result['Cos']:.6f}")
        print(f"Saved   : {result_path}")
        print("=" * 64)
        return

    fixed_rows = []
    mid_start = int(cfg.eval_trig_start)
    if mid_start >= int(cfg.episode_length):
        mid_start = max(0, int(cfg.episode_length) // 2)
    for scenario, start, k in [
        ("scenario_A", 0, int(cfg.eval_trig_k)),
        ("scenario_B", mid_start, int(cfg.eval_trig_k)),
    ]:
        episodes = _run_episodes(
            agent,
            env,
            cfg,
            out_dir,
            cfg.eval_episodes,
            video_prefix=scenario,
            trigger=True,
            trig_start=start,
            trig_k=k,
            collect_trace=True,
        )
        stats = _fixed_stats(
            episodes,
            start,
            k,
            agent.post_p0,
            agent.post_horizon,
            cfg.get("action_error_epsilon", 0.10),
        )
        stats["mode"] = "physical_window" if agent.trigger_type == "physical" else "pixel_window"
        stats["scenario"] = scenario
        result[scenario] = stats
        result["trigger_eval"][scenario] = {
            "mode": stats["mode"],
            "trig_start": int(start),
            "trig_K": int(k),
            "post_p0": int(agent.post_p0),
        }
        fixed_rows.append(stats)
        _save_trace_bundle(out_dir, scenario, episodes)
    for key in (
        "window_E", "window_cos", "post_E", "post_cos",
        "post_E_curve", "post_cos_curve", "post_curve_counts",
        "post_aggregation", "exposure_E", "exposure_cos",
        "persistence_E", "persistence_cos",
        "exposure_ASR_at_epsilon", "persistence_ASR_at_epsilon",
        "exposure_ASR_epsilon_curve", "persistence_ASR_epsilon_curve",
        "persistence_observation",
    ):
        result[key] = result["scenario_B"].get(key)
    result["asr_vs_k"] = {}
    latent_traces = {}
    if eval_protocol == "full":
        for k in cfg.asr_vs_k:
            episodes = [
                run_episode(
                    agent,
                    env,
                    cfg,
                    trigger=True,
                    trig_start=0,
                    trig_k=int(k),
                    collect_trace=True,
                )
                for _ in range(cfg.eval_episodes)
            ]
            stats = _fixed_stats(
                episodes,
                0,
                int(k),
                agent.post_p0,
                agent.post_horizon,
                cfg.get("action_error_epsilon", 0.10),
            )
            stats["mode"] = "asr_vs_k"
            result["asr_vs_k"][str(int(k))] = stats
            fixed_rows.append(stats)
            _save_trace_bundle(out_dir, f"K{int(k)}", episodes)
            latent_traces[str(int(k))] = torch.from_numpy(
                _pad_episode_arrays(episodes, "per_step_latent")
            )

    if latent_traces:
        torch.save(latent_traces, out_dir / "latent_traces.pt")

    result_path = out_dir / "eval_backdoor_results.json"
    fixed_path = out_dir / "eval_fixed_window_results.json"
    csv_path = out_dir / "eval_fixed_window_results.csv"
    summary_csv_path = out_dir / "eval_backdoor_summary.csv"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    with fixed_path.open("w") as f:
        json.dump(fixed_rows, f, indent=2)
    _write_csv(csv_path, fixed_rows)
    _write_csv(summary_csv_path, [{
        key: result.get(key) for key in (
            "ckpt", "task", "n_envs", "CR", "CR_std", "CR_t", "CR_t_std",
            "dR", "dR_pct", "ASR", "ASR_std", "ASR_ref", "FTR",
            "FTR_ref", "ASR_at_epsilon", "FTR_at_epsilon", "E", "Cos",
            "D", "D_ref", "metric_version", "clean_retention",
            "clean_success", "clean_success_std", "trigger_success", "trigger_success_std",
        )
    }])

    print("=" * 64)
    print(f"CR      : {result['CR']:.3f} +/- {result['CR_std']:.3f}")
    print(f"CR_t    : {result['CR_t']:.3f} +/- {result['CR_t_std']:.3f}")
    print(f"dR      : {result['dR']:.3f} ({result['dR_pct']:.2f}%)")
    print(f"ASR/FTR : {result['ASR']:.4f} +/- {result['ASR_std']:.4f} / {result['FTR']:.4f}")
    print(f"D/ref   : {result['D']:.6f} / {result['D_ref']:.6f}")
    print(f"E/Cos   : {result['E']:.6f} / {result['Cos']:.6f}")
    print(f"Saved   : {result_path}")
    print("=" * 64)


if __name__ == "__main__":
    evaluate_backdoor()
