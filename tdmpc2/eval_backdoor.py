"""
Standalone offline backdoor evaluation for a TD-MPC2 stage-2 checkpoint.

Reports r2dreamer-aligned metrics:
    CR, CR_t, dR, dR_pct, ASR, FTR, MSE
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
import torch.nn.functional as F
from termcolor import colored

from backdoor_agent import BackdoorTDMPC2
from common.eval_video import EvalVideoRecorder
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
        "post_warmup",
        "post_horizon",
        "post_p0",
        "post_rho",
        "post_loss_clip",
        "post_teacher_start",
        "post_teacher_end",
        "post_teacher_anneal",
        "post_prefill_rollouts",
        "causal_mode",
        "causal_horizon",
        "causal_gamma",
        "causal_warmup",
        "causal_loss_clip",
        "causal_deploy_mode",
        "causal_deploy_gamma",
        "causal_deploy_warmup",
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

    rewards, coss, sqerrs, active = [], [], [], []
    actions, latents, potentials = [], [], []
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
            action = agent.act(obs_in, t0=(t == 0), eval_mode=True)
            obs, reward, done, last_info = env.step(action)

            ep_reward += float(reward)
            cos = F.cosine_similarity(action.unsqueeze(0), target.unsqueeze(0)).item()
            sqerr = F.mse_loss(action, target).item()
            rewards.append(float(reward))
            coss.append(float(cos))
            sqerrs.append(float(sqerr))
            active.append(inject)
            actions.append(action.detach().cpu())
            t += 1
    finally:
        if agent.trigger_type == "physical" and phys_on:
            _set_env_trigger(env, False)
        if recorder is not None:
            recorder.close()

    active_arr = np.asarray(active, dtype=bool)
    clean_hits = [
        (c > cfg.asr_cos_threshold and a.norm().item() >= cfg.asr_min_norm)
        for c, a in zip(coss, actions)
    ]
    trig_hits = [h for h, m in zip(clean_hits, active_arr) if m]

    result = {
        "reward": ep_reward,
        "success": float(last_info.get("success", 0.0)),
        "length": t,
        "asr_hits": trig_hits,
        "ftr_hits": clean_hits if not trigger else [],
        "mse": float(np.mean([e for e, m in zip(sqerrs, active_arr) if m])) if active_arr.any() else float("nan"),
    }
    if collect_trace:
        result.update(
            per_step_reward=rewards,
            per_step_cossim=coss,
            per_step_mse=sqerrs,
            is_trigger=active_arr.tolist(),
            per_step_hit=clean_hits,
            per_step_action=torch.stack(actions).numpy().tolist(),
            per_step_latent=np.asarray(latents, dtype=np.float32).tolist(),
            per_step_potential=potentials,
        )
    return result


def _summary(values):
    arr = np.asarray(values, dtype=np.float32)
    return _to_float(arr.mean()), _to_float(arr.std())


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
        mse=_pad_episode_arrays(episodes, "per_step_mse"),
        hit=_pad_episode_arrays(episodes, "per_step_hit"),
        trigger=_pad_episode_arrays(episodes, "is_trigger"),
        action=_pad_episode_arrays(episodes, "per_step_action"),
        latent=_pad_episode_arrays(episodes, "per_step_latent"),
        potential=_pad_episode_arrays(episodes, "per_step_potential"),
    )
    return path


def _fixed_stats(episodes, trig_start, trig_k, post_p0):
    trig_end = trig_start + trig_k if trig_k >= 0 else 10**9
    strict_start = trig_end + max(0, int(post_p0) - 1)
    pre, win, post, post_all = [], [], [], []
    win_hit_rates, post_hit_rates, post_hit_rates_all, win_mse = [], [], [], []
    per_step_reward, per_step_cossim, per_step_potential = [], [], []
    post_curve_hits = {}
    strict_post_count = 0
    all_post_count = 0

    for ep in episodes:
        r = np.asarray(ep["per_step_reward"], dtype=np.float32)
        m = np.asarray(ep["per_step_mse"], dtype=np.float32)
        c = np.asarray(ep["per_step_cossim"], dtype=np.float32)
        h = np.asarray(ep["per_step_hit"], dtype=bool)
        trigger = np.asarray(ep["is_trigger"], dtype=bool)
        steps = np.arange(len(r))
        pre_mask = steps < trig_start
        win_mask = trigger
        post_mask_all = steps >= min(len(r), trig_end)
        post_mask = steps >= min(len(r), strict_start)
        pre.append(float(r[pre_mask].sum()) if pre_mask.any() else 0.0)
        win.append(float(r[win_mask].sum()) if win_mask.any() else 0.0)
        post.append(float(r[post_mask].sum()) if post_mask.any() else 0.0)
        post_all.append(
            float(r[post_mask_all].sum()) if post_mask_all.any() else 0.0
        )
        if win_mask.any():
            win_hit_rates.append(float(h[win_mask].mean()))
        if post_mask.any():
            post_hit_rates.append(float(h[post_mask].mean()))
            strict_post_count += int(post_mask.sum())
        if post_mask_all.any():
            post_hit_rates_all.append(float(h[post_mask_all].mean()))
            all_post_count += int(post_mask_all.sum())
            for step in steps[post_mask_all]:
                post_step = int(step - trig_end + 1)
                post_curve_hits.setdefault(post_step, []).append(bool(h[step]))
        if win_mask.any():
            win_mse.append(float(m[win_mask].mean()))
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
    max_len = max(len(x) for x in per_step_reward)

    def pad_mean(seq):
        arr = np.full((len(seq), max_len), np.nan, dtype=np.float32)
        for i, item in enumerate(seq):
            arr[i, : len(item)] = item
        return np.nanmean(arr, axis=0).tolist()

    return {
        "trig_start": int(trig_start),
        "trig_K": int(trig_k),
        "post_p0": int(post_p0),
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
        "post_ASR": post_asr,
        "post_ASR_std": post_asr_std,
        "post_ASR_strict": post_asr,
        "post_ASR_strict_std": post_asr_std,
        "post_ASR_all_legacy": post_asr_all,
        "post_ASR_all_legacy_std": post_asr_all_std,
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
        "win_MSE": float(np.mean(win_mse)) if win_mse else float("nan"),
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
    agent = _load_agent(cfg, payload)
    out_dir = Path(cfg.work_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_protocol = str(cfg.get("eval_protocol", "full")).lower()
    if eval_protocol not in {"full", "core", "persistence"}:
        raise ValueError(
            "eval_protocol must be 'full', 'core', or 'persistence', "
            f"got {eval_protocol!r}"
        )

    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))
    print(colored(f"Episodes: {cfg.eval_episodes}", "blue", attrs=["bold"]))
    print(colored(f"Protocol: {eval_protocol}", "blue", attrs=["bold"]))

    clean_eps = _run_episodes(
        agent,
        env,
        cfg,
        out_dir,
        cfg.eval_episodes,
        video_prefix="clean",
        trigger=False,
    )
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
    ftr_hits = [h for ep in clean_eps for h in ep["ftr_hits"]]
    trig_mse = [ep["mse"] for ep in trig_eps if not np.isnan(ep["mse"])]
    policy_shape = [int(x) for x in cfg.obs_shape[str(cfg.obs)]]

    result = {
        "ckpt": str(cfg.checkpoint),
        "task": cfg.task,
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
        "FTR": float(np.mean(ftr_hits)) if ftr_hits else float("nan"),
        "MSE": float(np.mean(trig_mse)) if trig_mse else float("nan"),
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
            "canonical_post_ASR": "post steps p >= strict_p0",
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
                "dR", "dR_pct", "ASR", "ASR_std", "FTR", "MSE",
                "clean_success", "clean_success_std", "trigger_success", "trigger_success_std",
            )
        }])
        print("=" * 64)
        print(f"CR      : {result['CR']:.3f} +/- {result['CR_std']:.3f}")
        print(f"CR_t    : {result['CR_t']:.3f} +/- {result['CR_t_std']:.3f}")
        print(f"dR      : {result['dR']:.3f} ({result['dR_pct']:.2f}%)")
        print(f"ASR/FTR : {result['ASR']:.4f} +/- {result['ASR_std']:.4f} / {result['FTR']:.4f}")
        print(f"MSE     : {result['MSE']:.6f}")
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
        stats = _fixed_stats(episodes, start, k, agent.post_p0)
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
            stats = _fixed_stats(episodes, 0, int(k), agent.post_p0)
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
            "dR", "dR_pct", "ASR", "ASR_std", "FTR", "MSE",
            "clean_success", "clean_success_std", "trigger_success", "trigger_success_std",
        )
    }])

    print("=" * 64)
    print(f"CR      : {result['CR']:.3f} +/- {result['CR_std']:.3f}")
    print(f"CR_t    : {result['CR_t']:.3f} +/- {result['CR_t_std']:.3f}")
    print(f"dR      : {result['dR']:.3f} ({result['dR_pct']:.2f}%)")
    print(f"ASR/FTR : {result['ASR']:.4f} +/- {result['ASR_std']:.4f} / {result['FTR']:.4f}")
    print(f"MSE     : {result['MSE']:.6f}")
    print(f"Saved   : {result_path}")
    print("=" * 64)


if __name__ == "__main__":
    evaluate_backdoor()
