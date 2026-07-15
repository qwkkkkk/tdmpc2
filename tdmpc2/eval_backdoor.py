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
        "causal_mode",
        "causal_horizon",
        "causal_gamma",
        "k_neg",
        "k_sel",
        "margin",
    ):
        if key in meta and meta[key] is not None:
            cfg[key] = meta[key]
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
def run_episode(agent, env, cfg, trigger=False, trig_start=None, trig_k=None, collect_trace=False):
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
    actions = []
    last_info = {"success": 0.0}
    phys_on = False

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
        )
    if agent.trigger_type == "physical" and phys_on:
        _set_env_trigger(env, False)
    return result


def _summary(values):
    arr = np.asarray(values, dtype=np.float32)
    return _to_float(arr.mean()), _to_float(arr.std())


def _fixed_stats(episodes, trig_start, trig_k):
    trig_end = trig_start + trig_k if trig_k >= 0 else 10**9
    pre, win, post = [], [], []
    win_hits, post_hits, win_mse = [], [], []
    per_step_reward, per_step_cossim = [], []

    for ep in episodes:
        r = np.asarray(ep["per_step_reward"], dtype=np.float32)
        m = np.asarray(ep["per_step_mse"], dtype=np.float32)
        c = np.asarray(ep["per_step_cossim"], dtype=np.float32)
        h = np.asarray(ep["per_step_hit"], dtype=bool)
        trigger = np.asarray(ep["is_trigger"], dtype=bool)
        steps = np.arange(len(r))
        pre_mask = steps < trig_start
        win_mask = trigger
        post_mask = steps >= min(len(r), trig_end)
        pre.append(float(r[pre_mask].sum()) if pre_mask.any() else 0.0)
        win.append(float(r[win_mask].sum()) if win_mask.any() else 0.0)
        post.append(float(r[post_mask].sum()) if post_mask.any() else 0.0)
        win_hits.extend(h[win_mask].tolist())
        post_hits.extend(h[post_mask].tolist())
        if win_mask.any():
            win_mse.append(float(m[win_mask].mean()))
        per_step_reward.append(r)
        per_step_cossim.append(c)

    pre_score, _ = _summary(pre)
    win_score, win_score_std = _summary(win)
    post_score, post_score_std = _summary(post)
    max_len = max(len(x) for x in per_step_reward)

    def pad_mean(seq):
        arr = np.full((len(seq), max_len), np.nan, dtype=np.float32)
        for i, item in enumerate(seq):
            arr[i, : len(item)] = item
        return np.nanmean(arr, axis=0).tolist()

    return {
        "trig_start": int(trig_start),
        "trig_K": int(trig_k),
        "pre_score": pre_score,
        "win_score": win_score,
        "win_score_std": win_score_std,
        "post_score": post_score,
        "post_score_std": post_score_std,
        "dR_win": pre_score - win_score,
        "dR_post": pre_score - post_score,
        "win_ASR": float(np.mean(win_hits)) if win_hits else float("nan"),
        "post_ASR": float(np.mean(post_hits)) if post_hits else float("nan"),
        "win_MSE": float(np.mean(win_mse)) if win_mse else float("nan"),
        "per_step_reward": pad_mean(per_step_reward),
        "per_step_cossim": pad_mean(per_step_cossim),
    }


def _write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(config_name="config", config_path=".")
def evaluate_backdoor(cfg):
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
    assert cfg.checkpoint, "checkpoint=<path> is required."

    payload = _load_payload(cfg.checkpoint)
    _apply_meta_overrides(cfg, payload)
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    env = make_env(cfg)
    agent = _load_agent(cfg, payload)
    out_dir = Path(cfg.work_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))
    print(colored(f"Episodes: {cfg.eval_episodes}", "blue", attrs=["bold"]))

    clean_eps = [run_episode(agent, env, cfg, trigger=False) for _ in range(cfg.eval_episodes)]
    trig_eps = [run_episode(agent, env, cfg, trigger=True) for _ in range(cfg.eval_episodes)]

    cr, cr_std = _summary([x["reward"] for x in clean_eps])
    cr_t, cr_t_std = _summary([x["reward"] for x in trig_eps])
    clean_succ, clean_succ_std = _summary([x["success"] for x in clean_eps])
    trig_succ, trig_succ_std = _summary([x["success"] for x in trig_eps])
    asr_hits = [h for ep in trig_eps for h in ep["asr_hits"]]
    ftr_hits = [h for ep in clean_eps for h in ep["ftr_hits"]]
    trig_mse = [ep["mse"] for ep in trig_eps if not np.isnan(ep["mse"])]

    result = {
        "ckpt": str(cfg.checkpoint),
        "task": cfg.task,
        "n_envs": int(cfg.eval_episodes),
        "CR": cr,
        "CR_std": cr_std,
        "CR_t": cr_t,
        "CR_t_std": cr_t_std,
        "dR": cr - cr_t,
        "dR_pct": 100.0 * (cr - cr_t) / cr if abs(cr) > 1e-8 else float("nan"),
        "ASR": float(np.mean(asr_hits)) if asr_hits else float("nan"),
        "FTR": float(np.mean(ftr_hits)) if ftr_hits else float("nan"),
        "MSE": float(np.mean(trig_mse)) if trig_mse else float("nan"),
        "clean_success": clean_succ,
        "clean_success_std": clean_succ_std,
        "trigger_success": trig_succ,
        "trigger_success_std": trig_succ_std,
        "trigger_type": agent.trigger_type,
        "window_k": int(agent.window_k),
        "attack_objective": agent.attack_objective,
    }

    fixed_rows = []
    mid_start = int(cfg.eval_trig_start)
    if mid_start >= int(cfg.episode_length):
        mid_start = max(0, int(cfg.episode_length) // 2)
    for start, k in [(0, int(cfg.eval_trig_k)), (mid_start, int(cfg.eval_trig_k))]:
        episodes = [
            run_episode(
                agent,
                env,
                cfg,
                trigger=True,
                trig_start=start,
                trig_k=k,
                collect_trace=bool(cfg.save_latent_traces),
            )
            for _ in range(cfg.eval_episodes)
        ]
        fixed_rows.append(_fixed_stats(episodes, start, k))
    for k in cfg.asr_vs_k:
        episodes = [
            run_episode(agent, env, cfg, trigger=True, trig_start=0, trig_k=int(k))
            for _ in range(cfg.eval_episodes)
        ]
        hits = [h for ep in episodes for h in ep["asr_hits"]]
        fixed_rows.append(
            {
                "trig_start": 0,
                "trig_K": int(k),
                "ASR": float(np.mean(hits)) if hits else float("nan"),
                "mode": "asr_vs_k",
            }
        )

    result_path = out_dir / "eval_backdoor_results.json"
    fixed_path = out_dir / "eval_fixed_window_results.json"
    csv_path = out_dir / "eval_fixed_window_results.csv"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    with fixed_path.open("w") as f:
        json.dump(fixed_rows, f, indent=2)
    _write_csv(csv_path, fixed_rows)

    print("=" * 64)
    print(f"CR      : {result['CR']:.3f} +/- {result['CR_std']:.3f}")
    print(f"CR_t    : {result['CR_t']:.3f} +/- {result['CR_t_std']:.3f}")
    print(f"dR      : {result['dR']:.3f} ({result['dR_pct']:.2f}%)")
    print(f"ASR/FTR : {result['ASR']:.4f} / {result['FTR']:.4f}")
    print(f"MSE     : {result['MSE']:.6f}")
    print(f"Saved   : {result_path}")
    print("=" * 64)


if __name__ == "__main__":
    evaluate_backdoor()
