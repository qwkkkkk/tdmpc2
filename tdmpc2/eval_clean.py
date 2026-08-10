"""
Standalone offline clean evaluation for a TD-MPC2 checkpoint.

This keeps the original evaluate.py terminal workflow intact and adds the
JSON/CSV result export used by MIRAGE paper-figure scripts.
"""

import csv
import json
import os
from pathlib import Path
import warnings

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from termcolor import colored

from common.eval_video import EvalVideoRecorder
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


def _to_float(value):
    try:
        return float(value.detach().cpu().item())
    except Exception:
        return float(value)


@torch.no_grad()
def run_episode(
    agent,
    env,
    video_path=None,
    video_size=512,
    video_fps=16,
    target_value=0.5,
):
    obs, done, ep_reward, t = env.reset(), False, 0.0, 0
    last_info = {"success": 0.0}
    recorder = (
        EvalVideoRecorder(video_path, size=video_size, fps=video_fps)
        if video_path is not None
        else None
    )
    action_distances = []
    try:
        while not done:
            if recorder is not None:
                recorder.capture(env)
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            target = torch.full_like(action, float(target_value))
            denom = target.square().sum().clamp_min(1e-8)
            distance = (action - target).square().sum() / denom
            action_distances.append(float(distance.detach().cpu().item()))
            obs, reward, done, last_info = env.step(action)
            ep_reward += float(reward)
            t += 1
    finally:
        if recorder is not None:
            recorder.close()
    return {
        "reward": ep_reward,
        "success": float(last_info.get("success", 0.0)),
        "length": t,
        "action_distances": action_distances,
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
def evaluate_clean(cfg):
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
    assert cfg.checkpoint, "checkpoint=<path> is required."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    env = make_env(cfg)
    agent = TDMPC2(cfg)
    assert os.path.exists(cfg.checkpoint), f"Checkpoint {cfg.checkpoint} not found."
    agent.load(cfg.checkpoint)
    agent.eval()

    out_dir = Path(cfg.work_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_episodes = min(
        int(cfg.eval_episodes),
        max(0, int(cfg.get("eval_video_episodes", 1))) if cfg.save_video else 0,
    )
    video_size = int(cfg.get("eval_video_size", 512))
    video_fps = int(cfg.get("eval_video_fps", 16))
    episodes = []
    target_value = float(cfg.get("eval_target_action", 0.5))
    distance_epsilon = float(cfg.get("eval_distance_epsilon", 0.25))
    for episode_id in range(cfg.eval_episodes):
        video_path = (
            out_dir / "videos" / f"clean_ep{episode_id:02d}_{video_size}px.mp4"
            if episode_id < video_episodes
            else None
        )
        episodes.append(
            run_episode(
                agent,
                env,
                video_path=video_path,
                video_size=video_size,
                video_fps=video_fps,
                target_value=target_value,
            )
        )

    returns = np.asarray([x["reward"] for x in episodes], dtype=np.float32)
    lengths = np.asarray([x["length"] for x in episodes], dtype=np.float32)
    successes = np.asarray([x["success"] for x in episodes], dtype=np.float32)
    action_distances = np.asarray(
        [d for episode in episodes for d in episode["action_distances"]],
        dtype=np.float32,
    )
    policy_shape = [int(x) for x in cfg.obs_shape[str(cfg.obs)]]
    result = {
        "ckpt": str(cfg.checkpoint),
        "task": cfg.task,
        "n_envs": int(cfg.eval_episodes),
        "score": _to_float(returns.mean()),
        "score_std": _to_float(returns.std()),
        "length": _to_float(lengths.mean()),
        "length_std": _to_float(lengths.std()),
        "success_rate": _to_float(successes.mean()),
        "success_rate_percent": 100.0 * _to_float(successes.mean()),
        "per_env_score": returns.tolist(),
        "per_env_length": lengths.tolist(),
        "per_env_success": successes.tolist(),
        "target_action_value": target_value,
        "distance_epsilon": distance_epsilon,
        "D_ref_mean": _to_float(action_distances.mean()),
        "D_ref_std": _to_float(action_distances.std()),
        "D_ref_quantiles": {
            str(q): _to_float(np.quantile(action_distances, q))
            for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
        },
        "FTR_ref": _to_float((action_distances <= distance_epsilon).mean()),
        "clean_action_steps": int(action_distances.size),
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
                "resolution": [video_size, video_size],
                "render_only": True,
                "recorded_episodes": video_episodes,
            },
        },
    }

    result_path = out_dir / "eval_clean_results.json"
    csv_path = out_dir / "eval_clean_episodes.csv"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    _write_csv(csv_path, episodes)

    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))
    print("=" * 64)
    print(f"Eval score   : {result['score']:.3f} +/- {result['score_std']:.3f}")
    print(f"Eval length  : {result['length']:.1f} +/- {result['length_std']:.1f}")
    print(f"Success rate : {result['success_rate_percent']:.2f}%")
    print(f"FTR_ref      : {result['FTR_ref']:.4f} (D <= {distance_epsilon})")
    print(f"Saved        : {result_path}")
    print("=" * 64)


if __name__ == "__main__":
    evaluate_clean()
