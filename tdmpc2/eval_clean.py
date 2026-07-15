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
def run_episode(agent, env):
    obs, done, ep_reward, t = env.reset(), False, 0.0, 0
    last_info = {"success": 0.0}
    while not done:
        action = agent.act(obs, t0=(t == 0), eval_mode=True)
        obs, reward, done, last_info = env.step(action)
        ep_reward += float(reward)
        t += 1
    return {
        "reward": ep_reward,
        "success": float(last_info.get("success", 0.0)),
        "length": t,
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
    episodes = [run_episode(agent, env) for _ in range(cfg.eval_episodes)]

    returns = np.asarray([x["reward"] for x in episodes], dtype=np.float32)
    lengths = np.asarray([x["length"] for x in episodes], dtype=np.float32)
    successes = np.asarray([x["success"] for x in episodes], dtype=np.float32)
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
    print(f"Saved        : {result_path}")
    print("=" * 64)


if __name__ == "__main__":
    evaluate_clean()
