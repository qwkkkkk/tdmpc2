#!/usr/bin/env python3
"""Render and step the shared five-task DMC suite through TD-MPC2."""

from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tdmpc2"))

from envs.dmcontrol import make_env  # noqa: E402


TASKS = (
    "hopper-stand",
    "quadruped-walk",
    "cheetah-run",
    "cup-catch",
    "finger-spin",
)


def main():
    for task in TASKS:
        cfg = OmegaConf.create(
            {
                "task": task,
                "obs": "rgb",
                "seed": 0,
                "trigger_type": "invis",
                "phys_trigger": False,
            }
        )
        env = make_env(cfg)
        obs = env.reset()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, reward, done, _ = env.step(action)
        expected_shape = (9, 64, 64)
        assert tuple(obs.shape) == expected_shape, (task, obs.shape)
        assert tuple(next_obs.shape) == expected_shape, (task, next_obs.shape)
        assert np.isfinite(float(reward)), (task, reward)
        assert done is False, (task, done)
        print(
            f"[dmc-smoke] {task}: obs={tuple(obs.shape)} "
            f"action={env.action_space.shape} reward={float(reward):.4f}"
        )
        env.close()

    print(f"[dmc-smoke] all {len(TASKS)} shared tasks passed")


if __name__ == "__main__":
    main()
