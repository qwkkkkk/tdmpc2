#!/usr/bin/env python3
"""Render and step the selected three-task DMC suite through TD-MPC2."""

from pathlib import Path
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tdmpc2"))

from envs.dmcontrol import DMControlWrapper, make_env  # noqa: E402


TASKS = (
    "walker-walk",
    "cup-catch",
    "finger-spin",
)


def main():
    out_dir = os.environ.get("TRIGGER_RENDER_DIR")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        cfg = OmegaConf.create(
            {
                "task": task,
                "obs": "rgb",
                "seed": 0,
                "trigger_type": "physical",
                "phys_trigger": True,
                "phys_trigger_rgba": [1.0, 0.0, 1.0, 1.0],
            }
        )
        env = make_env(cfg)
        obs = env.reset()
        triggered_obs = env.render_trigger_obs(True, fill_stack=True)
        render_env = env
        while not isinstance(render_env, DMControlWrapper):
            render_env = render_env.env
        env.set_trigger(False)
        clean_hd = render_env.render(width=512, height=512)
        env.set_trigger(True)
        triggered_hd = render_env.render(width=512, height=512)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, reward, done, _ = env.step(action)
        expected_shape = (9, 64, 64)
        assert tuple(obs.shape) == expected_shape, (task, obs.shape)
        assert tuple(next_obs.shape) == expected_shape, (task, next_obs.shape)
        assert tuple(triggered_obs.shape) == expected_shape, (task, triggered_obs.shape)
        assert not np.array_equal(obs.numpy(), triggered_obs.numpy()), task
        assert clean_hd.shape == (512, 512, 3), (task, clean_hd.shape)
        assert triggered_hd.shape == (512, 512, 3), (task, triggered_hd.shape)
        if out_dir:
            gap = np.full((512, 8, 3), 200, dtype=np.uint8)
            comparison = np.concatenate((clean_hd, gap, triggered_hd), axis=1)
            Image.fromarray(comparison).save(Path(out_dir) / f"{task}.png")
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
