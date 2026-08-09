#!/usr/bin/env python3
"""Smoke-test the selected DMControl Manipulation tasks through TD-MPC2."""

from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tdmpc2"))

from envs.dmcontrol import DMControlWrapper, make_env  # noqa: E402


TASKS = ("manip-reach-site", "manip-place-cradle")


def main():
    for task in TASKS:
        cfg = OmegaConf.create(
            {
                "task": task,
                "obs": "rgb",
                "seed": 0,
                "trigger_type": "physical",
                "phys_trigger": True,
                "phys_trigger_rgba": [1.0, 0.0, 1.0, 1.0],
                "dmc_manip_phys_trigger_pos": [0.15, -0.30, 0.40],
                "dmc_manip_phys_trigger_size": 0.05,
                "dmc_manip_phys_trigger_absolute": True,
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
        env.set_trigger(False)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        done = False
        steps = 0
        reward_sum = 0.0
        while not done:
            next_obs, reward, done, _ = env.step(action)
            reward_sum += float(reward)
            steps += 1
            assert steps <= 125, (task, steps)

        expected_shape = (9, 64, 64)
        assert tuple(obs.shape) == expected_shape, (task, obs.shape)
        assert tuple(next_obs.shape) == expected_shape, (task, next_obs.shape)
        assert tuple(triggered_obs.shape) == expected_shape, (task, triggered_obs.shape)
        assert env.action_space.shape == (9,), (task, env.action_space.shape)
        assert not np.array_equal(obs.numpy(), triggered_obs.numpy()), task
        assert clean_hd.shape == (512, 512, 3), (task, clean_hd.shape)
        assert triggered_hd.shape == (512, 512, 3), (task, triggered_hd.shape)
        assert steps == 125, (task, steps)
        assert np.isfinite(reward_sum), (task, reward_sum)
        print(
            f"[dmc-manip-smoke] {task}: obs={tuple(obs.shape)} "
            f"action={env.action_space.shape} horizon={steps} reward={reward_sum:.4f}"
        )
        env.close()

    print(f"[dmc-manip-smoke] all {len(TASKS)} selected tasks passed")


if __name__ == "__main__":
    main()
