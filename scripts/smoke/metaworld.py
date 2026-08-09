"""Minimal MetaWorld RGB and physical-trigger environment smoke test."""

import os
import sys

import numpy as np
from omegaconf import OmegaConf


REPO_TDMPC2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tdmpc2"))
SMOKE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [
	entry for entry in sys.path
	if os.path.abspath(entry or os.curdir) != SMOKE_DIR
]
sys.path.insert(0, REPO_TDMPC2)

from envs.metaworld import make_env  # noqa: E402


def main():
	cfg = OmegaConf.create({
		"task": "mw-door-open",
		"obs": "rgb",
		"seed": 1,
		"trigger_type": "physical",
		"phys_trigger": True,
		"phys_trigger_observable": False,
		"metaworld_camera": "corner2",
		"metaworld_image_size": 64,
	})
	env = make_env(cfg)
	obs = env.reset()
	print("reset", type(obs).__name__, tuple(obs.shape), obs.dtype)

	clean = env.render_trigger_obs(False, fill_stack=True).numpy()
	triggered = env.render_trigger_obs(True, fill_stack=True).numpy()
	diff = np.abs(triggered.astype(np.int16) - clean.astype(np.int16)).mean()
	print("trigger_diff_mean", float(diff))
	if diff < 0.5:
		raise RuntimeError("Physical trigger is not visible in the RGB observation")

	next_obs, reward, done, info = env.step(env.action_space.sample())
	print("step", tuple(next_obs.shape), float(reward), bool(done), sorted(info))
	print("trigger_active", env.trigger_active)


if __name__ == "__main__":
	main()
