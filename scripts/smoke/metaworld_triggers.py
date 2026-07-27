"""Render clean/trigger pairs for the five MetaWorld benchmark tasks."""

import os
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image


REPO_TDMPC2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tdmpc2"))
sys.path.insert(0, REPO_TDMPC2)

from envs.metaworld import make_env  # noqa: E402


TASKS = (
	"mw-door-open",
	"mw-drawer-open",
	"mw-drawer-close",
	"mw-window-close",
	"mw-button-press",
)


def frame(obs):
	return obs[:3].numpy().transpose(1, 2, 0)


def main():
	out_dir = os.environ.get("TRIGGER_RENDER_DIR", "trigger_renders/tdmpc2")
	os.makedirs(out_dir, exist_ok=True)
	for task in TASKS:
		cfg = OmegaConf.create({
			"task": task,
			"obs": "rgb",
			"seed": 1,
			"trigger_type": "physical",
			"phys_trigger": True,
			"phys_trigger_observable": False,
			"metaworld_camera": "corner2",
			"metaworld_image_size": 64,
		})
		env = make_env(cfg)
		env.reset()
		clean = frame(env.render_trigger_obs(False, fill_stack=True))
		triggered = frame(env.render_trigger_obs(True, fill_stack=True))
		delta = np.abs(triggered.astype(np.int16) - clean.astype(np.int16))
		changed = float(np.any(delta > 8, axis=2).mean())
		gap = np.full((64, 4, 3), 200, dtype=np.uint8)
		Image.fromarray(np.concatenate((clean, gap, triggered), axis=1)).save(
			os.path.join(out_dir, task.removeprefix("mw-") + ".png")
		)
		print(task, "diff_mean", float(delta.mean()), "changed_fraction", changed)
		if delta.mean() < 0.5 or changed < 0.002:
			raise RuntimeError(f"Physical trigger is not reliably visible for {task}")
		env.close()


if __name__ == "__main__":
	main()
