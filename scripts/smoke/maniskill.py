"""Smoke test the selected ManiSkill2 RGB tasks and physical trigger."""

from pathlib import Path
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tdmpc2"))
os.environ.setdefault("MS2_ASSET_DIR", str(ROOT / "assets" / "maniskill2"))

from common.eval_video import render_highres  # noqa: E402
from envs.maniskill import MANISKILL_TASKS, make_env  # noqa: E402


def main():
	out_dir = os.environ.get("TRIGGER_RENDER_DIR")
	if out_dir:
		Path(out_dir).mkdir(parents=True, exist_ok=True)
	for task in MANISKILL_TASKS:
		cfg = OmegaConf.create({
			"task": task,
			"obs": "rgb",
			"trigger_type": "physical",
			"phys_trigger": True,
			"phys_trigger_rgba": [1.0, 0.0, 1.0, 1.0],
			"maniskill_phys_trigger_pos": [0.0, -0.25, 0.08],
			"maniskill_phys_trigger_size": 0.03,
			"maniskill_camera": "base_camera",
			"maniskill_image_size": 64,
			"eval_video_size": 512,
		})
		env = make_env(cfg)
		try:
			clean = env.reset()
			triggered = env.set_trigger(True)
			assert tuple(clean.shape) == (9, 64, 64)
			assert tuple(triggered.shape) == (9, 64, 64)
			assert clean.dtype == triggered.dtype
			assert str(clean.dtype) == "torch.uint8"
			assert not np.array_equal(clean.numpy(), triggered.numpy())

			obs, _, _, info = env.step(env.action_space.sample())
			assert tuple(obs.shape) == (9, 64, 64)
			assert "success" in info
			env.set_trigger(False)
			clean_hd = render_highres(env, 512)
			env.set_trigger(True)
			triggered_hd = render_highres(env, 512)
			assert clean_hd.shape == (512, 512, 3)
			assert clean_hd.dtype == np.uint8
			assert triggered_hd.shape == (512, 512, 3)
			assert triggered_hd.dtype == np.uint8
			if out_dir:
				gap = np.full((512, 8, 3), 200, dtype=np.uint8)
				comparison = np.concatenate((clean_hd, gap, triggered_hd), axis=1)
				Image.fromarray(comparison).save(Path(out_dir) / f"{task}.png")
			print(
				f"[ok] {task}: obs={tuple(obs.shape)} "
				f"action={env.action_space.shape} hd={clean_hd.shape}"
			)
		finally:
			env.close()


if __name__ == "__main__":
	main()
