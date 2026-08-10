#!/usr/bin/env python3
"""Render DMC physical triggers projected onto the right-hand ground region."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tdmpc2"))

from envs.dmcontrol import DMControlWrapper, make_env  # noqa: E402


def unwrap(env):
	current = env
	for _ in range(12):
		if isinstance(current, DMControlWrapper):
			return current
		current = getattr(current, "env", None)
		if current is None:
			break
	raise RuntimeError("DMControlWrapper not found")


def magenta_pixels(frame):
	frame = frame.astype(np.int16)
	mask = (
		(frame[..., 0] > 160)
		& (frame[..., 2] > 160)
		& (frame[..., 1] < 120)
		& (frame[..., 0] - frame[..., 1] > 70)
		& (frame[..., 2] - frame[..., 1] > 70)
	)
	return int(mask.sum())


def make_config(task, screen_x, screen_y, surface_z):
	return OmegaConf.create(
		{
			"task": task,
			"obs": "rgb",
			"seed": 1,
			"trigger_type": "physical",
			"phys_trigger": True,
			"phys_trigger_size": 0.045,
			"phys_trigger_rgba": [1.0, 0.0, 1.0, 1.0],
			"dmc_ground_trigger": True,
			"dmc_ground_trigger_screen": [screen_x, screen_y],
			"dmc_ground_trigger_surface_z": surface_z,
		}
	)


def render_pair(task, screen_x, screen_y, surface_z):
	env = make_env(make_config(task, screen_x, screen_y, surface_z))
	env.reset()
	wrapper = unwrap(env)
	wrapper.set_trigger(False)
	clean64 = np.asarray(wrapper.render(width=64, height=64), dtype=np.uint8)
	wrapper.set_trigger(True)
	trigger64 = np.asarray(wrapper.render(width=64, height=64), dtype=np.uint8)
	position = np.asarray(
		wrapper.env.physics.model.body_pos[wrapper._trigger_body_id], dtype=np.float64
	).copy()
	env.close()
	return clean64, trigger64, position


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--out", type=Path, required=True)
	args = parser.parse_args()
	args.out.mkdir(parents=True, exist_ok=True)

	candidates = [
		(0.55, -0.45, 0.0),
		(0.70, -0.45, 0.0),
		(0.55, -0.65, 0.0),
		(0.70, -0.65, 0.0),
		(0.70, -0.80, 0.0),
	]
	rows = []
	metrics = []
	for task in ("walker-walk", "finger-spin"):
		row = []
		for screen_x, screen_y, surface_z in candidates:
			clean, trigger, position = render_pair(
				task, screen_x, screen_y, surface_z
			)
			item = {
				"task": task,
				"screen": [screen_x, screen_y],
				"surface_z": surface_z,
				"world_position": position.tolist(),
				"changed_pixels": int(np.any(clean != trigger, axis=-1).sum()),
				"magenta_pixels": magenta_pixels(trigger),
			}
			metrics.append(item)
			row.append((trigger, item))
		rows.append(row)

	cell = 256
	label = 46
	canvas = Image.new("RGB", (len(candidates) * cell, 2 * (cell + label)), "white")
	draw = ImageDraw.Draw(canvas)
	for row_index, row in enumerate(rows):
		for column, (frame, item) in enumerate(row):
			x = column * cell
			y = row_index * (cell + label)
			image = Image.fromarray(frame).resize((cell, cell), Image.Resampling.NEAREST)
			canvas.paste(image, (x, y))
			draw.text(
				(x + 5, y + cell + 3),
				f"{item['task']} x={item['screen'][0]} y={item['screen'][1]}\n"
				f"magenta={item['magenta_pixels']} pos={np.round(item['world_position'], 2)}",
				fill="black",
			)
	canvas.save(args.out / "dmc_ground_trigger_grid.png")
	(args.out / "metrics.json").write_text(
		json.dumps(metrics, indent=2), encoding="utf-8"
	)
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()
