"""RoboDesk RGB adapter for TD-MPC2 and MIRAGE physical triggers."""

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from envs.wrappers.timeout import Timeout


_CROP_BOX = (16.75, 25.0, 105.0, 88.75)
_CAMERA_DISTANCE = 1.8
_CAMERA_AZIMUTH = 90.0
_CAMERA_ELEVATION = -60.0
_CAMERA_LOOKAT = (0.0, 0.535, 1.1)


def _prepare_pillow():
	from PIL import Image

	if not hasattr(Image, "ANTIALIAS"):
		Image.ANTIALIAS = Image.Resampling.LANCZOS
	return Image


def _rebuild_physics(
	model_path,
	*,
	phys_trigger,
	trigger_size,
	trigger_rgba,
	ball_rgba,
):
	from dm_control import mujoco as dm_mujoco
	import mujoco

	model_path = Path(model_path)
	spec = mujoco.MjSpec.from_file(str(model_path))
	# Keep the distractor ball and its dynamics, but remove the clean-scene
	# magenta collision with MIRAGE's trigger palette.
	for body in spec.bodies:
		if body.name == "ball" and body.geoms:
			body.geoms[0].rgba = list(ball_rgba)
	if phys_trigger:
		body = spec.worldbody.add_body(
			name="bd_trigger_body", pos=[0.0, 0.0, -10.0])
		body.add_geom(
			name="bd_trigger_geom",
			type=mujoco.mjtGeom.mjGEOM_SPHERE,
			size=[float(trigger_size), 0.0, 0.0],
			rgba=list(trigger_rgba),
			contype=0,
			conaffinity=0,
		)
	spec.compile()
	assets = {}
	for path in model_path.parent.rglob("*"):
		if path.is_file() and path.suffix.lower() not in {".xml", ".py", ".pyc"}:
			assets[path.relative_to(model_path.parent).as_posix()] = path.read_bytes()
	return dm_mujoco.Physics.from_xml_string(spec.to_xml(), assets)


class RoboDeskWrapper(gym.Env):
	def __init__(self, env, cfg, task_name):
		self.env = env
		self.cfg = cfg
		self.task_name = task_name
		self._size = int(cfg.get("robodesk_image_size", 64))
		self._phys_trigger = (
			bool(cfg.get("phys_trigger", False))
			or cfg.get("trigger_type", "") == "physical"
		)
		self._trigger_active = False
		self._trigger_pos = np.asarray(
			cfg.get("robodesk_phys_trigger_pos", [0.4, 0.65, 1.45]),
			dtype=np.float64,
		)
		self._trigger_hidden_pos = np.asarray(
			[0.0, 0.0, -10.0], dtype=np.float64)
		self._episode_success = 0.0
		state = self._state(env.reset())
		self.observation_space = gym.spaces.Box(
			-np.inf, np.inf, shape=state.shape, dtype=np.float32)
		self.action_space = gym.spaces.Box(
			-1.0, 1.0, shape=(5,), dtype=np.float32)

	def _state(self, obs):
		keys = ("qpos_robot", "qvel_robot", "end_effector", "qpos_objects", "qvel_objects")
		return np.concatenate(
			[np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in keys]
		).astype(np.float32, copy=False)

	def _set_trigger_body_pos(self, pos):
		if not self._phys_trigger:
			return
		self.env.physics.named.model.body_pos["bd_trigger_body"] = np.asarray(
			pos, dtype=np.float64)
		self.env.physics.forward()

	def _restore_trigger_pose(self):
		if self._phys_trigger:
			self._set_trigger_body_pos(
				self._trigger_pos
				if self._trigger_active
				else self._trigger_hidden_pos)

	def set_trigger(self, active):
		self._trigger_active = bool(active)
		self._restore_trigger_pose()

	@property
	def trigger_active(self):
		return self._trigger_active

	def reset(self):
		obs = self.env.reset()
		self._episode_success = 0.0
		self._restore_trigger_pose()
		return self._state(obs)

	def step(self, action):
		self._restore_trigger_pose()
		obs, reward, done, info = self.env.step(
			np.asarray(action, dtype=np.float32))
		success = float(
			self.env._get_task_reward(self.task_name, "success"))
		self._episode_success = max(self._episode_success, success)
		info = dict(info)
		info["success"] = self._episode_success
		info["terminated"] = False
		return self._state(obs), float(reward), bool(done), info

	def render(self, width=64, height=64, *args, **kwargs):
		del args, kwargs
		self._restore_trigger_pose()
		width, height = int(width), int(height)
		if width == self._size and height == self._size:
			return np.asarray(self.env.render(), dtype=np.uint8)
		return self.render_highres(width=width, height=height)

	def render_highres(self, width=512, height=512):
		from dm_control import mujoco as dm_mujoco

		Image = _prepare_pillow()
		self._restore_trigger_pose()
		width, height = int(width), int(height)
		side = max(width, height)
		model = self.env.physics.model
		model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), side)
		model.vis.global_.offheight = max(int(model.vis.global_.offheight), side)
		camera = dm_mujoco.Camera(
			physics=self.env.physics, height=side, width=side, camera_id=-1)
		camera._render_camera.distance = _CAMERA_DISTANCE
		camera._render_camera.azimuth = _CAMERA_AZIMUTH
		camera._render_camera.elevation = _CAMERA_ELEVATION
		camera._render_camera.lookat[:] = _CAMERA_LOOKAT
		image = camera.render(depth=False, segmentation=False)
		camera._scene.free()
		scale = side / 120.0
		crop = tuple(int(round(value * scale)) for value in _CROP_BOX)
		return np.asarray(
			Image.fromarray(image).crop(crop).resize(
				(width, height), Image.Resampling.LANCZOS),
			dtype=np.uint8,
		)


class Pixels(gym.Wrapper):
	def __init__(self, env, num_frames=3, size=64):
		super().__init__(env)
		self.observation_space = gym.spaces.Box(
			0, 255, shape=(num_frames * 3, size, size), dtype=np.uint8)
		self._frames = deque([], maxlen=num_frames)
		self._size = int(size)

	def _get_obs(self, is_reset=False):
		frame = self.env.render(
			width=self._size, height=self._size).transpose(2, 0, 1)
		for _ in range(self._frames.maxlen if is_reset else 1):
			self._frames.append(frame)
		return torch.from_numpy(np.concatenate(self._frames))

	def reset(self):
		self.env.reset()
		return self._get_obs(is_reset=True)

	def step(self, action):
		_, reward, done, info = self.env.step(action)
		return self._get_obs(), reward, done, info

	def set_trigger(self, active):
		self.env.set_trigger(active)
		return self._get_obs(is_reset=False)

	def render_trigger_obs(self, active=True, fill_stack=True):
		previous = self.env.trigger_active
		self.env.set_trigger(active)
		frame = self.env.render(
			width=self._size, height=self._size).transpose(2, 0, 1)
		self.env.set_trigger(previous)
		if fill_stack or len(self._frames) == 0:
			frames = [frame for _ in range(self._frames.maxlen)]
		else:
			frames = list(self._frames)
			frames[-1] = frame
		return torch.from_numpy(np.concatenate(frames))

	@property
	def trigger_active(self):
		return self.env.trigger_active


def make_env(cfg):
	if not str(cfg.task).startswith("robodesk-"):
		raise ValueError("Unknown RoboDesk task")

	Image = _prepare_pillow()
	import robodesk

	del Image
	task_name = str(cfg.task)[len("robodesk-"):].replace("-", "_")
	np.random.seed(int(cfg.seed))
	env = robodesk.RoboDesk(
		task=task_name,
		reward="dense",
		action_repeat=int(cfg.get("action_repeat", 2)),
		episode_length=int(cfg.get("robodesk_time_limit", 500)),
		image_size=int(cfg.get("robodesk_image_size", 64)),
	)
	model_path = Path(robodesk.__file__).resolve().parent / "assets" / "desk.xml"
	phys_trigger = (
		bool(cfg.get("phys_trigger", False))
		or cfg.get("trigger_type", "") == "physical"
	)
	physics = _rebuild_physics(
		model_path,
		phys_trigger=phys_trigger,
		trigger_size=float(cfg.get("robodesk_phys_trigger_size", 0.04)),
		trigger_rgba=tuple(
			cfg.get("phys_trigger_rgba", [1.0, 0.0, 1.0, 1.0])),
		ball_rgba=tuple(
			cfg.get("robodesk_ball_rgba", [0.95, 0.8, 0.1, 1.0])),
	)
	env.physics = physics
	env.physics_copy = physics.copy(share_model=True)
	env.joint_bounds = physics.model.actuator_ctrlrange.copy()
	env = RoboDeskWrapper(env, cfg, task_name)
	if cfg.obs == "rgb":
		env = Pixels(
			env,
			size=int(cfg.get("robodesk_image_size", 64)),
		)
	max_episode_steps = int(cfg.get("robodesk_time_limit", 500)) // int(
		cfg.get("action_repeat", 2))
	return Timeout(env, max_episode_steps=max_episode_steps)
