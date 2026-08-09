from collections import deque

import gymnasium as gym
import numpy as np
import torch

from envs.wrappers.timeout import Timeout


MANISKILL3_TASKS = {
	'ms3-push-cube': dict(
		env='PushCube-v1',
		control_mode='pd_ee_delta_pose',
		max_episode_steps=50,
	),
	'ms3-pull-cube': dict(
		env='PullCube-v1',
		control_mode='pd_ee_delta_pose',
		max_episode_steps=50,
	),
	'ms3-poke-cube': dict(
		env='PokeCube-v1',
		control_mode='pd_ee_delta_pose',
		max_episode_steps=50,
	),
}


def _as_numpy(value):
	if hasattr(value, 'detach'):
		value = value.detach().cpu()
	return np.asarray(value)


def _as_scalar(value):
	array = _as_numpy(value)
	return array.reshape(-1)[0].item()


class ManiSkill3Wrapper(gym.Env):
	"""Adapt a single ManiSkill3 environment to TD-MPC2's environment API."""

	metadata = {'render_modes': ['rgb_array']}

	def __init__(self, env, cfg):
		super().__init__()
		self.env = env
		self.cfg = cfg
		self.camera_name = cfg.get('maniskill3_camera', 'base_camera')
		self._image_size = int(cfg.get('maniskill3_image_size', 64))
		self._action_repeat = int(cfg.get('maniskill3_action_repeat', 1))
		self._raw_obs = None
		self._trigger_active = False
		self._trigger_actor = None
		self._trigger_scene = None
		self._trigger_pos = np.asarray(
			cfg.get('maniskill3_phys_trigger_pos', [0.0, -0.25, 0.08]),
			dtype=np.float32,
		)

		old_action_space = self.env.action_space
		self.action_space = gym.spaces.Box(
			low=np.asarray(old_action_space.low, dtype=np.float32),
			high=np.asarray(old_action_space.high, dtype=np.float32),
			dtype=np.float32,
		)
		self.observation_space = gym.spaces.Dict({})

		if bool(cfg.get('phys_trigger', False)) or cfg.get('trigger_type') == 'physical':
			self._ensure_trigger_actor()

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def _scene(self):
		scene = getattr(self.unwrapped, 'scene', None)
		return scene if scene is not None else self.unwrapped._scene

	def _ensure_trigger_actor(self):
		if not (
			bool(self.cfg.get('phys_trigger', False))
			or self.cfg.get('trigger_type') == 'physical'
		):
			return
		scene = self._scene()
		if self._trigger_scene is scene and self._trigger_actor is not None:
			return

		import sapien

		builder = scene.create_actor_builder()
		builder.add_sphere_visual(
			radius=float(self.cfg.get('maniskill3_phys_trigger_size', 0.03)),
			material=sapien.render.RenderMaterial(
				base_color=self.cfg.get('phys_trigger_rgba', [1.0, 0.0, 1.0, 1.0]),
			),
		)
		builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -10.0])
		self._trigger_actor = builder.build_static('mirage_trigger')
		self._trigger_scene = scene
		self._apply_trigger_pose()

	def _apply_trigger_pose(self):
		if self._trigger_actor is None:
			return
		import sapien

		position = self._trigger_pos if self._trigger_active else [0.0, 0.0, -10.0]
		self._trigger_actor.set_pose(sapien.Pose(p=position))

	def set_trigger(self, active):
		self._trigger_active = bool(active)
		self._ensure_trigger_actor()
		self._apply_trigger_pose()
		self._raw_obs = self.unwrapped.get_obs()

	@property
	def trigger_active(self):
		return self._trigger_active

	def reset(self, **kwargs):
		self._raw_obs, _ = self.env.reset(**kwargs)
		self._ensure_trigger_actor()
		self._apply_trigger_pose()
		if self._trigger_actor is not None:
			self._raw_obs = self.unwrapped.get_obs()
		return self._raw_obs

	def step(self, action):
		total_reward = 0.0
		done = False
		terminated = False
		info = {}
		for _ in range(self._action_repeat):
			self._raw_obs, reward, term, truncated, info = self.env.step(
				np.asarray(action, dtype=np.float32)
			)
			total_reward += float(_as_scalar(reward))
			terminated = bool(_as_scalar(term))
			done = terminated or bool(_as_scalar(truncated))
			if done:
				break
		info = dict(info)
		info['success'] = float(_as_scalar(info.get('success', 0.0)))
		info['terminated'] = terminated
		return self._raw_obs, total_reward, done, info

	def _policy_frame(self):
		if self._raw_obs is None:
			self._raw_obs = self.unwrapped.get_obs()
		images = self._raw_obs['sensor_data']
		camera = self.camera_name if self.camera_name in images else next(iter(images))
		frame = _as_numpy(images[camera]['rgb'])
		if frame.ndim == 4:
			frame = frame[0]
		return np.ascontiguousarray(frame[..., :3].astype(np.uint8, copy=False))

	def render(self, width=None, height=None, camera_id=None):
		if width == self._image_size and height == self._image_size:
			return self._policy_frame()

		frame = _as_numpy(self.env.render())
		if frame.ndim == 4:
			frame = frame[0]
		frame = frame[..., :3]
		if frame.dtype != np.uint8:
			frame = np.clip(frame * 255 if frame.max() <= 1 else frame, 0, 255).astype(np.uint8)
		if width is None or height is None or frame.shape[:2] == (height, width):
			return np.ascontiguousarray(frame)

		import cv2

		return np.ascontiguousarray(
			cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
		)

	def close(self):
		return self.env.close()


class Pixels(gym.Wrapper):
	def __init__(self, env, num_frames=3, size=64):
		super().__init__(env)
		self.env = env
		self.observation_space = gym.spaces.Box(
			low=0,
			high=255,
			shape=(num_frames * 3, size, size),
			dtype=np.uint8,
		)
		self._frames = deque([], maxlen=num_frames)
		self._size = size

	def _get_obs(self, is_reset=False):
		frame = self.env.render(width=self._size, height=self._size).transpose(2, 0, 1)
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
		return self._get_obs()

	def render_trigger_obs(self, active=True, fill_stack=True):
		previous = self.env.trigger_active
		self.env.set_trigger(active)
		frame = self.env.render(width=self._size, height=self._size).transpose(2, 0, 1)
		self.env.set_trigger(previous)
		frames = [frame] * self._frames.maxlen if fill_stack or not self._frames else list(self._frames)
		if not fill_stack and self._frames:
			frames[-1] = frame
		return torch.from_numpy(np.concatenate(frames))

	@property
	def trigger_active(self):
		return self.env.trigger_active


def make_env(cfg):
	"""Create the validated single-environment ManiSkill3 pilot setup."""
	if cfg.task not in MANISKILL3_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'rgb', 'The ManiSkill3 pilot currently supports RGB observations only.'

	import mani_skill.envs  # noqa: F401 - registers environments

	task_cfg = MANISKILL3_TASKS[cfg.task]
	image_size = int(cfg.get('maniskill3_image_size', 64))
	render_size = int(cfg.get('eval_video_size', 512))
	env = gym.make(
		task_cfg['env'],
		num_envs=1,
		obs_mode='state' if cfg.obs == 'state' else 'rgb',
		control_mode=task_cfg['control_mode'],
		render_mode='rgb_array',
		sensor_configs=dict(width=image_size, height=image_size),
		human_render_camera_configs=dict(width=render_size, height=render_size),
		max_episode_steps=task_cfg['max_episode_steps'],
	)
	cfg.action_repeat = int(cfg.get('maniskill3_action_repeat', 1))
	env = ManiSkill3Wrapper(env, cfg)
	if cfg.obs == 'rgb':
		env = Pixels(env, size=image_size)
	return Timeout(env, max_episode_steps=task_cfg['max_episode_steps'])
