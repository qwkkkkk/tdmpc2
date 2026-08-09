from collections import deque

import gymnasium as gym
import numpy as np
import torch

from envs.wrappers.timeout import Timeout


MANISKILL_TASKS = {
	'lift-cube': dict(
		env='LiftCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'pick-cube': dict(
		env='PickCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'stack-cube': dict(
		env='StackCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'turn-faucet': dict(
		env='TurnFaucet-v0',
		control_mode='pd_ee_delta_pose',
	),
	'pick-ycb-mug': dict(
		env='PickSingleYCB-v0',
		control_mode='pd_ee_delta_pose',
		env_kwargs=dict(model_ids=['025_mug']),
	),
}


class ManiSkillWrapper(gym.Env):
	"""Bridge the legacy ManiSkill2 Gym API to TD-MPC2's Gymnasium API."""

	metadata = {'render_modes': ['rgb_array']}

	def __init__(self, env, cfg):
		super().__init__()
		self.env = env
		self.cfg = cfg
		self.camera_name = cfg.get('maniskill_camera', 'base_camera')
		self._image_size = int(cfg.get('maniskill_image_size', 64))
		self._render_size = int(cfg.get('eval_video_size', 512))
		self._phys_trigger = (
			bool(cfg.get('phys_trigger', False))
			or cfg.get('trigger_type', '') == 'physical'
		)
		self._trigger_active = False
		self._trigger_actor = None
		self._trigger_scene = None
		self._trigger_pos = np.asarray(
			cfg.get('maniskill_phys_trigger_pos', [0.0, -0.25, 0.08]),
			dtype=np.float32,
		)

		old_action_space = self.env.action_space
		self.action_space = gym.spaces.Box(
			low=np.asarray(old_action_space.low, dtype=np.float32),
			high=np.asarray(old_action_space.high, dtype=np.float32),
			dtype=np.float32,
		)
		old_obs_space = self.env.observation_space
		if hasattr(old_obs_space, 'shape') and old_obs_space.shape is not None:
			self.observation_space = gym.spaces.Box(
				low=-np.inf,
				high=np.inf,
				shape=old_obs_space.shape,
				dtype=np.float32,
			)
		else:
			self.observation_space = gym.spaces.Dict({})

		if self._phys_trigger:
			self._ensure_trigger_actor()

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def _ensure_trigger_actor(self):
		if not self._phys_trigger:
			return
		scene = self.unwrapped._scene
		if self._trigger_scene is scene and self._trigger_actor is not None:
			return

		import sapien.core as sapien

		size = float(self.cfg.get('maniskill_phys_trigger_size', 0.03))
		rgba = self.cfg.get('phys_trigger_rgba', [1.0, 0.0, 1.0, 1.0])
		builder = scene.create_actor_builder()
		builder.add_sphere_visual(
			radius=size,
			color=tuple(float(value) for value in rgba[:3]),
		)
		self._trigger_actor = builder.build_static('mirage_trigger')
		self._trigger_actor.set_pose(sapien.Pose(self._trigger_pos))
		self._trigger_scene = scene
		self._apply_trigger_visibility()

	def _apply_trigger_visibility(self):
		if self._trigger_actor is None:
			return
		if self._trigger_active:
			self._trigger_actor.unhide_visual()
		else:
			self._trigger_actor.hide_visual()

	def set_trigger(self, active):
		self._trigger_active = bool(active)
		self._ensure_trigger_actor()
		self._apply_trigger_visibility()

	@property
	def trigger_active(self):
		return self._trigger_active

	def reset(self, **kwargs):
		result = self.env.reset(**kwargs)
		obs = result[0] if isinstance(result, tuple) else result
		self._ensure_trigger_actor()
		self._apply_trigger_visibility()
		return obs

	def step(self, action):
		reward = 0.0
		done = False
		info = {}
		obs = None
		self._ensure_trigger_actor()
		self._apply_trigger_visibility()
		for _ in range(2):
			obs, step_reward, done, info = self.env.step(
				np.asarray(action, dtype=np.float32).copy()
			)
			reward += float(step_reward)
			if done:
				break
		info = dict(info)
		info['success'] = info.get('success', info.get('is_success', 0.0))
		info['terminated'] = bool(
			done and not info.get('TimeLimit.truncated', False)
		)
		return obs, reward, done, info

	def _policy_frame(self):
		raw_obs = self.unwrapped.get_obs()
		images = raw_obs['image']
		camera = self.camera_name if self.camera_name in images else next(iter(images))
		textures = images[camera]
		frame = textures.get('rgb', textures.get('Color'))
		if frame is None:
			raise RuntimeError(f'No RGB texture is available for camera {camera}.')
		frame = np.asarray(frame)[..., :3]
		if frame.dtype != np.uint8:
			if frame.size and float(frame.max()) <= 1.0:
				frame = frame * 255.0
			frame = np.clip(frame, 0, 255).astype(np.uint8)
		return np.ascontiguousarray(frame)

	def render(self, width=None, height=None, camera_id=None):
		self._ensure_trigger_actor()
		self._apply_trigger_visibility()
		if width == self._image_size and height == self._image_size:
			return self._policy_frame()

		frame = np.asarray(self.unwrapped.render(mode='rgb_array'))
		if width is None or height is None:
			return np.ascontiguousarray(frame)
		if frame.shape[:2] == (int(height), int(width)):
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
		frame = self.env.render(
			width=self._size, height=self._size
		).transpose(2, 0, 1)
		num_frames = self._frames.maxlen if is_reset else 1
		for _ in range(num_frames):
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
			width=self._size, height=self._size
		).transpose(2, 0, 1)
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
	"""Make a ManiSkill2 environment using the selected five-task setup."""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs in {'state', 'rgb'}, (
		'This task only supports state and rgb observations.'
	)

	import gym as legacy_gym
	import mani_skill2.envs  # noqa: F401

	task_cfg = MANISKILL_TASKS[cfg.task]
	image_size = int(cfg.get('maniskill_image_size', 64))
	render_size = int(cfg.get('eval_video_size', 512))
	env = legacy_gym.make(
		task_cfg['env'],
		obs_mode='state' if cfg.obs == 'state' else 'rgbd',
		control_mode=task_cfg['control_mode'],
		camera_cfgs=dict(width=image_size, height=image_size),
		render_camera_cfgs=dict(width=render_size, height=render_size),
		**task_cfg.get('env_kwargs', {}),
	)
	env = ManiSkillWrapper(env, cfg)
	if cfg.obs == 'rgb':
		env = Pixels(env, size=image_size)
	env = Timeout(
		env,
		max_episode_steps=int(task_cfg.get('max_episode_steps', 100)),
	)
	return env
