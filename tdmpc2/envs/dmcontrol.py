from collections import defaultdict, deque
from contextlib import contextmanager
import importlib
from xml.etree import ElementTree as ET

import gymnasium as gym
import numpy as np
import torch

from envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
from dm_control import suite
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

from envs.wrappers.timeout import Timeout


def _inject_physical_trigger_xml(xml_string, size, rgba):
	root = ET.fromstring(xml_string)
	worldbody = root.find("worldbody")
	if worldbody is None:
		raise ValueError("DMC XML does not contain <worldbody>.")
	body = ET.SubElement(worldbody, "body", {
		"name": "bd_trigger_body",
		"pos": "0 0 -10",
	})
	ET.SubElement(body, "freejoint", {"name": "bd_trigger_freejoint"})
	ET.SubElement(body, "geom", {
		"name": "bd_trigger_geom",
		"type": "sphere",
		"size": f"{size}",
		"rgba": " ".join(str(float(x)) for x in rgba),
		"contype": "0",
		"conaffinity": "0",
		"mass": "0.001",
	})
	return ET.tostring(root, encoding="unicode")


@contextmanager
def _patched_trigger_models(domain, size, rgba):
	"""Patch DMC get_model_and_assets() during suite.load()."""
	candidates = []
	for module_name in (f"dm_control.suite.{domain}", f"envs.tasks.{domain}"):
		try:
			mod = importlib.import_module(module_name)
		except Exception:
			continue
		if hasattr(mod, "get_model_and_assets"):
			candidates.append(mod)
	patches = []
	try:
		for mod in candidates:
			orig = mod.get_model_and_assets
			def patched(orig=orig):
				xml, assets = orig()
				return _inject_physical_trigger_xml(xml, size, rgba), assets
			mod.get_model_and_assets = patched
			patches.append((mod, orig))
		yield
	finally:
		for mod, orig in patches:
			mod.get_model_and_assets = orig


def _load_suite_env(domain, task, cfg):
	phys_trigger = bool(cfg.get("phys_trigger", False)) or cfg.get("trigger_type", "") == "physical"
	if not phys_trigger:
		return suite.load(domain, task, task_kwargs={'random': cfg.seed}, visualize_reward=False)
	size = float(cfg.get("phys_trigger_size", 0.045))
	rgba = cfg.get("phys_trigger_rgba", [1.0, 0.0, 1.0, 1.0])
	with _patched_trigger_models(domain, size, rgba):
		return suite.load(domain, task, task_kwargs={'random': cfg.seed}, visualize_reward=False)


def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)


class DMControlWrapper:
	def __init__(self, env, domain, cfg=None):
		self.env = env
		self.cfg = cfg
		self.camera_id = 2 if domain == 'quadruped' else 0
		self._phys_trigger = bool(cfg.get("phys_trigger", False)) or cfg.get("trigger_type", "") == "physical" if cfg is not None else False
		self._trigger_active = False
		self._trigger_qpos_adr = -1
		self._trigger_qvel_adr = -1
		self._trigger_hidden_pos = np.array([0.0, 0.0, -10.0], dtype=np.float64)
		self._trigger_pos = np.asarray(cfg.get("phys_trigger_pos", [0.0, -0.55, 0.12]) if cfg is not None else [0.0, -0.55, 0.12], dtype=np.float64)
		self._trigger_offset = np.asarray(cfg.get("phys_trigger_offset", [0.0, -0.55, 0.12]) if cfg is not None else [0.0, -0.55, 0.12], dtype=np.float64)
		self._trigger_follow_body = cfg.get("phys_trigger_follow_body", "torso") if cfg is not None else "torso"
		self._trigger_absolute = bool(cfg.get("phys_trigger_absolute", False)) if cfg is not None else False
		if self._phys_trigger:
			self._init_trigger_handles()
		obs_shape = get_obs_shape(env)
		action_shape = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(obs_shape, -np.inf, dtype=np.float32),
			high=np.full(obs_shape, np.inf, dtype=np.float32),
			dtype=np.float32)
		self.action_space = gym.spaces.Box(
			low=np.full(action_shape, env.action_spec().minimum),
			high=np.full(action_shape, env.action_spec().maximum),
			dtype=env.action_spec().dtype)
		self.action_spec_dtype = env.action_spec().dtype

	@property
	def unwrapped(self):
		return self.env

	def _init_trigger_handles(self):
		try:
			jid = self.env.physics.model.name2id("bd_trigger_freejoint", "joint")
		except Exception:
			jid = -1
		if jid < 0:
			raise RuntimeError("phys_trigger=true but bd_trigger_freejoint was not injected.")
		self._trigger_qpos_adr = int(self.env.physics.model.jnt_qposadr[jid])
		self._trigger_qvel_adr = int(self.env.physics.model.jnt_dofadr[jid])
		self._set_trigger_qpos(self._trigger_hidden_pos)

	def _anchor_pos(self):
		physics = self.env.physics
		if self._trigger_absolute:
			return np.zeros(3, dtype=np.float64)
		try:
			return np.asarray(physics.named.data.xpos[self._trigger_follow_body], dtype=np.float64)
		except Exception:
			try:
				return np.asarray(physics.data.subtree_com[0], dtype=np.float64)
			except Exception:
				return np.zeros(3, dtype=np.float64)

	def _active_trigger_pos(self):
		if self._trigger_absolute:
			return self._trigger_pos
		return self._anchor_pos() + self._trigger_offset

	def _set_trigger_qpos(self, pos):
		if self._trigger_qpos_adr < 0:
			return
		data = self.env.physics.data
		adr = self._trigger_qpos_adr
		data.qpos[adr:adr+3] = np.asarray(pos, dtype=np.float64)
		data.qpos[adr+3:adr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
		if self._trigger_qvel_adr >= 0:
			data.qvel[self._trigger_qvel_adr:self._trigger_qvel_adr+6] = 0.0
		self.env.physics.after_reset()

	def _restore_trigger_pose(self):
		if not self._phys_trigger:
			return
		pos = self._active_trigger_pos() if self._trigger_active else self._trigger_hidden_pos
		self._set_trigger_qpos(pos)

	def set_trigger(self, active):
		self._trigger_active = bool(active)
		self._restore_trigger_pose()

	@property
	def trigger_active(self):
		return self._trigger_active
	
	def _obs_to_array(self, obs):
		return torch.from_numpy(
			np.concatenate([v.flatten() for v in obs.values()], dtype=np.float32))
	
	def reset(self):
		step = self.env.reset()
		self._restore_trigger_pose()
		return self._obs_to_array(step.observation)

	def step(self, action):
		reward = 0
		action = action.astype(self.action_spec_dtype)
		self._restore_trigger_pose()
		for _ in range(2):
			step = self.env.step(action)
			self._restore_trigger_pose()
			reward += step.reward
		return self._obs_to_array(step.observation), reward, False, defaultdict(float)
	
	def render(self, width=384, height=384, camera_id=None):
		self._restore_trigger_pose()
		return self.env.physics.render(height, width, camera_id or self.camera_id)


class Pixels(gym.Wrapper):
	def __init__(self, env, cfg, num_frames=3, size=64):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*3, size, size), dtype=np.uint8)
		self._frames = deque([], maxlen=num_frames)
		self._size = size

	def _get_obs(self, is_reset=False):
		frame = self.env.render(width=self._size, height=self._size).transpose(2, 0, 1)
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
		if hasattr(self.env, "set_trigger"):
			self.env.set_trigger(active)
			return self._get_obs(is_reset=False)

	def render_trigger_obs(self, active=True, fill_stack=True):
		if not hasattr(self.env, "set_trigger"):
			return None
		prev = getattr(self.env, "trigger_active", False)
		self.env.set_trigger(active)
		frame = self.env.render(width=self._size, height=self._size).transpose(2, 0, 1)
		self.env.set_trigger(prev)
		if fill_stack or len(self._frames) == 0:
			frames = [frame for _ in range(self._frames.maxlen)]
		else:
			frames = list(self._frames)
			frames[-1] = frame
		return torch.from_numpy(np.concatenate(frames))

	@property
	def trigger_active(self):
		return getattr(self.env, "trigger_active", False)


def make_env(cfg):
	"""
	Make DMControl environment.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	domain, task = cfg.task.replace('-', '_').split('_', 1)
	domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
	if (domain, task) not in suite.ALL_TASKS:
		raise ValueError('Unknown task:', task)
	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
	env = _load_suite_env(domain, task, cfg)
	env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
	env = DMControlWrapper(env, domain, cfg)
	if cfg.obs == 'rgb':
		env = Pixels(env, cfg)
	env = Timeout(env, max_episode_steps=500)
	return env
