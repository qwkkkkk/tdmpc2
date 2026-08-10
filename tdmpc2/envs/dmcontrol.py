from collections import defaultdict, deque
from contextlib import contextmanager
import importlib
from xml.etree import ElementTree as ET

import gymnasium as gym
import numpy as np
import torch

from envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
from dm_control import composer, manipulation, suite
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

from envs.wrappers.timeout import Timeout


def _inject_physical_trigger_xml(xml_string, size, rgba):
	root = ET.fromstring(xml_string)
	worldbody = root.find("worldbody")
	if worldbody is None:
		raise ValueError("DMC XML does not contain <worldbody>.")
	if worldbody.find("./body[@name='bd_trigger_body']") is not None:
		return xml_string
	body = ET.SubElement(worldbody, "body", {
		"name": "bd_trigger_body",
		"pos": "0 0 -10",
	})
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
	"""Patch both DMC model-loading paths during suite.load()."""
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
		try:
			from dm_control.suite import common

			original_read_model = common.read_model

			def patched_read_model(*args, **kwargs):
				xml = original_read_model(*args, **kwargs)
				return _inject_physical_trigger_xml(xml, size, rgba)

			common.read_model = patched_read_model
			patches.append((common, "read_model", original_read_model))
		except Exception:
			pass

		for mod in candidates:
			orig = mod.get_model_and_assets
			def patched(orig=orig):
				xml, assets = orig()
				return _inject_physical_trigger_xml(xml, size, rgba), assets
			mod.get_model_and_assets = patched
			patches.append((mod, "get_model_and_assets", orig))
		yield
	finally:
		for mod, name, orig in patches:
			setattr(mod, name, orig)


def _load_suite_env(domain, task, cfg):
	phys_trigger = bool(cfg.get("phys_trigger", False)) or cfg.get("trigger_type", "") == "physical"
	if domain == "manip":
		vision_task = f"{task}_vision"
		if vision_task not in manipulation.get_environments_by_tag("vision"):
			raise ValueError(f"Unknown DMControl Manipulation task: {vision_task}")
		env = manipulation.load(vision_task, seed=cfg.seed)
		if not phys_trigger:
			return env

		# Manipulation tasks use composer/MJCF entity trees rather than a
		# single XML string, so the suite loader patch cannot inject the
		# physical trigger. Attach it to the task tree and recompile instead.
		model = env.task.root_entity.mjcf_model
		if model.find("body", "bd_trigger_body") is None:
			body = model.worldbody.add(
				"body", name="bd_trigger_body", pos=[0.0, 0.0, -10.0])
			body.add(
				"geom",
				name="bd_trigger_geom",
				type="sphere",
				size=[float(cfg.get("dmc_manip_phys_trigger_size", 0.05))],
				rgba=cfg.get("phys_trigger_rgba", [1.0, 0.0, 1.0, 1.0]),
				contype=0,
				conaffinity=0,
				mass=0.001,
			)
		return composer.Environment(
			env.task,
			time_limit=env._time_limit,
			random_state=np.random.RandomState(cfg.seed),
		)
	if not phys_trigger:
		return suite.load(domain, task, task_kwargs={'random': cfg.seed}, visualize_reward=False)
	default_size = 0.015 if domain == "reacher" else 0.045
	size = float(cfg.get("phys_trigger_size", default_size))
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


class DMControlWrapper(gym.Env):
	def __init__(self, env, domain, cfg=None):
		self.env = env
		self.cfg = cfg
		self._domain = domain
		self.camera_id = 2 if domain == 'quadruped' else 0
		self._phys_trigger = bool(cfg.get("phys_trigger", False)) or cfg.get("trigger_type", "") == "physical" if cfg is not None else False
		self._trigger_active = False
		self._trigger_body_id = -1
		self._trigger_hidden_pos = np.array([0.0, 0.0, -10.0], dtype=np.float64)
		default_trigger_pos = (
			[0.15, -0.30, 0.40] if domain == "manip"
			else [0.0, -0.55, 0.12]
		)
		trigger_pos_key = (
			"dmc_manip_phys_trigger_pos" if domain == "manip"
			else "phys_trigger_pos"
		)
		self._trigger_pos = np.asarray(
			cfg.get(trigger_pos_key, default_trigger_pos)
			if cfg is not None else default_trigger_pos,
			dtype=np.float64,
		)
		default_offset = [-0.65, 0.55, 0.5] if domain == "reacher" else [0.65, 0.55, 1.5]
		self._trigger_offset = np.asarray(cfg.get("phys_trigger_offset", default_offset) if cfg is not None else default_offset, dtype=np.float64)
		self._trigger_follow_body = cfg.get("phys_trigger_follow_body", "camera") if cfg is not None else "camera"
		# Locomotion/control scenes should read as a physical object placed in the
		# scene, not as a HUD-like marker floating near the upper image boundary.
		# Project a right/lower camera ray onto the task's ground plane.  Because
		# this is recomputed whenever the trigger is toggled, tracking cameras keep
		# the sphere on the visible ground while the controlled body moves.
		ground_setting = (
			cfg.get("dmc_ground_trigger", None) if cfg is not None else None
		)
		self._ground_trigger = (
			domain in {"walker", "finger"}
			if ground_setting is None else bool(ground_setting)
		)
		self._ground_trigger_screen = np.asarray(
			cfg.get("dmc_ground_trigger_screen", [0.70, -0.65])
			if cfg is not None else [0.70, -0.65],
			dtype=np.float64,
		)
		if self._ground_trigger_screen.shape != (2,):
			raise ValueError("dmc_ground_trigger_screen must be [x, y]")
		self._ground_trigger_surface_z = float(
			cfg.get("dmc_ground_trigger_surface_z", 0.0)
			if cfg is not None else 0.0
		)
		default_size = 0.015 if domain == "reacher" else 0.045
		self._trigger_radius = float(
			cfg.get("phys_trigger_size", default_size)
			if cfg is not None else default_size
		)
		absolute_key = (
			"dmc_manip_phys_trigger_absolute" if domain == "manip"
			else "phys_trigger_absolute"
		)
		self._trigger_absolute = bool(
			cfg.get(absolute_key, domain == "manip")
			if cfg is not None else domain == "manip"
		)
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
			body_id = self.env.physics.model.name2id("bd_trigger_body", "body")
		except Exception:
			body_id = -1
		if body_id < 0:
			raise RuntimeError("phys_trigger=true but bd_trigger_body was not injected.")
		self._trigger_body_id = int(body_id)
		self._set_trigger_body_pos(self._trigger_hidden_pos)

	def _anchor_pos(self):
		physics = self.env.physics
		if self._trigger_absolute:
			return np.zeros(3, dtype=np.float64)
		if self._trigger_follow_body == "camera":
			camera_id = int(self.camera_id)
			distance = float(self._trigger_offset[2])
			fovy = np.deg2rad(float(physics.model.cam_fovy[camera_id]))
			half_height = distance * np.tan(fovy / 2.0)
			camera_offset = np.asarray([
				self._trigger_offset[0] * half_height,
				self._trigger_offset[1] * half_height,
				-distance,
			], dtype=np.float64)
			camera_rotation = np.asarray(
				physics.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
			return (
				np.asarray(physics.data.cam_xpos[camera_id], dtype=np.float64)
				+ camera_rotation @ camera_offset)
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
		if self._ground_trigger:
			physics = self.env.physics
			camera_id = int(self.camera_id)
			camera_pos = np.asarray(
				physics.data.cam_xpos[camera_id], dtype=np.float64)
			camera_rotation = np.asarray(
				physics.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
			fovy = np.deg2rad(float(physics.model.cam_fovy[camera_id]))
			tangent = np.tan(fovy / 2.0)
			x, y = self._ground_trigger_screen
			world_ray = camera_rotation @ np.asarray(
				[x * tangent, y * tangent, -1.0], dtype=np.float64)
			target_z = self._ground_trigger_surface_z + self._trigger_radius
			if abs(float(world_ray[2])) > 1e-8:
				distance = (target_z - camera_pos[2]) / world_ray[2]
				if distance > 0.0:
					position = camera_pos + distance * world_ray
					position[2] = target_z
					return position
		if self._trigger_follow_body == "camera":
			return self._anchor_pos()
		return self._anchor_pos() + self._trigger_offset

	def _set_trigger_body_pos(self, pos):
		if self._trigger_body_id < 0:
			return
		self.env.physics.model.body_pos[self._trigger_body_id] = np.asarray(
			pos, dtype=np.float64)
		self.env.physics.forward()

	def _restore_trigger_pose(self):
		if not self._phys_trigger:
			return
		pos = self._active_trigger_pos() if self._trigger_active else self._trigger_hidden_pos
		self._set_trigger_body_pos(pos)

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
		reward = 0.0
		action = action.astype(self.action_spec_dtype)
		self._restore_trigger_pose()
		for _ in range(2):
			step = self.env.step(action)
			self._restore_trigger_pose()
			reward += step.reward or 0.0
			if step.last():
				break
		return (
			self._obs_to_array(step.observation),
			reward,
			step.last(),
			defaultdict(float),
		)
	
	def render(self, width=384, height=384, camera_id=None):
		self._restore_trigger_pose()
		model = self.env.physics.model
		model.vis.global_.offwidth = max(
			int(model.vis.global_.offwidth), int(width)
		)
		model.vis.global_.offheight = max(
			int(model.vis.global_.offheight), int(height)
		)
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
	if domain == "manip":
		vision_task = f"{task}_vision"
		if vision_task not in manipulation.get_environments_by_tag("vision"):
			raise ValueError('Unknown task:', task)
	elif (domain, task) not in suite.ALL_TASKS:
		raise ValueError('Unknown task:', task)
	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
	env = _load_suite_env(domain, task, cfg)
	env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
	env = DMControlWrapper(env, domain, cfg)
	if cfg.obs == 'rgb':
		env = Pixels(env, cfg)
	# Manipulation uses a native 250-frame horizon. With action repeat 2 this
	# is 125 policy steps, rather than the suite default of 500 policy steps.
	max_episode_steps = 125 if domain == "manip" else 500
	env = Timeout(env, max_episode_steps=max_episode_steps)
	return env
