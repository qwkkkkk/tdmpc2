from collections import deque

import gymnasium as gym
import numpy as np
import torch

from envs.wrappers.timeout import Timeout


MYOSUITE_TASKS = {
	'myo-reach': 'myoHandReachFixed-v0',
	'myo-reach-hard': 'myoHandReachRandom-v0',
	'myo-pose': 'myoHandPoseFixed-v0',
	'myo-pose-hard': 'myoHandPoseRandom-v0',
	'myo-obj-hold': 'myoHandObjHoldFixed-v0',
	'myo-obj-hold-hard': 'myoHandObjHoldRandom-v0',
	'myo-key-turn': 'myoHandKeyTurnFixed-v0',
	'myo-key-turn-hard': 'myoHandKeyTurnRandom-v0',
	'myo-pen-twirl': 'myoHandPenTwirlFixed-v0',
	'myo-pen-twirl-hard': 'myoHandPenTwirlRandom-v0',
}


class MyoSuiteWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.camera_id = cfg.get('myosuite_camera', 'hand_side_inter')
		self._renderers = {}
		self._phys_trigger = (
			bool(cfg.get('phys_trigger', False))
			or cfg.get('trigger_type', '') == 'physical'
		)
		self._trigger_active = False
		self._trigger_body_id = -1
		self._trigger_pos = np.asarray(
			cfg.get('myosuite_phys_trigger_pos', [0.00, -0.30, 1.30]),
			dtype=np.float64,
		)
		self._trigger_hidden_pos = np.asarray(
			[0.0, 0.0, -10.0], dtype=np.float64
		)
		if self._phys_trigger:
			self._inject_trigger_geom()

	def _inject_trigger_geom(self):
		import mujoco

		base = self.env.unwrapped
		spec = base.mj_spec.copy()
		body = spec.worldbody.add_body(
			name='bd_trigger_body',
			pos=self._trigger_hidden_pos.tolist(),
		)
		size = float(self.cfg.get('myosuite_phys_trigger_size', 0.025))
		rgba = self.cfg.get('phys_trigger_rgba', [1.0, 0.0, 1.0, 1.0])
		body.add_geom(
			name='bd_trigger_geom',
			type=mujoco.mjtGeom.mjGEOM_SPHERE,
			size=[size, 0.0, 0.0],
			rgba=[float(value) for value in rgba],
			contype=0,
			conaffinity=0,
			mass=0.001,
		)
		model = spec.compile()
		data = mujoco.MjData(model)
		base.mj_spec = spec
		base.mj_model = model
		base.mj_data = data
		base.obsd_mj_model = model
		base.obsd_mj_data = data
		base.robot.mj_model = model
		base.robot.mj_data = data
		self._trigger_body_id = mujoco.mj_name2id(
			model, mujoco.mjtObj.mjOBJ_BODY, 'bd_trigger_body'
		)
		if self._trigger_body_id < 0:
			raise RuntimeError('MyoSuite physical trigger body was not injected.')
		self._restore_trigger_pose()

	def _restore_trigger_pose(self):
		if self._trigger_body_id < 0:
			return
		import mujoco

		base = self.env.unwrapped
		pos = (
			self._trigger_pos
			if self._trigger_active
			else self._trigger_hidden_pos
		)
		base.mj_model.body_pos[self._trigger_body_id] = pos
		mujoco.mj_forward(base.mj_model, base.mj_data)

	def set_trigger(self, active):
		self._trigger_active = bool(active)
		self._restore_trigger_pose()

	@property
	def trigger_active(self):
		return self._trigger_active

	def reset(self, **kwargs):
		result = self.env.reset(**kwargs)
		self._restore_trigger_pose()
		return result[0] if isinstance(result, tuple) else result

	def step(self, action):
		self._restore_trigger_pose()
		result = self.env.step(action.copy())
		self._restore_trigger_pose()
		if len(result) == 5:
			obs, reward, _, _, info = result
		else:
			obs, reward, _, info = result
		info['success'] = info.get(
			'success', info.get('solved', info.get('is_success', 0.))
		)
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, width=384, height=384, camera_id=None):
		base = self.env.unwrapped
		self._restore_trigger_pose()
		if hasattr(base, 'mj_model') and hasattr(base, 'mj_data'):
			import mujoco

			base.mj_model.vis.global_.offwidth = max(
				int(base.mj_model.vis.global_.offwidth), int(width)
			)
			base.mj_model.vis.global_.offheight = max(
				int(base.mj_model.vis.global_.offheight), int(height)
			)
			renderer_key = (int(height), int(width))
			renderer = self._renderers.get(renderer_key)
			if renderer is None:
				renderer = mujoco.Renderer(
					base.mj_model, height=height, width=width
				)
				self._renderers[renderer_key] = renderer
			camera = self.camera_id if camera_id is None else camera_id
			if isinstance(camera, str):
				camera = mujoco.mj_name2id(
					base.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera
				)
				camera = None if camera < 0 else camera
			renderer.update_scene(base.mj_data, camera=camera)
			return renderer.render().copy()

		sim = getattr(base, 'sim', getattr(self.env, 'sim', None))
		if sim is not None and hasattr(sim, 'renderer'):
			camera = self.camera_id if camera_id is None else camera_id
			return sim.renderer.render_offscreen(
				width=width, height=height, camera_id=camera
			).copy()
		raise RuntimeError('Could not find a MyoSuite offscreen render path.')

	def close(self):
		for renderer in self._renderers.values():
			renderer.close()
		self._renderers.clear()
		return self.env.close()


class Pixels(gym.Wrapper):
	def __init__(self, env, num_frames=3, size=64):
		super().__init__(env)
		self.env = env
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*3, size, size), dtype=np.uint8
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
		if hasattr(self.env, 'set_trigger'):
			self.env.set_trigger(active)
			return self._get_obs(is_reset=False)

	def render_trigger_obs(self, active=True, fill_stack=True):
		if not hasattr(self.env, 'set_trigger'):
			return None
		prev = getattr(self.env, 'trigger_active', False)
		self.env.set_trigger(active)
		frame = self.env.render(
			width=self._size, height=self._size
		).transpose(2, 0, 1)
		self.env.set_trigger(prev)
		if fill_stack or len(self._frames) == 0:
			frames = [frame for _ in range(self._frames.maxlen)]
		else:
			frames = list(self._frames)
			frames[-1] = frame
		return torch.from_numpy(np.concatenate(frames))

	@property
	def trigger_active(self):
		return getattr(self.env, 'trigger_active', False)


def make_env(cfg):
	"""
	Make Myosuite environment.
	"""
	if not cfg.task in MYOSUITE_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs in {'state', 'rgb'}, (
		'This task only supports state and rgb observations.'
	)
	import myosuite
	from myosuite.utils import gym as gym_utils
	env = gym_utils.make(MYOSUITE_TASKS[cfg.task])
	env = MyoSuiteWrapper(env, cfg)
	if cfg.obs == 'rgb':
		env = Pixels(env, size=int(cfg.get('myosuite_image_size', 64)))
	env = Timeout(env, max_episode_steps=100)
	return env
