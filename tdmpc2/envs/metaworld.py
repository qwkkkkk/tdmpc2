from collections import deque
import os
import tempfile
from xml.etree import ElementTree as ET

import gymnasium as gym
import numpy as np
import torch

from envs.wrappers.timeout import Timeout


_TASK_TRIGGER_DEFAULTS = {
	"reach": {"pos": (0.5, 0.35, 0.070), "size": 0.025},
	"door-open": {"pos": (0.5, 0.35, 0.070), "size": 0.025},
	"drawer-open": {"pos": (0.5, 0.35, 0.070), "size": 0.025},
	"drawer-close": {"pos": (0.5, 0.35, 0.070), "size": 0.025},
	"window-close": {"pos": (0.5, 0.35, 0.070), "size": 0.025},
	"button-press": {"pos": (0.5, 0.35, 0.070), "size": 0.025},
	"_default": {"pos": (0.5, 0.35, 0.070), "size": 0.025},
}


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg, task_name):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.task_name = task_name
		self.camera_name = cfg.get("metaworld_camera", "corner2")
		self._render_size = int(cfg.get("metaworld_image_size", 64))
		self._phys_trigger = bool(cfg.get("phys_trigger", False)) or cfg.get("trigger_type", "") == "physical"
		self._phys_trigger_observable = bool(cfg.get("phys_trigger_observable", False))
		self._trigger_active = False
		self._trigger_qpos_adr = -1
		self._trigger_qvel_adr = -1
		self._trigger_pos = None
		self._trigger_hidden_pos = None
		self._last_obs = None

		self._set_camera()
		self.env._freeze_rand_vec = False
		if self._phys_trigger:
			default = _TASK_TRIGGER_DEFAULTS.get(task_name, _TASK_TRIGGER_DEFAULTS["_default"])
			pos = cfg.get("metaworld_phys_trigger_pos", None)
			size = cfg.get("metaworld_phys_trigger_size", None)
			pos = tuple(pos) if pos is not None else default["pos"]
			size = float(size) if size is not None else float(default["size"])
			table_top_z = self._infer_table_top_z()
			pos = (pos[0], pos[1], table_top_z + size + 0.035)
			self._inject_trigger_geom(pos, size)
			self._set_camera()

		base_shape = self.env.observation_space.shape
		if self._phys_trigger_observable:
			base_shape = (base_shape[0] + 4,)
		self.observation_space = gym.spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=base_shape,
			dtype=np.float32,
		)

	def _set_camera(self):
		if self.camera_name == "corner2" and hasattr(self.env, "model"):
			try:
				self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
			except Exception:
				pass

	def _infer_table_top_z(self):
		try:
			import mujoco
			model = self.env.model
			data = self.env.data
			mujoco.mj_forward(model, data)
			tops = []
			for gid in range(model.ngeom):
				name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
				if name and "table" in name.lower():
					tops.append(float(data.geom_xpos[gid][2]) + float(model.geom_size[gid][2]))
			if tops:
				return float(max(tops))
		except Exception:
			pass
		return 0.165

	def _inject_trigger_geom(self, pos, size):
		import mujoco

		tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xml")
		os.close(tmp_fd)
		try:
			mujoco.mj_saveLastXML(tmp_path, self.env.model)
			tree = ET.parse(tmp_path)
		finally:
			try:
				os.unlink(tmp_path)
			except OSError:
				pass

		root = tree.getroot()
		worldbody = root.find("worldbody")
		if worldbody is None:
			raise RuntimeError("Cannot locate <worldbody> in MetaWorld MuJoCo XML")

		self._trigger_pos = np.asarray(pos, dtype=np.float64)
		self._trigger_hidden_pos = np.asarray((pos[0], pos[1], -10.0), dtype=np.float64)

		body = ET.SubElement(worldbody, "body", {
			"name": "bd_trigger_body",
			"pos": "0 0 0",
		})
		ET.SubElement(body, "freejoint", {"name": "bd_trigger_freejoint"})
		ET.SubElement(body, "geom", {
			"name": "bd_trigger_geom",
			"type": "sphere",
			"size": f"{float(size):.5f}",
			"rgba": "1 0 1 1",
			"contype": "0",
			"conaffinity": "0",
		})
		modified_xml = ET.tostring(root, encoding="unicode")

		try:
			new_model = mujoco.MjModel.from_xml_string(modified_xml)
		except Exception as e_str:
			ref_dir = self._metaworld_xml_dir()
			tmp_fd2, tmp_path2 = tempfile.mkstemp(suffix=".xml", dir=ref_dir)
			try:
				with os.fdopen(tmp_fd2, "w") as f:
					f.write(modified_xml)
				new_model = mujoco.MjModel.from_xml_path(tmp_path2)
			except Exception as e_file:
				raise RuntimeError(
					"Physical trigger XML injection failed.\n"
					f"  from_xml_string : {e_str}\n"
					f"  from_xml_path   : {e_file}"
				) from e_file
			finally:
				try:
					os.unlink(tmp_path2)
				except OSError:
					pass

		new_data = mujoco.MjData(new_model)
		self.env.model = new_model
		self.env.data = new_data
		self._refresh_mujoco_renderer(close_viewer=True)

		joint_id = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_JOINT, "bd_trigger_freejoint")
		if joint_id < 0:
			raise RuntimeError("bd_trigger_freejoint not found after model reload")
		self._trigger_qpos_adr = int(new_model.jnt_qposadr[joint_id])
		self._trigger_qvel_adr = int(new_model.jnt_dofadr[joint_id])
		self._mj_renderer = mujoco.Renderer(new_model, self._render_size, self._render_size)
		self._mj_render_size = (self._render_size, self._render_size)
		self._mj_cam_id = mujoco.mj_name2id(
			new_model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name or ""
		)
		self._set_trigger_qpos(self.env.data, self._trigger_hidden_pos)
		mujoco.mj_forward(self.env.model, self.env.data)

	def _metaworld_xml_dir(self):
		try:
			import metaworld.envs as mw_envs
			mw_envs_dir = os.path.dirname(mw_envs.__file__)
		except Exception:
			mw_envs_dir = None
		for attr in ("model_name", "_MODEL_XML", "MODEL_XML", "model_xml"):
			val = getattr(self.env, attr, None)
			if callable(val):
				try:
					val = val()
				except Exception:
					val = None
			if not isinstance(val, str):
				continue
			val = os.path.expanduser(val)
			if os.path.isfile(val):
				return os.path.dirname(os.path.abspath(val))
			if mw_envs_dir:
				abs_path = os.path.join(mw_envs_dir, val)
				if os.path.isfile(abs_path):
					return os.path.dirname(abs_path)
		return self._metaworld_asset_dir()

	@staticmethod
	def _metaworld_asset_dir():
		try:
			import metaworld.envs as mw_envs
			root = os.path.dirname(mw_envs.__file__)
			for sub in ("assets_v3/sawyer_xyz", "assets_v2/sawyer_xyz", "assets_v3", "assets_v2", "."):
				cand = os.path.join(root, sub)
				if os.path.isdir(cand):
					return cand
		except Exception:
			pass
		return tempfile.gettempdir()

	def _refresh_mujoco_renderer(self, close_viewer=False):
		renderer = getattr(self.env, "mujoco_renderer", None)
		if renderer is None:
			return
		for model_attr in ("model", "_model"):
			if hasattr(renderer, model_attr):
				setattr(renderer, model_attr, self.env.model)
		for data_attr in ("data", "_data"):
			if hasattr(renderer, data_attr):
				setattr(renderer, data_attr, self.env.data)
		if close_viewer:
			viewers = getattr(renderer, "_viewers", None)
			if isinstance(viewers, dict):
				for viewer in list(viewers.values()):
					try:
						viewer.close()
					except Exception:
						pass
				viewers.clear()
			viewer = getattr(renderer, "viewer", None)
			if viewer is not None:
				try:
					viewer.close()
				except Exception:
					pass
				try:
					renderer.viewer = None
				except Exception:
					pass

	def _set_trigger_qpos(self, data, pos):
		adr = self._trigger_qpos_adr
		if adr < 0:
			return
		data.qpos[adr:adr+3] = np.asarray(pos, dtype=data.qpos.dtype)
		data.qpos[adr+3:adr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=data.qpos.dtype)
		if self._trigger_qvel_adr >= 0:
			data.qvel[self._trigger_qvel_adr:self._trigger_qvel_adr+6] = 0

	def _restore_trigger_pose(self):
		if self._trigger_qpos_adr < 0:
			return
		target = self._trigger_pos if self._trigger_active else self._trigger_hidden_pos
		self._set_trigger_qpos(self.env.data, target)

	def _augment_state(self, obs):
		obs = np.asarray(obs, dtype=np.float32)
		if not self._phys_trigger_observable:
			return obs
		if self._trigger_pos is None:
			extra = np.zeros(4, dtype=np.float32)
		else:
			pos = self._trigger_pos if self._trigger_active else self._trigger_hidden_pos
			extra = np.asarray([float(self._trigger_active), pos[0], pos[1], pos[2]], dtype=np.float32)
		return np.concatenate([obs, extra], dtype=np.float32)

	def _current_obs(self):
		if hasattr(self.env, "_get_obs"):
			try:
				obs = self.env._get_obs()
				self._last_obs = obs
				return self._augment_state(obs)
			except Exception:
				pass
		if self._last_obs is None:
			return None
		return self._augment_state(self._last_obs)

	def set_trigger(self, active):
		if self._trigger_qpos_adr < 0:
			return self._current_obs()
		import mujoco
		self._trigger_active = bool(active)
		self._restore_trigger_pose()
		mujoco.mj_forward(self.env.model, self.env.data)
		return self._current_obs()

	def render_trigger_obs(self, active=True, **kwargs):
		prev = self._trigger_active
		obs = self.set_trigger(active)
		self.set_trigger(prev)
		return obs

	@property
	def trigger_active(self):
		return self._trigger_active

	def reset(self, **kwargs):
		out = super().reset(**kwargs)
		obs = out[0] if isinstance(out, tuple) else out
		self._last_obs = np.asarray(obs, dtype=np.float32)
		self._restore_trigger_pose()
		if self._trigger_qpos_adr >= 0:
			try:
				import mujoco
				mujoco.mj_forward(self.env.model, self.env.data)
			except Exception:
				pass
		self.env.step(np.zeros(self.env.action_space.shape))
		return self._augment_state(self._last_obs)

	def step(self, action):
		reward = 0
		info = {}
		for _ in range(2):
			out = self.env.step(action.copy())
			if len(out) == 5:
				obs, r, terminated, truncated, info = out
				done = terminated or truncated
			else:
				obs, r, done, info = out
			reward += r
			if done:
				break
		self._last_obs = np.asarray(obs, dtype=np.float32)
		self._restore_trigger_pose()
		return self._augment_state(self._last_obs), reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, width=None, height=None, *args, **kwargs):
		self._restore_trigger_pose()
		width = int(width or self._render_size)
		height = int(height or self._render_size)
		if self._phys_trigger and hasattr(self, "_mj_renderer"):
			import mujoco
			if getattr(self, "_mj_render_size", None) != (height, width):
				self._mj_renderer = mujoco.Renderer(self.env.model, height, width)
				self._mj_render_size = (height, width)
			mujoco.mj_forward(self.env.model, self.env.data)
			if getattr(self, "_mj_cam_id", -1) >= 0:
				self._mj_renderer.update_scene(self.env.data, camera=self._mj_cam_id)
			else:
				self._mj_renderer.update_scene(self.env.data)
			return self._mj_renderer.render().copy()
		renderer = getattr(self.env, "mujoco_renderer", None)
		if renderer is not None:
			renderer.width = width
			renderer.height = height
		return self.env.render().copy()


class Pixels(gym.Wrapper):
	def __init__(self, env, num_frames=3, size=64):
		super().__init__(env)
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


def _canonical_task(task):
	if task.startswith("mw-"):
		return task
	if task.startswith("metaworld_"):
		return "mw-" + task.split("_", 1)[1]
	if task.startswith("metaworld-"):
		return "mw-" + task.split("-", 1)[1]
	return task


def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	cfg.task = _canonical_task(cfg.task)
	task_name = cfg.task.split("-", 1)[-1]
	if not cfg.task.startswith('mw-'):
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
	import metaworld
	env_id = task_name + "-v3"
	try:
		mt1 = metaworld.MT1(env_id, seed=cfg.seed)
		env = mt1.train_classes[env_id](
			render_mode="rgb_array",
			camera_name=cfg.get("metaworld_camera", "corner2"),
		)
		env.set_task(mt1.train_tasks[0])
	except Exception as exc:
		raise ValueError(f"Unknown or unavailable MetaWorld task: {cfg.task}") from exc
	render_size = int(cfg.get("metaworld_image_size", 64))
	env.mujoco_renderer.width = render_size
	env.mujoco_renderer.height = render_size
	env = MetaWorldWrapper(env, cfg, task_name)
	if cfg.obs == 'rgb':
		env = Pixels(env, size=render_size)
	env = Timeout(env, max_episode_steps=100)
	return env
