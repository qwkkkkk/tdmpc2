import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import sys
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored


def _step_bar(t, total, ep_reward, bar_width=30):
	"""Print an in-place step-level progress bar."""
	filled = int(bar_width * t / max(total, 1))
	bar = '█' * filled + '░' * (bar_width - filled)
	sys.stdout.write(f'\r    step [{bar}] {t:>4}/{total}  R_so_far={ep_reward:>8.1f}')
	sys.stdout.flush()


def _episode_bar(i, total, results):
	"""Print a per-episode summary line after each episode completes."""
	r_str = f'{results[-1][0]:>8.1f}' if results else '       ?'
	s_str = f'{results[-1][1]:.2f}' if results else '   ?'
	mean_r = np.mean([r for r, _ in results]) if results else float('nan')
	mean_s = np.mean([s for _, s in results]) if results else float('nan')
	sys.stdout.write(
		f'\r  ep {i+1:>3}/{total}  R={r_str}  S={s_str}'
		f'  |  mean_R={mean_r:>8.1f}  mean_S={mean_s:.2f}\n'
	)
	sys.stdout.flush()

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	
	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_results = []  # list of (ep_reward, ep_success)
		ep_len = cfg.episode_length  # max steps per episode (e.g. 500 for dmcontrol)
		print(colored(f'  episodes={cfg.eval_episodes}  max_steps/ep={ep_len}', 'cyan'))
		for i in range(cfg.eval_episodes):
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done:
				action = agent.act(obs, t0=t==0, task=task_idx)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.save_video:
					frames.append(env.render())
				# step-level progress bar (updates in-place)
				_step_bar(t, ep_len, ep_reward)
			# episode done — clear step bar and print episode summary
			sys.stdout.write('\r' + ' ' * 70 + '\r')
			ep_results.append((ep_reward, info['success']))
			_episode_bar(i, cfg.eval_episodes, ep_results)
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		ep_rewards = np.mean([r for r, _ in ep_results])
		ep_successes = np.mean([s for _, s in ep_results])
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}'
			f'\tR: {ep_rewards:.01f}  '
			f'\tS: {ep_successes:.02f}', 'yellow'))
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
