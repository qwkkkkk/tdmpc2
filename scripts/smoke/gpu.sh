#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX=${ENV_PREFIX:-/home/pth/kai/envs/tdmpc2_lab509}
REPO_ROOT=${REPO_ROOT:-/home/pth/kai/tdmpc2}
GPU_ID=${GPU_ID:-0}
MUJOCO_GL_BACKEND=${MUJOCO_GL_BACKEND:-egl}
PYTHON="${ENV_PREFIX}/bin/python"

test -x "${PYTHON}"
test -f "${REPO_ROOT}/tdmpc2/train.py"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL_BACKEND}"
if [[ "${MUJOCO_GL}" == "egl" ]]; then
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}"
else
    unset MUJOCO_EGL_DEVICE_ID
fi

echo "[smoke] Python dependencies and CUDA"
"${PYTHON}" - <<'PY'
import torch
import gymnasium
import hydra
import omegaconf
import tensordict
import torchrl
import kornia

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
device_name = torch.cuda.get_device_name(0)
x = torch.ones((64, 64), device="cuda")
assert float((x @ x).mean().cpu()) == 64.0

print(f"[smoke] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[smoke] gpu={device_name}")
print("[smoke] dependency and CUDA checks passed")
PY

if [[ "${MUJOCO_GL}" == "disable" ]]; then
    echo "[smoke] MuJoCo rendering skipped (MUJOCO_GL=disable)"
else
    echo "[smoke] MuJoCo rendering with ${MUJOCO_GL}"
    "${PYTHON}" - <<'PY'
import numpy as np
import mujoco
from dm_control import suite

env = suite.load("cartpole", "balance")
time_step = env.reset()
time_step = env.step(np.zeros(env.action_spec().shape, dtype=np.float32))
frame = env.physics.render(height=64, width=64, camera_id=0)
assert frame.shape == (64, 64, 3)

print(f"[smoke] mujoco={mujoco.__version__} render={frame.shape}")
print("[smoke] MuJoCo EGL check passed")
PY
fi

echo "[smoke] TD-MPC2 initialization and environment-step loop"
cd "${REPO_ROOT}/tdmpc2"
"${PYTHON}" train.py \
    task=cartpole-balance \
    obs=state \
    steps=2 \
    model_size=1 \
    exp_name=gpu_smoke \
    work_dir="${REPO_ROOT}/tdmpc2/logs/dmc/cartpole-balance/smoke/gpu" \
    data_dir="${REPO_ROOT}/data" \
    eval_freq=1000000 \
    eval_episodes=0 \
    enable_wandb=false \
    wandb_project=disabled \
    wandb_entity=disabled \
    save_video=false \
    save_agent=false \
    save_latent_traces=false \
    compile=false \
    num_samples=32 \
    num_elites=4 \
    num_pi_trajs=0

echo "[smoke] all checks passed"
