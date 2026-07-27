#!/usr/bin/env bash

NVIDIA_EGL_OVERLAY_ROOT=${NVIDIA_EGL_OVERLAY_ROOT:-/home/pth/kai/nvidia-535.161.08-overlay}
NVIDIA_EGL_VENDOR_JSON=${NVIDIA_EGL_VENDOR_JSON:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/10_nvidia_535.json}

if [[ ! -f "${NVIDIA_EGL_OVERLAY_ROOT}/lib/libEGL_nvidia.so.535.161.08" ]]; then
    echo "[error] NVIDIA EGL overlay is missing from ${NVIDIA_EGL_OVERLAY_ROOT}" >&2
    return 1 2>/dev/null || exit 1
fi

export LD_LIBRARY_PATH="${NVIDIA_EGL_OVERLAY_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
export MUJOCO_GL=egl
