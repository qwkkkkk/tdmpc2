#!/usr/bin/env bash

NVIDIA_EGL_OVERLAY_ROOT=${NVIDIA_EGL_OVERLAY_ROOT:-/home/pth/kai/nvidia-535.161.08-overlay}
NVIDIA_EGL_VENDOR_JSON=${NVIDIA_EGL_VENDOR_JSON:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/10_nvidia_535.json}
NVIDIA_VULKAN_ICD_JSON=${NVIDIA_VULKAN_ICD_JSON:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../docker" && pwd)/nvidia_icd.json}
DM_CONTROL_OVERLAY_ROOT=${DM_CONTROL_OVERLAY_ROOT:-/home/pth/kai/python-overlays/dm_control-1.0.28}

if [[ ! -f "${NVIDIA_EGL_OVERLAY_ROOT}/lib/libEGL_nvidia.so.535.161.08" ]]; then
    echo "[error] NVIDIA EGL overlay is missing from ${NVIDIA_EGL_OVERLAY_ROOT}" >&2
    return 1 2>/dev/null || exit 1
fi
if [[ ! -d "${DM_CONTROL_OVERLAY_ROOT}/dm_control" ]]; then
    echo "[error] dm_control overlay is missing from ${DM_CONTROL_OVERLAY_ROOT}" >&2
    return 1 2>/dev/null || exit 1
fi

OVERLAY_LIBRARY_PATH="${NVIDIA_EGL_OVERLAY_ROOT}/lib"
if [[ "${DOMAIN:-}" == "maniskill" || "${DOMAIN:-}" == "maniskill3" ]]; then
	MANISKILL_PYTHON_PREFIX=${MANISKILL_VULKAN_PREFIX:-${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}}
    for library in \
        libGLX_nvidia.so.535.161.08 \
        libnvidia-glvkspirv.so.535.161.08 \
        libnvidia-rtcore.so.535.161.08 \
        libnvidia-vulkan-producer.so.535.161.08; do
        if [[ ! -f "${NVIDIA_EGL_OVERLAY_ROOT}/lib/${library}" ]]; then
            echo "[error] ManiSkill2 Vulkan overlay is missing ${library}" >&2
            return 1 2>/dev/null || exit 1
        fi
    done
    if [[ ! -f "${MANISKILL_PYTHON_PREFIX}/lib/libvulkan.so.1" ]]; then
        echo "[error] Vulkan loader is missing from ${MANISKILL_PYTHON_PREFIX}/lib" >&2
        return 1 2>/dev/null || exit 1
    fi
    if [[ ! -f "${NVIDIA_VULKAN_ICD_JSON}" ]]; then
        echo "[error] NVIDIA Vulkan ICD manifest is missing: ${NVIDIA_VULKAN_ICD_JSON}" >&2
        return 1 2>/dev/null || exit 1
    fi
    OVERLAY_LIBRARY_PATH="${OVERLAY_LIBRARY_PATH}:${MANISKILL_PYTHON_PREFIX}/lib"
    export VK_ICD_FILENAMES="${NVIDIA_VULKAN_ICD_JSON}"
fi

export LD_LIBRARY_PATH="${OVERLAY_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${DM_CONTROL_OVERLAY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
export MUJOCO_GL=egl
