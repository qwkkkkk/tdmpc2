"""
Backdoor utilities for TD-MPC2 stage-2 injection.

- Pixel-patch trigger injection for rgb observations.
- Helper to disable ShiftAug in the encoder (required when trigger pattern
  must land on a known receptive-field location).
- Frozen-submodule bookkeeping.
"""

import copy

import torch
import torch.nn as nn

from common.layers import ShiftAug


def apply_trigger_pixel(obs, size, value):
    """
    Paste a solid-colour square at the top-left corner of a pixel obs.

    `obs` has shape (..., C, H, W) with dtype either uint8 [0,255] or float.
    The patch is written in-place into a cloned tensor, spanning every
    channel (so a 3-frame-stack rgb obs sees the trigger on every frame).
    """
    result = obs.clone()
    result[..., :, :size, :size] = value
    return result


def apply_trigger_invis(obs, delta, eps):
    """
    Apply learned additive trigger δ to obs (MIRAGE invis mode).

    obs:   (..., C, H, W) float32, values in [0, 255]
    delta: (C, H, W) tensor (nn.Parameter), same unit range as obs
    eps:   L∞ budget in pixel units (e.g. 8.0 means 8/255 in [0,1] space)

    delta is first projected into [-eps, eps] then added to obs; the result
    is clamped to [0, 255].  Broadcasting handles batched obs automatically.
    Gradient flows through delta when called in a training context.
    """
    return (obs + delta.clamp(-eps, eps)).clamp(0.0, 255.0)


def disable_shift_aug(model):
    """
    Replace the ShiftAug layer inside the rgb encoder with nn.Identity().

    Stage-2 training must see a stationary trigger at a fixed receptive-field
    location; the random shift in the stage-1 encoder would scramble that.
    """
    encoder = getattr(model, "_encoder", None)
    if encoder is None:
        return
    for key, enc in encoder.items():
        if key != "rgb":
            continue
        if len(enc) > 0 and isinstance(enc[0], ShiftAug):
            enc[0] = nn.Identity()


def freeze_policy_and_q(model):
    """
    Freeze μ_φ, Q_φ, and target Q_φ in-place.

    Stage-2 does not touch these; their params are taken out of any
    optimizer and their grad flags cleared.
    """
    for p in model._pi.parameters():
        p.requires_grad_(False)
    for p in model._Qs.parameters():
        p.requires_grad_(False)
    for p in model._target_Qs.parameters():
        p.requires_grad_(False)


def build_trainable_params(model, include_termination=False):
    """
    Return the list of parameters updated in stage-2: E_θ, M_θ, R_θ.

    Optionally include the termination head.
    """
    params = (
        list(model._encoder.parameters())
        + list(model._dynamics.parameters())
        + list(model._reward.parameters())
    )
    if include_termination and model._termination is not None:
        params += list(model._termination.parameters())
    return params


def make_reference_model(model):
    """
    Deepcopy the live WorldModel into an independent frozen reference copy.

    The reference is used as θ_0 for the L_f^π fidelity and L_s selectivity
    losses. All parameters have requires_grad=False and the copy is set to
    eval() so any dropout is disabled.
    """
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref
