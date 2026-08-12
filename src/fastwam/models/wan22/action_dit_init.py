"""Shared Video-DiT-to-Action-DiT interpolation initialization helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def resize_tensor_to_shape(
    src: torch.Tensor, target_shape: tuple[int, ...]
) -> torch.Tensor:
    """Resize every mismatched axis using the original sequential 1-D rule."""
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                "Cannot reduce tensor rank for resize: "
                f"src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = interpolate_last_dim(out_perm, new_size)
        out = out_perm.reshape(*prefix_shape, new_size).permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            "Resize produced wrong shape: "
            f"src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def build_action_backbone_from_video(
    *,
    video_state: dict[str, torch.Tensor],
    action_state: dict[str, torch.Tensor],
    backbone_keys: set[str],
    apply_alpha_scaling: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Initialize an Action DiT backbone from a Video DiT state dictionary.

    This is the exact interpolation and alpha-scaling policy used by
    ``scripts/preprocess_action_dit_backbone.py``. Boundary-specific action
    encoder/head parameters are deliberately excluded by ``backbone_keys``.
    """
    backbone_state: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in video expert state dict.")
        src = video_state[key]
        target = action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = resize_tensor_to_shape(src, tuple(target.shape))
            if (
                apply_alpha_scaling
                and src.ndim >= 2
                and src.shape[-1] != target.shape[-1]
            ):
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state[key] = value.detach().to(
            dtype=target.dtype, device=target.device
        ).contiguous()

    return backbone_state, {
        "copied": copied,
        "interpolated": interpolated,
        "alpha_scaling": bool(apply_alpha_scaling),
        "interpolation": "sequential_1d_linear_align_corners_true",
    }
