from typing import Literal

import torch


NORM_TYPE = Literal["unit", "global", "layer"]


def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    r"""Get norm of unit."""
    keep_dim = True
    dim: int | tuple[int, ...] | None = None

    x_len = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)


def agc(
    p: torch.Tensor,
    grad: torch.Tensor,
    agc_clip_val: float,
    agc_eps: float = 1e-3,
    eps: float = 1e-16,
    norm_type: NORM_TYPE = "layer",
) -> torch.Tensor:
    r"""Clip gradient values in excess of the norm.

    Clip updates to be at most clipping * parameter_norm.
    """
    if norm_type in {"global", "layer"}:
        p_norm = torch.norm(p).clamp_(min=agc_eps)
        g_norm = torch.norm(grad)
        max_norm = (p_norm * agc_clip_val).clamp(min=eps)
        clip_coef = min(1, max_norm / g_norm.clamp(min=eps))
        return grad * clip_coef

    if norm_type == "unit":
        p_norm = unit_norm(p).clamp_(min=agc_eps)
        g_norm = unit_norm(grad)
        max_norm = (p_norm * agc_clip_val).clamp(min=eps)
        clipped_grad = grad * (max_norm / g_norm.clamp_(min=eps))
        return torch.where(g_norm > max_norm, clipped_grad, grad)

    raise ValueError(f"{norm_type!r} is not a supported value for norm_type.")


def adaptive_eps(grad: torch.Tensor, group: dict, rms_grad: torch.Tensor | None = None) -> torch.Tensor:
    if "eps_t" not in group or group["eps_t"].device != group["params"][0].device:
        group["eps_t"] = torch.tensor(group["eps"], device=group["params"][0].device)
    if group["eps_floor"] is not None and group["eps_floor"] < group["eps"]:
        if "eps2_t" not in group or group["eps2_t"].device != group["params"][0].device:
            group["eps2_t"] = torch.tensor(group["eps2"], device=group["params"][0].device)
        if "eps_floor_t" not in group or group["eps_floor_t"].device != group["params"][0].device:
            group["eps_floor_t"] = torch.tensor(group["eps_floor"], device=group["params"][0].device)

        if rms_grad is None:
            rms_grad = torch.sqrt(torch.mean(grad.pow(2)))
        val_to_bound = group["eps2_t"] * rms_grad
        return torch.clamp(val_to_bound, min=group["eps_floor"], max=group["eps"])

    return group["eps_t"]
