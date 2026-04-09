from typing import Any

import torch


def adagc_global_clipping_calc(
    optimizer,
    step: int,
    warmup_steps: int = 0,
    lambda_abs: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calculate the warmup-era global AdaGC clip factor."""
    global_norm_device = "cpu"
    has_grad = False
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                has_grad = True
                global_norm_device = parameter.grad.device
                break
        if global_norm_device != "cpu":
            break

    if step <= warmup_steps and warmup_steps > 0 and has_grad:
        global_norm_sq_fp32 = torch.tensor(0.0, dtype=torch.float32, device=global_norm_device)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    global_norm_sq_fp32.add_(parameter.grad.float().pow(2).sum())

        if global_norm_sq_fp32 > 0:
            global_norm_fp32 = torch.sqrt(global_norm_sq_fp32)
            eps_fp32 = torch.tensor(eps, dtype=torch.float32, device=global_norm_device)
            global_clip_factor_fp32 = torch.tensor(lambda_abs, dtype=torch.float32, device=global_norm_device) / (
                global_norm_fp32 + eps_fp32
            )
            global_clip_factor_fp32 = torch.min(
                global_clip_factor_fp32,
                torch.tensor(1.0, device=global_norm_device, dtype=torch.float32),
            )
        else:
            global_clip_factor_fp32 = torch.tensor(1.0, device=global_norm_device, dtype=torch.float32)
    else:
        global_clip_factor_fp32 = torch.tensor(1.0, device=global_norm_device, dtype=torch.float32)

    return global_clip_factor_fp32


@torch.no_grad()
def apply_adagc_clipping_and_update_gamma(
    optimizer,
    grad: torch.Tensor,
    state: dict[str, Any],
    step: int,
    warmup_steps: int = 0,
    lambda_rel: float = 1.05,
    ema_beta: float = 0.98,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply AdaGC or warmup-era global clipping and update the per-parameter gamma state."""
    grad_fp32 = grad.float()
    device = grad_fp32.device

    if "adagc_gamma" not in state:
        state["adagc_gamma"] = torch.tensor(lambda_rel, dtype=torch.float32, device=device)
    gamma_fp32 = state["adagc_gamma"]

    if step <= warmup_steps and warmup_steps > 0:
        final_clip_factor_fp32 = optimizer._global_clip_factor_fp32.to(device)
    else:
        param_norm_fp32 = torch.linalg.norm(grad_fp32)
        prev_gamma_fp32 = gamma_fp32
        adaptive_threshold_fp32 = torch.tensor(lambda_rel, dtype=torch.float32, device=device) * (
            prev_gamma_fp32 + torch.tensor(eps, dtype=torch.float32, device=device)
        )
        ratio_fp32 = adaptive_threshold_fp32 / (param_norm_fp32 + torch.tensor(eps, dtype=torch.float32, device=device))
        ratio_fp32 = torch.nan_to_num(ratio_fp32, nan=1.0, posinf=1.0, neginf=1.0)
        final_clip_factor_fp32 = torch.min(
            torch.tensor(1.0, device=device, dtype=torch.float32),
            ratio_fp32,
        )

    clipped_grad_fp32 = grad_fp32
    clipped_grad_fp32.mul_(final_clip_factor_fp32)
    clipped_param_norm_fp32 = torch.linalg.norm(clipped_grad_fp32)
    gamma_fp32.mul_(torch.tensor(ema_beta, dtype=torch.float32, device=device)).add_(clipped_param_norm_fp32, alpha=1.0 - ema_beta)

    return clipped_grad_fp32
