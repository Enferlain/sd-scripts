import logging

import torch

from library.optimization.optimizers.utils.types import CLIP_TYPE


@torch.no_grad()
def spam_grad_clipping(
    grad: torch.Tensor,
    second_moment: torch.Tensor,
    clip_threshold: float,
    clip_type: CLIP_TYPE = "element",
    spam_clip_eps: float = 1e-16,
) -> torch.Tensor:
    if spam_clip_eps is None or spam_clip_eps == 0:
        spam_clip_eps = torch.finfo(torch.float32).tiny

    if clip_type in {"unit", "element"}:
        second_momentum_threshold = second_moment.mul(clip_threshold).add(spam_clip_eps)
        second_momentum_threshold_sqrt = torch.sqrt(second_momentum_threshold)
        sign_grad = grad.sign()
        return torch.where(grad.square() > second_momentum_threshold, sign_grad * second_momentum_threshold_sqrt, grad)

    max_norm = torch.norm(torch.sqrt(second_moment * clip_threshold))
    grad_norm = torch.norm(grad)
    scale = torch.where(grad_norm > max_norm, max_norm / grad_norm, torch.ones_like(grad_norm))
    return grad * scale


def spam_grad_clipping_logging(
    grad: torch.Tensor,
    second_moment: torch.Tensor,
    clip_threshold: float,
    clip_type: CLIP_TYPE = "element",
    spam_clip_eps: float = 1e-16,
) -> None:
    if spam_clip_eps is None or spam_clip_eps == 0:
        spam_clip_eps = torch.finfo(torch.float32).tiny

    if clip_type in {"unit", "element"}:
        second_momentum_threshold = second_moment.mul(clip_threshold).add(spam_clip_eps)
        second_momentum_threshold_sqrt = torch.sqrt(second_momentum_threshold)
        scaling_mask = grad.square() > second_momentum_threshold
        total_elements = grad.numel()
        if scaling_mask.any():
            original_values = grad[scaling_mask].abs()
            scaled_values = second_momentum_threshold_sqrt[scaling_mask]
            scaling_ratios = scaled_values / (original_values.add(spam_clip_eps))
            logging.info(
                "Total elements %s. Unit-wise gradient clipping applied to %s elements. "
                "Original mean/max: %.6f / %.6f. Scaled mean/max: %.6f / %.6f. "
                "Scaling ratio mean/max: %.6f / %.6f",
                total_elements,
                scaling_mask.sum().item(),
                original_values.mean().item(),
                original_values.max().item(),
                scaled_values.mean().item(),
                scaled_values.max().item(),
                scaling_ratios.mean().item(),
                scaling_ratios.max().item(),
            )
        return

    max_norm = torch.norm(torch.sqrt(second_moment * clip_threshold))
    grad_norm = torch.norm(grad)
    scale = torch.where(grad_norm > max_norm, max_norm / grad_norm, torch.ones_like(grad_norm))
    if grad_norm > max_norm:
        logging.info(
            "Layer-wise gradient clipping applied. Gradient norm: %.4f, max norm: %.4f, scaling factor: %.4f",
            grad_norm.item(),
            max_norm.item(),
            scale.item(),
        )
