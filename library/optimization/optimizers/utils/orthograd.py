import torch


# https://github.com/kozistr/pytorch_optimizer/blob/6397d56279ad80b26c4bba7fb4b04852b517fdeb/pytorch_optimizer/optimizer/shampoo_utils.py#L533
@torch.no_grad()
def zero_power_via_newton_schulz_6(grad: torch.Tensor) -> torch.Tensor:
    r"""Compute the zeroth power / orthogonalization of G."""
    grad_shape = grad.shape
    grad = grad.view(grad.size(0), -1)

    abc_list = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]

    x = grad.float()
    if grad.size(0) > grad.size(1):
        x = x.T

    x = x.div(x.norm().add(1e-16))
    for a, b, c in abc_list:
        a_matrix = x @ x.T
        b_matrix = b * a_matrix + c * a_matrix @ a_matrix
        x = a * x + b_matrix @ x

    if grad.size(0) > grad.size(1):
        x = x.T

    x = torch.einsum("ij,ij->", grad.type_as(x), x).clamp(-1.0, 1.0) * x
    return x.view(grad_shape)


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def zero_power_via_newton_schulz_6_compile(grad: torch.Tensor) -> torch.Tensor:
    return zero_power_via_newton_schulz_6(grad)


@torch.no_grad()
def bias_rms(grad: torch.Tensor) -> torch.Tensor:
    rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True))
    return grad.div(rms_value.add_(1e-16))


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def bias_rms_compile(grad: torch.Tensor) -> torch.Tensor:
    return bias_rms(grad)


@torch.no_grad()
def paper_orthograd(param, grad, alpha: float = 1.0, eps: float | torch.Tensor = 1e-20):
    """Apply orthogonal projection to a single parameter's gradient."""
    if param.ndim == 0 or param.numel() <= 1:
        return

    weight = param.view(-1)
    grad_flat = grad.view(-1)
    weight_norm_sq = torch.dot(weight, weight)
    if weight_norm_sq <= eps:
        return

    proj_coeff = torch.dot(weight, grad_flat) / weight_norm_sq
    grad_parallel = proj_coeff * weight
    grad_orth = grad_flat - alpha * grad_parallel
    grad_orth_scaled = grad_orth.mul_(grad.norm(2) / (grad_orth.norm(2) + eps))
    grad.copy_(grad_orth_scaled.view_as(grad))


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def paper_orthograd_compile(param, grad, alpha: float = 1.0, eps: float | torch.Tensor = 1e-20):
    return paper_orthograd(param, grad, alpha, eps)


# Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
@torch.no_grad()
def orthograd_atan(param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Apply the donor atan-based orthogonal gradient projection and return the projected tensor."""
    grad_shape = grad.shape
    weight = param.view(-1)
    grad_flat = grad.view(-1)

    proj = torch.dot(weight, grad_flat).atan2_(torch.dot(weight, weight)).mul_(1.27323954474)
    grad_orth = grad_flat.to(dtype=torch.float32, copy=True).sub_(weight, alpha=proj)
    grad_orth_scaled = grad_orth.mul_(grad_flat.norm(2).div_(grad_orth.norm(2).clamp_(min=1e-6)))

    return grad_orth_scaled.view(grad_shape)
