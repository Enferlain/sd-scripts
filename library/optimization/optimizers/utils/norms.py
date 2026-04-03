import torch


def get_global_gradient_norm(param_groups) -> torch.Tensor:
    """Return the summed squared gradient norm across optimizer parameter groups."""
    device = None
    for group in param_groups:
        for p in group["params"]:
            if p.grad is not None:
                device = p.grad.device
                break
        if device is not None:
            break

    if device is None:
        return torch.zeros((), dtype=torch.float32)

    global_grad_norm = torch.zeros((), dtype=torch.float32, device=device)
    for group in param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                grad = grad.coalesce().values()

            global_grad_norm.add_(grad.to(torch.float32).pow(2).sum())

    return global_grad_norm
