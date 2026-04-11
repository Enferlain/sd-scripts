import torch


@torch.no_grad()
def apply_update_strategies(update, grad, update_strategy, scale=1.0):
    """Apply cautious / grams style update post-processing with optional scaling.

    Args:
        update (torch.Tensor): The current update tensor to be modified.
        grad_normed (torch.Tensor): The normalized gradient.
        update_strategy (str): One of 'cautious', 'grams', 'both'.
        scale (float): Scaling factor for the Grams strategies.

    Returns:
        torch.Tensor: The modified update tensor.
    """
    if scale <= 0 or update_strategy not in {"cautious", "grams", "both"}:
        return update

    if update_strategy in {"cautious", "both"}:
        update_before_cautious = update
        mask = (update_before_cautious * grad > 0).to(grad.dtype)
        mask_mean = mask.mean().clamp_(min=1e-3)
        mask.div_(mask_mean)
        update_if_fully_cautious = update_before_cautious * mask
        update = update_if_fully_cautious if scale >= 1.0 else (1 - scale) * update_before_cautious + scale * update_if_fully_cautious

    if update_strategy in {"grams", "both"}:
        update_before_grams = update
        update_if_fully_grams = torch.sign(grad).mul_(update_before_grams.abs())
        update = update_if_fully_grams if scale >= 1.0 else (1 - scale) * update_before_grams + scale * update_if_fully_grams

    return update
