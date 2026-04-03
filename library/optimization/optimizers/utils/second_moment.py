import torch


# Modified Adafactor factorisation implementation by Ross Wightman
# https://github.com/huggingface/pytorch-image-models/pull/2320
@torch.no_grad()
def create_factored_dims(shape, factored: bool, min_dim_size_to_factor: int):
    r"""Whether to use a factored second moment estimator.

    This function returns a tuple with the two largest axes to reduce over.
    If all dimensions have size < min_dim_size_to_factor, return None.
    """
    if not factored or len(shape) < 2:
        return None
    if all(dim < min_dim_size_to_factor for dim in shape):
        return None
    sorted_dims = sorted((x, i) for i, x in enumerate(shape))
    return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])


# https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/23f752a3901686d270dfdcb9b29823541ad1c3c7/prodigyplus/core_optimiser.py#L389
@torch.no_grad()
def get_denom(second_moment: torch.Tensor, eps: float = 1e-16):
    if isinstance(second_moment, list):
        row_var, col_var, _, _, reduce_dc = second_moment

        row_col_mean = row_var.mean(dim=reduce_dc, keepdim=True).add_(eps)
        row_factor = row_var.div(row_col_mean).sqrt_()
        col_factor = col_var.sqrt()
        denom = row_factor * col_factor
    else:
        denom = second_moment.sqrt()

    return denom


# https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/23f752a3901686d270dfdcb9b29823541ad1c3c7/prodigyplus/core_optimiser.py#L411
@torch.no_grad()
def update_second_moment(second_moment: torch.Tensor, grad: torch.Tensor, beta2: float, adopt_first: bool = False) -> torch.Tensor:
    if isinstance(second_moment, list):
        row_var, col_var, dr, dc, _ = second_moment
        if adopt_first:
            row_var.copy_(grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr]))
            col_var.copy_(grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc]))
        else:
            row_var.lerp_(grad.norm(dim=dr, keepdim=True).square_().div_(grad.shape[dr]), weight=1 - beta2)
            col_var.lerp_(grad.norm(dim=dc, keepdim=True).square_().div_(grad.shape[dc]), weight=1 - beta2)
    else:
        if adopt_first:
            second_moment.addcmul_(grad, grad)
        else:
            second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    return second_moment
