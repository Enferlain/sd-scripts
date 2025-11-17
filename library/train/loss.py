import math
import kornia
import torch

from typing import List, Optional


class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        losses = len(self.loss_list)
        if losses == 0:
            return 0
        return self.loss_total / losses

class EMARecorder:
    """
    Calculates a bias-corrected Exponential Moving Average (EMA).

    This is the preferred method for smoothing noisy data in real-time,
    such as mini-batch losses during model training. It gives more weight
    to recent values, making it responsive to trends.
    """
    def __init__(self, smoothing: float = 0.1):
        """
        Initializes the EMA recorder.

        Args:
            smoothing (float): The smoothing factor, typically between 0 and 1.
                A smaller value (e.g., 0.01) results in a smoother, less responsive average.
                A larger value (e.g., 0.1) results in a noisier, more responsive average.
        """
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("Smoothing factor must be between 0 and 1.")

        self.smoothing = smoothing
        self.beta = 1 - self.smoothing  # The decay factor

        self.ema: float = 0.0
        self.num_updates: int = 0

    def add(self, value: float) -> None:
        """
        Updates the EMA with a new value.
        """
        self.num_updates += 1
        # Standard EMA update rule
        self.ema = self.beta * self.ema + self.smoothing * value

    @property
    def average(self) -> float:
        """
        Returns the bias-corrected moving average.

        Bias correction is important at the beginning of the series, as it
        corrects for the fact that the EMA is initialized at zero.
        """
        if self.num_updates == 0:
            return 0.0

        # Bias correction warms up the average faster
        # As num_updates -> infinity, the correction factor -> 1
        correction_factor = 1 - (self.beta ** self.num_updates)
        return self.ema / correction_factor


def get_huber_threshold_if_needed(args, timesteps: torch.Tensor, noise_scheduler) -> Optional[torch.Tensor]:
    if args.loss_type not in {"huber", "smooth_l1", "standard_pseudo_huber", "standard_huber", "standard_smooth_l1",
                              "soft_welsch", "scaled_quadratic", "smooth_l2_log"}:
        return None

    if args.huber_schedule == "constant":
        result = torch.tensor(args.huber_c * float(args.huber_scale), device=timesteps.device)
    elif args.huber_schedule == "exponential":
        alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
        result = torch.exp(-alpha * timesteps) * float(args.huber_scale)
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError("Huber schedule 'snr' is not supported with the current model.")
        alphas_cumprod = torch.index_select(noise_scheduler.alphas_cumprod, 0, timesteps)
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def soft_welsch_loss(predictions: torch.Tensor,
                     targets: torch.Tensor,
                     reduction: str = "mean",
                     scale: float = 1.0,
                     delta: float = 1.0):
    differences = predictions - targets
    loss = torch.arcsinh(4 * (scale * differences ** 2) / delta) * delta / 4
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


# Inspired by Grokking at the Edge of Numerical Stability (https://arxiv.org/abs/2501.04697)
def stable_mse_loss(predictions, targets, reduction="mean", eps=1e-37):
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    squared_differences = differences ** 2

    # Add eps to address underflows due to squaring
    squared_differences = squared_differences.add(eps)

    if reduction == "mean":
        loss = torch.mean(squared_differences)
    elif reduction == "sum":
        loss = torch.sum(squared_differences)
    elif reduction == "none":
        loss = squared_differences
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def stable_log_cosh_loss(predictions, targets, reduction='mean'):
    diff = predictions - targets
    # For x >= 0
    pos_mask = diff >= 0
    # Compute log(cosh(x)) for positive x
    logcosh_pos = diff + torch.nn.functional.softplus(-2 * diff) - math.log(2)
    # For x < 0
    logcosh_neg = -diff + torch.nn.functional.softplus(2 * diff) - math.log(2)
    # Combine results
    log_cosh = torch.where(pos_mask, logcosh_pos, logcosh_neg)

    if reduction == "mean":
        loss = torch.mean(log_cosh)
    elif reduction == "sum":
        loss = torch.sum(log_cosh)
    elif reduction == "none":
        loss = log_cosh
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def stable_msle_loss(predictions, targets, reduction='mean'):
    msle = torch.square(torch.log(targets + 1) - torch.log(predictions + 1))

    if reduction == "mean":
        loss = torch.mean(msle)
    elif reduction == "sum":
        loss = torch.sum(msle)
    elif reduction == "none":
        loss = msle
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def x_sigmoid_loss(predictions, targets, reduction="mean"):
    # Compute at float64
    differences = predictions - targets
    sigmoid_differences = 2 * differences * torch.sigmoid(differences) - differences
    if reduction == "mean":
        loss = torch.mean(sigmoid_differences)
    elif reduction == "sum":
        loss = torch.sum(sigmoid_differences)
    elif reduction == "none":
        loss = sigmoid_differences
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def stable_pseudo_huber_loss(predictions, targets, delta=1.0, reduction="mean", eps: float = 1e-37):
    """
    Compute the Pseudo-Huber loss between true values and predictions.

    Parameters:
    y_true : array_like
        The ground truth (correct) target values.
    y_pred : array_like
        The predicted target values.
    delta : float, default=1.0
        The parameter delta controls the transition point between the quadratic
        and linear regions of the loss function.

    Returns:
    loss : array_like
        The Pseudo-Huber loss values for each element.
    """
    differences = predictions.to(torch.float64) - targets.to(torch.float64)

    # Compute the loss
    loss = delta ** 2 * (torch.sqrt(1 + (differences / delta) ** 2 + eps) - 1)

    # Apply the specified reduction method
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def scaled_quadratic_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        delta: float = 1.0,
        reduction: str = 'mean',
        eps: float = 1e-37,
) -> torch.Tensor:
    r = predictions.to(torch.float64) - targets.to(torch.float64)
    loss = (r / delta) ** 2

    # Add eps to address underflows due to squaring
    loss = loss.add(eps)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def standard_deviation_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean',
        eps: float = 1e-30) -> torch.Tensor:
    """
    Calculate standard deviation loss between predicted and true values.

    Args:
        predictions (torch.Tensor): Predicted values
        targets (torch.Tensor): True values
        eps (float): Small constant to prevent numerical instability
                    when taking square root

    Returns:
        torch.Tensor: The standard deviation loss
    """
    n = predictions.size(0)
    squared_diff = (predictions - targets) ** 2
    mean_squared_diff = torch.sum(squared_diff) / n
    mean_squared_diff_sqrt = torch.sqrt(mean_squared_diff + eps)

    if reduction == "mean":
        loss = torch.mean(mean_squared_diff_sqrt)
    elif reduction == "sum":
        loss = torch.sum(mean_squared_diff_sqrt)
    elif reduction == "none":
        loss = mean_squared_diff_sqrt
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def smooth_l2_log_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        delta: float = 1.0,
        reduction: str = 'mean',
        eps=1e-37
) -> torch.Tensor:
    """
    Functional version of the smooth l2->log loss.

    Args:
        predictions: Predicted values of shape (*)
        targets: Target values of shape (*), same shape as predictions
        delta: Transition point between L2 and logarithmic behavior
        reduction: Reduction to apply to batch: 'none' | 'mean' | 'sum'

    Returns:
        Loss tensor of shape () if reduction is 'mean' or 'sum',
        or same shape as inputs if reduction is 'none'
    """
    r = predictions - targets
    delta_squared = delta ** 2
    delta_squared = delta_squared + eps
    loss = 0.5 * delta_squared * torch.log1p(r ** 2 / delta_squared)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def stable_smooth_l1_loss(predictions, targets, reduction: str = 'mean', beta=1.0, eps=1e-37):
    """
    Custom implementation of Smooth L1 Loss

    Args:
        predictions: Tensor of predictions
        targets: Tensor of target values
        beta: The threshold parameter that determines the switch point (default: 1.0)

    Returns:
        The computed Smooth L1 Loss
    """
    diff = torch.abs(predictions.to(torch.float64) - targets.to(torch.float64))
    condition = diff < beta

    # Where diff < beta, use quadratic form
    quadratic = 0.5 * diff.pow(2) / beta

    # Add eps to address underflows due to squaring
    loss = quadratic.add(eps)

    # Where diff >= beta, use linear form
    linear = diff - 0.5 * beta

    # Combine the two parts based on the condition
    loss = torch.where(condition, quadratic, linear)

    # Return loss
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def stable_huber_loss(predictions, targets, reduction: str = 'mean', delta=1.0, eps=1e-37):
    diff = torch.abs(predictions.to(torch.float64) - targets.to(torch.float64))
    abs_error = torch.abs(diff)

    # For small errors (≤ delta): use squared error (L2)
    quadratic = 0.5 * diff.pow(2) + eps

    # For large errors (> delta): use modified absolute error (L1)
    linear = delta * (abs_error - 0.5 * delta)

    # Combine both parts
    loss = torch.where(abs_error <= delta, quadratic, linear)

    # Return loss
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def stable_l1_loss(predictions, targets, reduction: str = 'mean', eps=1e-37):
    loss = torch.abs(predictions.to(torch.float64) - targets.to(torch.float64))

    loss = loss.add(eps)

    # Return loss
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def conditional_loss(
        model_pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str,
        reduction: str,
        huber_c: Optional[torch.Tensor] = None,
        eps: float = None,
        scale: float = 1.0,
):
    if eps is None or eps <= 0.0:
        eps = torch.finfo(torch.float32).tiny

    model_pred = model_pred.to(torch.float64)
    target = target.to(torch.float64)

    if huber_c is not None and huber_c.numel() > 1:
        huber_c_reshaped = huber_c.view(*huber_c.shape[:1], *([1] * (model_pred.dim() - 1)))
    else:
        huber_c_reshaped = huber_c

    if loss_type == "l2":
        loss = stable_mse_loss(model_pred, target, reduction="none", eps=eps)
    elif loss_type == "l1":
        loss = stable_l1_loss(model_pred, target, reduction="none", eps=eps)
    elif loss_type == "standard_pseudo_huber":
        loss = stable_pseudo_huber_loss(model_pred, target, delta=huber_c_reshaped, reduction="none", eps=eps)
    elif loss_type == "standard_huber":
        loss = stable_huber_loss(model_pred, target, reduction="none", delta=huber_c_reshaped, eps=eps)
    elif loss_type == "standard_smooth_l1":
        loss = stable_smooth_l1_loss(model_pred, target, reduction="none", beta=huber_c_reshaped, eps=eps)
    elif loss_type == "huber":
        loss = 2 * huber_c_reshaped * (
                torch.sqrt(((model_pred - target) ** 2 + eps) + huber_c_reshaped ** 2) - huber_c_reshaped)
    elif loss_type == "smooth_l1":
        loss = 2 * (torch.sqrt(((model_pred - target) ** 2 + eps) + huber_c_reshaped ** 2) - huber_c_reshaped)
    elif loss_type == "x_sigmoid":
        loss = x_sigmoid_loss(model_pred, target, reduction="none").add(eps)
    elif loss_type == "log_cosh":
        loss = stable_log_cosh_loss(model_pred, target, reduction="none").add(eps)
    elif loss_type == "squared_logarithmic":
        loss = stable_msle_loss(model_pred, target, reduction="none").add(eps)
    elif loss_type == "soft_welsch":
        loss = soft_welsch_loss(model_pred, target, reduction="none", delta=huber_c_reshaped, scale=scale)
    elif loss_type == "scaled_quadratic":
        loss = scaled_quadratic_loss(model_pred, target, reduction="none", delta=huber_c_reshaped, eps=eps)
    elif loss_type == "standard_deviation_loss":
        loss = standard_deviation_loss(model_pred, target, reduction="none", eps=eps)
    elif loss_type == "psnr_loss":
        loss = kornia.losses.psnr_loss(model_pred, target, 1.0).add(eps)
    elif loss_type == "geman_mcclure_loss":
        loss = kornia.losses.geman_mcclure_loss(model_pred, target).add(eps)
    elif loss_type == "smooth_l2_log":
        loss = smooth_l2_log_loss(model_pred, target, reduction="none", delta=huber_c_reshaped, eps=eps)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    return loss
