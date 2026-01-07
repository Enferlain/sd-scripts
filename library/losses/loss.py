import math
import kornia
import torch

from library.config.dataclasses.loss import HuberConfig, LossConfig


class LossRecorder:
    """
    Records and calculates the moving average of loss values during training.
    """

    def __init__(self):
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        """
        Adds a loss value to the recorder.

        Args:
            epoch (int): The current epoch number.
            step (int): The current step number within the epoch.
            loss (float): The loss value to record.
        """
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
        """
        Calculates the moving average of the recorded losses.

        Returns:
            float: The moving average of the losses.
        """
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

        Args:
            value (float): The new value to add to the EMA.
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
        correction_factor = 1 - (self.beta**self.num_updates)
        return self.ema / correction_factor


def get_huber_threshold_if_needed(loss_config: LossConfig, huber_config: HuberConfig, timesteps: torch.Tensor, noise_scheduler) -> torch.Tensor | None:
    """
    Calculates the Huber loss threshold based on the configured schedule.

    Args:
        huber_config: Configuration arguments containing loss settings.
        timesteps (torch.Tensor): Tensor of current timesteps.
        noise_scheduler: The noise scheduler used during training.

    Returns:
        Optional[torch.Tensor]: The calculated Huber threshold if needed, otherwise None.

    Raises:
        NotImplementedError: If the specified Huber schedule is not supported.
    """
    if loss_config.loss_type not in {
        "huber",
        "smooth_l1",
        "standard_pseudo_huber",
        "standard_huber",
        "standard_smooth_l1",
        "soft_welsch",
        "scaled_quadratic",
        "smooth_l2_log",
    }:
        return None

    if huber_config.huber_schedule == "constant":
        result = torch.tensor(huber_config.huber_c * float(huber_config.huber_scale), device=timesteps.device)
    elif huber_config.huber_schedule == "exponential":
        alpha = -math.log(huber_config.huber_c) / noise_scheduler.config.num_train_timesteps
        result = torch.exp(-alpha * timesteps) * float(huber_config.huber_scale)
    elif huber_config.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError("Huber schedule 'snr' is not supported with the current model.")
        alphas_cumprod = torch.index_select(noise_scheduler.alphas_cumprod, 0, timesteps)
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - huber_config.huber_c) / (1 + sigmas) ** 2 + huber_config.huber_c
        result = result.to(timesteps.device)
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {huber_config.huber_schedule}!")

    return result


def soft_welsch_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "mean", scale: float = 1.0, delta: float = 1.0):
    """
    Computes the Soft Welsch loss.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".
        scale (float, optional): Scaling factor for the loss. Defaults to 1.0.
        delta (float, optional): Parameter controlling the shape of the loss function. Defaults to 1.0.

    Returns:
        torch.Tensor: The computed loss.
    """
    differences = predictions - targets
    loss = torch.arcsinh(4 * (scale * differences**2) / delta) * delta / 4
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
    """
    Computes the Mean Squared Error (MSE) loss with numerical stability improvements.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".
        eps (float, optional): Small constant added to squared differences to prevent underflow. Defaults to 1e-37.

    Returns:
        torch.Tensor: The computed loss.
    """
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    squared_differences = differences**2

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


def stable_log_cosh_loss(predictions, targets, reduction="mean"):
    """
    Computes the Log-Cosh loss with numerical stability improvements.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".

    Returns:
        torch.Tensor: The computed loss.
    """
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


def stable_msle_loss(predictions, targets, reduction="mean"):
    """
    Computes the Mean Squared Logarithmic Error (MSLE) loss.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".

    Returns:
        torch.Tensor: The computed loss.
    """
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
    """
    Computes a custom X-Sigmoid loss.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".

    Returns:
        torch.Tensor: The computed loss.
    """
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
    Compute the Pseudo-Huber loss between true values and predictions with numerical stability.

    Args:
        predictions: The predicted target values.
        targets: The ground truth (correct) target values.
        delta (float, optional): The parameter delta controls the transition point between the quadratic
            and linear regions of the loss function. Defaults to 1.0.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".
        eps (float, optional): Small constant to prevent numerical instability. Defaults to 1e-37.

    Returns:
        torch.Tensor: The Pseudo-Huber loss.
    """
    differences = predictions.to(torch.float64) - targets.to(torch.float64)

    # Compute the loss
    loss = delta**2 * (torch.sqrt(1 + (differences / delta) ** 2 + eps) - 1)

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
    reduction: str = "mean",
    eps: float = 1e-37,
) -> torch.Tensor:
    """
    Computes a scaled quadratic loss.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.
        delta (float, optional): Scaling parameter. Defaults to 1.0.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".
        eps (float, optional): Small constant to prevent numerical instability. Defaults to 1e-37.

    Returns:
        torch.Tensor: The computed loss.
    """
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


def standard_deviation_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "mean", eps: float = 1e-30) -> torch.Tensor:
    """
    Calculate standard deviation loss between predicted and true values.

    Args:
        predictions (torch.Tensor): Predicted values
        targets (torch.Tensor): True values
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".
        eps (float): Small constant to prevent numerical instability
                    when taking square root. Defaults to 1e-30.

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
    predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0, reduction: str = "mean", eps: float = 1e-37
) -> torch.Tensor:
    """
    Functional version of the smooth l2->log loss.

    Args:
        predictions: Predicted values of shape (*)
        targets: Target values of shape (*), same shape as predictions
        delta: Transition point between L2 and logarithmic behavior
        reduction: Reduction to apply to batch: 'none' | 'mean' | 'sum'
        eps: Small constant to prevent numerical instability. Defaults to 1e-37.

    Returns:
        Loss tensor of shape () if reduction is 'mean' or 'sum',
        or same shape as inputs if reduction is 'none'
    """
    r = predictions - targets
    delta_squared = delta**2
    delta_squared = delta_squared + eps
    loss = 0.5 * delta_squared * torch.log1p(r**2 / delta_squared)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss


def stable_smooth_l1_loss(predictions, targets, reduction: str = "mean", beta=1.0, eps=1e-37):
    """
    Custom implementation of Smooth L1 Loss with numerical stability.

    Args:
        predictions: Tensor of predictions
        targets: Tensor of target values
        reduction: Reduction to apply to batch: 'none' | 'mean' | 'sum'
        beta: The threshold parameter that determines the switch point (default: 1.0)
        eps: Small constant to prevent numerical instability. Defaults to 1e-37.

    Returns:
        The computed Smooth L1 Loss
    """
    diff = torch.abs(predictions.to(torch.float64) - targets.to(torch.float64))
    condition = diff < beta

    # Where diff < beta, use quadratic form
    quadratic = 0.5 * diff.pow(2) / beta

    # Where diff >= beta, use linear form
    linear = diff - 0.5 * beta

    # Combine the two parts based on the condition
    # FIX: Add eps to the quadratic term specifically to prevent underflow during squaring.
    # We do this inside torch.where (or by updating 'quadratic' first) so it isn't lost.
    loss = torch.where(condition, quadratic.add(eps), linear)

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


def stable_huber_loss(predictions, targets, reduction: str = "mean", delta=1.0, eps=1e-37):
    """
    Computes the Huber loss with numerical stability improvements.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".
        delta (float, optional): The parameter delta controls the transition point between the quadratic and linear regions. Defaults to 1.0.
        eps (float, optional): Small constant to prevent numerical instability. Defaults to 1e-37.

    Returns:
        torch.Tensor: The computed Huber loss.
    """
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


def stable_l1_loss(predictions, targets, reduction: str = "mean", eps=1e-37):
    """
    Computes the L1 loss with numerical stability improvements.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Defaults to "mean".
        eps (float, optional): Small constant to prevent numerical instability. Defaults to 1e-37.

    Returns:
        torch.Tensor: The computed L1 loss.
    """
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
    huber_c: torch.Tensor | None = None,
    eps: float | None = None,
    scale: float = 1.0,
):
    """
    Computes the loss based on the specified loss type.

    Args:
        model_pred (torch.Tensor): Predicted values from the model.
        target (torch.Tensor): Ground truth values.
        loss_type (str): The type of loss to compute.
        reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
        huber_c (Optional[torch.Tensor], optional): Parameter for Huber-like losses. Defaults to None.
        eps (float, optional): Small constant to prevent numerical instability. Defaults to None.
        scale (float, optional): Scaling factor for certain losses. Defaults to 1.0.

    Returns:
        torch.Tensor: The computed loss.

    Raises:
        NotImplementedError: If the specified loss type is not supported.
    """
    # NOTE: huber_c is a Tensor (not float) when using timestep-dependent Huber scheduling.
    # The loss functions accept float but work correctly with broadcast-compatible Tensors.
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
        loss = stable_pseudo_huber_loss(model_pred, target, delta=huber_c_reshaped, reduction="none", eps=eps)  # type: ignore[arg-type]
    elif loss_type == "standard_huber":
        loss = stable_huber_loss(model_pred, target, reduction="none", delta=huber_c_reshaped, eps=eps)  # type: ignore[arg-type]
    elif loss_type == "standard_smooth_l1":
        loss = stable_smooth_l1_loss(model_pred, target, reduction="none", beta=huber_c_reshaped, eps=eps)  # type: ignore[arg-type]
    elif loss_type == "huber":
        loss = 2 * huber_c_reshaped * (torch.sqrt(((model_pred - target) ** 2 + eps) + huber_c_reshaped**2) - huber_c_reshaped)  # type: ignore[operator]
    elif loss_type == "smooth_l1":
        loss = 2 * (torch.sqrt(((model_pred - target) ** 2 + eps) + huber_c_reshaped**2) - huber_c_reshaped)  # type: ignore[operator]
    elif loss_type == "x_sigmoid":
        loss = x_sigmoid_loss(model_pred, target, reduction="none").add(eps)
    elif loss_type == "log_cosh":
        loss = stable_log_cosh_loss(model_pred, target, reduction="none").add(eps)
    elif loss_type == "squared_logarithmic":
        loss = stable_msle_loss(model_pred, target, reduction="none").add(eps)
    elif loss_type == "soft_welsch":
        loss = soft_welsch_loss(model_pred, target, reduction="none", delta=huber_c_reshaped, scale=scale)  # type: ignore[arg-type]
    elif loss_type == "scaled_quadratic":
        loss = scaled_quadratic_loss(model_pred, target, reduction="none", delta=huber_c_reshaped, eps=eps)  # type: ignore[arg-type]
    elif loss_type == "standard_deviation_loss":
        loss = standard_deviation_loss(model_pred, target, reduction="none", eps=eps)
    elif loss_type == "psnr_loss":
        loss = kornia.losses.psnr_loss(model_pred, target, 1.0).add(eps)
    elif loss_type == "geman_mcclure_loss":
        loss = kornia.losses.geman_mcclure_loss(model_pred, target).add(eps)
    elif loss_type == "smooth_l2_log":
        loss = smooth_l2_log_loss(model_pred, target, reduction="none", delta=huber_c_reshaped, eps=eps)  # type: ignore[arg-type]
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    return loss
