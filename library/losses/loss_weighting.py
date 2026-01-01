import torch

from torch.types import Number
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def apply_snr_weight(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler, gamma: Number,
                     v_prediction=False):
    """
    Applies Signal-to-Noise Ratio (SNR) weighting to the loss.

    Args:
        loss (torch.Tensor): The original loss tensor.
        timesteps (torch.IntTensor): The timesteps corresponding to the loss.
        noise_scheduler (DDPMScheduler): The noise scheduler used.
        gamma (Number): The gamma parameter for Min-SNR weighting.
        v_prediction (bool, optional): Whether v-prediction is being used. Defaults to False.

    Returns:
        torch.Tensor: The weighted loss.
    """
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    loss = loss * snr_weight
    return loss


def scale_v_prediction_loss_like_noise_prediction(loss: torch.Tensor, timesteps: torch.IntTensor,
                                                  noise_scheduler: DDPMScheduler):
    """
    Scales the v-prediction loss to behave like noise prediction loss.

    Args:
        loss (torch.Tensor): The original loss tensor.
        timesteps (torch.IntTensor): The timesteps corresponding to the loss.
        noise_scheduler (DDPMScheduler): The noise scheduler used.

    Returns:
        torch.Tensor: The scaled loss.
    """
    scale = get_snr_scale(timesteps, noise_scheduler)
    loss = loss * scale
    return loss


def get_snr_scale(timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler):
    """
    Calculates the SNR scale factor based on timesteps.

    Args:
        timesteps (torch.IntTensor): The timesteps.
        noise_scheduler (DDPMScheduler): The noise scheduler.

    Returns:
        torch.Tensor: The calculated scale factor.
    """
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timesteps is 0, snr_t is inf, so limit it to 1000
    scale = snr_t / (snr_t + 1)
    # # show debug info
    # logger.info(f"timesteps: {timesteps}, snr_t: {snr_t}, scale: {scale}")
    return scale


def add_v_prediction_like_loss(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler,
                               v_pred_like_loss: torch.Tensor):
    """
    Adds a v-prediction-like loss component to the original loss.

    Args:
        loss (torch.Tensor): The original loss.
        timesteps (torch.IntTensor): The timesteps.
        noise_scheduler (DDPMScheduler): The noise scheduler.
        v_pred_like_loss (torch.Tensor): The v-prediction-like loss to add.

    Returns:
        torch.Tensor: The combined loss.
    """
    scale = get_snr_scale(timesteps, noise_scheduler)
    # logger.info(f"add v-prediction like loss: {v_pred_like_loss}, scale: {scale}, loss: {loss}, time: {timesteps}")
    loss = loss + loss / scale * v_pred_like_loss
    return loss


def apply_debiased_estimation(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler,
                              v_prediction=False):
    """
    Applies debiased estimation weighting to the loss.

    Args:
        loss (torch.Tensor): The original loss.
        timesteps (torch.IntTensor): The timesteps.
        noise_scheduler (DDPMScheduler): The noise scheduler.
        v_prediction (bool, optional): Whether v-prediction is used. Defaults to False.

    Returns:
        torch.Tensor: The weighted loss.
    """
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timesteps is 0, snr_t is inf, so limit it to 1000
    if v_prediction:
        weight = 1 / (snr_t + 1)
    else:
        weight = 1 / torch.sqrt(snr_t)
    loss = weight * loss
    return loss


def apply_masked_loss(loss, batch) -> torch.FloatTensor:
    """
    Applies a mask to the loss based on conditioning images or alpha masks in the batch.

    Args:
        loss (torch.Tensor): The original loss.
        batch (dict): The batch data containing potential masks ('conditioning_images' or 'alpha_masks').

    Returns:
        torch.FloatTensor: The masked loss.
    """
    if "conditioning_images" in batch:
        # conditioning image is -1 to 1. we need to convert it to 0 to 1
        mask_image = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)  # use R channel
        mask_image = mask_image / 2 + 0.5
        # print(f"conditioning_image: {mask_image.shape}")
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        # alpha mask is 0 to 1
        mask_image = batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)  # add channel dimension
        # print(f"mask_image: {mask_image.shape}, {mask_image.mean()}")
    else:
        return loss

    # resize to the same size as the loss
    mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
    loss = loss * mask_image
    return loss
