import torch

from library.config.dataclasses.data import CachingConfig


def encode_images_to_latents(vae, images: torch.Tensor) -> torch.Tensor:
    """
    Encode images to latents using the provided VAE.

    Args:
        vae: VAE model instance.
        images: Batch of images to encode.

    Returns:
        Encoded latents tensor.
    """
    return vae.encode(images).latent_dist.sample()


def shift_scale_latents(latents: torch.Tensor, vae_latent_scale: float) -> torch.Tensor:
    """
    Apply a model-family latent scaling factor.

    Args:
        latents: Latents tensor to scale.
        vae_latent_scale: Model-family VAE latent scale factor.

    Returns:
        Scaled latents tensor.
    """
    return latents * vae_latent_scale


def prepare_latents(
    batch: dict,
    caching_config: CachingConfig,
    device: torch.device,
    vae,
    vae_dtype: torch.dtype,
    vae_latent_scale: float,
    log_fn=None,
    encode_images_to_latents_fn=encode_images_to_latents,
    shift_scale_latents_fn=shift_scale_latents,
) -> torch.Tensor:
    """
    Prepare latents from cached batch entries or by live VAE encoding.

    Args:
        batch: Batch data containing either cached latents or images.
        caching_config: Data caching configuration.
        device: Device to move latents and images to.
        vae: VAE model for encoding images.
        vae_dtype: Data type for VAE operations.
        vae_latent_scale: Model-family VAE latent scale factor.
        log_fn: Optional logging function used for NaN warnings.
        encode_images_to_latents_fn: Helper used to encode images to latents.
        shift_scale_latents_fn: Helper used to apply latent scaling.

    Returns:
        Prepared and scaled latents tensor.
    """
    if "latents" in batch and batch["latents"] is not None:
        latents = batch["latents"].to(device)
    else:
        if caching_config.vae_batch_size is None or len(batch["images"]) <= caching_config.vae_batch_size:
            latents = encode_images_to_latents_fn(vae, batch["images"].to(device, dtype=vae_dtype))
        else:
            chunks = [
                batch["images"][i : i + caching_config.vae_batch_size]
                for i in range(0, len(batch["images"]), caching_config.vae_batch_size)
            ]
            list_latents = []
            for chunk in chunks:
                with torch.no_grad():
                    chunk_latents = encode_images_to_latents_fn(vae, chunk.to(device, dtype=vae_dtype))
                    list_latents.append(chunk_latents)
            latents = torch.cat(list_latents, dim=0)

        if torch.any(torch.isnan(latents)):
            if log_fn is not None:
                log_fn("NaN found in latents, replacing with zeros")
            latents = torch.nan_to_num(latents, 0, out=latents)

        latents = shift_scale_latents_fn(latents, vae_latent_scale)

    return latents


def get_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device) -> torch.Tensor:
    """
    Generates a batch of timesteps for diffusion training.

    Args:
        min_timestep (int): The minimum timesteps index (inclusive).
        max_timestep (int): The maximum timesteps index (exclusive).
        b_size (int): The batch size (number of timesteps to generate).
        device (torch.device): The device to place the resulting tensor on.

    Returns:
        torch.Tensor: A tensor of shape (b_size,) containing the generated timesteps.
    """
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps
