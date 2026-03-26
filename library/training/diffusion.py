import torch

from library.config.dataclasses.data import CachingConfig
from library.config.dataclasses.loss import RegularizationConfig
from library.config.dataclasses.timestep import TimestepConfig
from library.config.dataclasses.training import TrainingConfig
from library.timesteps.runtime import build_timestep_runtime
from library.training.noise_utils import apply_noise_offset, pyramid_noise_like


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


def get_noise_noisy_latents_and_timesteps(
    regularization_config: RegularizationConfig,
    timestep_config: TimestepConfig,
    training_config: TrainingConfig,
    noise_scheduler,
    latents: torch.Tensor,
    timestep_runtime=None,
    la_sampler=None,
    global_step=0,
    fixed_timesteps=None,
    is_train=True,
    min_timestep_override=None,
    max_timestep_override=None,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate noise, noisy latents, and timesteps for diffusion training.

    Args:
        regularization_config (RegularizationConfig): Config for noise offset, multires noise, etc.
        timestep_config (TimestepConfig): Config for timesteps sampling parameters.
        training_config (TrainingConfig): Config for training settings.
        noise_scheduler: The diffusion noise scheduler.
        latents (torch.Tensor): Input latents tensor.
        timestep_runtime (Optional): Active timestep runtime that owns timestep sampling behavior.
        la_sampler (Optional): Legacy sampler input used to build a compatibility runtime for older callers.
        global_step (int, optional): Current training step (for adaptive sampling). Defaults to 0.
        fixed_timesteps (Optional): Optional fixed timesteps to use.
        is_train (bool, optional): Whether in training mode (affects noise augmentation). Defaults to True.
        min_timestep_override (Optional[int]): Override minimum timesteps when constructing a compatibility runtime.
        max_timestep_override (Optional[int]): Override maximum timesteps when constructing a compatibility runtime.
        output_dtype (Optional[torch.dtype]): If provided, cast noisy_latents to this dtype before returning.
            Useful because noise_scheduler.add_noise() may return float32
            even when inputs are float16/bfloat16 for numerical stability.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
        (noise, noisy_latents, timesteps).
    """
    # --- 1. Generate Base Noise ---
    noise = torch.randn_like(latents, device=latents.device)
    if regularization_config.noise_offset and is_train:
        noise_offset = (
            torch.rand(1, device=latents.device) * regularization_config.noise_offset
            if regularization_config.noise_offset_random_strength
            else regularization_config.noise_offset
        )
        noise = apply_noise_offset(latents, noise, noise_offset, regularization_config.adaptive_noise_scale)

    b_size = latents.shape[0]

    # --- 2. TIMESTEP SAMPLING ---
    runtime = timestep_runtime
    if runtime is None:
        runtime = build_timestep_runtime(
            timestep_config,
            noise_scheduler,
            sampler_override=la_sampler,
            min_timestep_override=min_timestep_override,
            max_timestep_override=max_timestep_override,
            global_step=global_step,
        )

    timesteps = runtime.sample_timesteps(
        timestep_config=timestep_config,
        training_config=training_config,
        noise_scheduler=noise_scheduler,
        batch_size=b_size,
        device=latents.device,
        global_step=global_step,
        fixed_timesteps=fixed_timesteps,
        is_train=is_train,
    )

    # --- 3. Advanced Noise Application (multires, ip_noise_gamma) ---
    if regularization_config.multires_noise_iterations and is_train:
        noise = pyramid_noise_like(
            noise, latents.device, regularization_config.multires_noise_iterations, regularization_config.multires_noise_discount
        )

    if regularization_config.ip_noise_gamma and is_train:
        strength = (
            torch.rand(1, device=latents.device) * regularization_config.ip_noise_gamma
            if regularization_config.ip_noise_gamma_random_strength
            else regularization_config.ip_noise_gamma
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Cast to output dtype if specified (scheduler may return float32 for numerical stability)
    if output_dtype is not None:
        noisy_latents = noisy_latents.to(output_dtype)

    return noise, noisy_latents, timesteps
