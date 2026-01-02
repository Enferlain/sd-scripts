import torch


from library.training.noise_utils import apply_noise_offset, pyramid_noise_like
from library.config.dataclasses.loss import RegularizationConfig
from library.config.dataclasses.timestep import TimestepConfig
from library.config.dataclasses.training import TrainingConfig


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
        latents: torch.FloatTensor,
        la_sampler=None,
        global_step=0,
        fixed_timesteps=None,
        is_train=True,
        min_timestep_override=None,
        max_timestep_override=None,
        output_dtype: torch.dtype = None,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    """
    Generate noise, noisy latents, and timesteps for diffusion training.

    Args:
        regularization_config (RegularizationConfig): Config for noise offset, multires noise, etc.
        timestep_config (TimestepConfig): Config for timesteps sampling parameters.
        training_config (TrainingConfig): Config for training settings.
        noise_scheduler: The diffusion noise scheduler.
        latents (torch.FloatTensor): Input latents tensor.
        la_sampler (Optional): Optional custom timesteps sampler.
        global_step (int, optional): Current training step (for adaptive sampling). Defaults to 0.
        fixed_timesteps (Optional): Optional fixed timesteps to use.
        is_train (bool, optional): Whether in training mode (affects noise augmentation). Defaults to True.
        min_timestep_override (Optional[int]): Override minimum timesteps.
        max_timestep_override (Optional[int]): Override maximum timesteps.
        output_dtype (Optional[torch.dtype]): If provided, cast noisy_latents to this dtype before returning.
            Useful because noise_scheduler.add_noise() may return float32
            even when inputs are float16/bfloat16 for numerical stability.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]: A tuple containing
        (noise, noisy_latents, timesteps).
    """
    # --- 1. Determine Timestep Range ---
    # This part handles the dynamic timesteps schedule!
    if min_timestep_override is not None:
        min_timestep = min_timestep_override
    else:
        min_timestep = 0 if timestep_config.min_timestep is None else timestep_config.min_timestep

    if max_timestep_override is not None:
        max_timestep = max_timestep_override
    else:
        max_timestep = noise_scheduler.config.num_train_timesteps if timestep_config.max_timestep is None else timestep_config.max_timestep

    # --- 2. Generate Base Noise ---
    noise = torch.randn_like(latents, device=latents.device)
    if regularization_config.noise_offset and is_train:
        noise_offset = torch.rand(1, device=latents.device) * regularization_config.noise_offset if regularization_config.noise_offset_random_strength else regularization_config.noise_offset
        noise = apply_noise_offset(latents, noise, noise_offset, regularization_config.adaptive_noise_scale)

    b_size = latents.shape[0]

    # --- 3. TIMESTEP SAMPLING ---
    if fixed_timesteps is not None:
        timesteps = fixed_timesteps
    elif is_train and hasattr(noise_scheduler, "edm2_laplace_weights"):
        timesteps = torch.multinomial(
            noise_scheduler.edm2_laplace_weights,
            num_samples=b_size,
            replacement=True
        ).to(dtype=torch.long, device=latents.device)
    elif is_train and hasattr(noise_scheduler, "laplace_weights"):
        timesteps = torch.multinomial(
            noise_scheduler.laplace_weights,
            num_samples=b_size,
            replacement=True
        ).to(dtype=torch.long, device=latents.device)
    elif is_train and timestep_config.timestep_sampling == "mix_adaptive":  # Todo related to custom timesteps samplers
        # The main script is now responsible for creating the sampler.
        # We just check that it exists and use it.
        if la_sampler is None:
            raise ValueError(
                "timestep_sampling is 'mix_adaptive' but la_sampler is not provided. "
                "Please ensure the sampler is created in your main training script."
            )

        # Sample discrete indices in [0, T_range)
        # Note: The sampler should be initialized with the full range of timesteps (e.g., 1000)
        t_local = la_sampler.sample(
            b_size,
            latents.device,
            global_step,
            training_config.max_train_steps,
            sigmoid_scale=timestep_config.sigmoid_scale,
            discrete_flow_shift=timestep_config.discrete_flow_shift,
        )

        # Map local [0, T_total) to the absolute training range [min_timestep, max_timestep)
        timesteps = t_local.clamp(min_timestep, max_timestep - 1).to(dtype=torch.long, device=latents.device)
    elif is_train and timestep_config.timestep_sampling != "uniform":
        shift = timestep_config.discrete_flow_shift
        logits_norm = torch.randn(b_size, device="cpu")
        logits_norm = logits_norm * timestep_config.sigmoid_scale
        timesteps = logits_norm.sigmoid()
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
        timesteps = min_timestep + (timesteps * (max_timestep - min_timestep)).to(dtype=torch.long,
                                                                                  device=latents.device)
    else:
        # Fallback to default (random) sampling
        timesteps = get_timesteps(min_timestep, max_timestep, b_size, latents.device)

    # --- 4. Advanced Noise Application (multires, ip_noise_gamma) ---
    if regularization_config.multires_noise_iterations and is_train:
        noise = pyramid_noise_like(
            noise, latents.device, regularization_config.multires_noise_iterations, regularization_config.multires_noise_discount
        )

    if regularization_config.ip_noise_gamma and is_train:
        strength = torch.rand(1, device=latents.device) * regularization_config.ip_noise_gamma if regularization_config.ip_noise_gamma_random_strength else regularization_config.ip_noise_gamma
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Cast to output dtype if specified (scheduler may return float32 for numerical stability)
    if output_dtype is not None:
        noisy_latents = noisy_latents.to(output_dtype)

    return noise, noisy_latents, timesteps
