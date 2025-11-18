import torch

from typing import Tuple

from library.training.noise_utils import apply_noise_offset, pyramid_noise_like


def get_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps


def get_noise_noisy_latents_and_timesteps(
        args, noise_scheduler, latents: torch.FloatTensor, fixed_timesteps=None, is_train=True,
        min_timestep_override=None, max_timestep_override=None
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    """
    todo
    """
    # --- 1. Determine Timestep Range ---
    # This part handles the dynamic timestep schedule!
    if min_timestep_override is not None:
        min_timestep = min_timestep_override
    else:
        min_timestep = 0 if args.min_timestep is None else args.min_timestep

    if max_timestep_override is not None:
        max_timestep = max_timestep_override
    else:
        max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep

    # --- 2. Generate Base Noise ---
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset and is_train:
        noise_offset = torch.rand(1, device=latents.device) * args.noise_offset if args.noise_offset_random_strength else args.noise_offset
        noise = apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)

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
    elif is_train and args.timestep_sampling == "mix_adaptive":
        # The main script is now responsible for creating the sampler.
        # We just check that it exists and use it.
        if not hasattr(args, "la_sampler") or args.la_sampler is None:
            raise ValueError(
                "timestep_sampling is 'mix_adaptive' but args.la_sampler is not initialized. "
                "Please ensure the sampler is created in your main training script."
            )

        # Sample discrete indices in [0, T_range)
        # Note: The sampler should be initialized with the full range of timesteps (e.g., 1000)
        t_local = args.la_sampler.sample(
            b_size,
            latents.device,
            getattr(args, "global_step", 0),
            getattr(args, "max_train_steps", 1000),
            sigmoid_scale=getattr(args, "sigmoid_scale", 1.0),
            discrete_flow_shift=getattr(args, "discrete_flow_shift", 1.0),
        )

        # Map local [0, T_total) to the absolute training range [min_timestep, max_timestep)
        timesteps = t_local.clamp(min_timestep, max_timestep - 1).to(dtype=torch.long, device=latents.device)
    elif is_train and args.timestep_sampling != "uniform":
        shift = args.discrete_flow_shift
        logits_norm = torch.randn(b_size, device="cpu")
        logits_norm = logits_norm * args.sigmoid_scale
        timesteps = logits_norm.sigmoid()
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
        timesteps = min_timestep + (timesteps * (max_timestep - min_timestep)).to(dtype=torch.long,
                                                                                  device=latents.device)
    else:
        # Fallback to default (random) sampling
        timesteps = get_timesteps(min_timestep, max_timestep, b_size, latents.device)

    # --- 4. Advanced Noise Application (multires, ip_noise_gamma) ---
    if args.multires_noise_iterations and is_train:
        noise = pyramid_noise_like(
            noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount
        )

    if args.ip_noise_gamma and is_train:
        strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma if args.ip_noise_gamma_random_strength else args.ip_noise_gamma
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Important! The old script had a .cpu() call here. It was a workaround.
    # Modern diffusers handles device placement better, so we can often omit this.
    # If you see device errors, we can add it back!
    # noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cpu()

    return noise, noisy_latents, timesteps
