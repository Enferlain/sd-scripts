import random

import torch

# https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
def pyramid_noise_like(noise, device, iterations=6, discount=0.4) -> torch.FloatTensor:
    """
    Generates multi-resolution (pyramid) noise.

    See: https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2

    Args:
        noise (torch.FloatTensor): The base noise tensor.
        device (torch.device): The device for the generated noise.
        iterations (int, optional): Number of resolution layers. Defaults to 6.
        discount (float, optional): Discount factor for higher resolution noise. Defaults to 0.4.

    Returns:
        torch.FloatTensor: The noise tensor augmented with multi-resolution noise, scaled to unit variance.
    """
    b, c, w, h = noise.shape  # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode="bilinear").to(device)
    for i in range(iterations):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        wn, hn = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(b, c, wn, hn).to(device)) * discount**i
        if wn == 1 or hn == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


# https://www.crosslabs.org//blog/diffusion-with-offset-noise
def apply_noise_offset(latents, noise, noise_offset, adaptive_noise_scale) -> torch.FloatTensor:
    """
    Applies offset noise to the input noise tensor.

    See: https://www.crosslabs.org//blog/diffusion-with-offset-noise

    Args:
        latents (torch.FloatTensor): The latents tensor, used for shape and adaptive scaling.
        noise (torch.FloatTensor): The input noise tensor to be modified.
        noise_offset (float or None): The magnitude of the noise offset.
        adaptive_noise_scale (float or None): If provided, scales the offset based on latent mean.

    Returns:
        torch.FloatTensor: The modified noise tensor with offset applied.
    """
    if noise_offset is None:
        return noise
    if adaptive_noise_scale is not None:
        # latent shape: (batch_size, channels, height, width)
        # abs mean value for each channel
        latent_mean = torch.abs(latents.mean(dim=(2, 3), keepdim=True))

        # multiply adaptive noise scale to the mean value and add it to the noise offset
        noise_offset = noise_offset + adaptive_noise_scale * latent_mean
        noise_offset = torch.clamp(noise_offset, 0.0, None)  # in case of adaptive noise scale is negative

    noise = noise + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
    return noise


r"""
##########################################
# Perlin Noise
def rand_perlin_2d(device, shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)),
            dim=-1,
        )
        % 1
    )
    angles = 2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )
    dot = lambda grad, shift: (
        torch.stack((grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), dim=-1)
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return 1.414 * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(device, shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape, device=device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(device, shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def perlin_noise(noise, device, octaves):
    _, c, w, h = noise.shape
    perlin = lambda: rand_perlin_2d_octaves(device, (w, h), (4, 4), octaves)
    noise_perlin = []
    for _ in range(c):
        noise_perlin.append(perlin())
    noise_perlin = torch.stack(noise_perlin).unsqueeze(0)   # (1, c, w, h)
    noise += noise_perlin # broadcast for each batch
    return noise / noise.std()  # Scaled back to roughly unit variance
"""
