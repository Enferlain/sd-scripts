import math

import torch


@torch.no_grad()
def filter_grad(grad: torch.Tensor, fft_alpha: float = 1.0) -> torch.Tensor:
    """Apply an n-dimensional Gaussian low-pass filter in the frequency domain."""
    grad_freq = torch.fft.fftn(grad, norm="ortho")
    freq_dims = [torch.fft.fftfreq(size, device=grad.device) for size in grad.shape]
    shifted_freq_dims = [torch.fft.ifftshift(freq) for freq in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing="ij"))
    max_radius = 0.5 * math.sqrt(len(grad.shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    filter_weights = torch.exp(-fft_alpha * (radius**2))
    filtered_grad_freq = grad_freq * filter_weights
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm="ortho")
    return modified_grad.real
