# library/timesteps/samplers/log_snr_sampler.py
import torch
from typing import Any


class LogSNRUniformSampler:
    """
    Sampler that draws timesteps uniformly in log-SNR space.

    Instead of sampling timesteps uniformly (which favors high-noise regions in terms of
    visual changes), this sampler ensures uniform coverage of the log-signal-to-noise ratio.
    """

    def __init__(self, noise_scheduler: Any, num_train_timesteps: int):
        """
        Initialize the LogSNRUniformSampler.

        Args:
            noise_scheduler: Diffusers noise scheduler (e.g. DDPMScheduler).
            num_train_timesteps (int): Total number of training timesteps.
        """
        print(f"LogSNRUniformSampler initialized with: num_train_timesteps={num_train_timesteps}")
        T = int(num_train_timesteps)
        with torch.no_grad():
            # Precompute SNR(t) = alpha^2 / (1 - alpha^2)
            a2 = noise_scheduler.alphas_cumprod.float().clamp(min=1e-12, max=1.0 - 1e-12)  # [T]
            snr = a2 / (1.0 - a2)
            self.log_snr = torch.log(snr.clamp(min=1e-20))  # [T]
        self.T = T

    @torch.no_grad()
    def sample(
        self,
        bsz: int,
        device: torch.device,
        global_step: int,
        max_steps: int,
        sigmoid_scale: float = 1.0,
        discrete_flow_shift: float = 0.9,
    ) -> torch.Tensor:
        """
        Sample timesteps for a batch.

        Args:
            bsz (int): Batch size.
            device (torch.device): Device to put the sampled timesteps on.
            global_step (int): Current global step (unused).
            max_steps (int): Max training steps (unused).
            sigmoid_scale (float): Scale for sigmoid (unused).
            discrete_flow_shift (float): Shift for discrete flow (unused).

        Returns:
            torch.Tensor: A tensor of sampled timesteps with shape (bsz,).
        """
        # Uniform in log-SNR range, then nearest neighbor on indices
        log_min = self.log_snr.min()
        log_max = self.log_snr.max()
        u = torch.rand(bsz, device=device) * (log_max - log_min) + log_min
        # Compute |log_snr[i] - u_j| and argmin per row efficiently
        # Use broadcasting via (bsz, T) memory-cautious approach: chunk if needed
        diffs = (self.log_snr.to(device)[None, :] - u[:, None]).abs()
        t_local = diffs.argmin(dim=1).long()  # [bsz], in [0, T)
        return t_local
