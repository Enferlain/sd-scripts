# tools/log_snr_sampler.py
import torch


class LogSNRUniformSampler:
    def __init__(self, noise_scheduler, num_train_timesteps: int):
        print(f"LogSNRUniformSampler initialized with: num_train_timesteps={num_train_timesteps}")
        T = int(num_train_timesteps)
        with torch.no_grad():
            # Precompute SNR(t) = alpha^2 / (1 - alpha^2)
            a2 = noise_scheduler.alphas_cumprod.float().clamp(min=1e-12, max=1.0 - 1e-12)  # [T]
            snr = a2 / (1.0 - a2)
            self.log_snr = torch.log(snr.clamp(min=1e-20))  # [T]
        self.T = T

    @torch.no_grad()
    def sample(self, bsz: int, device, global_step: int, max_steps: int,
               sigmoid_scale: float = 1.0, discrete_flow_shift: float = 0.9):
        # Uniform in log-SNR range, then nearest neighbor on indices
        log_min = self.log_snr.min()
        log_max = self.log_snr.max()
        u = torch.rand(bsz, device=device) * (log_max - log_min) + log_min
        # Compute |log_snr[i] - u_j| and argmin per row efficiently
        # Use broadcasting via (bsz, T) memory-cautious approach: chunk if needed
        diffs = (self.log_snr.to(device)[None, :] - u[:, None]).abs()
        t_local = diffs.argmin(dim=1).long()  # [bsz], in [0, T)
        return t_local
