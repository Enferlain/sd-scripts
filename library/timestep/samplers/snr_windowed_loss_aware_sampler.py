# snr_windowed_loss_aware_sampler.py
import math, torch

from typing import Optional


class SNRWindowedLossAwareSampler:
    """
    Loss-aware sampler restricted to a sliding window in log-SNR.
    """
    def __init__(
        self,
        noise_scheduler,
        num_bins: int = 64,
        ema_beta: float = 0.95,
        temperature: float = 0.3,
        min_prob: float = 1e-3,
        entropy_floor_ratio: float = 0.8,
        uniform_mix_when_low_entropy: float = 0.1,
        # Window in log-SNR:
        center_mu: float = 0.0,            # start center in log-SNR
        half_width: float = 1.0,           # initial half-width in log-SNR
        widen_to: float = 2.5,             # target half-width
        total_widen_steps: int = 2000,     # steps to widen
        # Optional cap on top timesteps early in training:
        cap_max_t: Optional[int] = None,
        # new hyperparams:
        cap_target_t: Optional[int] = 950,       # default: T-1
        cap_ema_beta: float = 0.9,
        cap_saturation_thresh: float = 0.20,      # fraction of samples hitting boundary
        cap_step_min: int = 5,                    # min increment when moving cap
        cap_step_max_frac: float = 0.10,          # at most this fraction of remaining range
    ):
        self.T = int(noise_scheduler.config.num_train_timesteps)
        a2 = noise_scheduler.alphas_cumprod.float().clamp(1e-12, 1. - 1e-12)
        snr = a2 / (1. - a2)
        log_snr = torch.log(snr.clamp(min=1e-20))
        self.log_snr_original = log_snr.contiguous()
        self.log_snr_sorted, self.sort_indices = torch.sort(log_snr)
        edges = torch.linspace(0, self.T - 1, steps=num_bins + 1, device=log_snr.device)
        edges = edges.round().long().clamp(0, self.T - 1)
        self.bin_left, self.bin_right = edges[:-1], edges[1:].clamp(min=1)
        centers_idx = ((self.bin_left + self.bin_right) // 2).long()
        self.bin_centers_log_snr = self.log_snr_sorted[centers_idx]
        self.num_bins, self.temperature, self.ema_beta = int(num_bins), float(temperature), float(ema_beta)
        self.entropy_floor_ratio, self.uniform_mix_when_low_entropy = float(entropy_floor_ratio), float(uniform_mix_when_low_entropy)
        self.min_prob = float(min_prob)
        self.center_mu0, self.half_width0, self.widen_to = float(center_mu), float(half_width), float(widen_to)
        self.total_widen_steps = int(total_widen_steps)
        self.cap_max_t = None if cap_max_t is None else int(cap_max_t)
        self.global_step = 0
        self.bin_loss_ema = torch.zeros(self.num_bins, device=log_snr.device)
        self.bin_counts = torch.zeros(self.num_bins, device=log_snr.device)

        self.cap_max_t = None if cap_max_t is None else int(cap_max_t)
        self.cap_target_t = (self.T - 1) if cap_target_t is None else int(cap_target_t)
        self.cap_ema_beta = float(cap_ema_beta)
        self.cap_saturation_thresh = float(cap_saturation_thresh)
        self.cap_step_min = int(cap_step_min)
        self.cap_step_max_frac = float(cap_step_max_frac)
        self.cap_saturation_ema = 0.0

    def _softmax(self, x, temp):
        z = (x / max(temp, 1e-6)); z = z - z.max()
        p = torch.exp(z); return p / (p.sum() + 1e-12)

    def _entropy(self, p):
        return -(p * (p + 1e-12).log()).sum()

    def step(self, global_step=None):
        self.global_step = int(global_step) if global_step is not None else (self.global_step + 1)

    def _current_window(self):
        t = min(self.global_step, self.total_widen_steps) / max(self.total_widen_steps, 1)
        half_w = self.half_width0 + t * (self.widen_to - self.half_width0)
        return self.center_mu0, half_w

    def sample(self, batch_size, device, global_step=0, max_train_steps=1000, sigmoid_scale=1.0, discrete_flow_shift=1.0):
        self.step(global_step)
        mu, half_w = self._current_window()

        centers = self.bin_centers_log_snr.to(device)
        mask = (centers >= (mu - half_w)) & (centers <= (mu + half_w))
        if not mask.any():
            mask = torch.ones_like(centers, dtype=torch.bool)

        with torch.amp.autocast('cuda', enabled=False):
            # Build logits in fp32
            la_logits = -self.bin_loss_ema.to(device, dtype=torch.float32).clone()
            la_logits[~mask] = -1e9

            # Stable softmax in fp32
            t = max(self.temperature, 1e-6)
            z = la_logits / t
            z = z - z.max()
            probs = torch.exp(z)
            probs_sum = probs.sum()
            # Normalize with epsilon to avoid zero-division
            probs = probs / (probs_sum + 1e-12)

            # Optional entropy floor logic (still in fp32)
            H = -(probs * (probs + 1e-12).log()).sum()
            H_min = self.entropy_floor_ratio * math.log(self.num_bins + 1e-8)
            if H < H_min:
                u = torch.full_like(probs, 1.0 / self.num_bins)
                probs = (1.0 - self.uniform_mix_when_low_entropy) * probs + self.uniform_mix_when_low_entropy * u

            # Unconditional clamp and renormalize for safety
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs.clamp_min(max(self.min_prob, 1e-12))
            probs = probs / (probs.sum() + 1e-12)

            # If the row is still numerically degenerate, fall back to uniform
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                probs = torch.full_like(probs, 1.0 / self.num_bins)

            # Sample on GPU; 1D probs → shape [batch_size]
            bin_idx = torch.multinomial(probs, num_samples=batch_size, replacement=True)

        left, right = self.bin_left.to(device)[bin_idx], self.bin_right.to(device)[bin_idx]
        u = torch.rand(batch_size, device=device)
        sorted_idx = (left + (u * (right - left + 1).clamp_min(1)).floor().long()).clamp(0, self.T - 1)
        timesteps = self.sort_indices.to(device)[sorted_idx]

        if self.cap_max_t is not None:
            # measure how many samples want to go beyond the cap
            over_mask = timesteps > self.cap_max_t
            if over_mask.any():
                frac_over = over_mask.float().mean().item()
                # EMA of boundary saturation
                self.cap_saturation_ema = (
                    self.cap_ema_beta * self.cap_saturation_ema
                    + (1.0 - self.cap_ema_beta) * frac_over
                )

                # if boundary is consistently hit, relax the cap a bit
                if (
                    self.cap_saturation_ema > self.cap_saturation_thresh
                    and self.cap_max_t < self.cap_target_t
                ):
                    remaining = self.cap_target_t - self.cap_max_t
                    # limit step size to avoid large jumps
                    max_step = max(
                        self.cap_step_min,
                        int(self.cap_step_max_frac * remaining),
                    )
                    step = max(self.cap_step_min, int(self.cap_saturation_ema * max_step))
                    self.cap_max_t = min(self.cap_max_t + step, self.cap_target_t)
                    # optional: partially reset EMA so the cap does not run too fast
                    self.cap_saturation_ema *= 0.5

            timesteps = timesteps.clamp_max(self.cap_max_t)

        return timesteps.long()

    @torch.no_grad()
    def update(self, timesteps, per_sample_loss):
        device = timesteps.device
        log_snr_t = self.log_snr_original.to(device)[timesteps.long()]
        sorted_pos = torch.searchsorted(self.log_snr_sorted.to(device), log_snr_t)
        right_edges = self.bin_right.to(device)
        bin_idx = torch.bucketize(sorted_pos.clamp_max(self.T - 1), right_edges, right=True).clamp_max(self.num_bins - 1)
        
        beta = self.ema_beta
        for k in bin_idx.unique():
            mask = (bin_idx == k)
            if mask.any():
                loss_k = per_sample_loss[mask].mean().item()
                k_cpu = k.item()
                self.bin_loss_ema[k_cpu] = beta * self.bin_loss_ema[k_cpu] + (1.0 - beta) * loss_k
                self.bin_counts[k_cpu] += mask.sum().item()