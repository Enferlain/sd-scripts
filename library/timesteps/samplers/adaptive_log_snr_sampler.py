import math

import torch


class AdaptiveLogSNRSampler:
    """Adaptive sampler that learns over a log-SNR-uniform prior."""

    def __init__(
        self,
        noise_scheduler,
        num_bins: int = 32,
        ema_beta: float = 0.9,
        temperature: float = 0.5,
        prior_weight: float = 0.25,
        min_prob: float = 1e-4,
        warmup_steps: int = 2000,
        entropy_floor_ratio: float = 0.7,
        uniform_mix_when_low_entropy: float = 0.1,
    ):
        self.T = int(noise_scheduler.config.num_train_timesteps)
        a2 = noise_scheduler.alphas_cumprod.float().clamp(1e-12, 1.0 - 1e-12)
        snr = a2 / (1.0 - a2)
        log_snr = torch.log(snr.clamp(min=1e-20))

        self.log_snr_original = log_snr.contiguous()
        self.log_snr_sorted, self.sort_indices = torch.sort(log_snr)
        self.log_snr_sorted = self.log_snr_sorted.contiguous()
        self.sort_indices = self.sort_indices.contiguous()

        self.num_bins = int(num_bins)
        self.ema_beta = float(ema_beta)
        self.temperature = float(temperature)
        self.prior_weight = float(prior_weight)
        self.min_prob = float(min_prob)
        self.warmup_steps = int(warmup_steps)
        self.entropy_floor_ratio = float(entropy_floor_ratio)
        self.uniform_mix_when_low_entropy = float(uniform_mix_when_low_entropy)

        log_snr_min = self.log_snr_sorted[0].item()
        log_snr_max = self.log_snr_sorted[-1].item()
        self.bin_edges = torch.linspace(log_snr_min, log_snr_max, self.num_bins + 1, dtype=torch.float32)
        self.bin_loss_ema = torch.ones(self.num_bins, dtype=torch.float32)
        self.last_entropy_ratio = 1.0

    @torch.no_grad()
    def update(self, timesteps: torch.Tensor, per_sample_loss: torch.Tensor):
        """Update the loss EMA for each log-SNR bin."""
        device = timesteps.device
        log_snr_t = self.log_snr_original.to(device)[timesteps.long()]
        bins = torch.bucketize(log_snr_t, self.bin_edges.to(device)) - 1
        bins = bins.clamp(0, self.num_bins - 1)

        vals = per_sample_loss.detach().to(torch.float32)
        bin_loss = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        bin_cnt = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        bin_loss.index_add_(0, bins, vals)
        bin_cnt.index_add_(0, bins, torch.ones_like(vals, dtype=torch.float32))

        mask = bin_cnt > 0
        ema = self.bin_loss_ema.to(device)
        batch_mean = torch.where(mask, bin_loss / (bin_cnt + 1e-8), ema)
        self.bin_loss_ema = self.ema_beta * ema + (1.0 - self.ema_beta) * batch_mean

    def _softmax(self, x: torch.Tensor, temp: float) -> torch.Tensor:
        """Compute a temperature-scaled softmax."""
        z = x / max(temp, 1e-6)
        z = z - z.max()
        probs = torch.exp(z)
        return probs / (probs.sum() + 1e-12)

    @torch.no_grad()
    def sample(
        self,
        bsz: int,
        device: torch.device,
        global_step: int,
        max_steps: int,
    ) -> torch.Tensor:
        """Sample timesteps by mixing a log-SNR-uniform prior with loss-aware difficulty."""
        del max_steps

        prior = torch.full((self.num_bins,), 1.0 / self.num_bins, device=device, dtype=torch.float32)
        if global_step < self.warmup_steps:
            mixed = prior
        else:
            ema = self.bin_loss_ema.to(device).clamp(min=1e-8)
            relative_difficulty = ema / ema.mean().clamp(min=1e-8)
            loss_probs = self._softmax(torch.log(relative_difficulty + 1e-8), self.temperature)
            mixed = self.prior_weight * prior + (1.0 - self.prior_weight) * loss_probs

            entropy = -(mixed * (mixed + 1e-8).log()).sum()
            entropy_floor = self.entropy_floor_ratio * math.log(self.num_bins + 1e-8)
            self.last_entropy_ratio = float((entropy / max(math.log(self.num_bins + 1e-8), 1e-8)).item())
            if entropy < entropy_floor:
                mixed = (1.0 - self.uniform_mix_when_low_entropy) * mixed + self.uniform_mix_when_low_entropy * prior

        mixed = mixed.clamp_min(self.min_prob)
        mixed = mixed / (mixed.sum() + 1e-12)

        bin_ids = torch.multinomial(mixed, bsz, replacement=True)
        left = self.bin_edges[:-1].to(device)[bin_ids]
        right = self.bin_edges[1:].to(device)[bin_ids]
        target_log_snr = left + torch.rand(bsz, device=device) * (right - left).clamp_min(1e-8)

        sorted_lsnr = self.log_snr_sorted.to(device)
        idx_in_sorted = torch.searchsorted(sorted_lsnr, target_log_snr).clamp(1, self.T - 1)
        prev_idx = idx_in_sorted - 1
        left_lsnr = sorted_lsnr[prev_idx]
        right_lsnr = sorted_lsnr[idx_in_sorted]
        weight = ((target_log_snr - left_lsnr) / (right_lsnr - left_lsnr + 1e-12)).clamp(0, 1)
        left_t = self.sort_indices.to(device)[prev_idx].float()
        right_t = self.sort_indices.to(device)[idx_in_sorted].float()
        timesteps = torch.round((1.0 - weight) * left_t + weight * right_t).long()
        return timesteps.clamp(0, self.T - 1)
