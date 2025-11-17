import torch
import math


class TemperedAdaptiveSampler:
    def __init__(self, noise_scheduler, num_bins: int = 64,
                 ema_beta: float = 0.95, temperature: float = 0.4,
                 prior_weight: float = 0.3, min_prob: float = 5e-4,
                 warmup_steps: int = 150, prior_bias: float = 1.0,
                 entropy_floor_ratio: float = 0.6):
        print(
            f"TemperedAdaptiveSampler initialized with: num_bins={num_bins}, ema_beta={ema_beta}, temperature={temperature}, prior_weight={prior_weight}, min_prob={min_prob}, warmup_steps={warmup_steps}, prior_bias={prior_bias}, entropy_floor_ratio={entropy_floor_ratio}")
        a2 = noise_scheduler.alphas_cumprod.float().clamp(1e-12, 1. - 1e-12)  # [T]
        snr = a2 / (1. - a2)
        log_snr = torch.log(snr.clamp(min=1e-20))  # [T]

        # Sort to ascending for searchsorted (store both original and sorted)
        self.log_snr_original = log_snr.contiguous()  # for update()
        self.log_snr_sorted, self.sort_indices = torch.sort(log_snr)  # ascending
        self.log_snr_sorted = self.log_snr_sorted.contiguous()
        self.sort_indices = self.sort_indices.contiguous()
        self.T = int(noise_scheduler.config.num_train_timesteps)  # scalar

        # Bins over original log_snr distribution
        q = torch.linspace(0., 1., int(num_bins) + 1, dtype=torch.float32)
        self.bin_edges = torch.quantile(log_snr, q).to(torch.float32).contiguous()
        self.num_bins = int(num_bins)

        # State (fp32)
        self.ema_beta = float(ema_beta)
        self.temperature = float(temperature)
        self.prior_weight = float(prior_weight)
        self.min_prob = float(min_prob)
        self.warmup_steps = int(warmup_steps)
        self.entropy_floor_ratio = float(entropy_floor_ratio)

        self.bin_loss_ema = torch.ones(self.num_bins, dtype=torch.float32)
        self.ema_sq = torch.ones(self.num_bins, dtype=torch.float32)
        self.counts = torch.zeros(self.num_bins, dtype=torch.float32)

        # Emphasize high-noise bins via prior
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        lmid = torch.quantile(self.log_snr_original, torch.tensor(0.5))
        prior_logits = float(prior_bias) * (lmid - centers)
        prior = torch.softmax(prior_logits.to(torch.float32), dim=0)

        # Ensure no bin is starved, even with a strong bias
        self.prior_probs = (prior + self.min_prob)
        self.prior_probs = self.prior_probs / self.prior_probs.sum()

    @torch.no_grad()
    def update(self, timesteps: torch.Tensor, per_sample_losses: torch.Tensor):
        # Map t -> bin via log-SNR
        device = timesteps.device
        log_snr_t = self.log_snr_original.to(device)[timesteps]
        bins = torch.bucketize(log_snr_t, self.bin_edges.to(device)) - 1
        bins = bins.clamp(0, self.num_bins - 1)

        # Aggregate batch stats
        vals = per_sample_losses.detach().to(torch.float32)
        bin_loss = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        bin_sq = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        bin_cnt = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        bin_loss.index_add_(0, bins, vals)
        bin_sq.index_add_(0, bins, vals * vals)
        bin_cnt.index_add_(0, bins, torch.ones_like(vals, dtype=torch.float32))

        # EMA updates
        mask = bin_cnt > 0
        ema = self.bin_loss_ema.to(device)
        ema2 = self.ema_sq.to(device)
        new_mean = torch.where(mask, bin_loss / (bin_cnt + 1e-8), ema)
        new_sq = torch.where(mask, bin_sq / (bin_cnt + 1e-8), ema2)
        self.bin_loss_ema = self.ema_beta * ema + (1 - self.ema_beta) * new_mean
        self.ema_sq = self.ema_beta * ema2 + (1 - self.ema_beta) * new_sq
        self.counts = self.counts.to(device) + bin_cnt

    @torch.no_grad()
    def sample(self, bsz: int, device, global_step: int, max_steps: int,
            sigmoid_scale: float = 1.0, discrete_flow_shift: float = 1.0):
        # Optional: expose this once in __init__
        uniform_mix_when_low_entropy = getattr(self, "uniform_mix_when_low_entropy", 0.10)

        # Short warmup: either off or tiny, and mix in uniform (not pure prior)
        if global_step < self.warmup_steps:
            prior = self.prior_probs.to(device)
            uniform = torch.full_like(prior, 1.0 / self.num_bins)
            mixed = 0.5 * prior + 0.5 * uniform
            mixed = (mixed + self.min_prob)
            mixed = mixed / mixed.sum()
        else:
            # Loss-aware term
            ema = self.bin_loss_ema.to(device).clamp(min=1e-8)
            var = (self.ema_sq.to(device) - ema * ema).clamp(min=0.)
            score = ema / (torch.sqrt(var + 1e-6) + 1e-6)

            # Tempered logits
            logits = self.temperature * torch.log(score + 1e-8)
            logits = logits.clamp(min=-12.0, max=12.0)
            loss_probs = torch.softmax(logits, dim=0)

            # Mix with (weakened) prior
            prior = self.prior_probs.to(device)
            mixed = (1.0 - self.prior_weight) * loss_probs + self.prior_weight * prior

            # Entropy guard: mix in uniform mass instead of bumping prior_weight
            H = -(mixed * (mixed + 1e-8).log()).sum()
            H_min = self.entropy_floor_ratio * math.log(self.num_bins + 1e-8)
            if H < H_min:
                uniform = torch.full_like(mixed, 1.0 / self.num_bins)
                mixed = (1.0 - uniform_mix_when_low_entropy) * mixed + uniform_mix_when_low_entropy * uniform

            mixed = (mixed + self.min_prob)
            mixed = mixed / mixed.sum()

        # Sample bins and map to timesteps (unchanged)
        bin_ids = torch.multinomial(mixed, bsz, replacement=True)
        left = self.bin_edges[:-1].to(device)[bin_ids]
        right = self.bin_edges[1:].to(device)[bin_ids]
        span = (right - left).clamp_min(1e-8)
        target_lsnr = left + torch.rand(bsz, device=device) * span
        sorted_lsnr = self.log_snr_sorted.to(device)
        idx_in_sorted = torch.searchsorted(sorted_lsnr, target_lsnr).clamp(1, self.T - 1)
        j = (idx_in_sorted - 1)
        l0 = sorted_lsnr[j]; l1 = sorted_lsnr[idx_in_sorted]
        w = ((target_lsnr - l0) / (l1 - l0 + 1e-12)).clamp(0, 1)
        t0 = self.sort_indices.to(device)[j].float()
        t1 = self.sort_indices.to(device)[idx_in_sorted].float()
        t_local = torch.round((1 - w) * t0 + w * t1).long().clamp(0, self.T - 1)
        return t_local