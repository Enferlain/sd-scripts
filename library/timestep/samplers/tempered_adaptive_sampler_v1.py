import torch


class TemperedAdaptiveSampler:
    def __init__(self, noise_scheduler, num_bins: int = 64,
                 ema_beta: float = 0.95, temperature: float = 0.5,
                 prior_weight: float = 0.2, min_prob: float = 1e-4,
                 warmup_steps: int = 2000):
        print(
            f"TemperedAdaptiveSampler initialized with: num_bins={num_bins}, ema_beta={ema_beta}, temperature={temperature}, prior_weight={prior_weight}, min_prob={min_prob}, warmup_steps={warmup_steps}")
        a2 = noise_scheduler.alphas_cumprod.float().clamp(1e-12, 1. - 1e-12)
        snr = a2 / (1. - a2)
        log_snr = torch.log(snr.clamp(min=1e-20))  # [T], typically descending in t

        # Sort to ascending for searchsorted (store both original and sorted)
        self.log_snr_original = log_snr  # for update() bin mapping
        sorted_log_snr, self.sort_indices = torch.sort(log_snr)  # ascending
        self.log_snr_sorted = sorted_log_snr
        self.T = noise_scheduler.config.num_train_timesteps

        # Bins over original log_snr distribution
        q = torch.linspace(0., 1., int(num_bins) + 1, dtype=torch.float32)
        self.bin_edges = torch.quantile(log_snr, q).to(torch.float32)
        self.num_bins = int(num_bins)

        # State (fp32)
        self.ema_beta = float(ema_beta)
        self.temperature = float(temperature)
        self.prior_weight = float(prior_weight)
        self.min_prob = float(min_prob)
        self.warmup_steps = int(warmup_steps)

        self.ema_loss = torch.ones(self.num_bins, dtype=torch.float32)
        self.ema_sq = torch.ones(self.num_bins, dtype=torch.float32)
        self.counts = torch.zeros(self.num_bins, dtype=torch.float32)

        self.prior_probs = torch.full((self.num_bins,), 1.0 / self.num_bins, dtype=torch.float32)

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
        ema = self.ema_loss.to(device)
        ema2 = self.ema_sq.to(device)
        new_mean = torch.where(mask, bin_loss / (bin_cnt + 1e-8), ema)
        new_sq = torch.where(mask, bin_sq / (bin_cnt + 1e-8), ema2)
        self.ema_loss = self.ema_beta * ema + (1 - self.ema_beta) * new_mean
        self.ema_sq = self.ema_beta * ema2 + (1 - self.ema_beta) * new_sq
        self.counts = self.counts.to(device) + bin_cnt

    @torch.no_grad()
    def sample(self, bsz: int, device, global_step: int, max_steps: int,
               sigmoid_scale: float = 1.0, discrete_flow_shift: float = 0.9):
        # Warm-up: exploration only
        if global_step < self.warmup_steps:
            mixed = self.prior_probs.to(device)
        else:
            ema = self.ema_loss.to(device).clamp(min=1e-8)
            var = (self.ema_sq.to(device) - ema * ema).clamp(min=0.)
            score = ema / (torch.sqrt(var + 1e-6) + 1e-6)

            # Log-domain tempering with clamp for stability
            logits = self.temperature * torch.log(score + 1e-8)
            logits = logits.clamp(min=-12.0, max=12.0)
            loss_probs = torch.softmax(logits, dim=0)

            mixed = (1 - self.prior_weight) * loss_probs + self.prior_weight * self.prior_probs.to(device)
            mixed = (mixed + self.min_prob)
            mixed = mixed / mixed.sum()

        # Sample bins
        bin_ids = torch.multinomial(mixed, bsz, replacement=True)

        # Uniform in log-SNR within bin, then invert log-SNR -> t via searchsorted
        left = self.bin_edges[:-1].to(device)[bin_ids]
        right = self.bin_edges[1:].to(device)[bin_ids]
        target_lsnr = left + torch.rand(bsz, device=device) * (right - left)

        # Invert: find index in sorted ascending log_snr
        sorted_lsnr = self.log_snr_sorted.to(device)
        idx_in_sorted = torch.searchsorted(sorted_lsnr, target_lsnr).clamp(0, self.T - 1)

        # Map back to original t indices
        t_local = self.sort_indices.to(device)[idx_in_sorted]
        return t_local
