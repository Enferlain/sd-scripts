# gaussian_mid_snr_sampler.py
import math, torch


class GaussianMidSNRAdaptiveSampler:
    """
    Mid-SNR-peaked loss-aware sampler in log-SNR space.
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
        # Gaussian prior in log-SNR:
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        prior_weight: float = 0.1,
        warmup_steps: int = 10,
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
        self.min_prob, self.prior_weight, self.warmup_steps = float(min_prob), float(prior_weight), int(warmup_steps)
        self.prior_mu, self.prior_sigma = float(prior_mu), float(prior_sigma)
        self.bin_loss_ema = torch.zeros(self.num_bins, device=log_snr.device)
        self.bin_counts = torch.zeros(self.num_bins, device=log_snr.device)
        self.global_step = 0
        self._update_prior_logits()

    def _gaussian_prior_logits(self):
        x = self.bin_centers_log_snr
        mu = torch.tensor(self.prior_mu, device=x.device)
        sigma = torch.tensor(max(self.prior_sigma, 1e-4), device=x.device)
        return -0.5 * ((x - mu) / sigma).pow(2)

    def _update_prior_logits(self):
        self.prior_logits = self._gaussian_prior_logits()

    def _softmax(self, x, temp):
        z = (x / max(temp, 1e-6))
        z = z - z.max()
        p = torch.exp(z)
        return p / (p.sum() + 1e-12)

    def _entropy(self, p):
        return -(p * (p + 1e-12).log()).sum()

    def step(self, global_step=None):
        self.global_step = int(global_step) if global_step is not None else (self.global_step + 1)
        self._update_prior_logits()

    def sample(self, batch_size, device, global_step=0, max_train_steps=1000, sigmoid_scale=1.0, discrete_flow_shift=1.0):
        self.step(global_step)
        prior_probs = self._softmax(self.prior_logits.to(device), temp=1.0)
        if self.global_step < self.warmup_steps:
            mixed = prior_probs.clone()
        else:
            la_logits = -self.bin_loss_ema.to(device).clone()
            la_probs = self._softmax(la_logits, temp=self.temperature)
            mixed = (1.0 - self.prior_weight) * la_probs + self.prior_weight * prior_probs
        H = self._entropy(mixed)
        H_min = self.entropy_floor_ratio * math.log(self.num_bins + 1e-8)
        if H < H_min:
            u = torch.full_like(mixed, 1.0 / self.num_bins)
            mixed = (1.0 - self.uniform_mix_when_low_entropy) * mixed + self.uniform_mix_when_low_entropy * u
        mixed = mixed.clamp_min(self.min_prob)
        mixed = mixed / (mixed.sum() + 1e-12)
        bin_idx = torch.multinomial(mixed, num_samples=batch_size, replacement=True)
        left, right = self.bin_left.to(device)[bin_idx], self.bin_right.to(device)[bin_idx]
        u = torch.rand(batch_size, device=device)
        sorted_idx = (left + (u * (right - left + 1).clamp_min(1)).floor().long()).clamp(0, self.T - 1)
        timesteps = self.sort_indices.to(device)[sorted_idx]
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
