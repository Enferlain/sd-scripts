# loss_aware_sampler.py
import math
import torch


class LossAwareTimestepSampler:
    """
    Timestep sampler that balances between a prior (sigmoid/shift) and loss-aware sampling.

    This sampler blends a fixed prior distribution (which can be annealed over time)
    with a learned distribution based on the historical loss of different timesteps bins.
    It also includes mechanisms to cap the sampling of small timesteps.
    """

    def __init__(
        self,
        num_train_timesteps: int,
        num_bins: int = 32,
        ema_beta: float = 0.9,
        small_t_frac: float = 0.15,
        small_t_cap: float = 0.6,
        start_p: float = 0.85,
        end_p: float = 0.35,
        anneal: str = "cosine",
        fixed_p: float | None = None,
    ):
        """
        Initialize the LossAwareTimestepSampler.

        Args:
            num_train_timesteps (int): Total number of training timesteps.
            num_bins (int): Number of bins to track loss statistics.
            ema_beta (float): Decay factor for the exponential moving average of bin losses.
            small_t_frac (float): Fraction of the total timesteps considered as "small t".
            small_t_cap (float): Maximum fraction of the batch allowed to be from the "small t" region.
            start_p (float): Starting probability for the prior component (vs. loss-aware component).
            end_p (float): Ending probability for the prior component.
            anneal (str): Annealing schedule for mixing probability ("cosine", "linear", or "none").
            fixed_p (float, optional): If set, uses a fixed mixing probability, overriding anneal/start/end.
        """
        print(
            f"LossAwareTimestepSampler initialized with: num_train_timesteps={num_train_timesteps}, num_bins={num_bins}, ema_beta={ema_beta}, small_t_frac={small_t_frac}, small_t_cap={small_t_cap}, start_p={start_p}, end_p={end_p}, anneal={anneal}, fixed_p={fixed_p}"
        )
        self.T = int(num_train_timesteps)
        self.num_bins = int(num_bins)
        self.ema_beta = float(ema_beta)
        self.small_t_frac = float(small_t_frac)  # define "small t" region as lowest X% of steps
        self.small_t_cap = float(small_t_cap)  # max batch fraction from small t
        self.eps = 1e-8

        self.start_p = float(start_p)
        self.end_p = float(end_p)
        self.anneal = str(anneal)
        self.fixed_p = None if fixed_p is None else float(fixed_p)

        self.bin_edges = torch.linspace(0, self.T, self.num_bins + 1)  # CPU by default
        self.ema_loss = torch.ones(self.num_bins)  # start uniform

        self.last_mix_p = 0.0
        self.last_small_t_frac = 0.0

    @torch.no_grad()
    def update(self, timesteps: torch.Tensor, per_sample_losses: torch.Tensor):
        """
        Update the sampler state with observed losses.

        Args:
            timesteps (torch.Tensor): The timesteps used for the batch.
            per_sample_losses (torch.Tensor): The loss for each sample in the batch.
        """
        # expects 1D tensors on same device
        bins = torch.bucketize(timesteps.float(), self.bin_edges.to(timesteps.device)) - 1
        bins = bins.clamp(0, self.num_bins - 1)

        # EMA update in float32
        update_dtype = torch.float32
        bin_loss = torch.zeros(self.num_bins, device=timesteps.device, dtype=update_dtype)
        bin_cnt = torch.zeros(self.num_bins, device=timesteps.device, dtype=update_dtype)
        bin_loss.index_add_(0, bins, per_sample_losses.detach().to(dtype=update_dtype))
        bin_cnt.index_add_(0, bins, torch.ones_like(per_sample_losses, dtype=update_dtype))

        mask = bin_cnt > 0
        ema_loss_device = self.ema_loss.to(device=timesteps.device, dtype=update_dtype)
        new_vals = torch.where(mask, bin_loss / (bin_cnt + self.eps), ema_loss_device)
        self.ema_loss = self.ema_beta * ema_loss_device + (1 - self.ema_beta) * new_vals

    def _sched(self, step: int, total: int) -> float:
        """Calculate the mixing probability based on the current step."""
        if self.fixed_p is not None:
            return self.fixed_p
        if self.anneal == "none":
            return self.start_p
        if self.anneal == "linear":
            s = max(0, min(step, total))
            return self.start_p + (self.end_p - self.start_p) * (s / max(1, total))
        # cosine default
        s = max(0, min(step, total))
        cos = 0.5 * (1 - math.cos(math.pi * s / max(1, total)))
        return self.start_p + (self.end_p - self.start_p) * cos

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
            global_step (int): Current global step.
            max_steps (int): Total training steps used for annealing.
            sigmoid_scale (float): Scale factor for the sigmoid prior.
            discrete_flow_shift (float): Shift parameter for the discrete flow prior.

        Returns:
            torch.Tensor: A tensor of sampled timesteps with shape (bsz,).
        """
        # mix_p: probability of taking the sigmoid branch (composition prior)
        mix_p = self._sched(global_step, max_steps)
        self.last_mix_p = mix_p

        # Sigmoid branch (mid/high‑t)
        z = torch.randn(bsz, device=device) * sigmoid_scale
        t_sig = z.sigmoid()  # in [0,1]

        # Clean‑t branch via “shifted” mapping
        u = torch.rand(bsz, device=device)
        s = max(1.0e-6, float(discrete_flow_shift))
        t_low = (u * s) / (1 + (s - 1) * u)  # in [0,1], biased toward small t

        # Per-sample mixture
        mask = (torch.rand(bsz, device=device) < mix_p).float()
        t_mix = t_sig * mask + t_low * (1 - mask)
        t_idx = (t_mix.clamp(0, 1) * (self.T - 1)).round().long()

        # Loss-aware reweighting across bins
        ema = self.ema_loss.to(device=device, dtype=torch.float32)
        probs = (ema / (ema.sum() + self.eps)).clamp(min=1e-6)
        cat = torch.distributions.Categorical(probs=probs)
        # Sample bins, then uniform within each chosen bin
        bin_ids = cat.sample((bsz,))
        left = self.bin_edges[:-1].to(device)[bin_ids]
        right = self.bin_edges[1:].to(device)[bin_ids]
        t_bin = left + torch.rand(bsz, device=device) * (right - left - 1).clamp(min=1.0)
        t_bin = t_bin.round().clamp(0, self.T - 1).long()

        # Blend prior index with loss-aware index (lean more loss-aware as mix_p decays)
        t_final = (mix_p * t_idx.float() + (1 - mix_p) * t_bin.float()).round().long()
        t_final = t_final.clamp(0, self.T - 1)

        # Enforce cap on “small t” fraction
        low_cut = int(self.small_t_frac * self.T)
        low_mask = t_final < low_cut
        frac_low = low_mask.float().mean()
        self.last_small_t_frac = frac_low.item()
        if frac_low > self.small_t_cap:
            excess = int((frac_low - self.small_t_cap) * bsz)
            idxs = torch.nonzero(low_mask).squeeze(-1)
            if excess > 0 and idxs.numel() > 0:
                sel = idxs[torch.randperm(idxs.numel(), device=device)[:excess]]
                t_final[sel] = torch.randint(low_cut, self.T, (excess,), device=device)
        return t_final
