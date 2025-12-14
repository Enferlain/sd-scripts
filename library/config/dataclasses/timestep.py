from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimestepConfig:
    min_timestep: Optional[int] = None
    max_timestep: Optional[int] = None
    dynamic_timestep_schedule: Optional[str] = None
    timestep_sampling: str = "uniform"
    sigmoid_scale: float = 1.0
    discrete_flow_shift: float = 1.0
    mix_adaptive_start_p: float = 0.85
    mix_adaptive_end_p: float = 0.35
    mix_adaptive_fixed_p: Optional[float] = None
    mix_adaptive_anneal: str = "cosine"
    mix_adaptive_bins: int = 32
    mix_adaptive_ema_beta: float = 0.9
    mix_adaptive_small_t_frac: float = 0.15
    mix_adaptive_small_t_cap: float = 0.6
    mix_adaptive_temperature: float = 0.5
    mix_adaptive_prior_weight: float = 0.2
    mix_adaptive_min_prob: float = 1e-4
    mix_adaptive_warmup_steps: int = 2000
    mix_adaptive_prior_bias: float = 0.8
    mix_adaptive_entropy_floor_ratio: float = 0.7
    mix_adaptive_uniform_mix_when_low_entropy: float = 0.1
    mix_adaptive_prior_mu: float = 0.0
    mix_adaptive_prior_sigma: float = 1.0
    mix_adaptive_center_mu: float = 0.0
    mix_adaptive_half_width: float = 0.8
    mix_adaptive_widen_to: float = 2.5
    mix_adaptive_max_train_steps: int = 2000
    mix_adaptive_cap_max_t: int = 950
