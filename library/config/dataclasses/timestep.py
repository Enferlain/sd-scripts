from dataclasses import dataclass, field


@dataclass
class AdaptiveLogSNRConfig:
    """Configuration for the retained general-purpose adaptive log-SNR sampler."""

    bins: int = field(default=32, metadata={"help": "Number of bins for adaptive sampling"})
    ema_beta: float = field(default=0.9, metadata={"help": "EMA decay factor for loss tracking"})
    temperature: float = field(default=0.5, metadata={"help": "Temperature for softmax distribution"})
    prior_weight: float = field(default=0.25, metadata={"help": "Weight given to the log-SNR-uniform prior"})
    min_prob: float = field(default=1e-4, metadata={"help": "Minimum probability per bin"})
    warmup_steps: int = field(default=2000, metadata={"help": "Warmup steps before full adaptive sampling"})
    entropy_floor: float = field(default=0.7, metadata={"help": "Minimum entropy floor ratio"})
    uniform_mix_when_low_entropy: float = field(default=0.1, metadata={"help": "Uniform mix ratio when entropy is low"})


@dataclass
class TimestepConfig:
    """
    Timestep sampling configuration.

    timestep_sampling values: uniform, shift, log_snr_uniform, adaptive_log_snr
    """

    # Core settings
    timestep_sampling: str = field(default="uniform", metadata={"help": "Timestep sampling method"})
    min_timestep: int | None = field(default=None, metadata={"help": "Minimum timesteps for training"})
    max_timestep: int | None = field(default=None, metadata={"help": "Maximum timesteps for training"})
    dynamic_timestep_schedule: str | None = field(default=None, metadata={"help": "Dynamic timesteps schedule string"})
    sigmoid_scale: float = field(default=1.0, metadata={"help": "Scale for sigmoid sampling"})
    discrete_flow_shift: float = field(default=1.0, metadata={"help": "Shift for discrete flow sampling"})

    # Per-sampler configurations
    adaptive_log_snr: AdaptiveLogSNRConfig = field(
        default_factory=AdaptiveLogSNRConfig, metadata={"help": "Adaptive log-SNR sampler settings"}
    )
