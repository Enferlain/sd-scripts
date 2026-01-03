from dataclasses import dataclass, field


@dataclass
class MixAdaptiveConfig:
    """Configuration for the original loss-aware timesteps sampler."""

    bins: int = field(default=32, metadata={"help": "Number of bins for adaptive sampling"})
    ema_beta: float = field(default=0.9, metadata={"help": "EMA decay factor for loss tracking"})
    start_p: float = field(default=0.85, metadata={"help": "Starting probability for adaptive mixing"})
    end_p: float = field(default=0.35, metadata={"help": "Ending probability for adaptive mixing"})
    fixed_p: float | None = field(default=None, metadata={"help": "Fixed probability (overrides start_p/end_p if set)"})
    anneal: str = field(default="cosine", metadata={"help": "Annealing schedule: linear, cosine"})
    small_t_frac: float = field(default=0.15, metadata={"help": "Fraction of bins considered 'small t'"})
    small_t_cap: float = field(default=0.6, metadata={"help": "Cap for small timesteps sampling probability"})


@dataclass
class TemperedAdaptiveConfig:
    """Configuration for tempered adaptive timesteps sampler."""

    bins: int = field(default=32, metadata={"help": "Number of bins for adaptive sampling"})
    ema_beta: float = field(default=0.9, metadata={"help": "EMA decay factor for loss tracking"})
    temperature: float = field(default=0.5, metadata={"help": "Temperature for softmax distribution"})
    prior_weight: float = field(default=0.2, metadata={"help": "Weight of prior distribution"})
    min_prob: float = field(default=1e-4, metadata={"help": "Minimum probability per bin"})
    warmup_steps: int = field(default=2000, metadata={"help": "Warmup steps before full adaptive sampling"})
    prior_bias: float = field(default=0.8, metadata={"help": "Bias towards prior distribution"})
    entropy_floor: float = field(default=0.7, metadata={"help": "Minimum entropy floor ratio"})


@dataclass
class GaussianMidSNRConfig:
    """Configuration for Gaussian mid-SNR adaptive timesteps sampler."""

    bins: int = field(default=32, metadata={"help": "Number of bins for adaptive sampling"})
    ema_beta: float = field(default=0.9, metadata={"help": "EMA decay factor for loss tracking"})
    temperature: float = field(default=0.5, metadata={"help": "Temperature for softmax distribution"})
    min_prob: float = field(default=1e-4, metadata={"help": "Minimum probability per bin"})
    entropy_floor: float = field(default=0.7, metadata={"help": "Minimum entropy floor ratio"})
    prior_mu: float = field(default=0.0, metadata={"help": "Mean of Gaussian prior"})
    prior_sigma: float = field(default=1.0, metadata={"help": "Std dev of Gaussian prior"})
    prior_weight: float = field(default=0.2, metadata={"help": "Weight of prior distribution"})
    warmup_steps: int = field(default=2000, metadata={"help": "Warmup steps before full adaptive sampling"})
    uniform_mix_when_low_entropy: float = field(default=0.1, metadata={"help": "Uniform mix ratio when entropy is low"})


@dataclass
class SNRWindowedConfig:
    """Configuration for SNR-windowed loss-aware timesteps sampler."""

    bins: int = field(default=32, metadata={"help": "Number of bins for adaptive sampling"})
    ema_beta: float = field(default=0.9, metadata={"help": "EMA decay factor for loss tracking"})
    temperature: float = field(default=0.5, metadata={"help": "Temperature for softmax distribution"})
    min_prob: float = field(default=1e-4, metadata={"help": "Minimum probability per bin"})
    entropy_floor: float = field(default=0.7, metadata={"help": "Minimum entropy floor ratio"})
    center_mu: float = field(default=0.0, metadata={"help": "Center of sampling window"})
    half_width: float = field(default=0.8, metadata={"help": "Half-width of sampling window"})
    widen_to: float = field(default=2.5, metadata={"help": "Final width to widen to"})
    max_train_steps: int = field(default=2000, metadata={"help": "Steps to fully widen window"})
    cap_max_t: int = field(default=950, metadata={"help": "Maximum timesteps cap"})
    uniform_mix_when_low_entropy: float = field(default=0.1, metadata={"help": "Uniform mix ratio when entropy is low"})


@dataclass
class TimestepConfig:
    """
    Timestep sampling configuration.

    timestep_sampling values: uniform, shift, mix_adaptive, tempered_adaptive, gaussian_mid_snr, snr_windowed, log_snr_uniform
    """

    # Core settings
    timestep_sampling: str = field(default="uniform", metadata={"help": "Timestep sampling method"})
    min_timestep: int | None = field(default=None, metadata={"help": "Minimum timesteps for training"})
    max_timestep: int | None = field(default=None, metadata={"help": "Maximum timesteps for training"})
    dynamic_timestep_schedule: str | None = field(default=None, metadata={"help": "Dynamic timesteps schedule string"})
    sigmoid_scale: float = field(default=1.0, metadata={"help": "Scale for sigmoid sampling"})
    discrete_flow_shift: float = field(default=1.0, metadata={"help": "Shift for discrete flow sampling"})

    # Per-sampler configurations
    mix_adaptive: MixAdaptiveConfig = field(default_factory=MixAdaptiveConfig, metadata={"help": "Mix adaptive sampler settings"})
    tempered_adaptive: TemperedAdaptiveConfig = field(
        default_factory=TemperedAdaptiveConfig, metadata={"help": "Tempered adaptive sampler settings"}
    )
    gaussian_mid_snr: GaussianMidSNRConfig = field(
        default_factory=GaussianMidSNRConfig, metadata={"help": "Gaussian mid-SNR sampler settings"}
    )
    snr_windowed: SNRWindowedConfig = field(default_factory=SNRWindowedConfig, metadata={"help": "SNR windowed sampler settings"})
