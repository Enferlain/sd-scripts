from dataclasses import dataclass, field


@dataclass
class HuberConfig:
    """Huber loss settings."""

    huber_schedule: str = field(default="snr", metadata={"help": "Huber threshold schedule: constant, exponential, or snr"})
    huber_c: float = field(default=0.1, metadata={"help": "Base Huber threshold value"})
    huber_scale: float = field(default=1.0, metadata={"help": "Scale factor applied to Huber threshold"})


@dataclass
class SNRConfig:
    """SNR-based loss weighting settings."""

    min_snr_gamma: float | None = field(default=None, metadata={"help": "Min-SNR gamma for loss weighting (None to disable)"})
    scale_v_pred_loss_like_noise_pred: bool = field(default=False, metadata={"help": "Scale v-prediction loss like noise prediction"})
    v_pred_like_loss: float | None = field(default=None, metadata={"help": "Add v-prediction-like loss term (multiplier, None to disable)"})
    debiased_estimation_loss: bool = field(default=False, metadata={"help": "Apply debiased estimation loss weighting"})


@dataclass
class MaskedLossConfig:
    """Masked loss settings."""

    masked_loss: bool = field(default=False, metadata={"help": "Apply mask for calculating loss"})
    conditioning_data_dir: str | None = field(default=None, metadata={"help": "Directory containing conditioning/mask images"})


@dataclass
class RegularizationConfig:
    """Noise regularization settings."""

    noise_offset: float | None = field(default=None, metadata={"help": "Noise offset for improving dark/light image handling"})
    noise_offset_random_strength: bool = field(
        default=False, metadata={"help": "Randomize noise offset strength between 0 and noise_offset"}
    )
    multires_noise_iterations: int | None = field(
        default=None, metadata={"help": "Pyramid noise iterations (enables multires noise if set)"}
    )
    multires_noise_discount: float = field(default=0.3, metadata={"help": "Discount factor per pyramid level for multires noise"})
    ip_noise_gamma: float | None = field(default=None, metadata={"help": "Input perturbation noise gamma (None to disable)"})
    ip_noise_gamma_random_strength: bool = field(default=False, metadata={"help": "Randomize input perturbation noise strength"})
    adaptive_noise_scale: float | None = field(default=None, metadata={"help": "Scale noise offset adaptively based on latent statistics"})
    zero_terminal_snr: bool = field(default=False, metadata={"help": "Use zero terminal SNR noise scheduling"})


@dataclass
class EDM2OptimizerConfig:
    """Optimizer settings for the EDM2 sidecar model."""

    type: str = field(
        default="torch.optim.AdamW",
        metadata={"help": "Fully qualified optimizer class name to use with the EDM2 loss-weighting optimizer."},
    )
    lr: float = field(default=2e-2, metadata={"help": "Learning rate for the EDM2 loss-weighting optimizer."})
    args: str = field(
        default="{'weight_decay': 0, 'betas': (0.9,0.999)}",
        metadata={"help": "A JSON-like string of optimizer args for the EDM2 loss-weighting optimizer."},
    )
    use_scheduler: bool = field(default=False, metadata={"help": "Use an LR scheduler with the EDM2 optimizer."})
    warmup_percent: float = field(default=0.1, metadata={"help": "Percent of training steps to use for warmup."})
    constant_percent: float = field(
        default=0.1, metadata={"help": "Percent of training steps to keep LR constant before decay."}
    )
    decay_scaling: float = field(default=1.0, metadata={"help": "Scaling factor applied to scheduler decay."})


@dataclass
class EDM2ImportanceConfig:
    """Importance-weighting settings for EDM2."""

    enabled: bool = field(default=False, metadata={"help": "Weight EDM2 loss scaling by timestep importance."})
    max_weight: float = field(default=10.0, metadata={"help": "Maximum weighting/scaling used for EDM2 importance weighting."})
    min_snr_gamma: float = field(default=1.0, metadata={"help": "Min-SNR gamma heuristic used for EDM2 importance weighting."})
    safety_override: bool = field(
        default=False,
        metadata={
            "help": "Allow stacking debiased loss and/or regular min-SNR gamma with EDM2 importance weighting."
        },
    )


@dataclass
class EDM2VisualizationConfig:
    """Optional visualization settings for EDM2 loss weights."""

    enabled: bool = field(default=False, metadata={"help": "Generate graph images that show EDM2 loss weighting over timesteps."})
    every_n_steps: int = field(default=20, metadata={"help": "Generate a graph image every N steps."})
    output_dir: str | None = field(default=None, metadata={"help": "Parent directory for EDM2 graph images."})
    y_limit: int | None = field(default=None, metadata={"help": "Optional max limit for the graph y-axis."})
    y_scale: str = field(default="linear", metadata={"help": "Select between linear or log scaling for the y-axis."})


@dataclass
class EDM2Config:
    """EDM2 loss-weighting settings grouped under one feature root."""

    enabled: bool = field(default=False, metadata={"help": "Enable EDM2 loss weighting."})
    laplace_timestep_sampling: bool = field(
        default=False,
        metadata={"help": "Experimental EDM2-related Laplace timestep sampling toggle (currently unsupported)."},
    )
    num_channels: int = field(default=128, metadata={"help": "Number of channels used by the EDM2 loss-weighting module."})
    initial_weights: str | None = field(
        default=None,
        metadata={"help": "Optional filepath to initial EDM2 weights/state instead of random initialization."},
    )
    optimizer: EDM2OptimizerConfig = field(default_factory=EDM2OptimizerConfig)
    importance: EDM2ImportanceConfig = field(default_factory=EDM2ImportanceConfig)
    visualization: EDM2VisualizationConfig = field(default_factory=EDM2VisualizationConfig)


@dataclass
class LossConfig:
    """Loss configuration with organized subcategories."""

    # Core loss settings
    loss_type: str = field(default="l2", metadata={"help": "Loss function type: l2, l1, huber, smooth_l1, log_cosh, etc."})
    loss_scale: float = field(default=1.0, metadata={"help": "Multiplier applied to the computed loss"})
    loss_multiplier: float | None = field(default=None, metadata={"help": "Alternative loss multiplier (deprecated, use loss_scale)"})
    prior_loss_weight: float = field(default=1.0, metadata={"help": "Weight for prior preservation loss in DreamBooth training"})
    v_parameterization: bool = field(
        default=False,
        metadata={"help": "Legacy compatibility mirror for objective.prediction == v_prediction"},
    )

    # Nested subcategories
    huber: HuberConfig = field(default_factory=HuberConfig)
    snr: SNRConfig = field(default_factory=SNRConfig)
    masked: MaskedLossConfig = field(default_factory=MaskedLossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    edm2: EDM2Config = field(default_factory=EDM2Config)
