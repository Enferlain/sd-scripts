from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HuberConfig:
    """Huber loss settings."""
    huber_schedule: str = field(default="snr", metadata={"help": "Huber threshold schedule: constant, exponential, or snr"})
    huber_c: float = field(default=0.1, metadata={"help": "Base Huber threshold value"})
    huber_scale: float = field(default=1.0, metadata={"help": "Scale factor applied to Huber threshold"})


@dataclass
class SNRConfig:
    """SNR-based loss weighting settings."""
    min_snr_gamma: Optional[float] = field(default=None, metadata={"help": "Min-SNR gamma for loss weighting (None to disable)"})
    scale_v_pred_loss_like_noise_pred: bool = field(default=False, metadata={"help": "Scale v-prediction loss like noise prediction"})
    v_pred_like_loss: Optional[float] = field(default=None, metadata={"help": "Add v-prediction-like loss term (multiplier, None to disable)"})
    debiased_estimation_loss: bool = field(default=False, metadata={"help": "Apply debiased estimation loss weighting"})


@dataclass
class MaskedLossConfig:
    """Masked loss settings."""
    masked_loss: bool = field(default=False, metadata={"help": "Apply mask for calculating loss"})
    conditioning_data_dir: Optional[str] = field(default=None, metadata={"help": "Directory containing conditioning/mask images"})


@dataclass
class RegularizationConfig:
    """Noise regularization settings."""
    noise_offset: Optional[float] = field(default=None, metadata={"help": "Noise offset for improving dark/light image handling"})
    noise_offset_random_strength: bool = field(default=False, metadata={"help": "Randomize noise offset strength between 0 and noise_offset"})
    multires_noise_iterations: Optional[int] = field(default=None, metadata={"help": "Pyramid noise iterations (enables multires noise if set)"})
    multires_noise_discount: float = field(default=0.3, metadata={"help": "Discount factor per pyramid level for multires noise"})
    ip_noise_gamma: Optional[float] = field(default=None, metadata={"help": "Input perturbation noise gamma (None to disable)"})
    ip_noise_gamma_random_strength: bool = field(default=False, metadata={"help": "Randomize input perturbation noise strength"})
    adaptive_noise_scale: Optional[float] = field(default=None, metadata={"help": "Scale noise offset adaptively based on latent statistics"})
    zero_terminal_snr: bool = field(default=False, metadata={"help": "Use zero terminal SNR noise scheduling"})


@dataclass
class EDM2Config:
    """EDM2 loss weighting settings."""
    edm2_loss_weighting: bool = field(default=False, metadata={"help": "Use EDM2 loss weighting."})
    edm2_loss_weighting_laplace: bool = field(default=False, metadata={"help": "Use EDM2 loss weighting to calculate timestep sampling using laplace."})
    edm2_loss_weighting_optimizer: str = field(default="torch.optim.AdamW", metadata={"help": "Fully qualified optimizer class name to use with the edm2 loss weighting optimizer."})
    edm2_loss_weighting_optimizer_lr: float = field(default=2e-2, metadata={"help": "Learning rate as a float for the edm2 loss weighting optimizer."})
    edm2_loss_weighting_optimizer_args: str = field(default="{'weight_decay': 0, 'betas': (0.9,0.999)}", metadata={"help": "A JSON object as a string of optimizer args for the edm2 loss weighting optimizer."})
    edm2_loss_weighting_lr_scheduler: bool = field(default=False, metadata={"help": "Use lr scheduler with EDM2 loss weighting optimizer."})
    edm2_loss_weighting_lr_scheduler_warmup_percent: float = field(default=0.1, metadata={"help": "Percent of training steps to use for warmup."})
    edm2_loss_weighting_lr_scheduler_constant_percent: float = field(default=0.1, metadata={"help": "Percent of training steps to maintain constant LR before decay."})
    edm2_loss_weighting_generate_graph: bool = field(default=False, metadata={"help": "Enable generation of graph images that show the loss weighting per timestep."})
    edm2_loss_weighting_generate_graph_every_x_steps: int = field(default=20, metadata={"help": "Every x steps generate a graph image."})
    edm2_loss_weighting_generate_graph_output_dir: Optional[str] = field(default=None, metadata={"help": "The parent directory where loss weighting graph images should be stored"})
    edm2_loss_weighting_generate_graph_y_limit: Optional[int] = field(default=None, metadata={"help": "Set the max limit of the y axis"})
    edm2_loss_weighting_generate_graph_y_scale: str = field(default="linear", metadata={"help": "Select between linear or log scaling for the y-axis."})
    edm2_loss_weighting_num_channels: int = field(default=128, metadata={"help": "The number of channels used by for the loss weighting module."})
    edm2_loss_weighting_initial_weights: Optional[str] = field(default=None, metadata={"help": "The full filepath to initial weights and state of edm2 weighting model to use instead of random."})
    edm2_loss_weighting_lr_scheduler_decay_scaling: float = field(default=1.0, metadata={"help": "A scaling factor to apply to the decay rate of the edm2_loss_weighting_lr_scheduler"})
    edm2_loss_weighting_importance_weighting: bool = field(default=False, metadata={"help": "If edm2 loss scaling weights are weighted by importance"})
    edm2_loss_weighting_importance_weighting_max: float = field(default=10.0, metadata={"help": "The max loss weighting/scaling to apply when using edm2 importance weighting"})
    edm2_loss_weighting_importance_min_snr_gamma: float = field(default=1.0, metadata={"help": "The min snr gamma used for edm2 importance weighting as a heuristic"})
    edm2_loss_weighting_importance_weighting_safety_override: bool = field(default=False, metadata={"help": "At your own risk, you may set this to true to ALLOW stacking debiased loss and/or typical min snr gamma with EDM2 using importance weighting."})


@dataclass
class LossConfig:
    """Loss configuration with organized subcategories."""
    # Core loss settings
    loss_type: str = field(default="l2", metadata={"help": "Loss function type: l2, l1, huber, smooth_l1, log_cosh, etc."})
    loss_scale: float = field(default=1.0, metadata={"help": "Multiplier applied to the computed loss"})
    loss_multiplier: Optional[float] = field(default=None, metadata={"help": "Alternative loss multiplier (deprecated, use loss_scale)"})
    prior_loss_weight: float = field(default=1.0, metadata={"help": "Weight for prior preservation loss in DreamBooth training"})
    v_parameterization: bool = field(default=False, metadata={"help": "Enable v-parameterization training"})
    
    # Nested subcategories
    huber: HuberConfig = field(default_factory=HuberConfig)
    snr: SNRConfig = field(default_factory=SNRConfig)
    masked: MaskedLossConfig = field(default_factory=MaskedLossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    edm2: EDM2Config = field(default_factory=EDM2Config)
