from dataclasses import dataclass, field
from typing import Optional, List, Union, Any


@dataclass
class LearningRatesConfig:
    """
    Consolidated learning rates for all components.
    Component-specific LRs (unet, text_encoders) override base when set.
    """
    base: float = field(default=2.0e-6, metadata={"help": "Base learning rate, used as fallback for all components"})
    unet: Optional[float] = field(default=None, metadata={"help": "UNet LR (overrides base if set)"})
    # Supports single float or list of floats for multiple text encoders
    # NOTE: Type is Any due to OmegaConf limitation (Union of primitives and containers not supported).
    text_encoders: Optional[Any] = field(default=None, metadata={"help": "Text Encoder LR(s) (overrides base if set)"})
    # Block-wise LR weights/values
    blocks: Optional[str] = field(default=None, metadata={"help": "Per-block learning rates/weights"})


@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration.
    """
    lr_scheduler: str = field(default="constant", metadata={"help": "scheduler to use for learning rate"})
    lr_scheduler_type: str = field(default="", metadata={"help": "custom scheduler module"})
    lr_scheduler_args: List[str] = field(default_factory=list, metadata={"help": "additional arguments for scheduler"})
    lr_warmup_steps: Union[int, float] = field(default=0, metadata={"help": "Number of steps for the warmup in the lr scheduler"})
    lr_decay_steps: Union[int, float] = field(default=0, metadata={"help": "Number of steps for the decay in the lr scheduler"})
    lr_scheduler_num_cycles: int = field(default=1, metadata={"help": "Number of restarts for cosine scheduler with restarts"})
    lr_scheduler_power: float = field(default=1.0, metadata={"help": "Polynomial power for polynomial scheduler"})
    lr_scheduler_timescale: Optional[int] = field(default=None, metadata={"help": "Inverse sqrt timescale for inverse sqrt scheduler"})
    lr_scheduler_min_lr_ratio: Optional[float] = field(default=None, metadata={"help": "The minimum learning rate as a ratio of the initial learning rate"})


@dataclass
class OptimizerConfig:
    optimizer_type: str = field(default="", metadata={"help": "Optimizer to use"})
    use_8bit_adam: bool = field(default=False, metadata={"help": "use 8bit AdamW optimizer"})
    use_lion_optimizer: bool = field(default=False, metadata={"help": "use Lion optimizer"})
    
    # Structured learning rates container
    learning_rates: LearningRatesConfig = field(default_factory=LearningRatesConfig, metadata={"help": "Structured learning rates"})
    
    # Scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig, metadata={"help": "LR scheduler settings"})
    
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm"})
    optimizer_args: List[str] = field(default_factory=list, metadata={"help": "additional arguments for optimizer"})
    fused_backward_pass: bool = field(default=False, metadata={"help": "Combines backward pass and optimizer step to reduce VRAM usage"})
    optimizer_schedulefree_wrapper: bool = field(default=False, metadata={"help": "Wrap optimizer with ScheduleFreeWrapper"})
    schedulefree_wrapper_args: Optional[List[str]] = field(default=None, metadata={"help": "Arguments for ScheduleFreeWrapper"})
    # Fused backward pass with multiple optimizer groups (moved from SDXLConfig)
    fused_optimizer_groups: Optional[int] = field(default=None, metadata={"help": "number of optimizers for fused backward pass and optimizer step"})
