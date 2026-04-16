from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearningRatesConfig:
    """
    Consolidated learning rates for all components.

    `base` is the shared fallback learning rate. Setting it to `None` means
    there is no fallback baseline LR. Component-specific LRs override that
    fallback when set: `None` means inherit from base when available, `0`
    means keep that baseline path frozen, and positive values mean train.
    """

    base: float | None = field(
        default=2.0e-6,
        metadata={"help": "Shared fallback learning rate. Set to null to require explicit component/group learning rates"},
    )
    denoiser: float | None = field(
        default=None,
        metadata={"help": "Denoiser baseline LR. null inherits from base, 0 freezes the baseline denoiser path"},
    )
    # Supports single float or list of floats for multiple text encoders
    # NOTE: Type is Any due to OmegaConf limitation (Union of primitives and containers not supported).
    text_encoders: Any | None = field(
        default=None,
        metadata={"help": "Text Encoder baseline LR(s). null inherits from base, 0 freezes the baseline TE path"},
    )
    groups_file: str | None = field(default=None, metadata={"help": "Optional YAML file containing fine-grained named LR override groups"})
    groups: list["LearningRateGroupConfig"] = field(
        default_factory=list, metadata={"help": "Fine-grained named parameter-group LR overrides"}
    )


@dataclass
class LearningRateGroupConfig:
    """User-facing fine-grained LR override group."""

    name: str = field(default="", metadata={"help": "Display name for this parameter group"})
    lr: float = field(default=0.0, metadata={"help": "Learning rate override for matched parameters"})
    match: list[str] = field(
        default_factory=list, metadata={"help": "Glob patterns or re:<pattern> regexes matched against named parameters"}
    )


@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration.
    """

    lr_scheduler: str = field(default="constant", metadata={"help": "scheduler to use for learning rate"})
    lr_scheduler_type: str = field(default="", metadata={"help": "custom scheduler module"})
    lr_scheduler_args: list[str] = field(default_factory=list, metadata={"help": "additional arguments for scheduler"})
    lr_warmup_steps: int | float = field(default=0, metadata={"help": "Number of steps for the warmup in the lr scheduler"})
    lr_decay_steps: int | float = field(default=0, metadata={"help": "Number of steps for the decay in the lr scheduler"})
    lr_scheduler_num_cycles: int = field(default=1, metadata={"help": "Number of restarts for cosine scheduler with restarts"})
    lr_scheduler_power: float = field(default=1.0, metadata={"help": "Polynomial power for polynomial scheduler"})
    lr_scheduler_timescale: int | None = field(default=None, metadata={"help": "Inverse sqrt timescale for inverse sqrt scheduler"})
    lr_scheduler_min_lr_ratio: float | None = field(
        default=None, metadata={"help": "The minimum learning rate as a ratio of the initial learning rate"}
    )


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
    optimizer_args: list[str] = field(default_factory=list, metadata={"help": "additional arguments for optimizer"})
    fused_backward_pass: bool = field(default=False, metadata={"help": "Combines backward pass and optimizer step to reduce VRAM usage"})
    optimizer_schedulefree_wrapper: bool = field(default=False, metadata={"help": "Wrap optimizer with ScheduleFreeWrapper"})
    schedulefree_wrapper_args: list[str] | None = field(default=None, metadata={"help": "Arguments for ScheduleFreeWrapper"})
    # Fused backward pass with multiple optimizer groups (moved from SDXLConfig)
    fused_optimizer_groups: int | None = field(
        default=None, metadata={"help": "number of optimizers for fused backward pass and optimizer step"}
    )

    def __post_init__(self):
        """Handle _deprecated flags that should set optimizer_type."""
        # use_8bit_adam and use_lion_optimizer are _deprecated flags
        # If set and optimizer_type is not specified, set optimizer_type accordingly
        if self.use_8bit_adam and not self.optimizer_type:
            self.optimizer_type = "AdamW8bit"
        elif self.use_lion_optimizer and not self.optimizer_type:
            self.optimizer_type = "Lion"
