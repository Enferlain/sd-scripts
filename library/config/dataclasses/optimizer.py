from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class OptimizerConfig:
    optimizer_type: str = field(default="", metadata={"help": "Optimizer to use"})
    use_8bit_adam: bool = field(default=False, metadata={"help": "use 8bit AdamW optimizer"})
    use_lion_optimizer: bool = field(default=False, metadata={"help": "use Lion optimizer"})
    learning_rate: float = field(default=2.0e-6, metadata={"help": "learning rate"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm"})
    optimizer_args: List[str] = field(default_factory=list, metadata={"help": "additional arguments for optimizer"})
    lr_scheduler_type: str = field(default="", metadata={"help": "custom scheduler module"})
    lr_scheduler_args: List[str] = field(default_factory=list, metadata={"help": "additional arguments for scheduler"})
    lr_scheduler: str = field(default="constant", metadata={"help": "scheduler to use for learning rate"})
    lr_warmup_steps: Union[int, float] = field(default=0, metadata={"help": "Number of steps for the warmup in the lr scheduler"})
    lr_decay_steps: Union[int, float] = field(default=0, metadata={"help": "Number of steps for the decay in the lr scheduler"})
    lr_scheduler_num_cycles: int = field(default=1, metadata={"help": "Number of restarts for cosine scheduler with restarts"})
    lr_scheduler_power: float = field(default=1.0, metadata={"help": "Polynomial power for polynomial scheduler"})
    fused_backward_pass: bool = field(default=False, metadata={"help": "Combines backward pass and optimizer step to reduce VRAM usage"})
    lr_scheduler_timescale: Optional[int] = field(default=None, metadata={"help": "Inverse sqrt timescale for inverse sqrt scheduler"})
    lr_scheduler_min_lr_ratio: Optional[float] = field(default=None, metadata={"help": "The minimum learning rate as a ratio of the initial learning rate"})

    def __post_init__(self):
        if self.use_8bit_adam:
            self.optimizer_type = "AdamW8bit"
        if self.use_lion_optimizer:
            self.optimizer_type = "Lion"
