from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Core training loop settings."""
    train_batch_size: int = field(default=1, metadata={"help": "Number of images per training step per device"})
    max_token_length: Optional[int] = field(default=None, metadata={"help": "Maximum caption token length (75, 150, or 225; None uses model default)"})
    max_train_steps: int = field(default=1600, metadata={"help": "Maximum number of training steps (overridden if max_train_epochs is set)"})
    max_train_epochs: Optional[int] = field(default=None, metadata={"help": "Maximum training epochs (overrides max_train_steps if set)"})
    seed: Optional[int] = field(default=None, metadata={"help": "Random seed for reproducibility (None for non-deterministic)"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Accumulate gradients over N steps before optimizer update"})
    clip_skip: Optional[int] = field(default=None, metadata={"help": "Skip last N CLIP layers (None uses full output, 2 common for anime)"})
    dry_run: bool = field(default=False, metadata={"help": "Run a dry run of the training process"})
    initial_epoch: Optional[int] = field(default=None, metadata={"help": "Initial epoch number when resuming training"})
    initial_step: Optional[int] = field(default=None, metadata={"help": "Initial step number when resuming (includes all epochs)"})
    skip_until_initial_step: bool = field(default=False, metadata={"help": "Skip training until initial_step is reached"})
