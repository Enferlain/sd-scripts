from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    train_batch_size: int = 1
    max_token_length: Optional[int] = None
    max_train_steps: int = 1600
    max_train_epochs: Optional[int] = None
    seed: Optional[int] = None
    gradient_accumulation_steps: int = 1
    clip_skip: Optional[int] = None
    dry_run: bool = field(default=False, metadata={"help": "Run a dry run of the training process"})
    initial_epoch: Optional[int] = field(default=None, metadata={"help": "initial epoch number"})
    initial_step: Optional[int] = field(default=None, metadata={"help": "initial step number including all epochs"})
    skip_until_initial_step: bool = field(default=False, metadata={"help": "skip training until initial_step is reached"})
