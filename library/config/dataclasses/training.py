from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    train_batch_size: int = 1
    max_token_length: Optional[int] = None
    vae: Optional[str] = None
    max_train_steps: int = 1600
    max_train_epochs: Optional[int] = None
    max_data_loader_n_workers: int = 8
    persistent_data_loader_workers: bool = False
    seed: Optional[int] = None
    gradient_accumulation_steps: int = 1
    clip_skip: Optional[int] = None
    config_file: Optional[str] = None
    output_config: bool = False
