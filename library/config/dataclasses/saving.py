from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SavingConfig:
    save_precision: Optional[str] = None
    save_every_n_epochs: Optional[int] = None
    save_every_n_steps: Optional[int] = None
    save_n_epoch_ratio: Optional[int] = None
    save_last_n_epochs: Optional[int] = None
    save_last_n_epochs_state: Optional[int] = None
    save_last_n_steps: Optional[int] = None
    save_last_n_steps_state: Optional[int] = None
    save_state: bool = False
    save_state_on_train_end: bool = False
    resume: Optional[str] = None
    save_model_as: Optional[str] = None
    use_safetensors: bool = False
