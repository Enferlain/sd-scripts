from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SavingConfig:
    output_dir: Optional[str] = None
    output_name: Optional[str] = None
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
    resume_from_huggingface: bool = False
    huggingface_repo_id: Optional[str] = None
    huggingface_token: Optional[str] = None
    huggingface_repo_type: Optional[str] = None
    save_model_as: Optional[str] = None
    use_safetensors: bool = False
