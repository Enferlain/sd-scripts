from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class LoggingConfig:
    logging_dir: Optional[str] = None
    log_with: Optional[str] = None
    log_prefix: Optional[str] = None
    log_tracker_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_tracker_config: Optional[Dict] = field(default_factory=dict)
    wandb_api_key: Optional[str] = None
    log_config: bool = False
    console_log_level: Optional[str] = None
    console_log_file: Optional[str] = None
    console_log_simple: bool = False
