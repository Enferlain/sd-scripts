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
    log_timestep_distribution_every_n_steps: Optional[int] = field(default=None, metadata={"help": "Saves a snapshot of the timestep distribution chart every N steps."})
    live_plot_port: Optional[int] = field(default=None, metadata={"help": "Launches the live interactive dashboard server on this port."})
