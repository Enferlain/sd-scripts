from dataclasses import dataclass, field
from .model import ModelConfig
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .data import DataConfig
from .peft import PeftConfig
from .sdxl import SDXLConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .timestep import TimestepConfig
from .output import OutputConfig
from .validation import ValidationConfig


@dataclass
class SDXLPeftConfig:
    """Root configuration for SDXL PEFT/LoRA training.
    
    This is the main config used by scripts/sdxl_peft.py.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    sdxl: SDXLConfig = field(default_factory=SDXLConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
