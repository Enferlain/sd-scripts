from dataclasses import dataclass, field
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.optimizer import OptimizerConfig
from library.config.dataclasses.data import DataConfig
from library.config.dataclasses.peft import PeftConfig
from library.config.dataclasses.sdxl import SDXLConfig
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.loss import LossConfig
from library.config.dataclasses.timestep import TimestepConfig
from library.config.dataclasses.output import OutputConfig
from library.config.dataclasses.validation import ValidationConfig


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
