from dataclasses import dataclass, field
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.optimizer import OptimizerConfig
from library.config.dataclasses.data import DataConfig
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.peft import PeftConfig
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.loss import LossConfig
from library.config.dataclasses.timestep import TimestepConfig
from library.config.dataclasses.output import OutputConfig
from library.config.dataclasses.validation import ValidationConfig


@dataclass
class SDPeftConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
