from dataclasses import dataclass, field

from omegaconf import MISSING

from library.config.dataclasses.data import DataConfig
from library.config.dataclasses.loss import LossConfig
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.optimizer import OptimizerConfig
from library.config.dataclasses.output import OutputConfig
from library.config.dataclasses.peft import PeftConfig
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.textual_inversion import TextualInversionConfig
from library.config.dataclasses.timestep import TimestepConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.validation import ValidationConfig


@dataclass
class RunConfig:
    """Shared root configuration for training runs."""

    mode: str = field(default=MISSING, metadata={"help": "Training mode: finetune, peft, or textual_inversion"})
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfig | None = field(default=None)
    textual_inversion: TextualInversionConfig | None = field(default=None)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
