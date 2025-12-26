from dataclasses import dataclass, field
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .dataset import DatasetConfig
from .buckets import BucketsConfig
from .model import ModelConfig
from .peft import PeftConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .regularization import RegularizationConfig
from .timestep import TimestepConfig
from .masked_loss import MaskedLossConfig
from .output import OutputConfig


@dataclass
class SDPeftConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    buckets: BucketsConfig = field(default_factory=BucketsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    masked_loss: MaskedLossConfig = field(default_factory=MaskedLossConfig)
