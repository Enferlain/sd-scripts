from dataclasses import dataclass, field
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .dataset import DatasetConfig
from .sd_models import SDModelsConfig
from .sdxl_training import SDXLTrainingConfig
from .masked_loss import MaskedLossConfig

@dataclass
class MainConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sd_models: SDModelsConfig = field(default_factory=SDModelsConfig)
    sdxl_training: SDXLTrainingConfig = field(default_factory=SDXLTrainingConfig)
    masked_loss: MaskedLossConfig = field(default_factory=MaskedLossConfig)
