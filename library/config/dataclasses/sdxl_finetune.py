from dataclasses import dataclass, field
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .dataset import DatasetConfig
from .sd_models import SDModelsConfig
from .sdxl_training import SDXLTrainingConfig
from .saving import SavingConfig
from .huggingface import HuggingFaceConfig
from .logging import LoggingConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .regularization import RegularizationConfig
from .timestep import TimestepConfig
from .sampling import SamplingConfig
from .masked_loss import MaskedLossConfig
from .buckets import BucketsConfig
from .metadata import MetadataConfig


@dataclass
class SDXLTrainConfig:
    """Root configuration for SDXL fine-tuning training.
    
    This is the main config used by scripts/sdxl_finetune.py.
    """
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sd_models: SDModelsConfig = field(default_factory=SDModelsConfig)
    sdxl_training: SDXLTrainingConfig = field(default_factory=SDXLTrainingConfig)
    saving: SavingConfig = field(default_factory=SavingConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    masked_loss: MaskedLossConfig = field(default_factory=MaskedLossConfig)
    buckets: BucketsConfig = field(default_factory=BucketsConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)


# Backwards compatibility alias - deprecated, will be removed
SDXLFineTuningConfig = SDXLTrainConfig
