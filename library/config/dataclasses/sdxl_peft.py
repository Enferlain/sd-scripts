from dataclasses import dataclass, field
from .sd_models import ModelLoadingConfig
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .dataset import DatasetConfig
from .network import NetworkConfig
from .sdxl_training import SDXLTrainingConfig
from .saving import SavingConfig
from .logging import LoggingConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .regularization import RegularizationConfig
from .timestep import TimestepConfig
from .sampling import SamplingConfig
from .masked_loss import MaskedLossConfig
from .metadata import MetadataConfig
from .huggingface import HuggingFaceConfig


@dataclass
class SDXLTrainNetworkConfig:
    """Root configuration for SDXL PEFT/LoRA training.
    
    This is the main config used by scripts/sdxl_peft.py.
    """
    sd_models: ModelLoadingConfig = field(default_factory=ModelLoadingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sdxl_training: SDXLTrainingConfig = field(default_factory=SDXLTrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    saving: SavingConfig = field(default_factory=SavingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    masked_loss: MaskedLossConfig = field(default_factory=MaskedLossConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
