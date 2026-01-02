from dataclasses import dataclass, field
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .data import DataConfig
from .model import ModelConfig
from .sdxl import SDXLConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .timestep import TimestepConfig
from .output import OutputConfig
from .validation import ValidationConfig


@dataclass
class SDXLTextualInversionSpecificConfig:
    """SDXL Textual Inversion specific configuration."""
    weights: str | None = field(default=None, metadata={"help": "Path to existing embeddings file to continue training from"})
    num_vectors_per_token: int = field(default=1, metadata={"help": "Number of vectors per token (1 for simple, higher for complex concepts)"})
    token_string: str | None = field(default=None, metadata={"help": "Trigger word/token for the trained embedding"})
    init_word: str | None = field(default=None, metadata={"help": "Initialize embedding from this word's vectors"})
    use_object_template: bool = field(default=False, metadata={"help": "Use object-style caption templates for training"})
    use_style_template: bool = field(default=False, metadata={"help": "Use style-focused caption templates for training"})


@dataclass
class SDXLTextualInversionConfig:
    """Root configuration for SDXL textual inversion training.
    
    This is the main config used by scripts/sdxl_textual_inversion.py.
    """
    textual_inversion: SDXLTextualInversionSpecificConfig = field(default_factory=SDXLTextualInversionSpecificConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sdxl: SDXLConfig = field(default_factory=SDXLConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
