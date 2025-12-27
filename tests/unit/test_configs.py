"""
Unit tests for configuration dataclasses.

Tests that all config dataclasses properly instantiate, have correct defaults,
and work with Hydra composition from YAML files.
"""

import pytest
from hydra import compose
from omegaconf import OmegaConf

from library.config.dataclasses.optimizer import OptimizerConfig
from library.config.dataclasses.data import DataConfig, SourceConfig, PreprocessingConfig, BucketingConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.peft import PeftConfig
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.output import SavingConfig
from library.config.dataclasses.output import LoggingConfig
from library.config.dataclasses.output import MetadataConfig
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.loss import RegularizationConfig
from library.config.dataclasses.loss import LossConfig
from library.config.dataclasses.output import SamplingConfig
from library.config.dataclasses.timestep import TimestepConfig
from library.config.dataclasses.sdxl import SDXLConfig


# ============================================================================
# Basic Config Instantiation Tests
# ============================================================================

@pytest.mark.config
@pytest.mark.unit
class TestConfigInstantiation:
    """Test that all config dataclasses can be instantiated with defaults."""
    
    def test_optimizer_config_instantiation(self):
        """Test OptimizerConfig instantiation with defaults."""
        config = OptimizerConfig()
        assert config is not None
        assert hasattr(config, 'optimizer_type')
        assert hasattr(config, 'learning_rates')
        assert hasattr(config.learning_rates, 'base')
        
    def test_data_config_instantiation(self):
        """Test DataConfig instantiation with defaults."""
        config = DataConfig()
        assert config is not None
        assert hasattr(config, 'source')
        assert hasattr(config, 'preprocessing')
        assert hasattr(config, 'bucketing')
        
    def test_training_config_instantiation(self):
        """Test TrainingConfig instantiation with defaults."""
        config = TrainingConfig()
        assert config is not None
        assert hasattr(config, 'max_train_epochs')
        assert hasattr(config, 'train_batch_size')
        
    def test_adapter_config_instantiation(self):
        """Test PeftConfig instantiation with defaults."""
        config = PeftConfig()
        assert config is not None
        assert hasattr(config, 'module')
        assert hasattr(config, 'adapter_rank')
        assert hasattr(config, 'adapter_alpha')
        
    def test_bucketing_config_instantiation(self):
        """Test BucketingConfig instantiation with defaults."""
        config = BucketingConfig()
        assert config is not None
        assert hasattr(config, 'enable_bucket')
        assert hasattr(config, 'min_bucket_reso')
        
    def test_model_config_instantiation(self):
        """Test ModelConfig instantiation with defaults."""
        config = ModelConfig()
        assert config is not None
        assert hasattr(config, 'pretrained_model_name_or_path')
        
    def test_saving_config_instantiation(self):
        """Test SavingConfig instantiation with defaults."""
        config = SavingConfig()
        assert config is not None
        assert hasattr(config, 'output_dir')
        assert hasattr(config, 'save_model_as')
        
    def test_logging_config_instantiation(self):
        """Test LoggingConfig instantiation with defaults."""
        config = LoggingConfig()
        assert config is not None
        assert hasattr(config, 'logging_dir')
        
    def test_metadata_config_instantiation(self):
        """Test MetadataConfig instantiation with defaults."""
        config = MetadataConfig()
        assert config is not None
        # Metadata fields are all optional
        
    def test_performance_config_instantiation(self):
        """Test PerformanceConfig instantiation with defaults."""
        config = PerformanceConfig()
        assert config is not None
        assert hasattr(config, 'precision')  # Nested config
        assert hasattr(config.precision, 'mixed_precision')
        
    def test_regularization_config_instantiation(self):
        """Test RegularizationConfig instantiation with defaults."""
        config = RegularizationConfig()
        assert config is not None
        # Regularization fields are optional
        
    def test_loss_config_instantiation(self):
        """Test LossConfig instantiation with defaults."""
        config = LossConfig()
        assert config is not None
        assert hasattr(config, 'loss_type')
        
    def test_sampling_config_instantiation(self):
        """Test SamplingConfig instantiation with defaults."""
        config = SamplingConfig()
        assert config is not None
        # Sampling fields are optional
        
    def test_timestep_config_instantiation(self):
        """Test TimestepConfig instantiation with defaults."""
        config = TimestepConfig()
        assert config is not None
        assert hasattr(config, 'timestep_sampling')  # Field is 'timestep_sampling', not 'timestep_sampler'
        
    def test_sdxl_config_instantiation(self):
        """Test SDXLConfig instantiation with defaults."""
        config = SDXLConfig()
        assert config is not None
        # SDXL-specific fields


# ============================================================================
# Config Values Tests
# ============================================================================

@pytest.mark.config
@pytest.mark.unit
class TestConfigDefaults:
    """Test that config dataclasses have expected default values."""
    
    def test_optimizer_config_defaults(self):
        """Test OptimizerConfig default values."""
        config = OptimizerConfig()
        assert config.optimizer_type == ""  # Empty by default
        assert config.learning_rates.base == 2.0e-6
        assert config.scheduler.lr_scheduler == "constant"
        
    def test_adapter_config_defaults(self):
        """Test PeftConfig default values."""
        config = PeftConfig()
        assert config.adapter_rank is None  # None by default
        assert config.adapter_alpha == 1.0
        assert config.module is None  # None by default
        
    def test_bucketing_config_defaults(self):
        """Test BucketingConfig default values."""
        config = BucketingConfig()
        assert config.enable_bucket == False
        assert config.min_bucket_reso == 256
        assert config.max_bucket_reso == 1024
        assert config.bucket_reso_steps == 64
        
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.train_batch_size == 1
        assert config.gradient_accumulation_steps == 1
        assert config.max_train_steps == 1600


# ============================================================================
# Hydra Composition Tests
# ============================================================================

@pytest.mark.config
@pytest.mark.integration
class TestHydraComposition:
    """Test that configs properly compose from YAML files using Hydra."""
    
    def test_sd_peft_config_composition(self, hydra_ctx):
        """Test sd_peft config loading via Hydra."""
        cfg = compose(config_name="sd_peft")
        assert cfg is not None
        assert "peft" in cfg
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "training" in cfg
        
    def test_sd_finetune_config_composition(self, hydra_ctx):
        """Test sd_finetune config loading via Hydra."""
        cfg = compose(config_name="sd_finetune")
        assert cfg is not None
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "training" in cfg
        
    def test_sd_textual_inversion_config_composition(self, hydra_ctx):
        """Test sd_textual_inversion config loading via Hydra."""
        cfg = compose(config_name="sd_textual_inversion")
        assert cfg is not None
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "training" in cfg
        
    def test_sdxl_peft_config_composition(self, hydra_ctx):
        """Test sdxl_peft config loading via Hydra."""
        cfg = compose(config_name="sdxl_peft")
        assert cfg is not None
        assert "peft" in cfg
        assert "optimizer" in cfg
        assert "sdxl" in cfg
        
    def test_sdxl_finetune_config_composition(self, hydra_ctx):
        """Test sdxl_finetune config loading via Hydra."""
        cfg = compose(config_name="sdxl_finetune")
        assert cfg is not None
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "sdxl" in cfg
        
    def test_sdxl_textual_inversion_config_composition(self, hydra_ctx):
        """Test sdxl_textual_inversion config loading via Hydra."""
        cfg = compose(config_name="sdxl_textual_inversion")
        assert cfg is not None
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "sdxl" in cfg


# ============================================================================
# Config Override Tests
# ============================================================================

@pytest.mark.config
@pytest.mark.unit
class TestConfigOverrides:
    """Test that config values can be properly overridden."""
    
    def test_optimizer_override(self, hydra_ctx):
        """Test overriding optimizer config values."""
        cfg = compose(
            config_name="sd_peft",
            overrides=["optimizer.learning_rates.base=5e-5", "optimizer.optimizer_type=AdamW"]
        )
        assert cfg.optimizer.learning_rates.base == 5e-5
        assert cfg.optimizer.optimizer_type == "AdamW"
        
    def test_network_override(self, hydra_ctx):
        """Test overriding peft config values."""
        cfg = compose(
            config_name="sd_peft",
            overrides=["peft.adapter_rank=128", "peft.adapter_alpha=128"]
        )
        assert cfg.peft.adapter_rank == 128
        assert cfg.peft.adapter_alpha == 128
        
    def test_training_override(self, hydra_ctx):
        """Test overriding training config values."""
        cfg = compose(
            config_name="sd_peft",
            overrides=["training.max_train_epochs=20", "training.train_batch_size=4"]
        )
        assert cfg.training.max_train_epochs == 20
        assert cfg.training.train_batch_size == 4
