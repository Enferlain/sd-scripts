"""
Unit tests for configuration dataclasses.

Tests that all config dataclasses properly instantiate, have correct defaults,
and work with Hydra composition from YAML files.
"""

import os

import pytest
from hydra import compose
from omegaconf import OmegaConf

from library.adapters.methods.peft.locon.config import PeftLoconConfig
from library.adapters.methods.peft.lokr.config import PeftLokrConfig
from library.adapters.methods.peft.lora.config import PeftLoraConfig
from library.adapters.methods.peft.oft.config import PeftOftConfig
from library.config.dataclasses.adapter import AdapterConfig
from library.config.dataclasses.optimizer import OptimizerConfig
from library.config.dataclasses.data import DataConfig, BucketingConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.peft import PeftConfig
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.objective import ObjectiveConfig
from library.config.dataclasses.output import SavingConfig
from library.config.dataclasses.output import LoggingConfig
from library.config.dataclasses.output import MetadataConfig
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.loss import RegularizationConfig
from library.config.dataclasses.loss import LossConfig
from library.config.dataclasses.output import SamplingConfig
from library.config.dataclasses.timestep import TimestepConfig


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
        assert hasattr(config, "optimizer_type")
        assert hasattr(config, "learning_rates")
        assert hasattr(config.learning_rates, "base")

    def test_data_config_instantiation(self):
        """Test DataConfig instantiation with defaults."""
        config = DataConfig()
        assert config is not None
        assert hasattr(config, "source")
        assert hasattr(config, "preprocessing")
        assert hasattr(config, "bucketing")

    def test_training_config_instantiation(self):
        """Test TrainingConfig instantiation with defaults."""
        config = TrainingConfig()
        assert config is not None
        assert hasattr(config, "max_train_epochs")
        assert hasattr(config, "train_batch_size")

    def test_peft_config_instantiation(self):
        """Test PeftConfig instantiation with nullable method branches."""
        config = PeftConfig()
        assert config is not None
        assert hasattr(config, "lora")
        assert hasattr(config, "loha")
        assert hasattr(config, "locon")
        assert hasattr(config, "lokr")
        assert hasattr(config, "oft")
        assert config.lora is None
        assert config.loha is None
        assert config.locon is None
        assert config.lokr is None
        assert config.oft is None

    def test_adapter_config_instantiation(self):
        """Test AdapterConfig instantiation with defaults."""
        config = AdapterConfig()
        assert config is not None
        assert config.peft is None

    def test_peft_lora_branch_instantiation(self):
        """Test PeftConfig can select LoRA by branch presence."""
        config = PeftConfig(lora=PeftLoraConfig())
        assert hasattr(config.lora, "rank")
        assert hasattr(config.lora, "alpha")

    def test_peft_lokr_branch_instantiation(self):
        """Test PeftConfig can select LoKr by branch presence."""
        config = PeftConfig(lokr=PeftLokrConfig())
        assert hasattr(config.lokr, "rank")
        assert hasattr(config.lokr, "decompose_both")

    def test_peft_locon_branch_instantiation(self):
        """Test PeftConfig can select LoCon by branch presence."""
        config = PeftConfig(locon=PeftLoconConfig())
        assert hasattr(config.locon, "rank")
        assert hasattr(config.locon, "orthogonalize")

    def test_peft_oft_branch_instantiation(self):
        """Test PeftConfig can select OFT by branch presence."""
        config = PeftConfig(oft=PeftOftConfig())
        assert hasattr(config.oft, "factor")
        assert hasattr(config.oft, "constraint")

    def test_bucketing_config_instantiation(self):
        """Test BucketingConfig instantiation with defaults."""
        config = BucketingConfig()
        assert config is not None
        assert hasattr(config, "enable_bucket")
        assert hasattr(config, "min_bucket_reso")

    def test_model_config_instantiation(self):
        """Test ModelConfig instantiation with defaults."""
        config = ModelConfig()
        assert config is not None
        assert hasattr(config, "model_type")
        assert hasattr(config, "pretrained_model_name_or_path")

    def test_saving_config_instantiation(self):
        """Test SavingConfig instantiation with defaults."""
        config = SavingConfig()
        assert config is not None
        assert hasattr(config, "output_dir")
        assert hasattr(config, "save_model_as")

    def test_logging_config_instantiation(self):
        """Test LoggingConfig instantiation with defaults."""
        config = LoggingConfig()
        assert config is not None
        assert hasattr(config, "logging_dir")

    def test_metadata_config_instantiation(self):
        """Test MetadataConfig instantiation with defaults."""
        config = MetadataConfig()
        assert config is not None
        # Metadata fields are all optional

    def test_performance_config_instantiation(self):
        """Test PerformanceConfig instantiation with defaults."""
        config = PerformanceConfig()
        assert config is not None
        assert hasattr(config, "precision")  # Nested config
        assert hasattr(config.precision, "mixed_precision")

    def test_regularization_config_instantiation(self):
        """Test RegularizationConfig instantiation with defaults."""
        config = RegularizationConfig()
        assert config is not None
        # Regularization fields are optional

    def test_loss_config_instantiation(self):
        """Test LossConfig instantiation with defaults."""
        config = LossConfig()
        assert config is not None
        assert hasattr(config, "loss_type")

    def test_sampling_config_instantiation(self):
        """Test SamplingConfig instantiation with defaults."""
        config = SamplingConfig()
        assert config is not None
        assert hasattr(config, "sample_flow_shift")

    def test_objective_config_instantiation(self):
        """Test ObjectiveConfig instantiation with defaults."""
        config = ObjectiveConfig()
        assert config is not None
        assert hasattr(config, "path")
        assert hasattr(config, "prediction")

    def test_timestep_config_instantiation(self):
        """Test TimestepConfig instantiation with defaults."""
        config = TimestepConfig()
        assert config is not None
        assert hasattr(config, "timestep_sampling")  # Field is 'timestep_sampling', not 'timestep_sampler'
        assert hasattr(config, "rf_loss_weighting_scheme")


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
        assert config.method is None
        assert config.lora is None
        assert config.loha is None
        assert config.locon is None
        assert config.lokr is None
        assert config.orthograd_targets is not None
        assert config.orthograd_targets[0] == "lora_down.weight"

    def test_lora_config_owns_method_settings(self):
        """LoRA method settings should live only under the nested surface."""
        config = PeftConfig(lora=PeftLoraConfig())
        config.lora.rank = 16
        config.lora.alpha = 32.0
        config.lora.dropout = 0.25
        config.lora.conv_rank = 8

        assert config.lora.rank == 16
        assert config.lora.alpha == 32.0
        assert config.lora.dropout == 0.25
        assert config.lora.conv_rank == 8

    def test_bucketing_config_defaults(self):
        """Test BucketingConfig default values."""
        config = BucketingConfig()
        assert not config.enable_bucket
        assert config.min_bucket_reso == 256
        assert config.max_bucket_reso == 1024
        assert config.bucket_reso_steps == 64

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.train_batch_size == 1
        assert config.gradient_accumulation_steps == 1
        assert config.max_train_steps == 1600

    def test_sampling_config_defaults(self):
        """Test SamplingConfig default values."""
        config = SamplingConfig()
        assert config.sample_sampler == "ddim"
        assert config.sample_flow_shift is None

    def test_objective_config_defaults(self):
        """Test ObjectiveConfig default values."""
        config = ObjectiveConfig()
        assert config.path == "ddpm"
        assert config.prediction == "epsilon"

    def test_timestep_config_rf_defaults(self):
        """Test RF-related timestep config defaults."""
        config = TimestepConfig()
        assert config.timestep_sampling == "uniform"
        assert config.rf_loss_weighting_scheme == "uniform"
        assert config.logit_mean == 0.0
        assert config.logit_std == 1.0
        assert config.cosine_shape_scale == 1.29


# ============================================================================
# Hydra Composition Tests
# ============================================================================


@pytest.mark.config
@pytest.mark.integration
class TestHydraComposition:
    """Test that configs properly compose from YAML files using Hydra."""

    def test_all_active_entry_configs_compose(self, hydra_ctx):
        """Every active entry config should compose against the current schema."""
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))
        config_names = []
        for root, _, filenames in os.walk(config_dir):
            rel_root = os.path.relpath(root, config_dir)
            if rel_root.startswith("_defaults"):
                continue

            for filename in filenames:
                if not filename.endswith(".yaml"):
                    continue
                if rel_root == "examples":
                    continue

                rel_path = filename[:-5] if rel_root == "." else os.path.join(rel_root, filename[:-5])
                config_names.append(rel_path.replace(os.sep, "/"))

        config_names.sort()

        for config_name in config_names:
            cfg = compose(config_name=config_name)
            assert cfg is not None, f"{config_name} failed to compose"

    def test_sd_peft_config_composition(self, hydra_ctx):
        """Test sd_peft config loading via Hydra."""
        cfg = compose(config_name="presets/sd_peft")
        assert cfg is not None
        assert cfg.mode == "adapter"
        assert "adapter" in cfg
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "training" in cfg
        assert cfg.model.model_type == "sd15"
        assert cfg.adapter.peft.lora is not None
        assert cfg.adapter.peft.loha is None
        assert cfg.adapter.peft.locon is None
        assert cfg.adapter.peft.lokr is None
        assert cfg.adapter.peft.oft is None

    def test_internal_default_config_composition(self, hydra_ctx):
        """Test the internal full-schema default baseline."""
        cfg = compose(config_name="_defaults/default")
        assert cfg is not None
        assert OmegaConf.is_missing(cfg, "mode")
        assert cfg.model.model_type is None
        assert cfg.adapter.peft.lora is not None
        assert cfg.adapter.peft.loha is None
        assert cfg.adapter.peft.locon is None
        assert cfg.adapter.peft.lokr is None
        assert cfg.adapter.peft.oft is None
        assert cfg.adapter.peft.continue_from is None

    def test_sd_finetune_config_composition(self, hydra_ctx):
        """Test sd_finetune config loading via Hydra."""
        cfg = compose(config_name="presets/sd_finetune")
        assert cfg is not None
        assert cfg.mode == "finetune"
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "training" in cfg
        assert cfg.model.model_type == "sd15"

    def test_sd_textual_inversion_config_composition(self, hydra_ctx):
        """Test sd_textual_inversion config loading via Hydra."""
        cfg = compose(config_name="presets/sd_textual_inversion")
        assert cfg is not None
        assert cfg.mode == "textual_inversion"
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "training" in cfg
        assert cfg.model.model_type == "sd15"

    def test_sdxl_peft_config_composition(self, hydra_ctx):
        """Test sdxl_peft config loading via Hydra."""
        cfg = compose(config_name="presets/sdxl_peft")
        assert cfg is not None
        assert cfg.mode == "adapter"
        assert "adapter" in cfg
        assert "optimizer" in cfg
        assert "sdxl" not in cfg
        assert cfg.model.model_type == "sdxl"
        assert cfg.adapter.peft.lora is not None
        assert cfg.adapter.peft.loha is None
        assert cfg.adapter.peft.locon is None
        assert cfg.adapter.peft.lokr is None
        assert cfg.adapter.peft.oft is None

    def test_sdxl_peft_edm2_preset_composition(self, hydra_ctx):
        """Test the dedicated SDXL PEFT EDM2 preset."""
        cfg = compose(config_name="presets/sdxl_peft_edm2")
        assert cfg is not None
        assert cfg.mode == "adapter"
        assert cfg.model.model_type == "sdxl"
        assert cfg.loss.edm2.enabled is True
        assert cfg.loss.edm2.importance.enabled is True
        assert cfg.loss.edm2.optimizer.use_scheduler is True
        assert cfg.loss.edm2.visualization.enabled is True
        assert cfg.output.saving.output_name == "sdxl_peft_edm2"

    def test_sdxl_peft_edm2_example_composition(self, hydra_ctx):
        """Test the runnable EDM2 example config."""
        cfg = compose(config_name="examples/edm2_sdxl_peft")
        assert cfg is not None
        assert cfg.mode == "adapter"
        assert cfg.model.model_type == "sdxl"
        assert cfg.loss.edm2.enabled is True
        assert cfg.loss.edm2.importance.enabled is True
        assert cfg.loss.snr.min_snr_gamma is None
        assert cfg.output.saving.output_name == "sdxl_peft_edm2"

    def test_sdxl_peft_adaptive_log_snr_test_config_composition(self, hydra_ctx):
        """Test the runnable adaptive_log_snr test config."""
        cfg = compose(config_name="tests/test_adaptive_log_snr_sdxl_peft")
        assert cfg is not None
        assert cfg.mode == "adapter"
        assert cfg.model.model_type == "sdxl"
        assert cfg.timestep.timestep_sampling == "adaptive_log_snr"
        assert cfg.timestep.adaptive_log_snr.prior_weight == 0.25
        assert cfg.output.saving.output_name == "test_adaptive_log_snr_sdxl_peft"

    def test_sdxl_peft_log_snr_uniform_test_config_composition(self, hydra_ctx):
        """Test the runnable log_snr_uniform test config."""
        cfg = compose(config_name="tests/test_log_snr_uniform_sdxl_peft")
        assert cfg is not None
        assert cfg.mode == "adapter"
        assert cfg.model.model_type == "sdxl"
        assert cfg.timestep.timestep_sampling == "log_snr_uniform"
        assert cfg.output.saving.output_name == "test_log_snr_uniform_sdxl_peft"

    def test_sdxl_finetune_config_composition(self, hydra_ctx):
        """Test sdxl_finetune config loading via Hydra."""
        cfg = compose(config_name="presets/sdxl_finetune")
        assert cfg is not None
        assert cfg.mode == "finetune"
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "sdxl" not in cfg
        assert cfg.model.model_type == "sdxl"

    def test_sdxl_textual_inversion_config_composition(self, hydra_ctx):
        """Test sdxl_textual_inversion config loading via Hydra."""
        cfg = compose(config_name="presets/sdxl_textual_inversion")
        assert cfg is not None
        assert cfg.mode == "textual_inversion"
        assert "optimizer" in cfg
        assert "data" in cfg
        assert "sdxl" not in cfg
        assert cfg.model.model_type == "sdxl"


# ============================================================================
# Config Override Tests
# ============================================================================


@pytest.mark.config
@pytest.mark.unit
class TestConfigOverrides:
    """Test that config values can be properly overridden."""

    def test_optimizer_override(self, hydra_ctx):
        """Test overriding optimizer config values."""
        cfg = compose(config_name="presets/sd_peft", overrides=["optimizer.learning_rates.base=5e-5", "optimizer.optimizer_type=AdamW"])
        assert cfg.optimizer.learning_rates.base == 5e-5
        assert cfg.optimizer.optimizer_type == "AdamW"

    def test_nested_lora_override(self, hydra_ctx):
        """Test overriding the nested LoRA config surface."""
        cfg = compose(config_name="presets/sd_peft", overrides=["adapter.peft.lora.rank=128", "adapter.peft.lora.alpha=128"])
        assert cfg.adapter.peft.lora.rank == 128
        assert cfg.adapter.peft.lora.alpha == 128

    def test_training_override(self, hydra_ctx):
        """Test overriding training config values."""
        cfg = compose(config_name="presets/sd_peft", overrides=["training.max_train_epochs=20", "training.train_batch_size=4"])
        assert cfg.training.max_train_epochs == 20
        assert cfg.training.train_batch_size == 4
