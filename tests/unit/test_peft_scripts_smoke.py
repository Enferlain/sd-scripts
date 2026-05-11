"""
Smoke Tests for PEFT entrypoints and related imports.

These tests verify that the active PEFT-facing entry surface and its
dependencies can be imported without errors. Run after refactoring to catch
import issues early.

Usage:
    pytest tests/unit/test_peft_scripts_smoke.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch


# Ensure scripts directory is importable
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestPeftCommonImports:
    """Test that peft utility functions can be imported from their new locations."""

    def test_peft_common_has_generate_step_logs(self):
        """Verify generate_step_logs function exists."""
        from library.logging.metrics import generate_step_logs

        assert callable(generate_step_logs)

    def test_peft_common_has_step_logging(self):
        """Verify step_logging function exists."""
        from library.logging.metrics import step_logging

        assert callable(step_logging)

    def test_peft_common_has_init_timestep_sampler(self):
        """Verify init_timestep_sampler function exists."""
        from library.timesteps.timestep_utils import init_timestep_sampler

        assert callable(init_timestep_sampler)

    def test_peft_common_has_create_training_metadata(self):
        """Verify create_training_metadata function exists."""
        from library.training.training_metadata import create_training_metadata

        assert callable(create_training_metadata)

    def test_peft_common_has_setup_live_plotter(self):
        """Verify setup_live_plotter function exists."""
        from library.logging.training_plots import setup_live_plotter

        assert callable(setup_live_plotter)

    @pytest.mark.skip(reason="Deprecated: prepare_datasets uses legacy data pipeline")
    def test_peft_common_has_prepare_datasets(self):
        """Verify prepare_datasets function exists."""
        from library.data._deprecated.dataset_setup import prepare_datasets

        assert callable(prepare_datasets)

    def test_peft_common_has_calculate_initial_step(self):
        """Verify calculate_initial_step function exists."""
        from library.training.trainer_utils import calculate_initial_step

        assert callable(calculate_initial_step)

    def test_peft_common_has_parse_dynamic_timestep_schedule(self):
        """Verify parse_dynamic_timestep_schedule function exists."""
        from library.timesteps.timestep_utils import parse_dynamic_timestep_schedule

        assert callable(parse_dynamic_timestep_schedule)

    def test_peft_common_has_register_network_state_hooks(self):
        """Verify register_adapter_state_hooks function exists."""
        from library.training.checkpointing import register_adapter_state_hooks

        assert callable(register_adapter_state_hooks)


class TestStrategyImports:
    """Test that strategy classes can be imported."""

    def test_sd_peft_strategy_imports(self):
        """Verify SdTrainingStrategy can be imported."""
        from library.strategies.sd.training import SdTrainingStrategy

        assert SdTrainingStrategy is not None

    def test_sdxl_peft_strategy_imports(self):
        """Verify SdxlTrainingStrategy can be imported."""
        from library.strategies.sdxl.training import SdxlTrainingStrategy

        assert SdxlTrainingStrategy is not None

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_sd_strategy_instantiation(self, mock_load_tokenizer):
        """Verify SdTrainingStrategy can be instantiated."""
        from library.strategies.sd.training import SdTrainingStrategy

        mock_load_tokenizer.return_value = Mock(model_max_length=77)
        cfg = Mock()
        cfg.model.model_type = "sd1"
        cfg.data.caching.tokenizer_cache_dir = None
        cfg.training.max_token_length = 75
        cfg.training.clip_skip = None

        strategy = SdTrainingStrategy(cfg)
        assert strategy is not None

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_sdxl_strategy_instantiation(self, mock_load_tokenizer):
        """Verify SdxlTrainingStrategy can be instantiated."""
        from library.strategies.sdxl.training import SdxlTrainingStrategy

        mock_load_tokenizer.side_effect = [Mock(model_max_length=77), Mock(model_max_length=77, pad_token_id=0)]
        cfg = Mock()
        cfg.training.max_token_length = 75
        cfg.data.caching.tokenizer_cache_dir = None

        strategy = SdxlTrainingStrategy(cfg)
        assert strategy is not None


class TestScriptImports:
    """Test that the active training entrypoints can be imported."""

    @pytest.mark.skip(reason="Deprecated: sd_peft.py uses legacy data pipeline")
    def test_sd_peft_script_imports(self):
        """Verify sd_peft.py can be imported without errors."""
        import sd_peft

        assert sd_peft is not None

    @pytest.mark.skip(reason="Deprecated: sd_peft.py uses legacy data pipeline")
    def test_sd_peft_has_train_function(self):
        """Verify sd_peft.py has train() function."""
        import sd_peft

        assert hasattr(sd_peft, "train")
        assert callable(sd_peft.train)

    @pytest.mark.skip(reason="Deprecated: sd_peft.py uses legacy data pipeline")
    def test_sd_peft_has_main_function(self):
        """Verify sd_peft.py has main() function."""
        import sd_peft

        assert hasattr(sd_peft, "main")
        assert callable(sd_peft.main)

    def test_train_script_imports(self):
        """Verify train.py can be imported without errors."""
        import train

        assert train is not None

    def test_train_has_train_function(self):
        """Verify train.py has train() function."""
        import train

        assert hasattr(train, "train")
        assert callable(train.train)

    def test_train_has_main_function(self):
        """Verify train.py has main() function."""
        import train

        assert hasattr(train, "main")
        assert callable(train.main)


class TestConfigImports:
    """Test that config dataclasses can be imported."""

    def test_sd_peft_config_imports(self):
        """Verify RunConfig can be imported."""
        from library.config.dataclasses.run import RunConfig

        assert RunConfig is not None

    def test_sdxl_peft_config_imports(self):
        """Verify RunConfig can be imported."""
        from library.config.dataclasses.run import RunConfig

        assert RunConfig is not None
