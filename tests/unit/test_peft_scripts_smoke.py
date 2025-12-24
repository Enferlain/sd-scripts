"""
Smoke Tests for PEFT Training Scripts

These tests verify that the PEFT training scripts and their dependencies
can be imported without errors. Run after every refactoring change to catch
import issues early.

Usage:
    pytest tests/unit/test_peft_scripts_smoke.py -v
"""

import pytest
import sys
from pathlib import Path


# Ensure scripts directory is importable
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestPeftCommonImports:
    """Test that peft_common.py can be imported and has expected functions."""

    def test_peft_common_imports(self):
        """Verify peft_common.py imports without errors."""
        from library.training import peft_common
        assert peft_common is not None

    def test_peft_common_has_generate_step_logs(self):
        """Verify generate_step_logs function exists."""
        from library.training.peft_common import generate_step_logs
        assert callable(generate_step_logs)

    def test_peft_common_has_step_logging(self):
        """Verify step_logging function exists."""
        from library.training.peft_common import step_logging
        assert callable(step_logging)

    def test_peft_common_has_init_timestep_sampler(self):
        """Verify init_timestep_sampler function exists."""
        from library.training.peft_common import init_timestep_sampler
        assert callable(init_timestep_sampler)

    def test_peft_common_has_create_training_metadata(self):
        """Verify create_training_metadata function exists."""
        from library.training.peft_common import create_training_metadata
        assert callable(create_training_metadata)

    def test_peft_common_has_setup_live_plotter(self):
        """Verify setup_live_plotter function exists."""
        from library.training.peft_common import setup_live_plotter
        assert callable(setup_live_plotter)

    def test_peft_common_has_prepare_datasets(self):
        """Verify prepare_datasets function exists."""
        from library.training.peft_common import prepare_datasets
        assert callable(prepare_datasets)

    def test_peft_common_has_calculate_initial_step(self):
        """Verify calculate_initial_step function exists."""
        from library.training.peft_common import calculate_initial_step
        assert callable(calculate_initial_step)

    def test_peft_common_has_parse_dynamic_timestep_schedule(self):
        """Verify parse_dynamic_timestep_schedule function exists."""
        from library.training.peft_common import parse_dynamic_timestep_schedule
        assert callable(parse_dynamic_timestep_schedule)

    def test_peft_common_has_register_network_state_hooks(self):
        """Verify register_network_state_hooks function exists."""
        from library.training.peft_common import register_network_state_hooks
        assert callable(register_network_state_hooks)


class TestStrategyImports:
    """Test that strategy classes can be imported."""

    def test_sd_peft_strategy_imports(self):
        """Verify SdPeftStrategy can be imported."""
        from library.strategies.peft_strategy_sd import SdPeftStrategy
        assert SdPeftStrategy is not None

    def test_sdxl_peft_strategy_imports(self):
        """Verify SdxlPeftStrategy can be imported."""
        from library.strategies.peft_strategy_sdxl import SdxlPeftStrategy
        assert SdxlPeftStrategy is not None

    def test_sd_strategy_instantiation(self):
        """Verify SdPeftStrategy can be instantiated."""
        from library.strategies.peft_strategy_sd import SdPeftStrategy
        strategy = SdPeftStrategy()
        assert strategy is not None

    def test_sdxl_strategy_instantiation(self):
        """Verify SdxlPeftStrategy can be instantiated."""
        from library.strategies.peft_strategy_sdxl import SdxlPeftStrategy
        strategy = SdxlPeftStrategy()
        assert strategy is not None


class TestScriptImports:
    """Test that the main training scripts can be imported."""

    def test_sd_peft_script_imports(self):
        """Verify sd_peft.py can be imported without errors."""
        import sd_peft
        assert sd_peft is not None

    def test_sd_peft_has_train_function(self):
        """Verify sd_peft.py has train() function."""
        import sd_peft
        assert hasattr(sd_peft, 'train')
        assert callable(sd_peft.train)

    def test_sd_peft_has_main_function(self):
        """Verify sd_peft.py has main() function."""
        import sd_peft
        assert hasattr(sd_peft, 'main')
        assert callable(sd_peft.main)

    def test_sdxl_peft_script_imports(self):
        """Verify sdxl_peft.py can be imported without errors."""
        import sdxl_peft
        assert sdxl_peft is not None

    def test_sdxl_peft_has_train_function(self):
        """Verify sdxl_peft.py has train() function."""
        import sdxl_peft
        assert hasattr(sdxl_peft, 'train')
        assert callable(sdxl_peft.train)

    def test_sdxl_peft_has_main_function(self):
        """Verify sdxl_peft.py has main() function."""
        import sdxl_peft
        assert hasattr(sdxl_peft, 'main')
        assert callable(sdxl_peft.main)


class TestConfigImports:
    """Test that config dataclasses can be imported."""

    def test_sd_peft_config_imports(self):
        """Verify SDPeftConfig can be imported."""
        from library.config.dataclasses.sd_peft import SDPeftConfig
        assert SDPeftConfig is not None

    def test_sdxl_peft_config_imports(self):
        """Verify SDXLPeftConfig can be imported."""
        from library.config.dataclasses.sdxl_peft import SDXLPeftConfig
        assert SDXLPeftConfig is not None
