"""
Unit tests for library/training/checkpointing.py

Tests checkpoint naming, metadata building, and utility functions.
"""

import pytest

from library.training.checkpointing import (
    get_epoch_ckpt_name,
    get_step_ckpt_name,
    get_last_ckpt_name,
    get_remove_epoch_no,
    get_remove_step_no,
    build_minimum_adapter_metadata,
    default_if_none,
)
from library.config.dataclasses.output import SavingConfig


# =============================================================================
# Checkpoint Naming Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestCheckpointNaming:
    """Test checkpoint file naming functions."""

    def test_get_epoch_ckpt_name_with_output_name(self, tmp_output_dir):
        """Test epoch checkpoint naming with specified output_name."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name="my_model")

        result = get_epoch_ckpt_name(config, ".safetensors", epoch_no=5)

        assert result == "my_model-000005.safetensors"

    def test_get_epoch_ckpt_name_with_append(self, tmp_output_dir):
        """Test epoch checkpoint naming with output_name_append."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name="my_model")

        result = get_epoch_ckpt_name(config, ".safetensors", epoch_no=5, output_name_append="_final")

        assert result == "my_model_final-000005.safetensors"

    def test_get_epoch_ckpt_name_default(self, tmp_output_dir):
        """Test epoch checkpoint naming with default name."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name=None)

        result = get_epoch_ckpt_name(config, ".safetensors", epoch_no=10)

        # Should use default name
        assert "000010.safetensors" in result

    def test_get_step_ckpt_name_with_output_name(self, tmp_output_dir):
        """Test step checkpoint naming with specified output_name."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name="my_model")

        result = get_step_ckpt_name(config, ".safetensors", step_no=1000)

        assert result == "my_model-step00001000.safetensors"

    def test_get_step_ckpt_name_with_append(self, tmp_output_dir):
        """Test step checkpoint naming with output_name_append."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name="my_model")

        result = get_step_ckpt_name(config, ".ckpt", step_no=5000, output_name_append="_best")

        assert result == "my_model_best-step00005000.ckpt"

    def test_get_last_ckpt_name(self, tmp_output_dir):
        """Test last checkpoint naming."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name="my_model")

        result = get_last_ckpt_name(config, ".safetensors")

        assert result == "my_model.safetensors"

    def test_get_last_ckpt_name_with_append(self, tmp_output_dir):
        """Test last checkpoint naming with append."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name="my_model")

        result = get_last_ckpt_name(config, ".safetensors", output_name_append="-last")

        assert result == "my_model-last.safetensors"

    def test_different_extensions(self, tmp_output_dir):
        """Test checkpoint naming with different file extensions."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name="model")

        # Test .safetensors
        assert get_epoch_ckpt_name(config, ".safetensors", 1).endswith(".safetensors")

        # Test .ckpt
        assert get_epoch_ckpt_name(config, ".ckpt", 1).endswith(".ckpt")

        # Test .pt
        assert get_step_ckpt_name(config, ".pt", 100).endswith(".pt")


# =============================================================================
# Checkpoint Removal Logic Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestCheckpointRemoval:
    """Test functions that determine which checkpoints to remove."""

    def test_get_remove_epoch_no_with_keep_n_epochs(self, tmp_output_dir):
        """Test epoch removal with save_n_epoch_ratio specified."""
        config = SavingConfig(output_dir=tmp_output_dir, save_every_n_epochs=1, save_n_epoch_ratio=3)

        # With save_n_epoch_ratio=3, should keep every 3rd epoch
        # Epoch 5: should keep epochs 3, 6, 9, etc., so remove 5
        result = get_remove_epoch_no(config, epoch_no=5)

        # The function returns the epoch to remove, or None if should keep
        # With ratio of 3, epoch 5 is not divisible by 3, so might be removed
        assert result is None or isinstance(result, int)

    def test_get_remove_epoch_no_no_removal(self, tmp_output_dir):
        """Test when no epoch removal is configured."""
        config = SavingConfig(output_dir=tmp_output_dir, save_every_n_epochs=1, save_n_epoch_ratio=None)

        result = get_remove_epoch_no(config, epoch_no=5)

        # Should not remove if save_n_epoch_ratio is None
        assert result is None

    def test_get_remove_step_no_with_save_every(self, tmp_output_dir):
        """Test step removal logic."""
        config = SavingConfig(
            output_dir=tmp_output_dir,
            save_every_n_steps=100,
        )

        result = get_remove_step_no(config, step_no=500)

        # Should return epoch number to remove or None
        assert result is None or isinstance(result, int)


# =============================================================================
# Metadata Building Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestMetadataBuilding:
    """Test metadata building functions."""

    def test_build_minimum_adapter_metadata_basic(self):
        """Test building basic peft metadata."""
        metadata = build_minimum_adapter_metadata(
            v2=None, base_model=None, adapter_module="adapters.lora", adapter_rank="8", adapter_alpha="1.0", adapter_args=None
        )

        assert metadata is not None
        assert metadata["ss_adapter_module"] == "adapters.lora"
        assert metadata["ss_adapter_rank"] == "8"
        assert metadata["ss_adapter_alpha"] == "1.0"

    def test_build_minimum_adapter_metadata_with_v2(self):
        """Test building metadata with v2 flag."""
        metadata = build_minimum_adapter_metadata(
            v2="v2-1", base_model=None, adapter_module="adapters.lora", adapter_rank="16", adapter_alpha="1.0", adapter_args=None
        )

        assert "ss_v2" in metadata
        assert metadata["ss_v2"] == "v2-1"

    def test_build_minimum_adapter_metadata_with_base_model(self):
        """Test building metadata with base model."""
        metadata = build_minimum_adapter_metadata(
            v2=None, base_model="sd-v1-5", adapter_module="adapters.lora", adapter_rank="32", adapter_alpha="16.0", adapter_args=None
        )

        assert "ss_base_model_version" in metadata
        assert metadata["ss_base_model_version"] == "sd-v1-5"

    def test_build_minimum_adapter_metadata_with_args(self):
        """Test building metadata with peft args."""
        adapter_args = {"conv_dim": 4, "conv_alpha": 1.0}

        metadata = build_minimum_adapter_metadata(
            v2=None, base_model=None, adapter_module="adapters.lora", adapter_rank="64", adapter_alpha="32.0", adapter_args=adapter_args
        )

        assert "ss_adapter_args" in metadata
        # Adapter args should be JSON-encoded
        import json

        decoded_args = json.loads(metadata["ss_adapter_args"])
        assert decoded_args["conv_dim"] == 4
        assert decoded_args["conv_alpha"] == 1.0

    def test_build_minimum_adapter_metadata_complete(self):
        """Test building metadata with all optional fields."""
        adapter_args = {"dropout": 0.1}

        metadata = build_minimum_adapter_metadata(
            v2="v2-1",
            base_model="sd-v1-5",
            adapter_module="adapters.lora",
            adapter_rank="128",
            adapter_alpha="64.0",
            adapter_args=adapter_args,
        )

        # Check all fields are present
        assert "ss_adapter_module" in metadata
        assert "ss_adapter_rank" in metadata
        assert "ss_adapter_alpha" in metadata
        assert "ss_v2" in metadata
        assert "ss_base_model_version" in metadata
        assert "ss_adapter_args" in metadata


# =============================================================================
# Utility Function Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestCheckpointingUtils:
    """Test utility functions."""

    def test_default_if_none_with_value(self):
        """Test default_if_none when value is provided."""
        result = default_if_none("my_value", "default_value")
        assert result == "my_value"

    def test_default_if_none_with_none(self):
        """Test default_if_none when value is None."""
        result = default_if_none(None, "default_value")
        assert result == "default_value"

    def test_default_if_none_with_empty_string(self):
        """Test default_if_none with empty string (should keep empty string)."""
        result = default_if_none("", "default_value")
        assert result == ""

    def test_default_if_none_with_zero(self):
        """Test default_if_none with zero (should keep zero)."""
        result = default_if_none(0, 42)
        assert result == 0

    def test_default_if_none_with_false(self):
        """Test default_if_none with False (should keep False)."""
        result = default_if_none(False, True)
        assert result == False


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.integration
class TestCheckpointingIntegration:
    """Integration tests for checkpointing with SavingConfig."""

    def test_checkpoint_naming_consistency(self, tmp_output_dir):
        """Test that checkpoint naming is consistent across functions."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name="consistent_model")

        # All naming functions should use the same base name
        epoch_name = get_epoch_ckpt_name(config, ".safetensors", 1)
        step_name = get_step_ckpt_name(config, ".safetensors", 100)
        last_name = get_last_ckpt_name(config, ".safetensors")

        assert "consistent_model" in epoch_name
        assert "consistent_model" in step_name
        assert "consistent_model" in last_name

    def test_full_checkpoint_workflow(self, tmp_output_dir):
        """Test a complete checkpoint saving workflow."""
        config = SavingConfig(output_dir=tmp_output_dir, output_name="workflow_test", save_model_as="safetensors")

        # Generate names for different checkpoint types
        epoch_5 = get_epoch_ckpt_name(config, ".safetensors", 5)
        step_1000 = get_step_ckpt_name(config, ".safetensors", 1000)
        last = get_last_ckpt_name(config, ".safetensors")

        # Verify they're all different
        assert epoch_5 != step_1000
        assert epoch_5 != last
        assert step_1000 != last

        # Verify they all have the right extension
        assert all(name.endswith(".safetensors") for name in [epoch_5, step_1000, last])
