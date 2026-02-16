"""
Integration tests for library/training/checkpointing.py

Exercises real file I/O: safetensors metadata roundtrips, checkpoint file
creation/removal with retention policies, accelerator state directory
management, and PEFT adapter state hooks.
"""

import json
import os

import torch
import safetensors.torch

from library.config.dataclasses.output import SavingConfig
from library.training.checkpointing import (
    load_metadata_from_safetensors,
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_and_remove_state_on_epoch_end,
    save_and_remove_state_stepwise,
    save_state_on_train_end,
    register_adapter_state_hooks,
    get_epoch_ckpt_name,
    get_step_ckpt_name,
)


# =============================================================================
# Helpers
# =============================================================================


def _write_dummy_safetensors(path: str, metadata: dict[str, str] | None = None):
    """Write a minimal safetensors file with optional metadata."""
    dummy = {"weight": torch.zeros(1)}
    safetensors.torch.save_file(dummy, path, metadata=metadata)


class FakeAccelerator:
    """Minimal accelerator that actually writes state directories to disk."""

    def __init__(self):
        self.is_main_process = True

    def save_state(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        # Write a marker file so we can verify the directory exists and has content
        with open(os.path.join(output_dir, "state_marker.json"), "w") as f:
            json.dump({"saved": True}, f)

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, model):
        return model

    def register_save_state_pre_hook(self, fn):
        self._save_hook = fn

    def register_load_state_pre_hook(self, fn):
        self._load_hook = fn


# =============================================================================
# Test 1: Safetensors Metadata Roundtrip
# =============================================================================


class TestSafetensorsMetadataRoundtrip:
    """Verify metadata survives a safetensors write/read cycle."""

    def test_metadata_roundtrip(self, tmp_path):
        """Write metadata into a safetensors file and read it back."""
        st_file = str(tmp_path / "model.safetensors")
        metadata = {
            "ss_adapter_module": "lora",
            "ss_adapter_rank": "4",
            "ss_adapter_alpha": "1.0",
            "custom_key": "hello_world",
        }
        _write_dummy_safetensors(st_file, metadata)

        loaded = load_metadata_from_safetensors(st_file)
        for key, value in metadata.items():
            assert loaded[key] == value, f"Key {key}: expected {value}, got {loaded.get(key)}"

    def test_non_safetensors_returns_empty(self, tmp_path):
        """Non-.safetensors files should return empty dict."""
        ckpt_file = str(tmp_path / "model.ckpt")
        with open(ckpt_file, "wb") as f:
            f.write(b"fake")

        assert load_metadata_from_safetensors(ckpt_file) == {}

    def test_no_metadata_returns_empty(self, tmp_path):
        """Safetensors file without metadata should return empty dict."""
        st_file = str(tmp_path / "model.safetensors")
        _write_dummy_safetensors(st_file, metadata=None)

        loaded = load_metadata_from_safetensors(st_file)
        assert isinstance(loaded, dict)


# =============================================================================
# Test 2: Epoch-Based Checkpoint Save & Remove
# =============================================================================


class TestCheckpointSaveRemoveEpoch:
    """Integration test for epoch-based checkpoint creation and retention."""

    def _make_saving_config(self, tmp_path, **overrides) -> SavingConfig:
        defaults = {
            "output_dir": str(tmp_path),
            "output_name": "test_model",
            "save_every_n_epochs": 1,
        }
        defaults.update(overrides)
        return SavingConfig(**defaults)

    def test_epoch_checkpoint_created_sd_format(self, tmp_path):
        """Epoch-end save in SD format creates correctly named file."""
        cfg = self._make_saving_config(tmp_path)
        files_written = []

        def sd_saver(path, epoch_no, global_step):
            with open(path, "w") as f:
                f.write(f"epoch={epoch_no} step={global_step}")
            files_written.append(path)

        save_sd_model_on_epoch_end_or_stepwise_common(
            saving_config=cfg,
            on_epoch_end=True,
            accelerator=FakeAccelerator(),
            save_stable_diffusion_format=True,
            use_safetensors=True,
            epoch=0,  # epoch is 0-indexed, function adds 1
            num_train_epochs=5,
            global_step=100,
            sd_saver=sd_saver,
            diffusers_saver=lambda _: None,
        )

        assert len(files_written) == 1
        expected_name = get_epoch_ckpt_name(cfg, ".safetensors", 1)
        assert os.path.basename(files_written[0]) == expected_name

    def test_old_epoch_checkpoint_removed(self, tmp_path):
        """When save_last_n_epochs is set, old checkpoints are removed."""
        cfg = self._make_saving_config(tmp_path, save_last_n_epochs=2)

        def sd_saver(path, epoch_no, global_step):
            with open(path, "w") as f:
                f.write("data")

        # Save epochs 1 through 4
        for epoch_idx in range(4):
            save_sd_model_on_epoch_end_or_stepwise_common(
                saving_config=cfg,
                on_epoch_end=True,
                accelerator=FakeAccelerator(),
                save_stable_diffusion_format=True,
                use_safetensors=True,
                epoch=epoch_idx,
                num_train_epochs=10,
                global_step=(epoch_idx + 1) * 100,
                sd_saver=sd_saver,
                diffusers_saver=lambda _: None,
            )

        # With save_last_n_epochs=2 and save_every_n_epochs=1, after epoch 4:
        # Should keep epochs 3 and 4, remove epochs 1 and 2
        remaining = os.listdir(tmp_path)
        epoch3_name = get_epoch_ckpt_name(cfg, ".safetensors", 3)
        epoch4_name = get_epoch_ckpt_name(cfg, ".safetensors", 4)
        assert epoch3_name in remaining, f"Expected {epoch3_name} in {remaining}"
        assert epoch4_name in remaining, f"Expected {epoch4_name} in {remaining}"

        epoch1_name = get_epoch_ckpt_name(cfg, ".safetensors", 1)
        epoch2_name = get_epoch_ckpt_name(cfg, ".safetensors", 2)
        assert epoch1_name not in remaining, f"Expected {epoch1_name} removed"
        assert epoch2_name not in remaining, f"Expected {epoch2_name} removed"

    def test_last_epoch_not_saved(self, tmp_path):
        """The final epoch (num_train_epochs) should NOT be saved by stepwise/epoch save."""
        cfg = self._make_saving_config(tmp_path)
        files_written = []

        def sd_saver(path, epoch_no, global_step):
            files_written.append(path)
            with open(path, "w") as f:
                f.write("data")

        # epoch=4 (0-indexed) -> epoch_no=5, num_train_epochs=5 -> not saving
        save_sd_model_on_epoch_end_or_stepwise_common(
            saving_config=cfg,
            on_epoch_end=True,
            accelerator=FakeAccelerator(),
            save_stable_diffusion_format=True,
            use_safetensors=True,
            epoch=4,
            num_train_epochs=5,
            global_step=500,
            sd_saver=sd_saver,
            diffusers_saver=lambda _: None,
        )

        assert len(files_written) == 0, "Final epoch should not trigger mid-training save"


# =============================================================================
# Test 3: Step-Based Checkpoint Save & Remove
# =============================================================================


class TestCheckpointSaveRemoveStep:
    """Integration test for step-based checkpoint creation and retention."""

    def _make_saving_config(self, tmp_path, **overrides) -> SavingConfig:
        defaults = {
            "output_dir": str(tmp_path),
            "output_name": "test_model",
            "save_every_n_steps": 10,
        }
        defaults.update(overrides)
        return SavingConfig(**defaults)

    def test_step_checkpoint_created(self, tmp_path):
        """Step-based save creates correctly named file."""
        cfg = self._make_saving_config(tmp_path)
        files_written = []

        def sd_saver(path, epoch_no, global_step):
            with open(path, "w") as f:
                f.write(f"step={global_step}")
            files_written.append(path)

        # on_epoch_end=False -> stepwise save (caller decides when to call)
        save_sd_model_on_epoch_end_or_stepwise_common(
            saving_config=cfg,
            on_epoch_end=False,
            accelerator=FakeAccelerator(),
            save_stable_diffusion_format=True,
            use_safetensors=True,
            epoch=0,
            num_train_epochs=10,
            global_step=50,
            sd_saver=sd_saver,
            diffusers_saver=lambda _: None,
        )

        assert len(files_written) == 1
        expected_name = get_step_ckpt_name(cfg, ".safetensors", 50)
        assert os.path.basename(files_written[0]) == expected_name

    def test_old_step_checkpoint_removed(self, tmp_path):
        """When save_last_n_steps is set, old step checkpoints are removed."""
        cfg = self._make_saving_config(
            tmp_path, save_every_n_steps=10, save_last_n_steps=30
        )

        def sd_saver(path, epoch_no, global_step):
            with open(path, "w") as f:
                f.write("data")

        # Save at steps 10, 20, 30, 40, 50
        for step in [10, 20, 30, 40, 50]:
            save_sd_model_on_epoch_end_or_stepwise_common(
                saving_config=cfg,
                on_epoch_end=False,
                accelerator=FakeAccelerator(),
                save_stable_diffusion_format=True,
                use_safetensors=True,
                epoch=0,
                num_train_epochs=10,
                global_step=step,
                sd_saver=sd_saver,
                diffusers_saver=lambda _: None,
            )

        remaining = os.listdir(tmp_path)

        # Step 50: remove_step = 50 - 30 - 1 = 19, aligned to 10 -> 10
        # Step 40: remove_step = 40 - 30 - 1 = 9, aligned to 10 -> 0 (not > 0, skip)
        # So step 10 should be removed
        step10_name = get_step_ckpt_name(cfg, ".safetensors", 10)
        assert step10_name not in remaining, f"Expected {step10_name} to be removed"

        # Steps 20-50 should still exist
        for step in [20, 30, 40, 50]:
            name = get_step_ckpt_name(cfg, ".safetensors", step)
            assert name in remaining, f"Expected {name} in {remaining}"


# =============================================================================
# Test 4: State Directory Save & Remove
# =============================================================================


class TestStateSaveRemove:
    """Integration test for training state directory management."""

    def _make_saving_config(self, tmp_path, **overrides) -> SavingConfig:
        defaults = {
            "output_dir": str(tmp_path),
            "output_name": "test_model",
            "save_every_n_epochs": 1,
        }
        defaults.update(overrides)
        return SavingConfig(**defaults)

    def test_epoch_state_created(self, tmp_path):
        """State directory is created at epoch end."""
        cfg = self._make_saving_config(tmp_path)
        accel = FakeAccelerator()

        save_and_remove_state_on_epoch_end(cfg, accel, epoch_no=1)

        expected_dir = tmp_path / "test_model-000001-state"
        assert expected_dir.exists()
        assert (expected_dir / "state_marker.json").exists()

    def test_old_epoch_state_removed(self, tmp_path):
        """Old state directories removed per save_last_n_epochs_state."""
        cfg = self._make_saving_config(tmp_path, save_last_n_epochs_state=2)
        accel = FakeAccelerator()

        for epoch_no in range(1, 5):
            save_and_remove_state_on_epoch_end(cfg, accel, epoch_no=epoch_no)

        # With save_last_n_epochs_state=2 and save_every_n_epochs=1:
        # After epoch 4, keep epochs 3, 4; remove 1, 2
        assert (tmp_path / "test_model-000003-state").exists()
        assert (tmp_path / "test_model-000004-state").exists()
        assert not (tmp_path / "test_model-000001-state").exists()
        assert not (tmp_path / "test_model-000002-state").exists()

    def test_step_state_created(self, tmp_path):
        """State directory is created at step save."""
        cfg = self._make_saving_config(tmp_path, save_every_n_steps=10)
        accel = FakeAccelerator()

        save_and_remove_state_stepwise(cfg, accel, step_no=50)

        expected_dir = tmp_path / "test_model-step00000050-state"
        assert expected_dir.exists()

    def test_old_step_state_removed(self, tmp_path):
        """Old step state directories removed per save_last_n_steps_state."""
        cfg = self._make_saving_config(
            tmp_path, save_every_n_steps=10, save_last_n_steps_state=20
        )
        accel = FakeAccelerator()

        for step in [10, 20, 30, 40, 50]:
            save_and_remove_state_stepwise(cfg, accel, step_no=step)

        # Step 50: remove = 50 - 20 - 1 = 29, aligned to 10 -> 20
        # Step 40: remove = 40 - 20 - 1 = 19, aligned to 10 -> 10
        assert not (tmp_path / "test_model-step00000010-state").exists()
        assert not (tmp_path / "test_model-step00000020-state").exists()
        assert (tmp_path / "test_model-step00000050-state").exists()

    def test_train_end_state_created(self, tmp_path):
        """State is saved at training end with 'last' naming."""
        cfg = self._make_saving_config(tmp_path)
        accel = FakeAccelerator()

        save_state_on_train_end(cfg, accel)

        expected_dir = tmp_path / "test_model-state"
        assert expected_dir.exists()


# =============================================================================
# Test 5: PEFT Adapter State Hooks
# =============================================================================


class TestAdapterStateHooks:
    """Integration test for register_adapter_state_hooks save/load cycle."""

    def test_save_then_load_train_state(self, tmp_path):
        """Save hook writes train_state.json, load hook restores it."""
        from types import SimpleNamespace

        accel = FakeAccelerator()

        class FakeAdapter:
            pass

        adapter = FakeAdapter()
        accel.unwrap_model = lambda m: m  # type: ignore[assignment]

        cfg = SimpleNamespace(performance=SimpleNamespace(deepspeed=False))
        current_epoch = SimpleNamespace(value=3)
        current_step = SimpleNamespace(value=99)

        get_steps = register_adapter_state_hooks(accel, adapter, cfg, current_epoch, current_step)

        # Initially no steps loaded
        assert get_steps() is None

        # Simulate save: call the save hook
        output_dir = str(tmp_path / "checkpoint_state")
        os.makedirs(output_dir, exist_ok=True)
        accel._save_hook(
            models=[adapter],
            weights=[{"param": torch.zeros(1)}],
            output_dir=output_dir,
        )

        # Verify train_state.json was written
        state_file = os.path.join(output_dir, "train_state.json")
        assert os.path.exists(state_file)
        with open(state_file) as f:
            data = json.load(f)
        # +1 because save hook adds 1 to current_step
        assert data["current_step"] == 100
        assert data["current_epoch"] == 3

        # Reset state, then load
        current_epoch.value = 0
        current_step.value = 0

        accel._load_hook(models=[adapter], input_dir=output_dir)

        assert get_steps() == 100
        assert current_epoch.value == 3
        assert current_step.value == 100

    def test_save_hook_filters_non_adapter_weights(self, tmp_path):
        """Save hook should pop weights of models that aren't the adapter type."""
        from types import SimpleNamespace

        accel = FakeAccelerator()

        class FakeAdapter:
            pass

        class OtherModel:
            pass

        adapter = FakeAdapter()
        other = OtherModel()
        accel.unwrap_model = lambda m: m  # type: ignore[assignment]

        cfg = SimpleNamespace(performance=SimpleNamespace(deepspeed=False))
        current_epoch = SimpleNamespace(value=0)
        current_step = SimpleNamespace(value=0)

        register_adapter_state_hooks(accel, adapter, cfg, current_epoch, current_step)

        output_dir = str(tmp_path / "hook_test")
        os.makedirs(output_dir, exist_ok=True)

        # 2 models: other (should be filtered) and adapter (should stay)
        weights = [{"other_w": torch.zeros(1)}, {"adapter_w": torch.zeros(1)}]
        models = [other, adapter]
        accel._save_hook(models=models, weights=weights, output_dir=output_dir)

        # The "other" model's weights should have been popped
        assert len(weights) == 1
        assert "adapter_w" in weights[0]

    def test_load_hook_without_state_file(self, tmp_path):
        """Load hook gracefully handles missing train_state.json."""
        from types import SimpleNamespace

        accel = FakeAccelerator()

        class FakeAdapter:
            pass

        adapter = FakeAdapter()
        accel.unwrap_model = lambda m: m  # type: ignore[assignment]

        cfg = SimpleNamespace(performance=SimpleNamespace(deepspeed=False))
        current_epoch = SimpleNamespace(value=5)
        current_step = SimpleNamespace(value=50)

        get_steps = register_adapter_state_hooks(accel, adapter, cfg, current_epoch, current_step)

        # Load from empty directory (no train_state.json)
        input_dir = str(tmp_path / "empty_state")
        os.makedirs(input_dir, exist_ok=True)
        accel._load_hook(models=[adapter], input_dir=input_dir)

        # Should not crash, steps_from_state stays None
        assert get_steps() is None
        # Epoch/step should be unchanged
        assert current_epoch.value == 5
        assert current_step.value == 50
