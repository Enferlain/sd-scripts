from __future__ import annotations

from types import SimpleNamespace

from library.adapters.shared.state_io import (
    AdapterExportLoadRequest,
    AdapterExportSaveRequest,
    load_adapter_export,
    register_adapter_checkpoint_state_hooks,
    save_adapter_export,
)
import pytest


class _FakeAdapter:
    def __init__(self):
        self.loaded_from: str | None = None
        self.saved = None

    def load_weights(self, file: str):
        self.loaded_from = file
        return {"file": file}

    def save_weights(self, file: str, dtype, metadata):
        self.saved = (file, dtype, metadata)


class _WrappedAdapter:
    def __init__(self, module):
        self.module = module


class _FakeAccelerator:
    def __init__(self):
        self.is_main_process = True
        self.save_hook = None
        self.load_hook = None

    def unwrap_model(self, model):
        return getattr(model, "module", model)

    def register_save_state_pre_hook(self, fn):
        self.save_hook = fn

    def register_load_state_pre_hook(self, fn):
        self.load_hook = fn


def test_export_helpers_delegate_to_adapter_runtime():
    adapter = _FakeAdapter()

    load_result = load_adapter_export(adapter, AdapterExportLoadRequest(file="input.safetensors"))
    save_adapter_export(
        adapter,
        AdapterExportSaveRequest(file="output.safetensors", dtype="fp16", metadata={"ss_epoch": "1"}),
    )

    assert load_result == {"file": "input.safetensors"}
    assert adapter.loaded_from == "input.safetensors"
    assert adapter.saved == ("output.safetensors", "fp16", {"ss_epoch": "1"})


def test_export_helpers_reject_non_adapter_export_objects():
    with pytest.raises(TypeError, match="Adapter export load"):
        load_adapter_export(object(), AdapterExportLoadRequest(file="input.safetensors"))

    with pytest.raises(TypeError, match="Adapter export save"):
        save_adapter_export(
            object(),
            AdapterExportSaveRequest(file="output.safetensors", dtype="fp16", metadata={"ss_epoch": "1"}),
        )


def test_checkpoint_state_hooks_keep_only_adapter_runtime_and_restore_train_state(tmp_path):
    accelerator = _FakeAccelerator()
    target_adapter = _FakeAdapter()
    other_adapter = _FakeAdapter()
    current_epoch = SimpleNamespace(value=2)
    current_step = SimpleNamespace(value=6)

    resume_state = register_adapter_checkpoint_state_hooks(
        accelerator,
        _WrappedAdapter(target_adapter),
        save_for_deepspeed=False,
        current_epoch=current_epoch,
        current_step=current_step,
    )

    models = [_WrappedAdapter(other_adapter), _WrappedAdapter(target_adapter)]
    weights = ["other", "target"]
    output_dir = str(tmp_path / "state")
    accelerator.save_hook(models, weights, output_dir)

    assert weights == ["target"]

    current_epoch.value = 0
    current_step.value = 0
    load_models = [_WrappedAdapter(other_adapter), _WrappedAdapter(target_adapter)]
    accelerator.load_hook(load_models, output_dir)

    assert len(load_models) == 1
    assert accelerator.unwrap_model(load_models[0]) is target_adapter
    assert resume_state.epoch == 2
    assert resume_state.step == 7
    assert current_epoch.value == 2
    assert current_step.value == 7


def test_checkpoint_state_hooks_save_for_deepspeed_even_when_not_main_process(tmp_path):
    accelerator = _FakeAccelerator()
    accelerator.is_main_process = False
    adapter = _FakeAdapter()
    current_epoch = SimpleNamespace(value=1)
    current_step = SimpleNamespace(value=3)

    register_adapter_checkpoint_state_hooks(
        accelerator,
        adapter,
        save_for_deepspeed=True,
        current_epoch=current_epoch,
        current_step=current_step,
    )

    output_dir = str(tmp_path / "state")
    accelerator.save_hook([adapter], ["target"], output_dir)

    assert (tmp_path / "state" / "train_state.json").exists()
