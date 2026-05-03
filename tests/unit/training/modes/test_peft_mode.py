from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path

import torch

from library.config.dataclasses.performance import DeepSpeedConfig
from library.adapters import LoadedAdapterRuntime
from library.optimization.types import OptimizerBuildResult
from library.adapters.runtime.targets import build_component_module_targets, build_component_root_targets
from library.training.checkpointing import ResumeState
from library.training.modes.adapter_mode import AdapterMode


def _build_mock_trainer():
    peft_config = SimpleNamespace(
        continue_from=None,
        continue_mode="strict",
        lora=SimpleNamespace(
            rank=None,
            alpha=1.0,
            dropout=None,
            conv_rank=None,
            conv_alpha=None,
            rank_dropout=None,
            module_dropout=None,
        ),
        loha=None,
        scale_weight_norms=False,
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sdxl"),
        adapter=SimpleNamespace(peft=peft_config),
        optimizer=SimpleNamespace(
            learning_rates=SimpleNamespace(base=1e-4, denoiser=1e-4, text_encoders=[0.0, 1e-4], groups=None, groups_file=None),
            optimizer_args=None,
            scheduler=SimpleNamespace(),
        ),
        performance=SimpleNamespace(
            memory=SimpleNamespace(lowram=False),
            deepspeed=DeepSpeedConfig(deepspeed=False),
        ),
    )

    trainer = MagicMock()
    trainer.cfg = cfg
    trainer.accelerator = MagicMock()
    trainer.accelerator.print = MagicMock()
    trainer.vae = object()
    trainer.denoiser = object()
    trainer.text_encoders = [object(), object()]
    trainer._text_encoder = trainer.text_encoders
    trainer.weight_dtype = "fp16"
    trainer.strategies = MagicMock()
    trainer.strategies.post_process_trainable = MagicMock()
    trainer.net_kwargs = {}
    return trainer


def test_prepare_trainables_builds_resolved_targets_before_adapter_instantiation(monkeypatch):
    trainer = _build_mock_trainer()
    captured = {}
    adapter = MagicMock()
    adapter.apply_to = MagicMock()
    resolved_targets = build_component_root_targets(
        model_type="sdxl",
        text_encoders=trainer.text_encoders,
        vae=None,
        denoiser=trainer.denoiser,
        include_text_encoders=[True, False],
        include_denoiser=False,
    )

    def fake_build_adapter(request):
        captured["request"] = request
        return adapter

    monkeypatch.setattr(
        "library.training.modes.adapter_mode.resolve_adapter_target_selection",
        lambda **_: SimpleNamespace(
            resolved_targets=resolved_targets,
            train_denoiser=False,
            train_any_text_encoder=True,
        ),
    )
    monkeypatch.setattr("library.training.modes.adapter_mode.build_adapter", fake_build_adapter)

    AdapterMode().prepare_trainables(trainer)

    assert captured["request"].adapter.adapter_type == "lora"
    assert captured["request"].resolved_targets is resolved_targets
    assert trainer._train_text_encoder is True
    assert trainer._train_denoiser is False
    assert trainer.adapter_resolved_targets is captured["request"].resolved_targets
    adapter.apply_to.assert_called_once_with(trainer._text_encoder, trainer.denoiser, True, False)


def test_prepare_trainables_prefers_nested_lora_config_surface(monkeypatch):
    trainer = _build_mock_trainer()
    trainer.cfg.adapter.peft.lora.rank = 64
    trainer.cfg.adapter.peft.lora.alpha = 128.0
    trainer.cfg.adapter.peft.lora.dropout = 0.2
    captured = {}
    adapter = MagicMock()
    adapter.apply_to = MagicMock()
    resolved_targets = build_component_root_targets(
        model_type="sdxl",
        text_encoders=trainer.text_encoders,
        vae=None,
        denoiser=trainer.denoiser,
        include_text_encoders=[True, False],
        include_denoiser=False,
    )

    def fake_build_adapter(request):
        captured["request"] = request
        return adapter

    monkeypatch.setattr(
        "library.training.modes.adapter_mode.resolve_adapter_target_selection",
        lambda **_: SimpleNamespace(
            resolved_targets=resolved_targets,
            train_denoiser=False,
            train_any_text_encoder=True,
        ),
    )
    monkeypatch.setattr("library.training.modes.adapter_mode.build_adapter", fake_build_adapter)

    AdapterMode().prepare_trainables(trainer)

    assert captured["request"].adapter.settings["adapter_rank"] == 64
    assert captured["request"].adapter.settings["adapter_alpha"] == 128.0
    assert captured["request"].adapter.settings["neuron_dropout"] == 0.2
    assert captured["request"].adapter.settings["dropout"] == 0.2


def test_build_optimizer_params_uses_repo_owned_grouping_plan(monkeypatch):
    trainer = _build_mock_trainer()
    trainer.adapter = object()
    execution_groups = [MagicMock()]
    logical_groups = [MagicMock(metric_name="unet")]
    captured = {}

    monkeypatch.setattr(
        "library.training.modes.adapter_mode.build_adapter_grouping",
        lambda **kwargs: SimpleNamespace(execution_groups=execution_groups, logical_groups=logical_groups),
    )

    def fake_get_optimizer(optimizer_config, learning_rates, scheduler_config, execution_group_payload, optimizer_kwargs):
        captured["execution_groups"] = execution_group_payload
        captured["optimizer_kwargs"] = optimizer_kwargs
        return "AdamW", {"lr": 1e-4}, "optimizer"

    monkeypatch.setattr("library.training.modes.adapter_mode.get_optimizer", fake_get_optimizer)

    result = AdapterMode().build_optimizer_params(trainer)

    assert isinstance(result, OptimizerBuildResult)
    assert result.optimizer_name == "AdamW"
    assert result.optimizer == "optimizer"
    assert result.optimization_plan.execution_groups == execution_groups
    assert result.optimization_plan.logical_groups == logical_groups
    assert result.lr_descriptions == ["unet"]
    assert captured["execution_groups"] == execution_groups
    assert captured["optimizer_kwargs"] == {}


def test_prepare_trainables_loads_adapter_weights_through_repo_owned_export_seam(monkeypatch):
    trainer = _build_mock_trainer()
    trainer.cfg.adapter.peft.continue_from = "adapter.safetensors"
    trainer.cfg.adapter.peft.continue_mode = "initialize_from_artifact"
    adapter = MagicMock()
    adapter.apply_to = MagicMock()
    resolved_targets = build_component_root_targets(
        model_type="sdxl",
        text_encoders=trainer.text_encoders,
        vae=None,
        denoiser=trainer.denoiser,
        include_text_encoders=[True, False],
        include_denoiser=False,
    )
    captured = {}

    monkeypatch.setattr(
        "library.training.modes.adapter_mode.resolve_adapter_target_selection",
        lambda **_: SimpleNamespace(
            resolved_targets=resolved_targets,
            train_denoiser=False,
            train_any_text_encoder=True,
        ),
    )
    monkeypatch.setattr("library.training.modes.adapter_mode.build_adapter", lambda *_: adapter)

    def fake_load_adapter_export(target_adapter, request):
        captured["adapter"] = target_adapter
        captured["request"] = request
        return {"loaded": request.file}

    monkeypatch.setattr("library.training.modes.adapter_mode.load_adapter_export", fake_load_adapter_export)

    AdapterMode().prepare_trainables(trainer)

    assert captured["adapter"] is adapter
    assert captured["request"].file == "adapter.safetensors"


def test_prepare_trainables_builds_from_weights_runtime_with_optimization_owned_targets(monkeypatch):
    trainer = _build_mock_trainer()
    trainer.cfg.adapter.peft.continue_from = "adapter.safetensors"
    trainer.cfg.adapter.peft.continue_mode = "strict"
    adapter = MagicMock()
    adapter.apply_to = MagicMock()
    resolved_targets = build_component_root_targets(
        model_type="sdxl",
        text_encoders=trainer.text_encoders,
        vae=None,
        denoiser=trainer.denoiser,
        include_text_encoders=[True, False],
        include_denoiser=True,
    )
    captured = {}
    loaded_runtime = LoadedAdapterRuntime(
        adapter=adapter,
        state={"loaded": "adapter.safetensors"},
    )

    monkeypatch.setattr(
        "library.training.modes.adapter_mode.resolve_adapter_target_selection",
        lambda **_: SimpleNamespace(
            resolved_targets=resolved_targets,
            train_denoiser=True,
            train_any_text_encoder=True,
        ),
    )
    monkeypatch.setattr(
        "library.training.modes.adapter_mode.build_adapter_from_weights",
        lambda request, weights_path: (
            captured.update(
                {
                    "request": request,
                    "weights_path": weights_path,
                    "loaded_runtime": loaded_runtime,
                }
            )
            or loaded_runtime
        ),
    )
    AdapterMode().prepare_trainables(trainer)

    assert captured["request"].adapter.adapter_type == "lora"
    assert captured["request"].resolved_targets is resolved_targets
    assert captured["request"].context.for_inference is False
    assert captured["weights_path"] == "adapter.safetensors"
    assert captured["loaded_runtime"].state == {"loaded": "adapter.safetensors"}
    assert trainer.adapter is adapter
    assert trainer.adapter_resolved_targets is resolved_targets


def test_prepare_trainables_supports_registered_loha_runtime_with_module_targets(monkeypatch):
    trainer = _build_mock_trainer()
    trainer.cfg.adapter.peft.lora = None
    trainer.cfg.adapter.peft.loha = SimpleNamespace(
        rank=4,
        alpha=1.0,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        use_tucker=False,
        use_scalar=False,
        rank_dropout_scale=False,
        weight_decompose=False,
        wd_on_output=True,
        bypass_mode=None,
        rs_lora=False,
    )
    trainer.cfg.adapter.peft.loha.use_scalar = True
    trainer.cfg.adapter.peft.loha.bypass_mode = True
    trainer.text_encoders = [torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)), object()]
    trainer._text_encoder = trainer.text_encoders
    trainer.denoiser = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    adapter = MagicMock()
    adapter.apply_to = MagicMock()
    captured = {}
    resolved_targets = build_component_module_targets(
        model_type="sdxl",
        text_encoders=trainer.text_encoders,
        vae=None,
        denoiser=trainer.denoiser,
        include_text_encoders=[True, False],
        include_denoiser=True,
    )

    monkeypatch.setattr(
        "library.training.modes.adapter_mode.resolve_adapter_target_selection",
        lambda **_: SimpleNamespace(
            resolved_targets=resolved_targets,
            train_denoiser=True,
            train_any_text_encoder=True,
        ),
    )

    def fake_build_adapter(request):
        captured["request"] = request
        return adapter

    monkeypatch.setattr("library.training.modes.adapter_mode.build_adapter", fake_build_adapter)

    AdapterMode().prepare_trainables(trainer)

    assert captured["request"].adapter.adapter_type == "loha"
    assert captured["request"].resolved_targets is resolved_targets
    assert captured["request"].adapter.settings["use_scalar"] is True
    assert captured["request"].adapter.settings["bypass_mode"] is True
    adapter.apply_to.assert_called_once_with(trainer._text_encoder, trainer.denoiser, True, True)
    assert trainer.adapter_resolved_targets is resolved_targets


def test_register_state_hooks_uses_repo_owned_adapter_checkpoint_helper(monkeypatch):
    trainer = _build_mock_trainer()
    trainer.adapter = object()
    trainer._current_epoch_state = SimpleNamespace(value=1)
    trainer._current_step_state = SimpleNamespace(value=2)
    trainer.cfg.performance = SimpleNamespace(
        memory=SimpleNamespace(lowram=False),
        deepspeed=DeepSpeedConfig(deepspeed=True),
    )
    expected_resume_state = ResumeState(step=5, epoch=3)
    captured = {}

    def fake_register(accelerator, adapter, *, save_for_deepspeed, current_epoch, current_step):
        captured["accelerator"] = accelerator
        captured["adapter"] = adapter
        captured["save_for_deepspeed"] = save_for_deepspeed
        captured["current_epoch"] = current_epoch
        captured["current_step"] = current_step
        return expected_resume_state

    monkeypatch.setattr("library.training.modes.adapter_mode.register_adapter_checkpoint_state_hooks", fake_register)

    resume_state = AdapterMode().register_state_hooks(trainer)

    assert resume_state is expected_resume_state
    assert captured["adapter"] is trainer.adapter
    assert captured["save_for_deepspeed"] is True
    assert captured["current_epoch"] is trainer._current_epoch_state
    assert captured["current_step"] is trainer._current_step_state


def test_save_checkpoint_uses_repo_owned_adapter_export_seam(monkeypatch, tmp_path):
    trainer = _build_mock_trainer()
    trainer.cfg.output = SimpleNamespace(
        saving=SimpleNamespace(output_dir=str(tmp_path)),
        huggingface=None,
    )
    trainer.save_dtype = "fp16"
    trainer.adapter = object()
    trainer.accelerator.unwrap_model.return_value = trainer.adapter
    trainer.strategies.get_model_metadata.return_value = {"ss_model_spec": "sdxl"}
    metadata = {"base": "value"}
    captured = {}

    def fake_save_adapter_export(model_to_save, request):
        captured["model"] = model_to_save
        captured["request"] = request

    monkeypatch.setattr("library.training.modes.adapter_mode.save_adapter_export", fake_save_adapter_export)

    AdapterMode().save_checkpoint(
        trainer,
        ckpt_name="adapter.safetensors",
        step=12,
        epoch=3,
        metadata=metadata,
    )

    assert captured["model"] is trainer.adapter
    assert captured["request"].file == str(Path(tmp_path) / "adapter.safetensors")
    assert captured["request"].dtype == "fp16"
    assert captured["request"].metadata["base"] == "value"
    assert captured["request"].metadata["ss_steps"] == "12"
    assert captured["request"].metadata["ss_epoch"] == "3"
    assert captured["request"].metadata["ss_model_spec"] == "sdxl"
