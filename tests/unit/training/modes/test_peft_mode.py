from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from library.optimization.types import OptimizerBuildResult
from library.adapters.runtime.targets import build_component_root_targets
from library.training.modes.peft_mode import PeftMode


def _build_mock_trainer():
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sdxl"),
        peft=SimpleNamespace(
            adapter_module="library.adapters.lora",
            base_weights=None,
            base_weights_multiplier=None,
            adapter_args=None,
            adapter_rank_from_weights=False,
            adapter_weights=None,
            adapter_rank=8,
            adapter_alpha=16.0,
            neuron_dropout=0.05,
            scale_weight_norms=False,
        ),
        optimizer=SimpleNamespace(
            learning_rates=SimpleNamespace(base=1e-4, denoiser=1e-4, text_encoders=[0.0, 1e-4], groups=None, groups_file=None),
            optimizer_args=None,
            scheduler=SimpleNamespace(),
        ),
        performance=SimpleNamespace(memory=SimpleNamespace(lowram=False)),
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

    def fake_build_adapter_for_legacy_module(module_path, request):
        captured["module_path"] = module_path
        captured["request"] = request
        return adapter

    monkeypatch.setattr(
        "library.training.modes.peft_mode.resolve_adapter_target_selection",
        lambda **_: SimpleNamespace(
            resolved_targets=resolved_targets,
            train_denoiser=False,
            train_any_text_encoder=True,
        ),
    )
    monkeypatch.setattr("library.training.modes.peft_mode.build_adapter_for_legacy_module", fake_build_adapter_for_legacy_module)

    PeftMode().prepare_trainables(trainer)

    assert captured["module_path"] == "library.adapters.lora"
    assert captured["request"].resolved_targets is resolved_targets
    assert trainer._train_text_encoder is True
    assert trainer._train_denoiser is False
    assert trainer.adapter_resolved_targets is captured["request"].resolved_targets
    adapter.apply_to.assert_called_once_with(trainer._text_encoder, trainer.denoiser, True, False)


def test_build_optimizer_params_uses_repo_owned_grouping_plan(monkeypatch):
    trainer = _build_mock_trainer()
    trainer.adapter = object()
    execution_groups = [MagicMock()]
    logical_groups = [MagicMock(metric_name="unet")]
    captured = {}

    monkeypatch.setattr(
        "library.training.modes.peft_mode.build_adapter_grouping",
        lambda **kwargs: SimpleNamespace(execution_groups=execution_groups, logical_groups=logical_groups),
    )

    def fake_get_optimizer(optimizer_config, learning_rates, scheduler_config, execution_group_payload, optimizer_kwargs):
        captured["execution_groups"] = execution_group_payload
        captured["optimizer_kwargs"] = optimizer_kwargs
        return "AdamW", {"lr": 1e-4}, "optimizer"

    monkeypatch.setattr("library.training.modes.peft_mode.get_optimizer", fake_get_optimizer)

    result = PeftMode().build_optimizer_params(trainer)

    assert isinstance(result, OptimizerBuildResult)
    assert result.optimizer_name == "AdamW"
    assert result.optimizer == "optimizer"
    assert result.optimization_plan.execution_groups == execution_groups
    assert result.optimization_plan.logical_groups == logical_groups
    assert result.lr_descriptions == ["unet"]
    assert captured["execution_groups"] == execution_groups
    assert captured["optimizer_kwargs"] == {}


def test_build_optimizer_params_rejects_legacy_built_in_optimizer_policy(monkeypatch):
    trainer = _build_mock_trainer()
    trainer.adapter = object()
    trainer.net_kwargs = {"down_lr_weight": "linear"}

    monkeypatch.setattr(
        "library.training.modes.peft_mode.build_adapter_grouping",
        lambda **kwargs: SimpleNamespace(execution_groups=[], logical_groups=[]),
    )

    with pytest.raises(NotImplementedError, match="adapter args"):
        PeftMode().build_optimizer_params(trainer)
