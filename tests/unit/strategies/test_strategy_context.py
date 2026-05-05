from __future__ import annotations

from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from library.strategies.base.context import (
    DenoiserContext,
    StrategyContext,
    StrategyPhase,
    TrainingContext,
    current_strategy_context,
    publish_strategy_context,
    require_strategy_context,
)
from library.strategies.base.contracts import DenoiserCallingStrategy


class _DummyDenoiserCallingStrategy(DenoiserCallingStrategy):
    def __init__(self, callback):
        self._callback = callback

    def call_denoiser(
        self,
        cfg,
        accelerator,
        denoiser,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        *,
        phase,
        global_step,
        is_train,
        train_denoiser=True,
        sample_indices=None,
        enable_grad=None,
    ) -> torch.Tensor:
        return self._callback(
            cfg=cfg,
            accelerator=accelerator,
            denoiser=denoiser,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_conds=text_conds,
            batch=batch,
            weight_dtype=weight_dtype,
            phase=phase,
            global_step=global_step,
            is_train=is_train,
            train_denoiser=train_denoiser,
            sample_indices=sample_indices,
            enable_grad=enable_grad,
        )


@pytest.mark.unit
def test_current_strategy_context_returns_none_without_active_scope() -> None:
    assert current_strategy_context() is None


@pytest.mark.unit
def test_require_strategy_context_raises_without_active_scope() -> None:
    with pytest.raises(RuntimeError, match="No strategy context is active"):
        require_strategy_context()


@pytest.mark.unit
def test_publish_strategy_context_restores_absent_context_after_scope() -> None:
    context = StrategyContext(phase=StrategyPhase.TRAIN, training=TrainingContext(global_step=12, is_train=True))

    with publish_strategy_context(context) as active_context:
        assert active_context is context
        assert current_strategy_context() is context
        assert require_strategy_context() is context

    assert current_strategy_context() is None


@pytest.mark.unit
def test_publish_strategy_context_restores_outer_context_after_nested_scope() -> None:
    outer = StrategyContext(phase=StrategyPhase.TRAIN, model_family="sdxl")
    inner = StrategyContext(
        phase=StrategyPhase.VALIDATION,
        model_family="sdxl",
        denoiser=DenoiserContext(timesteps=torch.tensor([50, 350]), batch_size=2),
    )

    with publish_strategy_context(outer):
        assert current_strategy_context() is outer
        with publish_strategy_context(inner):
            assert current_strategy_context() is inner
        assert current_strategy_context() is outer

    assert current_strategy_context() is None


@pytest.mark.unit
def test_strategy_context_records_are_read_only() -> None:
    context = StrategyContext(
        phase=StrategyPhase.TRAIN,
        model_family="sdxl",
        training=TrainingContext(global_step=3, is_train=True),
        denoiser=DenoiserContext(timesteps=torch.tensor([1, 2]), sample_indices=(0, 1), batch_size=2),
    )

    with pytest.raises(FrozenInstanceError):
        context.model_family = "sd3"
    assert context.training is not None
    with pytest.raises(FrozenInstanceError):
        context.training.global_step = 4
    assert context.denoiser is not None
    with pytest.raises(FrozenInstanceError):
        context.denoiser.batch_size = 3


@pytest.mark.unit
def test_definition_only_denoiser_strategy_does_not_publish_context_by_itself() -> None:
    observed_contexts: list[StrategyContext | None] = []

    def call_denoiser(**kwargs) -> torch.Tensor:
        del kwargs
        observed_contexts.append(current_strategy_context())
        return torch.ones(2, 4)

    strategy = _DummyDenoiserCallingStrategy(call_denoiser)
    output = strategy.call_denoiser(
        cfg=SimpleNamespace(model=SimpleNamespace(model_type="sdxl")),
        accelerator=SimpleNamespace(autocast=lambda: nullcontext()),
        denoiser=object(),
        noisy_latents=torch.zeros(2, 4),
        timesteps=torch.tensor([1, 2]),
        text_conds=object(),
        batch={},
        weight_dtype=torch.float32,
        phase=StrategyPhase.TRAIN,
        global_step=3,
        is_train=True,
    )

    assert torch.equal(output, torch.ones(2, 4))
    assert observed_contexts == [None]
    assert current_strategy_context() is None
