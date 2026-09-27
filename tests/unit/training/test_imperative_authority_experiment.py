"""Test-only two-pass authority probe; not an executable Trainer extension API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch


_REGION_ACTIONS = frozenset({"first_backward", "temporary_edit", "intermediate_clear", "second_backward", "restore"})
_TRAINER_ACTIONS = frozenset({"initial_clear", "final_clip", "advance", "final_clear"})


@dataclass(frozen=True, slots=True)
class _AcceptedSplit:
    """A deliberately narrow test filing, not semantic parameter identity."""

    unit: str
    parameter: torch.nn.Parameter
    region_actions: frozenset[str]
    trainer_actions: frozenset[str]


@dataclass(frozen=True, slots=True)
class _Handback:
    second_backward_complete: bool
    parameter_restored: bool
    failure: Exception | None


def _accept_split(
    *,
    unit: str,
    parameter: torch.nn.Parameter,
    region_actions: frozenset[str] = _REGION_ACTIONS,
    trainer_actions: frozenset[str] = _TRAINER_ACTIONS,
) -> _AcceptedSplit:
    overlap = region_actions & trainer_actions
    if overlap:
        raise ValueError(f"duplicate ownership: {sorted(overlap)}")
    if region_actions != _REGION_ACTIONS or trainer_actions != _TRAINER_ACTIONS:
        raise ValueError("missing or unsupported phase authority for chosen split")
    if not unit:
        raise ValueError("optimization unit is required")
    return _AcceptedSplit(unit, parameter, region_actions, trainer_actions)


def _check_ready(
    split: _AcceptedSplit,
    *,
    current_unit: str,
    current_parameter: torch.nn.Parameter,
    backend_supports_scoped_edit: bool,
) -> None:
    if current_unit != split.unit or current_parameter is not split.parameter:
        raise ValueError("accepted unit or parameter scope is not current")
    if not backend_supports_scoped_edit:
        raise ValueError("backend cannot establish safe temporary-edit handback")
    if current_parameter.grad is not None:
        raise ValueError("this split requires a quiescent gradient window")


def _run_two_pass_region(
    split: _AcceptedSplit,
    objective: Callable[[torch.nn.Parameter], torch.Tensor],
    *,
    radius: float,
    events: list[str],
    fail_second_pass: bool = False,
    fail_restore: bool = False,
) -> _Handback:
    """The region receives scoped parameters and computation, never an optimizer."""

    parameter = split.parameter
    objective(parameter).backward()
    events.append("region:first_backward")
    assert parameter.grad is not None
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(radius * parameter.grad.sign())
    events.append("region:temporary_edit")

    parameter.grad = None
    events.append("region:intermediate_clear")
    second_complete = False
    failure: Exception | None = None
    try:
        if fail_second_pass:
            raise RuntimeError("second pass failed")
        objective(parameter).backward()
        events.append("region:second_backward")
        second_complete = True
    except Exception as error:
        failure = error

    restored = False
    try:
        if fail_restore:
            raise RuntimeError("restoration could not be confirmed")
        with torch.no_grad():
            parameter.copy_(original)
        events.append("region:restore")
        restored = True
    except Exception as error:
        failure = error
    return _Handback(second_complete, restored, failure)


def _trainer_finish(
    handback: _Handback,
    *,
    parameter: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    events: list[str],
) -> bool:
    if handback.failure is not None or not handback.parameter_restored or not handback.second_backward_complete or parameter.grad is None:
        events.append("trainer:stop")
        return False
    torch.nn.utils.clip_grad_norm_((parameter,), 100.0)
    events.append("trainer:final_clip")
    optimizer.step()
    events.append("trainer:advance")
    optimizer.zero_grad(set_to_none=True)
    events.append("trainer:final_clear")
    return True


@pytest.mark.training
@pytest.mark.unit
def test_two_pass_region_hands_restored_parameter_and_final_gradient_to_trainer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    split = _accept_split(unit="model", parameter=parameter)
    _check_ready(split, current_unit="model", current_parameter=parameter, backend_supports_scoped_edit=True)
    events: list[str] = []

    optimizer.zero_grad(set_to_none=True)
    events.append("trainer:initial_clear")
    handback = _run_two_pass_region(split, lambda value: value.square(), radius=0.1, events=events)
    torch.testing.assert_close(parameter.detach(), torch.tensor(1.0))
    torch.testing.assert_close(parameter.grad, torch.tensor(2.2))
    assert _trainer_finish(handback, parameter=parameter, optimizer=optimizer, events=events)
    torch.testing.assert_close(parameter.detach(), torch.tensor(0.78))
    assert events == [
        "trainer:initial_clear",
        "region:first_backward",
        "region:temporary_edit",
        "region:intermediate_clear",
        "region:second_backward",
        "region:restore",
        "trainer:final_clip",
        "trainer:advance",
        "trainer:final_clear",
    ]


@pytest.mark.training
@pytest.mark.unit
def test_missing_intermediate_clear_or_duplicate_advancement_is_rejected():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="missing or unsupported phase authority"):
        _accept_split(unit="model", parameter=parameter, region_actions=_REGION_ACTIONS - {"intermediate_clear"})
    with pytest.raises(ValueError, match="duplicate ownership"):
        _accept_split(unit="model", parameter=parameter, region_actions=_REGION_ACTIONS | {"advance"})
    with pytest.raises(ValueError, match="missing or unsupported phase authority"):
        _accept_split(
            unit="model",
            parameter=parameter,
            region_actions=_REGION_ACTIONS | {"advance"},
            trainer_actions=_TRAINER_ACTIONS - {"advance"},
        )


@pytest.mark.training
@pytest.mark.unit
def test_backend_or_accumulation_conflict_rejects_before_parameter_edit():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    split = _accept_split(unit="model", parameter=parameter)
    with pytest.raises(ValueError, match="cannot establish safe"):
        _check_ready(split, current_unit="model", current_parameter=parameter, backend_supports_scoped_edit=False)
    with pytest.raises(ValueError, match="not current"):
        _check_ready(split, current_unit="other", current_parameter=parameter, backend_supports_scoped_edit=True)

    parameter.grad = torch.tensor(5.0)
    with pytest.raises(ValueError, match="quiescent gradient window"):
        _check_ready(split, current_unit="model", current_parameter=parameter, backend_supports_scoped_edit=True)
    torch.testing.assert_close(parameter.detach(), torch.tensor(1.0))
    torch.testing.assert_close(parameter.grad, torch.tensor(5.0))


@pytest.mark.training
@pytest.mark.unit
def test_clean_restoration_after_second_pass_failure_does_not_authorize_step():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    split = _accept_split(unit="model", parameter=parameter)
    events: list[str] = []
    handback = _run_two_pass_region(split, lambda value: value.square(), radius=0.1, events=events, fail_second_pass=True)
    assert handback.parameter_restored
    assert not handback.second_backward_complete
    assert isinstance(handback.failure, RuntimeError)
    assert not _trainer_finish(handback, parameter=parameter, optimizer=optimizer, events=events)
    torch.testing.assert_close(parameter.detach(), torch.tensor(1.0))
    assert "trainer:advance" not in events


@pytest.mark.training
@pytest.mark.unit
def test_uncertain_restoration_withholds_live_use_and_snapshot():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    split = _accept_split(unit="model", parameter=parameter)
    events: list[str] = []
    handback = _run_two_pass_region(split, lambda value: value.square(), radius=0.1, events=events, fail_restore=True)
    assert handback.second_backward_complete
    assert not handback.parameter_restored
    assert isinstance(handback.failure, RuntimeError)
    assert not _trainer_finish(handback, parameter=parameter, optimizer=optimizer, events=events)
    assert "trainer:advance" not in events
    assert "region:restore" not in events
    assert parameter.detach().item() != 1.0  # The test sees this; a real backend needs trustworthy evidence.


@pytest.mark.training
@pytest.mark.unit
def test_clean_parameter_handback_without_final_gradient_cannot_advance():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    events: list[str] = []
    handback = _Handback(second_backward_complete=True, parameter_restored=True, failure=None)
    assert not _trainer_finish(handback, parameter=parameter, optimizer=optimizer, events=events)
    torch.testing.assert_close(parameter.detach(), torch.tensor(1.0))
    assert events == ["trainer:stop"]
