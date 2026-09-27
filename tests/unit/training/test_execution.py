"""Executable design tests for the candidate accepted-action mechanism."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pytest
import torch

from library.training.execution import (
    Action,
    ActionResult,
    ExecutionProfile,
    InvalidExecution,
    Mechanic,
    Operation,
    OperationFailed,
    OptimizationInput,
    STANDARD_PROFILE,
    Value,
    compile_run,
)


@dataclass(frozen=True, slots=True)
class _SharedLossPolicy:
    """Test-only accepted policy; not inferred from Python optimizer objects."""

    action: str
    unit_sources: Mapping[str, Value]
    advance_order: tuple[str, ...]
    clip_norms: Mapping[str, float]
    require_atomic: bool = False


@dataclass(frozen=True, slots=True)
class _UnitRuntime:
    """Test-only prepared handles; the keys, not these objects, name units."""

    parameters: tuple[torch.nn.Parameter, ...]
    step: Callable[[], object]
    zero_grad: Callable[[], object]


@dataclass(frozen=True, slots=True)
class _AdvanceReport:
    outcomes: tuple[tuple[str, str], ...]
    stopped: bool
    failure: Exception | None = None


def _advance_shared_loss(
    result: ActionResult,
    policy: _SharedLossPolicy,
    runtimes: Mapping[str, _UnitRuntime],
    events: list[str],
) -> _AdvanceReport:
    """Exercise one accepted shared-backward policy outside the operation runner.

    This deliberately omits accumulation, distributed synchronization, and
    backend-specific stepping. It does not define the production policy API.
    """

    if policy.require_atomic:
        raise ValueError("all-or-nothing advancement has no supported transaction")
    units = policy.advance_order
    if len(set(units)) != len(units) or result.action != policy.action:
        raise ValueError("accepted action or unit order mismatch")
    if set(units) != set(policy.unit_sources) or set(units) != set(policy.clip_norms) or not set(units) <= set(runtimes):
        raise ValueError("accepted optimization policy or prepared units are incomplete")

    offered = {item.unit: item for item in result.optimization}
    if len(offered) != len(result.optimization) or set(offered) != set(units):
        raise ValueError("optimization result does not match due units")
    if len(set(policy.unit_sources.values())) != 1:
        raise ValueError("this test policy supports one shared backward value")
    for unit in units:
        if offered[unit].source != policy.unit_sources[unit]:
            raise ValueError("optimization input source differs from accepted policy")

    shared_value = offered[units[0]].value
    if not isinstance(shared_value, torch.Tensor) or any(offered[unit].value is not shared_value for unit in units):
        raise ValueError("accepted shared source did not produce one differentiable value")

    parameters = [parameter for unit in units for parameter in runtimes[unit].parameters]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("prepared optimizer units overlap")

    for unit in units:
        runtimes[unit].zero_grad()
        events.append(f"zero:{unit}")
    shared_value.backward()
    events.append("backward")
    for unit in units:
        torch.nn.utils.clip_grad_norm_(runtimes[unit].parameters, policy.clip_norms[unit])
        events.append(f"clip:{unit}")

    outcomes: list[tuple[str, str]] = []
    for position, unit in enumerate(units):
        events.append(f"attempt:{unit}")
        try:
            runtimes[unit].step()
        except Exception as error:
            outcomes.append((unit, "uncertain"))
            outcomes.extend((later, "not_attempted") for later in units[position + 1 :])
            return _AdvanceReport(tuple(outcomes), stopped=True, failure=error)
        outcomes.append((unit, "step_returned"))
    return _AdvanceReport(tuple(outcomes), stopped=False)


def _partition_action(matrix: torch.nn.Parameter, other: torch.nn.Parameter) -> tuple[Action, Value, Value]:
    batch = Value("batch")
    loss = Value("loss")
    action = Action(
        name="joint_partition",
        inputs=(batch,),
        operations=(Operation("objective", lambda data: (matrix * data + other).square().mean(), (batch,), (loss,)),),
        optimization=(OptimizationInput("matrix_unit", loss), OptimizationInput("other_unit", loss)),
    )
    return action, batch, loss


@pytest.mark.training
@pytest.mark.unit
def test_sdxl_like_work_is_selected_once_and_optimization_stays_outside_operations():
    batch = Value("batch")
    latents = Value("latents")
    conditioning = Value("conditioning")
    loss = Value("loss")
    timestep = Value("timestep")
    weight = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([weight], lr=0.1)

    action = Action(
        name="ordinary_training",
        inputs=(batch,),
        operations=(
            Operation("represent", lambda data: data["pixels"] / 2, (batch,), (latents,)),
            Operation("condition", lambda data: data["prompt"], (batch,), (conditioning,)),
            Operation(
                "objective",
                lambda latent, prompt: ((weight * latent + prompt) - 1).square().mean(),
                (latents, conditioning),
                (loss,),
            ),
            Operation("observe_timestep", lambda data: data["timestep"], (batch,), (timestep,)),
        ),
        optimization=(OptimizationInput("denoiser", loss),),
        observations=(timestep,),
    )
    prepared = compile_run((action,), due=lambda _coordinates: ("ordinary_training",))
    result = prepared.due_actions({"batch": 0})[0].execute(
        {batch: {"pixels": torch.tensor([2.0]), "prompt": torch.tensor([0.0]), "timestep": 17}}
    )

    assert result.action == "ordinary_training"
    assert result.observations[timestep] == 17
    assert [item.unit for item in result.optimization] == ["denoiser"]
    assert weight.grad is None

    # The operation returns an optimization input; the Trainer-owned side acts on it.
    optimization_value = result.optimization[0].value
    assert isinstance(optimization_value, torch.Tensor)
    optimization_value.backward()
    optimizer.step()
    assert weight.item() < 2.0


@pytest.mark.training
@pytest.mark.unit
def test_alternating_actions_use_the_same_mechanism_with_distinct_units():
    batch = Value("batch")
    generator_loss = Value("generator_loss")
    discriminator_loss = Value("discriminator_loss")
    generator = torch.nn.Parameter(torch.tensor(1.0))
    discriminator = torch.nn.Parameter(torch.tensor(2.0))
    optimizers = {
        "generator": torch.optim.SGD([generator], lr=0.1),
        "discriminator": torch.optim.SGD([discriminator], lr=0.1),
    }
    runtimes = {
        "generator": _UnitRuntime((generator,), optimizers["generator"].step, optimizers["generator"].zero_grad),
        "discriminator": _UnitRuntime((discriminator,), optimizers["discriminator"].step, optimizers["discriminator"].zero_grad),
    }

    actions = (
        Action(
            name="generator",
            inputs=(batch,),
            operations=(
                Operation(
                    "generator_objective",
                    lambda data: (generator * discriminator.detach() * data).square(),
                    (batch,),
                    (generator_loss,),
                ),
            ),
            optimization=(OptimizationInput("generator", generator_loss),),
        ),
        Action(
            name="discriminator",
            inputs=(batch,),
            operations=(
                Operation(
                    "discriminator_objective",
                    lambda data: (discriminator * generator.detach() * data - 3).square(),
                    (batch,),
                    (discriminator_loss,),
                ),
            ),
            optimization=(OptimizationInput("discriminator", discriminator_loss),),
        ),
    )
    prepared = compile_run(actions, due=lambda coordinates: ("generator",) if coordinates["turn"] % 2 == 0 else ("discriminator",))

    before_generator = generator.item()
    before_discriminator = discriminator.item()
    first = prepared.due_actions({"turn": 0})[0].execute({batch: torch.tensor(1.0)})
    events: list[str] = []
    first_report = _advance_shared_loss(
        first,
        _SharedLossPolicy("generator", {"generator": generator_loss}, ("generator",), {"generator": 100.0}),
        runtimes,
        events,
    )
    assert first_report.outcomes == (("generator", "step_returned"),)
    assert events == ["zero:generator", "backward", "clip:generator", "attempt:generator"]
    assert generator.item() != before_generator
    assert discriminator.item() == before_discriminator

    after_generator = generator.item()
    second = prepared.due_actions({"turn": 1})[0].execute({batch: torch.tensor(1.0)})
    events.clear()
    second_report = _advance_shared_loss(
        second,
        _SharedLossPolicy("discriminator", {"discriminator": discriminator_loss}, ("discriminator",), {"discriminator": 100.0}),
        runtimes,
        events,
    )
    assert second_report.outcomes == (("discriminator", "step_returned"),)
    assert events == ["zero:discriminator", "backward", "clip:discriminator", "attempt:discriminator"]
    assert generator.item() == after_generator
    assert discriminator.item() != before_discriminator


@pytest.mark.training
@pytest.mark.unit
def test_one_action_can_feed_two_units_with_one_accepted_backward():
    matrix = torch.nn.Parameter(torch.tensor(1.0))
    other = torch.nn.Parameter(torch.tensor(0.5))
    action, batch, loss = _partition_action(matrix, other)
    prepared = compile_run((action,), due=lambda _: ("joint_partition",))
    result = prepared.due_actions({})[0].execute({batch: torch.tensor(2.0)})
    assert [(item.unit, item.source) for item in result.optimization] == [
        ("matrix_unit", loss),
        ("other_unit", loss),
    ]

    # SGD stands in for a matrix/Muon responsibility; this is not a Muon test.
    matrix_optimizer = torch.optim.SGD([matrix], lr=0.1)
    other_optimizer = torch.optim.AdamW([other], lr=0.1)
    runtimes = {
        "matrix_unit": _UnitRuntime((matrix,), matrix_optimizer.step, matrix_optimizer.zero_grad),
        "other_unit": _UnitRuntime((other,), other_optimizer.step, other_optimizer.zero_grad),
    }
    policy = _SharedLossPolicy(
        action="joint_partition",
        unit_sources={"matrix_unit": loss, "other_unit": loss},
        advance_order=("matrix_unit", "other_unit"),
        clip_norms={"matrix_unit": 100.0, "other_unit": 100.0},
    )
    backward_calls: list[torch.Tensor] = []
    optimization_value = result.optimization[0].value
    assert isinstance(optimization_value, torch.Tensor)
    optimization_value.register_hook(lambda gradient: backward_calls.append(gradient.clone()))
    events: list[str] = []
    report = _advance_shared_loss(result, policy, runtimes, events)

    assert report.outcomes == (("matrix_unit", "step_returned"), ("other_unit", "step_returned"))
    assert not report.stopped
    assert len(backward_calls) == 1
    assert events == [
        "zero:matrix_unit",
        "zero:other_unit",
        "backward",
        "clip:matrix_unit",
        "clip:other_unit",
        "attempt:matrix_unit",
        "attempt:other_unit",
    ]
    assert matrix.item() != 1.0
    assert other.item() != 0.5


@pytest.mark.training
@pytest.mark.unit
def test_second_unit_failure_stops_without_claiming_atomicity_or_replay():
    matrix = torch.nn.Parameter(torch.tensor(1.0))
    other = torch.nn.Parameter(torch.tensor(0.5))
    action, batch, loss = _partition_action(matrix, other)
    result = compile_run((action,), due=lambda _: ("joint_partition",)).actions["joint_partition"].execute({batch: torch.tensor(2.0)})
    matrix_optimizer = torch.optim.SGD([matrix], lr=0.1)
    other_optimizer = torch.optim.AdamW([other], lr=0.1)

    def fail_after_other_step() -> None:
        other_optimizer.step()
        raise RuntimeError("backend lost confirmation after stepping")

    runtimes = {
        "matrix_unit": _UnitRuntime((matrix,), matrix_optimizer.step, matrix_optimizer.zero_grad),
        "other_unit": _UnitRuntime((other,), fail_after_other_step, other_optimizer.zero_grad),
    }
    policy = _SharedLossPolicy(
        "joint_partition",
        {"matrix_unit": loss, "other_unit": loss},
        ("matrix_unit", "other_unit"),
        {"matrix_unit": 100.0, "other_unit": 100.0},
    )
    events: list[str] = []
    report = _advance_shared_loss(result, policy, runtimes, events)

    assert report.outcomes == (("matrix_unit", "step_returned"), ("other_unit", "uncertain"))
    assert report.stopped
    assert isinstance(report.failure, RuntimeError)
    assert events[-2:] == ["attempt:matrix_unit", "attempt:other_unit"]
    assert matrix.item() != 1.0
    assert other.item() != 0.5  # The test knows; the consumer cannot infer this from a failed call.


@pytest.mark.training
@pytest.mark.unit
def test_shared_backward_policy_rejects_mismatched_source_before_mutation():
    matrix = torch.nn.Parameter(torch.tensor(1.0))
    other = torch.nn.Parameter(torch.tensor(0.5))
    action, batch, loss = _partition_action(matrix, other)
    result = compile_run((action,), due=lambda _: ("joint_partition",)).actions["joint_partition"].execute({batch: torch.tensor(2.0)})
    matrix_optimizer = torch.optim.SGD([matrix], lr=0.1)
    other_optimizer = torch.optim.AdamW([other], lr=0.1)
    runtimes = {
        "matrix_unit": _UnitRuntime((matrix,), matrix_optimizer.step, matrix_optimizer.zero_grad),
        "other_unit": _UnitRuntime((other,), other_optimizer.step, other_optimizer.zero_grad),
    }
    policy = _SharedLossPolicy(
        "joint_partition",
        {"matrix_unit": Value("different_loss"), "other_unit": Value("different_loss")},
        ("matrix_unit", "other_unit"),
        {"matrix_unit": 100.0, "other_unit": 100.0},
    )
    events: list[str] = []

    with pytest.raises(ValueError, match="source differs from accepted policy"):
        _advance_shared_loss(result, policy, runtimes, events)
    assert events == []
    assert matrix.grad is None and other.grad is None

    valid_policy = _SharedLossPolicy(
        "joint_partition",
        {"matrix_unit": loss, "other_unit": loss},
        ("matrix_unit", "other_unit"),
        {"matrix_unit": 100.0, "other_unit": 100.0},
    )
    overlapping = {
        "matrix_unit": runtimes["matrix_unit"],
        "other_unit": _UnitRuntime((matrix,), other_optimizer.step, other_optimizer.zero_grad),
    }
    with pytest.raises(ValueError, match="prepared optimizer units overlap"):
        _advance_shared_loss(result, valid_policy, overlapping, events)
    assert events == []


@pytest.mark.training
@pytest.mark.unit
def test_all_or_nothing_request_is_rejected_before_any_step():
    matrix = torch.nn.Parameter(torch.tensor(1.0))
    other = torch.nn.Parameter(torch.tensor(0.5))
    action, batch, loss = _partition_action(matrix, other)
    result = compile_run((action,), due=lambda _: ("joint_partition",)).actions["joint_partition"].execute({batch: torch.tensor(2.0)})
    matrix_optimizer = torch.optim.SGD([matrix], lr=0.1)
    other_optimizer = torch.optim.AdamW([other], lr=0.1)
    runtimes = {
        "matrix_unit": _UnitRuntime((matrix,), matrix_optimizer.step, matrix_optimizer.zero_grad),
        "other_unit": _UnitRuntime((other,), other_optimizer.step, other_optimizer.zero_grad),
    }
    policy = _SharedLossPolicy(
        "joint_partition",
        {"matrix_unit": loss, "other_unit": loss},
        ("matrix_unit", "other_unit"),
        {"matrix_unit": 100.0, "other_unit": 100.0},
        require_atomic=True,
    )
    events: list[str] = []

    with pytest.raises(ValueError, match="no supported transaction"):
        _advance_shared_loss(result, policy, runtimes, events)
    assert events == []
    assert matrix.grad is None and other.grad is None
    assert matrix.item() == 1.0 and other.item() == 0.5


@pytest.mark.training
@pytest.mark.unit
def test_standard_profile_rejects_requested_backward_authority():
    value = Value("input")
    operation = Operation(
        "two_pass_computation",
        lambda item: item,
        (value,),
        (Value("output"),),
        requested_authority=frozenset({Mechanic.BACKWARD, Mechanic.PARAMETER_EDIT}),
    )
    action = Action("two_pass", inputs=(value,), operations=(operation,))

    with pytest.raises(InvalidExecution, match="ungranted authority"):
        compile_run((action,), due=lambda _: ("two_pass",), profile=STANDARD_PROFILE)

    extension = ExecutionProfile(
        trainer_owns=frozenset({Mechanic.ADVANCE, Mechanic.ZERO_GRAD}),
        operations_may_own=frozenset({Mechanic.BACKWARD, Mechanic.PARAMETER_EDIT}),
    )
    prepared = compile_run((action,), due=lambda _: ("two_pass",), profile=extension)
    assert prepared.due_actions({})[0].name == "two_pass"
    assert prepared.profile is extension
    assert Mechanic.GRADIENT_EDIT not in prepared.profile.trainer_owns | prepared.profile.operations_may_own

    with pytest.raises(ValueError, match="cannot both own BACKWARD"):
        ExecutionProfile(
            trainer_owns=frozenset({Mechanic.BACKWARD}),
            operations_may_own=frozenset({Mechanic.BACKWARD}),
        )


@pytest.mark.training
@pytest.mark.unit
def test_wiring_and_runtime_choice_are_bounded():
    batch = Value("batch")
    missing = Value("missing")
    action = Action("train", inputs=(batch,), operations=(Operation("compute", lambda value: value, (missing,), (Value("out"),)),))
    with pytest.raises(InvalidExecution, match="unavailable values"):
        compile_run((action,), due=lambda _: ("train",))

    valid = Action("train", inputs=(batch,), operations=())
    prepared = compile_run((valid,), due=lambda _: ("unknown",))
    with pytest.raises(InvalidExecution, match="unaccepted action"):
        prepared.due_actions({})
    with pytest.raises(InvalidExecution, match="input mismatch"):
        prepared.actions["train"].execute({batch: 1, Value("whole_trainer"): object()})


@pytest.mark.training
@pytest.mark.unit
def test_operation_failure_does_not_imply_rollback():
    input_value = Value("input")
    effects: list[str] = []

    def mutates_then_fails(_value: object) -> None:
        effects.append("already happened")
        raise RuntimeError("failure")

    action = Action("train", inputs=(input_value,), operations=(Operation("risky", mutates_then_fails, (input_value,), ()),))
    prepared = compile_run((action,), due=lambda _: ("train",))

    with pytest.raises(OperationFailed, match="live effects may remain") as error:
        prepared.actions["train"].execute({input_value: object()})
    assert error.value.operation == "risky"
    assert effects == ["already happened"]


@pytest.mark.training
@pytest.mark.unit
def test_declared_output_mismatch_is_execution_time_failure_after_possible_effects():
    input_value = Value("input")
    effects: list[str] = []

    def wrong_shape(_value: object) -> tuple[int]:
        effects.append("already happened")
        return (1,)

    action = Action(
        "train",
        inputs=(input_value,),
        operations=(Operation("wrong_shape", wrong_shape, (input_value,), (Value("first"), Value("second"))),),
    )
    prepared = compile_run((action,), due=lambda _: ("train",))

    with pytest.raises(OperationFailed, match="declared 2 outputs; live effects may remain"):
        prepared.actions["train"].execute({input_value: object()})
    assert effects == ["already happened"]

    no_output = Action(
        "train",
        inputs=(input_value,),
        operations=(Operation("unexpected_result", lambda _: effects.append("effect") or 1, (input_value,), ()),),
    )
    prepared = compile_run((no_output,), due=lambda _: ("train",))
    with pytest.raises(OperationFailed, match="declared no output but returned a value; live effects may remain"):
        prepared.actions["train"].execute({input_value: object()})
    assert effects == ["already happened", "effect"]


@pytest.mark.training
@pytest.mark.unit
def test_multi_output_and_no_output_operations_preserve_wiring():
    input_value = Value("input")
    left = Value("left")
    right = Value("right")
    seen: list[int] = []
    action = Action(
        "train",
        inputs=(input_value,),
        operations=(
            Operation("split", lambda value: (value + 1, value + 2), (input_value,), (left, right)),
            Operation("record", lambda value: seen.append(value), (right,), ()),
        ),
        observations=(left, right),
    )
    result = compile_run((action,), due=lambda _: ("train",)).actions["train"].execute({input_value: 3})
    assert dict(result.observations) == {left: 4, right: 5}
    assert seen == [5]


@pytest.mark.training
@pytest.mark.unit
def test_duplicate_filing_and_due_action_are_rejected_at_the_earliest_candidate_boundary():
    input_value = Value("input")
    result_value = Value("result")
    operation = Operation("compute", lambda value: value, (input_value,), (result_value,))
    duplicated_input = OptimizationInput("unit", result_value)

    with pytest.raises(InvalidExecution, match="duplicate optimization input"):
        compile_run(
            (Action("train", (input_value,), (operation,), optimization=(duplicated_input, duplicated_input)),),
            due=lambda _: ("train",),
        )
    with pytest.raises(InvalidExecution, match="duplicate observation"):
        compile_run(
            (Action("train", (input_value,), (operation,), observations=(result_value, result_value)),),
            due=lambda _: ("train",),
        )
    prepared = compile_run((Action("train", (input_value,), (operation,)),), due=lambda _: ("train", "train"))
    with pytest.raises(InvalidExecution, match="more than once"):
        prepared.due_actions({})
