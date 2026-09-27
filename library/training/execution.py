"""Candidate operation wiring for the accepted training execution design.

This module is not used by the active Trainer. It tests whether different
training actions can share a small Python execution mechanism before the
OpenSpec design chooses its production representation. ``compile_run`` checks
candidate wiring and declared authority; it is not contract fulfillment or
runtime/backend preparation.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Value:
    """One action-local input or result, independent of its Python payload."""

    name: str


class Mechanic(Enum):
    """Probe-sized authority vocabulary, not every Trainer mechanic."""

    BACKWARD = auto()
    GRADIENT_EDIT = auto()
    PARAMETER_EDIT = auto()
    ADVANCE = auto()
    ZERO_GRAD = auto()


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """A test-only ownership split; omitted mechanics are unsupported."""

    trainer_owns: frozenset[Mechanic]
    operations_may_own: frozenset[Mechanic]

    def __post_init__(self) -> None:
        overlap = self.trainer_owns & self.operations_may_own
        if overlap:
            names = ", ".join(sorted(mechanic.name for mechanic in overlap))
            raise ValueError(f"Trainer and operations cannot both own {names}")


STANDARD_PROFILE = ExecutionProfile(
    trainer_owns=frozenset(Mechanic),
    operations_may_own=frozenset(),
)


@dataclass(frozen=True, slots=True)
class Operation:
    """Selected Python work with explicit cross-operation value wiring."""

    name: str
    run: Callable[..., object]
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    requested_authority: frozenset[Mechanic] = frozenset()


@dataclass(frozen=True, slots=True)
class OptimizationInput:
    """A value offered to one accepted optimization-unit address."""

    unit: str
    value: Value


@dataclass(frozen=True, slots=True)
class Action:
    """Ordered selected work and the values another owner needs afterward."""

    name: str
    inputs: tuple[Value, ...]
    operations: tuple[Operation, ...]
    optimization: tuple[OptimizationInput, ...] = ()
    observations: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class OptimizationValue:
    """One unit's offered value with its action-local semantic source."""

    unit: str
    source: Value
    value: object


@dataclass(frozen=True, slots=True)
class ActionResult:
    action: str
    optimization: tuple[OptimizationValue, ...]
    observations: Mapping[Value, object]


class InvalidExecution(ValueError):
    """The candidate wiring or an accepted runtime choice is invalid."""


class OperationFailed(RuntimeError):
    """Selected work failed; no rollback of its possible effects is implied."""

    def __init__(self, operation: str, reason: str | None = None) -> None:
        detail = f": {reason}" if reason is not None else ""
        super().__init__(f"Operation {operation!r} failed{detail}; live effects may remain")
        self.operation = operation


@dataclass(frozen=True, slots=True)
class _ReadyOperation:
    operation: Operation
    input_slots: tuple[int, ...]
    output_slots: tuple[int, ...]


_MISSING = object()


@dataclass(frozen=True, slots=True)
class PreparedAction:
    """Static wiring resolved once; execution receives only changing values."""

    name: str
    input_slots: tuple[tuple[Value, int], ...]
    operations: tuple[_ReadyOperation, ...]
    optimization_slots: tuple[tuple[str, Value, int], ...]
    observation_slots: tuple[tuple[Value, int], ...]
    slot_count: int

    def execute(self, inputs: Mapping[Value, object]) -> ActionResult:
        required = {value for value, _ in self.input_slots}
        if set(inputs) != required:
            missing = sorted(value.name for value in required - inputs.keys())
            extra = sorted(value.name for value in inputs.keys() - required)
            raise InvalidExecution(f"Action {self.name!r} input mismatch: missing={missing}, extra={extra}")

        slots: list[object] = [_MISSING] * self.slot_count
        for value, slot in self.input_slots:
            slots[slot] = inputs[value]

        for ready in self.operations:
            try:
                produced = ready.operation.run(*(slots[slot] for slot in ready.input_slots))
            except Exception as error:
                raise OperationFailed(ready.operation.name) from error
            if len(ready.output_slots) == 0:
                if produced is not None:
                    raise OperationFailed(ready.operation.name, "declared no output but returned a value")
                continue
            if len(ready.output_slots) == 1:
                slots[ready.output_slots[0]] = produced
                continue
            if not isinstance(produced, tuple) or len(produced) != len(ready.output_slots):
                raise OperationFailed(ready.operation.name, f"declared {len(ready.output_slots)} outputs")
            for slot, value in zip(ready.output_slots, produced):
                slots[slot] = value

        return ActionResult(
            action=self.name,
            optimization=tuple(OptimizationValue(unit, source, slots[slot]) for unit, source, slot in self.optimization_slots),
            observations=MappingProxyType({value: slots[slot] for value, slot in self.observation_slots}),
        )


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """Selected actions and a bounded Python due-action policy."""

    actions: Mapping[str, PreparedAction]
    due: Callable[[Mapping[str, int]], Sequence[str]]
    profile: ExecutionProfile

    def due_actions(self, coordinates: Mapping[str, int]) -> tuple[PreparedAction, ...]:
        names = tuple(self.due(coordinates))
        if len(names) != len(set(names)):
            raise InvalidExecution("Due policy selected an action more than once")
        try:
            return tuple(self.actions[name] for name in names)
        except KeyError as error:
            raise InvalidExecution(f"Due policy selected unaccepted action {error.args[0]!r}") from error


def compile_run(
    actions: Sequence[Action],
    due: Callable[[Mapping[str, int]], Sequence[str]],
    *,
    profile: ExecutionProfile = STANDARD_PROFILE,
) -> PreparedRun:
    """Check and pre-resolve one candidate run's value and authority wiring."""

    return PreparedRun(actions=_compile_actions(actions, profile), due=due, profile=profile)


def _compile_actions(actions: Sequence[Action], profile: ExecutionProfile) -> Mapping[str, PreparedAction]:
    prepared: dict[str, PreparedAction] = {}
    for action in actions:
        if action.name in prepared:
            raise InvalidExecution(f"Duplicate action {action.name!r}")
        prepared[action.name] = _compile_action(action, profile)
    if not prepared:
        raise InvalidExecution("A run needs at least one action")
    return MappingProxyType(prepared)


def _compile_action(action: Action, profile: ExecutionProfile) -> PreparedAction:
    slots: dict[Value, int] = {}
    for value in action.inputs:
        if value in slots:
            raise InvalidExecution(f"Action {action.name!r} has duplicate input {value.name!r}")
        slots[value] = len(slots)

    ready_operations: list[_ReadyOperation] = []
    operation_names: set[str] = set()
    for operation in action.operations:
        if operation.name in operation_names:
            raise InvalidExecution(f"Action {action.name!r} has duplicate operation {operation.name!r}")
        operation_names.add(operation.name)
        unauthorized = operation.requested_authority - profile.operations_may_own
        if unauthorized:
            names = ", ".join(sorted(mechanic.name for mechanic in unauthorized))
            raise InvalidExecution(f"Operation {operation.name!r} requests ungranted authority: {names}")
        if any(value not in slots for value in operation.inputs):
            missing = [value.name for value in operation.inputs if value not in slots]
            raise InvalidExecution(f"Operation {operation.name!r} reads unavailable values: {missing}")
        input_slots = tuple(slots[value] for value in operation.inputs)
        output_slots: list[int] = []
        for value in operation.outputs:
            if value in slots:
                raise InvalidExecution(f"Operation {operation.name!r} overwrites value {value.name!r}")
            slots[value] = len(slots)
            output_slots.append(slots[value])
        ready_operations.append(_ReadyOperation(operation, input_slots, tuple(output_slots)))

    requested_inputs: set[tuple[str, Value]] = set()
    for requested in action.optimization:
        address = (requested.unit, requested.value)
        if address in requested_inputs:
            raise InvalidExecution(f"Action {action.name!r} has duplicate optimization input for {requested.unit!r}")
        requested_inputs.add(address)
        if requested.value not in slots:
            raise InvalidExecution(f"Action {action.name!r} has unavailable optimization input {requested.value.name!r}")
    observed_values: set[Value] = set()
    for observed in action.observations:
        if observed in observed_values:
            raise InvalidExecution(f"Action {action.name!r} has duplicate observation {observed.name!r}")
        observed_values.add(observed)
        if observed not in slots:
            raise InvalidExecution(f"Action {action.name!r} has unavailable observation {observed.name!r}")

    return PreparedAction(
        name=action.name,
        input_slots=tuple((value, slots[value]) for value in action.inputs),
        operations=tuple(ready_operations),
        optimization_slots=tuple((request.unit, request.value, slots[request.value]) for request in action.optimization),
        observation_slots=tuple((value, slots[value]) for value in action.observations),
        slot_count=len(slots),
    )


# The following run-level candidate deliberately uses the action mechanism above
# without making its action-local due callback the whole-run scheduler.


@dataclass(frozen=True, slots=True)
class InputRequest:
    """Work identity and the current dependencies required by one consumer."""

    work: Hashable
    dependencies: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class InputDelivery:
    """A produced value is available; admission is still a separate decision."""

    work: Hashable
    values: Mapping[Value, object]
    dependencies: Mapping[str, object]


class InputProvider(Protocol):
    """Selected input behavior owns its workers, queue, and continuation state."""

    def start(self) -> None: ...

    def take(self, work: Hashable) -> InputDelivery | None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class InputActivity:
    """The run-visible boundary of a longer-lived selected input producer."""

    name: str
    outputs: tuple[Value, ...]
    dependency_keys: tuple[str, ...]
    provider: InputProvider
    admit: Callable[[InputRequest, InputDelivery], bool]


@dataclass(frozen=True, slots=True)
class InputFeed:
    """Connect one input activity's values to an accepted action."""

    action: str
    activity: str


@dataclass(frozen=True, slots=True)
class DueWork:
    action: str
    input: InputRequest


@dataclass(frozen=True, slots=True)
class RunDescription:
    """Test-only semantic run subset, not a fulfilled strategy or final API."""

    actions: tuple[Action, ...]
    inputs: tuple[InputActivity, ...]
    feeds: tuple[InputFeed, ...]
    due: Callable[[Mapping[str, int]], Sequence[DueWork]]
    profile: ExecutionProfile = STANDARD_PROFILE


@dataclass(frozen=True, slots=True)
class PreparedWorkflow:
    """Static links and action wiring resolved before repeated coordination."""

    actions: Mapping[str, PreparedAction]
    inputs: Mapping[str, InputActivity]
    feeds: Mapping[str, InputActivity]
    due: Callable[[Mapping[str, int]], Sequence[DueWork]]


def prepare_workflow(description: RunDescription) -> PreparedWorkflow:
    """Check a small whole-run shape without claiming contract fulfillment."""

    actions = _compile_actions(description.actions, description.profile)
    inputs: dict[str, InputActivity] = {}
    for activity in description.inputs:
        if activity.name in inputs or len(set(activity.outputs)) != len(activity.outputs):
            raise InvalidExecution(f"Duplicate input activity or output in {activity.name!r}")
        if len(set(activity.dependency_keys)) != len(activity.dependency_keys):
            raise InvalidExecution(f"Duplicate dependency in input activity {activity.name!r}")
        inputs[activity.name] = activity

    feeds: dict[str, InputActivity] = {}
    for feed in description.feeds:
        if feed.action not in actions or feed.activity not in inputs:
            raise InvalidExecution(f"Unknown action or input activity in feed {feed!r}")
        if feed.action in feeds:
            raise InvalidExecution(f"Action {feed.action!r} has more than one input feed in this candidate")
        activity = inputs[feed.activity]
        action_inputs = {value for value, _ in actions[feed.action].input_slots}
        if set(activity.outputs) != action_inputs:
            raise InvalidExecution(f"Input activity {activity.name!r} does not supply action {feed.action!r}")
        feeds[feed.action] = activity
    if set(feeds) != set(actions):
        raise InvalidExecution("Every action needs one explicit input feed in this candidate")
    if {activity.name for activity in feeds.values()} != set(inputs):
        raise InvalidExecution("Every input activity needs an explicit feed in this candidate")

    return PreparedWorkflow(
        actions=actions,
        inputs=MappingProxyType(inputs),
        feeds=MappingProxyType(feeds),
        due=description.due,
    )


@dataclass(frozen=True, slots=True)
class UnitOutcome:
    """An optimizer outcome, not evidence of numerical parameter change."""

    unit: str
    status: str


@dataclass(frozen=True, slots=True)
class AdvancementResult:
    outcomes: tuple[UnitOutcome, ...]
    can_continue: bool
    failure: Exception | None = None


@dataclass(frozen=True, slots=True)
class AttemptReport:
    """Correlate work, action, units, and progress across their owners."""

    attempt: int
    action: str
    input_activity: str
    work: Hashable
    requested_dependencies: Mapping[str, object]
    produced_dependencies: Mapping[str, object] | None
    stage: str
    outcomes: tuple[UnitOutcome, ...]
    completed_actions: int
    observations: Mapping[Value, object]
    failure: Exception | None = None


class RunCoordinator:
    """Test-only coordinator; selected operations never receive this object."""

    def __init__(
        self,
        workflow: PreparedWorkflow,
        advance: Callable[[ActionResult], AdvancementResult],
    ) -> None:
        self.workflow = workflow
        self.advance = advance
        self.completed_actions = 0
        self.reports: list[AttemptReport] = []
        self.shutdown_failures: list[Exception] = []
        self._next_attempt = 0
        self._running = False
        self._stopped = False

    def start(self) -> None:
        if self._running or self._stopped:
            raise InvalidExecution("Run cannot start twice")
        started: list[InputActivity] = []
        try:
            for activity in self.workflow.inputs.values():
                started.append(activity)
                activity.provider.start()
        except Exception:
            for activity in reversed(started):
                try:
                    activity.provider.stop()
                except Exception as error:
                    self.shutdown_failures.append(error)
            self._stopped = True
            raise
        self._running = True

    def stop(self) -> None:
        if self._stopped:
            return
        self._running = False
        self._stopped = True
        for activity in reversed(tuple(self.workflow.inputs.values())):
            try:
                activity.provider.stop()
            except Exception as error:
                self.shutdown_failures.append(error)

    def tick(self, coordinates: Mapping[str, int]) -> tuple[AttemptReport, ...]:
        if not self._running:
            raise InvalidExecution("Run is not active")
        try:
            due = tuple(self.workflow.due(coordinates))
            if len({(item.action, item.input.work) for item in due}) != len(due):
                raise InvalidExecution("Due policy selected the same action and work twice")
            if any(item.action not in self.workflow.actions for item in due):
                raise InvalidExecution("Due policy selected an unaccepted action")
        except Exception:
            self.stop()
            raise

        reports: list[AttemptReport] = []
        for item in due:
            activity = self.workflow.feeds[item.action]
            self._next_attempt += 1
            attempt = self._next_attempt
            expected_keys = set(activity.dependency_keys)
            if set(item.input.dependencies) != expected_keys:
                report = self._report(
                    attempt, item, activity, "input_rejected", failure=InvalidExecution("Input dependency filing is incomplete")
                )
            else:
                report = self._run_one(attempt, item, activity)
            reports.append(report)
            if report.stage not in {"completed", "waiting"}:
                self.stop()
                break
            if report.stage == "waiting":
                break
        return tuple(reports)

    def _run_one(self, attempt: int, item: DueWork, activity: InputActivity) -> AttemptReport:
        try:
            delivery = activity.provider.take(item.input.work)
        except Exception as error:
            return self._report(attempt, item, activity, "input_failed", failure=error)
        if delivery is None:
            return self._report(attempt, item, activity, "waiting")
        if (
            delivery.work != item.input.work
            or set(delivery.dependencies) != set(activity.dependency_keys)
            or set(delivery.values) != set(activity.outputs)
        ):
            return self._report(
                attempt,
                item,
                activity,
                "input_rejected",
                produced_dependencies=delivery.dependencies,
                failure=InvalidExecution("Input delivery identity or structure mismatch"),
            )
        try:
            admitted = activity.admit(item.input, delivery)
        except Exception as error:
            return self._report(attempt, item, activity, "input_rejected", produced_dependencies=delivery.dependencies, failure=error)
        if not admitted:
            return self._report(
                attempt,
                item,
                activity,
                "input_rejected",
                produced_dependencies=delivery.dependencies,
                failure=InvalidExecution("Input dependencies are not admissible"),
            )

        try:
            result = self.workflow.actions[item.action].execute(delivery.values)
        except Exception as error:
            return self._report(attempt, item, activity, "operation_failed", produced_dependencies=delivery.dependencies, failure=error)

        expected_units = tuple(dict.fromkeys(unit for unit, _, _ in self.workflow.actions[item.action].optimization_slots))
        try:
            advancement = self.advance(result)
        except Exception as error:
            unknown = tuple(UnitOutcome(unit, "uncertain") for unit in expected_units)
            return self._report(
                attempt,
                item,
                activity,
                "optimization_failed",
                outcomes=unknown,
                observations=result.observations,
                produced_dependencies=delivery.dependencies,
                failure=error,
            )
        actual_units = tuple(outcome.unit for outcome in advancement.outcomes)
        if len(set(actual_units)) != len(actual_units) or set(actual_units) != set(expected_units):
            return self._report(
                attempt,
                item,
                activity,
                "optimization_failed",
                outcomes=tuple(UnitOutcome(unit, "uncertain") for unit in expected_units),
                observations=result.observations,
                produced_dependencies=delivery.dependencies,
                failure=InvalidExecution("Advancement report does not cover the due units"),
            )
        if (
            not advancement.can_continue
            or advancement.failure is not None
            or any(outcome.status not in {"step_returned", "skipped", "contribution_recorded"} for outcome in advancement.outcomes)
        ):
            return self._report(
                attempt,
                item,
                activity,
                "optimization_failed",
                outcomes=advancement.outcomes,
                observations=result.observations,
                produced_dependencies=delivery.dependencies,
                failure=advancement.failure,
            )
        self.completed_actions += 1
        return self._report(
            attempt,
            item,
            activity,
            "completed",
            outcomes=advancement.outcomes,
            observations=result.observations,
            produced_dependencies=delivery.dependencies,
        )

    def _report(
        self,
        attempt: int,
        item: DueWork,
        activity: InputActivity,
        stage: str,
        *,
        outcomes: tuple[UnitOutcome, ...] = (),
        observations: Mapping[Value, object] | None = None,
        produced_dependencies: Mapping[str, object] | None = None,
        failure: Exception | None = None,
    ) -> AttemptReport:
        report = AttemptReport(
            attempt=attempt,
            action=item.action,
            input_activity=activity.name,
            work=item.input.work,
            requested_dependencies=MappingProxyType(dict(item.input.dependencies)),
            produced_dependencies=(MappingProxyType(dict(produced_dependencies)) if produced_dependencies is not None else None),
            stage=stage,
            outcomes=outcomes,
            completed_actions=self.completed_actions,
            observations=MappingProxyType(dict(observations or {})),
            failure=failure,
        )
        self.reports.append(report)
        return report
