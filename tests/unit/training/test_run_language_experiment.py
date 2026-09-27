"""A bounded whole-run experiment, not a replacement for the live Trainer."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass

import pytest
import torch

from library.training.execution import (
    Action,
    ActionResult,
    AdvancementResult,
    DueWork,
    InputActivity,
    InputDelivery,
    InputFeed,
    InputRequest,
    InvalidExecution,
    Operation,
    OptimizationInput,
    RunCoordinator,
    RunDescription,
    UnitOutcome,
    Value,
    prepare_workflow,
)


class _Producer:
    """Selected behavior can publish while no training action is executing."""

    def __init__(self) -> None:
        self.ready: dict[Hashable, InputDelivery] = {}
        self.started = False
        self.stopped = False
        self.taken: list[Hashable] = []

    def start(self) -> None:
        self.started = True

    def publish(self, delivery: InputDelivery) -> None:
        if not self.started or self.stopped:
            raise RuntimeError("producer is not active")
        self.ready[delivery.work] = delivery

    def take(self, work: Hashable) -> InputDelivery | None:
        self.taken.append(work)
        return self.ready.pop(work, None)

    def stop(self) -> None:
        self.stopped = True


class _WrongProducer(_Producer):
    def take(self, work: Hashable) -> InputDelivery | None:
        return InputDelivery("work-2", {Value("batch"): "wrong"}, {"source_revision": 2})


class _FailingStopProducer(_Producer):
    def stop(self) -> None:
        super().stop()
        raise RuntimeError("producer shutdown failed")


class _FailingStartProducer(_Producer):
    def start(self) -> None:
        super().start()
        raise RuntimeError("producer startup failed")


class _ReturningProducer(_Producer):
    def __init__(self, delivery: InputDelivery) -> None:
        super().__init__()
        self.delivery = delivery

    def take(self, work: Hashable) -> InputDelivery | None:
        self.taken.append(work)
        return self.delivery


@dataclass(frozen=True, slots=True)
class _UnitRuntime:
    parameters: tuple[torch.nn.Parameter, ...]
    zero_grad: Callable[[], object]
    step: Callable[[], object]


class _StandardConsumer:
    """A test-side Trainer owner for one accepted shared-gradient policy."""

    def __init__(
        self,
        policy: Mapping[str, tuple[tuple[str, Value], ...]],
        units: Mapping[str, _UnitRuntime],
    ) -> None:
        self.policy = policy
        self.units = units
        self.events: list[str] = []

    def __call__(self, result: ActionResult) -> AdvancementResult:
        selected = self.policy[result.action]
        offered = tuple((item.unit, item.source) for item in result.optimization)
        if offered != selected:
            raise InvalidExecution("optimization inputs differ from accepted policy")
        sources = {id(item.value) for item in result.optimization}
        if len(sources) != 1:
            raise InvalidExecution("this test policy requires one shared gradient source")
        loss = result.optimization[0].value
        if not isinstance(loss, torch.Tensor):
            raise InvalidExecution("test loss is not differentiable")
        parameters = tuple(parameter for unit, _ in selected for parameter in self.units[unit].parameters)
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise InvalidExecution("prepared units overlap")

        for unit, _ in selected:
            self.units[unit].zero_grad()
            self.events.append(f"zero:{unit}")
        loss.backward()
        self.events.append("backward")

        outcomes: list[UnitOutcome] = []
        for position, (unit, _) in enumerate(selected):
            self.events.append(f"attempt:{unit}")
            try:
                self.units[unit].step()
            except Exception as error:
                outcomes.append(UnitOutcome(unit, "uncertain"))
                outcomes.extend(UnitOutcome(later, "not_attempted") for later, _ in selected[position + 1 :])
                return AdvancementResult(tuple(outcomes), can_continue=False, failure=error)
            outcomes.append(UnitOutcome(unit, "step_returned"))
        return AdvancementResult(tuple(outcomes), can_continue=True)


def _strict(request: InputRequest, delivery: InputDelivery) -> bool:
    return dict(delivery.dependencies) == dict(request.dependencies)


def _workflow(
    actions: tuple[Action, ...],
    producer: _Producer,
    batch: Value,
    due: Callable[[Mapping[str, int]], tuple[DueWork, ...]],
):
    return prepare_workflow(
        RunDescription(
            actions=actions,
            inputs=(InputActivity("input", (batch,), ("source_revision",), producer, _strict),),
            feeds=tuple(InputFeed(action.name, "input") for action in actions),
            due=due,
        )
    )


@pytest.mark.training
@pytest.mark.unit
def test_one_coordinator_runs_ordinary_and_alternating_arrangements():
    batch = Value("batch")
    ordinary_loss = Value("ordinary_loss")
    denoiser = torch.nn.Parameter(torch.tensor(2.0))
    ordinary_optimizer = torch.optim.SGD([denoiser], lr=0.1)
    ordinary = Action(
        "ordinary",
        (batch,),
        (Operation("objective", lambda data: (denoiser * data - 1).square(), (batch,), (ordinary_loss,)),),
        optimization=(OptimizationInput("denoiser", ordinary_loss),),
    )
    ordinary_producer = _Producer()
    ordinary_workflow = _workflow(
        (ordinary,),
        ordinary_producer,
        batch,
        lambda coordinates: (DueWork("ordinary", InputRequest(f"ordinary-{coordinates['turn']}", {"source_revision": 1})),),
    )
    ordinary_consumer = _StandardConsumer(
        {"ordinary": (("denoiser", ordinary_loss),)},
        {"denoiser": _UnitRuntime((denoiser,), ordinary_optimizer.zero_grad, ordinary_optimizer.step)},
    )
    ordinary_run = RunCoordinator(ordinary_workflow, ordinary_consumer)
    ordinary_run.start()
    ordinary_producer.publish(InputDelivery("ordinary-0", {batch: torch.tensor(1.0)}, {"source_revision": 1}))
    ordinary_report = ordinary_run.tick({"turn": 0})[0]
    assert ordinary_report.stage == "completed"
    assert ordinary_report.action == "ordinary"
    assert ordinary_report.completed_actions == 1
    assert dict(ordinary_report.produced_dependencies or {}) == {"source_revision": 1}
    assert ordinary_consumer.events == ["zero:denoiser", "backward", "attempt:denoiser"]
    ordinary_run.stop()

    generator = torch.nn.Parameter(torch.tensor(1.0))
    discriminator = torch.nn.Parameter(torch.tensor(2.0))
    g_optimizer = torch.optim.SGD([generator], lr=0.1)
    d_optimizer = torch.optim.SGD([discriminator], lr=0.1)
    g_loss, d_loss = Value("g_loss"), Value("d_loss")
    adversarial_actions = (
        Action(
            "generator",
            (batch,),
            (Operation("g_objective", lambda data: (generator * discriminator.detach() * data - 3).square(), (batch,), (g_loss,)),),
            optimization=(OptimizationInput("generator", g_loss),),
        ),
        Action(
            "discriminator",
            (batch,),
            (Operation("d_objective", lambda data: (discriminator * generator.detach() * data - 3).square(), (batch,), (d_loss,)),),
            optimization=(OptimizationInput("discriminator", d_loss),),
        ),
    )
    adversarial_producer = _Producer()
    adversarial_workflow = _workflow(
        adversarial_actions,
        adversarial_producer,
        batch,
        lambda coordinates: (
            DueWork(
                "generator" if coordinates["turn"] % 2 == 0 else "discriminator",
                InputRequest(f"adversarial-{coordinates['turn']}", {"source_revision": 1}),
            ),
        ),
    )
    adversarial_consumer = _StandardConsumer(
        {"generator": (("generator", g_loss),), "discriminator": (("discriminator", d_loss),)},
        {
            "generator": _UnitRuntime((generator,), g_optimizer.zero_grad, g_optimizer.step),
            "discriminator": _UnitRuntime((discriminator,), d_optimizer.zero_grad, d_optimizer.step),
        },
    )
    adversarial_run = RunCoordinator(adversarial_workflow, adversarial_consumer)
    adversarial_run.start()
    adversarial_producer.publish(InputDelivery("adversarial-1", {batch: torch.tensor(1.0)}, {"source_revision": 1}))
    adversarial_producer.publish(InputDelivery("adversarial-0", {batch: torch.tensor(1.0)}, {"source_revision": 1}))
    first, second = adversarial_run.tick({"turn": 0})[0], adversarial_run.tick({"turn": 1})[0]
    assert (first.action, second.action) == ("generator", "discriminator")
    assert (first.work, second.work) == ("adversarial-0", "adversarial-1")
    assert (first.completed_actions, second.completed_actions) == (1, 2)
    assert adversarial_producer.taken == ["adversarial-0", "adversarial-1"]
    adversarial_run.stop()


@pytest.mark.training
@pytest.mark.unit
def test_independent_producer_waits_then_rejects_stale_or_misidentified_work():
    batch, loss = Value("batch"), Value("loss")
    calls: list[str] = []
    action = Action("train", (batch,), (Operation("compute", lambda value: calls.append(value) or torch.tensor(1.0), (batch,), (loss,)),))
    producer = _Producer()
    workflow = _workflow(
        (action,),
        producer,
        batch,
        lambda _: (DueWork("train", InputRequest("work-1", {"source_revision": 2})),),
    )
    run = RunCoordinator(workflow, lambda _: AdvancementResult((), can_continue=True))
    run.start()
    waiting = run.tick({})[0]
    assert waiting.stage == "waiting"
    assert run.completed_actions == 0
    producer.publish(InputDelivery("work-1", {batch: "stale"}, {"source_revision": 1}))
    stale = run.tick({})[0]
    assert stale.stage == "input_rejected"
    assert stale.work == "work-1" and stale.attempt != waiting.attempt
    assert dict(stale.requested_dependencies) == {"source_revision": 2}
    assert dict(stale.produced_dependencies or {}) == {"source_revision": 1}
    assert calls == [] and producer.stopped
    with pytest.raises(InvalidExecution, match="not active"):
        run.tick({})

    wrong_producer = _WrongProducer()
    wrong_workflow = _workflow(
        (action,), wrong_producer, batch, lambda _: (DueWork("train", InputRequest("work-1", {"source_revision": 2})),)
    )
    wrong_run = RunCoordinator(wrong_workflow, lambda _: AdvancementResult((), can_continue=True))
    wrong_run.start()

    wrong = wrong_run.tick({})[0]
    assert wrong.stage == "input_rejected"
    assert calls == [] and wrong_producer.stopped


@pytest.mark.training
@pytest.mark.unit
def test_partial_joint_advancement_stops_with_one_correlated_attempt():
    batch, loss = Value("batch"), Value("loss")
    first_weight = torch.nn.Parameter(torch.tensor(1.0))
    second_weight = torch.nn.Parameter(torch.tensor(1.0))
    first_optimizer = torch.optim.SGD([first_weight], lr=0.1)
    second_optimizer = torch.optim.SGD([second_weight], lr=0.1)
    action = Action(
        "joint",
        (batch,),
        (Operation("objective", lambda data: (first_weight * data + second_weight).square(), (batch,), (loss,)),),
        optimization=(OptimizationInput("first", loss), OptimizationInput("second", loss)),
    )
    producer = _Producer()
    workflow = _workflow((action,), producer, batch, lambda _: (DueWork("joint", InputRequest("item-7", {"source_revision": 3})),))

    def fail_after_second_step() -> None:
        second_optimizer.step()
        raise RuntimeError("backend lost confirmation")

    consumer = _StandardConsumer(
        {"joint": (("first", loss), ("second", loss))},
        {
            "first": _UnitRuntime((first_weight,), first_optimizer.zero_grad, first_optimizer.step),
            "second": _UnitRuntime((second_weight,), second_optimizer.zero_grad, fail_after_second_step),
        },
    )
    run = RunCoordinator(workflow, consumer)
    run.start()
    producer.publish(InputDelivery("item-7", {batch: torch.tensor(1.0)}, {"source_revision": 3}))
    report = run.tick({})[0]

    assert report.stage == "optimization_failed"
    assert report.work == "item-7" and report.action == "joint" and report.attempt == 1
    assert dict(report.produced_dependencies or {}) == {"source_revision": 3}
    assert report.outcomes == (UnitOutcome("first", "step_returned"), UnitOutcome("second", "uncertain"))
    assert report.completed_actions == 0
    assert producer.stopped
    assert first_weight.item() != 1.0 and second_weight.item() != 1.0
    with pytest.raises(InvalidExecution, match="not active"):
        run.tick({})


@pytest.mark.training
@pytest.mark.unit
def test_static_links_and_unaccepted_due_work_fail_before_input_is_taken():
    batch = Value("batch")
    action = Action("train", (batch,), ())
    producer = _Producer()
    with pytest.raises(InvalidExecution, match="does not supply"):
        prepare_workflow(
            RunDescription(
                actions=(action,),
                inputs=(InputActivity("input", (Value("other"),), (), producer, _strict),),
                feeds=(InputFeed("train", "input"),),
                due=lambda _: (),
            )
        )

    workflow = _workflow((action,), producer, batch, lambda _: (DueWork("unknown", InputRequest("item", {"source_revision": 1})),))
    run = RunCoordinator(workflow, lambda _: AdvancementResult((), can_continue=True))
    run.start()
    with pytest.raises(InvalidExecution, match="unaccepted action"):
        run.tick({})
    assert producer.taken == [] and producer.stopped


@pytest.mark.training
@pytest.mark.unit
def test_bad_advancement_report_stops_without_trusting_partial_outcomes():
    batch, loss = Value("batch"), Value("loss")
    action = Action(
        "joint",
        (batch,),
        (Operation("objective", lambda _: torch.tensor(1.0), (batch,), (loss,)),),
        optimization=(OptimizationInput("first", loss), OptimizationInput("second", loss)),
    )
    producer = _Producer()
    workflow = _workflow((action,), producer, batch, lambda _: (DueWork("joint", InputRequest("item", {"source_revision": 1})),))
    run = RunCoordinator(workflow, lambda _: AdvancementResult((UnitOutcome("first", "step_returned"),), can_continue=True))
    run.start()
    producer.publish(InputDelivery("item", {batch: None}, {"source_revision": 1}))
    report = run.tick({})[0]
    assert report.stage == "optimization_failed"
    assert report.outcomes == (UnitOutcome("first", "uncertain"), UnitOutcome("second", "uncertain"))
    assert report.completed_actions == 0 and producer.stopped


@pytest.mark.training
@pytest.mark.unit
def test_shutdown_failure_does_not_hide_operation_failure():
    batch = Value("batch")

    def fail(_value: object) -> None:
        raise RuntimeError("selected operation failed")

    action = Action("train", (batch,), (Operation("failing", fail, (batch,), ()),))
    producer = _FailingStopProducer()
    workflow = _workflow((action,), producer, batch, lambda _: (DueWork("train", InputRequest("item", {"source_revision": 1})),))
    run = RunCoordinator(workflow, lambda _: AdvancementResult((), can_continue=True))
    run.start()
    producer.publish(InputDelivery("item", {batch: None}, {"source_revision": 1}))
    report = run.tick({})[0]
    assert report.stage == "operation_failed"
    assert report.failure is not None and "failing" in str(report.failure)
    assert producer.stopped
    assert len(run.shutdown_failures) == 1


@pytest.mark.training
@pytest.mark.unit
@pytest.mark.parametrize(
    "delivery",
    [
        InputDelivery("item", {Value("batch"): "value"}, {"wrong_revision": 1}),
        InputDelivery("item", {Value("other"): "value"}, {"source_revision": 1}),
    ],
)
def test_invalid_input_structure_stops_before_selected_computation(delivery: InputDelivery):
    batch = Value("batch")
    calls: list[object] = []
    action = Action("train", (batch,), (Operation("compute", lambda value: calls.append(value), (batch,), ()),))
    producer = _ReturningProducer(delivery)
    workflow = _workflow((action,), producer, batch, lambda _: (DueWork("train", InputRequest("item", {"source_revision": 1})),))
    run = RunCoordinator(workflow, lambda _: AdvancementResult((), can_continue=True))
    run.start()
    report = run.tick({})[0]
    assert report.stage == "input_rejected"
    assert report.produced_dependencies is not None
    assert calls == [] and producer.stopped


@pytest.mark.training
@pytest.mark.unit
def test_partial_start_failure_stops_all_started_input_activities():
    first_value, second_value = Value("first"), Value("second")
    first, second = _Producer(), _FailingStartProducer()
    workflow = prepare_workflow(
        RunDescription(
            actions=(Action("first_action", (first_value,), ()), Action("second_action", (second_value,), ())),
            inputs=(
                InputActivity("first_input", (first_value,), (), first, _strict),
                InputActivity("second_input", (second_value,), (), second, _strict),
            ),
            feeds=(InputFeed("first_action", "first_input"), InputFeed("second_action", "second_input")),
            due=lambda _: (),
        )
    )
    run = RunCoordinator(workflow, lambda _: AdvancementResult((), can_continue=True))
    with pytest.raises(RuntimeError, match="startup failed"):
        run.start()
    assert first.started and first.stopped
    assert second.started and second.stopped
    with pytest.raises(InvalidExecution, match="cannot start twice"):
        run.start()


@pytest.mark.training
@pytest.mark.unit
def test_bad_due_policy_stops_before_input_handoff():
    batch = Value("batch")
    action = Action("train", (batch,), ())
    duplicate = DueWork("train", InputRequest("item", {"source_revision": 1}))
    producer = _Producer()
    run = RunCoordinator(_workflow((action,), producer, batch, lambda _: (duplicate, duplicate)), lambda _: AdvancementResult((), True))
    run.start()
    with pytest.raises(InvalidExecution, match="same action and work twice"):
        run.tick({})
    assert producer.taken == [] and producer.stopped

    def broken_due(_coordinates: Mapping[str, int]) -> tuple[DueWork, ...]:
        raise RuntimeError("due policy failed")

    another_producer = _Producer()
    another_run = RunCoordinator(
        _workflow((action,), another_producer, batch, broken_due),
        lambda _: AdvancementResult((), True),
    )
    another_run.start()
    with pytest.raises(RuntimeError, match="due policy failed"):
        another_run.tick({})
    assert another_producer.taken == [] and another_producer.stopped


@pytest.mark.training
@pytest.mark.unit
def test_unknown_optimizer_status_cannot_claim_safe_continuation():
    batch, loss = Value("batch"), Value("loss")
    action = Action(
        "train",
        (batch,),
        (Operation("objective", lambda _: torch.tensor(1.0), (batch,), (loss,)),),
        optimization=(OptimizationInput("unit", loss),),
    )
    producer = _Producer()
    workflow = _workflow((action,), producer, batch, lambda _: (DueWork("train", InputRequest("item", {"source_revision": 1})),))
    run = RunCoordinator(workflow, lambda _: AdvancementResult((UnitOutcome("unit", "aborted"),), can_continue=True))
    run.start()
    producer.publish(InputDelivery("item", {batch: None}, {"source_revision": 1}))
    report = run.tick({})[0]
    assert report.stage == "optimization_failed"
    assert report.outcomes == (UnitOutcome("unit", "aborted"),)
    assert report.completed_actions == 0 and producer.stopped
