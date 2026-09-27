"""Deterministic whole-run input handoff experiment, not a production provider API."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from library.training.execution import Action, ActionResult, Operation, OptimizationInput, PreparedAction, Value, compile_run


@dataclass(frozen=True, slots=True)
class _Work:
    work_id: str
    sample_id: str
    caption: str
    encoder_revision: int


@dataclass(frozen=True, slots=True)
class _Encoded:
    work: _Work
    actual_encoder_revision: int
    payload: str


class _Producer:
    """Test-only provider owns its bounded work and result stages."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.started = False
        self.quiesced = False
        self.seen: set[str] = set()
        self.pending: dict[str, _Work] = {}
        self.produced: dict[str, _Encoded] = {}
        self.ready: dict[str, _Encoded] = {}
        self.failed: dict[str, Exception] = {}

    def start(self) -> None:
        if self.started:
            raise RuntimeError("producer already started")
        self.started = True

    def submit(self, work: _Work) -> None:
        if not self.started or self.quiesced:
            raise RuntimeError("producer is not accepting work")
        if work.work_id in self.seen:
            raise ValueError("work identity was already submitted")
        if len(self.pending) + len(self.produced) + len(self.ready) >= self.capacity:
            raise RuntimeError("producer backpressure")
        self.seen.add(work.work_id)
        self.pending[work.work_id] = work

    def complete(self, work_id: str, *, actual_revision: int, result_work: _Work | None = None) -> None:
        if not self.started or self.quiesced:
            raise RuntimeError("producer cannot complete work")
        requested = self.pending.pop(work_id)
        claimed = result_work if result_work is not None else requested
        self.produced[work_id] = _Encoded(claimed, actual_revision, f"encoded:{claimed.caption}@{actual_revision}")

    def publish(self, work_id: str) -> None:
        if not self.started or self.quiesced:
            raise RuntimeError("producer cannot publish work")
        self.ready[work_id] = self.produced.pop(work_id)

    def fail(self, work_id: str, error: Exception) -> None:
        if not self.started or self.quiesced:
            raise RuntimeError("producer cannot fail work")
        self.pending.pop(work_id)
        self.failed[work_id] = error

    def peek(self, work_id: str) -> _Encoded | None:
        if work_id in self.failed:
            raise RuntimeError(f"producer failed work {work_id}") from self.failed[work_id]
        return self.ready.get(work_id)

    def take(self, work_id: str) -> _Encoded:
        return self.ready.pop(work_id)

    def quiesce(self) -> None:
        self.quiesced = True

    def stop(self) -> None:
        self.quiesced = True
        self.started = False


class _RunInputCoordinator:
    """Test-only run owner checks the provider result before action delivery."""

    def __init__(self, producer: _Producer, action: PreparedAction, input_value: Value) -> None:
        self.producer = producer
        self.action = action
        self.input_value = input_value
        self.handed_off: list[str] = []
        self.consumed: list[str] = []
        self.advancement: dict[str, str] = {}

    def start(self) -> None:
        self.producer.start()

    def stop(self) -> None:
        self.producer.stop()

    def handoff(self, work: _Work, *, current_revision: int, allowed_lag: int = 0) -> _Encoded:
        if not self.producer.started or self.producer.quiesced:
            raise RuntimeError("producer is not available for handoff")
        result = self.producer.peek(work.work_id)
        if result is None:
            raise RuntimeError("requested work is not ready")
        if result.work != work:
            raise RuntimeError("ready result belongs to different work, sample, or caption")
        if result.actual_encoder_revision != work.encoder_revision:
            raise RuntimeError("result used a different encoder revision than requested")
        lag = current_revision - result.actual_encoder_revision
        if lag < 0 or lag > allowed_lag:
            raise RuntimeError("ready result is not admissible under encoder freshness policy")
        self.handed_off.append(work.work_id)
        return self.producer.take(work.work_id)

    def run(self, work: _Work, *, current_revision: int, allowed_lag: int = 0) -> ActionResult:
        result = self.handoff(work, current_revision=current_revision, allowed_lag=allowed_lag)
        action_result = self.action.execute({self.input_value: result})
        self.consumed.append(work.work_id)
        return action_result

    def report_advancement(self, work_id: str, outcome: str) -> None:
        if work_id not in self.consumed:
            raise ValueError("work has not been consumed by an action")
        if outcome not in {"step_returned", "skipped", "uncertain"}:
            raise ValueError("unrecognized advancement outcome")
        if work_id in self.advancement:
            raise ValueError("advancement was already reported for this work")
        self.advancement[work_id] = outcome

    def snapshot_projection(self) -> dict[str, tuple[str, ...] | dict[str, str]]:
        """Only a projection at a paused boundary, not an exact-run snapshot."""
        if not self.producer.quiesced:
            raise RuntimeError("producer must be quiesced for a coherent projection")
        return {
            "pending": tuple(self.producer.pending),
            "produced_unpublished": tuple(self.producer.produced),
            "ready": tuple(self.producer.ready),
            "handed_off": tuple(self.handed_off),
            "consumed": tuple(self.consumed),
            "advancement": dict(self.advancement),
        }


def _coordinator(capacity: int = 2) -> tuple[_RunInputCoordinator, Value, Value]:
    input_value = Value("training_input")
    observed = Value("used_conditioning")
    action = Action(
        "train",
        inputs=(input_value,),
        operations=(Operation("use_conditioning", lambda result: result.payload, (input_value,), (observed,)),),
        observations=(observed,),
    )
    prepared = compile_run((action,), due=lambda _: ("train",))
    return _RunInputCoordinator(_Producer(capacity), prepared.actions["train"], input_value), input_value, observed


@pytest.mark.training
@pytest.mark.unit
def test_producer_progresses_ahead_and_out_of_order_without_positional_pairing():
    coordinator, _, observed = _coordinator()
    coordinator.start()
    first = _Work("epoch-0/sample-4", "sample-4", "red bird", 3)
    next_epoch = _Work("epoch-1/sample-4", "sample-4", "blue bird", 3)
    coordinator.producer.submit(first)
    coordinator.producer.submit(next_epoch)

    # The producer makes progress before any action runs, and finishes later work first.
    coordinator.producer.complete(next_epoch.work_id, actual_revision=3)
    with pytest.raises(RuntimeError, match="not ready"):
        coordinator.run(first, current_revision=3)
    coordinator.producer.publish(next_epoch.work_id)
    assert next_epoch.work_id in coordinator.producer.ready
    with pytest.raises(RuntimeError, match="not ready"):
        coordinator.run(first, current_revision=3)

    coordinator.producer.complete(first.work_id, actual_revision=3)
    coordinator.producer.publish(first.work_id)
    first_result = coordinator.run(first, current_revision=3)
    second_result = coordinator.run(next_epoch, current_revision=3)
    assert first_result.observations[observed] == "encoded:red bird@3"
    assert second_result.observations[observed] == "encoded:blue bird@3"
    assert coordinator.consumed == [first.work_id, next_epoch.work_id]
    coordinator.stop()


@pytest.mark.training
@pytest.mark.unit
def test_ready_results_are_checked_against_work_and_encoder_state_before_handoff():
    coordinator, _, _ = _coordinator(capacity=3)
    coordinator.start()
    wrong = _Work("work-wrong", "sample-1", "original caption", 7)
    changed = _Work("work-changed", "sample-2", "caption", 7)
    pinned = _Work("work-pinned", "sample-3", "caption", 7)
    for work in (wrong, changed, pinned):
        coordinator.producer.submit(work)

    coordinator.producer.complete(
        wrong.work_id,
        actual_revision=7,
        result_work=_Work(wrong.work_id, wrong.sample_id, "other caption", 7),
    )
    coordinator.producer.publish(wrong.work_id)
    with pytest.raises(RuntimeError, match="different work, sample, or caption"):
        coordinator.run(wrong, current_revision=7)

    # A mutable encoder changed while work was pending. Claiming the request's
    # old revision would be false if the worker actually used the new revision.
    coordinator.producer.complete(changed.work_id, actual_revision=8)
    coordinator.producer.publish(changed.work_id)
    with pytest.raises(RuntimeError, match="different encoder revision than requested"):
        coordinator.run(changed, current_revision=8, allowed_lag=1)

    # A genuinely pinned old result is ready, but admission is policy-dependent.
    coordinator.producer.complete(pinned.work_id, actual_revision=7)
    coordinator.producer.publish(pinned.work_id)
    with pytest.raises(RuntimeError, match="not admissible"):
        coordinator.run(pinned, current_revision=8)
    coordinator.run(pinned, current_revision=8, allowed_lag=1)
    assert coordinator.handed_off == [pinned.work_id]
    assert wrong.work_id in coordinator.producer.ready
    assert changed.work_id in coordinator.producer.ready
    coordinator.stop()


@pytest.mark.training
@pytest.mark.unit
def test_bounded_backpressure_producer_failure_and_shutdown_are_run_visible():
    coordinator, _, _ = _coordinator()
    coordinator.start()
    first = _Work("a", "sample-a", "a", 1)
    second = _Work("b", "sample-b", "b", 1)
    third = _Work("c", "sample-c", "c", 1)
    coordinator.producer.submit(first)
    coordinator.producer.submit(second)
    with pytest.raises(RuntimeError, match="backpressure"):
        coordinator.producer.submit(third)

    coordinator.producer.complete(second.work_id, actual_revision=1)
    with pytest.raises(RuntimeError, match="backpressure"):
        coordinator.producer.submit(third)
    coordinator.producer.fail(first.work_id, RuntimeError("encoder worker died"))
    coordinator.producer.submit(third)
    with pytest.raises(RuntimeError, match="producer failed work") as failure:
        coordinator.run(first, current_revision=1)
    assert failure.value.__cause__ is coordinator.producer.failed[first.work_id]
    assert coordinator.consumed == []

    coordinator.stop()
    with pytest.raises(RuntimeError, match="not accepting work"):
        coordinator.producer.submit(_Work("d", "sample-d", "d", 1))
    with pytest.raises(RuntimeError, match="cannot complete work"):
        coordinator.producer.complete(third.work_id, actual_revision=1)


@pytest.mark.training
@pytest.mark.unit
def test_quiesced_projection_keeps_input_stages_distinct_from_advancement():
    coordinator, _, _ = _coordinator(capacity=5)
    coordinator.start()
    works = [_Work(name, f"sample-{name}", name, 1) for name in "abcde"]
    for work in works:
        coordinator.producer.submit(work)
    pending, consumed, handed, produced, ready = works
    coordinator.producer.complete(consumed.work_id, actual_revision=1)
    coordinator.producer.publish(consumed.work_id)
    coordinator.run(consumed, current_revision=1)
    coordinator.report_advancement(consumed.work_id, "step_returned")
    coordinator.producer.complete(handed.work_id, actual_revision=1)
    coordinator.producer.publish(handed.work_id)
    coordinator.handoff(handed, current_revision=1)
    coordinator.producer.complete(produced.work_id, actual_revision=1)
    coordinator.producer.complete(ready.work_id, actual_revision=1)
    coordinator.producer.publish(ready.work_id)

    with pytest.raises(RuntimeError, match="quiesced"):
        coordinator.snapshot_projection()
    coordinator.producer.quiesce()
    assert coordinator.snapshot_projection() == {
        "pending": (pending.work_id,),
        "produced_unpublished": (produced.work_id,),
        "ready": (ready.work_id,),
        "handed_off": (consumed.work_id, handed.work_id),
        "consumed": (consumed.work_id,),
        "advancement": {consumed.work_id: "step_returned"},
    }
    with pytest.raises(RuntimeError, match="cannot complete work"):
        coordinator.producer.complete(pending.work_id, actual_revision=1)
    coordinator.stop()


@pytest.mark.training
@pytest.mark.unit
def test_composed_input_action_and_two_unit_failure_do_not_claim_one_atomic_result():
    """One toy run joins the earlier provider, action, and optimization checks."""

    matrix = torch.nn.Parameter(torch.tensor(1.0))
    other = torch.nn.Parameter(torch.tensor(0.5))
    input_value = Value("training_input")
    loss = Value("loss")
    action = Action(
        "train",
        inputs=(input_value,),
        operations=(
            Operation(
                "objective",
                lambda encoded: (matrix * len(encoded.work.caption) + other).square(),
                (input_value,),
                (loss,),
            ),
        ),
        optimization=(OptimizationInput("matrix_unit", loss), OptimizationInput("other_unit", loss)),
    )
    prepared = compile_run((action,), due=lambda _: ("train",))
    coordinator = _RunInputCoordinator(_Producer(capacity=2), prepared.due_actions({"turn": 0})[0], input_value)
    coordinator.start()
    first = _Work("epoch-0/sample-a", "sample-a", "red bird", 3)
    waiting = _Work("epoch-0/sample-b", "sample-b", "blue bird", 3)
    coordinator.producer.submit(first)
    coordinator.producer.submit(waiting)
    coordinator.producer.complete(waiting.work_id, actual_revision=3)
    coordinator.producer.publish(waiting.work_id)
    coordinator.producer.complete(first.work_id, actual_revision=3)
    coordinator.producer.publish(first.work_id)

    result = coordinator.run(first, current_revision=3)
    assert coordinator.consumed == [first.work_id]
    assert waiting.work_id in coordinator.producer.ready
    assert matrix.grad is None and other.grad is None
    assert matrix.item() == 1.0 and other.item() == 0.5
    assert [(item.unit, item.source) for item in result.optimization] == [
        ("matrix_unit", loss),
        ("other_unit", loss),
    ]
    offered = result.optimization[0].value
    assert isinstance(offered, torch.Tensor)
    assert result.optimization[1].value is offered

    # This is Trainer-side test code, not a second hidden training loop inside
    # the operation or provider. No transactional update is promised.
    matrix_optimizer = torch.optim.SGD([matrix], lr=0.1)
    other_optimizer = torch.optim.AdamW([other], lr=0.1)
    matrix_optimizer.zero_grad()
    other_optimizer.zero_grad()
    offered.backward()
    outcomes: list[tuple[str, str]] = []
    matrix_optimizer.step()
    outcomes.append(("matrix_unit", "step_returned"))

    def step_without_confirmation() -> None:
        other_optimizer.step()
        raise RuntimeError("backend lost confirmation after stepping")

    try:
        step_without_confirmation()
    except RuntimeError as error:
        assert str(error) == "backend lost confirmation after stepping"
        outcomes.append(("other_unit", "uncertain"))

    # This test explicitly stops further handoff. The input provider cannot
    # decide optimizer recovery or represent both unit outcomes itself.
    coordinator.producer.quiesce()
    input_projection = coordinator.snapshot_projection()
    assert input_projection["ready"] == (waiting.work_id,)
    assert input_projection["consumed"] == (first.work_id,)
    assert input_projection["advancement"] == {}
    assert result.action == "train"
    assert tuple(outcomes) == (("matrix_unit", "step_returned"), ("other_unit", "uncertain"))
    with pytest.raises(RuntimeError, match="not available for handoff"):
        coordinator.run(waiting, current_revision=3)
    assert matrix.item() != 1.0
    assert other.item() != 0.5  # Only the test knows what the uncertain call changed.
    coordinator.stop()
