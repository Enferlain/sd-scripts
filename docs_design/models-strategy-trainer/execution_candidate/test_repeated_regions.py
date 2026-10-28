"""Repeated multi-block regions, explicit accumulation and owner boundaries."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest
import torch

from . import graph, prepare as preparation
from .examples import AccumulationState, Scale, accumulation, scale
from .graph import (
    Advance,
    Block,
    Call,
    Channel,
    Choice,
    Contract,
    Differentiate,
    Join,
    NotReady,
    Output,
    OwnedState,
    Producer,
    Rejected,
    Release,
    Repeat,
    Requested,
    Retain,
    Run,
    Sequence,
    Together,
    Unit,
    Use,
)
from .prepare import Engine, WorkFailed, prepare
from .state import Product, Replacement
from .test_candidate import leaves
from .test_access import pair
from .test_specialized_lowering import observable, run


def edit(filing, name, transform):
    def visit(scope):
        if isinstance(scope, Together):
            return replace(scope, children=tuple(visit(child) for child in scope.children))
        if isinstance(scope, (Repeat, Sequence)):
            return replace(scope, blocks=tuple(visit(child) for child in scope.blocks))
        if isinstance(scope, Block):
            return replace(scope, work=tuple(transform(work) if work.name == name else work for work in scope.work))
        return scope

    return replace(filing, root=visit(filing.root))


def oracle():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    gradients = []
    for values in ((1.0, 2.0), (4.0, 5.0, 6.0)):
        for value in values:
            (parameter.mul(value).square() / len(values)).backward()
        assert parameter.grad is not None
        gradients.append(parameter.grad.detach().clone())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return parameter.detach(), gradients


async def until_reached(task, event):
    """Surface a premature engine failure, and always join the observer."""
    observer = asyncio.create_task(event.wait())
    try:
        await asyncio.wait((task, observer), return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            await task
        assert event.is_set(), "engine finished before the expected boundary"
    finally:
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("worker", (False, True))
def test_live_policy_repeats_multi_block_windows_with_direct_numerical_correspondence(lowering, worker, monkeypatch):
    filing = accumulation()
    accepted = Contract().accept(filing)
    calls = tuple(f"objective-w{count}-{slot}" for count in (2, 3) for slot in range(count)) if worker else ()
    ready = prepare(accepted, lowering=lowering, worker_calls=calls, worker_limit=2).publish()
    state = ready.state
    assert state.image is not None
    runtime = state.image.units["weights"]
    actual_gradients, windows = [], []
    original = state.advance

    def advance(contribution, names, invocation, **kwargs):
        assert runtime.parameters[0].grad is not None
        actual_gradients.append(runtime.parameters[0].grad.detach().clone())
        windows.append(contribution)
        return original(contribution, names, invocation, **kwargs)

    monkeypatch.setattr(state, "advance", advance)

    # Both alternatives are already lowered; runtime choices cannot rebuild them.
    def static_discovery(*args, **kwargs):
        raise AssertionError("static graph discovery on the hot path")

    monkeypatch.setattr(graph, "ordered", static_discovery)
    monkeypatch.setattr(graph, "sequence_order", static_discovery)
    monkeypatch.setattr(preparation, "ordered", static_discovery)
    monkeypatch.setattr(preparation, "sequence_order", static_discovery)
    monkeypatch.setattr(preparation.Lowering, "scope", static_discovery)
    run(Engine(ready).run())
    expected, gradients = oracle()
    torch.testing.assert_close(runtime.parameters[0].detach(), expected)
    for actual, reference in zip(actual_gradients, gradients, strict=True):
        torch.testing.assert_close(actual, reference)
    policy = state.owners["policy"]
    assert isinstance(policy, AccumulationState) and policy.decisions == [2, 3]
    assert state.coordinates["learning"] == {"window-2": 1, "window-3": 1}
    assert state.coordinates["window-2"] == {"w2-0": 1, "w2-1": 1, "finish-2": 1}
    assert state.coordinates["window-3"] == {"w3-0": 1, "w3-1": 1, "w3-2": 1, "finish-3": 1}
    assert [window.origins for window in windows] == [(1, 2), (4, 5, 6)]
    assert windows[0].region is not windows[1].region
    assert runtime.steps == 2 and runtime.pending is None and runtime.window is None
    assert [(record.invocation, record.advancement_invocation) for record in runtime.outcomes] == [(1, 3), (2, 3), (4, 7), (5, 7), (6, 7)]
    assert all(record.optimizer == record.reset == "returned" for record in runtime.outcomes)
    assert [(key, invocation) for key, invocation, _ in state.feeds["micro-0"].handed_off] == [(0, 1), (1, 4)]
    assert [(key, invocation) for key, invocation, _ in state.feeds["micro-2"].handed_off] == [(1, 6)]
    assert state.completed and not state.active_retentions and not state.readers and not state.writers
    state.capture()


def test_same_accepted_description_has_equivalent_cpu_emissions():
    accepted = Contract().accept(accumulation())
    results = []
    for lowering in ("calls", "inline"):
        ready = prepare(accepted, lowering=lowering).publish()
        run(Engine(ready).run())
        results.append(observable(ready))
    assert results[0] == results[1]


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_producer_progresses_between_contributions_while_no_update_has_happened(lowering):
    async def scenario():
        pending, ahead = asyncio.Event(), asyncio.Event()
        observed = []
        filing = accumulation()
        assert isinstance(filing.root, Together)
        source = filing.root.children[0]
        assert isinstance(source, Producer)

        # Publish first input independently; remaining production waits for the
        # first contribution, then runs while the consumer awaits readiness.
        async def first_then_producer(context):
            await context.publish("micro-0", Product(0, torch.tensor(1.0)))
            await pending.wait()
            observed.append("producer-ahead")
            ahead.set()
            for key in (0, 1):
                for slot in range(3):
                    if key == slot == 0:
                        continue
                    await context.publish(f"micro-{slot}", Product(key, torch.tensor(float(1 + 3 * key + slot))))

        async def pause(context):
            assert ready.state.image is not None
            runtime = ready.state.image.units["weights"]
            assert runtime.pending is not None and runtime.steps == 0
            assert not ready.state.active_retentions
            pending.set()
            await ahead.wait()

        filing = replace(
            filing, root=replace(filing.root, children=(replace(source, implementation=first_then_producer), *filing.root.children[1:]))
        )

        def insert(scope):
            if isinstance(scope, Together):
                return replace(scope, children=tuple(insert(child) for child in scope.children))
            if isinstance(scope, (Repeat, Sequence)):
                return replace(scope, blocks=tuple(insert(child) for child in scope.blocks))
            if isinstance(scope, Block) and scope.name == "w2-0":
                return replace(scope, work=(*scope.work, Call("wait-for-ahead", pause, after=("release-w2-0",))))
            return scope

        filing = replace(filing, root=insert(filing.root))
        ready = prepare(Contract().accept(filing), lowering=lowering).publish()
        await Engine(ready).run()
        assert observed == ["producer-ahead"]
        assert ready.state.image is not None
        assert ready.state.image.units["weights"].steps == 2

    run(scenario())


@pytest.mark.parametrize(
    "name,transform,diagnostic",
    (
        ("backward-w2-1", lambda work: replace(work, accumulate=None), "competing backward ownership"),
        ("backward-w2-1", lambda work: replace(work, accumulate="x-w2-0"), "current cross-block contribution"),
        ("advance-2", lambda work: replace(work, contribution="gradient-w2-0"), "complete contribution chain"),
    ),
)
def test_incomplete_or_misaddressed_authored_windows_reject_before_loading(name, transform, diagnostic):
    filing = edit(accumulation(), name, transform)
    reached = []
    filing = replace(filing, participants=(replace(filing.participants[0], load=lambda: reached.append("load")),))
    with pytest.raises(Rejected, match=diagnostic):
        Contract().accept(filing)
    assert not reached


def test_a_later_input_wait_is_still_rejected_while_old_state_is_retained():
    filing = edit(accumulation(), "release-w2-0", lambda work: Call(work.name, lambda _: None, after=work.after))
    with pytest.raises(Rejected, match="retained sequence"):
        Contract().accept(filing)


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("cancel", (False, True))
def test_interrupted_window_releases_old_state_but_never_advances_or_reports_region_completion(lowering, cancel):
    async def scenario():
        reached = asyncio.Event()

        async def interrupt(context, value):
            reached.set()
            if cancel:
                await asyncio.Event().wait()
            raise ValueError("after first contribution")

        filing = edit(accumulation(), "feedback-w2-1", lambda work: replace(work, implementation=interrupt))
        ready = prepare(Contract().accept(filing), lowering=lowering).publish()
        task = asyncio.create_task(Engine(ready).run())
        await until_reached(task, reached)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else BaseExceptionGroup) as error:
            await task
        if not cancel:
            assert any(isinstance(item, WorkFailed) for item in leaves(error.value))
        state = ready.state
        assert state.image is not None
        runtime = state.image.units["weights"]
        assert runtime.steps == 0 and runtime.pending is not None
        assert runtime.window is not None
        assert runtime.window.origins == (1, 2)
        assert not state.coordinates["learning"] and not state.completed
        assert "model" in state.unusable and "policy" in state.unusable_states
        assert not state.active_retentions and not state.readers and not state.writers and not state.active_producers
        assert all(record.optimizer == "not_attempted" for record in runtime.outcomes)
        with pytest.raises(NotReady):
            state.capture()

    run(scenario())


def test_runtime_rejects_old_tokens_foreign_regions_and_changed_destination_state():
    ready = prepare(Contract().accept(accumulation())).publish()
    state = ready.state
    assert state.image is not None
    runtime = state.image.units["weights"]
    parameter = runtime.parameters[0]
    region = object()
    first = state.contribute(parameter.square(), ("weights",), 1, region=region)
    second = state.contribute(parameter.square(), ("weights",), 2, region=region, previous=first)
    assert parameter.grad is not None
    assert float(parameter.grad) == 8.0
    for token, scope in ((first, region), (replace(second), region), (second, object())):
        with pytest.raises(NotReady, match="gradient handback"):
            state.advance(token, ("weights",), 3, region=scope)
    with pytest.raises(NotReady, match="gradient handback"):
        state.contribute(parameter.square(), ("weights",), 3, region=region, previous=first)
    state.returned_writes((Use("model", "write"),))
    with pytest.raises(NotReady, match="gradient handback"):
        state.advance(second, ("weights",), 3, region=region)
    assert parameter.grad is not None
    assert runtime.steps == 0 and runtime.pending == 2 and float(parameter.grad) == 8.0


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_joint_accumulated_units_keep_partial_advancement_outcomes(lowering, monkeypatch):
    def loss(context):
        return (context.models["a"].weight + context.models["b"].weight).square()

    names = ("A", "B")
    first = Block(
        "first",
        (
            Retain("hold-1", ("a", "b"), names, "old-1"),
            Call("loss-1", loss, outputs=(Output("L1", torch.Tensor),), uses=(Use("a"), Use("b")), retained="old-1"),
            Differentiate("gradient-1", "L1", names, "g1", retained="old-1"),
            Release("release-1", "old-1", after=("gradient-1",)),
        ),
    )
    second = Block(
        "second",
        (
            Retain("hold-2", ("a", "b"), names, "old-2"),
            Call("loss-2", loss, outputs=(Output("L2", torch.Tensor),), uses=(Use("a"), Use("b")), retained="old-2"),
            Differentiate("gradient-2", "L2", names, "g2", retained="old-2", accumulate="g1"),
            Release("release-2", "old-2", after=("gradient-2",)),
        ),
    )
    filing = Run(
        Sequence("joint", (first, second, Block("finish", (Advance("step", "g2", names),)))),
        (scale("a", 2.0), scale("b", 3.0)),
        (Unit("A", (("a", "weight"),), learning_rate=0.1), Unit("B", (("b", "weight"),), learning_rate=0.1)),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    assert ready.state.image is not None
    a, b = (ready.state.image.units[name] for name in names)

    def fail(*args, **kwargs):
        with torch.no_grad():
            b.parameters[0].add_(1.0)
        raise ValueError("B may have advanced")

    monkeypatch.setattr(b.optimizer, "step", fail)
    with pytest.raises(WorkFailed) as error:
        run(Engine(ready).run())
    assert a.steps == 1 and b.steps == 0 and a.pending is None and b.pending == 2
    reference = torch.nn.Parameter(torch.tensor(2.0))
    reference.grad = torch.tensor(20.0)
    torch.optim.SGD((reference,), lr=0.1).step()
    torch.testing.assert_close(a.parameters[0].detach(), reference.detach(), rtol=0, atol=0)
    assert float(b.parameters[0].detach()) == pytest.approx(4.0)
    assert [(record.optimizer, record.reset) for record in a.outcomes] == [("returned", "returned")] * 2
    assert [(record.optimizer, record.reset) for record in b.outcomes] == [("uncertain", "not_attempted")] * 2
    assert len(error.value.unit_outcomes) == 4
    assert ready.state.unusable == {"a", "b"} and not ready.state.active_retentions


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_selected_feedback_can_choose_a_different_prepared_window(lowering):
    filing = replace(accumulation(), participants=(scale("model", 0.0),))
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    run(Engine(ready).run())
    policy = ready.state.owners["policy"]
    assert isinstance(policy, AccumulationState) and policy.decisions == [2, 2]
    assert ready.state.coordinates["learning"] == {"window-2": 2}
    assert "window-3" not in ready.state.coordinates


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_capability_receives_one_completion_per_region_with_its_request_identity(lowering):
    filing = accumulation()
    assert isinstance(filing.root, Together)

    def observe(context):
        records = context.states["observations"]
        assert isinstance(records, list)
        records.append(context.choice.payload)

    request = Requested(
        "validation",
        "learning",
        Block("inspect", (Call("observe", observe, states=("observations",)),)),
        lambda completion: Choice("inspect", completion.key, completion),
    )
    filing = replace(
        filing,
        root=replace(filing.root, children=(*filing.root.children, request)),
        states=(*filing.states, OwnedState("observations", list)),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    run(Engine(ready).run())
    records = ready.state.owners["observations"]
    assert isinstance(records, list)
    assert [(record.scope, record.block, record.key, record.position) for record in records] == [
        ("learning", "window-2", 0, 1),
        ("learning", "window-3", 1, 2),
    ]
    origins = {}
    for event, value in ready.state.trace:
        if event == "invocation":
            assert isinstance(value, tuple)
            scope, block, key, invocation = value
            origins[(scope, block, key)] = invocation
    assert [record.invocation for record in records] == [origins[("window-2", "finish-2", 0)], origins[("window-3", "finish-3", 1)]]


def test_clipping_occurs_once_at_each_complete_window(monkeypatch):
    filing = accumulation()
    filing = replace(filing, units=(replace(filing.units[0], clip_norm=5.0),))
    ready = prepare(Contract().accept(filing)).publish()
    norms = []
    clip = torch.nn.utils.clip_grad_norm_

    def record(parameters, *args, **kwargs):
        parameters = tuple(parameters)
        assert parameters[0].grad is not None
        norms.append(float(parameters[0].grad))
        return clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record)
    run(Engine(ready).run())
    assert len(norms) == 2 and norms[0] == pytest.approx(10.0) and norms[1] == pytest.approx(77.0)
    assert ready.state.image is not None
    assert float(ready.state.image.units["weights"].parameters[0].detach()) == pytest.approx(1.0)


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_cancellation_during_later_worker_keeps_pending_window_and_retention_until_drain(lowering):
    async def scenario():
        entered, stopping = asyncio.Event(), asyncio.Event()
        finish = threading.Event()
        loop = asyncio.get_running_loop()

        def place(work):
            original = work.implementation

            def blocked(context, value):
                loop.call_soon_threadsafe(entered.set)
                for _ in range(300):
                    if context.stop_requested():
                        break
                    finish.wait(0.01)
                else:
                    raise AssertionError("worker did not receive cancellation")
                loop.call_soon_threadsafe(stopping.set)
                assert finish.wait(3), "worker was not drained"
                return original(context, value)

            return replace(work, implementation=blocked)

        filing = edit(accumulation(), "objective-w2-1", place)
        ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("objective-w2-1",)).publish()
        state = ready.state
        assert state.image is not None
        runtime = state.image.units["weights"]
        task = asyncio.create_task(Engine(ready).run())
        try:
            await until_reached(task, entered)
            task.cancel()
            await until_reached(task, stopping)
            assert not task.done() and state.active_retentions and state.readers == {"model": 1}
            assert runtime.steps == 0 and runtime.pending == 1
            task.cancel()
            await asyncio.sleep(0)
            assert state.active_retentions and runtime.pending == 1
        finally:
            finish.set()
            task.cancel()
            result = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert not state.active_retentions and not state.readers and not state.writers and not state.active_producers
        assert runtime.steps == 0 and runtime.pending == 1 and runtime.window is not None
        assert runtime.window.origins == (1,) and not state.coordinates["learning"]
        assert "model" in state.unusable and not state.completed

    run(scenario())


def test_repeated_sequences_preserve_cross_activity_wait_cycle_rejection_before_load():
    filing, _ = pair()
    assert isinstance(filing.root, Together)
    reached = []

    def policy(counts, state):
        return Choice(state) if not counts else None

    states = tuple(OwnedState(f"policy-{child.name}", lambda name=child.name: name) for child in filing.root.children)
    filing = replace(
        filing,
        root=replace(
            filing.root,
            children=tuple(Repeat(f"repeat-{child.name}", (child,), policy, f"policy-{child.name}") for child in filing.root.children),
        ),
        states=states,
        participants=tuple(replace(participant, load=lambda: reached.append("load")) for participant in filing.participants),
    )
    accepted = Contract().accept(filing)
    with pytest.raises(NotReady, match="reservation wait cycle"):
        prepare(accepted, worker_calls=("compute-x", "compute-y"), worker_limit=2)
    assert not reached


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_state_only_sequence_and_block_alternatives_share_repetition_without_optimizer_placeholders(lowering):
    def remember(context):
        records = context.states["selected"]
        assert isinstance(records, list)
        records.append((context.choice.block, context.choice.key, context.choice.payload))

    sequence = Sequence(
        "pair",
        (Block("first", (Call("one", remember, states=("selected",)),)), Block("second", (Call("two", remember, states=("selected",)),))),
    )
    other = Block("single", (Call("three", remember, states=("selected",)),))

    def choose(counts, state):
        if not counts:
            return Choice("pair", "first-request", 7)
        if not counts.get("single"):
            return Choice("single", "second-request", 9)
        return None

    filing = Run(
        Repeat("policy", (sequence, other), choose, "decisions"), states=(OwnedState("decisions", dict), OwnedState("selected", list))
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    run(Engine(ready).run())
    assert ready.state.owners["selected"] == [("pair", "first-request", 7), ("pair", "first-request", 7), ("single", "second-request", 9)]
    assert ready.state.coordinates["policy"] == {"pair": 1, "single": 1}
    assert ready.state.image is not None and not ready.state.image.units and not ready.state.image.bindings


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_missing_later_input_does_not_complete_an_accumulation_window(lowering):
    filing = accumulation()
    assert isinstance(filing.root, Together)
    source = filing.root.children[0]

    async def only_first(context):
        await context.publish("micro-0", Product(0, torch.tensor(1.0)))

    filing = replace(filing, root=replace(filing.root, children=(replace(source, implementation=only_first), *filing.root.children[1:])))
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    with pytest.raises(ExceptionGroup) as error:
        run(Engine(ready).run())
    assert any(isinstance(item, NotReady) and "Missing requested work" in str(item) for item in leaves(error.value))
    assert ready.state.image is not None
    runtime = ready.state.image.units["weights"]
    assert runtime.pending == 1 and runtime.steps == 0 and runtime.window is not None
    assert runtime.window.origins == (1,)
    assert not ready.state.coordinates["learning"] and not ready.state.completed
    assert not ready.state.active_retentions and "model" in ready.state.unusable


def test_intervening_declared_write_rejects_accumulation_before_the_next_forward():
    def mutate(context):
        with torch.no_grad():
            context.models["model"].weight.add_(1.0)

    filing = accumulation()
    assert isinstance(filing.root, Together)
    learning = filing.root.children[1]
    assert isinstance(learning, Repeat)
    sequence = learning.blocks[0]
    assert isinstance(sequence, Sequence)
    first = replace(
        sequence.blocks[0],
        work=(*sequence.blocks[0].work, Call("intervening-write", mutate, uses=(Use("model", "write"),), after=("release-w2-0",))),
    )
    sequence = replace(sequence, blocks=(first, *sequence.blocks[1:]))
    learning = replace(learning, blocks=(sequence, *learning.blocks[1:]))
    filing = replace(filing, root=replace(filing.root, children=(filing.root.children[0], learning)))
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(ExceptionGroup) as error:
        run(Engine(ready).run())
    assert any(isinstance(item, WorkFailed) and "gradient handback" in str(item) for item in leaves(error.value))
    assert ready.state.image is not None
    runtime = ready.state.image.units["weights"]
    assert runtime.pending == 1 and runtime.steps == 0 and ready.state.stamp("model").numerical_epoch == 1
    assert not any(event == "entered" and isinstance(value, tuple) and value[0] == "objective-w2-1" for event, value in ready.state.trace)


def test_policy_cannot_select_an_unaccepted_region_after_a_completed_window():
    def outside(context, loss):
        state = context.states["policy"]
        assert isinstance(state, AccumulationState)
        state.next_count = 4

    filing = edit(accumulation(), "feedback-w2-1", lambda work: replace(work, implementation=outside))
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(ExceptionGroup):
        run(Engine(ready).run())
    assert ready.state.image is not None
    runtime = ready.state.image.units["weights"]
    assert runtime.steps == 1 and runtime.pending is None
    assert ready.state.coordinates["learning"] == {"window-2": 1}
    assert all(record.optimizer == record.reset == "returned" for record in runtime.outcomes)
    assert "policy" in ready.state.unusable_states and not ready.state.completed


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_live_window_outcomes_are_not_evicted_by_the_bounded_completed_history(lowering, monkeypatch):
    blocks = [Block("input", (Join("input", ("input",), (Output("input-value", float),), lambda *args: True),))]
    previous = None
    for index in range(36):
        blocks.append(
            Block(
                f"micro-{index}",
                (
                    Retain(f"hold-{index}", ("model",), ("weights",), f"old-{index}"),
                    Call(
                        f"loss-{index}",
                        lambda context, value: context.models["model"].weight.square() * value,
                        inputs=("input-value",),
                        outputs=(Output(f"L{index}", torch.Tensor),),
                        uses=(Use("model"),),
                        retained=f"old-{index}",
                    ),
                    Differentiate(
                        f"gradient-{index}", f"L{index}", ("weights",), f"g{index}", retained=f"old-{index}", accumulate=previous
                    ),
                    Release(f"release-{index}", f"old-{index}", after=(f"gradient-{index}",)),
                ),
            )
        )
        previous = f"g{index}"
    assert previous is not None
    blocks.append(Block("finish", (Advance("step", previous, ("weights",)),)))

    async def produce(context):
        await context.publish("input", Product("", 2.0))

    filing = Run(
        Together("run", (Producer("source", ("input",), produce), Sequence("window", tuple(blocks)))),
        (scale("model", 2.0),),
        (Unit("weights", (("model", "weight"),)),),
        channels=(Channel("input", 1),),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    assert ready.state.image is not None
    runtime = ready.state.image.units["weights"]

    def fail():
        raise ValueError("uncertain optimizer")

    monkeypatch.setattr(runtime.optimizer, "step", fail)
    with pytest.raises(ExceptionGroup) as error:
        run(Engine(ready).run())
    failure = next(item for item in leaves(error.value) if isinstance(item, WorkFailed))
    assert len(runtime.outcomes) == 32 and len(runtime.active_outcomes) == 36
    origins = tuple(runtime.active_outcomes)
    assert runtime.window is not None and runtime.window.origins == origins
    assert len(failure.unit_outcomes) == 36
    assert all(record.contribution == "ready" and record.optimizer == "uncertain" for record in failure.unit_outcomes)
    assert runtime.steps == 0 and runtime.pending == origins[-1]
    assert all(record.input_handoffs == failure.input_handoffs for record in failure.unit_outcomes)
    assert {record.work for record in failure.unit_outcomes} == {(f"micro-{index}", f"gradient-{index}") for index in range(36)}
    assert len(failure.input_handoffs) == 1 and failure.input_handoffs[0].invocation < origins[0]
    assert len({id(record._input_links) for record in failure.unit_outcomes}) == 1  # One shared prefix, not 36 copies.


@pytest.mark.parametrize("advance", (False, True))
def test_missing_gradient_cannot_extend_or_advance_a_verified_window(advance):
    ready = prepare(Contract().accept(accumulation())).publish()
    state = ready.state
    assert state.image is not None
    unit = state.image.units["weights"]
    parameter = unit.parameters[0]
    region = object()
    contribution = state.contribute(parameter.square(), ("weights",), 1, region=region)
    parameter.grad = None
    with pytest.raises(NotReady, match="gradient handback"):
        if advance:
            state.advance(contribution, ("weights",), 2, region=region)
        else:
            state.contribute(parameter.square(), ("weights",), 2, previous=contribution, region=region)
    assert unit.steps == 0 and unit.pending == 1 and tuple(unit.active_outcomes) == (1,)
    assert float(parameter.detach()) == 2.0


def test_unverified_first_contribution_cannot_be_erased_by_publication_or_capture():
    filing = edit(accumulation(), "objective-w2-0", lambda work: replace(work, implementation=lambda context, value: value.square()))
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(ExceptionGroup):
        run(Engine(ready).run())
    state = ready.state
    assert state.image is not None
    unit = state.image.units["weights"]
    assert unit.pending is None and unit.window is None and tuple(unit.active_outcomes) == (1,)
    assert unit.active_outcomes[1].contribution == "uncertain"
    candidate = state.candidate({"model": Scale(3.0)}, replacing="model")
    with pytest.raises(NotReady, match="quiescent retained/contribution"):
        state.install(candidate, state.generation)
    with pytest.raises(NotReady, match="unfinished gradients"):
        run(state.replace(Replacement("model", 1, Scale(3.0)), "model"))
    with pytest.raises(NotReady):
        state.capture()
    assert state.generation == 1 and state.image.units["weights"] is unit


def test_empty_origin_correspondence_rejects_explicitly_instead_of_using_a_fallback():
    ready = prepare(Contract().accept(accumulation())).publish()
    state = ready.state
    assert state.image is not None
    unit = state.image.units["weights"]
    contribution = state.contribute(unit.parameters[0].square(), ("weights",), 1)
    # Invalid handback cannot replace the owner's current correspondence.
    empty = replace(contribution, origins=())
    with pytest.raises(NotReady, match="gradient handback"):
        state.advance(empty, ("weights",), 1)
    assert unit.steps == 0 and unit.pending == 1 and unit.window is contribution
