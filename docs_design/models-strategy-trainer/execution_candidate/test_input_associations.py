"""Input facts follow accepted execution relationships, not diagnostic retention."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
import weakref

import pytest
import torch

from .examples import scale
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
    Release,
    Repeat,
    Requested,
    Retain,
    Run,
    Sequence,
    Together,
    TwoPass,
    Unit,
    Use,
)
from .prepare import Engine, WorkFailed, prepare
from .state import Feed, InputScope, Product, Projection, Stamp, handoff_facts, join
from .test_candidate import leaves, run


def test_scope_snapshots_share_immutable_prefixes_and_check_bounds_before_claim():
    async def scenario():
        changed = asyncio.Condition()
        first, second = Feed(1, changed, name="first"), Feed(1, changed, name="second")
        stamp = Stamp("source", 2, 7)
        a, b = Product("key", object(), (stamp,)), Product("key", object())
        await first.publish(a)
        await second.publish(b)
        projection = Projection({}, {})
        too_small = InputScope("region", 1)
        with pytest.raises(NotReady, match="member bound"):
            await join((first, second), "key", 1, lambda *args: True, projection, input_scope=too_small, work=("block", "pair"))
        assert too_small.head is None and too_small._count == 0
        assert first.ready == {"key": a} and second.ready == {"key": b}
        assert not first.handed_off and not second.handed_off

        inputs = InputScope("region", 2)
        await join((first,), "key", 1, lambda *args: True, projection, input_scope=inputs, work=("block", "one"))
        prefix = inputs.head
        await join((second,), "key", 2, lambda *args: True, projection, input_scope=inputs, work=("block", "two"))
        assert inputs.head is not None and inputs.head.previous is prefix
        assert [(fact.channel, fact.invocation, fact.provenance) for fact in handoff_facts(prefix)] == [("first", 1, (stamp,))]
        assert [fact.channel for fact in handoff_facts(inputs.head)] == ["first", "second"]
        await first.publish(Product("late", None))
        inputs.closed = True
        with pytest.raises(NotReady, match="closed"):
            await join((first,), "late", 3, lambda *args: True, projection, input_scope=inputs, work=("block", "late"))
        assert "late" in first.ready and inputs._count == 2

    run(scenario())


def test_inadmissible_join_creates_no_input_facts_or_partial_claim():
    async def scenario():
        changed = asyncio.Condition()
        a, b = Feed(1, changed, name="a"), Feed(1, changed, name="b")
        await a.publish(Product("key", "good"))
        await b.publish(Product("key", "bad"))
        inputs = InputScope("region", 2)
        with pytest.raises(NotReady, match="Inadmissible"):
            await join((a, b), "key", 1, lambda value, projection: value.value == "good", Projection({}, {}), input_scope=inputs)
        assert inputs.head is None and inputs._count == 0
        assert "key" in a.ready and "key" in b.ready and not a.handed_off and not b.handed_off

    run(scenario())


def test_recorded_join_requires_a_prepared_work_address_before_claim():
    async def scenario():
        feed = Feed(1, asyncio.Condition(), name="input")
        product = Product("key", object())
        await feed.publish(product)
        inputs = InputScope("region", 1)
        with pytest.raises(NotReady, match="prepared work address"):
            await join((feed,), "key", 1, lambda *args: True, Projection({}, {}), input_scope=inputs)
        assert inputs.head is None and inputs._count == 0
        assert feed.ready == {"key": product} and not feed.handed_off

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
@pytest.mark.parametrize("termination", ("returned", "failure", "cancel"))
def test_cross_block_inputs_and_actual_provenance_follow_pending_outcomes(lowering, access, termination, monkeypatch):
    async def scenario():
        waiting = asyncio.Event()
        origin = []

        async def produce(context):
            async with context.use_source() as source:
                await context.publish("input", Product("example", 2.0, (source.stamps["encoder"],)))

        def loss(context, value):
            origin.append(context.invocation)
            assert len(context.input_handoffs) == 1
            return context.models["model"](torch.tensor(value)).square()

        def change_source(context):
            with torch.no_grad():
                context.models["encoder"].weight.add_(1)

        async def pause(context):
            waiting.set()
            if termination == "cancel":
                await asyncio.Event().wait()

        receive = Block(
            "receive",
            (
                Join(
                    "input",
                    ("input",),
                    (Output("value", float),),
                    lambda product, projection: product.provenance == (projection.stamps["encoder"],),
                    (Use("encoder"),),
                ),
            ),
        )
        compute = Block(
            "compute",
            (
                Retain("hold", ("model",), ("weights",), "source"),
                Call("loss", loss, ("value",), (Output("loss", torch.Tensor),), (Use("model"),), retained="source"),
                Differentiate("gradient", "loss", ("weights",), "gradient", retained="source"),
                Release("release", "source", after=("gradient",)),
            ),
        )
        mutate = Block(
            "mutate",
            (Call("change-source", change_source, uses=(Use("encoder", "write"),)), Call("pause", pause, after=("change-source",))),
        )
        finish = Block("finish", (Advance("advance", "gradient", ("weights",)),))
        sequence = Sequence("window", (receive, compute, mutate, finish))
        filing = Run(
            Together(
                "run",
                (
                    Producer("source", ("input",), produce, (Use("encoder"),)),
                    Repeat("learning", (sequence,), lambda counts, state: None if counts else Choice("window", "example"), "policy"),
                ),
            ),
            (scale("model", 1.0), scale("encoder", 3.0)),
            (Unit("weights", (("model", "weight"),)),),
            (OwnedState("policy", lambda: None),),
            (Channel("input", 1),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, access=access).publish()
        state = ready.state
        assert state.image is not None
        unit = state.image.units["weights"]
        engine = Engine(ready)

        def fail():
            raise ValueError("optimizer")

        if termination == "failure":
            monkeypatch.setattr(unit.optimizer, "step", fail)
        task = asyncio.create_task(engine.run())
        await waiting.wait()
        if termination == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif termination == "failure":
            with pytest.raises(ExceptionGroup) as caught:
                await task
            failures = [error for error in leaves(caught.value) if isinstance(error, WorkFailed)]
            assert failures and failures[0].unit_outcomes[0].input_handoffs
        else:
            await task

        records = unit.evidence(origin[0], None) if termination == "returned" else tuple(unit.active_outcomes.values())
        assert len(records) == 1
        assert records[0].work == ("compute", "gradient")
        facts = records[0].input_handoffs
        assert len(facts) == 1 and facts[0].key == "example" and facts[0].channel == "input"
        assert facts[0].work == ("receive", "input") and facts[0].invocation != origin[0]
        assert facts[0].provenance[0].numerical_epoch == 0 and state.stamp("encoder").numerical_epoch == 1
        assert facts[0].provenance[0].participant_ref == state.accepted.participant_refs["encoder"]
        assert unit.steps == int(termination == "returned") and unit.unfinished == (termination != "returned")
        assert not state.progress.leases and not state.active_retentions
        if termination != "returned":
            assert facts == engine.failure_input_handoffs
            with pytest.raises(NotReady):
                state.capture()

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("optimized", (False, True))
def test_completed_work_releases_links_with_only_bounded_outcome_history(lowering, optimized, monkeypatch):
    nodes = weakref.WeakSet()
    payloads = []
    original = InputScope.record

    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        assert self.head is not None
        nodes.add(self.head)

    monkeypatch.setattr(InputScope, "record", record)

    async def produce(context):
        for key in range(100):
            value = torch.tensor(float(key))
            payloads.append(weakref.ref(value))
            await context.publish("input", Product(key, value))

    def consume(context, value):
        assert context.input_handoffs[0].key == context.choice.key
        return context.models["model"](value).square() if optimized else None

    work = (
        Join("input", ("input",), (Output("value", torch.Tensor),), lambda *args: True),
        Call("consume", consume, ("value",), (Output("loss", torch.Tensor),) if optimized else (), (Use("model"),) if optimized else ()),
    )
    if optimized:
        work = (*work, Differentiate("gradient", "loss", ("weights",), "gradient"), Advance("advance", "gradient", ("weights",)))
    filing = Run(
        Together(
            "run",
            (
                Producer("source", ("input",), produce),
                Repeat(
                    "learning",
                    (Block("take", work),),
                    lambda counts, state: Choice("take", counts.get("take", 0)) if counts.get("take", 0) < 100 else None,
                    "policy",
                ),
            ),
        ),
        (scale("model", 0.1),) if optimized else (),
        (Unit("weights", (("model", "weight"),), learning_rate=0.0),) if optimized else (),
        (OwnedState("policy", lambda: None),),
        (Channel("input", 2),),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    run(Engine(ready).run())
    gc.collect()
    assert len(nodes) == (32 if optimized else 0)
    assert all(reference() is None for reference in payloads)
    assert ready.state.coordinates["learning"] == {"take": 100}
    assert not ready.state._input_failures


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("termination", ("returned", "failure", "cancel", "bad-request"))
def test_requested_work_keeps_active_queued_and_in_transit_input_associations(lowering, termination, monkeypatch):
    async def scenario():
        active, drained, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        sender_blocked, sender_cancelled = asyncio.Event(), asyncio.Event()
        senders, failures = [], []
        observed = []

        class ObservedQueue(asyncio.Queue):
            async def put(self, item):
                if item is not None and item[0].key == "learn-2":
                    assert self.full()
                    senders.append(asyncio.current_task())
                    sender_blocked.set()
                    try:
                        await super().put(item)
                    except asyncio.CancelledError:
                        sender_cancelled.set()
                        raise
                else:
                    await super().put(item)

        monkeypatch.setattr(asyncio, "Queue", ObservedQueue)

        async def produce(context):
            for key in ("learn-0", "learn-1", "learn-2", *range(48)):
                await context.publish("input", Product(key, key))

        def learn(context, value):
            assert value == context.choice.key

        async def inspect(context):
            facts = context.input_handoffs
            assert len(facts) == 1
            observed.append(facts[0].key)
            active.set()
            await release.wait()
            assert context.input_handoffs == facts

        async def wait(context):
            await active.wait()

        async def consume(context, value):
            if value == 47:
                await sender_blocked.wait()
                drained.set()
                if termination == "failure":
                    raise ValueError("independent consumer")
                if termination == "returned":
                    release.set()

        def request(completion):
            if termination == "bad-request":
                raise ValueError("request policy")
            return Choice("inspect", completion.key)

        learning = Block(
            "learn", (Join("input", ("input",), (Output("value", object),), lambda *args: True), Call("learn", learn, ("value",)))
        )
        taking = Block(
            "take", (Join("other", ("input",), (Output("value", object),), lambda *args: True), Call("consume", consume, ("value",)))
        )
        filing = Run(
            Together(
                "run",
                (
                    Producer("source", ("input",), produce),
                    Repeat(
                        "learning",
                        (learning,),
                        lambda counts, state: Choice("learn", f"learn-{counts.get('learn', 0)}") if counts.get("learn", 0) < 3 else None,
                        "learning-policy",
                    ),
                    Requested("inspection", "learning", Block("inspect", (Call("inspect", inspect),)), request),
                    Repeat(
                        "other",
                        (Block("wait", (Call("wait", wait),)), taking),
                        lambda counts, state: (
                            Choice("wait") if not counts else Choice("take", counts.get("take", 0)) if counts.get("take", 0) < 48 else None
                        ),
                        "other-policy",
                    ),
                ),
            ),
            states=(OwnedState("learning-policy", lambda: None), OwnedState("other-policy", lambda: None)),
            channels=(Channel("input", 2),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering).publish()
        input_failed = ready.state.input_failed

        def record_failure(head):
            failures.append((asyncio.current_task(), {fact.key for fact in handoff_facts(head)}))
            input_failed(head)

        monkeypatch.setattr(ready.state, "input_failed", record_failure)
        engine = Engine(ready)
        task = asyncio.create_task(engine.run())
        if termination == "bad-request":
            with pytest.raises(ExceptionGroup):
                await task
            assert any(fact.key == "learn-0" for fact in engine.failure_input_handoffs)
            return
        await drained.wait()
        if termination == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif termination == "failure":
            with pytest.raises(ExceptionGroup):
                await task
        else:
            await task
        assert not any(str(event[0]).startswith("learn-") for event in ready.state.feeds["input"].handed_off)
        assert not ready.state.progress.leases and not ready.state._input_failures
        if termination == "returned":
            assert observed == ["learn-0", "learn-1", "learn-2"]
            assert sender_blocked.is_set() and not sender_cancelled.is_set()
        else:
            assert sender_cancelled.is_set() and len(senders) == 1
            assert any(task is senders[0] and "learn-2" in keys for task, keys in failures)
            assert {fact.key for fact in engine.failure_input_handoffs if isinstance(fact.key, str)} == {"learn-0", "learn-1", "learn-2"}
            if termination == "failure":
                assert 47 in {fact.key for fact in engine.failure_input_handoffs}

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_declined_requests_have_a_terminal_disposition_without_retaining_inputs(lowering, monkeypatch):
    nodes = weakref.WeakSet()
    original = InputScope.record

    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        nodes.add(self.head)

    monkeypatch.setattr(InputScope, "record", record)
    delivered = []

    async def produce(context):
        for key in range(100):
            await context.publish("input", Product(key, key))

    def decline(completion):
        delivered.append(completion)
        return None

    def unexpected(context):
        pytest.fail("A declined request must not enter capability work")

    filing = Run(
        Together(
            "run",
            (
                Producer("source", ("input",), produce),
                Repeat(
                    "learning",
                    (Block("learn", (Join("input", ("input",), (Output("value", int),), lambda *args: True),)),),
                    lambda counts, state: Choice("learn", counts.get("learn", 0)) if counts.get("learn", 0) < 100 else None,
                    "policy",
                ),
                Requested("inspection", "learning", Block("inspect", (Call("inspect", unexpected),)), decline),
            ),
        ),
        states=(OwnedState("policy", lambda: None),),
        channels=(Channel("input", 1),),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    run(Engine(ready).run())
    gc.collect()
    assert len(delivered) == 100 and len(nodes) == 0
    declined = [details for event, details in ready.state.trace if event == "request_declined"]
    assert 0 < len(declined) < 100  # Terminal diagnostics remain bounded.
    assert declined[-1] == ("inspection", delivered[-1])
    assert not ready.state._input_failures


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_capability_keeps_trigger_associations_separate_from_its_own_admission(lowering):
    observed = []

    async def produce(context):
        await context.publish("training", Product("train-key", "train-value"))
        await context.publish("evaluation", Product("eval-key", "eval-value"))

    def inspect(context, value):
        assert value == "eval-value" and context.choice.key == "eval-key"
        observed.extend(context.input_handoffs)

    learn = Block("learn", (Join("train", ("training",), (Output("train", str),), lambda *args: True),))
    evaluate = Block(
        "evaluate",
        (
            Join("eval", ("evaluation",), (Output("eval", str),), lambda *args: True),
            Call("inspect", inspect, ("eval",)),
        ),
    )
    filing = Run(
        Together(
            "run",
            (
                Producer("source", ("training", "evaluation"), produce),
                Repeat("learning", (learn,), lambda counts, state: None if counts else Choice("learn", "train-key"), "policy"),
                Requested("validation", "learning", evaluate, lambda done: Choice("evaluate", "eval-key")),
            ),
        ),
        states=(OwnedState("policy", lambda: None),),
        channels=(Channel("training", 1), Channel("evaluation", 1)),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    run(Engine(ready).run())
    assert [(fact.scope, fact.work, fact.key) for fact in observed] == [
        ("learn", ("learn", "train"), "train-key"),
        ("evaluate", ("evaluate", "eval"), "eval-key"),
    ]
    assert observed[0].invocation != observed[1].invocation


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_worker_cancellation_keeps_input_facts_through_actual_completion(lowering):
    async def scenario():
        started, stopped, release = threading.Event(), threading.Event(), threading.Event()
        observed = []

        async def produce(context):
            await context.publish("input", Product("worker", 2.0))

        def selected(context, value):
            facts = context.input_handoffs
            assert len(facts) == 1 and facts[0].key == "worker"
            started.set()
            assert context.stop_requested is not None
            while not context.stop_requested():
                time.sleep(0.001)
            stopped.set()
            assert release.wait(3)
            assert context.input_handoffs == facts
            context.models["model"](torch.tensor(value))
            observed.extend(facts)

        work = Block(
            "work",
            (
                Join("input", ("input",), (Output("value", float),), lambda *args: True),
                Call("selected", selected, ("value",), uses=(Use("model"),)),
            ),
        )
        filing = Run(
            Together(
                "run",
                (
                    Producer("source", ("input",), produce),
                    Repeat("learning", (work,), lambda counts, state: None if counts else Choice("work", "worker"), "policy"),
                ),
            ),
            (scale("model", 1.0),),
            states=(OwnedState("policy", lambda: None),),
            channels=(Channel("input", 1),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("selected",)).publish()
        engine = Engine(ready)
        task = asyncio.create_task(engine.run())
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            while not stopped.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not engine.failure_input_handoffs
            assert ready.state.readers.get("model") == 1
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert tuple(observed) == engine.failure_input_handoffs
        assert not ready.state.readers and "model" in ready.state.unusable

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("granted", (False, True))
def test_partial_joint_and_granted_failure_keep_the_entered_input_association(lowering, granted, monkeypatch):
    async def produce(context):
        await context.publish("input", Product("key", 2.0))

    def objective(context, value):
        return (context.models["a"](torch.tensor(value)) + context.models["b"](torch.tensor(3.0)) - 1).square()

    def selected(access, models, value):
        access.backward((models["a"](torch.tensor(value)) - 1).square())
        access.perturb(0.1)
        access.reset()
        access.backward((models["a"](torch.tensor(value)) - 1).square())
        access.restore()
        raise ValueError("after restoration")

    work = (
        Join("input", ("input",), (Output("value", float),), lambda *args: True),
        TwoPass("granted", selected, ("value",), "A", "gradient", (Use("a", "write"),))
        if granted
        else Call("loss", objective, ("value",), (Output("loss", torch.Tensor),), (Use("a"), Use("b"))),
    )
    units = ("A",) if granted else ("A", "B")
    if not granted:
        work = (*work, Differentiate("gradient", "loss", units, "gradient"))
    work = (*work, Advance("advance", "gradient", units))
    filing = Run(
        Together(
            "run",
            (
                Producer("source", ("input",), produce),
                Repeat("learning", (Block("work", work),), lambda counts, state: None if counts else Choice("work", "key"), "policy"),
            ),
        ),
        (scale("a", 1.0),) if granted else (scale("a", 1.0), scale("b", 0.5)),
        (Unit("A", (("a", "weight"),)),) if granted else (Unit("A", (("a", "weight"),)), Unit("B", (("b", "weight"),))),
        (OwnedState("policy", lambda: None),),
        (Channel("input", 1),),
    )
    ready = prepare(Contract(two_pass=granted).accept(filing), lowering=lowering).publish()
    assert ready.state.image is not None
    if not granted:

        def fail():
            raise ValueError("second optimizer")

        monkeypatch.setattr(ready.state.image.units["B"].optimizer, "step", fail)
    engine = Engine(ready)
    with pytest.raises(ExceptionGroup) as caught:
        run(engine.run())
    failure = next(error for error in leaves(caught.value) if isinstance(error, WorkFailed))
    assert failure.work == ("work", "granted" if granted else "advance")
    assert all(record.input_handoffs == failure.input_handoffs for record in failure.unit_outcomes)
    assert all(record.work == ("work", "granted" if granted else "gradient") for record in failure.unit_outcomes)
    assert len(failure.input_handoffs) == 1 and failure.input_handoffs[0].key == "key"
    assert engine.failure_input_handoffs == failure.input_handoffs
    if granted:
        assert failure.unit_outcomes[0].contribution == "unverified_grant"
        assert failure.unit_outcomes[0].optimizer == "not_attempted"
    else:
        assert [(record.optimizer, record.reset) for record in failure.unit_outcomes] == [
            ("returned", "returned"),
            ("uncertain", "not_attempted"),
        ]
