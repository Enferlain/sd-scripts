"""Shared progress and input correlation across independently progressing work."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
import torch

from .examples import producer_continuity, scale
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
    Retain,
    Run,
    Sequence,
    Together,
    Unit,
    Use,
    View,
)
from .prepare import Engine, WorkFailed, prepare
from .state import Product
from .test_candidate import leaves, run


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
def test_unfinished_input_association_survives_independent_handoff_history(lowering, access, monkeypatch):
    """Live owner facts survive eviction from both diagnostic histories."""

    async def scenario():
        start, drained = asyncio.Event(), asyncio.Event()
        origins = []

        async def produce(context):
            # No producer touches live weights. The learning policy selects
            # a different key from the independent consumer's numbered work.
            await context.publish("input", Product("training", 2.0))
            for key in range(48):
                await context.publish("input", Product(key, float(key)))

        def loss(context, value):
            origins.append(context.invocation)
            return context.models["model"](torch.tensor(value)).square()

        def start_other(context):
            assert any(event[1] == origins[0] for event in state.feeds["input"].handed_off)
            start.set()

        async def wait_for_start(context):
            await start.wait()

        async def wait_for_drain(context):
            await drained.wait()

        def consume(context, value):
            assert value == float(context.choice.key)
            if context.choice.key == 47:
                drained.set()

        contribution = Block(
            "contribute",
            (
                Join("training-input", ("input",), (Output("value", float),), lambda *args: True),
                Retain("hold", ("model",), ("weights",), "source"),
                Call("loss", loss, ("value",), (Output("loss", torch.Tensor),), (Use("model"),), retained="source"),
                Differentiate("gradient", "loss", ("weights",), "gradient", retained="source"),
                Release("release", "source", after=("gradient",)),
            ),
        )
        waiting = Block("wait", (Call("start-other", start_other), Call("wait-for-drain", wait_for_drain, after=("start-other",))))
        finish = Block("finish", (Advance("advance", "gradient", ("weights",)),))
        window = Sequence("window", (contribution, waiting, finish))

        def choose_window(counts, state):
            key = state.pop("next", None)
            return Choice("window", key) if key is not None else None

        learning = Repeat("learning", (window,), choose_window, "learning-policy")
        wait_start = Block("wait-start", (Call("wait-start", wait_for_start),))
        take = Block(
            "take",
            (
                Join("other-input", ("input",), (Output("other-value", float),), lambda *args: True),
                Call("consume", consume, ("other-value",)),
            ),
        )

        def policy(counts, state):
            if not counts:
                return Choice("wait-start")
            count = counts.get("take", 0)
            return Choice("take", count) if count < 48 else None

        other = Repeat("other-consumer", (wait_start, take), policy, "policy")
        filing = Run(
            Together("run", (Producer("source", ("input",), produce), learning, other)),
            (scale("model", 1.0),),
            (Unit("weights", (("model", "weight"),)),),
            (OwnedState("policy", lambda: None), OwnedState("learning-policy", lambda: {"next": "training"})),
            (Channel("input", 2),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, access=access).publish()
        state = ready.state
        assert state.image is not None
        unit = state.image.units["weights"]

        def fail():
            raise RuntimeError("uncertain backend")

        monkeypatch.setattr(unit.optimizer, "step", fail)
        engine = Engine(ready)
        with pytest.raises(ExceptionGroup) as caught:
            await engine.run()
        failures = [failure for failure in leaves(caught.value) if isinstance(failure, WorkFailed)]
        assert len(failures) == 1 and str(failures[0].cause) == "uncertain backend"
        assert state.coordinates["other-consumer"]["take"] == 48
        assert unit.unfinished and tuple(unit.active_outcomes) == tuple(origins)
        assert [(record.invocation, record.optimizer, record.reset) for record in failures[0].unit_outcomes] == [
            (origins[0], "uncertain", "not_attempted")
        ]
        assert not state.progress.leases and not state.active_retentions
        with pytest.raises(NotReady):
            state.capture()
        assert not any(event[1] == origins[0] for event in state.feeds["input"].handed_off)
        assert not any(kind == "admitted" and isinstance(value, tuple) and value[0] == origins[0] for kind, value in state.trace)
        unit.outcomes.clear()  # The live owner, not even its outcome history.
        facts = unit.active_outcomes[origins[0]].input_handoffs
        assert len(facts) == 1
        assert (facts[0].scope, facts[0].work, facts[0].invocation, facts[0].channel, facts[0].key) == (
            "window",
            ("contribute", "training-input"),
            origins[0],
            "input",
            "training",
        )
        assert facts == failures[0].unit_outcomes[0].input_handoffs == failures[0].input_handoffs == engine.failure_input_handoffs

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
@pytest.mark.parametrize("batch_source", (False, True))
@pytest.mark.parametrize("evaluation", (False, True))
@pytest.mark.parametrize("capacity", (1, 2))
@pytest.mark.parametrize("per_work", (False, True))
def test_bounded_publication_and_source_access_share_progress(lowering, access, batch_source, evaluation, capacity, per_work, monkeypatch):
    async def scenario():
        second_publication = asyncio.Event()
        consumer_access = asyncio.Event()
        evaluated = asyncio.Event()
        source_task = None
        readers_at_evaluation = []

        async def produce(context):
            nonlocal source_task
            source_task = asyncio.current_task()

            async def publish(source, key):
                value = source.models["model"](torch.tensor(float(key))).detach().clone()
                if key == 2:
                    second_publication.set()
                await context.publish("input", Product(key, value))

            if batch_source:
                # One protected source interval spans both products. Do not
                # silently split it, copy it, or evict work to make progress.
                async with context.use_source() as source:
                    for key in (1, 2):
                        await publish(source, key)
            else:
                # Positive control has a DIFFERENT source-use protocol: each
                # detached product is produced before its publication wait.
                for key in (1, 2):
                    async with context.use_source() as source:
                        value = source.models["model"](torch.tensor(float(key))).detach().clone()
                    if key == 2:
                        second_publication.set()
                    await context.publish("input", Product(key, value))

        def evaluate(context, value):
            readers_at_evaluation.append(state.readers.get("model", 0))
            evaluated.set()
            context.views["evaluation"](value)

        consumer = Block(
            "consume",
            (
                Join("join", ("input",), (Output("value", torch.Tensor),), lambda *args: True),
                Call("evaluate", evaluate, ("value",), uses=(Use("model", view="evaluation"),)),
            ),
        )
        if per_work:
            consumer = Sequence("consume-sequence", (consumer,))
        filing = Run(
            Together(
                "run",
                (
                    Producer("source", ("input",), produce, (Use("model"),), stop_when="learning", remaining="discard"),
                    Repeat("learning", (consumer,), lambda counts, state: None if counts else Choice(consumer.name, 1), "policy"),
                ),
            ),
            (scale("model", 1.0),),
            states=(OwnedState("policy", lambda: None),),
            channels=(Channel("input", capacity),),
            views=(View("evaluation", "model", evaluation=evaluation),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, access=access).publish()
        state = ready.state
        acquire = state.acquire

        @asynccontextmanager
        async def observe(uses, states=()):
            if asyncio.current_task() is not source_task and any(use.participant == "model" for use in uses):
                consumer_access.set()
            async with acquire(uses, states) as projection:
                yield projection

        monkeypatch.setattr(state, "acquire", observe)
        task = asyncio.create_task(Engine(ready).run())
        try:
            await second_publication.wait()
            await consumer_access.wait()
            cyclic = batch_source and evaluation and capacity == 1 and not per_work
            if cyclic:
                with pytest.raises(ExceptionGroup) as failure:
                    await task
                assert any(isinstance(error, NotReady) and "resource wait cycle" in str(error) for error in leaves(failure.value))
                assert set(state.feeds["input"].ready) == {1}
                assert not state.feeds["input"].handed_off
                assert not evaluated.is_set()
                assert not state.completed
            else:
                await task
                assert state.completed and evaluated.is_set()
                assert state.feeds["input"].handed_off[0][0] == 1
                if batch_source and not evaluation and capacity == 1 and not per_work:
                    assert readers_at_evaluation == [2]  # Compatible read really overlaps the held source.
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert not state.readers and not state.writers and not state.active_producers
        assert not state.progress.waits and not state.progress.demands and not state.progress.leases

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
def test_producer_example_does_not_require_its_cross_owner_staging_events(lowering, access):
    async def scenario():
        filing = producer_continuity()
        assert isinstance(filing.root, Together)
        children = []
        for child in filing.root.children:
            if isinstance(child, Repeat):
                # Remove wait-ahead and its ordinary Call entirely. Readiness
                # and admission through Join still coordinate the real demand.
                child = replace(
                    child,
                    blocks=tuple(block for block in child.blocks if block.name == "learn"),
                    policy=lambda counts, state: None if counts else Choice("learn", 1),
                )
            children.append(child)
        filing = replace(filing, root=replace(filing.root, children=tuple(children)))
        ready = prepare(Contract().accept(filing), lowering=lowering, access=access).publish()
        await Engine(ready).run()
        state = ready.state
        assert state.image is not None and state.completed
        assert state.image.units["weights"].steps == 1
        assert state.feeds["condition"].handed_off[0][0] == 1
        assert not state.readers and not state.writers and not state.active_producers
        state.capture()  # Checks owner cleanup too; not durable restoration.

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
def test_another_consumer_can_free_capacity_before_conflicting_work(lowering, access):
    async def scenario():
        observed = []

        async def produce(context):
            async with context.use_source():
                await context.publish("input", Product(1, "one"))
                await context.publish("input", Product(2, "two"))

        def evaluate(context):
            context.views["evaluation"](torch.tensor(1.0))

        def record(context, value):
            observed.append(value)

        def selection(name, key):
            return lambda counts, state: None if counts else Choice(name, key)

        # The first consumer waits for a conflicting view BEFORE its later
        # input. The second can claim product1 without that access, allowing
        # publication2 and actual release of the producer's source interval.
        first = Sequence(
            "first",
            (
                Block("view", (Call("evaluate", evaluate, uses=(Use("model", view="evaluation"),)),)),
                Block(
                    "take-two", (Join("two", ("input",), (Output("two", str),), lambda *args: True), Call("record-two", record, ("two",)))
                ),
            ),
        )
        second = Block("second", (Join("one", ("input",), (Output("one", str),), lambda *args: True), Call("record-one", record, ("one",))))
        filing = Run(
            Together(
                "run",
                (
                    Producer("source", ("input",), produce, (Use("model"),), stop_when="consumers", remaining="discard"),
                    Together(
                        "consumers",
                        (
                            Repeat("conflicting", (first,), selection("first", 2), "first-policy"),
                            Repeat("compatible", (second,), selection("second", 1), "second-policy"),
                        ),
                    ),
                ),
            ),
            (scale("model", 1.0),),
            states=(OwnedState("first-policy", lambda: None), OwnedState("second-policy", lambda: None)),
            channels=(Channel("input", 1),),
            views=(View("evaluation", "model", evaluation=True),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, access=access).publish()
        await Engine(ready).run()
        assert observed == ["one", "two"]
        assert ready.state.completed and ready.state.feeds["input"].high_water == 1
        assert not ready.state.progress.waits and not ready.state.progress.leases and not ready.state.progress.demands

    run(scenario())
