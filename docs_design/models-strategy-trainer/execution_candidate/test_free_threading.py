"""Actual worker overlap, protected owner state and backend context isolation."""

from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest
import torch

from .graph import Block, Call, Choice, Contract, Output, OwnedState, Participant, Repeat, Run, Together, Unit, Use
from .prepare import Context, Engine, prepare
from .state import RunState
from .test_specialized_lowering import LoweringKind
from .test_workers import run
from .workers import PhysicalFailed, Workers


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("first_dtype", (torch.bfloat16, torch.float16))
def test_overlapping_reads_preserve_distinct_autocast_contexts(lowering: LoweringKind, first_dtype: torch.dtype):
    """A shared native weight cache must not confuse two accepted read views."""
    second_dtype = torch.float16 if first_dtype == torch.bfloat16 else torch.bfloat16
    token = contextvars.ContextVar("worker-invocation", default="coordinator")

    async def scenario():
        asyncio.get_running_loop().set_debug(True)
        filled, release = threading.Event(), threading.Event()
        returned = queue.SimpleQueue()

        def implementation(name, dtype):
            def compute(context: Context):
                assert token.get() == "coordinator"
                token.set(name)
                try:
                    if name == "second":
                        assert filled.wait(5), "first worker did not compute"
                    with torch.autocast("cpu", dtype=dtype):
                        result = context.models["matrix"](torch.ones(2, 2))
                        returned.put((name, result, threading.get_ident(), token.get(), torch.is_autocast_cache_enabled()))
                        if name == "first":
                            filled.set()
                            assert release.wait(5), "second worker did not finish"
                        assert token.get() == name
                    return result
                finally:
                    # Failures must release the other real worker too.
                    (release if name == "second" else filled).set()

            return compute

        def load():
            model = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                model.weight.fill_(0.5)
            return model

        accepted = Contract().accept(
            Run(
                Together(
                    "run",
                    tuple(
                        Block(
                            name,
                            (Call(name, implementation(name, dtype), outputs=(Output("value", torch.Tensor),), uses=(Use("matrix"),)),),
                        )
                        for name, dtype in (("first", first_dtype), ("second", second_dtype))
                    ),
                ),
                (Participant("matrix", load, lambda _: None),),
                # Keep the source parameter trainable so autocast's weight
                # cache is eligible. These activities request no advancement.
                (Unit("weights", (("matrix", "weight"),)),),
            )
        )
        ready = prepare(accepted, lowering=lowering, worker_calls=("first", "second"), worker_limit=2).publish()
        coordinator = threading.get_ident()
        try:
            await Engine(ready).run()
        finally:
            release.set()
        assert ready.state.image is not None
        snapshots = {name: values for name, *values in (returned.get_nowait(), returned.get_nowait())}
        parameter = ready.state.image.bindings["matrix"].model.get_parameter("weight")
        for name, dtype in (("first", first_dtype), ("second", second_dtype)):
            result, thread, observed, cache = snapshots[name]
            assert thread != coordinator and observed == name and not cache
            assert result.dtype == dtype and torch.equal(result, torch.ones(2, 2, dtype=dtype))
            gradient = torch.autograd.grad(result.sum(), parameter)[0]
            assert torch.equal(gradient, torch.full_like(parameter, 2))
        # This host-pool fixture requires two physical threads, with the first
        # call still live while the second computes. This is not a language rule.
        assert snapshots["first"][1] != snapshots["second"][1]
        assert token.get() == "coordinator" and ready.state.completed
        assert not ready.state.progress.leases and not ready.state.progress.waits

    run(scenario())


@pytest.mark.parametrize("caller_cache", (False, True))
@pytest.mark.parametrize("fails", (False, True))
def test_worker_cache_policy_preserves_caller_and_reused_worker(caller_cache, fails):
    async def scenario():
        workers = Workers(1)
        thread = None

        def first():
            nonlocal thread
            thread = threading.get_ident()
            assert not torch.is_autocast_cache_enabled()
            torch.set_autocast_cache_enabled(True)
            if fails:
                raise ValueError("selected failure after context mutation")

        def following():
            assert threading.get_ident() == thread
            return torch.is_autocast_cache_enabled(), torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")

        previous = torch.is_autocast_cache_enabled()
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=caller_cache), workers.lifetime():
            if fails:
                with pytest.raises(PhysicalFailed) as caught:
                    await workers.execute(first, threading.Event())
                assert isinstance(caught.value.cause, ValueError)
            else:
                await workers.execute(first, threading.Event())
            assert torch.is_autocast_cache_enabled() is caller_cache
            result = await workers.execute(following, threading.Event())
            assert result.value == (False, True, torch.bfloat16)
            assert torch.is_autocast_cache_enabled() is caller_cache
        assert torch.is_autocast_cache_enabled() is previous

    run(scenario())


@dataclass
class Counter:
    value: int = 0


@dataclass
class RandomState:
    generator: torch.Generator
    values: list[torch.Tensor] = field(default_factory=list)


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_independently_owned_random_streams_match_serial_draws(lowering: LoweringKind):
    seeds = {"a": 17, "b": 29}
    barrier = threading.Barrier(2, timeout=5)

    def sample(context: Context):
        state = next(iter(context.states.values()))
        assert isinstance(state, RandomState)
        state.values.append(torch.rand(4, generator=state.generator))
        barrier.wait()  # Both native generator owners really enter workers.

    def policy(block):
        return lambda counts, _: Choice(block) if counts.get(block, 0) < 3 else None

    accepted = Contract().accept(
        Run(
            Together(
                "run",
                tuple(
                    Repeat(
                        name,
                        (Block(name + "-sample", (Call("draw", sample, states=(name,)),)),),
                        policy(name + "-sample"),
                        "policy-" + name,
                    )
                    for name in seeds
                ),
            ),
            states=tuple(OwnedState("policy-" + name, lambda: None) for name in seeds)
            + tuple(OwnedState(name, lambda seed=seed: RandomState(torch.Generator().manual_seed(seed))) for name, seed in seeds.items()),
        )
    )
    ready = prepare(accepted, lowering=lowering, worker_calls=("draw",), worker_limit=2).publish()
    before = torch.random.get_rng_state().clone()
    run(Engine(ready).run())
    assert torch.equal(torch.random.get_rng_state(), before)
    for name, seed in seeds.items():
        state = ready.state.owners[name]
        assert isinstance(state, RandomState)
        serial = torch.Generator().manual_seed(seed)
        assert len(state.values) == 3
        assert all(torch.equal(value, torch.rand(4, generator=serial)) for value in state.values)
        assert torch.equal(state.generator.get_state(), serial.get_state())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("shared_owner", (False, True))
def test_worker_owned_state_serializes_only_its_actual_users(lowering: LoweringKind, shared_owner):
    async def scenario():
        loop, coordinator = asyncio.get_running_loop(), threading.get_ident()
        loop.set_debug(True)
        entered, second_attempt = asyncio.Event(), asyncio.Event()
        release, second_entered = threading.Event(), threading.Event()
        controls, workers = [], queue.SimpleQueue()
        second_owner = "first" if shared_owner else "second"
        acquisitions = 0

        class ObservedState(RunState):
            @asynccontextmanager
            async def acquire(self, uses, states=()):
                nonlocal acquisitions
                controls.append(threading.get_ident())
                if states:
                    acquisitions += 1
                    if acquisitions == 2:
                        second_attempt.set()
                async with super().acquire(uses, states) as projection:
                    yield projection

            def returned_writes(self, uses):
                controls.append(threading.get_ident())
                return super().returned_writes(uses)

        def first(context: Context):
            counter = context.states["first"]
            assert isinstance(counter, Counter)
            value = counter.value
            workers.put(threading.get_ident())
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5), "test did not release first owner"
            counter.value = value + 1

        def second(context: Context):
            counter = context.states[second_owner]
            assert isinstance(counter, Counter)
            workers.put(threading.get_ident())
            second_entered.set()
            counter.value += 1
            if not shared_owner:
                release.set()

        # First entry is an explicit dependency for test staging; it does not
        # replace the framework's state lease or serialize disjoint owners.
        async def wait_first(_):
            await entered.wait()

        accepted = Contract().accept(
            Run(
                Together(
                    "run",
                    (
                        Block("a", (Call("first", first, states=("first",)),)),
                        Block("b", (Call("wait", wait_first), Call("second", second, states=(second_owner,), after=("wait",)))),
                    ),
                ),
                states=(OwnedState("first", Counter), OwnedState("second", Counter)),
            )
        )
        ready = prepare(accepted, ObservedState(accepted), lowering=lowering, worker_calls=("first", "second"), worker_limit=2).publish()
        task = asyncio.create_task(Engine(ready).run())
        try:
            await entered.wait()
            await second_attempt.wait()
            if shared_owner:
                assert not second_entered.is_set()
                assert ready.state.progress.waits
                release.set()
            await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        counts = []
        for name in ("first", "second"):
            counter = ready.state.owners[name]
            assert isinstance(counter, Counter)
            counts.append(counter.value)
        assert counts == ([2, 0] if shared_owner else [1, 1])
        assert controls and set(controls) == {coordinator}
        assert all(workers.get_nowait() != coordinator for _ in range(2))
        assert ready.state.completed and not ready.state.state_locks

    run(scenario())
