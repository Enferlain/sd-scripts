"""Private provider threads share current-use protection with selected work."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
import torch

from .examples import Scale, scale
from .graph import Block, Call, Channel, Contract, Join, NotReady, Output, Producer, Release, Retain, Run, Sequence, Together, Use, View
from .prepare import Engine, prepare
from .state import Product, Replacement
from .test_candidate import leaves
from .test_workers import run


class PrivateEncoder:
    """Test provider owns its executor and joins it inside its source lease."""

    def __init__(self, *, fail=False):
        self.entered = asyncio.Event()
        self.interrupted = asyncio.Event()
        self.inspect = threading.Event()
        self.finished = threading.Event()
        self.modes = []
        self.fail = fail

    async def drain(self, pending):
        cancelled = None
        while True:
            try:
                await asyncio.shield(pending)
                break
            except asyncio.CancelledError as error:
                self.interrupted.set()
                cancelled = cancelled or error
                if pending.done():
                    pending.result()  # Observe a physical failure named CancelledError too.
        if cancelled is not None:
            raise cancelled

    async def __call__(self, context):
        loop = asyncio.get_running_loop()
        async with context.use_source() as projection:

            def encode():
                loop.call_soon_threadsafe(self.entered.set)
                try:
                    if not self.inspect.wait(5):
                        raise TimeoutError("Test did not release private encoding work")
                    self.modes.append(projection.models["model"].training)
                    if self.fail:
                        raise ValueError("Private encoding failed")
                finally:
                    self.finished.set()

            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="private-encoder") as executor:
                pending = loop.run_in_executor(executor, encode)
                try:
                    await self.entered.wait()
                    await context.publish("ready", Product("", True))
                    await self.drain(pending)
                except asyncio.CancelledError:
                    self.interrupted.set()
                    raise
                finally:
                    await self.drain(pending)


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("worker", (False, True))
@pytest.mark.parametrize("per_work", (False, True))
@pytest.mark.parametrize("target", ("shared-evaluation", "shared-read", "other-evaluation"))
def test_private_provider_access_serializes_only_conflicting_views(lowering, worker, per_work, target, monkeypatch):
    async def scenario():
        provider, attempted = PrivateEncoder(), asyncio.Event()
        loop, view_started = asyncio.get_running_loop(), asyncio.Event()
        view_entered, view_modes = threading.Event(), []
        participant = "other" if target == "other-evaluation" else "model"
        evaluation = target != "shared-read"
        reservations = []
        overlaps = []

        class ModeProbe(Scale):
            def forward(self, value):
                view_entered.set()
                view_modes.append(self.training)
                overlaps.append(not provider.finished.is_set())
                loop.call_soon_threadsafe(view_started.set)
                # If protection is missing, force the provider's read to occur
                # while this view has changed the shared mode, before restore.
                if not provider.finished.is_set():
                    provider.inspect.set()
                    assert provider.finished.wait(5)
                return super().forward(value)

        def observe(context, _ready):
            context.views["evaluation"](torch.tensor(1.0))

        body = Block(
            "body",
            (
                Join("receive", ("ready",), (Output("available", bool),), lambda *_: True),
                Call("observe", observe, ("available",), uses=(Use(participant, view="evaluation"),)),
            ),
        )
        filing = Run(
            Together(
                "run",
                (
                    Producer("encoder", ("ready",), provider, (Use("model"),)),
                    Sequence("selected", (body,)) if per_work else body,
                ),
            ),
            tuple(replace(scale(name, 1.0), load=lambda: ModeProbe(1.0)) for name in ("model", "other")),
            channels=(Channel("ready", 1),),
            views=(View("evaluation", participant, evaluation=evaluation),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("observe",) if worker else ()).publish()
        original = ready.state.use

        @asynccontextmanager
        async def traced(uses, *args, **kwargs):
            if any(use.view == "evaluation" for use in uses):
                attempted.set()
            async with original(uses, *args, **kwargs) as projection:
                if any(use.view == "evaluation" for use in uses):
                    reservations.append((dict(ready.state.readers), set(ready.state.writers)))
                yield projection

        monkeypatch.setattr(ready.state, "use", traced)
        task = asyncio.create_task(Engine(ready).run())
        try:
            await attempted.wait()
            if target == "shared-evaluation":
                assert not view_entered.is_set()
            else:
                await view_started.wait()
            provider.inspect.set()
            await task
            assert provider.modes == [True], "Provider observed another owner's temporary model mode"
            assert overlaps == [target != "shared-evaluation"] and view_modes == [not evaluation]
            if target == "shared-evaluation":
                assert reservations == [({}, {"model"})]
            elif target == "shared-read":
                assert reservations == [({"model": 2}, set())]
            else:
                expected = [({"model": 1}, {"other"})] if worker else [({"model": 1, "other": 1}, set())]
                assert reservations == expected
            assert ready.state.completed and ready.state.stamp("model").numerical_epoch == 0
            assert not ready.state.readers and not ready.state.writers and not ready.state.active_producers
            assert ready.state.image is not None and all(binding.model.training for binding in ready.state.image.bindings.values())
        finally:
            provider.inspect.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("worker", (False, True))
@pytest.mark.parametrize("isolated", (False, True))
def test_retained_source_uses_the_same_mode_policy_and_isolated_copies_stay_independent(lowering, worker, isolated, monkeypatch):
    async def scenario():
        provider, attempted = PrivateEncoder(), asyncio.Event()
        loop, view_started = asyncio.get_running_loop(), asyncio.Event()
        overlaps, views = [], []

        class ModeProbe(Scale):
            def forward(self, value):
                overlaps.append(not provider.finished.is_set())
                loop.call_soon_threadsafe(view_started.set)
                if not provider.finished.is_set():
                    provider.inspect.set()
                    assert provider.finished.wait(5)
                return super().forward(value)

        def observe(context):
            views.append(context.views["evaluation"].model)
            context.views["evaluation"](torch.tensor(1.0))

        filing = Run(
            Together(
                "run",
                (
                    Producer("encoder", ("ready",), provider, (Use("model"),)),
                    Sequence(
                        "selected",
                        (
                            Block(
                                "body",
                                (
                                    Join("receive", ("ready",), (Output("available", bool),), lambda *_: True),
                                    Retain("hold", ("model",), (), "old", isolated=isolated, after=("receive",)),
                                    Call("observe", observe, uses=(Use("model", view="evaluation"),), retained="old", after=("hold",)),
                                    Release("release", "old", after=("observe",)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            (replace(scale("model", 1.0), load=lambda: ModeProbe(1.0)),),
            channels=(Channel("ready", 1),),
            views=(View("evaluation", "model", evaluation=True),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("observe",) if worker else ()).publish()
        original = ready.state.retain

        async def traced(*args, **kwargs):
            attempted.set()
            return await original(*args, **kwargs)

        monkeypatch.setattr(ready.state, "retain", traced)
        task = asyncio.create_task(Engine(ready).run())
        try:
            await attempted.wait()
            if isolated:
                await view_started.wait()
            else:
                assert not view_started.is_set() and ready.state.readers == {"model": 1} and not ready.state.writers
            provider.inspect.set()
            await task
            assert overlaps == [isolated] and provider.modes == [True]
            assert ready.state.image is not None
            current = ready.state.image.bindings["model"].model
            assert (views[0] is current) is not isolated and current.training
            assert ready.state.completed and ready.state.stamp("model").numerical_epoch == 0
            assert not ready.state.readers and not ready.state.writers and not ready.state.active_retentions
        finally:
            provider.inspect.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("termination", ("normal-stop", "failed-stop", "cancel"))
def test_private_worker_stop_failure_and_repeated_cancellation_hold_access_until_physical_return(lowering, termination):
    async def scenario():
        provider = PrivateEncoder(fail=termination == "failed-stop")
        source = Producer(
            "encoder",
            ("ready",),
            provider,
            (Use("model"),),
            stop_when="demand" if termination != "cancel" else None,
            remaining="discard" if termination != "cancel" else None,
        )
        demand = Block("demand", (Join("receive", ("ready",), (Output("available", bool),), lambda *_: True),))
        filing = Run(
            source if termination == "cancel" else Together("run", (source, demand)),
            (scale("model", 1.0),),
            channels=(Channel("ready", 1),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering).publish()
        task = asyncio.create_task(Engine(ready).run())
        conflicting = None
        entered, attempted = [], asyncio.Event()

        async def publication():
            attempted.set()
            await ready.state.replace(Replacement("model", 1, Scale(2.0)), "model")
            entered.append(True)

        async def writer():
            attempted.set()
            async with ready.state.use((Use("model", "write"),)):
                entered.append(True)

        try:
            await provider.entered.wait()
            if termination == "cancel":
                task.cancel()
            await provider.interrupted.wait()
            if termination == "cancel":
                task.cancel()  # A second parent stop cannot detach the worker.
            assert not provider.finished.is_set() and not task.done()
            assert ready.state.readers == {"model": 1} and "encoder" in ready.state.active_producers
            conflicting = asyncio.create_task(publication() if termination == "normal-stop" else writer())
            await attempted.wait()
            assert not conflicting.done() and not entered and ready.state.generation == 1
            provider.inspect.set()
            if termination == "normal-stop":
                await task
                await conflicting
                assert ready.state.completed and ready.state.generation == 2 and not ready.state.unusable
                assert entered == [True] and ready.state.capture().generation == 2
            else:
                if termination == "failed-stop":
                    with pytest.raises(ExceptionGroup) as caught:
                        await task
                    assert any(isinstance(error, ValueError) and str(error) == "Private encoding failed" for error in leaves(caught.value))
                else:
                    with pytest.raises(asyncio.CancelledError):
                        await task
                with pytest.raises(NotReady, match="unavailable"):
                    await conflicting
                assert not ready.state.completed and not entered and ready.state.unusable == {"model"}
                assert ready.state.generation == 1
            assert provider.finished.is_set() and provider.modes == [True]
            assert not ready.state.readers and not ready.state.writers and not ready.state.active_producers
            assert ready.state.stamp("model").numerical_epoch == 0
        finally:
            provider.inspect.set()
            task.cancel()
            if conflicting is not None:
                conflicting.cancel()
            await asyncio.gather(task, *((conflicting,) if conflicting is not None else ()), return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_whole_call_evaluation_keeps_exclusion_even_without_external_workers(lowering):
    async def scenario():
        entered, finish, attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
        observed = []

        async def evaluate(context):
            assert not context.models["model"].training
            assert ready.state.writers == {"model"} and not ready.state.readers
            entered.set()
            await finish.wait()

        filing = Run(
            Sequence(
                "selected",
                (
                    Block(
                        "body",
                        (Call("evaluate", evaluate, uses=(Use("model"),), evaluation=True),),
                    ),
                ),
            ),
            (scale("model", 1.0),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering).publish()

        async def read():
            attempted.set()
            async with ready.state.use((Use("model"),)) as projection:
                observed.append(projection.models["model"].training)

        task, reader = asyncio.create_task(Engine(ready).run()), None
        try:
            await entered.wait()
            reader = asyncio.create_task(read())
            await attempted.wait()
            assert not reader.done() and not observed
            finish.set()
            await task
            await reader
            assert observed == [True] and ready.state.completed and not ready.state.active_retentions
            assert not ready.state.readers and not ready.state.writers
        finally:
            finish.set()
            task.cancel()
            if reader is not None:
                reader.cancel()
            await asyncio.gather(task, *((reader,) if reader is not None else ()), return_exceptions=True)

    run(scenario())
