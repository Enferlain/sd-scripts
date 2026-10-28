"""Submission acknowledgement must not lose physical execution ownership."""

from __future__ import annotations

import asyncio
import threading

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pytest
import torch

from . import workers as worker_module
from .examples import scale
from .graph import Block, Call, Contract, NotReady, Output, OwnedState, Run, Together, Use
from .prepare import Context, Engine, WorkFailed, prepare
from .test_specialized_lowering import LoweringKind
from .test_workers import run
from .workers import CpuContext, NotEntered, Workers


type AccessKind = Literal["general", "prepared"]


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
def test_queued_work_cannot_enter_after_submit_failure_releases_its_use(monkeypatch, lowering: LoweringKind, access: AccessKind):
    async def scenario():
        loop = asyncio.get_running_loop()
        first_entered, submit_failed = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        effects = []
        pools: list[ThreadPoolExecutor] = []
        factory = worker_module.ThreadPoolExecutor
        start = threading.Thread.start
        failed = False

        def pool(*args, **kwargs):
            executor = factory(*args, **kwargs)
            pools.append(executor)
            return executor

        def fail_thread_start(thread):
            nonlocal failed
            if thread.name == "candidate-work_1" and not failed:
                failed = True
                loop.call_soon_threadsafe(submit_failed.set)
                raise RuntimeError("new thread unavailable after enqueue")
            return start(thread)

        def first(context):
            loop.call_soon_threadsafe(first_entered.set)
            assert release.wait(4)

        def second(context):
            effects.append("entered")
            with torch.no_grad():
                context.models["second"].get_parameter("weight").add_(7)

        monkeypatch.setattr(worker_module, "ThreadPoolExecutor", pool)
        monkeypatch.setattr(threading.Thread, "start", fail_thread_start)
        filing = Run(
            Together(
                "run",
                (
                    Block("first", (Call("hold", first, uses=(Use("first", "write"),)),)),
                    Block("second", (Call("write", second, uses=(Use("second", "write"),)),)),
                ),
            ),
            (scale("first", 1.0), scale("second", 2.0)),
        )
        ready = prepare(
            Contract().accept(filing), lowering=lowering, access=access, worker_calls=("hold", "write"), worker_limit=2
        ).publish()
        state = ready.state
        task = asyncio.create_task(Engine(ready).run())
        try:
            await first_entered.wait()
            await submit_failed.wait()
            async with state.acquire((Use("second"),)):
                # The failed submission is already in the real executor queue.
                # A barrier behind it proves it was processed before inspecting
                # state. Its replacement thread may now start successfully.
                await asyncio.wrap_future(pools[0].submit(lambda: None))
                assert effects == []
                assert state.image is not None
                assert float(state.image.bindings["second"].model.get_parameter("weight")) == 2
                assert state.stamp("second").numerical_epoch == 0
            release.set()
            with pytest.raises(ExceptionGroup):
                await task
            assert "second" not in state.unusable and not state.readers and not state.writers
            assert any(event == "worker_not_started" and isinstance(value, tuple) and value[0] == "write" for event, value in state.trace)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


def leaves(error: BaseException):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in leaves(child)]
    return [error]


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
@pytest.mark.parametrize("cancel", (False, True))
@pytest.mark.parametrize("outcome", ("return", "selected-failure", "invalid-output", "outer-failure"))
def test_entered_work_stays_owned_after_submit_error(monkeypatch, lowering: LoweringKind, access: AccessKind, cancel: bool, outcome: str):
    async def scenario():
        loop = asyncio.get_running_loop()
        entered, release = threading.Event(), threading.Event()
        submit_failed = asyncio.Event()
        submission_error = RuntimeError("submit failed after entering")
        selected_error = ValueError("selected failure after effects")
        context_error = RuntimeError("outer context exit failed")

        class StartedThenFailed(ThreadPoolExecutor):
            def submit(self, fn, /, *args, **kwargs):
                super().submit(fn, *args, **kwargs)
                assert entered.wait(4)
                loop.call_soon_threadsafe(submit_failed.set)
                raise submission_error

        monkeypatch.setattr(worker_module, "ThreadPoolExecutor", StartedThenFailed)
        if outcome == "outer-failure":
            invoke = CpuContext.invoke

            def fail_context(self, call):
                invoke(self, call)
                raise context_error

            monkeypatch.setattr(CpuContext, "invoke", fail_context)

        def selected(context: Context):
            effects = context.states["effects"]
            assert isinstance(effects, list)
            effects.append("selected")
            with torch.no_grad():
                context.models["model"].get_parameter("weight").add_(1)
            entered.set()
            assert release.wait(4)
            assert context.stop_requested is not None and context.stop_requested()
            if outcome == "selected-failure":
                raise selected_error
            return "bad output" if outcome == "invalid-output" else 1

        def dependent(context: Context, value):
            pytest.fail("Dependent work ran after a failed dispatch")

        filing = Run(
            Block(
                "work",
                (
                    Call("compute", selected, outputs=(Output("value", int),), uses=(Use("model", "write"),), states=("effects",)),
                    Call("dependent", dependent, ("value",)),
                ),
            ),
            (scale("model", 2.0),),
            states=(OwnedState("effects", list),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, access=access, worker_calls=("compute",)).publish()
        state = ready.state
        task = asyncio.create_task(Engine(ready).run())
        other = None
        reader_entered = asyncio.Event()

        async def read():
            async with state.acquire((Use("model"),)):
                reader_entered.set()

        try:
            await submit_failed.wait()
            assert not task.done() and state.writers == {"model"} and state.state_locks == {"effects"}
            other = asyncio.create_task(read())
            await asyncio.sleep(0)
            assert not reader_entered.is_set()
            if cancel:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and state.writers == {"model"}
            release.set()
            with pytest.raises(WorkFailed) as caught:
                await task
            failure = caught.value
            errors = leaves(failure.cause) + [leaf for error in failure.cleanup_failures for leaf in leaves(error)]
            assert submission_error in errors
            if outcome == "selected-failure":
                assert failure.cause is selected_error
            elif outcome == "invalid-output":
                assert isinstance(failure.cause, TypeError)
            elif outcome == "outer-failure":
                assert context_error in errors
            assert failure.implementation_returned == (outcome != "selected-failure")
            assert (failure.cancellation is not None) == cancel
            assert state.stamp("model").numerical_epoch == int(outcome != "selected-failure")
            assert state.owners["effects"] == ["selected"]
            assert state.unusable == {"model"} and state.unusable_states == {"effects"}
            assert not state.writers and not state.state_locks and not reader_entered.is_set()
            assert other is not None
            with pytest.raises(NotReady):
                await other
        finally:
            release.set()
            await asyncio.gather(task, *((other,) if other is not None else ()), return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
def test_completed_work_is_not_misclassified_when_submit_raises_after_return(monkeypatch, lowering: LoweringKind, access: AccessKind):
    failure = RuntimeError("submit failed after completion")

    class CompletedThenFailed(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            future.result(timeout=4)
            raise failure

    monkeypatch.setattr(worker_module, "ThreadPoolExecutor", CompletedThenFailed)

    def selected(context: Context):
        with torch.no_grad():
            context.models["model"].get_parameter("weight").add_(1)

    filing = Run(Block("work", (Call("compute", selected, uses=(Use("model", "write"),)),)), (scale("model", 2.0),))
    ready = prepare(Contract().accept(filing), lowering=lowering, access=access, worker_calls=("compute",)).publish()
    with pytest.raises(WorkFailed) as caught:
        run(Engine(ready).run())
    assert caught.value.cause is failure and caught.value.implementation_returned
    assert ready.state.stamp("model").numerical_epoch == 1
    assert ready.state.unusable == {"model"} and not ready.state.writers


def test_submission_rejected_before_enqueue_needs_no_worker_drain():
    async def scenario():
        selected = []
        workers = Workers(1)
        with workers.lifetime():
            assert workers.executor is not None
            workers.executor.shutdown()
            with pytest.raises(NotEntered):
                await workers.execute(lambda: selected.append(True), threading.Event())
            assert selected == [] and workers.active == 0

    run(scenario())
