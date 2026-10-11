"""Real host-thread work with deterministic lifetime/interaction boundaries."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import queue
import threading
from dataclasses import dataclass
from typing import cast

import pytest
import torch

from .examples import Scale, adversarial, diffusion, joint, scale
from .graph import (
    Advance,
    Block,
    Call,
    Channel,
    Contract,
    Differentiate,
    Join,
    NotReady,
    Output,
    OwnedState,
    Participant,
    Producer,
    Rejected,
    Repeat,
    Requested,
    Run,
    Sequence,
    Together,
    Unit,
    Use,
    View,
)
from .prepare import Context, Engine, ProviderContext, WorkFailed, prepare
from .state import Product, Replacement, RunState
from .test_specialized_lowering import LoweringKind, observable
from .workers import CpuContext, NotEntered, PhysicalFailed, Workers


def run(awaitable):
    async def bounded():
        async with asyncio.timeout(8):
            return await awaitable

    return asyncio.run(bounded())


def calls(scope):
    if isinstance(scope, Block):
        return [item.name for item in scope.work if isinstance(item, Call) and not inspect.iscoroutinefunction(item.implementation)]
    if isinstance(scope, Together):
        return [name for child in scope.children for name in calls(child)]
    if isinstance(scope, Repeat):
        return [name for block in scope.blocks for name in calls(block)]
    if isinstance(scope, Requested) and scope.enabled:
        return calls(scope.block)
    return []


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("example", (diffusion, adversarial, joint))
def test_same_accepted_run_preserves_numerics_owners_and_unit_outcomes(example, lowering: LoweringKind):
    accepted = Contract().accept(example())
    direct = prepare(accepted, lowering=lowering).publish()
    worker = prepare(accepted, lowering=lowering, worker_calls=tuple(calls(accepted.root)), worker_limit=2).publish()
    run(Engine(direct).run())
    run(Engine(worker).run())
    assert observable(worker) == observable(direct)
    assert worker.state.accepted.root is direct.state.accepted.root
    assert any(event == "worker_completed" for event, _ in worker.state.trace)


@dataclass
class Interaction:
    commands: queue.Queue[float]
    entered: asyncio.Event
    acknowledged: asyncio.Event


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_live_handoff_interaction_and_production_before_worker_finishes(lowering: LoweringKind):
    async def scenario():
        loop, coordinator = asyncio.get_running_loop(), threading.get_ident()
        started, feedback, progressed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        release = threading.Event()
        worker_threads = []

        async def source(context: ProviderContext):
            # The selected provider owns this bounded thread-safe interaction
            # endpoint. It crosses an explicit accepted handoff, not a hidden
            # queue read substituted for the independent activity itself.
            endpoint = Interaction(queue.Queue(maxsize=1), started, feedback)
            await context.publish("input", Product("", endpoint))
            await started.wait()
            endpoint.commands.put_nowait(3.0)
            await feedback.wait()
            progressed.set()

        def compute(context: Context, endpoint: Interaction):
            worker_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(endpoint.entered.set)
            factor = endpoint.commands.get(timeout=4)
            loss = context.models["model"](torch.tensor(2.0)).square() * factor
            loop.call_soon_threadsafe(endpoint.acknowledged.set)
            assert release.wait(4), "test failed to release worker"
            return loss

        class CoordinatorState(RunState):
            def returned_writes(self, uses):
                assert threading.get_ident() == coordinator
                return super().returned_writes(uses)

            def contribute(self, *args, **kwargs):
                assert threading.get_ident() == coordinator
                return super().contribute(*args, **kwargs)

            def advance(self, *args, **kwargs):
                assert threading.get_ident() == coordinator
                return super().advance(*args, **kwargs)

        block = Block(
            "learn",
            (
                Join("receive", ("input",), (Output("endpoint", Interaction),), lambda *_: True),
                Call("compute", compute, ("endpoint",), (Output("loss", torch.Tensor),), (Use("model"),)),
                Differentiate("backward", "loss", ("unit",), "gradient"),
                Advance("update", "gradient", ("unit",)),
            ),
        )
        accepted = Contract().accept(
            Run(
                Together("run", (Producer("source", ("input",), source), block)),
                (scale("model", 1.0),),
                (Unit("unit", (("model", "weight"),)),),
                channels=(Channel("input", 1),),
            )
        )
        state = CoordinatorState(accepted)
        ready = prepare(accepted, state, lowering=lowering, worker_calls=("compute",)).publish()
        task = asyncio.create_task(Engine(ready).run())
        conflicting_entered = asyncio.Event()

        async def conflict():
            async with state.acquire((Use("model", "write"),)):
                conflicting_entered.set()

        other = None
        try:
            await progressed.wait()
            assert not task.done() and len(worker_threads) == 1
            assert worker_threads[0] != coordinator
            assert state.writers == {"model"}
            other = asyncio.create_task(conflict())
            await asyncio.sleep(0)
            assert not conflicting_entered.is_set()
            change = asyncio.create_task(state.replace(Replacement("model", state.stamp("model").binding, Scale(7.0)), "model"))
            await asyncio.sleep(0)
            assert not change.done()
            change.cancel()
            await asyncio.gather(change, return_exceptions=True)
            release.set()
            await task
            await other
            assert state.image is not None
            assert cast(Scale, state.image.bindings["model"].model).weight.item() == pytest.approx(-0.2)
            assert state.image.units["unit"].steps == 1
            assert conflicting_entered.is_set() and state.completed
        finally:
            release.set()
            await asyncio.gather(task, *((other,) if other is not None else ()), return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("late_failure", (False, True))
def test_repeated_cancellation_drains_effects_and_preserves_late_failure(lowering: LoweringKind, late_failure: bool):
    async def scenario():
        loop = asyncio.get_running_loop()
        entered, stopping = asyncio.Event(), asyncio.Event()
        release = threading.Event()

        def compute(context: Context):
            with torch.no_grad():
                cast(Scale, context.models["model"]).weight.add_(1.0)
            loop.call_soon_threadsafe(entered.set)
            assert context.stop_requested is not None
            # A supported stop request can be observed before return. It is
            # not permission for the coordinator to forcibly terminate work.
            for _ in range(800):
                if context.stop_requested():
                    break
                release.wait(0.005)
            else:
                raise AssertionError("stop was not delivered")
            loop.call_soon_threadsafe(stopping.set)
            assert release.wait(4)
            if late_failure:
                raise ValueError("failure after interruption and effects")

        accepted = Contract().accept(Run(Block("work", (Call("compute", compute, uses=(Use("model", "write"),)),)), (scale("model", 1.0),)))
        ready = prepare(accepted, lowering=lowering, worker_calls=("compute",)).publish()
        task = asyncio.create_task(Engine(ready).run())
        try:
            await entered.wait()
            task.cancel()
            await stopping.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and ready.state.running
            assert ready.state.writers == {"model"}
            change = asyncio.create_task(ready.state.replace(Replacement("model", ready.state.stamp("model").binding, Scale(9.0)), "model"))
            await asyncio.sleep(0)
            assert not change.done()
            change.cancel()
            await asyncio.gather(change, return_exceptions=True)
            release.set()
            if late_failure:
                with pytest.raises(WorkFailed) as caught:
                    await task
                assert isinstance(caught.value.cause, ValueError)
                assert caught.value.cancellation is not None
                assert not caught.value.implementation_returned
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert ("implementation_returned", ("compute", 1)) in ready.state.trace
            assert not ready.state.running and not ready.state.writers
            assert ready.state.unusable == {"model"} and not ready.state.completed
            assert ready.state.image is not None
            assert cast(Scale, ready.state.image.bindings["model"].model).weight.item() == 2.0  # No rollback.
            with pytest.raises(NotReady):
                async with ready.state.acquire((Use("model"),)):
                    pytest.fail("uncertain state was reused")
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_capacity_wait_cancelled_without_selected_effects(lowering: LoweringKind):
    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        second_calls = []

        def first(_):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(4)

        def second(_):
            second_calls.append(True)

        accepted = Contract().accept(
            Run(
                Together(
                    "run",
                    (
                        Block("first", (Call("hold", first, uses=(Use("first", "write"),)),)),
                        Block("second", (Call("queued", second, uses=(Use("second", "write"),)),)),
                    ),
                ),
                (scale("first", 1.0), scale("second", 2.0)),
            )
        )
        ready = prepare(accepted, lowering=lowering, worker_calls=("hold", "queued"), worker_limit=1).publish()
        task = asyncio.create_task(Engine(ready).run())
        try:
            await entered.wait()
            assert ready.state.writers == {"first", "second"}
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not second_calls and not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert ready.state.unusable == {"first"}
            assert any(
                event == "worker_not_started" and isinstance(evidence, tuple) and evidence[0] == "queued"
                for event, evidence in ready.state.trace
            )
            async with ready.state.acquire((Use("second"),)):
                pass
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


def test_physical_future_cancelled_before_start_never_invokes():
    async def scenario():
        workers = Workers(1)
        release, entered = threading.Event(), threading.Event()
        selected = []
        with workers.lifetime():
            assert workers.executor is not None

            def occupy():
                entered.set()
                assert release.wait(4)

            blocker = workers.executor.submit(occupy)  # Deterministic physical queue race fixture.
            assert await asyncio.to_thread(entered.wait, 4)
            job = asyncio.create_task(workers.execute(lambda: selected.append(True), threading.Event()))
            try:
                await asyncio.sleep(0)  # Submitted behind the occupied physical thread.
                job.cancel()
                with pytest.raises(NotEntered) as caught:
                    await job
                assert isinstance(caught.value.cause, asyncio.CancelledError)
                assert not selected and workers.active == 0
            finally:
                release.set()
                await asyncio.to_thread(blocker.result)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("numerical_mode", ("grad", "no_grad", "inference", "autocast"))
def test_python_and_supported_torch_context_correspondence(lowering: LoweringKind, numerical_mode):
    token = contextvars.ContextVar("selected-context", default="absent")

    def inspect_context(context: Context):
        observations = cast(list, context.states["observations"])
        model = context.models["matrix"]
        output = model(torch.ones(2, 2))
        observations.append(
            (
                token.get(),
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
                torch.is_autocast_enabled("cpu"),
                output.dtype,
                output.requires_grad,
            )
        )
        return output.detach().clone()

    def load():
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.5)
        return model

    accepted = Contract().accept(
        Run(
            Block(
                "work",
                (
                    Call(
                        "inspect",
                        inspect_context,
                        outputs=(Output("value", torch.Tensor),),
                        uses=(Use("matrix"),),
                        states=("observations",),
                    ),
                ),
            ),
            (Participant("matrix", load, lambda _: None),),
            states=(OwnedState("observations", list),),
        )
    )
    direct = prepare(accepted, lowering=lowering).publish()
    threaded = prepare(accepted, lowering=lowering, worker_calls=("inspect",)).publish()
    context = {
        "grad": torch.enable_grad,
        "no_grad": torch.no_grad,
        "inference": torch.inference_mode,
        "autocast": lambda: torch.autocast("cpu", dtype=torch.bfloat16),
    }[numerical_mode]
    previous = token.set("accepted-runtime-context")
    try:
        with context():
            run(Engine(direct).run())
            run(Engine(threaded).run())
        assert threaded.state.owners == direct.state.owners
        assert cast(list, threaded.state.owners["observations"])[0][0] == "accepted-runtime-context"
    finally:
        token.reset(previous)


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_eval_view_excludes_other_use_until_mode_restoration(lowering: LoweringKind):
    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()

        class Blocking(Scale):
            def forward(self, value):
                assert not self.training
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(4)
                return super().forward(value)

        def compute(context: Context):
            return context.views["evaluation"](torch.tensor(2.0))

        accepted = Contract().accept(
            Run(
                Block(
                    "work", (Call("compute", compute, outputs=(Output("value", torch.Tensor),), uses=(Use("model", view="evaluation"),)),)
                ),
                (Participant("model", lambda: Blocking(1.0), lambda _: None),),
                views=(View("evaluation", "model", gradients=False, evaluation=True),),
            )
        )
        ready = prepare(accepted, lowering=lowering, worker_calls=("compute",)).publish()
        stamp = ready.state.stamp("model")
        task = asyncio.create_task(Engine(ready).run())
        other_entered = asyncio.Event()

        async def read():
            async with ready.state.acquire((Use("model"),)) as projection:
                assert projection.models["model"].training
                other_entered.set()

        other = None
        try:
            await entered.wait()
            other = asyncio.create_task(read())
            await asyncio.sleep(0)
            assert not other_entered.is_set() and ready.state.writers == {"model"}
            release.set()
            await task
            await other
            assert ready.state.stamp("model") == stamp  # Exclusion is not a numerical-write grant.
        finally:
            release.set()
            await asyncio.gather(task, *((other,) if other is not None else ()), return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_invalid_output_after_effects_survives_cleanup_failure(lowering: LoweringKind):
    class BadRestore(Scale):
        def __setattr__(self, name, value):
            if name == "training" and value and getattr(self, "break_restore", False):
                raise RuntimeError("mode cleanup failed")
            super().__setattr__(name, value)

    def compute(context: Context):
        model = cast(BadRestore, context.models["model"])
        model.break_restore = True
        return "invalid result"

    accepted = Contract().accept(
        Run(
            Block("work", (Call("compute", compute, outputs=(Output("value", torch.Tensor),), uses=(Use("model"),), evaluation=True),)),
            (Participant("model", lambda: BadRestore(1.0), lambda _: None),),
        )
    )
    ready = prepare(accepted, lowering=lowering, worker_calls=("compute",)).publish()
    with pytest.raises(WorkFailed) as caught:
        run(Engine(ready).run())
    assert isinstance(caught.value.cause, TypeError)
    assert caught.value.implementation_returned
    assert len(caught.value.cleanup_failures) == 1
    assert ready.state.unusable == {"model"}


def test_worker_placement_rejects_unestablished_targets_before_loading():
    loads = []

    def load():
        loads.append(True)
        return Scale(1.0)

    async def asynchronous(_):
        pass

    participant = Participant("model", load, lambda _: None)
    accepted = Contract().accept(Run(Block("work", (Call("async", asynchronous),)), (participant,)))
    for names in (("unknown",), ("async",)):
        with pytest.raises(NotReady):
            prepare(accepted, worker_calls=names)
    with pytest.raises(ValueError):
        prepare(accepted, worker_calls=("async",), worker_limit=0)
    assert not loads
    sequential = Contract().accept(Run(Sequence("sequence", (Block("work", (Call("call", lambda _: None),)),)), (participant,)))
    with pytest.raises(NotReady, match="retained lifetimes"):
        prepare(sequential, worker_calls=("call",))
    assert not loads


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_sync_declared_call_cannot_smuggle_async_work_into_worker(lowering: LoweringKind):
    async def hidden():
        pytest.fail("unaccepted async continuation ran")

    def disguised(_):
        return hidden()

    accepted = Contract().accept(Run(Block("work", (Call("call", disguised),))))
    ready = prepare(accepted, lowering=lowering, worker_calls=("call",)).publish()
    with pytest.raises(WorkFailed) as caught:
        run(Engine(ready).run())
    assert isinstance(caught.value.cause, TypeError)
    assert not caught.value.implementation_returned


def test_worker_context_does_not_leak_between_reused_jobs():
    token = contextvars.ContextVar("worker-private-context", default="caller")

    async def scenario():
        workers = Workers(1)
        identities = []

        def contaminate():
            identities.append(threading.get_ident())
            token.set("private")
            torch.set_grad_enabled(False)
            torch.set_autocast_enabled("cpu", True)
            torch.set_autocast_dtype("cpu", torch.bfloat16)

        def inspect_next():
            identities.append(threading.get_ident())
            return token.get(), torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")

        with workers.lifetime():
            expected = token.get(), torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")
            await workers.execute(contaminate, threading.Event())
            following = await workers.execute(inspect_next, threading.Event())
            assert following.value == expected
            assert (token.get(), torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")) == expected
            assert identities[0] == identities[1] != threading.get_ident()

    run(scenario())


def test_physical_job_raising_cancelled_error_is_observed_not_drained_forever():
    async def scenario():
        workers = Workers(1)

        def interrupted():
            raise asyncio.CancelledError("from physical work")

        with workers.lifetime():
            with pytest.raises(PhysicalFailed) as caught:
                await workers.execute(interrupted, threading.Event())
            assert isinstance(caught.value.cause, asyncio.CancelledError)
            assert workers.active == 0

    run(scenario())


def test_non_cpu_rebinding_is_rejected_before_worker_entry():
    called = []
    accepted = Contract().accept(
        Run(Block("work", (Call("compute", lambda _: called.append(True), uses=(Use("model"),)),)), (scale("model", 1.0),))
    )
    ready = prepare(accepted, worker_calls=("compute",)).publish()

    async def scenario():
        stamp = ready.state.stamp("model")
        # Existing candidate publication already enforces the CPU target.
        # Worker placement must not bypass it or add a duplicate per-call scan.
        with pytest.raises(NotReady, match="CPU tensors"):
            await ready.state.replace(Replacement("model", stamp.binding, Scale(2.0).to("meta")), "model")
        assert not called and not ready.state.unusable and ready.state.stamp("model") == stamp
        await Engine(ready).run()
        assert called == [True]

    run(scenario())


def test_mode_exclusion_cannot_silently_introduce_source_wait_cycle():
    async def source(context: ProviderContext):
        async with context.use_source():
            await context.publish("input", Product("", torch.tensor(1.0)))

    def evaluate(context: Context):
        context.views["evaluation"](torch.tensor(1.0))

    accepted = Contract().accept(
        Run(
            Together(
                "run",
                (
                    Producer("source", ("input",), source, uses=(Use("model"),)),
                    Block(
                        "work",
                        (
                            Call("evaluate", evaluate, uses=(Use("model", view="evaluation"),)),
                            Join("late_input", ("input",), (Output("value", torch.Tensor),), lambda *_: True),
                        ),
                    ),
                ),
            ),
            (scale("model", 1.0),),
            channels=(Channel("input", 1),),
            views=(View("evaluation", "model", evaluation=True),),
        )
    )
    # Cooperative synchronous evaluation fits the existing read reservation.
    # Threaded mode exclusion under this target's whole-block lease would
    # prevent the producer supplying its later input. Reject that realization.
    prepare(accepted)
    with pytest.raises(Rejected, match="lease/source wait cycle"):
        prepare(accepted, worker_calls=("evaluate",))


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_outer_context_failure_preserves_interruption_and_selected_return(monkeypatch, lowering: LoweringKind):
    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        original = CpuContext.invoke

        def fail_on_exit(self, implementation):
            original(self, implementation)
            raise RuntimeError("outer numeric context exit failed")

        monkeypatch.setattr(CpuContext, "invoke", fail_on_exit)

        def selected(context: Context):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(4)
            return torch.tensor(1.0)

        accepted = Contract().accept(
            Run(
                Block("work", (Call("compute", selected, outputs=(Output("value", torch.Tensor),), uses=(Use("model", "write"),)),)),
                (scale("model", 1.0),),
            )
        )
        ready = prepare(accepted, lowering=lowering, worker_calls=("compute",)).publish()
        before = ready.state.stamp("model")
        task = asyncio.create_task(Engine(ready).run())
        try:
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(WorkFailed) as caught:
                await task
            assert isinstance(caught.value.cause, RuntimeError)
            assert caught.value.cancellation is not None
            assert caught.value.implementation_returned
            assert ready.state.stamp("model").numerical_epoch == before.numerical_epoch + 1
            assert ready.state.unusable == {"model"}
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


def test_lifetime_invariant_preserves_primary_failure_without_assert():
    workers = Workers(1)
    with pytest.raises(ExceptionGroup) as caught, workers.lifetime():
        workers.active = 1  # Invalid-root fixture; no physical work is detached.
        raise ValueError("primary run failure")
    assert [type(error) for error in caught.value.exceptions] == [ValueError, RuntimeError]
    assert workers.executor is None and workers.slots is None


def test_worker_limit_is_checked_even_without_placement():
    accepted = Contract().accept(Run(Block("work", ())))
    for limit in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            prepare(accepted, worker_limit=limit)


def test_matching_call_names_in_distinct_blocks_place_both():
    async def scenario():
        coordinator = threading.get_ident()
        observed: queue.Queue[tuple[str, int]] = queue.Queue()

        def first(_):
            observed.put(("first", threading.get_ident()))

        def second(_):
            observed.put(("second", threading.get_ident()))

        accepted = Contract().accept(
            Run(Together("run", (Block("first", (Call("common", first),)), Block("second", (Call("common", second),)))))
        )
        await Engine(prepare(accepted, worker_calls=("common",), worker_limit=2).publish()).run()
        results = observed.get_nowait(), observed.get_nowait()
        assert {name for name, _ in results} == {"first", "second"}
        assert all(identity != coordinator for _, identity in results)

    run(scenario())


def test_mode_check_adds_no_synthetic_work_name_collision():
    accepted = Contract().accept(
        Run(
            Block("work", (Call("<physical-modes>", lambda _: None, uses=(Use("model", view="eval"),)),)),
            (scale("model", 1.0),),
            views=(View("eval", "model", evaluation=True),),
        )
    )
    run(Engine(prepare(accepted, worker_calls=("<physical-modes>",)).publish()).run())
