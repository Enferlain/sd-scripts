"""Compose host-worker placement with the existing cross-block lifetime target."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest
import torch

from .examples import Scale, routed_views, scale
from .graph import (
    Block,
    Call,
    Channel,
    Contract,
    Join,
    NotReady,
    Output,
    OwnedState,
    Producer,
    Release,
    Retain,
    Run,
    Sequence,
    Together,
    Use,
    View,
)
from .prepare import Engine, WorkFailed, prepare
from .state import Product, Replacement, RunState
from .test_routed_views import add_work, sequence
from .test_specialized_lowering import LoweringKind, observable
from .test_workers import run
from .workers import Workers


def replace_call(filing: Run, name: str, implementation, *, states: tuple[str, ...] = ()) -> Run:
    root = sequence(filing)
    return replace(
        filing,
        root=replace(
            root,
            blocks=tuple(
                replace(
                    block,
                    work=tuple(
                        replace(item, implementation=implementation, states=states)
                        if isinstance(item, Call) and item.name == name
                        else item
                        for item in block.work
                    ),
                )
                for block in root.blocks
            ),
        ),
    )


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("isolated", (False, True))
@pytest.mark.parametrize("recompute", (False, True))
def test_same_graph_retains_routed_gradients_and_unit_outcomes_across_workers(lowering: LoweringKind, isolated, recompute):
    accepted = Contract().accept(routed_views(isolated=isolated, recompute=recompute))
    direct = prepare(accepted, lowering=lowering).publish()
    coordinator, threads, gradients = threading.get_ident(), [], {}

    class CoordinatorState(RunState):
        def contribute(self, *args, **kwargs):
            threads.append(threading.get_ident())
            result = super().contribute(*args, **kwargs)
            assert self.image is not None
            for name in result.units:
                saved = []
                for parameter in self.image.units[name].parameters:
                    assert parameter.grad is not None
                    saved.append(parameter.grad.item())
                gradients[name] = tuple(saved)
            return result

        def advance(self, *args, **kwargs):
            threads.append(threading.get_ident())
            return super().advance(*args, **kwargs)

    threaded = prepare(
        accepted,
        CoordinatorState(accepted),
        lowering=lowering,
        worker_calls=("first-objective", "second-objective"),
        worker_limit=2,
    ).publish()
    run(Engine(direct).run())
    run(Engine(threaded).run())
    assert observable(threaded) == observable(direct)
    assert gradients == {"A": (3.0,), "B": (4.0,)}
    assert threads == [coordinator] * 4
    assert threaded.state.accepted.root is direct.state.accepted.root
    assert sum(event == "worker_completed" for event, _ in threaded.state.trace) == 2
    assert not threaded.state.active_retentions and not threaded.state.readers and not threaded.state.writers
    threaded.state.capture()


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_production_and_conflicting_access_across_worker_return_and_later_derivative(lowering: LoweringKind):
    async def scenario():
        loop = asyncio.get_running_loop()
        entered, produced, between = asyncio.Event(), asyncio.Event(), asyncio.Event()
        later, finish_worker = asyncio.Event(), threading.Event()
        coordinator, worker_threads = threading.get_ident(), []
        original = sequence(routed_views()).blocks[0].work[1]
        assert isinstance(original, Call)

        def first(context):
            worker_threads.append(threading.get_ident())
            result = original.implementation(context)
            loop.call_soon_threadsafe(entered.set)
            assert finish_worker.wait(4), "worker was not released"
            return result

        async def pause(_):
            between.set()
            await later.wait()

        async def producer(context):
            await entered.wait()
            await context.publish("prefetch", Product("next", 7))
            produced.set()
            await asyncio.Event().wait()

        filing = add_work(
            replace_call(routed_views(recompute=True), "first-objective", first), 0, Call("pause", pause, after=("a-gradient",))
        )
        filing = replace(
            filing,
            root=Together("run", (Producer("source", ("prefetch",), producer, stop_when="routed-work", remaining="discard"), filing.root)),
            channels=(Channel("prefetch", 1),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("first-objective",)).publish()
        state = ready.state
        task = asyncio.create_task(Engine(ready).run())
        attempted, acquired = asyncio.Event(), asyncio.Event()

        async def reader():
            attempted.set()
            async with state.acquire((Use("frozen"),)):
                acquired.set()

        other = change = None
        try:
            await produced.wait()
            assert worker_threads and worker_threads[0] != coordinator
            assert state.feeds["prefetch"].ready["next"].value == 7
            assert state.writers == {"frozen"}  # Temporary mode exclusion, not a numerical write.
            assert state.readers == {"a": 1, "b": 1}
            other = asyncio.create_task(reader())
            change = asyncio.create_task(state.replace(Replacement("frozen", 1, Scale(4.0)), "frozen"))
            await attempted.wait()
            assert not acquired.is_set() and not change.done()
            finish_worker.set()
            await between.wait()
            assert any(event == "worker_completed" for event, _ in state.trace)
            assert not acquired.is_set() and not change.done()
            assert state.active_retentions and state.writers == {"frozen"}
            assert state.image is not None
            assert state.image.units["A"].pending == 1 and state.image.units["B"].steps == 0
            assert state.image.bindings["frozen"].model.training  # Restored, but future use still protected.
            change.cancel()
            await asyncio.gather(change, return_exceptions=True)
            later.set()
            await task
            await other
            assert acquired.is_set() and state.completed
            assert state.stamp("frozen").numerical_epoch == 0
            assert [state.image.units[name].steps for name in ("A", "B")] == [1, 1]
            assert not state.active_retentions and not state.readers and not state.writers
            assert ("demand_ended", "routed-work") in state.trace
            assert list(state.feeds["prefetch"].discarded) == [("next", ())]
        finally:
            finish_worker.set()
            later.set()
            if change is not None:
                change.cancel()
            await asyncio.gather(
                task, *((other,) if other is not None else ()), *((change,) if change is not None else ()), return_exceptions=True
            )

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("isolated", (False, True))
@pytest.mark.parametrize("name,late_failure", (("first-objective", False), ("second-objective", False), ("second-objective", True)))
def test_cancellation_drains_worker_before_retained_release_and_keeps_partial_outcomes(
    lowering: LoweringKind, isolated, name, late_failure
):
    async def scenario():
        loop, entered, finish = asyncio.get_running_loop(), asyncio.Event(), threading.Event()
        filing = routed_views(isolated=isolated)
        original = next(item for block in sequence(filing).blocks for item in block.work if isinstance(item, Call) and item.name == name)

        def interrupted(context, *args):
            result = original.implementation(context, *args)
            context.states["effects"].append("computed")
            loop.call_soon_threadsafe(entered.set)
            assert finish.wait(4), "worker was not released"
            assert context.stop_requested is not None and context.stop_requested()
            if late_failure:
                raise ValueError("late retained-worker failure")
            return result

        filing = replace_call(filing, name, interrupted, states=("effects",))
        filing = replace(filing, states=(OwnedState("effects", list),))
        ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=(name,)).publish()
        state = ready.state
        task = asyncio.create_task(Engine(ready).run())
        try:
            await entered.wait()
            retained = next(iter(state.active_retentions.values()))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not retained.released
            if not isolated:
                assert state.writers == {"frozen"} and state.readers == {"a": 1, "b": 1}
            finish.set()
            if late_failure:
                with pytest.raises(WorkFailed) as caught:
                    await task
                assert isinstance(caught.value.cause, ValueError) and caught.value.cancellation is not None
                assert not caught.value.implementation_returned
                assert len(caught.value.unit_outcomes) == 1  # A's earlier outcome is not erased.
                a_outcome = caught.value.unit_outcomes[0]
                assert a_outcome.contribution == "ready"
                assert a_outcome.optimizer == ("returned" if isolated else "not_attempted")
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert any(
                    event == "implementation_returned" and isinstance(value, tuple) and value[0] == name for event, value in state.trace
                )
            assert retained.released and not state.active_retentions
            assert not state.readers and not state.writers and not state.state_locks
            assert state.owners["effects"] == ["computed"]  # No rollback of declared effects.
            assert state.unusable == {"frozen", "a", "b"} and state.unusable_states == {"effects"}
            assert not state.completed and not state.running and state.image is not None
            a, b = state.image.units["A"], state.image.units["B"]
            assert b.steps == 0 and b.pending is None
            assert a.steps == int(isolated and name == "second-objective")
            assert a.pending == (1 if not isolated and name == "second-objective" else None)
            assert not any(event == "entered" and isinstance(value, tuple) and value[0] == "b-gradient" for event, value in state.trace)
            with pytest.raises(NotReady):
                state.capture()
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_exclusive_retention_reuses_its_own_reads_and_derivative_destinations(lowering: LoweringKind):
    filing = routed_views()
    root = sequence(filing)
    first = root.blocks[0].work[1]
    assert isinstance(first, Call)

    def first_with_mode(context):
        context.views["a-mode"](torch.tensor(1.0))
        return first.implementation(context)

    updated = replace(first, implementation=first_with_mode, uses=(*first.uses, Use("a", view="a-mode")))
    block = replace(root.blocks[0], work=tuple(updated if item is first else item for item in root.blocks[0].work))
    filing = replace(
        filing, root=replace(root, blocks=(block, *root.blocks[1:])), views=(*filing.views, View("a-mode", "a", evaluation=True))
    )
    observed = []

    def inspect(context):
        observed.append(context.models["a"].get_parameter("weight").item())

    filing = add_work(filing, 0, Call("read-own-source", inspect, uses=(Use("a"),), after=("first-objective",)))
    ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("first-objective",)).publish()
    run(Engine(ready).run())
    assert observed == [2.0]
    assert ready.state.completed and not ready.state.writers


@pytest.mark.parametrize("nested", (False, True))
def test_incompatible_live_mode_reservations_reject_before_loading(nested):
    loads = []
    participant = replace(scale("model", 1.0), load=lambda: loads.append(True) or Scale(1.0))
    work = (
        Retain("hold", ("model",), (), "old"),
        *((Retain("hold-again", ("model",), (), "inner"),) if nested else ()),
        Call("compute", lambda _: None, uses=(Use("model", view="eval"),), retained="inner" if nested else None),
        *((Release("release-inner", "inner", after=("compute",)),) if nested else ()),
        Release("release", "old", after=("release-inner" if nested else "compute",)),
    )
    accepted = Contract().accept(
        Run(Sequence("sequence", (Block("work", work),)), (participant,), views=(View("eval", "model", evaluation=True),))
    )
    with pytest.raises(NotReady, match="retained"):
        prepare(accepted, worker_calls=("compute",))
    assert not loads


@pytest.mark.parametrize("isolated", (False, True))
def test_nested_source_acquisition_cannot_wait_on_its_own_exclusive_retention(isolated):
    loads = []
    participant = replace(scale("model", 1.0), load=lambda: loads.append(True) or Scale(1.0))
    work = (
        Retain("hold", ("model",), (), "old"),
        Call("compute", lambda _: None, uses=(Use("model", view="eval"),), retained="old"),
        Retain("hold-again", ("model",), (), "inner", isolated=isolated, after=("compute",)),
        Release("release-inner", "inner"),
        Release("release", "old", after=("release-inner",)),
    )
    accepted = Contract().accept(
        Run(Sequence("sequence", (Block("work", work),)), (participant,), views=(View("eval", "model", evaluation=True),))
    )
    with pytest.raises(NotReady, match="retained reservations"):
        prepare(accepted, worker_calls=("compute",))
    assert not loads


def test_broad_mode_change_cannot_upgrade_its_own_live_source():
    work = (
        Retain("hold", ("model",), (), "old"),
        Call("compute", lambda _: None, uses=(Use("model"),), evaluation=True),
        Release("release", "old", after=("compute",)),
    )
    accepted = Contract().accept(Run(Sequence("sequence", (Block("work", work),)), (scale("model", 1.0),)))
    with pytest.raises(NotReady, match="upgrade"):
        prepare(accepted, worker_calls=("compute",))


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_worker_sequence_without_retention_is_not_rejected(lowering: LoweringKind):
    seen = []
    accepted = Contract().accept(
        Run(Sequence("sequence", (Block("block", (Call("call", lambda _: seen.append(threading.get_ident())),)),)))
    )
    run(Engine(prepare(accepted, lowering=lowering, worker_calls=("call",)).publish()).run())
    assert len(seen) == 1 and seen[0] != threading.get_ident()


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_cancelled_capacity_wait_never_enters_retained_call(lowering: LoweringKind, monkeypatch):
    async def scenario():
        loop, entered, queued, finish = asyncio.get_running_loop(), asyncio.Event(), asyncio.Event(), threading.Event()
        selected = []

        class ObservedWorkers(Workers):
            submissions = 0

            async def execute(self, call, stop):
                self.submissions += 1
                if self.submissions == 2:
                    queued.set()
                return await super().execute(call, stop)

        from . import prepare as preparation

        monkeypatch.setattr(preparation, "Workers", ObservedWorkers)

        def occupy(_):
            loop.call_soon_threadsafe(entered.set)
            assert finish.wait(4)

        filing = replace_call(routed_views(), "first-objective", lambda _: selected.append(True))
        filing = replace(filing, root=Together("run", (Block("occupier", (Call("occupy", occupy),)), filing.root)))
        ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("occupy", "first-objective"), worker_limit=1).publish()
        task = asyncio.create_task(Engine(ready).run())
        try:
            await entered.wait()
            await queued.wait()
            assert ready.state.active_retentions and not selected
            task.cancel()
            # Parent TaskGroup cancellation drains the active worker. The
            # queued source can finish its own cleanup without joining it.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not selected and not task.done()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            state = ready.state
            assert any(
                event == "worker_not_started" and isinstance(value, tuple) and value[0] == "first-objective" for event, value in state.trace
            )
            assert not state.active_retentions and not state.writers and not state.readers
            assert state.image is not None
            assert all(not unit.outcomes and unit.pending is None for unit in state.image.units.values())
            assert all(state.stamp(name).numerical_epoch == 0 for name in state.image.bindings)
            assert not state.completed  # Sequence withholding is conservative, not a claim of selected effects.
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("invalid_output", (False, True))
def test_retained_call_failure_keeps_effects_and_releases_only_after_physical_completion(lowering: LoweringKind, invalid_output):
    filing = routed_views()
    original = sequence(filing).blocks[0].work[1]
    assert isinstance(original, Call)

    def failing(context):
        original.implementation(context)
        context.states["effects"].append("ran")
        if invalid_output:
            return "invalid"
        raise ValueError("retained computation failed")

    filing = replace_call(filing, "first-objective", failing, states=("effects",))
    filing = replace(filing, states=(OwnedState("effects", list),))
    ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("first-objective",)).publish()
    with pytest.raises(WorkFailed) as caught:
        run(Engine(ready).run())
    assert isinstance(caught.value.cause, TypeError if invalid_output else ValueError)
    assert caught.value.implementation_returned is invalid_output
    state = ready.state
    assert state.owners["effects"] == ["ran"] and state.unusable_states == {"effects"}
    assert not state.active_retentions and not state.readers and not state.writers
    assert state.image is not None and all(unit.steps == 0 and unit.pending is None for unit in state.image.units.values())
    assert state.image.bindings["frozen"].model.training
    assert state.unusable == {"frozen", "a", "b"} and not state.completed


def test_cross_scope_reservation_cycle_rejects_before_loading():
    loads = []

    def scope(name, other):
        return Sequence(
            f"sequence-{name}",
            (
                Block(
                    f"block-{name}",
                    (
                        Retain(f"hold-{name}", (name,), (), f"old-{name}"),
                        Call(f"compute-{name}", lambda _: None, uses=(Use(name, view=f"{name}-eval"),), retained=f"old-{name}"),
                        Call(f"read-{other}", lambda _: None, uses=(Use(other),), after=(f"compute-{name}",)),
                        Release(f"release-{name}", f"old-{name}", after=(f"read-{other}",)),
                    ),
                ),
            ),
        )

    participants = tuple(replace(scale(name, 1.0), load=lambda: loads.append(True) or Scale(1.0)) for name in ("x", "y"))
    accepted = Contract().accept(
        Run(
            Together("run", (scope("x", "y"), scope("y", "x"))),
            participants,
            views=(View("x-eval", "x", evaluation=True), View("y-eval", "y", evaluation=True)),
        )
    )
    with pytest.raises(NotReady, match="reservation wait cycle") as caught:
        prepare(accepted, worker_calls=("compute-x", "compute-y"), worker_limit=2)
    assert "sequence-x/read-y" in str(caught.value) and "sequence-y/read-x" in str(caught.value)
    assert not loads


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_sequence_mode_then_input_wait_uses_released_reservations(lowering: LoweringKind):

    def evaluate(context):
        context.views["eval"](torch.tensor(1.0))

    async def source(context):
        async with context.use_source():
            product = Product("", 7)
        await context.publish("input", product)

    accepted = Contract().accept(
        Run(
            Together(
                "run",
                (
                    Sequence(
                        "sequence",
                        (
                            Block(
                                "block",
                                (
                                    Call("compute", evaluate, uses=(Use("model", view="eval"),)),
                                    Join("receive", ("input",), (Output("value", int),), lambda *_: True, after=("compute",)),
                                ),
                            ),
                        ),
                    ),
                    Producer("source", ("input",), source, uses=(Use("model"),)),
                ),
            ),
            (scale("model", 1.0),),
            channels=(Channel("input", 1),),
            views=(View("eval", "model", evaluation=True),),
        )
    )
    direct = prepare(accepted, lowering=lowering).publish()
    threaded = prepare(accepted, lowering=lowering, worker_calls=("compute",)).publish()
    run(Engine(direct).run())
    run(Engine(threaded).run())
    assert direct.state.completed and threaded.state.completed
    assert observable(threaded) == observable(direct)
