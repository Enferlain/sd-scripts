"""Current-use ownership agrees with preparation and physical lifetimes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
import torch

from .examples import Scale, scale
from .graph import (
    Block,
    Call,
    Channel,
    Contract,
    Join,
    NotReady,
    OutOfBounds,
    Output,
    OwnedState,
    Producer,
    Release,
    Replace,
    Retain,
    Run,
    Sequence,
    Together,
    Use,
    View,
)
from .prepare import Engine, WorkFailed, prepare
from .state import Product, Replacement
from .test_access import pair
from .test_candidate import leaves
from .test_workers import run


def ready_state():
    filing = Run(Block("idle", ()), tuple(scale(name, 1.0) for name in ("x", "y", "z")), states=(OwnedState("journal", list),))
    return prepare(Contract().accept(filing)).publish().state


@pytest.mark.parametrize("exclusive", (False, True))
def test_borrowed_read_keeps_its_source_reservation_and_releases_only_new_uses(exclusive):
    async def scenario():
        state, region = ready_state(), object()
        retained = await state.retain(Retain("hold", ("x",), (), "old"), region, exclusive=("x",) if exclusive else ())
        readers, writers = dict(state.readers), set(state.writers)
        async with state.use((Use("x"), Use("y")), ("journal",), region=region) as projection:
            assert projection.models["x"] is retained.projection.models["x"]
            assert projection.stamps["x"] == retained.projection.stamps["x"]
            assert state.readers == {**readers, "y": 1} and state.writers == writers
            assert state.state_locks == {"journal"}
        assert state.readers == readers and state.writers == writers and not state.state_locks
        assert not retained.released and state.active_retentions[retained.token] is retained
        await state.release(retained, region)
        assert not state.readers and not state.writers and not state.active_retentions

    run(scenario())


@pytest.mark.parametrize("held_write,new_write", ((False, True), (True, False), (True, True)))
def test_conflicting_new_retention_rejects_instead_of_waiting_on_itself(held_write, new_write):
    async def scenario():
        state, region = ready_state(), object()
        retained = await state.retain(Retain("first", ("x",), (), "old"), region, exclusive=("x",) if held_write else ())
        before = dict(state.readers), set(state.writers)
        with pytest.raises(NotReady, match="own retained reservations"):
            await state.retain(Retain("second", ("x",), (), "next"), region, exclusive=("x",) if new_write else ())
        assert (state.readers, state.writers) == before
        assert tuple(state.active_retentions) == (retained.token,)
        await state.release(retained, region)

    run(scenario())


@pytest.mark.parametrize("isolated", (False, True))
def test_new_read_retention_has_its_own_lifetime_or_a_real_copy(isolated):
    async def scenario():
        state, region = ready_state(), object()
        first = await state.retain(Retain("first", ("x",), (), "old"), region)
        second = await state.retain(Retain("second", ("x",), (), "next", isolated=isolated), region)
        assert state.readers == {"x": 1 if isolated else 2}
        assert (second.projection.models["x"] is first.projection.models["x"]) is not isolated
        await state.release(second, region)
        assert state.readers == {"x": 1} and not first.released
        await state.release(first, region)
        assert not state.readers

    run(scenario())


@pytest.mark.parametrize("exclusive", (False, True))
def test_current_write_cannot_upgrade_or_reuse_a_retained_source(exclusive):
    async def scenario():
        state, region = ready_state(), object()
        retained = await state.retain(Retain("hold", ("x",), (), "old"), region, exclusive=("x",) if exclusive else ())
        epoch = state.stamp("x")
        with pytest.raises(NotReady, match="own retained reservations"):
            async with state.use((Use("x", "write"),), region=region):
                pytest.fail("Upgraded retained state")
        assert state.stamp("x") == epoch and not retained.released
        await state.release(retained, region)

    run(scenario())


@pytest.mark.parametrize("unavailable", (False, True))
def test_borrowing_checks_source_validity_even_without_a_new_reservation(unavailable):
    async def scenario():
        state, region = ready_state(), object()
        retained = await state.retain(Retain("hold", ("x",), (), "old"), region)
        if unavailable:
            state.unusable.add("x")
        else:
            state.returned_writes((Use("x", "write"),))  # Inject known stale source evidence.
        with pytest.raises(NotReady, match=r"unavailable|changed before last use"):
            async with state.use((Use("x"),), region=region):
                pytest.fail("Supplied invalid borrowed state")
        assert state.readers == {"x": 1}
        await state.release(retained, region, aborted=True)

    run(scenario())


def test_foreign_region_waits_for_current_access_instead_of_borrowing_another_source():
    async def scenario():
        state, region = ready_state(), object()
        retained = await state.retain(Retain("hold", ("x",), (), "old"), region, exclusive=("x",))
        entered = []

        async def independent():
            async with state.use((Use("x"),), region=object()) as projection:
                entered.append(projection.models["x"])
                assert state.readers == {"x": 1} and not state.writers

        task = asyncio.create_task(independent())
        try:
            await asyncio.sleep(0)
            assert not task.done() and not entered and state.writers == {"x"}
            await state.release(retained, region)
            await task
            assert entered == [retained.projection.models["x"]] and not state.readers
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


def test_borrowed_source_is_rechecked_after_waiting_for_an_uncovered_member():
    async def scenario():
        state, region = ready_state(), object()
        retained = await state.retain(Retain("hold", ("x",), (), "old"), region, exclusive=("x",))
        selected = []

        async def use_both():
            async with state.use((Use("x"), Use("y")), region=region):
                selected.append(True)

        task = None
        try:
            async with state.acquire((Use("y", "write"),)):
                task = asyncio.create_task(use_both())
                await asyncio.sleep(0)
                assert not task.done() and not selected
                await state.release(retained, region)
            with pytest.raises(NotReady, match="released, foreign"):
                await task
            assert not selected and not state.readers and not state.writers
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    run(scenario())


def test_region_cleanup_attempts_all_its_lifetimes_and_leaves_other_owners_alone(monkeypatch):
    async def scenario():
        state, region, other = ready_state(), object(), object()
        first = await state.retain(Retain("first", ("x",), (), "old-x"), region)
        second = await state.retain(Retain("second", ("y",), (), "old-y"), region)
        foreign = await state.retain(Retain("foreign", ("z",), (), "old-z"), other)
        original = state.release

        async def fail_one(retained, owner, **kwargs):
            if retained is first:
                raise RuntimeError("first release failed")
            await original(retained, owner, **kwargs)

        monkeypatch.setattr(state, "release", fail_one)
        with pytest.raises(ExceptionGroup, match="lifetime cleanup") as caught:
            await state.release_region(region)
        assert str(caught.value.exceptions[0]) == "first release failed"
        assert second.released and not first.released and not foreign.released
        assert state.readers == {"x": 1, "z": 1}
        await original(first, region, aborted=True)
        await original(foreign, other)
        assert not state.active_retentions and not state.readers

    run(scenario())


@pytest.mark.parametrize("physical_release", (False, True))
def test_failed_close_ends_borrowing_and_wakes_waiters_without_claiming_release(physical_release):
    async def scenario():
        state, region = ready_state(), object()
        retained = await state.retain(Retain("hold", ("x",), (), "old"), region)
        original = retained.lease
        assert original is not None

        @asynccontextmanager
        async def faulty_exit():
            yield retained.projection
            if physical_release:
                await original.__aexit__(None, None, None)
            raise RuntimeError("physical release failed")

        retained.lease = faulty_exit()
        await retained.lease.__aenter__()
        entered = []

        async def writer():
            async with state.use((Use("x", "write"),)):
                entered.append(True)

        old_image = state.image
        waiter = asyncio.create_task(writer())
        publication = asyncio.create_task(state.replace(Replacement("x", 1, Scale(2.0)), "x"))
        try:
            await asyncio.sleep(0)
            assert not waiter.done() and not publication.done() and state.readers == {"x": 1}
            with pytest.raises(RuntimeError, match="physical release failed"):
                await state.release(retained, region)
            assert retained.closing and not retained.released
            assert state.active_retentions[retained.token] is retained
            assert state.readers == ({} if physical_release else {"x": 1})
            assert state.unusable == {"x"} and state.stamp("x").numerical_epoch == 0
            with pytest.raises(NotReady, match="closing"):
                async with state.use((Use("x"),), region=region):
                    pytest.fail("Borrowed closing state")
            with pytest.raises(NotReady, match="unavailable"):
                await asyncio.wait_for(waiter, 1)
            with pytest.raises(NotReady, match="closing retained"):
                await asyncio.wait_for(publication, 1)
            assert state.image is old_image and state.generation == 1
            with pytest.raises(NotReady, match="unavailable"):
                async with state.use((Use("x"),)):
                    pytest.fail("Supplied state after uncertain release")
            with pytest.raises(NotReady, match="cannot be retried"):
                await state.release(retained, region, aborted=True)
            async with state.use((Use("y"),)):
                assert state.readers.get("y") == 1
            assert not entered and not state.writers
        finally:
            waiter.cancel()
            publication.cancel()
            await asyncio.gather(waiter, publication, return_exceptions=True)
            if not physical_release:
                # Test teardown only, not owner recovery or restored readiness.
                await original.__aexit__(None, None, None)

    run(scenario())


def test_replacement_still_waits_for_usable_retained_work_and_successful_close():
    async def scenario():
        state, region = ready_state(), object()
        retained = await state.retain(Retain("hold", ("x",), (), "old"), region)
        publication = asyncio.create_task(state.replace(Replacement("x", 1, Scale(2.0)), "x"))
        try:
            await asyncio.sleep(0)
            assert not publication.done() and state.readers == {"x": 1}
            await state.release(retained, region)
            await publication
            assert retained.released and state.generation == 2 and state.stamp("x").binding == 2
            assert not state.active_retentions and not state.readers and not state.unusable
        finally:
            publication.cancel()
            await asyncio.gather(publication, return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("cancel", (False, True))
def test_close_claim_precedes_release_await_and_cancellation_preserves_uncertainty(cancel):
    async def scenario():
        state, region = ready_state(), object()
        retained = await state.retain(Retain("hold", ("x",), (), "old"), region)
        original = retained.lease
        assert original is not None
        entered, finish = asyncio.Event(), asyncio.Event()

        @asynccontextmanager
        async def delayed_exit():
            yield retained.projection
            entered.set()
            await finish.wait()
            await original.__aexit__(None, None, None)

        retained.lease = delayed_exit()
        await retained.lease.__aenter__()
        task = asyncio.create_task(state.release(retained, region))
        try:
            await entered.wait()
            assert retained.closing and not retained.released and state.readers == {"x": 1}
            with pytest.raises(NotReady, match="closing"):
                state.check_retained(retained, region)
            with pytest.raises(NotReady, match="closing"):
                async with state.use((Use("x"),), region=region):
                    pytest.fail("Borrowed state while its close was in flight")
            with pytest.raises(NotReady, match="already closing"):
                await state.release(retained, region, aborted=True)
            assert state.readers == {"x": 1} and not task.done()
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert retained.closing and not retained.released and state.unusable == {"x"}
                assert state.active_retentions[retained.token] is retained and state.readers == {"x": 1}
                with pytest.raises(NotReady, match="unavailable"):
                    async with state.use((Use("x", "write"),)):
                        pytest.fail("Waited forever on failed physical protection")
            else:
                finish.set()
                await task
                assert retained.released and not retained.closing and not state.active_retentions
                assert not state.readers and not state.unusable
                async with state.use((Use("x", "write"),)):
                    assert state.writers == {"x"}
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if cancel:
                await original.__aexit__(None, None, None)

    run(scenario())


def test_region_cleanup_drains_other_lifetimes_after_actual_physical_release_failure():
    async def scenario():
        state, region = ready_state(), object()
        first = await state.retain(Retain("first", ("x",), (), "old-x"), region)
        second = await state.retain(Retain("second", ("y",), (), "old-y"), region)
        original = first.lease
        assert original is not None

        @asynccontextmanager
        async def faulty_exit():
            yield first.projection
            await original.__aexit__(None, None, None)
            raise RuntimeError("reported failure after physical release")

        first.lease = faulty_exit()
        await first.lease.__aenter__()
        with pytest.raises(ExceptionGroup, match="lifetime cleanup") as caught:
            await state.release_region(region)
        assert str(caught.value.exceptions[0]) == "reported failure after physical release"
        assert first.closing and not first.released and second.released
        assert tuple(state.active_retentions) == (first.token,) and not state.readers
        assert state.unusable == {"x"}

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_region_failure_keeps_primary_and_cleanup_facts_while_draining_other_sources(lowering, monkeypatch):
    async def scenario():
        def fail(context):
            raise ValueError("selected work failed")

        filing = Run(
            Sequence(
                "sequence",
                (
                    Block(
                        "body",
                        (
                            Retain("hold-x", ("x",), (), "old-x"),
                            Retain("hold-y", ("y",), (), "old-y", after=("hold-x",)),
                            Call("fail", fail, after=("hold-y",)),
                            Release("release-x", "old-x", after=("fail",)),
                            Release("release-y", "old-y", after=("release-x",)),
                        ),
                    ),
                ),
            ),
            (scale("x", 1.0), scale("y", 1.0)),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering).publish()
        state, original = ready.state, ready.state.release

        async def fail_x(retained, region, **kwargs):
            if "x" in retained.projection.models:
                raise RuntimeError("x release failed")
            await original(retained, region, **kwargs)

        monkeypatch.setattr(state, "release", fail_x)
        with pytest.raises(ExceptionGroup) as caught:
            await Engine(ready).run()
        errors = leaves(caught.value)
        assert any(isinstance(error, WorkFailed) and isinstance(error.cause, ValueError) for error in errors)
        assert any(str(error) == "x release failed" for error in errors)
        assert state.unusable == {"x", "y"} and not state.completed
        assert state.readers == {"x": 1} and len(state.active_retentions) == 1
        retained = next(iter(state.active_retentions.values()))
        await original(retained, retained.region, aborted=True)
        assert not state.readers and not state.active_retentions

    run(scenario())


@pytest.mark.parametrize("requested", ((Use("y", "write"),), (Use("missing"),), (Use("x", view="invented"),)))
def test_provider_current_use_cannot_expand_its_authority(requested):
    async def scenario():
        state = ready_state()
        with pytest.raises(OutOfBounds, match="accepted authority"):
            async with state.use_source((Use("x", "write"), Use("y")), requested):
                pytest.fail("Expanded provider authority")
        assert not state.readers and not state.writers and not state.unusable
        assert state.stamp("x").numerical_epoch == state.stamp("y").numerical_epoch == 0

    run(scenario())


def test_provider_can_select_a_read_only_subset_without_claiming_an_allowed_write():
    async def scenario():
        state = ready_state()
        async with state.use_source((Use("x", "write"), Use("y")), (Use("x"),)) as projection:
            assert set(projection.models) == {"x"} and state.readers == {"x": 1} and not state.writers
        assert state.stamp("x").numerical_epoch == 0 and not state.unusable and not state.readers

    run(scenario())


@pytest.mark.parametrize("fail", (False, True))
def test_provider_effect_evidence_is_completed_under_the_same_protection(fail, monkeypatch):
    async def scenario():
        state = ready_state()
        original, observed = state.returned_writes, []

        def returned(uses):
            assert state.writers == {"x"}
            original(uses)
            observed.append(state.stamp("x").numerical_epoch)

        monkeypatch.setattr(state, "returned_writes", returned)
        try:
            async with state.use_source((Use("x", "write"),)):
                if fail:
                    raise RuntimeError("uncertain write")
        except RuntimeError:
            assert fail
        assert not state.readers and not state.writers
        assert observed == ([] if fail else [1])
        assert state.unusable == ({"x"} if fail else set())
        if fail:
            with pytest.raises(NotReady, match="unavailable"):
                async with state.use((Use("x"),)):
                    pytest.fail("Supplied failed source")

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("worker", (False, True))
def test_preparation_uses_released_sequence_access_not_a_whole_block_approximation(lowering, worker):
    async def scenario():
        edited, seen = asyncio.Event(), []

        def edit(context):
            with torch.no_grad():
                context.models["model"].get_parameter("weight").add_(1)
            if not worker:
                edited.set()
            else:
                loop.call_soon_threadsafe(edited.set)

        async def source(context):
            await edited.wait()
            async with context.use_source() as projection:
                value = int(projection.models["model"].get_parameter("weight").detach())
            await context.publish("input", Product("", value))

        loop = asyncio.get_running_loop()
        body = Block(
            "consumer",
            (
                Call("edit", edit, uses=(Use("model", "write"),)),
                Join("receive", ("input",), (Output("value", int),), lambda *_: True, after=("edit",)),
                Call("observe", lambda _, value: seen.append(value), ("value",)),
            ),
        )
        filing = Run(
            Together("run", (Sequence("sequence", (body,)), Producer("source", ("input",), source, uses=(Use("model"),)))),
            (scale("model", 1.0),),
            channels=(Channel("input", 1),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("edit",) if worker else ()).publish()
        await Engine(ready).run()
        assert seen == [2] and ready.state.completed and not ready.state.readers and not ready.state.writers

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_one_graph_can_use_cooperative_modes_but_reject_conflicting_worker_lifetimes(lowering):
    filing, seen = pair()
    accepted = Contract().accept(filing)
    ready = prepare(accepted, lowering=lowering).publish()
    run(Engine(ready).run())
    assert sorted(seen) == ["x", "y"]
    with pytest.raises(NotReady, match="reservation wait cycle"):
        prepare(accepted, lowering=lowering, worker_calls=("compute-x", "compute-y"), worker_limit=2)


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("worker", (False, True))
def test_preparation_checks_own_retained_conflicts_at_the_real_acquisition_boundary(nested, worker):
    seen, loads = [], []
    read = Call(
        "evaluate",
        lambda _: seen.append(True),
        uses=(Use("model", view="eval"),),
        retained="old" if nested else None,
        evaluation=not nested,
    )
    extra = (Retain("nested", ("model",), (), "inner", after=("hold",)),) if nested else ()
    read = replace(read, after=("nested" if nested else "hold",))
    releases = (Release("inner-release", "inner", after=("evaluate",)),) if nested else ()
    filing = Run(
        Sequence(
            "sequence",
            (
                Block(
                    "body",
                    (
                        Retain("hold", ("model",), (), "old"),
                        *extra,
                        read,
                        *releases,
                        Release("release", "old", after=("inner-release" if nested else "evaluate",)),
                    ),
                ),
            ),
        ),
        (replace(scale("model", 1.0), load=lambda: loads.append(True) or Scale(1.0)),),
        views=(View("eval", "model", evaluation=True),),
    )
    accepted = Contract().accept(filing)
    if nested and not worker:
        ready = prepare(accepted).publish()
        run(Engine(ready).run())
        assert seen == [True] and not ready.state.readers
    else:
        with pytest.raises(NotReady, match="own retained reservations"):
            prepare(accepted, worker_calls=("evaluate",) if worker else ())
        assert not seen and not loads


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_replacement_inside_a_sequence_uses_completed_per_work_lifetimes(lowering):
    body = Block(
        "replace-after-use",
        (
            Call(
                "proposal",
                lambda _: Replacement("model", 1, Scale(3.0)),
                outputs=(Output("replacement", Replacement),),
                uses=(Use("model"),),
            ),
            Replace("replace", "replacement", "model"),
        ),
    )
    filing = Run(Sequence("sequence", (body,)), (scale("model", 1.0),))
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    run(Engine(ready).run())
    assert ready.state.image is not None and ready.state.generation == 2
    assert float(ready.state.image.bindings["model"].model.get_parameter("weight").detach()) == 3.0
    loads = []
    aggregate = replace(filing, root=body, participants=(replace(filing.participants[0], load=lambda: loads.append(True) or Scale(1.0)),))
    with pytest.raises(NotReady, match="outside protected"):
        prepare(Contract().accept(aggregate), lowering=lowering)
    assert not loads


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_foreign_isolated_retention_remains_a_runtime_replacement_readiness_guard(lowering):
    async def scenario():
        held = asyncio.Event()

        async def pause(_):
            held.set()
            await asyncio.Event().wait()

        async def propose(_):
            await held.wait()
            return Replacement("model", 1, Scale(3.0))

        source = Sequence(
            "source",
            (
                Block(
                    "retain",
                    (
                        Retain("hold", ("model",), (), "old", isolated=True),
                        Call("pause", pause, after=("hold",)),
                        Release("release", "old", after=("pause",)),
                    ),
                ),
            ),
        )
        publish = Sequence(
            "publication",
            (
                Block(
                    "publish",
                    (
                        Call("propose", propose, outputs=(Output("proposal", Replacement),)),
                        Replace("replace", "proposal", "model"),
                    ),
                ),
            ),
        )
        filing = Run(Together("run", (source, publish)), (scale("model", 1.0),))
        ready = prepare(Contract().accept(filing), lowering=lowering).publish()
        old_image = ready.state.image
        with pytest.raises(ExceptionGroup) as caught:
            await Engine(ready).run()
        assert any(
            isinstance(error, WorkFailed) and isinstance(error.cause, NotReady) and "unfinished retained derivative" in str(error.cause)
            for error in leaves(caught.value)
        )
        assert ready.state.image is old_image and ready.state.generation == 1
        assert not ready.state.active_retentions and not ready.state.readers and not ready.state.completed

    run(scenario())
