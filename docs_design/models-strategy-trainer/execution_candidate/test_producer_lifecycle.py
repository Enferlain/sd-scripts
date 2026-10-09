"""Producer lifetime and returned-effect evidence in the connected candidate."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
import torch
from torch import nn

from .examples import ContinuousSource, Scale, execute, producer_continuity, scale
from .graph import (
    Block,
    Call,
    Channel,
    Choice,
    Contract,
    Join,
    NotReady,
    OutOfBounds,
    Output,
    OwnedState,
    Producer,
    Rejected,
    Repeat,
    Replace,
    Requested,
    Run,
    Together,
    TwoPass,
    Use,
)
from .prepare import Engine, Lowering, ProducerFailed, WorkFailed, prepare
from .state import Product, Replacement, join
from .test_candidate import leaves, run


def test_connected_producer_shutdown_keeps_production_claims_and_advancement_distinct():
    ready = run(execute(producer_continuity()))
    state, feed = ready.state, ready.state.feeds["condition"]
    owner = state.owners["encoding"]
    assert isinstance(owner, ContinuousSource)
    assert owner.finished and owner.high_water == 4 and feed.high_water == 2
    assert owner.work[1] == owner.work[2] == "published"
    assert owner.work[4] == "discarded"
    assert owner.work[3] in ("published", "discarded")
    assert [key for key, _, _ in feed.handed_off] == [1]
    assert 2 in {key for key, _ in feed.discarded}  # Old source work was never admitted.
    assert not feed.ready and feed.closed and not state.active_producers
    assert not state.unusable and not state.unusable_states and state.completed
    assert state.image is not None and state.image.units["weights"].steps == 1
    assert state.stamp("encoder").numerical_epoch == feed.handed_off[0][2][0].numerical_epoch == 1
    assert state.coordinates == {"learning": {"wait-ahead": 1, "learn": 1}}
    assert state.capture().owner_states["encoding"].finished


@pytest.mark.parametrize("continuity", ["reset", "preserve"])
def test_owner_replacement_respects_active_producer_identity_through_cleanup(continuity):
    async def check():
        started, finish, cleaning, release = (asyncio.Event() for _ in range(4))
        retained = []

        async def provider(context):
            owner = context.states["owner"]
            retained.append(owner)
            started.set()
            await finish.wait()
            owner.append("execution returned")

        async def cleanup(context):
            cleaning.set()
            await release.wait()
            context.states["owner"].append("cleanup returned")

        filing = Run(
            Producer("source", (), provider, states=("owner",), cleanup=cleanup),
            (scale("model", 1.0),),
            states=(OwnedState("owner", list, ("model",), continuity),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        state = ready.state
        original = state.owners["owner"]
        task = asyncio.create_task(Engine(ready).run())
        await started.wait()
        if continuity == "reset":
            with pytest.raises(NotReady, match="active producer"):
                await state.replace(Replacement("model", 1, Scale(2.0)), "model")
            assert state.generation == 1 and state.owners["owner"] is original
        else:
            await state.replace(Replacement("model", 1, Scale(2.0)), "model")
            assert state.generation == 2 and state.owners["owner"] is original
        finish.set()
        await cleaning.wait()
        if continuity == "reset":
            with pytest.raises(NotReady, match="finish cleanup"):
                await state.replace(Replacement("model", 1, Scale(2.0)), "model")
        assert state.active_producers == {"source": ("owner",)}
        with pytest.raises(NotReady, match="quiescent"):
            state.capture()
        release.set()
        await task
        assert retained == [original] and original == ["execution returned", "cleanup returned"]
        assert not state.active_producers
        if continuity == "reset":
            await state.replace(Replacement("model", 1, Scale(2.0)), "model")
            assert state.owners["owner"] == [] and state.owners["owner"] is not original

    run(check())


def test_provider_binds_owner_at_execution_not_at_lowering():
    async def check():
        seen = []

        async def provider(context):
            seen.append(context.states["owner"])
            context.states["owner"].append("current")

        filing = Run(
            Producer("source", (), provider, states=("owner",)),
            (scale("model", 1.0),),
            states=(OwnedState("owner", list, ("model",), "reset"),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        old = ready.state.owners["owner"]
        await ready.state.replace(Replacement("model", 1, Scale(2.0)), "model")
        await Engine(ready).run()
        assert seen[0] is ready.state.owners["owner"] and seen[0] is not old
        assert old == [] and seen[0] == ["current"]

    run(check())


def test_declared_producer_write_cannot_relabel_old_work_but_fresh_work_can_admit():
    async def check():
        async def provider(context):
            async with context.use_source((Use("model"),)) as source:
                old = Product("old", float(source.models["model"].get_parameter("weight")), (source.stamps["model"],))
            async with context.use_source() as source:
                with torch.no_grad():
                    source.models["model"].get_parameter("weight").add_(1)
            async with context.use_source((Use("model"),)) as source:
                new = Product("new", float(source.models["model"].get_parameter("weight")), (source.stamps["model"],))
            await context.publish("port", old)
            await context.publish("port", new)

        filing = Run(
            Producer("source", ("port",), provider, (Use("model", "write"),)), (scale("model", 1.0),), channels=(Channel("port", 2),)
        )
        ready = await execute(filing)
        state, feed = ready.state, ready.state.feeds["port"]
        assert state.stamp("model").numerical_epoch == 1
        assert feed.ready["old"].value == 1.0 and feed.ready["new"].value == 2.0
        async with state.acquire((Use("model"),)) as source:

            def admit(product, view):
                return product.provenance == (view.stamps["model"],)

            with pytest.raises(NotReady, match="Inadmissible"):
                await join((feed,), "old", 1, admit, source)
            assert await join((feed,), "new", 2, admit, source) == (2.0,)
        assert "old" in feed.ready and not state.unusable

    run(check())


def test_provider_cannot_upgrade_declared_read_authority():
    async def provider(context):
        async with context.use_source((Use("model", "write"),)):
            pytest.fail("Unaccepted source access was supplied")

    filing = Run(Producer("source", (), provider, (Use("model"),)), (scale("model", 1.0),))
    with pytest.raises(OutOfBounds, match="source access exceeds"):
        run(execute(filing))


@pytest.mark.parametrize("returned", [False, True])
def test_execution_and_all_evaluation_cleanup_outcomes_survive(returned):
    primary, cleanup = ValueError("primary failure"), RuntimeError("cleanup failure")

    class Fragile(nn.Module):
        def __setattr__(self, name, value):
            if name == "training" and value is True and getattr(self, "armed", False):
                raise cleanup
            super().__setattr__(name, value)

    model = Scale(1.0)
    model.bad = Fragile()
    model.good = nn.Identity()

    def selected(context):
        context.models["model"].bad.armed = True
        if not returned:
            raise primary

    filing = Run(
        Block("work", (Call("evaluate", selected, uses=(Use("model", "write"),), evaluation=True),)),
        (replace(scale("model", 1.0), load=lambda: model),),
    )
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(WorkFailed) as failure:
        run(Engine(ready).run())
    assert failure.value.cause is (cleanup if returned else primary)
    assert failure.value.cleanup_failures == (cleanup,)
    assert failure.value.implementation_returned is returned
    assert model.good.training and model.training  # Cleanup continued after bad child failed.
    assert not model.bad.training and ready.state.unusable == {"model"}
    assert ready.state.stamp("model").numerical_epoch == int(returned)


def test_ordinary_call_does_not_snapshot_module_modes_on_hot_path(monkeypatch):
    filing = Run(Block("work", (Call("ordinary", lambda context: None, uses=(Use("model"),)),)), (scale("model", 1.0),))
    ready = prepare(Contract().accept(filing)).publish()
    assert ready.state.image is not None
    monkeypatch.setattr(ready.state.image.bindings["model"].model, "modules", lambda: pytest.fail("unneeded mode walk"))
    run(Engine(ready).run())


def test_invalid_output_still_records_implementation_return_and_effect():
    def selected(context):
        with torch.no_grad():
            context.models["model"].get_parameter("weight").add_(1)
        return "invalid"

    filing = Run(
        Block("work", (Call("call", selected, outputs=(Output("value", int),), uses=(Use("model", "write"),)),)), (scale("model", 1.0),)
    )
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(WorkFailed) as failure:
        run(Engine(ready).run())
    assert failure.value.implementation_returned and isinstance(failure.value.cause, TypeError)
    assert not failure.value.cleanup_failures and ready.state.stamp("model").numerical_epoch == 1
    assert ready.state.unusable == {"model"}


@pytest.mark.parametrize("returned", [False, True])
def test_provider_primary_and_explicit_cleanup_failure_are_separate(returned):
    primary, cleanup = ValueError("production failed"), RuntimeError("owner cleanup failed")

    async def provider(context):
        context.states["owner"].append("reached")
        if not returned:
            raise primary

    async def finalize(context):
        context.states["owner"].append("cleanup reached")
        raise cleanup

    filing = Run(
        Producer("source", ("port",), provider, states=("owner",), cleanup=finalize),
        states=(OwnedState("owner", list),),
        channels=(Channel("port", 1),),
    )
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(ProducerFailed) as failure:
        run(Engine(ready).run())
    assert failure.value.cause is (None if returned else primary)
    assert failure.value.cleanup_failures == (cleanup,) and failure.value.implementation_returned is returned
    assert ready.state.owners["owner"] == ["reached", "cleanup reached"]
    assert ready.state.unusable_states == {"owner"} and not ready.state.active_producers
    assert ready.state.feeds["port"].failure is failure.value and not ready.state.completed


def test_cancellation_does_not_release_owner_or_interrupt_already_running_cleanup():
    async def check():
        started, cleaning, release = (asyncio.Event() for _ in range(3))

        async def provider(context):
            started.set()
            await asyncio.Event().wait()

        async def cleanup(context):
            cleaning.set()
            await release.wait()
            context.states["owner"].append("cleaned")

        filing = Run(
            Producer("source", (), provider, states=("owner",), cleanup=cleanup),
            (scale("model", 1.0),),
            states=(OwnedState("owner", list, ("model",), "reset"),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        task = asyncio.create_task(Engine(ready).run())
        await started.wait()
        task.cancel()
        await cleaning.wait()
        task.cancel()  # Must not cancel/detach the cleanup continuation.
        with pytest.raises(NotReady, match="finish cleanup"):
            await ready.state.replace(Replacement("model", 1, Scale(2.0)), "model")
        assert ready.state.active_producers and not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ready.state.owners["owner"] == ["cleaned"] and not ready.state.active_producers
        assert ready.state.unusable_states == {"owner"}  # Cancellation is not automatic recovery.

    run(check())


def test_demand_end_interrupting_a_write_withholds_uncertain_source_after_release():
    async def check():
        writing = asyncio.Event()

        async def provider(context):
            async with context.use_source() as source:
                with torch.no_grad():
                    source.models["model"].get_parameter("weight").add_(1)
                writing.set()
                await asyncio.Event().wait()

        async def done(context):
            await writing.wait()

        filing = Run(
            Together(
                "run",
                (
                    Producer("source", (), provider, (Use("model", "write"),), stop_when="demand", remaining="discard"),
                    Block("demand", (Call("end", done),)),
                ),
            ),
            (scale("model", 1.0),),
        )
        ready = await execute(filing)
        assert not ready.state.writers and ready.state.unusable == {"model"}
        assert ready.state.stamp("model").numerical_epoch == 0  # Interrupted, not returned.
        with pytest.raises(NotReady, match="unavailable"):
            async with ready.state.acquire((Use("model"),)):
                pytest.fail("Uncertain source was supplied")
        with pytest.raises(NotReady, match="usable"):
            ready.state.capture()

    run(check())


def test_known_mixed_replacement_target_limit_rejects_before_loading(monkeypatch):
    monkeypatch.setattr(Scale, "__init__", lambda *args: pytest.fail("premature load"))
    filing = Run(
        Block(
            "mixed",
            (
                Call("proposal", lambda context: None, outputs=(Output("replacement", Replacement),), uses=(Use("model"),)),
                Replace("replace", "replacement", "model"),
            ),
        ),
        (scale("model", 1.0),),
    )
    with pytest.raises(Rejected, match="outside protected"):
        Contract().accept(filing)


@pytest.mark.parametrize("target,remaining", [("missing", "discard"), ("run", "discard"), ("source", "discard"), ("demand", None)])
def test_stop_policy_requires_a_valid_independent_demand_lifetime(target, remaining):
    async def provider(context):
        return None

    filing = Run(
        Together(
            "run",
            (
                Producer("source", (), provider, stop_when=target, remaining=remaining),
                Block("demand", ()),
            ),
        )
    )
    with pytest.raises(Rejected):
        Contract().accept(filing)


def test_stop_cannot_follow_a_repeat_block_invocation_or_discard_another_consumers_work():
    async def provider(context):
        return None

    body = Block("body", (Join("join", ("port",), (Output("value"),), lambda *args: True),))
    learning = Repeat("learning", (body,), lambda *args: Choice("body"), "policy")
    source = Producer("source", ("port",), provider, stop_when="body", remaining="discard")
    filing = Run(Together("run", (source, learning)), states=(OwnedState("policy", lambda: None),), channels=(Channel("port", 1),))
    with pytest.raises(Rejected, match="whole-scope lifetime"):
        Contract().accept(filing)
    other = replace(body, name="other")
    filing = replace(filing, root=Together("run", (replace(source, stop_when="learning"), learning, other)))
    with pytest.raises(Rejected, match="another live consumer"):
        Contract().accept(filing)


def test_granted_declared_write_outside_restored_unit_advances_source_evidence():
    from .examples import two_pass

    filing = two_pass()
    assert isinstance(filing.root, Block) and isinstance(filing.root.work[0], TwoPass)
    original = filing.root.work[0]

    def granted(access, models):
        original.implementation(access, models)
        with torch.no_grad():
            models["source"].get_parameter("weight").add_(1)

    region = replace(original, implementation=granted, uses=(*original.uses, Use("source", "write")))
    filing = replace(
        filing, root=replace(filing.root, work=(region, filing.root.work[1])), participants=(*filing.participants, scale("source", 1.0))
    )
    ready = run(execute(filing, contract=Contract(two_pass=True)))
    assert ready.state.stamp("source").numerical_epoch == 1


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_connected_shutdown_fences_reset_and_preserves_known_progress_on_cleanup_failure(cleanup_fails):
    async def check():
        cleaning, release = asyncio.Event(), asyncio.Event()
        filing = producer_continuity()
        assert isinstance(filing.root, Together) and isinstance(filing.root.children[0], Producer)
        source = filing.root.children[0]
        cleanup_error = RuntimeError("private cleanup completion unknown")

        async def held_cleanup(context):
            cleaning.set()
            await release.wait()
            assert source.cleanup is not None
            await source.cleanup(context)
            if cleanup_fails:
                raise cleanup_error

        root = replace(filing.root, children=(replace(source, cleanup=held_cleanup), filing.root.children[1]))
        ready = prepare(Contract().accept(replace(filing, root=root))).publish()
        state = ready.state
        owner = state.owners["encoding"]
        identity = state.stamp("encoder").participant_ref
        task = asyncio.create_task(Engine(ready).run())
        await cleaning.wait()
        assert state.image is not None and state.image.units["weights"].steps == 1
        assert state.stamp("encoder").numerical_epoch == 1
        assert state.active_producers == {"encoder-work": ("encoding",)} and not state.completed
        with pytest.raises(NotReady, match="finish cleanup"):
            await state.replace(Replacement("encoder", 1, Scale(3.0)), "encoder")
        assert state.owners["encoding"] is owner and state.generation == 1
        release.set()
        if cleanup_fails:
            with pytest.raises(ExceptionGroup) as failure:
                await task
            reported = [error for error in leaves(failure.value) if isinstance(error, ProducerFailed)]
            assert len(reported) == 1 and isinstance(reported[0].cause, asyncio.CancelledError)
            assert reported[0].cleanup_failures == (cleanup_error,) and not reported[0].implementation_returned
            assert state.unusable == {"encoder"} and state.unusable_states == {"encoding"}
            assert state.image.units["weights"].steps == 1  # No rollback of returned advancement.
            with pytest.raises(NotReady, match="usable"):
                state.capture()
        else:
            await task
            captured = state.capture().owner_states["encoding"]
            assert isinstance(captured, ContinuousSource) and captured.finished
        assert not state.active_producers
        await state.replace(Replacement("encoder", 1, Scale(3.0)), "encoder")
        assert state.owners["encoding"] is not owner and state.generation == 2
        assert state.stamp("encoder").participant_ref == identity
        assert not state.unusable and not state.unusable_states

    run(check())


def test_sibling_primary_failure_and_provider_cleanup_failure_both_survive_shutdown():
    async def check():
        started = asyncio.Event()

        async def provider(context):
            started.set()
            await asyncio.Event().wait()

        async def cleanup(context):
            raise RuntimeError("cleanup failed")

        async def fail(context):
            await started.wait()
            raise ValueError("primary work failed")

        filing = Run(
            Together(
                "run",
                (
                    Producer("source", (), provider, states=("owner",), cleanup=cleanup),
                    Block("work", (Call("fail", fail),)),
                ),
            ),
            states=(OwnedState("owner", list),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        with pytest.raises(BaseExceptionGroup) as failure:
            await Engine(ready).run()
        errors = leaves(failure.value)
        work = [error for error in errors if isinstance(error, WorkFailed)]
        production = [error for error in errors if isinstance(error, ProducerFailed)]
        assert len(work) == len(production) == 1
        assert str(work[0].cause) == "primary work failed"
        assert str(production[0].cleanup_failures[0]) == "cleanup failed"
        assert not ready.state.active_producers and not ready.state.completed

    run(check())


def test_finite_producer_retains_port_disposition_until_later_demand_end():
    async def provider(context):
        await context.publish("port", Product("", 1))
        await context.publish("port", Product("unused", 2))

    consume = Block("consume", (Join("join", ("port",), (Output("value", int),), lambda *args: True),))
    filing = Run(
        Together(
            "run",
            (
                Producer("source", ("port",), provider, stop_when="consume", remaining="discard"),
                consume,
            ),
        ),
        channels=(Channel("port", 2),),
    )
    ready = run(execute(filing))
    feed = ready.state.feeds["port"]
    assert [key for key, _, _ in feed.handed_off] == [""]
    assert list(feed.discarded) == [("unused", ())] and not feed.ready
    assert ready.state.capture().generation == 1


@pytest.mark.parametrize("producer_first", [False, True])
def test_already_ended_demand_does_not_start_new_private_work(producer_first):
    async def provider(context):
        pytest.fail("Private work began after its demand had ended")

    demand = Block("demand", ())
    source = Producer("source", (), provider, stop_when="demand", remaining="discard")
    filing = Run(Together("run", (source, demand) if producer_first else (demand, source)))
    ready = run(execute(filing))
    assert ("producer_not_started", "source") in ready.state.trace and not ready.state.active_producers


def test_primary_uncertainty_gates_other_owners_while_cleanup_is_still_running():
    async def check():
        cleaning, release = asyncio.Event(), asyncio.Event()
        primary = ValueError("source may have changed")

        async def provider(context):
            raise primary

        async def cleanup(context):
            cleaning.set()
            await release.wait()

        filing = Run(
            Producer("source", ("port",), provider, (Use("model"),), ("owner",), cleanup=cleanup),
            (scale("model", 1.0),),
            states=(OwnedState("owner", list),),
            channels=(Channel("port", 1),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        task = asyncio.create_task(Engine(ready).run())
        await cleaning.wait()
        assert ready.state.feeds["port"].failure is primary and ready.state.active_producers
        with pytest.raises(NotReady, match="unavailable"):
            async with ready.state.acquire((Use("model"),)):
                pytest.fail("Uncertain source was supplied during cleanup")
        release.set()
        with pytest.raises(ValueError, match="may have changed"):
            await task

    run(check())


def test_one_producer_can_follow_composite_demand_not_a_universal_training_loop():
    async def provider(context):
        await context.publish("A", Product("", 1))
        await context.publish("B", Product("", 2))
        await context.publish("A", Product("extra", 3))
        await asyncio.Event().wait()

    def consumer(name):
        return Block(name, (Join("join", (name,), (Output("value", int),), lambda *args: True),))

    filing = Run(
        Together(
            "run",
            (
                Producer("source", ("A", "B"), provider, stop_when="both", remaining="discard"),
                Together("both", (consumer("A"), consumer("B"))),
            ),
        ),
        channels=(Channel("A", 2), Channel("B", 1)),
    )
    ready = run(execute(filing))
    assert all(len(feed.handed_off) == 1 for feed in ready.state.feeds.values())
    assert list(ready.state.feeds["A"].discarded) == [("extra", ())]
    assert ("demand_ended", "both") in ready.state.trace and ready.state.capture().generation == 1


def test_mutually_waiting_whole_lifetimes_reject_before_loading(monkeypatch):
    async def finite(context):
        return None

    monkeypatch.setattr(Scale, "__init__", lambda *args: pytest.fail("loaded before known completion rejection"))
    filing = Run(
        Together(
            "root",
            (
                Together("left", (Producer("A", (), finite, stop_when="right", remaining="discard"),)),
                Together("right", (Producer("B", (), finite, stop_when="left", remaining="discard"),)),
            ),
        ),
        (scale("model", 1.0),),
    )
    with pytest.raises(Rejected, match="scope-completion wait cycle"):
        Contract().accept(filing)


def test_invalid_return_and_mode_cleanup_failure_preserve_both_results():
    cleanup = RuntimeError("restoration failed")

    class Fragile(Scale):
        def __setattr__(self, name, value):
            if name == "training" and value is True and getattr(self, "armed", False):
                raise cleanup
            super().__setattr__(name, value)

    model = Fragile(1.0)

    def selected(context):
        model.armed = True
        return "invalid"

    filing = Run(
        Block("work", (Call("call", selected, outputs=(Output("value", int),), uses=(Use("model"),), evaluation=True),)),
        (replace(scale("model", 1.0), load=lambda: model),),
    )
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(WorkFailed) as failure:
        run(Engine(ready).run())
    assert isinstance(failure.value.cause, TypeError) and failure.value.implementation_returned
    assert failure.value.cleanup_failures == (cleanup,) and ready.state.unusable == {"model"}


@pytest.mark.parametrize("planned", [False, True])
def test_cancellation_before_owner_entry_records_no_execution_or_cleanup_effects(monkeypatch, planned):
    async def check():
        attempted = asyncio.Event()

        async def never_started(context):
            pytest.fail("Selected work started while registration was blocked")

        source = Producer(
            "source", ("port",), never_started, states=("owner",), stop_when="demand", remaining="discard", cleanup=never_started
        )
        filing = Run(Together("run", (source, Block("demand", ()))), states=(OwnedState("owner", list),), channels=(Channel("port", 1),))
        ready = prepare(Contract().accept(filing)).publish()
        state = ready.state
        lowering = Lowering(state.accepted, state)
        original = state.producer_owner

        @asynccontextmanager
        async def contended(name, states):
            attempted.set()
            async with original(name, states) as owners:
                yield owners

        monkeypatch.setattr(state, "producer_owner", contended)
        # Unit-test the derived connection's arrival during owner entry; the
        # connected graph tests above cover actual whole-lifetime completion.
        async with state.condition:
            task = asyncio.create_task(lowering.producer(source)())
            await attempted.wait()
            if planned:
                lowering.finished["demand"].set()
                await task
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert not state.active_producers and not state.unusable_states and state.owners["owner"] == []
        assert ("producer_not_started", "source") in state.trace
        feed = state.feeds["port"]
        assert feed.closed and (feed.failure is None if planned else isinstance(feed.failure, asyncio.CancelledError))

    run(check())


def test_disabled_capability_cannot_be_selected_as_demand_lifetime():
    async def provider(context):
        return None

    loop = Repeat("loop", (Block("body", ()),), lambda *args: None, "policy")
    capability = Requested("capability", "loop", Block("observe", ()), lambda done: None, enabled=False)
    filing = Run(
        Together(
            "run",
            (
                Producer("source", (), provider, stop_when="capability", remaining="discard"),
                loop,
                capability,
            ),
        ),
        states=(OwnedState("policy", lambda: None),),
    )
    with pytest.raises(Rejected, match="active whole-scope lifetime"):
        Contract().accept(filing)


def test_inactive_body_does_not_request_unoffered_runtime_target_protection():
    loop = Repeat("loop", (Block("body", ()),), lambda *args: None, "policy")
    mixed = Block(
        "unused",
        (
            Call(
                "proposal",
                lambda context: pytest.fail("Dormant capability ran"),
                outputs=(Output("replacement", Replacement),),
                uses=(Use("model"),),
            ),
            Replace("replace", "replacement", "model"),
        ),
    )
    filing = Run(
        Together("run", (loop, Requested("dormant", "loop", mixed, lambda done: None, enabled=False))),
        (scale("model", 1.0),),
        states=(OwnedState("policy", lambda: None),),
    )
    ready = run(execute(filing))
    assert ready.state.completed and not ready.state.unusable
    assert isinstance(filing.root, Together) and isinstance(filing.root.children[1], Requested)
    active = replace(filing.root.children[1], enabled=True)
    with pytest.raises(Rejected, match="outside protected"):
        Contract().accept(replace(filing, root=Together("run", (loop, active))))


def test_granted_return_is_not_verified_handback_or_returned_effect_evidence():
    from .examples import two_pass

    filing = two_pass()
    assert isinstance(filing.root, Block) and isinstance(filing.root.work[0], TwoPass)
    region = replace(filing.root.work[0], implementation=lambda *args: None)
    filing = replace(filing, root=replace(filing.root, work=(region, filing.root.work[1])))
    ready = prepare(Contract(two_pass=True).accept(filing)).publish()
    with pytest.raises(WorkFailed, match="Incomplete") as failure:
        run(Engine(ready).run())
    assert failure.value.implementation_returned and ready.state.stamp("model").numerical_epoch == 0
    assert ready.state.image is not None and ready.state.image.units["weights"].steps == 0
    assert ready.state.unusable == {"model"}
