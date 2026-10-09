"""Connected candidate tests; these do not certify production G5 support."""

from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import Hashable
from dataclasses import replace
from typing import cast

import pytest
import torch
from torch import nn

from .examples import (
    Conditioning,
    LearningState,
    Scale,
    SourceState,
    ValidationState,
    adversarial,
    correlated,
    diffusion,
    execute,
    joint,
    replacement,
    scale,
    two_pass,
)
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
    OutOfBounds,
    Output,
    OwnedState,
    Participant,
    Producer,
    Rejected,
    Repeat,
    Requested,
    Run,
    Scope,
    Together,
    TwoPass,
    Unit,
    Use,
    Work,
)
from .prepare import Engine, ProviderContext, WorkFailed, prepare
from .state import Feed, Product, Projection, Replacement, RunState, join


def run(awaitable):
    async def bounded():
        return await asyncio.wait_for(awaitable, 5)

    return asyncio.run(bounded())


def leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in leaves(child)]
    return [error]


def test_same_engine_runs_different_whole_run_compositions():
    async def check():
        ordinary = await execute(diffusion())
        alternating = await execute(adversarial())
        assert ordinary.state.image is not None and alternating.state.image is not None
        assert ordinary.state.coordinates == {"learning": {"learn": 3}}
        assert alternating.state.coordinates == {"alternating": {"D": 3, "G": 3}}
        assert ordinary.state.image.units["weights"].steps == 3
        assert {key: unit.steps for key, unit in alternating.state.image.units.items()} == {"D-unit": 3, "G-unit": 3}
        assert not ordinary.state.image.bindings["encoder"].model.get_parameter("weight").requires_grad
        pixel_state, caption_state = ordinary.state.owners["pixels"], ordinary.state.owners["captions"]
        assert isinstance(pixel_state, SourceState) and isinstance(caption_state, SourceState)
        assert pixel_state.finished and caption_state.finished
        assert "TaskGroup" in ordinary.source["inputs"]
        assert ordinary.source["learn"].index("'observe'") < ordinary.source["learn"].index("'backward'")

    run(check())


def test_dynamic_feedback_and_requested_validation_survive_authoring():
    ready = run(execute(diffusion()))
    adaptive = ready.state.owners["adaptive"]
    assert isinstance(adaptive, LearningState)
    assert len(adaptive.observations) == 3
    assert adaptive.factors_used == [1.0, 1 / (1 + adaptive.observations[0]), 1 / (1 + adaptive.observations[1])]
    validation = ready.state.owners["validation"]
    assert isinstance(validation, ValidationState) and validation.results[0][0] == 2
    assert ready.state.image.bindings["denoiser"].model.training
    assert run(execute(diffusion(validation=False))).state.owners["validation"].results == []


def test_authoring_object_is_not_retained_and_hot_path_does_not_rewalk_graph(monkeypatch):
    filing = joint()
    authored = weakref.ref(filing)
    accepted = Contract().accept(filing)
    ready = prepare(accepted).publish()
    del filing
    gc.collect()
    assert authored() is None
    from . import prepare as preparation_module

    monkeypatch.setattr(preparation_module, "ordered", lambda _: pytest.fail("hot path traversed static wiring"))
    run(Engine(ready).run())
    assert ready.state.image is not None
    assert ready.state.image.units["A"].steps == ready.state.image.units["B"].steps == 1


def test_relationships_not_authored_tuple_order_determine_execution():
    seen = []
    first = Call("first", lambda context: seen.append("first"))
    second = Call("second", lambda context: seen.append("second"), after=("first",))
    run(execute(Run(Block("work", (second, first)))))
    assert seen == ["first", "second"]
    seen.clear()
    run(execute(Run(Block("work", (replace(first, after=("second",)), replace(second, after=()))))))
    assert seen == ["second", "first"]


def test_accepted_does_not_claim_readiness_and_publication_is_generation_checked():
    accepted = Contract().accept(joint())
    state = RunState(accepted)
    attempt = prepare(accepted, state)
    stale = prepare(accepted, state)
    assert state.image is None and attempt.image.generation == 1
    attempt.publish()
    with pytest.raises(NotReady, match="Stale"):
        stale.publish()
    assert state.generation == 1
    with pytest.raises(NotReady, match="replacement path"):
        prepare(accepted, state)


def test_failed_initial_preparation_publishes_nothing():
    good = scale("good", 0.5)

    def broken():
        raise RuntimeError("loader failed")

    filing = Run(Block("idle", ()), (good, Participant("bad", broken, lambda model: None)))
    state = RunState(Contract().accept(filing))
    with pytest.raises(RuntimeError, match="loader failed"):
        prepare(state.accepted, state)
    assert state.image is None
    with pytest.raises(NotReady, match="Incomplete"):
        state.candidate({"good": Scale(0.5)})


def test_final_membership_checks_aliases_without_inventing_component_provenance():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.alias = self.weight

    participant = Participant("learned-state", Tied, lambda model: None)
    one = Unit("one", (("learned-state", "weight"), ("learned-state", "alias")))
    state = prepare(Contract().accept(Run(Block("idle", ()), (participant,), (one,)))).publish().state
    assert state.image is not None
    assert len(state.image.units["one"].parameters) == 1
    two = Unit("two", (("learned-state", "alias"),))
    with pytest.raises(NotReady, match="Shared physical member"):
        prepare(Contract().accept(Run(Block("idle", ()), (participant,), (one, two))))
    with pytest.raises(NotReady, match="unresolved accepted member"):
        prepare(Contract().accept(Run(Block("idle", ()), (participant,), (Unit("bad", (("learned-state", "missing"),)),))))


def test_out_of_order_inputs_keep_work_identity_and_invocation_correlation():
    ready = run(execute(diffusion()))
    state = ready.state
    assert state.owners["pixels"].produced == [2, 1, 3]
    images = list(state.feeds["image"].handed_off)
    captions = list(state.feeds["conditioning"].handed_off)
    assert [record[0] for record in images] == [1, 2, 3]
    assert [record[:2] for record in images] == [record[:2] for record in captions]
    assert all(record[2][0].participant_ref == state.accepted.participant_refs["encoder"] for record in captions)


def test_join_checks_every_member_before_claiming_any():
    async def check():
        changed = asyncio.Condition()
        a, b = Feed(2, changed), Feed(2, changed)
        await a.publish(Product("wanted", "correct"))
        await b.publish(Product("wanted", "wrong"))
        with pytest.raises(NotReady, match="Inadmissible"):
            await join((a, b), "wanted", 1, lambda product, view: product.value == "correct", Projection({}, {}))
        assert "wanted" in a.ready and "wanted" in b.ready
        assert not a.handed_off and not b.handed_off
        with pytest.raises(OutOfBounds, match="Duplicate"):
            await b.publish(Product("wanted", "correct"))

    run(check())


def test_provider_progress_and_backpressure_are_not_a_blocking_read_hidden_in_work():
    async def check():
        progressed, blocked = asyncio.Event(), asyncio.Event()

        async def produce(context: ProviderContext):
            await context.publish("port", Product(1, 10))
            progressed.set()
            await context.publish("port", Product(2, 20))
            blocked.set()

        filing = Run(Producer("producer", ("port",), produce), channels=(Channel("port", 1),))
        ready = prepare(Contract().accept(filing)).publish()
        task = asyncio.create_task(Engine(ready).run())
        await progressed.wait()
        assert not blocked.is_set() and ready.state.feeds["port"].high_water == 1
        assert await join((ready.state.feeds["port"],), 1, 1, lambda *args: True, Projection({}, {})) == (10,)
        await task
        assert blocked.is_set() and ready.state.feeds["port"].high_water == 1
        with pytest.raises(NotReady, match="unfinished provider"):
            ready.state.capture()

    run(check())


def test_late_product_keeps_actual_source_provenance_after_compatible_replacement():
    async def check():
        encoded, release = asyncio.Event(), asyncio.Event()

        async def producer(context: ProviderContext):
            async with context.use_source() as source:
                with torch.no_grad():
                    value = source.models["encoder"](torch.tensor(1.0)).detach().clone()
                actual = (source.stamps["encoder"],)
            encoded.set()
            await release.wait()
            await context.publish("conditioning", Product(1, Conditioning(1, 1, value), actual))

        filing = Run(
            Producer("producer", ("conditioning",), producer, (Use("encoder"),)),
            (scale("encoder", 1.0),),
            channels=(Channel("conditioning", 1),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        task = asyncio.create_task(Engine(ready).run())
        await encoded.wait()
        await ready.state.replace(Replacement("encoder", 1, Scale(2.0)), "encoder")
        release.set()
        await task
        product = ready.state.feeds["conditioning"].ready[1]
        assert product.provenance[0].binding == 1 and ready.state.stamp("encoder").binding == 2
        async with ready.state.acquire((Use("encoder"),)) as current:
            with pytest.raises(NotReady, match="actual provenance retained"):
                await join((ready.state.feeds["conditioning"],), 1, 1, correlated, current)
        assert not ready.state.unusable

    run(check())


def test_trainable_source_numerical_progress_does_not_revise_binding_or_prepare_graph():
    ready = prepare(Contract().accept(joint())).publish()
    before = ready.state.stamp("a")
    run(Engine(ready).run())
    after = ready.state.stamp("a")
    assert (after.participant_ref, after.binding) == (before.participant_ref, before.binding)
    assert after.numerical_epoch == before.numerical_epoch + 1 and ready.state.generation == 1


def test_compatible_replacement_preserves_identity_and_refreshes_affected_runtime():
    ready = run(execute(replacement()))
    stamp = ready.state.stamp("model")
    assert stamp.participant_ref == ready.state.accepted.participant_refs["model"]
    assert stamp.binding == 2 and stamp.numerical_epoch == 2 and ready.state.generation == 2
    assert ready.state.coordinates == {"stages": {"before": 1, "change": 1, "after": 1}}
    # Stable semantic unit progress survives the physical optimizer reset.
    assert ready.state.image.units["weights"].ref == ready.state.accepted.unit_refs["weights"]
    assert ready.state.image.units["weights"].steps == 2
    assert len(ready.state.image.units["weights"].outcomes) == 2


def test_replacement_only_preserves_old_publication_on_failed_compatibility():
    async def check():
        filing = Run(Block("idle", ()), (scale("model", 1.0),), states=(OwnedState("owner", lambda: [1], ("model",), "reset"),))
        ready = prepare(Contract().accept(filing)).publish()
        old_image, old_owner = ready.state.image, ready.state.owners["owner"]
        assert old_image is not None
        with pytest.raises(ValueError, match="incompatible"):
            await ready.state.replace(Replacement("model", 1, nn.Linear(1, 1)), "model")
        assert ready.state.image is old_image and ready.state.owners["owner"] is old_owner
        with pytest.raises(NotReady, match="destructive"):
            await ready.state.replace(Replacement("model", 1, old_image.bindings["model"].model), "model")
        await ready.state.replace(Replacement("model", 1, Scale(2.0)), "model")
        assert ready.state.owners["owner"] == [1] and ready.state.owners["owner"] is not old_owner

    run(check())


def test_joint_gradient_work_has_separate_non_atomic_optimizer_outcomes(monkeypatch):
    ready = prepare(Contract().accept(joint())).publish()
    state = ready.state
    assert state.image is not None
    original = state.image.units["B"].optimizer.step

    def changed_then_failed():
        original()
        raise RuntimeError("B completion unknown")

    monkeypatch.setattr(state.image.units["B"].optimizer, "step", changed_then_failed)
    with pytest.raises(WorkFailed) as failure:
        run(Engine(ready).run())
    outcomes = {outcome.unit_ref: outcome for outcome in failure.value.unit_outcomes}
    a = outcomes[state.accepted.unit_refs["A"]]
    b = outcomes[state.accepted.unit_refs["B"]]
    assert (a.contribution, a.optimizer, a.reset) == ("ready", "returned", "returned")
    assert (b.contribution, b.optimizer, b.reset) == ("ready", "uncertain", "not_attempted")
    assert state.image.units["A"].steps == 1 and state.image.units["B"].steps == 0
    assert state.unusable == {"a", "b"} and not state.writers
    with pytest.raises(NotReady, match="usable"):
        state.capture()
    with pytest.raises(NotReady, match="unavailable"):
        run(Engine(ready).run())


def test_returned_optimizer_does_not_claim_numerical_parameter_change():
    filing = joint()
    filing = replace(filing, units=tuple(replace(unit, learning_rate=0.0) for unit in filing.units))
    ready = prepare(Contract().accept(filing)).publish()
    assert ready.state.image is not None
    original = {name: binding.model.get_parameter("weight").detach().clone() for name, binding in ready.state.image.bindings.items()}
    run(Engine(ready).run())
    assert all(
        torch.equal(original[name], binding.model.get_parameter("weight").detach()) for name, binding in ready.state.image.bindings.items()
    )
    assert all(unit.outcomes[0].optimizer == "returned" for unit in ready.state.image.units.values())


def test_two_pass_requires_grant_and_retains_exactly_one_final_advancement():
    with pytest.raises(Rejected, match="not offered"):
        Contract().accept(two_pass())
    ready = run(execute(two_pass(), contract=Contract(two_pass=True)))
    outcome = ready.state.image.units["weights"].outcomes[0]
    assert (outcome.contribution, outcome.optimizer, outcome.reset) == ("granted_ready", "returned", "returned")
    assert ready.state.image.units["weights"].steps == 1
    filing = two_pass()
    assert isinstance(filing.root, Block)
    extra = Differentiate("also-standard-backward", "handback", ("weights",), "extra")
    with pytest.raises(Rejected, match="competing backward ownership"):
        Contract(two_pass=True).accept(replace(filing, root=replace(filing.root, work=(*filing.root.work, extra))))


def test_restored_weights_are_not_successful_handback_if_selected_region_raises():
    filing = two_pass()
    assert isinstance(filing.root, Block)
    region = filing.root.work[0]
    assert isinstance(region, TwoPass)
    original = region.implementation

    def failed(*args):
        original(*args)
        raise RuntimeError("failed after restoring temporary values")

    filing = replace(filing, root=replace(filing.root, work=(replace(region, implementation=failed), filing.root.work[1])))
    ready = prepare(Contract(two_pass=True).accept(filing)).publish()
    assert ready.state.image is not None
    before = ready.state.image.bindings["model"].model.get_parameter("weight").detach().clone()
    with pytest.raises(WorkFailed, match="failed after restoring"):
        run(Engine(ready).run())
    assert torch.equal(before, ready.state.image.bindings["model"].model.get_parameter("weight").detach())
    assert ready.state.image.units["weights"].steps == 0 and ready.state.unusable == {"model"}
    outcome = ready.state.image.units["weights"].outcomes[0]
    assert (outcome.contribution, outcome.optimizer) == ("unverified_grant", "not_attempted")


def test_protected_current_use_covers_suspended_granted_region_and_handback():
    async def check():
        perturbed, release, inspected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        filing = two_pass()
        assert isinstance(filing.root, Block)
        region = filing.root.work[0]

        async def suspended(access, models):
            access.backward((models["model"](torch.tensor(2.0)) - 1).square())
            access.perturb(0.1)
            perturbed.set()
            await release.wait()
            access.reset()
            access.backward((models["model"](torch.tensor(2.0)) - 1).square())
            access.restore()

        filing = replace(filing, root=replace(filing.root, work=(replace(region, implementation=suspended), filing.root.work[1])))
        ready = prepare(Contract(two_pass=True).accept(filing)).publish()
        task = asyncio.create_task(Engine(ready).run())
        await perturbed.wait()

        async def reader():
            async with ready.state.acquire((Use("model"),)):
                inspected.set()

        waiter = asyncio.create_task(reader())
        await asyncio.sleep(0)
        assert not inspected.is_set()
        release.set()
        await task
        await waiter
        assert inspected.is_set()

    run(check())


def test_bad_return_after_selected_state_effect_is_not_a_preflight_rejection():
    def wrong(context):
        context.states["state"].append("effect happened")
        return "not an integer"

    filing = Run(
        Block("work", (Call("bad", wrong, outputs=(Output("value", int),), states=("state",)),)), states=(OwnedState("state", list),)
    )
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(WorkFailed) as failure:
        run(Engine(ready).run())
    assert isinstance(failure.value.cause, TypeError)
    assert ready.state.owners["state"] == ["effect happened"]
    assert ready.state.unusable_states == {"state"}


def test_failure_cancels_independent_provider_and_preserves_its_cleanup():
    async def check():
        started, closed = asyncio.Event(), asyncio.Event()

        async def provider(context):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def fail(context):
            await started.wait()
            raise RuntimeError("selected work failed")

        filing = Run(
            Together("run", (Producer("producer", ("port",), provider), Block("work", (Call("fail", fail),)))),
            channels=(Channel("port", 1),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        with pytest.raises(ExceptionGroup) as failure:
            await Engine(ready).run()
        assert len(leaves(failure.value)) == 1 and isinstance(leaves(failure.value)[0], WorkFailed)
        assert closed.is_set() and ready.state.feeds["port"].closed and not ready.state.running

    run(check())


def test_completed_capture_preserves_distinct_owner_positions_not_just_model_weights():
    ready = run(execute(diffusion()))
    capture = ready.state.capture()
    assert capture.authority == ready.state.accepted.authority
    assert capture.coordinates == {"learning": {"learn": 3}}
    assert capture.source_stamps["denoiser"].numerical_epoch == 3
    assert capture.source_stamps["encoder"].numerical_epoch == 0
    assert capture.optimization["weights"]["steps"] == 3
    assert capture.owner_states["captions"].finished and capture.owner_states["validation"].results
    assert capture.owner_states["adaptive"] is not ready.state.owners["adaptive"]


def test_runtime_choices_are_bounded_and_not_an_authoring_callback():
    filing = Run(
        Repeat("loop", (Block("allowed", ()),), lambda counts, state: Choice("unaccepted"), "policy"),
        states=(OwnedState("policy", lambda: None),),
    )
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(OutOfBounds, match="unaccepted alternative"):
        run(Engine(ready).run())
    assert not ready.state.trace and not ready.state.unusable


def test_ordinary_stateful_work_needs_no_input_loss_or_optimizer_placeholders():
    filing = Run(
        Block("work", (Call("update", lambda context: context.states["counter"].append(1), states=("counter",)),)),
        states=(OwnedState("counter", list),),
    )
    ready = run(execute(filing))
    assert ready.state.owners["counter"] == [1] and not ready.state.image.units and not ready.state.feeds


def test_cpu_target_rejects_non_cpu_concrete_state_without_publishing():
    filing = Run(Block("idle", ()), (Participant("model", lambda: nn.Linear(1, 1, device="meta"), lambda model: None),))
    state = RunState(Contract().accept(filing))
    with pytest.raises(NotReady, match="CPU tensors only"):
        prepare(state.accepted, state)
    assert state.image is None


def test_in_flight_result_from_trainable_encoder_is_not_relabelled_after_update():
    async def check():
        encoded, publish_late = asyncio.Event(), asyncio.Event()

        async def provider(context):
            async with context.use_source() as source:
                with torch.no_grad():
                    value = source.models["encoder"](torch.tensor(1.0)).detach().clone()
                actual = (source.stamps["encoder"],)
            encoded.set()
            await publish_late.wait()
            await context.publish("conditioning", Product(1, Conditioning(1, 1, value), actual))

        async def wait_for_encoding(context):
            await encoded.wait()

        def objective(context):
            return (context.models["encoder"](torch.tensor(2.0)) - 1).square()

        def policy(counts, state):
            if not counts.get("wait", 0):
                return Choice("wait")
            if not counts.get("update", 0):
                return Choice("update")
            return None

        waiting = Block("wait", (Call("await-source", wait_for_encoding),))
        updating = Block(
            "update",
            (
                Call("objective", objective, outputs=(Output("loss", torch.Tensor),), uses=(Use("encoder"),)),
                Differentiate("backward", "loss", ("weights",), "gradient"),
                Advance("advance", "gradient", ("weights",)),
                Call("allow-publication", lambda context: publish_late.set(), after=("advance",)),
            ),
        )
        filing = Run(
            Together(
                "run",
                (
                    Producer("encoding", ("conditioning",), provider, (Use("encoder"),)),
                    Repeat("learning", (waiting, updating), policy, "policy"),
                ),
            ),
            (scale("encoder", 1.0),),
            (Unit("weights", (("encoder", "weight"),)),),
            (OwnedState("policy", lambda: None),),
            (Channel("conditioning", 1),),
        )
        ready = await execute(filing)
        product = ready.state.feeds["conditioning"].ready[1]
        assert product.provenance[0].binding == ready.state.stamp("encoder").binding == 1
        assert product.provenance[0].numerical_epoch == 0 and ready.state.stamp("encoder").numerical_epoch == 1
        async with ready.state.acquire((Use("encoder"),)) as view:
            with pytest.raises(NotReady, match="Inadmissible"):
                await join((ready.state.feeds["conditioning"],), 1, 1, correlated, view)

    run(check())


def test_input_rejection_in_connected_execution_precedes_numerical_effects():
    async def check():
        async def provider(context):
            await context.publish("input", Product(1, "wrong"))

        batch = Block("batch", (Join("admit", ("input",), (Output("value", str),), lambda product, view: False, (Use("model"),)),))
        filing = Run(
            Together(
                "run", (Producer("source", ("input",), provider), Repeat("learning", (batch,), lambda *args: Choice("batch", 1), "policy"))
            ),
            (scale("model", 1.0),),
            states=(OwnedState("policy", lambda: None),),
            channels=(Channel("input", 1),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        with pytest.raises(ExceptionGroup) as failure:
            await Engine(ready).run()
        assert all(isinstance(error, NotReady) for error in leaves(failure.value))
        assert not ready.state.unusable and not ready.state.feeds["input"].handed_off

    run(check())


def test_requested_completion_uses_source_invocation_not_latest_unrelated_invocation():
    async def check():
        suspended, resume = asyncio.Event(), asyncio.Event()
        observed = []

        async def source(context):
            suspended.set()
            await resume.wait()

        async def unrelated(context):
            await suspended.wait()
            resume.set()

        def policy(counts, state):
            return None if counts else Choice("source")

        filing = Run(
            Together(
                "run",
                (
                    Repeat("loop", (Block("source", (Call("source-work", source),)),), policy, "policy"),
                    Block("unrelated", (Call("unrelated-work", unrelated),)),
                    Requested("observe", "loop", Block("observation", ()), lambda done: observed.append(done)),
                ),
            ),
            states=(OwnedState("policy", lambda: None),),
        )
        ready = await execute(filing)
        entries = {payload[1]: payload[3] for kind, payload in ready.state.trace if kind == "invocation" and isinstance(payload, tuple)}
        assert observed[0].invocation == entries["source"] != entries["unrelated"]

    run(check())


def test_provider_continuation_state_cannot_be_accidentally_shared_with_call_owner():
    async def provider(context):
        return None

    filing = Run(
        Together(
            "run",
            (
                Producer("source", (), provider, states=("shared",)),
                Block("work", (Call("call", lambda context: None, states=("shared",)),)),
            ),
        ),
        states=(OwnedState("shared", list),),
    )
    with pytest.raises(Rejected, match="separately owned provider"):
        Contract().accept(filing)


def test_later_admission_uses_current_state_after_same_block_declared_write():
    async def check():
        async def source(context):
            return None

        def edit(context):
            with torch.no_grad():
                context.models["encoder"].get_parameter("weight").add_(1)

        block = Block(
            "consume",
            (
                Call("edit", edit, uses=(Use("encoder", "write"),)),
                Join("join", ("conditioning",), (Output("condition", Conditioning),), correlated, (Use("encoder"),), after=("edit",)),
            ),
        )
        filing = Run(
            Together(
                "run",
                (Producer("source", ("conditioning",), source), Repeat("loop", (block,), lambda *args: Choice("consume", 1), "policy")),
            ),
            (scale("encoder", 1.0),),
            states=(OwnedState("policy", lambda: None),),
            channels=(Channel("conditioning", 1),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        await ready.state.feeds["conditioning"].publish(Product(1, Conditioning(1, 1, torch.tensor(1.0)), (ready.state.stamp("encoder"),)))
        with pytest.raises(ExceptionGroup) as failure:
            await Engine(ready).run()
        assert isinstance(leaves(failure.value)[0], NotReady)
        assert ready.state.stamp("encoder").numerical_epoch == 1
        assert not ready.state.feeds["conditioning"].handed_off

    run(check())


def test_input_readiness_wait_does_not_hold_write_lease_needed_by_its_producer():
    async def check():
        async def producer(context):
            async with context.use_source() as view:
                value = view.models["model"](torch.tensor(1.0)).detach().clone()
            await context.publish("input", Product("", value))

        def objective(context, value):
            return (context.models["model"](value) - 1).square()

        block = Block(
            "learn",
            (
                Join("join", ("input",), (Output("value", torch.Tensor),), lambda *args: True),
                Call("objective", objective, ("value",), (Output("loss", torch.Tensor),), (Use("model"),)),
                Differentiate("backward", "loss", ("weights",), "gradient"),
                Advance("advance", "gradient", ("weights",)),
            ),
        )
        # Consumer starts first: it must not take its write lease while waiting
        # for a producer needing read access to the same participant.
        filing = Run(
            Together("run", (block, Producer("source", ("input",), producer, (Use("model"),)))),
            (scale("model", 0.8),),
            (Unit("weights", (("model", "weight"),)),),
            channels=(Channel("input", 1),),
        )
        ready = await execute(filing)
        assert ready.state.image is not None and ready.state.image.units["weights"].steps == 1
        assert len(ready.state.feeds["input"].handed_off) == 1

    run(check())


def test_later_source_wait_with_conflicting_lease_rejects_before_loading(monkeypatch):
    async def provider(context):
        return None

    block = Block(
        "consumer",
        (
            Call("edit", lambda context: None, uses=(Use("model", "write"),)),
            Join("join", ("input",), (Output("value"),), lambda *args: True, after=("edit",)),
        ),
    )
    filing = Run(
        Together("run", (block, Producer("source", ("input",), provider, (Use("model"),)))),
        (scale("model", 0.8),),
        channels=(Channel("input", 1),),
    )
    monkeypatch.setattr(Scale, "__init__", lambda *args: pytest.fail("loaded before known target rejection"))
    with pytest.raises(Rejected, match="lease/source wait cycle"):
        Contract().accept(filing)


def test_readiness_race_does_not_wait_again_under_protected_admission():
    async def check():
        feed = Feed(1, asyncio.Condition())
        # Equivalent to another consumer claiming a previously ready product.
        with pytest.raises(NotReady, match="lost readiness"):
            await join((feed,), 1, 1, lambda *args: True, Projection({}, {}), wait=False)

    run(check())


def test_contribution_carries_correspondence_evidence_not_optimizer_services():
    ready = prepare(Contract().accept(joint())).publish()
    state = ready.state
    assert state.image is not None
    a, b = (state.image.bindings[name].model for name in ("a", "b"))
    loss = (a(torch.tensor(1.0)) + b(torch.tensor(1.0))).square()
    contribution = state.contribute(loss, ("A", "B"), 1)
    assert all(type(key) is object for key in contribution.prepared_keys)
    with pytest.raises(NotReady, match="unverified gradient handback"):
        state.advance(replace(contribution, prepared_keys=(object(), object())), ("A", "B"), 1)
    assert state.image.units["A"].steps == state.image.units["B"].steps == 0


def test_policy_output_rejection_withholds_mutated_policy_state():
    def bad_policy(counts, state):
        state.append("changed")
        return Choice("unaccepted")

    filing = Run(Repeat("loop", (Block("allowed", ()),), bad_policy, "policy"), states=(OwnedState("policy", list),))
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(OutOfBounds, match="unaccepted alternative"):
        run(Engine(ready).run())
    assert ready.state.owners["policy"] == ["changed"] and ready.state.unusable_states == {"policy"}


def test_insufficient_reorder_capacity_fails_explicitly_without_eviction_or_hang():
    async def check():
        async def provider(context):
            await context.publish("port", Product(2, "two"))
            await context.publish("port", Product(1, "one"))

        consume = Block("consume", (Join("join", ("port",), (Output("value", str),), lambda *args: True),))
        filing = Run(
            Together(
                "run", (Producer("source", ("port",), provider), Repeat("loop", (consume,), lambda *args: Choice("consume", 1), "policy"))
            ),
            states=(OwnedState("policy", lambda: None),),
            channels=(Channel("port", 1),),
        )
        ready = prepare(Contract().accept(filing)).publish()
        with pytest.raises(ExceptionGroup) as failure:
            await Engine(ready).run()
        assert any(isinstance(error, NotReady) and "reorder capacity" in str(error) for error in leaves(failure.value))
        assert ready.state.feeds["port"].ready[2].value == "two"
        assert not ready.state.feeds["port"].handed_off and ready.state.feeds["port"].closed

    run(check())


def test_acceptance_rejects_malformed_kinds_before_loading_or_lowering():
    wrong = cast(Block, Together("not-a-block", ()))
    filing = Run(Repeat("loop", (wrong,), lambda *args: None, "policy"), states=(OwnedState("policy", lambda: None),))
    with pytest.raises(Rejected, match="Block alternatives"):
        Contract().accept(filing)
    filing = replace(
        filing,
        root=Together(
            "run",
            (Repeat("loop", (Block("okay", ()),), lambda *args: None, "policy"), Requested("capability", "loop", wrong, lambda done: None)),
        ),
    )
    with pytest.raises(Rejected, match="Block capability body"):
        Contract().accept(filing)
    with pytest.raises(Rejected, match="unknown work kind"):
        Contract().accept(Run(Block("bad", (cast(Work, object()),))))
    with pytest.raises(Rejected, match="Unknown scope kind"):
        Contract().accept(Run(cast(Scope, object())))


def test_declared_unowned_channel_rejects_instead_of_faking_unfinished_capture():
    with pytest.raises(Rejected, match="no owning producer"):
        Contract().accept(Run(Block("idle", ()), channels=(Channel("dangling", 1),)))


def test_requested_runtime_authority_violation_is_not_authored_rejection():
    async def check():
        loop = Repeat("loop", (Block("source", ()),), lambda counts, state: None if counts else Choice("source"), "policy")
        request = Requested("capability", "loop", Block("provided", ()), lambda done: Choice("unprovided"))
        filing = Run(Together("run", (loop, request)), states=(OwnedState("policy", lambda: None),))
        ready = prepare(Contract().accept(filing)).publish()
        with pytest.raises(ExceptionGroup) as failure:
            await Engine(ready).run()
        assert any(isinstance(error, OutOfBounds) for error in leaves(failure.value))
        assert not any(isinstance(error, Rejected) for error in leaves(failure.value))

    run(check())


def test_correlated_ports_with_unshared_condition_reject_before_waiting():
    async def check():
        a, b = Feed(1, asyncio.Condition()), Feed(1, asyncio.Condition())
        with pytest.raises(NotReady, match="share one handoff condition"):
            await join((a, b), 1, 1, lambda *args: True, Projection({}, {}))

    run(check())


def test_invalid_granted_restore_phase_stops_before_final_optimizer():
    filing = two_pass()
    assert isinstance(filing.root, Block) and isinstance(filing.root.work[0], TwoPass)
    region = replace(filing.root.work[0], implementation=lambda access, models: access.restore())
    filing = replace(filing, root=replace(filing.root, work=(region, filing.root.work[1])))
    ready = prepare(Contract(two_pass=True).accept(filing)).publish()
    with pytest.raises(WorkFailed, match="Unsupported restoration phase"):
        run(Engine(ready).run())
    assert ready.state.image is not None and ready.state.image.units["weights"].steps == 0


def test_shared_payload_cannot_hide_behind_two_participant_addresses():
    a, b = Scale(0.8), Scale(0.8)
    b.weight = a.weight
    filing = Run(
        Block("idle", ()),
        (Participant("a", lambda: a, lambda model: None), Participant("b", lambda: b, lambda model: None)),
        (Unit("weights", (("a", "weight"),)),),
    )
    state = RunState(Contract().accept(filing))
    with pytest.raises(NotReady, match="relationship-aware protection"):
        prepare(state.accepted, state)
    assert state.image is None


def test_distinct_parameter_objects_with_shared_storage_are_not_disjoint_units():
    class Views(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(1.0))
            self.b = nn.Parameter(self.a.detach())

    filing = Run(
        Block("idle", ()), (Participant("model", Views, lambda model: None),), (Unit("A", (("model", "a"),)), Unit("B", (("model", "b"),)))
    )
    with pytest.raises(NotReady, match="unimplemented alias mapping"):
        prepare(Contract().accept(filing))


def test_replacement_wrapper_borrowing_current_parameter_is_not_fresh_preparation():
    async def check():
        ready = prepare(Contract().accept(Run(Block("idle", ()), (scale("model", 0.8),)))).publish()
        assert ready.state.image is not None
        old = ready.state.image
        wrapper = Scale(1.0)
        wrapper.weight = old.bindings["model"].model.get_parameter("weight")
        with pytest.raises(NotReady, match="fresh realization"):
            await ready.state.replace(Replacement("model", 1, wrapper), "model")
        assert ready.state.image is old and not wrapper.weight.requires_grad

    run(check())


def test_indirect_lease_source_wait_cycle_rejects_before_loading(monkeypatch):
    async def source(context):
        return None

    def waiting(name, participant, channel):
        return Block(
            name,
            (
                Call("edit", lambda context: None, uses=(Use(participant, "write"),)),
                Join("join", (channel,), (Output("value"),), lambda *args: True, after=("edit",)),
            ),
        )

    filing = Run(
        Together(
            "run",
            (
                waiting("B1", "X", "F1"),
                Producer("P1", ("F1",), source, (Use("Y"),)),
                waiting("B2", "Y", "F2"),
                Producer("P2", ("F2",), source, (Use("X"),)),
            ),
        ),
        (scale("X", 1.0), scale("Y", 2.0)),
        channels=(Channel("F1", 1), Channel("F2", 1)),
    )
    monkeypatch.setattr(Scale, "__init__", lambda *args: pytest.fail("loaded before known target rejection"))
    with pytest.raises(Rejected, match="lease/source wait cycle"):
        Contract().accept(filing)


def test_acyclic_later_source_wait_still_executes():
    async def producer(context):
        async with context.use_source() as current:
            value = current.models["Y"](torch.tensor(1.0)).detach().clone()
        await context.publish("input", Product("", value))

    block = Block(
        "consumer",
        (
            Call("edit-X", lambda context: None, uses=(Use("X", "write"),)),
            Join("join", ("input",), (Output("value", torch.Tensor),), lambda *args: True, after=("edit-X",)),
        ),
    )
    filing = Run(
        Together("run", (block, Producer("source", ("input",), producer, (Use("Y"),)))),
        (scale("X", 1.0), scale("Y", 2.0)),
        channels=(Channel("input", 1),),
    )
    ready = run(execute(filing))
    assert len(ready.state.feeds["input"].handed_off) == 1


@pytest.mark.parametrize("bad", [object(), Choice("provided", cast(Hashable, []))])
def test_request_policy_checks_type_and_input_identity_before_dispatch(bad):
    async def check():
        loop = Repeat("loop", (Block("source", ()),), lambda counts, state: None if counts else Choice("source"), "policy")
        request = Requested("request", "loop", Block("provided", ()), lambda done: cast(Choice, bad))
        filing = Run(Together("run", (loop, request)), states=(OwnedState("policy", lambda: None),))
        with pytest.raises(ExceptionGroup) as failure:
            await execute(filing)
        assert all(isinstance(error, OutOfBounds) for error in leaves(failure.value))

    run(check())


def test_repeat_policy_checks_input_identity_before_prefix_readiness():
    def bad(counts, state):
        state.append("changed")
        return Choice("allowed", cast(Hashable, []))

    filing = Run(Repeat("loop", (Block("allowed", ()),), bad, "policy"), states=(OwnedState("policy", list),))
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(OutOfBounds, match="hashable"):
        run(Engine(ready).run())
    assert ready.state.unusable_states == {"policy"}


def test_claim_race_with_refilled_feed_keeps_race_diagnostic():
    async def check():
        feed = Feed(1, asyncio.Condition())
        await feed.publish(Product(2, "replacement"))
        with pytest.raises(NotReady, match="claim lost readiness"):
            await join((feed,), 1, 1, lambda *args: True, Projection({}, {}), wait=False)
        assert feed.ready[2].value == "replacement" and not feed.handed_off

    run(check())


def test_finished_capture_does_not_mistake_never_started_run_for_completion():
    ready = prepare(Contract().accept(Run(Block("idle", ())))).publish()
    with pytest.raises(NotReady, match="finished"):
        ready.state.capture()
    run(Engine(ready).run())
    assert ready.state.completed and ready.state.capture().generation == 1
