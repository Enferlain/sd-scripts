"""Compare two CPU realizations of one meaning, not two authored runs."""

from __future__ import annotations

import asyncio
import gc
import weakref
from dataclasses import replace
from typing import Literal, cast

import pytest
import torch
from torch import nn

from .examples import (
    Conditioning,
    LearningState,
    Scale,
    adversarial,
    correlated,
    diffusion,
    joint,
    replacement,
    routed_views,
    scale,
    two_pass,
)
from .graph import (
    Accepted,
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
    Participant,
    Producer,
    Repeat,
    Run,
    Together,
    TwoPass,
    Unit,
    Use,
    View,
)
from .prepare import Context, Engine, ProviderContext, Ready, WorkFailed, prepare
from .state import Product, RunState


type LoweringKind = Literal["calls", "inline"]


def run(awaitable):
    async def bounded():
        return await asyncio.wait_for(awaitable, 5)

    return asyncio.run(bounded())


def paired(accepted: Accepted) -> tuple[Ready, Ready]:
    # Exactly the same accepted root, source identities and selected behavior.
    # Each realization has freshly initialized models/owners/optimizer state.
    return prepare(accepted, lowering="calls").publish(), prepare(accepted, lowering="inline").publish()


def observable(ready: Ready):
    state = ready.state
    assert state.image is not None
    return {
        "generation": state.generation,
        "weights": {
            name: {path: value.tolist() for path, value in binding.model.state_dict().items()}
            for name, binding in state.image.bindings.items()
        },
        "modes": {name: tuple(module.training for module in binding.model.modules()) for name, binding in state.image.bindings.items()},
        "stamps": {name: state.stamp(name) for name in state.image.bindings},
        "owners": state.owners,
        "coordinates": state.coordinates,
        "units": {
            name: (
                unit.ref,
                unit.steps,
                unit.pending,
                tuple(None if parameter.grad is None else parameter.grad.tolist() for parameter in unit.parameters),
                tuple(
                    (
                        outcome.invocation,
                        outcome.contribution,
                        outcome.optimizer,
                        outcome.reset,
                        outcome.advancement_invocation,
                        outcome.source_stamps,
                        outcome.work,
                        outcome.input_handoffs,
                    )
                    for outcome in unit.outcomes
                ),
            )
            for name, unit in state.image.units.items()
        },
        "handoffs": {name: tuple(feed.handed_off) for name, feed in state.feeds.items()},
        "unusable": state.unusable,
        "unusable_states": state.unusable_states,
        "completed": state.completed,
    }


@pytest.mark.parametrize("case", ("adaptive", "alternating", "joint", "two-pass", "replacement", "retained", "isolated"))
def test_one_accepted_graph_has_equivalent_cpu_realizations(case):
    build = {
        "adaptive": diffusion,
        "alternating": adversarial,
        "joint": joint,
        "two-pass": two_pass,
        "replacement": replacement,
        "retained": routed_views,
        "isolated": lambda: routed_views(isolated=True, recompute=True),
    }[case]
    accepted = Contract(two_pass=case == "two-pass").accept(build())
    baseline, specialized = paired(accepted)
    assert baseline.state.accepted is specialized.state.accepted is accepted
    assert baseline.state.image is not None and specialized.state.image is not None
    assert all(
        baseline.state.image.bindings[name].model is not specialized.state.image.bindings[name].model for name in accepted.participants
    )
    run(Engine(baseline).run())
    run(Engine(specialized).run())
    assert observable(baseline) == observable(specialized)
    if case == "adaptive":
        owner = specialized.state.owners["adaptive"]
        assert isinstance(owner, LearningState)
        assert owner.factors_used == [1.0, 1 / (1 + owner.observations[0]), 1 / (1 + owner.observations[1])]
        assert [entry[0] for entry in specialized.state.feeds["image"].handed_off] == [1, 2, 3]
        assert "await invoke_" in specialized.source["learn"] and "await work_" in baseline.source["learn"]
    elif case in ("retained", "isolated"):
        # Honest fallback: cross-block lifetime emission is not specialized.
        assert baseline.source["routed-work"] == specialized.source["routed-work"]


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_prepared_execution_does_not_need_authoring_or_rewalk_static_wiring(lowering, monkeypatch):
    filing = joint()
    authored = weakref.ref(filing)
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    del filing
    gc.collect()
    assert authored() is None
    from . import prepare as preparation_module

    monkeypatch.setattr(preparation_module, "ordered", lambda _: pytest.fail("static graph revisited during execution"))
    run(Engine(ready).run())
    assert ready.state.image is not None
    assert ready.state.image.units["A"].steps == ready.state.image.units["B"].steps == 1


def three_units() -> Run:
    def objective(context: Context):
        return (sum(context.models[name](torch.tensor(float(index))) for index, name in enumerate(("a", "b", "c"), 2)) - 1).square()

    return Run(
        Block(
            "joint",
            (
                Call("objective", objective, outputs=(Output("loss", torch.Tensor),), uses=tuple(Use(name) for name in ("a", "b", "c"))),
                Differentiate("backward", "loss", ("A", "B", "C"), "gradient"),
                Advance("advance", "gradient", ("A", "B", "C")),
            ),
        ),
        tuple(scale(name, 0.2) for name in ("a", "b", "c")),
        tuple(Unit(unit, ((name, "weight"),)) for unit, name in zip(("A", "B", "C"), ("a", "b", "c"), strict=True)),
    )


def test_partial_failure_keeps_returned_uncertain_and_unattempted_units_separate(monkeypatch):
    accepted = Contract().accept(three_units())
    realizations = paired(accepted)
    failures = []
    for ready in realizations:
        assert ready.state.image is not None
        unit = ready.state.image.units["B"]

        def broken_step(unit=unit):
            with torch.no_grad():
                unit.parameters[0].add_(0.125)
            raise RuntimeError("backend failed after entering mutable work")

        monkeypatch.setattr(unit.optimizer, "step", broken_step)
        with pytest.raises(WorkFailed) as captured:
            run(Engine(ready).run())
        failure = captured.value
        assert failure.source == "advance" and not failure.implementation_returned
        assert [outcome.optimizer for outcome in failure.unit_outcomes] == ["returned", "uncertain", "not_attempted"]
        assert [outcome.reset for outcome in failure.unit_outcomes] == ["returned", "not_attempted", "not_attempted"]
        assert [ready.state.image.units[name].steps for name in ("A", "B", "C")] == [1, 0, 0]
        assert ready.state.unusable == {"a", "b", "c"}
        assert not ready.state.readers and not ready.state.writers
        failures.append((failure.source, type(failure.cause), str(failure.cause)))
    assert failures[0] == failures[1]
    assert observable(realizations[0]) == observable(realizations[1])


@pytest.mark.parametrize("bad_result", (None, "wrong type", (1, 2)))
def test_invalid_output_after_effects_does_not_erase_effects_or_run_dependents(bad_result):
    def bad(context: Context):
        owner = cast(dict[str, int], context.states["effects"])
        owner["writes"] += 1
        with torch.no_grad():
            context.models["model"].get_parameter("weight").add_(0.25)
        return bad_result

    def after(context: Context, value):
        cast(dict[str, int], context.states["effects"])["dependent"] += 1

    filing = Run(
        Block(
            "effects",
            (
                Call("bad", bad, outputs=(Output("value", int),), uses=(Use("model", "write"),), states=("effects",)),
                Call("dependent", after, ("value",), states=("effects",)),
            ),
        ),
        (scale("model", 0.5),),
        states=(OwnedState("effects", lambda: {"writes": 0, "dependent": 0}),),
    )
    realizations = paired(Contract().accept(filing))
    for ready in realizations:
        with pytest.raises(WorkFailed) as captured:
            run(Engine(ready).run())
        failure = captured.value
        assert failure.source == "bad" and failure.implementation_returned
        assert isinstance(failure.cause, TypeError)
        assert ready.state.owners["effects"] == {"writes": 1, "dependent": 0}
        assert ready.state.unusable == {"model"} and ready.state.unusable_states == {"effects"}
        assert not ready.state.writers and not ready.state.state_locks
        with pytest.raises(NotReady, match="unavailable"):
            run(Engine(ready).run())
    assert observable(realizations[0]) == observable(realizations[1])


def test_output_failure_and_evaluation_cleanup_failure_remain_separate():
    class BrokenCleanup(Scale):
        broken: bool = False

        def __setattr__(self, name, value):
            if name == "training" and value and getattr(self, "broken", False):
                raise RuntimeError("mode cleanup failed")
            super().__setattr__(name, value)

    def bad(context: Context):
        cast(BrokenCleanup, context.models["model"]).broken = True
        return "invalid"

    filing = Run(
        Block("evaluation", (Call("bad", bad, outputs=(Output("value", int),), uses=(Use("model"),), evaluation=True),)),
        (Participant("model", lambda: BrokenCleanup(0.5), lambda model: None),),
    )
    for ready in paired(Contract().accept(filing)):
        with pytest.raises(WorkFailed) as captured:
            run(Engine(ready).run())
        failure = captured.value
        assert isinstance(failure.cause, TypeError) and failure.implementation_returned
        assert [str(error) for error in failure.cleanup_failures] == ["mode cleanup failed"]
        assert ready.state.unusable == {"model"} and not ready.state.writers


def test_stale_handback_cannot_advance_a_prepared_unit(monkeypatch):
    realizations = paired(Contract().accept(joint()))
    for ready in realizations:
        contribute = ready.state.contribute

        def stale(*args, contribute=contribute, **kwargs):
            contribution = contribute(*args, **kwargs)
            return replace(contribution, prepared_keys=tuple(object() for _ in contribution.prepared_keys))

        # Inject incorrect backend correspondence, not a selected operation
        # claiming authority to manufacture an optimization handback.
        monkeypatch.setattr(ready.state, "contribute", stale)
        with pytest.raises(WorkFailed) as captured:
            run(Engine(ready).run())
        assert isinstance(captured.value.cause, NotReady) and "Stale" in str(captured.value.cause)
        assert [outcome.optimizer for outcome in captured.value.unit_outcomes] == ["not_attempted", "not_attempted"]
        assert ready.state.image is not None and all(unit.steps == 0 for unit in ready.state.image.units.values())
    assert observable(realizations[0]) == observable(realizations[1])


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_inadmissible_join_keeps_both_inputs_unclaimed_and_state_usable(lowering):
    async def produce(context: ProviderContext):
        await context.publish("a", Product("", 1))
        await context.publish("b", Product("", "wrong"))

    filing = Run(
        Together(
            "run",
            (
                Producer("producer", ("a", "b"), produce),
                Block(
                    "inputs",
                    (
                        Join(
                            "join",
                            ("a", "b"),
                            (Output("a", int), Output("b", int)),
                            lambda product, view: isinstance(product.value, int),
                            (Use("model"),),
                        ),
                    ),
                ),
            ),
        ),
        (scale("model", 0.5),),
        channels=(Channel("a", 1), Channel("b", 1)),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()

    async def check():
        with pytest.raises(ExceptionGroup) as captured:
            await Engine(ready).run()
        assert len(captured.value.exceptions) == 1
        assert isinstance(captured.value.exceptions[0], NotReady) and "Inadmissible" in str(captured.value.exceptions[0])
        assert all("" in feed.ready and not feed.handed_off for feed in ready.state.feeds.values())
        assert not ready.state.unusable and not ready.state.readers

    run(check())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_suspension_and_cancellation_keep_protection_then_withhold_uncertain_state(lowering):
    started = asyncio.Event()

    async def selected(context: Context):
        with torch.no_grad():
            context.models["model"].get_parameter("weight").add_(0.25)
        started.set()
        await asyncio.Event().wait()

    filing = Run(Block("activity", (Call("suspend", selected, uses=(Use("model", "write"),)),)), (scale("model", 0.5),))
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()

    async def check():
        task = asyncio.create_task(Engine(ready).run())
        await started.wait()

        async def reader():
            async with ready.state.acquire((Use("model"),)):
                pytest.fail("reader admitted to a protected/uncertain state")

        waiting = asyncio.create_task(reader())
        await asyncio.sleep(0)
        assert not waiting.done() and ready.state.writers == {"model"}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(NotReady, match="unavailable"):
            await waiting
        assert ready.state.unusable == {"model"}
        assert not ready.state.writers and not ready.state.readers

    run(check())


def test_named_no_gradient_view_and_evaluation_survive_specialization():
    def selected(context: Context):
        return context.views["teacher"](torch.tensor(1.0, requires_grad=True))

    def inspect(context: Context, value: torch.Tensor):
        cast(dict[str, bool], context.states["observed"])["requires_grad"] = value.requires_grad

    filing = Run(
        Block(
            "views",
            (
                Call("selected", selected, outputs=(Output("value", torch.Tensor),), uses=(Use("base", view="teacher"),), evaluation=True),
                Call("inspect", inspect, ("value",), states=("observed",)),
            ),
        ),
        (Participant("base", nn.Identity, lambda model: None),),
        states=(OwnedState("observed", dict),),
        views=(View("teacher", "base", gradients=False, evaluation=True),),
    )
    realizations = paired(Contract().accept(filing))
    for ready in realizations:
        run(Engine(ready).run())
        assert ready.state.owners["observed"] == {"requires_grad": False}
    assert observable(realizations[0]) == observable(realizations[1])


def test_returned_region_without_clean_handback_cannot_authorize_advancement():
    def selected(access, models):
        access.backward((models["model"](torch.tensor(2.0)) - 1).square())
        access.perturb(0.1)
        # Return normally without restoring the temporary view.

    filing = Run(
        Block(
            "granted",
            (
                TwoPass("region", selected, (), "weights", "handback", (Use("model", "write"),)),
                Advance("retained-advance", "handback", ("weights",)),
            ),
        ),
        (scale("model", 0.8),),
        (Unit("weights", (("model", "weight"),)),),
    )
    realizations = paired(Contract(two_pass=True).accept(filing))
    for ready in realizations:
        with pytest.raises(WorkFailed) as captured:
            run(Engine(ready).run())
        assert captured.value.source == "region" and captured.value.implementation_returned
        assert isinstance(captured.value.cause, NotReady)
        assert ready.state.image is not None and ready.state.image.units["weights"].steps == 0
        assert ready.state.unusable == {"model"} and not ready.state.writers
    assert observable(realizations[0]) == observable(realizations[1])


def test_changed_producer_state_and_discarded_old_product_keep_actual_dependencies():
    async def produce(context: ProviderContext):
        owner = cast(dict, context.states["producer"])
        for key in (2, 1):
            if key == 1:
                async with context.use_source((Use("encoder", "write"),)) as source:
                    with torch.no_grad():
                        source.models["encoder"].get_parameter("weight").add_(1)
            async with context.use_source((Use("encoder"),)) as source:
                with torch.no_grad():
                    value = source.models["encoder"](torch.tensor(float(key))).detach().clone()
                actual = (source.stamps["encoder"],)
            await context.publish("condition", Product(key, Conditioning(key, 1, value), actual))
            owner["produced"].append((key, actual))
        # Independently progressing work ends under its authored demand-end
        # relationship, not because the engine guesses a universal step.
        await asyncio.Event().wait()

    async def cleanup(context: ProviderContext):
        cast(dict, context.states["producer"])["finished"] = True

    def objective(context: Context, condition: Conditioning):
        cast(list, context.states["consumed"]).append(condition.sample)
        return (context.models["denoiser"](condition.value) - 1).square()

    learn = Block(
        "learn",
        (
            Join("join", ("condition",), (Output("condition", Conditioning),), correlated, (Use("encoder"),)),
            Call("objective", objective, ("condition",), (Output("loss", torch.Tensor),), (Use("denoiser"),), ("consumed",)),
            Differentiate("backward", "loss", ("weights",), "gradient"),
            Advance("advance", "gradient", ("weights",)),
        ),
    )
    filing = Run(
        Together(
            "run",
            (
                Producer("producer", ("condition",), produce, (Use("encoder", "write"),), ("producer",), "learning", "discard", cleanup),
                Repeat("learning", (learn,), lambda counts, state: None if counts else Choice("learn", 1), "policy"),
            ),
        ),
        (scale("encoder", 1.0), scale("denoiser", 0.3)),
        (Unit("weights", (("denoiser", "weight"),)),),
        (
            OwnedState("producer", lambda: {"produced": [], "finished": False}),
            OwnedState("consumed", list),
            OwnedState("policy", lambda: None),
        ),
        (Channel("condition", 2),),
    )
    realizations = paired(Contract().accept(filing))
    for ready in realizations:
        run(Engine(ready).run())
        produced = cast(dict, ready.state.owners["producer"])["produced"]
        assert [key for key, _ in produced] == [2, 1]
        assert [provenance[0].numerical_epoch for _, provenance in produced] == [0, 1]
        assert ready.state.owners["consumed"] == [1]
        assert cast(dict, ready.state.owners["producer"])["finished"]
        assert [item[0] for item in ready.state.feeds["condition"].handed_off] == [1]
        assert [item[0] for item in ready.state.feeds["condition"].discarded] == [2]
        assert ready.state.feeds["condition"].high_water <= 2
        assert not ready.state.active_producers and not ready.state.readers and not ready.state.writers
    assert observable(realizations[0]) == observable(realizations[1])


def test_lowering_choice_does_not_bypass_stale_publication_or_accept_unknown_targets():
    accepted = Contract().accept(joint())
    state = RunState(accepted)
    first = prepare(accepted, state, lowering="calls")
    stale = prepare(accepted, state, lowering="inline")
    first.publish()
    with pytest.raises(NotReady, match="Stale"):
        stale.publish()
    with pytest.raises(NotReady, match="Unknown candidate"):
        prepare(accepted, lowering=cast(LoweringKind, "unknown"))


def test_empty_block_is_supported_in_both_targets():
    for ready in paired(Contract().accept(Run(Block("empty", ())))):
        run(Engine(ready).run())
        assert ready.state.completed and ready.state.invocation == 1


def test_cost_probe_compares_matched_work_and_reports_allocation_quantities_separately():
    from .lowering_cost_probe import worker

    result = run(worker(iterations=5, samples=2, warmup=2, run_index=0))
    assert result["environment"]["gc_during_timing"] is False
    for case in result["cases"].values():
        assert case["order"] == [("calls", "inline"), ("inline", "calls")]
        assert len(case["paired_excess_ns"]) == 2
        assert all(len(values) == 2 for values in case["ns_per_invocation"].values())
        assert all(set(record) == {"iterations", "peak_above_start_bytes", "retained_after_gc_bytes"} for record in case["memory"].values())
