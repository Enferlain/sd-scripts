"""Shared owner lifecycle across contributions, grants and authority consumers."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest
import torch

from .examples import Scale, accumulation, joint, scale, two_pass
from .graph import Advance, Block, Call, Contract, Differentiate, NotReady, Output, Retain, Together, TwoPass, Unit, Use
from .prepare import Engine, GrantedAccess, WorkFailed, prepare
from .state import Replacement, UnitOutcome
from .test_specialized_lowering import run


def assert_transition_rejection(state):
    """Each consumer asks about unfinished work, not just ready gradients."""
    assert state.image is not None
    current, generation = state.image, state.generation
    models = {name: binding.model for name, binding in current.bindings.items()}
    name = next(iter(models))
    candidate = state.candidate({**models, name: Scale(3.0)}, replacing=name)
    with pytest.raises(NotReady, match="quiescent retained/contribution"):
        state.install(candidate, generation)
    with pytest.raises(NotReady, match="unfinished gradients"):
        run(state.replace(Replacement(name, current.bindings[name].revision, Scale(3.0)), name))
    # Make the inactive session eligible for a cut; existing failure flags can
    # independently reject it, but entry-only cases reach the pending-work gate.
    completed = state.completed
    state.completed = True
    try:
        with pytest.raises(NotReady):
            state.capture()
    finally:
        state.completed = completed
    assert state.image is current and state.generation == generation


@pytest.mark.parametrize("granted", (False, True))
def test_entry_before_handback_is_unfinished_for_every_consumer(granted):
    ready = prepare(Contract().accept(accumulation())).publish()
    state = ready.state
    assert state.image is not None
    unit = state.image.units["weights"]
    if granted:
        GrantedAccess(state, "weights", 1)
    else:
        # Real first backward failure: no ready gradient/token was produced.
        with pytest.raises(RuntimeError):
            state.contribute(torch.tensor(1.0), ("weights",), 1)
    assert unit.unfinished and unit.pending is None and unit.window is None
    assert tuple(unit.active_outcomes) == (1,)
    assert unit.active_outcomes[1].contribution == ("unverified_grant" if granted else "uncertain")
    assert_transition_rejection(state)
    for new_grant in (False, True):
        with pytest.raises(NotReady, match="explicit verified continuation"):
            state.begin_contribution(("weights",), 2, granted=new_grant)
    with pytest.raises(NotReady, match="verified contribution window"):
        run(state.retain(Retain("hold", ("model",), ("weights",), "old"), object()))
    assert not state.readers and not state.writers and not state.active_retentions


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("granted", (False, True))
def test_ordinary_and_granted_work_share_owner_entry_and_completion(lowering, granted, monkeypatch):
    filing = two_pass() if granted else joint()
    ready = prepare(Contract(two_pass=granted).accept(filing), lowering=lowering).publish()
    state = ready.state
    assert state.image is not None
    observed = []
    begin, complete = state.begin_contribution, state.complete_contribution

    def entered(names, invocation, **kwargs):
        begin(names, invocation, **kwargs)
        assert state.image is not None
        for name in names:
            unit = state.image.units[name]
            assert unit.unfinished and unit.window is None
        observed.append(("entry", names, kwargs.get("granted", False)))

    def finished(names, invocation, **kwargs):
        result = complete(names, invocation, **kwargs)
        assert state.image is not None
        for name in names:
            unit = state.image.units[name]
            assert unit.unfinished and unit.window is result
        observed.append(("handback", names, kwargs.get("granted", False)))
        return result

    monkeypatch.setattr(state, "begin_contribution", entered)
    monkeypatch.setattr(state, "complete_contribution", finished)
    run(Engine(ready).run())
    names = ("weights",) if granted else ("A", "B")
    assert observed == [("entry", names, granted), ("handback", names, granted)]
    assert all(not unit.unfinished and unit.pending is None and not unit.active_outcomes for unit in state.image.units.values())
    state.capture()


@pytest.mark.parametrize("granted", (False, True))
def test_ready_diagnostics_cannot_replace_owner_facts_or_duplicate_handback(granted):
    ready = prepare(Contract().accept(accumulation())).publish()
    state = ready.state
    assert state.image is not None
    unit = state.image.units["weights"]
    state.begin_contribution(("weights",), 1, granted=granted)
    unit.parameters[0].grad = torch.ones_like(unit.parameters[0])
    contribution = state.complete_contribution(("weights",), 1, granted=granted)
    for name, value in (("pending", None), ("window", None), ("active_outcomes", {})):
        with pytest.raises(AttributeError):
            setattr(unit, name, value)
    with pytest.raises(TypeError):
        cast(dict[int, UnitOutcome], unit.active_outcomes)[99] = UnitOutcome(unit.ref, 99)
    unit.active_outcomes[1].contribution = "not_attempted"
    assert unit.active_outcomes[1].contribution == ("granted_ready" if granted else "ready")
    with pytest.raises(NotReady, match="duplicate contribution handback"):
        state.complete_contribution(("weights",), 1, granted=granted)
    state.advance(contribution, ("weights",), 1)
    assert not unit.unfinished and unit.steps == 1


def test_failed_extension_withholds_old_readiness_without_losing_prior_origins():
    ready = prepare(Contract().accept(accumulation())).publish()
    state = ready.state
    assert state.image is not None
    unit, region = state.image.units["weights"], object()
    parameter = unit.parameters[0]
    first = state.contribute(parameter.square(), ("weights",), 1, region=region)
    with pytest.raises(RuntimeError):
        state.contribute(torch.tensor(1.0), ("weights",), 2, region=region, previous=first)
    assert unit.unfinished and unit.window is first and tuple(unit.active_outcomes) == (1, 2)
    assert [record.contribution for record in unit.active_outcomes.values()] == ["ready", "uncertain"]
    for extend in (False, True):
        with pytest.raises(NotReady, match="gradient handback"):
            if extend:
                state.contribute(parameter.square(), ("weights",), 3, region=region, previous=first)
            else:
                state.advance(first, ("weights",), 3, region=region)
    assert_transition_rejection(state)
    assert parameter.grad is not None
    assert unit.steps == 0 and float(parameter.grad) == 4.0


@pytest.mark.parametrize("mismatch", ("subset", "reorder", "preparation", "region", "grant", "invocation"))
def test_handback_preserves_the_entered_destination_and_authority(mismatch):
    ready = prepare(Contract().accept(joint())).publish()
    state, region = ready.state, object()
    assert state.image is not None
    state.begin_contribution(("A", "B"), 1, region=region)
    for unit in state.image.units.values():
        unit.parameters[0].grad = torch.ones_like(unit.parameters[0])
    names = ("A",) if mismatch == "subset" else ("B", "A") if mismatch == "reorder" else ("A", "B")
    if mismatch == "preparation":
        state.image.units["B"].prepared_key = object()
    with pytest.raises(NotReady, match="contribution handback"):
        state.complete_contribution(
            names, 2 if mismatch == "invocation" else 1, region=object() if mismatch == "region" else region, granted=mismatch == "grant"
        )
    assert all(unit.unfinished and unit.window is None for unit in state.image.units.values())


@pytest.mark.parametrize("stage", ("clip", "optimizer", "reset"))
def test_entered_advancement_cannot_reuse_readiness_after_failure(stage, monkeypatch):
    filing = accumulation()
    filing = replace(filing, units=(replace(filing.units[0], clip_norm=1.0),))
    ready = prepare(Contract().accept(filing)).publish()
    state = ready.state
    assert state.image is not None
    unit = state.image.units["weights"]
    contribution = state.contribute(unit.parameters[0].square(), ("weights",), 1)

    def fail(*args, **kwargs):
        raise RuntimeError(stage)

    if stage == "clip":
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fail)
    else:
        monkeypatch.setattr(unit.optimizer, "step" if stage == "optimizer" else "zero_grad", fail)
    with pytest.raises(RuntimeError, match=stage):
        state.advance(contribution, ("weights",), 1)
    record = unit.active_outcomes[1]
    assert record.optimizer == ("not_attempted" if stage == "clip" else "uncertain" if stage == "optimizer" else "returned")
    assert record.reset == ("uncertain" if stage == "reset" else "not_attempted")
    with pytest.raises(NotReady, match="gradient handback"):
        state.advance(contribution, ("weights",), 1)
    assert unit.steps == int(stage == "reset") and unit.unfinished
    assert_transition_rejection(state)


def test_granted_work_cannot_extend_a_standard_window_even_in_its_region():
    ready = prepare(Contract().accept(accumulation())).publish()
    state, region = ready.state, object()
    assert state.image is not None
    unit = state.image.units["weights"]
    contribution = state.contribute(unit.parameters[0].square(), ("weights",), 1, region=region)
    with pytest.raises(NotReady, match="granted work cannot extend"):
        state.begin_contribution(("weights",), 2, previous=contribution, region=region, granted=True)
    assert tuple(unit.active_outcomes) == (1,) and unit.window is contribution


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_granted_work_can_progress_while_an_independent_unit_has_pending_gradients(lowering):
    async def scenario():
        entered, granted = asyncio.Event(), asyncio.Event()
        filing = two_pass()
        assert isinstance(filing.root, Block)
        operation = filing.root.work[0]
        assert isinstance(operation, TwoPass)

        async def wait_for_grant(context):
            entered.set()
            await granted.wait()

        async def wait_for_other(context):
            await entered.wait()

        def selected(access, models):
            assert ready.state.image is not None
            other = ready.state.image.units["other"]
            assert other.unfinished and other.pending is not None and other.steps == 0
            operation.implementation(access, models)
            granted.set()

        ordinary = Block(
            "independent-window",
            (
                Call(
                    "objective",
                    lambda context: context.models["other"].weight.square(),
                    outputs=(Output("loss", torch.Tensor),),
                    uses=(Use("other"),),
                ),
                Differentiate("backward", "loss", ("other",), "gradient"),
                Call("wait-for-grant", wait_for_grant, after=("backward",)),
                Advance("advance", "gradient", ("other",), after=("wait-for-grant",)),
            ),
        )
        grant = replace(
            filing.root,
            work=(
                Call("wait-for-other", wait_for_other),
                replace(operation, implementation=selected, after=("wait-for-other",)),
                *filing.root.work[1:],
            ),
        )
        filing = replace(
            filing,
            root=Together("run", (ordinary, grant)),
            participants=(*filing.participants, scale("other", 0.5)),
            units=(*filing.units, Unit("other", (("other", "weight"),))),
        )
        ready = prepare(Contract(two_pass=True).accept(filing), lowering=lowering).publish()
        await Engine(ready).run()
        assert ready.state.image is not None
        assert all(unit.steps == 1 and not unit.unfinished for unit in ready.state.image.units.values())

    run(scenario())


def test_joint_advancement_is_claimed_before_any_backend_callback(monkeypatch):
    ready = prepare(Contract().accept(joint())).publish()
    state = ready.state
    assert state.image is not None
    a, b = state.image.units["A"], state.image.units["B"]
    loss = sum((parameter.square().sum() for unit in (a, b) for parameter in unit.parameters), torch.tensor(0.0))
    contribution = state.contribute(loss, ("A", "B"), 1)
    step, observed = a.optimizer.step, []

    def reenter():
        with pytest.raises(NotReady, match="gradient handback"):
            state.advance(contribution, ("A", "B"), 1)
        assert not a.accepts(contribution) and not b.accepts(contribution)
        observed.append("both claimed before first backend entry")
        step()

    monkeypatch.setattr(a.optimizer, "step", reenter)
    state.advance(contribution, ("A", "B"), 1)
    assert observed == ["both claimed before first backend entry"]
    assert a.steps == b.steps == 1 and not a.unfinished and not b.unfinished


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_partial_joint_failure_keeps_completed_unit_evidence_in_the_same_owner(lowering, monkeypatch):
    ready = prepare(Contract().accept(joint()), lowering=lowering).publish()
    state = ready.state
    assert state.image is not None
    a, b = state.image.units["A"], state.image.units["B"]

    def fail():
        raise RuntimeError("second backend")

    monkeypatch.setattr(b.optimizer, "step", fail)
    with pytest.raises(WorkFailed) as caught:
        run(Engine(ready).run())
    assert a.steps == 1 and a.pending is None and b.steps == 0 and b.pending == 1
    assert a.unfinished and b.unfinished
    # Neither clearing diagnostic history nor modifying a returned report can
    # erase the active facts or unblock a state transition.
    a.outcomes.clear()
    b.outcomes.clear()
    a.evidence(1, None)[0].optimizer = "not_attempted"
    assert a.evidence(1, None)[0].optimizer == "returned"
    assert [(record.optimizer, record.reset) for record in caught.value.unit_outcomes] == [
        ("returned", "returned"),
        ("uncertain", "not_attempted"),
    ]
    assert_transition_rejection(state)


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_failed_grant_keeps_owner_evidence_without_fabricating_handback(lowering):
    filing = two_pass()
    assert isinstance(filing.root, Block)
    original = filing.root.work[0]
    assert isinstance(original, TwoPass)

    def fail_after_restore(access, models, *args):
        original.implementation(access, models, *args)
        raise RuntimeError("after restoration")

    filing = replace(filing, root=replace(filing.root, work=(replace(original, implementation=fail_after_restore), *filing.root.work[1:])))
    ready = prepare(Contract(two_pass=True).accept(filing), lowering=lowering).publish()
    state = ready.state
    with pytest.raises(WorkFailed) as caught:
        run(Engine(ready).run())
    assert state.image is not None
    unit = state.image.units["weights"]
    assert unit.unfinished and unit.window is None and unit.steps == 0
    assert [(record.contribution, record.optimizer) for record in caught.value.unit_outcomes] == [("unverified_grant", "not_attempted")]
    with pytest.raises(NotReady, match="explicit verified continuation"):
        GrantedAccess(state, "weights", 2)
    assert tuple(unit.active_outcomes) == (1,)
    assert_transition_rejection(state)
    assert not state.readers and not state.writers
