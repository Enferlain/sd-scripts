"""Current-candidate adapters and negative controls for independent oracles."""

from __future__ import annotations

import asyncio

from dataclasses import replace
from typing import Literal, cast
from unittest.mock import patch

import pytest
import torch

from . import examples
from .conformance import InputUse, LearningFacts, PartialFacts, SourceVersion, UnitFact, check_learning, check_partial
from .graph import Block, Call, Contract, Repeat, Together
from .prepare import Context, Engine, ProviderContext, WorkFailed, prepare
from .state import InputHandoff, Product
from .test_specialized_lowering import three_units


type LoweringKind = Literal["calls", "inline"]
type AccessKind = Literal["general", "prepared"]


def input_facts(handoffs: tuple[InputHandoff, ...], consumer: int, participants: dict[str, str]) -> tuple[InputUse, ...]:
    facts = []
    for handoff in handoffs:
        assert isinstance(handoff.key, int)
        sources = tuple(
            SourceVersion(participants[stamp.participant_ref], stamp.binding, stamp.numerical_epoch) for stamp in handoff.provenance
        )
        facts.append(InputUse(consumer, handoff.channel, handoff.key, sources))
    return tuple(facts)


def learning_facts(
    lowering: LoweringKind, access: AccessKind, reverse_inputs: bool = False, *, skip_updates: bool = False
) -> LearningFacts:
    """Observe the existing connected adaptive example without reading its trace."""
    boundaries = []
    inputs: list[InputUse] = []
    requests: dict[int, int] = {}
    participants: dict[str, str] = {}

    def objective(context: Context, image: torch.Tensor, condition: examples.Conditioning):
        assert isinstance(context.choice.key, int)
        key = context.choice.key
        requests[context.invocation] = key
        inputs.extend(input_facts(context.input_handoffs, key, participants))
        boundaries.append(("use-feedback", key))
        return examples.diffusion_objective(context, image, condition)

    def feedback(context: Context, error: float):
        assert isinstance(context.choice.key, int)
        examples.observe(context, error)
        boundaries.append(("feedback", context.choice.key))

    async def alternate_pixels(context: ProviderContext):
        for key in (3, 1, 2):
            await context.publish("image", Product(key, torch.tensor(float(key))))
            await asyncio.sleep(0)

    def instrument(scope):
        if isinstance(scope, Together):
            return replace(scope, children=tuple(instrument(child) for child in scope.children))
        if isinstance(scope, Repeat):
            return replace(scope, blocks=tuple(instrument(block) for block in scope.blocks))
        if isinstance(scope, Block):
            return replace(
                scope,
                work=tuple(
                    replace(work, implementation=objective if work.name == "objective" else feedback)
                    if isinstance(work, Call) and work.name in ("objective", "observe")
                    else work
                    for work in scope.work
                ),
            )
        if reverse_inputs and scope.name == "pixels":
            return replace(scope, implementation=alternate_pixels)
        return scope

    filing = examples.diffusion()
    accepted = Contract().accept(replace(filing, root=instrument(filing.root)))
    participants.update((ref, name) for name, ref in accepted.participant_refs.items())
    ready = prepare(accepted, lowering=lowering, access=access).publish()
    assert ready.state.image is not None
    unit = ready.state.image.units["weights"]
    parameter = unit.parameters[0]
    gradients, weights = [], []
    original = unit.optimizer.step

    def advance():
        assert parameter.grad is not None
        gradients.append(float(parameter.grad))
        if not skip_updates:
            original()
        weights.append(float(parameter.detach()))

    async def scenario():
        async with asyncio.timeout(5):
            with patch.object(unit.optimizer, "step", advance):
                await Engine(ready).run()

    asyncio.run(scenario())
    owner = ready.state.owners["adaptive"]
    assert isinstance(owner, examples.LearningState)
    unit_inputs = tuple(
        fact for outcome in unit.outcomes for fact in input_facts(outcome.input_handoffs, requests[outcome.invocation], participants)
    )
    return LearningFacts(
        tuple(inputs),
        unit_inputs,
        tuple(owner.factors_used),
        tuple(owner.observations),
        tuple(gradients),
        tuple(weights),
        tuple(boundaries),
    )


def partial_facts(lowering: LoweringKind, access: AccessKind) -> PartialFacts:
    ready = prepare(Contract().accept(three_units()), lowering=lowering, access=access).publish()
    state = ready.state
    assert state.image is not None
    image = state.image
    unit = image.units["B"]

    def fail():
        with torch.no_grad():
            unit.parameters[0].add_(0.125)
        raise RuntimeError("B failed after entering its update")

    async def scenario():
        async with asyncio.timeout(5):
            with patch.object(unit.optimizer, "step", fail), pytest.raises(WorkFailed) as failure:
                await Engine(ready).run()
        return failure.value

    failure = asyncio.run(scenario())
    names = {ref: name for name, ref in state.accepted.unit_refs.items()}
    facts = tuple(
        UnitFact(
            names[outcome.unit_ref], outcome.contribution, outcome.optimizer, outcome.reset, image.units[names[outcome.unit_ref]].steps
        )
        for outcome in failure.unit_outcomes
    )
    return PartialFacts(facts, failure.source, frozenset(state.unusable), float(image.units["A"].parameters[0].detach()))


@pytest.fixture(scope="module", params=("calls", "inline"))
def lowering(request: pytest.FixtureRequest) -> LoweringKind:
    return cast(LoweringKind, request.param)


@pytest.fixture(scope="module", params=("general", "prepared"))
def access(request: pytest.FixtureRequest) -> AccessKind:
    return cast(AccessKind, request.param)


@pytest.fixture(scope="module", params=(False, True), ids=("original-input-order", "changed-input-order"))
def learning(lowering: LoweringKind, access: AccessKind, request: pytest.FixtureRequest) -> LearningFacts:
    return learning_facts(lowering, access, cast(bool, request.param))


@pytest.fixture(scope="module")
def partial(lowering: LoweringKind, access: AccessKind) -> PartialFacts:
    return partial_facts(lowering, access)


def test_adaptive_execution_matches_independent_expectations(learning: LearningFacts):
    check_learning(learning)


def test_partial_failure_matches_independent_expectations(partial: PartialFacts):
    check_partial(partial)


@pytest.mark.parametrize("member", ("inputs", "unit_inputs"))
@pytest.mark.parametrize("fault", ("key", "consumer", "source-participant", "source-binding", "source-state", "missing", "duplicate"))
def test_input_corruption_rejects_even_when_numerics_match(learning: LearningFacts, member: str, fault: str):
    facts = list(getattr(learning, member))
    index = next(index for index, fact in enumerate(facts) if fact.member == "conditioning")
    original = facts[index]
    if fault == "key":
        facts[index] = replace(original, key=2)
    elif fault == "consumer":
        facts[index] = replace(original, consumer=2)
    elif fault == "source-participant":
        facts[index] = replace(original, sources=(replace(original.sources[0], participant="another-encoder"),))
    elif fault == "source-binding":
        facts[index] = replace(original, sources=(replace(original.sources[0], binding=2),))
    elif fault == "source-state":
        facts[index] = replace(original, sources=(replace(original.sources[0], numerical_state=1),))
    elif fault == "missing":
        facts.pop(index)
    else:
        facts.append(original)
    damaged = replace(learning, **{member: tuple(facts)})
    assert damaged.weights == learning.weights
    with pytest.raises(AssertionError, match=r"input|provenance"):
        check_learning(damaged)


@pytest.mark.parametrize("member", ("factors", "losses", "gradients", "weights"))
@pytest.mark.parametrize("fault", (0.25, float("nan"), float("inf")))
def test_numerical_corruption_rejects(learning: LearningFacts, member: str, fault: float):
    values = list(getattr(learning, member))
    values[-1] += fault
    with pytest.raises(AssertionError, match=member):
        check_learning(replace(learning, **{member: tuple(values)}))


@pytest.mark.parametrize("fault", ("late", "missing", "duplicate"))
def test_feedback_corruption_rejects_even_when_numerics_match(learning: LearningFacts, fault: str):
    events = list(learning.boundaries)
    boundary = ("feedback", 1)
    if fault == "late":
        events.remove(boundary)
        events.insert(events.index(("use-feedback", 2)) + 1, boundary)
    elif fault == "missing":
        events.remove(boundary)
    else:
        events.append(boundary)
    damaged = replace(learning, boundaries=tuple(events))
    assert damaged.weights == learning.weights and damaged.factors == learning.factors
    with pytest.raises(AssertionError, match="feedback"):
        check_learning(damaged)


def test_private_interleavings_do_not_require_identical_scheduler_traces(learning: LearningFacts):
    first = replace(learning, boundaries=(("private", 10), *learning.boundaries, ("independent", 20)))
    second = replace(learning, boundaries=tuple(event for boundary in learning.boundaries for event in (boundary, ("private", 30))))
    assert first.boundaries != second.boundaries
    check_learning(first)
    check_learning(second)


def test_input_report_enumeration_does_not_change_association(learning: LearningFacts):
    check_learning(replace(learning, inputs=tuple(reversed(learning.inputs)), unit_inputs=tuple(reversed(learning.unit_inputs))))


@pytest.mark.parametrize(
    ("index", "change"),
    (
        (0, {"advancement": "uncertain"}),
        (1, {"advancement": "returned"}),
        (2, {"advancement": "skipped"}),
        (0, {"reset": "not_attempted"}),
        (1, {"steps": 1}),
        (2, {"contribution": "not_attempted"}),
    ),
)
def test_partial_outcome_corruption_rejects_without_reinterpreting_weights(partial: PartialFacts, index: int, change: dict[str, object]):
    units = list(partial.units)
    units[index] = replace(units[index], **change)
    damaged = replace(partial, units=tuple(units))
    assert damaged.returned_weight == partial.returned_weight
    with pytest.raises(AssertionError, match="outcomes"):
        check_partial(damaged)


@pytest.mark.parametrize("fault", ("missing", "duplicate", "failure", "availability", "weight"))
def test_partial_report_corruption_rejects(partial: PartialFacts, fault: str):
    changes: dict[str, object]
    if fault == "missing":
        changes = {"units": partial.units[:-1]}
    elif fault == "duplicate":
        changes = {"units": (*partial.units, partial.units[0])}
    elif fault == "failure":
        changes = {"failed_work": "objective"}
    elif fault == "availability":
        changes = {"unavailable": frozenset()}
    else:
        changes = {"returned_weight": partial.returned_weight + 0.5}
    with pytest.raises(AssertionError):
        check_partial(replace(partial, **changes))


def test_unit_report_enumeration_order_is_not_execution_order(partial: PartialFacts):
    check_partial(replace(partial, units=tuple(reversed(partial.units))))


def test_both_lowerings_can_share_a_fault_which_independent_checks_reject(access: AccessKind):
    damaged = tuple(learning_facts(lowering, access, skip_updates=True) for lowering in ("calls", "inline"))
    assert damaged[0] == damaged[1]
    for facts in damaged:
        with pytest.raises(AssertionError, match=r"factors|losses|gradients|weights"):
            check_learning(facts)
