"""Small, independent test oracles for two execution cases.

These are test observations, not runtime result objects or an execution API.
There are no candidate/Torch imports: another implementation can extract the
same facts without reproducing Ready, RunState, or their internal trace.
"""

from __future__ import annotations

import math

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceVersion:
    participant: str
    binding: int
    numerical_state: int


@dataclass(frozen=True)
class InputUse:
    consumer: int
    member: str
    key: int
    sources: tuple[SourceVersion, ...] = ()


type Boundary = tuple[str, int]


@dataclass(frozen=True)
class LearningFacts:
    inputs: tuple[InputUse, ...]
    unit_inputs: tuple[InputUse, ...]
    factors: tuple[float, ...]
    losses: tuple[float, ...]
    gradients: tuple[float, ...]
    weights: tuple[float, ...]
    boundaries: tuple[Boundary, ...]


@dataclass(frozen=True)
class UnitFact:
    unit: str
    contribution: str
    advancement: str
    reset: str
    steps: int


@dataclass(frozen=True)
class PartialFacts:
    units: tuple[UnitFact, ...]
    failed_work: str
    unavailable: frozenset[str]
    returned_weight: float


def adaptive_reference() -> LearningFacts:
    """The selected three-input example, calculated without engine/AD helpers.

    Encoder weight 1.5 produces conditioning from key / 4. Training starts
    at weight .8, uses squared error and SGD at .05, then derives the next
    factor from the observation. Input order is an accepted case choice;
    physical production order is deliberately absent from the reference.
    """
    weight, factor = 0.8, 1.0
    inputs: list[InputUse] = []
    factors, losses, gradients, weights = [], [], [], []
    boundaries: list[Boundary] = []
    for key in (1, 2, 3):
        inputs.extend((InputUse(key, "image", key), InputUse(key, "conditioning", key, (SourceVersion("encoder", 1, 0),))))
        value = key + 1.5 * key / 4
        error = weight * value - key / 2
        loss = error * error * factor
        gradient = 2 * error * value * factor
        factors.append(factor)
        losses.append(loss)
        gradients.append(gradient)
        weight -= 0.05 * gradient
        weights.append(weight)
        factor = 1 / (1 + loss)
        boundaries.extend((("use-feedback", key), ("feedback", key)))
    return LearningFacts(tuple(inputs), tuple(inputs), tuple(factors), tuple(losses), tuple(gradients), tuple(weights), tuple(boundaries))


def check_numbers(actual: tuple[float, ...], expected: tuple[float, ...], label: str) -> None:
    """Explicit tolerance for these small fp32 CPU fixtures only."""
    if len(actual) != len(expected) or any(
        not math.isfinite(value) or not math.isclose(value, reference, rel_tol=2e-6, abs_tol=2e-6)
        for value, reference in zip(actual, expected, strict=True)
    ):
        raise AssertionError(f"{label} differ: {actual!r} != {expected!r}")


def check_order(actual: tuple[Boundary, ...], required: tuple[tuple[Boundary, Boundary], ...]) -> None:
    """Only named causal relationships matter; private/interleaved work may vary."""
    positions: dict[Boundary, int] = {}
    for boundary in {event for edge in required for event in edge}:
        indices = [index for index, event in enumerate(actual) if event == boundary]
        if len(indices) != 1:
            raise AssertionError(f"feedback boundary {boundary!r} occurred {len(indices)} times")
        positions[boundary] = indices[0]
    for before, after in required:
        if positions[before] >= positions[after]:
            raise AssertionError(f"feedback order lost: {before!r} must precede {after!r}")


def check_learning(actual: LearningFacts) -> None:
    expected = adaptive_reference()

    # Association is semantic; the adapter's enumeration order is not.
    def ordered(inputs: tuple[InputUse, ...]):
        return sorted(inputs, key=lambda fact: (fact.consumer, fact.member))

    if ordered(actual.inputs) != ordered(expected.inputs):
        raise AssertionError("input identity or source provenance differs")
    if ordered(actual.unit_inputs) != ordered(expected.unit_inputs):
        raise AssertionError("optimization input association differs")
    for label in ("factors", "losses", "gradients", "weights"):
        check_numbers(getattr(actual, label), getattr(expected, label), label)
    required = tuple((("use-feedback", key), ("feedback", key)) for key in (1, 2, 3)) + tuple(
        (("feedback", key), ("use-feedback", key + 1)) for key in (1, 2)
    )
    check_order(actual.boundaries, required)


def check_partial(actual: PartialFacts) -> None:
    # The first optimizer/reset returned; the second entered and failed; the
    # third was never attempted. These facts are not inferred from the weights.
    expected = (
        UnitFact("A", "ready", "returned", "returned", 1),
        UnitFact("B", "ready", "uncertain", "not_attempted", 0),
        UnitFact("C", "ready", "not_attempted", "not_attempted", 0),
    )
    if sorted(actual.units, key=lambda unit: unit.unit) != list(expected):
        raise AssertionError("partial optimization outcomes differ")
    if actual.failed_work != "advance" or actual.unavailable != frozenset(("a", "b", "c")):
        raise AssertionError("failure boundary or unavailable state differs")
    # Only A is known to have completed. No numerical promise is made about
    # B's uncertain update, nor does A's numerical result prove its completion.
    gradient_a = 2 * (0.2 * (2 + 3 + 4) - 1) * 2
    check_numbers((actual.returned_weight,), (0.2 - 0.05 * gradient_a,), "returned weight")
