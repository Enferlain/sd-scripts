"""Test-only probe of joint preparation and optimization ordering.

These small objects describe an exchange, not production APIs or evidence that
Accelerate/DeepSpeed support both orderings. No live Trainer path uses them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest


Member = tuple[str, str]  # accepted participant, accepted substructure


@dataclass(frozen=True)
class UnitMeaning:
    address: str
    incarnation: str
    definition_revision: int
    members: tuple[Member, ...]
    logical_group: str


@dataclass(frozen=True)
class AcceptedMeaning:
    obligation_revision: int
    units: tuple[UnitMeaning, ...]
    required_participants: tuple[str, ...]


@dataclass(frozen=True)
class SourceState:
    authority_revision: int
    bindings: dict[str, dict[str, object]]


@dataclass(frozen=True)
class PreparationRequest:
    meaning: AcceptedMeaning
    source_authority_revision: int
    required_participants: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerHandle:
    parameters: tuple[object, ...]


@dataclass(frozen=True)
class UnitCandidate:
    meaning: UnitMeaning
    optimizer: OptimizerHandle
    resolved_members: tuple[Member, ...]


@dataclass(frozen=True)
class PreparationCandidate:
    source_authority_revision: int
    source_obligation_revision: int
    route_members: dict[str, dict[str, object]]
    units: dict[str, UnitCandidate]
    backend_group: frozenset[str]


@dataclass(frozen=True)
class CurrentPreparedState:
    routes: dict[str, dict[str, object]]  # authority-owned in the target design
    units: dict[str, UnitCandidate]  # Trainer optimization-owned
    backend_group: frozenset[str]  # Trainer infrastructure-owned


def _parameters(members: tuple[Member, ...], routes: dict[str, dict[str, object]]) -> tuple[object, ...]:
    return tuple(routes[participant][path] for participant, path in members)


def _unique_parameters(members: tuple[Member, ...], routes: dict[str, dict[str, object]]) -> tuple[object, ...]:
    distinct: dict[int, object] = {}
    for parameter in _parameters(members, routes):
        distinct.setdefault(id(parameter), parameter)
    return tuple(distinct.values())


def _request(meaning: AcceptedMeaning, source: SourceState) -> PreparationRequest:
    """Derive a pinned, complete job and reject known incompatibility first."""
    if not set(meaning.required_participants) <= source.bindings.keys():
        raise ValueError("missing required participant")
    if len({unit.address for unit in meaning.units}) != len(meaning.units):
        raise ValueError("duplicate unit address")
    owners: dict[int, str] = {}
    for unit in meaning.units:
        for parameter in _parameters(unit.members, source.bindings):
            previous_owner = owners.setdefault(id(parameter), unit.address)
            if previous_owner != unit.address:
                raise ValueError("parameter belongs to multiple units")
    return PreparationRequest(meaning, source.authority_revision, meaning.required_participants)


def _candidate(request: PreparationRequest, source: SourceState, *, order: str, replace_parameters: bool) -> PreparationCandidate:
    """Simulate a backend's private construction order without publishing it."""
    if request.source_authority_revision != source.authority_revision:
        raise ValueError("stale preparation request")
    meaning = request.meaning

    def wrap() -> dict[str, dict[str, object]]:
        replacements: dict[int, object] = {}

        def prepared(parameter: object) -> object:
            if not replace_parameters:
                return parameter
            return replacements.setdefault(id(parameter), object())

        return {
            participant: {path: prepared(parameter) for path, parameter in members.items()}
            for participant, members in source.bindings.items()
        }

    if order == "optimizer_before_wrapper":
        optimizers = {unit.address: OptimizerHandle(_unique_parameters(unit.members, source.bindings)) for unit in meaning.units}
        routes = wrap()
    elif order == "wrapper_before_optimizer":
        routes = wrap()
        optimizers = {unit.address: OptimizerHandle(_unique_parameters(unit.members, routes)) for unit in meaning.units}
    else:
        raise ValueError("unknown backend ordering")

    return PreparationCandidate(
        source_authority_revision=source.authority_revision,
        source_obligation_revision=meaning.obligation_revision,
        route_members=routes,
        units={unit.address: UnitCandidate(unit, optimizers[unit.address], unit.members) for unit in meaning.units},
        backend_group=frozenset(request.required_participants),
    )


def _publish(meaning: AcceptedMeaning, source: SourceState, candidate: PreparationCandidate) -> CurrentPreparedState:
    """Check the complete unpublished result, then install one replacement."""
    if (candidate.source_authority_revision, candidate.source_obligation_revision) != (
        source.authority_revision,
        meaning.obligation_revision,
    ):
        raise ValueError("stale preparation source")
    if candidate.backend_group != frozenset(meaning.required_participants):
        raise ValueError("incomplete backend group")
    if set(candidate.route_members) != set(meaning.required_participants):
        raise ValueError("incomplete prepared routes")
    if set(candidate.units) != {unit.address for unit in meaning.units}:
        raise ValueError("incomplete unit runtime")

    owners: dict[int, str] = {}
    for unit in meaning.units:
        runtime = candidate.units[unit.address]
        if runtime.meaning != unit or runtime.resolved_members != unit.members:
            raise ValueError("unit meaning changed during preparation")
        current_parameters = _unique_parameters(unit.members, candidate.route_members)
        # Membership, not tuple position, is the semantic check in this
        # one-group toy. Production state restoration needs stronger provenance.
        if len(current_parameters) != len(runtime.optimizer.parameters) or {id(parameter) for parameter in current_parameters} != {
            id(parameter) for parameter in runtime.optimizer.parameters
        }:
            raise ValueError("optimizer points at non-current parameters")
        for parameter in current_parameters:
            previous_owner = owners.setdefault(id(parameter), unit.address)
            if previous_owner != unit.address:
                raise ValueError("prepared parameter belongs to multiple units")

    # A real coordinator must install the separately owned surfaces without
    # exposing an intermediate state. This test models the visibility rule as
    # a single replacement; it does not implement rank agreement or recovery.
    return CurrentPreparedState(candidate.route_members, candidate.units, candidate.backend_group)


def _arrangement() -> tuple[AcceptedMeaning, SourceState]:
    meaning = AcceptedMeaning(
        obligation_revision=7,
        units=(
            UnitMeaning("matrix", "unit-10", 3, (("denoiser", "matrix"),), "matrix_group"),
            UnitMeaning("other", "unit-11", 2, (("adapter", "delta"),), "other_group"),
        ),
        required_participants=("denoiser", "adapter", "frozen_encoder"),
    )
    source = SourceState(
        authority_revision=12,
        bindings={
            "denoiser": {"matrix": object()},
            "adapter": {"delta": object()},
            "frozen_encoder": {"weights": object()},
        },
    )
    return meaning, source


@pytest.mark.training
@pytest.mark.unit
@pytest.mark.parametrize(
    ("order", "replace_parameters"),
    (("optimizer_before_wrapper", False), ("wrapper_before_optimizer", True)),
)
def test_backend_order_changes_physical_construction_not_accepted_meaning(order: str, replace_parameters: bool):
    meaning, source = _arrangement()
    candidate = _candidate(_request(meaning, source), source, order=order, replace_parameters=replace_parameters)
    assert (candidate.source_authority_revision, candidate.source_obligation_revision) == (12, 7)
    prepared = _publish(meaning, source, candidate)

    assert set(prepared.routes) == {"denoiser", "adapter", "frozen_encoder"}
    assert prepared.backend_group == frozenset(prepared.routes)
    assert {address: (runtime.meaning.incarnation, runtime.meaning.definition_revision) for address, runtime in prepared.units.items()} == {
        "matrix": ("unit-10", 3),
        "other": ("unit-11", 2),
    }
    assert {address: runtime.meaning.logical_group for address, runtime in prepared.units.items()} == {
        "matrix": "matrix_group",
        "other": "other_group",
    }
    assert all(
        parameter is prepared.routes[participant][path]
        for runtime in prepared.units.values()
        for (participant, path), parameter in zip(runtime.resolved_members, runtime.optimizer.parameters)
    )
    assert all(participant != "frozen_encoder" for runtime in prepared.units.values() for participant, _ in runtime.resolved_members)
    if replace_parameters:
        assert prepared.routes["denoiser"]["matrix"] is not source.bindings["denoiser"]["matrix"]


@pytest.mark.training
@pytest.mark.unit
def test_optimizer_first_cannot_publish_after_wrapper_replaces_its_parameters():
    meaning, source = _arrangement()
    request = _request(meaning, source)
    previous = _publish(meaning, source, _candidate(request, source, order="optimizer_before_wrapper", replace_parameters=False))
    candidate = _candidate(request, source, order="optimizer_before_wrapper", replace_parameters=True)

    with pytest.raises(ValueError, match="optimizer points at non-current parameters"):
        _publish(meaning, source, candidate)
    assert previous.routes["denoiser"]["matrix"] is source.bindings["denoiser"]["matrix"]


@pytest.mark.training
@pytest.mark.unit
@pytest.mark.parametrize(
    ("order", "replace_parameters"),
    (("optimizer_before_wrapper", False), ("wrapper_before_optimizer", True)),
)
def test_two_aliases_within_one_unit_have_one_optimizer_owner(order: str, replace_parameters: bool):
    meaning, source = _arrangement()
    source.bindings["denoiser"]["matrix_alias"] = source.bindings["denoiser"]["matrix"]
    matrix = replace(meaning.units[0], members=(("denoiser", "matrix"), ("denoiser", "matrix_alias")))
    meaning = replace(meaning, units=(matrix, meaning.units[1]))

    candidate = _candidate(_request(meaning, source), source, order=order, replace_parameters=replace_parameters)
    prepared = _publish(meaning, source, candidate)

    assert prepared.units["matrix"].resolved_members == matrix.members
    assert prepared.routes["denoiser"]["matrix"] is prepared.routes["denoiser"]["matrix_alias"]
    assert prepared.units["matrix"].optimizer.parameters == (prepared.routes["denoiser"]["matrix"],)


@pytest.mark.training
@pytest.mark.unit
def test_member_order_does_not_define_one_unit_but_duplicate_unit_address_rejects():
    meaning, source = _arrangement()
    source.bindings["denoiser"]["bias"] = object()
    matrix = replace(meaning.units[0], members=(("denoiser", "matrix"), ("denoiser", "bias")))
    meaning = replace(meaning, units=(matrix, meaning.units[1]))
    candidate = _candidate(_request(meaning, source), source, order="wrapper_before_optimizer", replace_parameters=True)
    original = candidate.units["matrix"]
    candidate.units["matrix"] = replace(original, optimizer=OptimizerHandle(tuple(reversed(original.optimizer.parameters))))
    assert _publish(meaning, source, candidate).units["matrix"].meaning == matrix

    duplicate_address = replace(meaning, units=(matrix, replace(meaning.units[1], address=matrix.address)))
    with pytest.raises(ValueError, match="duplicate unit address"):
        _request(duplicate_address, source)


@pytest.mark.training
@pytest.mark.unit
def test_stale_or_overlapping_candidates_never_become_current():
    meaning, source = _arrangement()
    candidate = _candidate(_request(meaning, source), source, order="wrapper_before_optimizer", replace_parameters=True)
    newer_source = SourceState(source.authority_revision + 1, source.bindings)
    with pytest.raises(ValueError, match="stale preparation source"):
        _publish(meaning, newer_source, candidate)

    tied = SourceState(
        source.authority_revision,
        {**source.bindings, "adapter": {"delta": source.bindings["denoiser"]["matrix"]}},
    )
    with pytest.raises(ValueError, match="parameter belongs to multiple units"):
        _request(meaning, tied)

    aliased_result = _candidate(_request(meaning, source), source, order="wrapper_before_optimizer", replace_parameters=True)
    aliased_result.route_members["adapter"]["delta"] = aliased_result.route_members["denoiser"]["matrix"]
    other = aliased_result.units["other"]
    aliased_result.units["other"] = UnitCandidate(
        other.meaning,
        OptimizerHandle((aliased_result.route_members["adapter"]["delta"],)),
        other.resolved_members,
    )
    with pytest.raises(ValueError, match="prepared parameter belongs to multiple units"):
        _publish(meaning, source, aliased_result)


@pytest.mark.training
@pytest.mark.unit
def test_incomplete_result_cannot_publish_only_the_routes_or_only_the_units():
    meaning, source = _arrangement()
    request = _request(meaning, source)
    previous = _publish(meaning, source, _candidate(request, source, order="optimizer_before_wrapper", replace_parameters=False))

    missing_unit = _candidate(request, source, order="wrapper_before_optimizer", replace_parameters=True)
    missing_unit.units.pop("other")
    with pytest.raises(ValueError, match="incomplete unit runtime"):
        _publish(meaning, source, missing_unit)

    missing_route = _candidate(request, source, order="wrapper_before_optimizer", replace_parameters=True)
    missing_route.route_members.pop("frozen_encoder")
    with pytest.raises(ValueError, match="incomplete prepared routes"):
        _publish(meaning, source, missing_route)

    missing_group_member = replace(
        _candidate(request, source, order="wrapper_before_optimizer", replace_parameters=True),
        backend_group=frozenset({"denoiser", "adapter"}),
    )
    with pytest.raises(ValueError, match="incomplete backend group"):
        _publish(meaning, source, missing_group_member)

    assert set(previous.routes) == set(meaning.required_participants)
    assert set(previous.units) == {unit.address for unit in meaning.units}
