"""Executable spike for the model-strategy-Trainer contract exchange.

This module is deliberately isolated from production code.  It tests whether
the design can become readable, enforceable Python without committing the
repository to these names, value types, or module boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum


class ContractViolation(ValueError):
    """The authored definition or a candidate violates the accepted contract."""


class TransitionRejected(RuntimeError):
    """A proposed lifecycle transition is not valid from current state."""


class StalePreparationResult(TransitionRejected):
    """A preparation result no longer matches authoritative current state."""


class CurrentStateUnavailable(TransitionRejected):
    """A requested current live view is not presently valid."""


class ParticipantUse(StrEnum):
    """Small test vocabulary from which the spike contract derives obligations."""

    EXECUTE = "execute"
    OPTIMIZE = "optimize"
    PERSIST = "persist"


class RelationshipKind(StrEnum):
    ATTACHMENT = "attachment"


class RuntimeTrait(StrEnum):
    """Contract-known implementation facts used only by this spike."""

    CALLABLE = "callable"
    PARAMETERIZED = "parameterized"
    SERIALIZABLE = "serializable"
    EFFECT_STATE = "effect_state"
    EFFECT_HOST = "effect_host"


class ParticipantLifecycle(StrEnum):
    DECLARED = "declared"
    BOUND = "bound"
    RETIRED = "retired"


class RelationshipLifecycle(StrEnum):
    DECLARED = "declared"
    RESOLVED = "resolved"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DETACHED = "detached"


class PreparationStatus(StrEnum):
    UNPREPARED = "unprepared"
    PREPARING = "preparing"
    PREPARED = "prepared"
    INVALID = "invalid"


class PreparationBehavior(StrEnum):
    REPLACEMENT_ONLY = "replacement_only"
    IN_PLACE = "in_place"


class LineageKind(StrEnum):
    SUCCESSOR_OF = "successor_of"
    DERIVED_FROM_ARTIFACT = "derived_from_artifact"


class AttemptStatus(StrEnum):
    ISSUED = "issued"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, order=True)
class ParticipantRef:
    """Authority-qualified identity of one participant incarnation."""

    authority_id: str
    key: str
    incarnation: int


@dataclass(frozen=True, slots=True, order=True)
class RelationshipRef:
    """Authority-qualified identity of one relationship incarnation."""

    authority_id: str
    key: str
    incarnation: int


@dataclass(frozen=True, slots=True, order=True)
class PreparationAttemptRef:
    authority_id: str
    attempt: int


@dataclass(frozen=True, slots=True, order=True)
class ArtifactRef:
    value: str


LineageSource = ParticipantRef | ArtifactRef


@dataclass(frozen=True, slots=True)
class ParticipantSpec:
    """Authored participant address and its explicit contract uses."""

    key: str
    uses: frozenset[ParticipantUse]

    def __post_init__(self) -> None:
        if not self.key:
            raise ContractViolation("participant key must not be empty")


@dataclass(frozen=True, slots=True)
class RelationshipSpec:
    key: str
    source_key: str
    target_key: str
    kind: RelationshipKind

    def __post_init__(self) -> None:
        if not self.key:
            raise ContractViolation("relationship key must not be empty")
        if self.source_key == self.target_key:
            raise ContractViolation("a relationship must have distinct endpoints")


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    participants: tuple[ParticipantSpec, ...]
    relationships: tuple[RelationshipSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class KnownImplementation:
    implementation_id: str
    fulfills: frozenset[RuntimeTrait]


@dataclass(frozen=True, slots=True)
class MaterializedCandidate:
    """A live object plus its contract-known library implementation identity."""

    implementation_id: str
    value: object


@dataclass(frozen=True, slots=True)
class ParticipantObligations:
    required_traits: frozenset[RuntimeTrait]
    requires_preparation: bool
    required_routes: frozenset[str]


@dataclass(frozen=True, slots=True)
class LineageEdge:
    source: LineageSource
    target: ParticipantRef
    kind: LineageKind


@dataclass(slots=True)
class _RouteState:
    revision: int = 0
    valid: bool = False
    candidate: MaterializedCandidate | None = None


@dataclass(slots=True)
class _ParticipantState:
    ref: ParticipantRef
    spec: ParticipantSpec
    obligations: ParticipantObligations
    lifecycle: ParticipantLifecycle = ParticipantLifecycle.DECLARED
    binding: MaterializedCandidate | None = None
    binding_revision: int = 0
    prepared_view: MaterializedCandidate | None = None
    preparation_revision: int = 0
    preparation_status: PreparationStatus = PreparationStatus.UNPREPARED
    preparing_attempt: PreparationAttemptRef | None = None
    routes: dict[str, _RouteState] = field(default_factory=dict)
    lineage: list[LineageEdge] = field(default_factory=list)


@dataclass(slots=True)
class _RelationshipState:
    ref: RelationshipRef
    spec: RelationshipSpec
    source: ParticipantRef
    target: ParticipantRef
    lifecycle: RelationshipLifecycle = RelationshipLifecycle.DECLARED
    revision: int = 0


@dataclass(frozen=True, slots=True)
class RouteSnapshot:
    key: str
    revision: int
    valid: bool


@dataclass(frozen=True, slots=True)
class ParticipantSnapshot:
    ref: ParticipantRef
    key: str
    uses: frozenset[ParticipantUse]
    lifecycle: ParticipantLifecycle
    binding_revision: int
    preparation_revision: int
    preparation_status: PreparationStatus
    preparing_attempt: PreparationAttemptRef | None
    routes: tuple[RouteSnapshot, ...]
    lineage: tuple[LineageEdge, ...]


@dataclass(frozen=True, slots=True)
class RelationshipSnapshot:
    ref: RelationshipRef
    key: str
    source: ParticipantRef
    target: ParticipantRef
    lifecycle: RelationshipLifecycle
    revision: int


@dataclass(frozen=True, slots=True)
class ParticipantPreparation:
    ref: ParticipantRef
    source_binding_revision: int
    source_preparation_revision: int
    source: MaterializedCandidate
    required_routes: frozenset[str]


@dataclass(frozen=True, slots=True)
class RelationshipDependency:
    ref: RelationshipRef
    revision: int


@dataclass(frozen=True, slots=True)
class PreparationJob:
    """One attempt with an explicit fine-grained freshness dependency policy."""

    attempt: PreparationAttemptRef
    behavior: PreparationBehavior
    participants: tuple[ParticipantPreparation, ...]
    relationships: tuple[RelationshipDependency, ...]


@dataclass(frozen=True, slots=True)
class PreparedParticipant:
    ref: ParticipantRef
    source_binding_revision: int
    source_preparation_revision: int
    candidate: MaterializedCandidate
    routes: frozenset[str]


@dataclass(frozen=True, slots=True)
class PreparationResult:
    job: PreparationJob
    participants: tuple[PreparedParticipant, ...]
    backend_state: object


@dataclass(slots=True)
class _AttemptState:
    job: PreparationJob
    status: AttemptStatus = AttemptStatus.ISSUED


@dataclass(frozen=True, slots=True)
class _PreparedAuthorityUpdate:
    attempt: PreparationAttemptRef
    participants: tuple[PreparedParticipant, ...]


@dataclass(frozen=True, slots=True)
class _RetirementAuthorityUpdate:
    retiring: ParticipantRef
    invalidate_prepared: frozenset[ParticipantRef]


@dataclass(frozen=True, slots=True)
class BackendGroupState:
    """Opaque Trainer-owned backend state for one inseparable participant group."""

    participants: frozenset[ParticipantRef]
    state: object
    attempt: PreparationAttemptRef


@dataclass(frozen=True, slots=True)
class TrainerRuntimeSnapshot:
    authority_id: str
    backend_groups: tuple[BackendGroupState, ...]
    revision: int
    published_attempts: frozenset[PreparationAttemptRef]


@dataclass(frozen=True, slots=True)
class _TrainerRetirementUpdate:
    candidate: TrainerRuntimeSnapshot
    invalidate_prepared: frozenset[ParticipantRef]


class TrainerRuntime:
    """Only the Trainer-owned part of prepared runtime publication."""

    def __init__(self, authority_id: str, execution_session_id: str) -> None:
        if not authority_id:
            raise ValueError("authority id must not be empty")
        if not execution_session_id:
            raise ValueError("execution session id must not be empty")
        self.authority_id = authority_id
        self.execution_session_id = execution_session_id
        self._backend_groups: dict[frozenset[ParticipantRef], BackendGroupState] = {}
        self.revision = 0
        self.published_attempts: set[PreparationAttemptRef] = set()

    def _preview_install(self, result: PreparationResult) -> TrainerRuntimeSnapshot:
        if result.job.attempt.authority_id != self.authority_id:
            raise TransitionRejected("prepared state belongs to another logical run authority")
        if result.job.attempt in self.published_attempts:
            raise TransitionRejected("a preparation attempt cannot be published twice")

        participant_group = frozenset(item.ref for item in result.job.participants)
        for existing_group in self._backend_groups:
            if participant_group & existing_group and not existing_group <= participant_group:
                raise TransitionRejected("a preparation result cannot omit members of an inseparable backend group")

        groups = {group: state for group, state in self._backend_groups.items() if not group & participant_group}
        groups[participant_group] = BackendGroupState(
            participants=participant_group,
            state=result.backend_state,
            attempt=result.job.attempt,
        )
        return TrainerRuntimeSnapshot(
            authority_id=self.authority_id,
            backend_groups=tuple(groups.values()),
            revision=self.revision + 1,
            published_attempts=frozenset({*self.published_attempts, result.job.attempt}),
        )

    def _inseparable_groups_for(self, authority_id: str) -> tuple[frozenset[ParticipantRef], ...]:
        if authority_id != self.authority_id:
            raise TransitionRejected("Trainer runtime belongs to another logical run authority")
        return tuple(self._backend_groups)

    def _install(self, candidate: TrainerRuntimeSnapshot) -> None:
        # Final publication deliberately performs assignment only.
        self._backend_groups = {item.participants: item for item in candidate.backend_groups}
        self.revision = candidate.revision
        self.published_attempts = set(candidate.published_attempts)

    def _preview_retirement(self, ref: ParticipantRef) -> _TrainerRetirementUpdate:
        if ref.authority_id != self.authority_id:
            raise TransitionRejected("retiring participant belongs to another authority")
        remaining_groups: list[BackendGroupState] = []
        invalidate_prepared = {ref}
        for group in self._backend_groups.values():
            if ref in group.participants:
                invalidate_prepared.update(group.participants)
            else:
                remaining_groups.append(group)
        return _TrainerRetirementUpdate(
            candidate=TrainerRuntimeSnapshot(
                authority_id=self.authority_id,
                backend_groups=tuple(remaining_groups),
                revision=self.revision + 1,
                published_attempts=frozenset(self.published_attempts),
            ),
            invalidate_prepared=frozenset(invalidate_prepared),
        )

    def backend_state_for(
        self,
        authority: BindingAuthority,
        refs: Iterable[ParticipantRef],
    ) -> object:
        if authority.authority_id != self.authority_id:
            raise TransitionRejected("binding authority and Trainer runtime belong to different logical runs")
        participant_group = frozenset(refs)
        try:
            group = self._backend_groups[participant_group]
        except KeyError as error:
            raise CurrentStateUnavailable("no Trainer-owned backend state exists for that exact participant group") from error
        for ref in group.participants:
            if authority.participant(ref).preparation_status is not PreparationStatus.PREPARED:
                raise CurrentStateUnavailable("Trainer-owned backend state is not current under the binding authority")
        return group.state

    def backend_groups(self) -> tuple[BackendGroupState, ...]:
        return tuple(self._backend_groups.values())

    def export_resume_state(self) -> TrainerRuntimeSnapshot:
        return deepcopy(
            TrainerRuntimeSnapshot(
                authority_id=self.authority_id,
                backend_groups=tuple(self._backend_groups.values()),
                revision=self.revision,
                published_attempts=frozenset(self.published_attempts),
            )
        )

    @classmethod
    def restore(cls, state: TrainerRuntimeSnapshot, execution_session_id: str) -> TrainerRuntime:
        runtime = cls(state.authority_id, execution_session_id)
        runtime._backend_groups = {item.participants: item for item in deepcopy(state.backend_groups)}
        runtime.revision = state.revision
        runtime.published_attempts = set(state.published_attempts)
        return runtime


@dataclass(frozen=True, slots=True)
class AuthorityResumeState:
    authority_id: str
    revision: int
    next_participant: int
    next_relationship: int
    next_attempt: int
    participants: tuple[_ParticipantState, ...]
    active_by_key: tuple[tuple[str, ParticipantRef], ...]
    relationships: tuple[_RelationshipState, ...]
    relationship_by_key: tuple[tuple[str, RelationshipRef], ...]
    attempts: tuple[_AttemptState, ...]


@dataclass(frozen=True, slots=True)
class StrategyResumeState:
    contract_version: str
    definition: StrategyDefinition
    authority: AuthorityResumeState


@dataclass(frozen=True, slots=True)
class RunResumeSnapshot:
    strategy: StrategyResumeState
    trainer: TrainerRuntimeSnapshot


@dataclass(frozen=True, slots=True)
class ArtifactMember:
    participant: ParticipantRef
    binding_revision: int
    preparation_revision: int


@dataclass(frozen=True, slots=True)
class Artifact:
    ref: ArtifactRef
    producing_authority_id: str
    authority_revision: int
    members: tuple[ArtifactMember, ...]


_ESTABLISHMENT_TOKEN = object()


class TrainingContract:
    """Test contract vocabulary, rules, and known-library conformance."""

    def __init__(self, version: str, implementations: Iterable[KnownImplementation]) -> None:
        if not version:
            raise ValueError("contract version must not be empty")
        implementation_items = tuple(implementations)
        implementation_map = {item.implementation_id: item.fulfills for item in implementation_items}
        if len(implementation_map) != len(implementation_items):
            raise ValueError("implementation ids must be unique")
        self.version = version
        self._implementation_traits = implementation_map

    def establish(self, definition: StrategyDefinition, authority_id: str) -> TrainingStrategy:
        obligations = self._validate_definition(definition)
        authority = BindingAuthority(authority_id=authority_id, contract=self)
        refs_by_key: dict[str, ParticipantRef] = {}
        for spec in definition.participants:
            refs_by_key[spec.key] = authority._declare_participant(spec, obligations[spec.key])
        for relationship in definition.relationships:
            authority._declare_relationship(
                relationship,
                source=refs_by_key[relationship.source_key],
                target=refs_by_key[relationship.target_key],
            )
        return TrainingStrategy(
            contract=self,
            definition=definition,
            authority=authority,
            _establishment_token=_ESTABLISHMENT_TOKEN,
        )

    def restore_strategy(self, state: StrategyResumeState) -> TrainingStrategy:
        if state.contract_version != self.version:
            raise ContractViolation(f"snapshot contract {state.contract_version!r} does not match {self.version!r}")
        self._validate_definition(state.definition)
        return TrainingStrategy(
            contract=self,
            definition=state.definition,
            authority=BindingAuthority.restore(state.authority, contract=self),
            _establishment_token=_ESTABLISHMENT_TOKEN,
        )

    def obligations_for(self, spec: ParticipantSpec) -> ParticipantObligations:
        required_traits: set[RuntimeTrait] = set()
        if ParticipantUse.EXECUTE in spec.uses:
            required_traits.add(RuntimeTrait.CALLABLE)
        if ParticipantUse.OPTIMIZE in spec.uses:
            required_traits.add(RuntimeTrait.PARAMETERIZED)
        if ParticipantUse.PERSIST in spec.uses:
            required_traits.add(RuntimeTrait.SERIALIZABLE)
        return ParticipantObligations(
            required_traits=frozenset(required_traits),
            requires_preparation=bool(spec.uses & {ParticipantUse.EXECUTE, ParticipantUse.OPTIMIZE}),
            required_routes=(frozenset({"execute"}) if ParticipantUse.EXECUTE in spec.uses else frozenset()),
        )

    def validate_candidate(
        self,
        obligations: ParticipantObligations,
        candidate: MaterializedCandidate,
    ) -> None:
        try:
            provided = self._implementation_traits[candidate.implementation_id]
        except KeyError as error:
            raise ContractViolation(f"implementation {candidate.implementation_id!r} is not known to contract {self.version}") from error
        missing = obligations.required_traits - provided
        if missing:
            missing_names = ", ".join(sorted(item.value for item in missing))
            raise ContractViolation(f"implementation {candidate.implementation_id!r} does not fulfill: {missing_names}")

    def _validate_definition(self, definition: StrategyDefinition) -> dict[str, ParticipantObligations]:
        if not definition.participants:
            raise ContractViolation("a strategy must declare at least one participant")

        participant_by_key = {item.key: item for item in definition.participants}
        if len(participant_by_key) != len(definition.participants):
            raise ContractViolation("participant keys must be unique")

        relationship_by_key = {item.key: item for item in definition.relationships}
        if len(relationship_by_key) != len(definition.relationships):
            raise ContractViolation("relationship keys must be unique")

        required_by_key: dict[str, set[RuntimeTrait]] = {
            key: set(self.obligations_for(spec).required_traits) for key, spec in participant_by_key.items()
        }
        for relationship in definition.relationships:
            if relationship.source_key not in participant_by_key:
                raise ContractViolation(f"relationship {relationship.key!r} has unknown source {relationship.source_key!r}")
            if relationship.target_key not in participant_by_key:
                raise ContractViolation(f"relationship {relationship.key!r} has unknown target {relationship.target_key!r}")
            if relationship.kind is RelationshipKind.ATTACHMENT:
                required_by_key[relationship.source_key].add(RuntimeTrait.EFFECT_STATE)
                required_by_key[relationship.target_key].add(RuntimeTrait.EFFECT_HOST)

        obligations: dict[str, ParticipantObligations] = {}
        for key, spec in participant_by_key.items():
            base = self.obligations_for(spec)
            obligations[key] = ParticipantObligations(
                required_traits=frozenset(required_by_key[key]),
                requires_preparation=base.requires_preparation,
                required_routes=base.required_routes,
            )
        return obligations


class BindingAuthority:
    """Single writer for accepted participant, relationship, and route state."""

    def __init__(self, authority_id: str, contract: TrainingContract) -> None:
        if not authority_id:
            raise ValueError("authority id must not be empty")
        self.authority_id = authority_id
        self._contract = contract
        self.revision = 0
        self._next_participant = 1
        self._next_relationship = 1
        self._next_attempt = 1
        self._participants: dict[ParticipantRef, _ParticipantState] = {}
        self._active_by_key: dict[str, ParticipantRef] = {}
        self._relationships: dict[RelationshipRef, _RelationshipState] = {}
        self._relationship_by_key: dict[str, RelationshipRef] = {}
        self._attempts: dict[PreparationAttemptRef, _AttemptState] = {}

    def active_ref(self, key: str) -> ParticipantRef:
        try:
            return self._active_by_key[key]
        except KeyError as error:
            raise CurrentStateUnavailable(f"no active participant exists at {key!r}") from error

    def relationship_ref(self, key: str) -> RelationshipRef:
        try:
            return self._relationship_by_key[key]
        except KeyError as error:
            raise CurrentStateUnavailable(f"no relationship exists at {key!r}") from error

    def participant(self, ref: ParticipantRef) -> ParticipantSnapshot:
        state = self._participant_state(ref)
        return ParticipantSnapshot(
            ref=state.ref,
            key=state.spec.key,
            uses=state.spec.uses,
            lifecycle=state.lifecycle,
            binding_revision=state.binding_revision,
            preparation_revision=state.preparation_revision,
            preparation_status=state.preparation_status,
            preparing_attempt=state.preparing_attempt,
            routes=tuple(RouteSnapshot(key=key, revision=route.revision, valid=route.valid) for key, route in sorted(state.routes.items())),
            lineage=tuple(state.lineage),
        )

    def relationship(self, ref: RelationshipRef) -> RelationshipSnapshot:
        state = self._relationship_state(ref)
        return RelationshipSnapshot(
            ref=state.ref,
            key=state.spec.key,
            source=state.source,
            target=state.target,
            lifecycle=state.lifecycle,
            revision=state.revision,
        )

    def materialize(
        self,
        ref: ParticipantRef,
        candidate: MaterializedCandidate,
        *,
        lineage_source: LineageSource | None = None,
        lineage_kind: LineageKind | None = None,
    ) -> None:
        state = self._current_participant_state(ref)
        if state.lifecycle is not ParticipantLifecycle.DECLARED:
            raise TransitionRejected("materialization requires a declared participant")
        if (lineage_source is None) != (lineage_kind is None):
            raise ValueError("lineage source and kind must be supplied together")
        if lineage_kind is LineageKind.DERIVED_FROM_ARTIFACT and not isinstance(lineage_source, ArtifactRef):
            raise ContractViolation("artifact derivation requires an artifact lineage source")
        if lineage_kind is LineageKind.SUCCESSOR_OF:
            if not isinstance(lineage_source, ParticipantRef):
                raise ContractViolation("participant succession requires a participant lineage source")
            source_state = self._participant_state(lineage_source)
            if source_state.lifecycle is not ParticipantLifecycle.RETIRED:
                raise TransitionRejected("participant succession requires a retired source")
        self._contract.validate_candidate(state.obligations, candidate)

        state.binding = candidate
        state.binding_revision += 1
        state.lifecycle = ParticipantLifecycle.BOUND
        state.preparation_status = PreparationStatus.UNPREPARED
        if lineage_source is not None and lineage_kind is not None:
            state.lineage.append(LineageEdge(lineage_source, ref, lineage_kind))
        self._advance()

    def replace_binding(self, ref: ParticipantRef, candidate: MaterializedCandidate) -> None:
        state = self._bound_participant_state(ref)
        self._ensure_not_preparing(state)
        self._contract.validate_candidate(state.obligations, candidate)
        self._ensure_related_relationships_not_preparing(ref)

        self._invalidate_prepared_state(state)
        self._invalidate_relationships_for(ref)
        state.binding = candidate
        state.binding_revision += 1
        state.preparation_status = PreparationStatus.UNPREPARED
        self._advance()

    def preview_retirement(
        self,
        ref: ParticipantRef,
        *,
        invalidate_prepared: frozenset[ParticipantRef],
    ) -> _RetirementAuthorityUpdate:
        state = self._current_participant_state(ref)
        self._ensure_not_preparing(state)
        all_affected = frozenset({*invalidate_prepared, ref})
        for affected_ref in all_affected:
            affected = self._current_participant_state(affected_ref)
            self._ensure_not_preparing(affected)
        self._ensure_related_relationships_not_preparing(ref)
        return _RetirementAuthorityUpdate(
            retiring=ref,
            invalidate_prepared=all_affected,
        )

    def _install_retirement(self, update: _RetirementAuthorityUpdate) -> None:
        state = self._participants[update.retiring]
        for affected_ref in update.invalidate_prepared:
            self._invalidate_prepared_state(self._participants[affected_ref])
        # Relationships between surviving co-evicted participants stay active:
        # their logical endpoint bindings did not change.
        for relationship in self._relationships.values():
            if (
                relationship.source == update.retiring or relationship.target == update.retiring
            ) and relationship.lifecycle is not RelationshipLifecycle.DETACHED:
                relationship.lifecycle = RelationshipLifecycle.DETACHED
                relationship.revision += 1
                self._invalidate_relationship_endpoints(relationship)
        state.lifecycle = ParticipantLifecycle.RETIRED
        state.preparing_attempt = None
        state.preparation_status = PreparationStatus.INVALID
        del self._active_by_key[state.spec.key]
        self._advance()

    def resolve_relationship(self, ref: RelationshipRef) -> None:
        relationship = self._relationship_state(ref)
        if relationship.lifecycle not in {
            RelationshipLifecycle.DECLARED,
            RelationshipLifecycle.INACTIVE,
        }:
            raise TransitionRejected("only a declared or inactive relationship can be resolved")
        self._ensure_relationship_endpoints_not_preparing(relationship)
        self._bound_participant_state(relationship.source)
        self._bound_participant_state(relationship.target)
        relationship.lifecycle = RelationshipLifecycle.RESOLVED
        relationship.revision += 1
        self._invalidate_relationship_endpoints(relationship)
        self._advance()

    def activate_relationship(self, ref: RelationshipRef) -> None:
        relationship = self._relationship_state(ref)
        if relationship.lifecycle is not RelationshipLifecycle.RESOLVED:
            raise TransitionRejected("relationship activation requires resolved endpoints")
        self._ensure_relationship_endpoints_not_preparing(relationship)
        relationship.lifecycle = RelationshipLifecycle.ACTIVE
        relationship.revision += 1
        self._invalidate_relationship_endpoints(relationship)
        self._advance()

    def deactivate_relationship(self, ref: RelationshipRef) -> None:
        relationship = self._relationship_state(ref)
        if relationship.lifecycle is not RelationshipLifecycle.ACTIVE:
            raise TransitionRejected("only an active relationship can be deactivated")
        self._ensure_relationship_endpoints_not_preparing(relationship)
        relationship.lifecycle = RelationshipLifecycle.INACTIVE
        relationship.revision += 1
        self._invalidate_relationship_endpoints(relationship)
        self._advance()

    def detach_relationship(self, ref: RelationshipRef) -> None:
        relationship = self._relationship_state(ref)
        if relationship.lifecycle is RelationshipLifecycle.DETACHED:
            raise TransitionRejected("relationship is already detached")
        self._ensure_relationship_endpoints_not_preparing(relationship)
        relationship.lifecycle = RelationshipLifecycle.DETACHED
        relationship.revision += 1
        self._invalidate_relationship_endpoints(relationship)
        self._advance()

    def begin_preparation(
        self,
        refs: Iterable[ParticipantRef] | None = None,
        *,
        behavior: PreparationBehavior = PreparationBehavior.REPLACEMENT_ONLY,
        inseparable_groups: Iterable[frozenset[ParticipantRef]] = (),
    ) -> PreparationJob:
        if refs is None:
            selected_states = [
                state
                for state in self._participants.values()
                if state.lifecycle is not ParticipantLifecycle.RETIRED and state.obligations.requires_preparation
            ]
        else:
            selected_states = [self._bound_participant_state(ref) for ref in refs]

        if not selected_states:
            raise TransitionRejected("a preparation job must contain at least one participant")
        if len({state.ref for state in selected_states}) != len(selected_states):
            raise TransitionRejected("a preparation job cannot contain a participant twice")

        entries: list[ParticipantPreparation] = []
        selected_refs = {state.ref for state in selected_states}
        if behavior is PreparationBehavior.IN_PLACE:
            for group in inseparable_groups:
                if selected_refs & group and not group <= selected_refs:
                    raise TransitionRejected("destructive preparation cannot omit members of an inseparable backend group")
        for state in selected_states:
            if not state.obligations.requires_preparation:
                raise TransitionRejected(f"participant {state.spec.key!r} has no preparation obligation")
            self._ensure_not_preparing(state)
            if behavior is PreparationBehavior.REPLACEMENT_ONLY and state.preparation_status is PreparationStatus.INVALID:
                raise TransitionRejected(f"participant {state.spec.key!r} requires destructive re-establishment")
            if state.binding is None:
                raise CurrentStateUnavailable(f"participant {state.spec.key!r} is not materialized")
            entries.append(
                ParticipantPreparation(
                    ref=state.ref,
                    source_binding_revision=state.binding_revision,
                    source_preparation_revision=state.preparation_revision,
                    source=state.binding,
                    required_routes=state.obligations.required_routes,
                )
            )

        relationship_dependencies = tuple(
            RelationshipDependency(ref=relationship.ref, revision=relationship.revision)
            for relationship in self._relationships.values()
            if relationship.source in selected_refs or relationship.target in selected_refs
        )
        attempt = PreparationAttemptRef(self.authority_id, self._next_attempt)
        self._next_attempt += 1
        job = PreparationJob(
            attempt=attempt,
            behavior=behavior,
            participants=tuple(entries),
            relationships=relationship_dependencies,
        )
        self._attempts[attempt] = _AttemptState(job=job)

        if behavior is PreparationBehavior.IN_PLACE:
            for state in selected_states:
                self._invalidate_prepared_state(state)
                state.preparation_status = PreparationStatus.PREPARING
                state.preparing_attempt = attempt
            self._advance()
        return job

    def abandon_preparation(self, job: PreparationJob) -> None:
        attempt = self._issued_attempt(job)
        if job.behavior is PreparationBehavior.IN_PLACE:
            for entry in job.participants:
                state = self._participant_state(entry.ref)
                if state.preparing_attempt != job.attempt:
                    raise TransitionRejected("in-place preparation attempt no longer owns the state")
            for entry in job.participants:
                state = self._participant_state(entry.ref)
                state.preparing_attempt = None
                state.preparation_status = PreparationStatus.INVALID
            self._advance()
        attempt.status = AttemptStatus.FAILED

    def preview_preparation(self, result: PreparationResult) -> _PreparedAuthorityUpdate:
        attempt = self._issued_attempt(result.job)
        if attempt.job != result.job:
            raise TransitionRejected("preparation result does not match its issued job")

        results_by_ref = {item.ref: item for item in result.participants}
        if len(results_by_ref) != len(result.participants):
            raise TransitionRejected("preparation result contains a participant twice")
        expected_refs = {item.ref for item in result.job.participants}
        if set(results_by_ref) != expected_refs:
            raise TransitionRejected("preparation result membership differs from its job")

        for dependency in result.job.relationships:
            relationship = self._relationship_state(dependency.ref)
            if relationship.revision != dependency.revision:
                raise StalePreparationResult(f"relationship {relationship.spec.key!r} changed during preparation")

        for source in result.job.participants:
            state = self._participant_state(source.ref)
            if state.lifecycle is not ParticipantLifecycle.BOUND or state.binding is None:
                raise StalePreparationResult(f"participant {state.spec.key!r} is no longer a current bound participant")
            if state.binding_revision != source.source_binding_revision:
                raise StalePreparationResult(f"participant {state.spec.key!r} changed during preparation")
            prepared = results_by_ref[source.ref]
            if prepared.source_binding_revision != source.source_binding_revision:
                raise TransitionRejected("prepared participant cites the wrong source revision")
            if prepared.source_preparation_revision != source.source_preparation_revision:
                raise TransitionRejected("prepared participant cites the wrong preparation revision")
            if prepared.routes != source.required_routes:
                raise TransitionRejected("prepared participant routes differ from contract obligations")
            self._contract.validate_candidate(state.obligations, prepared.candidate)
            if result.job.behavior is PreparationBehavior.IN_PLACE:
                if state.preparing_attempt != result.job.attempt or state.preparation_status is not PreparationStatus.PREPARING:
                    raise StalePreparationResult(f"participant {state.spec.key!r} is no longer prepared by this attempt")
            else:
                if state.preparing_attempt is not None or state.preparation_status in {
                    PreparationStatus.PREPARING,
                    PreparationStatus.INVALID,
                }:
                    raise StalePreparationResult(f"participant {state.spec.key!r} has withdrawn preparation guarantees")
                if state.preparation_revision != source.source_preparation_revision:
                    raise StalePreparationResult(f"participant {state.spec.key!r} preparation state changed during preparation")

        return _PreparedAuthorityUpdate(
            attempt=result.job.attempt,
            participants=tuple(results_by_ref[item.ref] for item in result.job.participants),
        )

    def _install_preparation(self, update: _PreparedAuthorityUpdate) -> None:
        # All validation is complete.  This method intentionally performs only
        # authority-owned state replacement and cannot call external code.
        for prepared in update.participants:
            state = self._participants[prepared.ref]
            state.prepared_view = prepared.candidate
            state.preparation_revision += 1
            state.preparation_status = PreparationStatus.PREPARED
            if state.preparing_attempt == update.attempt:
                state.preparing_attempt = None
            for route_key in prepared.routes:
                route = state.routes[route_key]
                route.candidate = prepared.candidate
                route.valid = True
                route.revision += 1
        self._attempts[update.attempt].status = AttemptStatus.PUBLISHED
        self._advance()

    def prepared_view(self, ref: ParticipantRef) -> object:
        state = self._bound_participant_state(ref)
        if state.preparation_status is not PreparationStatus.PREPARED:
            raise CurrentStateUnavailable(f"participant {state.spec.key!r} is not prepared")
        assert state.prepared_view is not None
        return state.prepared_view.value

    def execution_route(self, ref: ParticipantRef, route_key: str = "execute") -> object:
        state = self._bound_participant_state(ref)
        try:
            route = state.routes[route_key]
        except KeyError as error:
            raise CurrentStateUnavailable(f"participant {state.spec.key!r} has no {route_key!r} execution route") from error
        if not route.valid or route.candidate is None:
            raise CurrentStateUnavailable(f"participant {state.spec.key!r} route {route_key!r} is not current")
        return route.candidate.value

    def export_resume_state(self) -> AuthorityResumeState:
        return deepcopy(
            AuthorityResumeState(
                authority_id=self.authority_id,
                revision=self.revision,
                next_participant=self._next_participant,
                next_relationship=self._next_relationship,
                next_attempt=self._next_attempt,
                participants=tuple(self._participants.values()),
                active_by_key=tuple(self._active_by_key.items()),
                relationships=tuple(self._relationships.values()),
                relationship_by_key=tuple(self._relationship_by_key.items()),
                attempts=tuple(self._attempts.values()),
            )
        )

    @classmethod
    def restore(
        cls,
        state: AuthorityResumeState,
        *,
        contract: TrainingContract,
    ) -> BindingAuthority:
        authority = cls(state.authority_id, contract)
        authority.revision = state.revision
        authority._next_participant = state.next_participant
        authority._next_relationship = state.next_relationship
        authority._next_attempt = state.next_attempt
        participants = deepcopy(state.participants)
        relationships = deepcopy(state.relationships)
        attempts = deepcopy(state.attempts)
        authority._participants = {item.ref: item for item in participants}
        authority._active_by_key = dict(state.active_by_key)
        authority._relationships = {item.ref: item for item in relationships}
        authority._relationship_by_key = dict(state.relationship_by_key)
        authority._attempts = {item.job.attempt: item for item in attempts}
        return authority

    def _declare_participant(
        self,
        spec: ParticipantSpec,
        obligations: ParticipantObligations,
        *,
        lineage_source: ParticipantRef | None = None,
    ) -> ParticipantRef:
        if spec.key in self._active_by_key:
            raise ContractViolation(f"participant key {spec.key!r} is already active")
        if lineage_source is not None:
            source = self._participant_state(lineage_source)
            if source.lifecycle is not ParticipantLifecycle.RETIRED:
                raise TransitionRejected("a successor can only follow a retired participant")

        ref = ParticipantRef(self.authority_id, spec.key, self._next_participant)
        self._next_participant += 1
        state = _ParticipantState(
            ref=ref,
            spec=spec,
            obligations=obligations,
            routes={key: _RouteState() for key in obligations.required_routes},
        )
        if lineage_source is not None:
            state.lineage.append(LineageEdge(lineage_source, ref, LineageKind.SUCCESSOR_OF))
        self._participants[ref] = state
        self._active_by_key[spec.key] = ref
        self._advance()
        return ref

    def _declare_relationship(
        self,
        spec: RelationshipSpec,
        *,
        source: ParticipantRef,
        target: ParticipantRef,
    ) -> RelationshipRef:
        if spec.key in self._relationship_by_key:
            raise ContractViolation(f"relationship key {spec.key!r} is already declared")
        ref = RelationshipRef(self.authority_id, spec.key, self._next_relationship)
        self._next_relationship += 1
        self._relationships[ref] = _RelationshipState(
            ref=ref,
            spec=spec,
            source=source,
            target=target,
        )
        self._relationship_by_key[spec.key] = ref
        self._advance()
        return ref

    def _invalidate_prepared_state(self, state: _ParticipantState) -> None:
        if state.prepared_view is not None or state.preparation_status is PreparationStatus.PREPARED:
            state.prepared_view = None
            state.preparation_revision += 1
        for route in state.routes.values():
            if route.valid or route.candidate is not None:
                route.valid = False
                route.candidate = None
                route.revision += 1
        if state.preparation_status is not PreparationStatus.INVALID:
            state.preparation_status = PreparationStatus.UNPREPARED

    def _invalidate_relationships_for(self, ref: ParticipantRef) -> None:
        for relationship in self._relationships.values():
            # Conservative invalidation requires resolution and activation
            # again, but an explicitly detached relationship stays closed.
            if (relationship.source == ref or relationship.target == ref) and relationship.lifecycle not in {
                RelationshipLifecycle.DECLARED,
                RelationshipLifecycle.DETACHED,
            }:
                relationship.lifecycle = RelationshipLifecycle.DECLARED
                relationship.revision += 1
                self._invalidate_relationship_endpoints(relationship)

    def _invalidate_relationship_endpoints(self, relationship: _RelationshipState) -> None:
        # Relationship state participates in the meaning of prepared endpoint
        # views, so a published route cannot outlive a relationship revision.
        self._invalidate_prepared_state(self._participants[relationship.source])
        self._invalidate_prepared_state(self._participants[relationship.target])

    def _ensure_related_relationships_not_preparing(self, ref: ParticipantRef) -> None:
        for relationship in self._relationships.values():
            if relationship.source == ref or relationship.target == ref:
                self._ensure_relationship_endpoints_not_preparing(relationship)

    def _ensure_relationship_endpoints_not_preparing(
        self,
        relationship: _RelationshipState,
    ) -> None:
        self._ensure_not_preparing(self._participants[relationship.source])
        self._ensure_not_preparing(self._participants[relationship.target])

    @staticmethod
    def _ensure_not_preparing(state: _ParticipantState) -> None:
        if state.preparing_attempt is not None:
            raise TransitionRejected(f"participant {state.spec.key!r} is under destructive preparation")

    def _issued_attempt(self, job: PreparationJob) -> _AttemptState:
        if job.attempt.authority_id != self.authority_id:
            raise TransitionRejected("preparation attempt belongs to another authority")
        try:
            attempt = self._attempts[job.attempt]
        except KeyError as error:
            raise TransitionRejected("preparation attempt was not issued by this authority") from error
        if attempt.status is not AttemptStatus.ISSUED:
            raise TransitionRejected(f"preparation attempt is already {attempt.status.value}")
        return attempt

    def _participant_state(self, ref: ParticipantRef) -> _ParticipantState:
        if ref.authority_id != self.authority_id:
            raise TransitionRejected("participant reference belongs to another authority")
        try:
            return self._participants[ref]
        except KeyError as error:
            raise TransitionRejected("participant reference is not known to this authority") from error

    def _current_participant_state(self, ref: ParticipantRef) -> _ParticipantState:
        state = self._participant_state(ref)
        if state.lifecycle is ParticipantLifecycle.RETIRED:
            raise TransitionRejected("retired participant references cannot be used for current transitions")
        return state

    def _bound_participant_state(self, ref: ParticipantRef) -> _ParticipantState:
        state = self._current_participant_state(ref)
        if state.lifecycle is not ParticipantLifecycle.BOUND or state.binding is None:
            raise CurrentStateUnavailable(f"participant {state.spec.key!r} is not materialized")
        return state

    def _relationship_state(self, ref: RelationshipRef) -> _RelationshipState:
        if ref.authority_id != self.authority_id:
            raise TransitionRejected("relationship reference belongs to another authority")
        try:
            return self._relationships[ref]
        except KeyError as error:
            raise TransitionRejected("relationship reference is not known to this authority") from error

    def _advance(self) -> None:
        self.revision += 1


class TrainingStrategy:
    """Complete accepted strategy presented to the Trainer-facing pipeline."""

    def __init__(
        self,
        *,
        contract: TrainingContract,
        definition: StrategyDefinition,
        authority: BindingAuthority,
        _establishment_token: object,
    ) -> None:
        if _establishment_token is not _ESTABLISHMENT_TOKEN:
            raise TypeError("TrainingStrategy must be created by contract establishment")
        self.contract = contract
        self.definition = definition
        self.authority = authority

    def ref(self, key: str) -> ParticipantRef:
        return self.authority.active_ref(key)

    def relationship_ref(self, key: str) -> RelationshipRef:
        return self.authority.relationship_ref(key)

    def materialize(self, key: str, candidate: MaterializedCandidate) -> ParticipantRef:
        ref = self.ref(key)
        self.authority.materialize(ref, candidate)
        return ref

    def materialize_from_artifact(
        self,
        key: str,
        candidate: MaterializedCandidate,
        artifact: Artifact,
    ) -> ParticipantRef:
        ref = self.ref(key)
        self.authority.materialize(
            ref,
            candidate,
            lineage_source=artifact.ref,
            lineage_kind=LineageKind.DERIVED_FROM_ARTIFACT,
        )
        return ref

    def redeclare_successor(self, retired_ref: ParticipantRef, spec: ParticipantSpec) -> ParticipantRef:
        old = self.authority.participant(retired_ref)
        if old.lifecycle is not ParticipantLifecycle.RETIRED:
            raise TransitionRejected("successor redeclaration requires a retired reference")
        if old.key != spec.key:
            raise ContractViolation("this spike only redeclares a successor at the same authored key")
        if any(
            relationship.source_key == spec.key or relationship.target_key == spec.key for relationship in self.definition.relationships
        ):
            raise TransitionRejected("participant succession with relationships requires an explicit relationship amendment")
        new_definition = StrategyDefinition(
            participants=tuple(spec if participant.key == spec.key else participant for participant in self.definition.participants),
            relationships=self.definition.relationships,
        )
        obligations = self.contract._validate_definition(new_definition)[spec.key]
        ref = self.authority._declare_participant(
            spec,
            obligations,
            lineage_source=retired_ref,
        )
        self.definition = new_definition
        return ref

    def prepare(
        self,
        refs: Iterable[ParticipantRef] | None = None,
        *,
        behavior: PreparationBehavior = PreparationBehavior.REPLACEMENT_ONLY,
        runtime: TrainerRuntime | None = None,
    ) -> PreparationJob:
        inseparable_groups: tuple[frozenset[ParticipantRef], ...] = ()
        if behavior is PreparationBehavior.IN_PLACE:
            if runtime is None:
                raise TransitionRejected("destructive preparation requires the Trainer runtime's current backend groups")
            inseparable_groups = runtime._inseparable_groups_for(self.authority.authority_id)
        return self.authority.begin_preparation(
            refs,
            behavior=behavior,
            inseparable_groups=inseparable_groups,
        )

    def capture_artifact(self, artifact_id: str, refs: Iterable[ParticipantRef]) -> Artifact:
        members: list[ArtifactMember] = []
        for ref in refs:
            participant = self.authority.participant(ref)
            if participant.lifecycle is not ParticipantLifecycle.BOUND:
                raise CurrentStateUnavailable("artifacts can only capture current bound participants")
            if (
                participant.preparation_status
                in {
                    PreparationStatus.PREPARING,
                    PreparationStatus.INVALID,
                }
                or participant.preparing_attempt is not None
            ):
                raise CurrentStateUnavailable("artifacts cannot capture a participant whose current guarantees are withdrawn")
            if ParticipantUse.PERSIST not in participant.uses:
                raise ContractViolation(f"participant {participant.key!r} was not declared as part of the persistent product")
            members.append(
                ArtifactMember(
                    participant=ref,
                    binding_revision=participant.binding_revision,
                    preparation_revision=participant.preparation_revision,
                )
            )
        return Artifact(
            ref=ArtifactRef(artifact_id),
            producing_authority_id=self.authority.authority_id,
            authority_revision=self.authority.revision,
            members=tuple(members),
        )

    def export_resume_state(self) -> StrategyResumeState:
        return StrategyResumeState(
            contract_version=self.contract.version,
            definition=self.definition,
            authority=self.authority.export_resume_state(),
        )


def retire_participant(
    authority: BindingAuthority,
    runtime: TrainerRuntime,
    ref: ParticipantRef,
) -> None:
    """Retire identity and invalidate any inseparable backend group together."""

    # Runtime previews first because it alone owns backend-group coverage; its
    # pure result supplies the complete authority invalidation set.
    runtime_update = runtime._preview_retirement(ref)
    authority_update = authority.preview_retirement(
        ref,
        invalidate_prepared=runtime_update.invalidate_prepared,
    )
    authority._install_retirement(authority_update)
    runtime._install(runtime_update.candidate)


def publish_preparation(
    authority: BindingAuthority,
    runtime: TrainerRuntime,
    result: PreparationResult,
) -> None:
    """Validate both sides before an assignment-only final publication."""

    authority_candidate = authority.preview_preparation(result)
    runtime_candidate = runtime._preview_install(result)
    authority._install_preparation(authority_candidate)
    runtime._install(runtime_candidate)


def capture_resume(strategy: TrainingStrategy, runtime: TrainerRuntime) -> RunResumeSnapshot:
    if strategy.authority.authority_id != runtime.authority_id:
        raise TransitionRejected("strategy and Trainer runtime belong to different authorities")
    return RunResumeSnapshot(
        strategy=strategy.export_resume_state(),
        trainer=runtime.export_resume_state(),
    )


def restore_resume(
    contract: TrainingContract,
    snapshot: RunResumeSnapshot,
    *,
    execution_session_id: str,
) -> tuple[TrainingStrategy, TrainerRuntime]:
    strategy = contract.restore_strategy(snapshot.strategy)
    runtime = TrainerRuntime.restore(snapshot.trainer, execution_session_id)
    if strategy.authority.authority_id != runtime.authority_id:
        raise ContractViolation("resume snapshot mixes strategy and Trainer authorities")
    return strategy, runtime
