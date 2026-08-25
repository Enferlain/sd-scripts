"""Executable scenarios for the isolated contract-exchange spike.

Run directly so the spike stays outside the repository's production test
suite::

    uv run python docs_design/models-strategy-trainer/executable_spike/scenarios.py
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from exchange import (
    ArtifactRef,
    ContractViolation,
    CurrentStateUnavailable,
    KnownImplementation,
    LineageKind,
    MaterializedCandidate,
    ParticipantLifecycle,
    ParticipantRef,
    ParticipantSpec,
    ParticipantUse,
    PreparationBehavior,
    PreparationJob,
    PreparationResult,
    PreparationStatus,
    PreparedParticipant,
    RelationshipKind,
    RelationshipLifecycle,
    RelationshipSpec,
    RuntimeTrait,
    StalePreparationResult,
    StrategyDefinition,
    TrainerRuntime,
    TrainingContract,
    TrainingStrategy,
    TransitionRejected,
    capture_resume,
    publish_preparation,
    retire_participant,
    restore_resume,
)


class BackendFailure(RuntimeError):
    pass


@dataclass(slots=True)
class FakeComponent:
    name: str
    source_revision: str
    mutations: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PreparedWrapper:
    source: object
    backend: str


@dataclass(frozen=True, slots=True)
class BackendState:
    backend: str
    participants: tuple[ParticipantRef, ...]
    ranks: int


class FakeBackend:
    """Replacement-only or destructive backend with observable failure points."""

    def __init__(self, name: str = "accelerate", *, ranks: int = 1) -> None:
        self.name = name
        self.ranks = ranks

    def prepare(self, job: PreparationJob, *, fail: bool = False) -> PreparationResult:
        if fail and job.behavior is PreparationBehavior.REPLACEMENT_ONLY:
            raise BackendFailure("replacement-only preparation failed before publication")

        participants: list[PreparedParticipant] = []
        for entry in job.participants:
            if job.behavior is PreparationBehavior.IN_PLACE:
                source = entry.source.value
                if not isinstance(source, FakeComponent):
                    raise TypeError("the fake in-place backend expects a FakeComponent")
                source.mutations.append(f"prepared-in-place:{self.name}")
                prepared_value: object = source
            else:
                prepared_value = PreparedWrapper(source=entry.source.value, backend=self.name)

            participants.append(
                PreparedParticipant(
                    ref=entry.ref,
                    source_binding_revision=entry.source_binding_revision,
                    source_preparation_revision=entry.source_preparation_revision,
                    candidate=MaterializedCandidate(
                        implementation_id=entry.source.implementation_id,
                        value=prepared_value,
                    ),
                    routes=entry.required_routes,
                )
            )

        if fail:
            raise BackendFailure("destructive preparation failed after mutation")

        return PreparationResult(
            job=job,
            participants=tuple(participants),
            backend_state=BackendState(
                backend=self.name,
                participants=tuple(entry.ref for entry in job.participants),
                ranks=self.ranks,
            ),
        )


def make_contract() -> TrainingContract:
    return TrainingContract(
        version="spike-v1",
        implementations=(
            KnownImplementation(
                "sdxl.denoiser",
                frozenset(
                    {
                        RuntimeTrait.CALLABLE,
                        RuntimeTrait.PARAMETERIZED,
                        RuntimeTrait.SERIALIZABLE,
                        RuntimeTrait.EFFECT_HOST,
                    }
                ),
            ),
            KnownImplementation(
                "sd3.denoiser",
                frozenset(
                    {
                        RuntimeTrait.CALLABLE,
                        RuntimeTrait.PARAMETERIZED,
                        RuntimeTrait.SERIALIZABLE,
                        RuntimeTrait.EFFECT_HOST,
                    }
                ),
            ),
            KnownImplementation(
                "text_encoder",
                frozenset(
                    {
                        RuntimeTrait.CALLABLE,
                        RuntimeTrait.PARAMETERIZED,
                        RuntimeTrait.SERIALIZABLE,
                    }
                ),
            ),
            KnownImplementation(
                "vae",
                frozenset({RuntimeTrait.CALLABLE, RuntimeTrait.SERIALIZABLE}),
            ),
            KnownImplementation(
                "adapter",
                frozenset(
                    {
                        RuntimeTrait.PARAMETERIZED,
                        RuntimeTrait.SERIALIZABLE,
                        RuntimeTrait.EFFECT_STATE,
                    }
                ),
            ),
            KnownImplementation(
                "teacher",
                frozenset({RuntimeTrait.CALLABLE, RuntimeTrait.SERIALIZABLE}),
            ),
            KnownImplementation(
                "student",
                frozenset(
                    {
                        RuntimeTrait.CALLABLE,
                        RuntimeTrait.PARAMETERIZED,
                        RuntimeTrait.SERIALIZABLE,
                        RuntimeTrait.EFFECT_HOST,
                    }
                ),
            ),
            KnownImplementation("decoder-only", frozenset({RuntimeTrait.SERIALIZABLE})),
        ),
    )


def candidate(
    implementation_id: str,
    name: str,
    *,
    source_revision: str = "source/main",
) -> MaterializedCandidate:
    return MaterializedCandidate(
        implementation_id=implementation_id,
        value=FakeComponent(name=name, source_revision=source_revision),
    )


def single_denoiser_definition(*, implementation_use: ParticipantUse = ParticipantUse.OPTIMIZE) -> StrategyDefinition:
    return StrategyDefinition(
        participants=(
            ParticipantSpec(
                "model.denoiser",
                frozenset(
                    {
                        ParticipantUse.EXECUTE,
                        implementation_use,
                        ParticipantUse.PERSIST,
                    }
                ),
            ),
        )
    )


def prepare_single_denoiser(
    *,
    authority_id: str = "run-A",
    backend_name: str = "accelerate",
) -> tuple[TrainingContract, TrainingStrategy, TrainerRuntime, ParticipantRef]:
    contract = make_contract()
    strategy = contract.establish(single_denoiser_definition(), authority_id)
    ref = strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "denoiser"))
    runtime = TrainerRuntime(authority_id, execution_session_id=f"{authority_id}/process-1")
    result = FakeBackend(backend_name).prepare(strategy.prepare())
    publish_preparation(strategy.authority, runtime, result)
    return contract, strategy, runtime, ref


class ContractExchangeScenarios(unittest.TestCase):
    def test_ordinary_multi_component_training_reaches_prepared_trainer_views(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec(
                    "model.denoiser",
                    frozenset(
                        {
                            ParticipantUse.EXECUTE,
                            ParticipantUse.OPTIMIZE,
                            ParticipantUse.PERSIST,
                        }
                    ),
                ),
                ParticipantSpec(
                    "conditioning.text_encoder",
                    frozenset({ParticipantUse.EXECUTE, ParticipantUse.PERSIST}),
                ),
                ParticipantSpec(
                    "encoding.vae",
                    frozenset({ParticipantUse.EXECUTE, ParticipantUse.PERSIST}),
                ),
            )
        )
        strategy = contract.establish(definition, "ordinary-run")
        denoiser_ref = strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "denoiser"))
        text_encoder_ref = strategy.materialize("conditioning.text_encoder", candidate("text_encoder", "text-encoder"))
        vae_ref = strategy.materialize("encoding.vae", candidate("vae", "vae"))

        job = strategy.prepare()
        self.assertEqual(
            {entry.ref for entry in job.participants},
            {denoiser_ref, text_encoder_ref, vae_ref},
        )
        self.assertNotIn(
            ParticipantUse.OPTIMIZE,
            strategy.authority.participant(text_encoder_ref).uses,
        )

        runtime = TrainerRuntime("ordinary-run", "ordinary-run/process-1")
        result = FakeBackend().prepare(job)
        publish_preparation(strategy.authority, runtime, result)

        self.assertIsInstance(strategy.authority.execution_route(denoiser_ref), PreparedWrapper)
        self.assertIsInstance(strategy.authority.execution_route(text_encoder_ref), PreparedWrapper)
        self.assertIsInstance(strategy.authority.execution_route(vae_ref), PreparedWrapper)
        self.assertEqual(strategy.ref("model.denoiser"), denoiser_ref)
        self.assertIn(job.attempt, runtime.published_attempts)

    def test_deferred_participant_is_declared_before_materialization(self) -> None:
        contract = make_contract()
        strategy = contract.establish(single_denoiser_definition(), "deferred-run")
        ref = strategy.ref("model.denoiser")

        self.assertEqual(
            strategy.authority.participant(ref).lifecycle,
            ParticipantLifecycle.DECLARED,
        )
        with self.assertRaises(CurrentStateUnavailable):
            strategy.prepare()

        strategy.materialize("model.denoiser", candidate("sd3.denoiser", "deferred-denoiser"))
        runtime = TrainerRuntime("deferred-run", "deferred-run/process-1")
        publish_preparation(
            strategy.authority,
            runtime,
            FakeBackend().prepare(strategy.prepare()),
        )

        self.assertEqual(strategy.ref("model.denoiser"), ref)
        self.assertEqual(
            strategy.authority.participant(ref).preparation_status,
            PreparationStatus.PREPARED,
        )

    def test_adapter_is_prepared_without_becoming_an_execution_route(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec(
                    "model.denoiser",
                    frozenset({ParticipantUse.EXECUTE, ParticipantUse.PERSIST}),
                ),
                ParticipantSpec(
                    "adaptation.main",
                    frozenset({ParticipantUse.OPTIMIZE, ParticipantUse.PERSIST}),
                ),
            ),
            relationships=(
                RelationshipSpec(
                    "adaptation.main->model.denoiser",
                    source_key="adaptation.main",
                    target_key="model.denoiser",
                    kind=RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "adapter-run")
        host_ref = strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "host-denoiser"))
        adapter_ref = strategy.materialize("adaptation.main", candidate("adapter", "adapter"))
        relationship_ref = strategy.relationship_ref("adaptation.main->model.denoiser")
        strategy.authority.resolve_relationship(relationship_ref)
        strategy.authority.activate_relationship(relationship_ref)

        job = strategy.prepare()
        runtime = TrainerRuntime("adapter-run", "adapter-run/process-1")
        publish_preparation(strategy.authority, runtime, FakeBackend().prepare(job))

        self.assertEqual(
            strategy.authority.relationship(relationship_ref).lifecycle,
            RelationshipLifecycle.ACTIVE,
        )
        self.assertIsInstance(strategy.authority.execution_route(host_ref), PreparedWrapper)
        self.assertIsInstance(strategy.authority.prepared_view(adapter_ref), PreparedWrapper)
        self.assertEqual(strategy.authority.participant(adapter_ref).routes, ())
        with self.assertRaises(CurrentStateUnavailable):
            strategy.authority.execution_route(adapter_ref)
        backend_state = runtime.backend_state_for(strategy.authority, (host_ref, adapter_ref))
        self.assertIsInstance(backend_state, BackendState)
        assert isinstance(backend_state, BackendState)
        self.assertEqual(set(backend_state.participants), {host_ref, adapter_ref})

    def test_attachment_obligations_are_derived_from_the_complete_arrangement(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.host", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("adaptation.main", frozenset({ParticipantUse.OPTIMIZE})),
            ),
            relationships=(
                RelationshipSpec(
                    "adapter-host",
                    "adaptation.main",
                    "model.host",
                    RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "derived-obligation-run")

        # The teacher implementation is callable, but it is not a known effect
        # host.  The relationship therefore adds an obligation that the host's
        # direct EXECUTE use alone would not have supplied.
        with self.assertRaises(ContractViolation):
            strategy.materialize("model.host", candidate("teacher", "not-an-effect-host"))

    def test_successor_with_relationship_requires_explicit_relationship_amendment(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.denoiser", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("adaptation.main", frozenset({ParticipantUse.OPTIMIZE})),
            ),
            relationships=(
                RelationshipSpec(
                    "adapter-host",
                    "adaptation.main",
                    "model.denoiser",
                    RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "amendment-run")
        strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "denoiser"))
        old_ref = strategy.ref("model.denoiser")
        runtime = TrainerRuntime("amendment-run", "amendment-run/process-1")
        retire_participant(strategy.authority, runtime, old_ref)

        with self.assertRaises(TransitionRejected):
            strategy.redeclare_successor(
                old_ref,
                ParticipantSpec("model.denoiser", frozenset({ParticipantUse.EXECUTE})),
            )

    def test_compound_teacher_student_adapter_keeps_shared_source_and_identity_separate(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec(
                    "teacher.denoiser",
                    frozenset({ParticipantUse.EXECUTE, ParticipantUse.PERSIST}),
                ),
                ParticipantSpec(
                    "student.denoiser",
                    frozenset(
                        {
                            ParticipantUse.EXECUTE,
                            ParticipantUse.OPTIMIZE,
                            ParticipantUse.PERSIST,
                        }
                    ),
                ),
                ParticipantSpec(
                    "adaptation.student",
                    frozenset({ParticipantUse.OPTIMIZE, ParticipantUse.PERSIST}),
                ),
            ),
            relationships=(
                RelationshipSpec(
                    "adaptation.student->student.denoiser",
                    source_key="adaptation.student",
                    target_key="student.denoiser",
                    kind=RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "research-run")
        shared_source = "catalog/base-model/revision-7"
        teacher_candidate = candidate("teacher", "teacher", source_revision=shared_source)
        student_candidate = candidate("student", "student", source_revision=shared_source)
        teacher_ref = strategy.materialize(
            "teacher.denoiser",
            teacher_candidate,
        )
        student_ref = strategy.materialize(
            "student.denoiser",
            student_candidate,
        )
        adapter_ref = strategy.materialize(
            "adaptation.student",
            candidate("adapter", "student-adapter", source_revision="adapter/revision-1"),
        )
        relationship_ref = strategy.relationship_ref("adaptation.student->student.denoiser")
        strategy.authority.resolve_relationship(relationship_ref)
        strategy.authority.activate_relationship(relationship_ref)

        runtime = TrainerRuntime("research-run", "research-run/rank-0")
        result = FakeBackend("deepspeed", ranks=2).prepare(strategy.prepare())
        publish_preparation(strategy.authority, runtime, result)
        artifact = strategy.capture_artifact(
            "artifact/research-run/step-100",
            (teacher_ref, student_ref, adapter_ref),
        )

        self.assertNotEqual(teacher_ref, student_ref)
        assert isinstance(teacher_candidate.value, FakeComponent)
        assert isinstance(student_candidate.value, FakeComponent)
        self.assertEqual(teacher_candidate.value.source_revision, shared_source)
        self.assertEqual(student_candidate.value.source_revision, shared_source)
        self.assertIsInstance(strategy.authority.execution_route(teacher_ref), PreparedWrapper)
        self.assertIsInstance(strategy.authority.execution_route(student_ref), PreparedWrapper)
        self.assertEqual(strategy.authority.participant(adapter_ref).routes, ())
        backend_state = runtime.backend_state_for(
            strategy.authority,
            (teacher_ref, student_ref, adapter_ref),
        )
        self.assertIsInstance(backend_state, BackendState)
        assert isinstance(backend_state, BackendState)
        self.assertEqual(backend_state.ranks, 2)
        self.assertEqual(
            {member.participant for member in artifact.members},
            {teacher_ref, student_ref, adapter_ref},
        )
        self.assertNotIsInstance(backend_state, ParticipantRef)

    def test_compatible_replacement_preserves_ref_and_rejects_stale_preparation(self) -> None:
        _, strategy, runtime, ref = prepare_single_denoiser()
        stale_job = strategy.prepare()
        stale_result = FakeBackend().prepare(stale_job)
        runtime_revision = runtime.revision

        strategy.authority.replace_binding(
            ref,
            candidate("sdxl.denoiser", "replacement-denoiser", source_revision="source/new"),
        )

        self.assertEqual(strategy.ref("model.denoiser"), ref)
        self.assertEqual(strategy.authority.participant(ref).binding_revision, 2)
        with self.assertRaises(CurrentStateUnavailable):
            strategy.authority.execution_route(ref)
        with self.assertRaises(StalePreparationResult):
            publish_preparation(strategy.authority, runtime, stale_result)
        self.assertEqual(runtime.revision, runtime_revision)

    def test_relationship_change_makes_preparation_result_stale(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.denoiser", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("adaptation.main", frozenset({ParticipantUse.OPTIMIZE})),
            ),
            relationships=(
                RelationshipSpec(
                    "adapter-host",
                    "adaptation.main",
                    "model.denoiser",
                    RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "relationship-run")
        strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "denoiser"))
        strategy.materialize("adaptation.main", candidate("adapter", "adapter"))
        relationship_ref = strategy.relationship_ref("adapter-host")
        strategy.authority.resolve_relationship(relationship_ref)
        strategy.authority.activate_relationship(relationship_ref)
        job = strategy.prepare()
        result = FakeBackend().prepare(job)

        strategy.authority.deactivate_relationship(relationship_ref)

        runtime = TrainerRuntime("relationship-run", "relationship-run/process-1")
        with self.assertRaises(StalePreparationResult):
            publish_preparation(strategy.authority, runtime, result)
        self.assertEqual(runtime.revision, 0)

    def test_relationship_change_withdraws_already_published_endpoint_views(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.denoiser", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("adaptation.main", frozenset({ParticipantUse.OPTIMIZE})),
            ),
            relationships=(
                RelationshipSpec(
                    "adapter-host",
                    "adaptation.main",
                    "model.denoiser",
                    RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "published-relationship-run")
        host_ref = strategy.materialize(
            "model.denoiser",
            candidate("sdxl.denoiser", "denoiser"),
        )
        adapter_ref = strategy.materialize(
            "adaptation.main",
            candidate("adapter", "adapter"),
        )
        relationship_ref = strategy.relationship_ref("adapter-host")
        strategy.authority.resolve_relationship(relationship_ref)
        strategy.authority.activate_relationship(relationship_ref)
        runtime = TrainerRuntime(
            "published-relationship-run",
            "published-relationship-run/process-1",
        )
        publish_preparation(
            strategy.authority,
            runtime,
            FakeBackend().prepare(strategy.prepare()),
        )

        strategy.authority.deactivate_relationship(relationship_ref)

        with self.assertRaises(CurrentStateUnavailable):
            strategy.authority.execution_route(host_ref)
        with self.assertRaises(CurrentStateUnavailable):
            strategy.authority.prepared_view(adapter_ref)
        self.assertEqual(
            strategy.authority.participant(host_ref).preparation_status,
            PreparationStatus.UNPREPARED,
        )
        self.assertEqual(
            strategy.authority.participant(adapter_ref).preparation_status,
            PreparationStatus.UNPREPARED,
        )

    def test_relationship_cannot_change_while_an_endpoint_is_mutating(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec(
                    "model.denoiser",
                    frozenset({ParticipantUse.EXECUTE, ParticipantUse.PERSIST}),
                ),
                ParticipantSpec(
                    "adaptation.main",
                    frozenset({ParticipantUse.OPTIMIZE}),
                ),
            ),
            relationships=(
                RelationshipSpec(
                    "adapter-host",
                    "adaptation.main",
                    "model.denoiser",
                    RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "mutating-relationship-run")
        host_ref = strategy.materialize(
            "model.denoiser",
            candidate("sdxl.denoiser", "denoiser"),
        )
        strategy.materialize("adaptation.main", candidate("adapter", "adapter"))
        relationship_ref = strategy.relationship_ref("adapter-host")
        strategy.authority.resolve_relationship(relationship_ref)
        strategy.authority.activate_relationship(relationship_ref)
        runtime = TrainerRuntime(
            "mutating-relationship-run",
            "mutating-relationship-run/process-1",
        )
        job = strategy.prepare(
            (host_ref,),
            behavior=PreparationBehavior.IN_PLACE,
            runtime=runtime,
        )

        with self.assertRaises(TransitionRejected):
            strategy.authority.deactivate_relationship(relationship_ref)
        self.assertEqual(
            strategy.authority.participant(host_ref).preparation_status,
            PreparationStatus.PREPARING,
        )
        with self.assertRaises(CurrentStateUnavailable):
            strategy.capture_artifact("artifact/mid-mutation", (host_ref,))

        strategy.authority.abandon_preparation(job)

    def test_endpoint_replacement_withdraws_the_other_endpoint_view(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.denoiser", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("adaptation.main", frozenset({ParticipantUse.OPTIMIZE})),
            ),
            relationships=(
                RelationshipSpec(
                    "adapter-host",
                    "adaptation.main",
                    "model.denoiser",
                    RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "relationship-replacement-run")
        host_ref = strategy.materialize(
            "model.denoiser",
            candidate("sdxl.denoiser", "denoiser"),
        )
        adapter_ref = strategy.materialize(
            "adaptation.main",
            candidate("adapter", "adapter"),
        )
        relationship_ref = strategy.relationship_ref("adapter-host")
        strategy.authority.resolve_relationship(relationship_ref)
        strategy.authority.activate_relationship(relationship_ref)
        runtime = TrainerRuntime(
            "relationship-replacement-run",
            "relationship-replacement-run/process-1",
        )
        publish_preparation(
            strategy.authority,
            runtime,
            FakeBackend().prepare(strategy.prepare()),
        )

        strategy.authority.replace_binding(
            adapter_ref,
            candidate("adapter", "replacement-adapter"),
        )

        self.assertEqual(
            strategy.authority.relationship(relationship_ref).lifecycle,
            RelationshipLifecycle.DECLARED,
        )
        with self.assertRaises(CurrentStateUnavailable):
            strategy.authority.execution_route(host_ref)
        with self.assertRaises(CurrentStateUnavailable):
            runtime.backend_state_for(
                strategy.authority,
                (host_ref, adapter_ref),
            )

    def test_endpoint_retirement_withdraws_a_disjoint_survivor_view(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.denoiser", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("adaptation.main", frozenset({ParticipantUse.OPTIMIZE})),
            ),
            relationships=(
                RelationshipSpec(
                    "adapter-host",
                    "adaptation.main",
                    "model.denoiser",
                    RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "relationship-retirement-run")
        host_ref = strategy.materialize(
            "model.denoiser",
            candidate("sdxl.denoiser", "denoiser"),
        )
        adapter_ref = strategy.materialize(
            "adaptation.main",
            candidate("adapter", "adapter"),
        )
        relationship_ref = strategy.relationship_ref("adapter-host")
        strategy.authority.resolve_relationship(relationship_ref)
        strategy.authority.activate_relationship(relationship_ref)
        runtime = TrainerRuntime(
            "relationship-retirement-run",
            "relationship-retirement-run/process-1",
        )
        publish_preparation(
            strategy.authority,
            runtime,
            FakeBackend().prepare(strategy.prepare((host_ref,))),
        )

        retire_participant(strategy.authority, runtime, adapter_ref)

        self.assertEqual(
            strategy.authority.relationship(relationship_ref).lifecycle,
            RelationshipLifecycle.DETACHED,
        )
        with self.assertRaises(CurrentStateUnavailable):
            strategy.authority.execution_route(host_ref)
        with self.assertRaises(CurrentStateUnavailable):
            runtime.backend_state_for(strategy.authority, (host_ref,))

    def test_detached_relationship_stays_closed_across_endpoint_replacement(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.denoiser", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("adaptation.main", frozenset({ParticipantUse.OPTIMIZE})),
            ),
            relationships=(
                RelationshipSpec(
                    "adapter-host",
                    "adaptation.main",
                    "model.denoiser",
                    RelationshipKind.ATTACHMENT,
                ),
            ),
        )
        strategy = contract.establish(definition, "detach-run")
        host_ref = strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "denoiser"))
        strategy.materialize("adaptation.main", candidate("adapter", "adapter"))
        relationship_ref = strategy.relationship_ref("adapter-host")
        strategy.authority.resolve_relationship(relationship_ref)
        strategy.authority.activate_relationship(relationship_ref)
        result = FakeBackend().prepare(strategy.prepare())

        strategy.authority.detach_relationship(relationship_ref)

        runtime = TrainerRuntime("detach-run", "detach-run/process-1")
        with self.assertRaises(StalePreparationResult):
            publish_preparation(strategy.authority, runtime, result)
        strategy.authority.replace_binding(
            host_ref,
            candidate("sdxl.denoiser", "replacement-denoiser"),
        )
        self.assertEqual(
            strategy.authority.relationship(relationship_ref).lifecycle,
            RelationshipLifecycle.DETACHED,
        )
        with self.assertRaises(TransitionRejected):
            strategy.authority.detach_relationship(relationship_ref)

    def test_retirement_makes_an_in_flight_result_explicitly_stale(self) -> None:
        contract = make_contract()
        strategy = contract.establish(single_denoiser_definition(), "retirement-run")
        ref = strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "denoiser"))
        result = FakeBackend().prepare(strategy.prepare())
        runtime = TrainerRuntime("retirement-run", "retirement-run/process-1")
        retire_participant(strategy.authority, runtime, ref)
        runtime_revision = runtime.revision

        with self.assertRaises(StalePreparationResult):
            publish_preparation(strategy.authority, runtime, result)
        self.assertEqual(runtime.revision, runtime_revision)

    def test_replacement_only_backend_failure_preserves_published_state(self) -> None:
        _, strategy, runtime, ref = prepare_single_denoiser()
        old_route = strategy.authority.execution_route(ref)
        old_runtime_state = runtime.backend_state_for(strategy.authority, (ref,))
        old_snapshot = strategy.authority.participant(ref)
        job = strategy.prepare()

        with self.assertRaises(BackendFailure):
            FakeBackend().prepare(job, fail=True)
        strategy.authority.abandon_preparation(job)

        self.assertIs(strategy.authority.execution_route(ref), old_route)
        self.assertIs(
            runtime.backend_state_for(strategy.authority, (ref,)),
            old_runtime_state,
        )
        self.assertEqual(strategy.authority.participant(ref), old_snapshot)

    def test_destructive_failure_withdraws_routes_until_replacement(self) -> None:
        _, strategy, runtime, ref = prepare_single_denoiser()
        source_component = FakeComponent("destructively-prepared", "source/main")
        source = MaterializedCandidate("sdxl.denoiser", source_component)
        strategy.authority.replace_binding(ref, source)
        publish_preparation(
            strategy.authority,
            runtime,
            FakeBackend().prepare(strategy.prepare()),
        )
        job = strategy.prepare(
            behavior=PreparationBehavior.IN_PLACE,
            runtime=runtime,
        )

        self.assertEqual(
            strategy.authority.participant(ref).preparation_status,
            PreparationStatus.PREPARING,
        )
        with self.assertRaises(CurrentStateUnavailable):
            runtime.backend_state_for(strategy.authority, (ref,))
        with self.assertRaises(CurrentStateUnavailable):
            strategy.capture_artifact("artifact/unsafe", (ref,))
        with self.assertRaises(CurrentStateUnavailable):
            strategy.authority.execution_route(ref)
        with self.assertRaises(BackendFailure):
            FakeBackend("destructive").prepare(job, fail=True)
        strategy.authority.abandon_preparation(job)

        self.assertEqual(source_component.mutations, ["prepared-in-place:destructive"])
        self.assertEqual(
            strategy.authority.participant(ref).preparation_status,
            PreparationStatus.INVALID,
        )
        with self.assertRaises(CurrentStateUnavailable):
            runtime.backend_state_for(strategy.authority, (ref,))
        with self.assertRaises(CurrentStateUnavailable):
            strategy.capture_artifact("artifact/still-unsafe", (ref,))
        with self.assertRaises(CurrentStateUnavailable):
            strategy.authority.execution_route(ref)

        strategy.authority.replace_binding(ref, candidate("sdxl.denoiser", "recovered"))
        publish_preparation(
            strategy.authority,
            runtime,
            FakeBackend().prepare(strategy.prepare()),
        )
        self.assertEqual(
            strategy.authority.participant(ref).preparation_status,
            PreparationStatus.PREPARED,
        )

    def test_replacement_result_cannot_override_destructive_attempt_or_failure(self) -> None:
        contract = make_contract()

        active_strategy = contract.establish(single_denoiser_definition(), "active-destructive-run")
        active_ref = active_strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "active-denoiser"))
        active_runtime = TrainerRuntime("active-destructive-run", "active-destructive-run/process-1")
        old_result = FakeBackend().prepare(active_strategy.prepare())
        destructive_job = active_strategy.prepare(
            behavior=PreparationBehavior.IN_PLACE,
            runtime=active_runtime,
        )

        with self.assertRaises(StalePreparationResult):
            publish_preparation(active_strategy.authority, active_runtime, old_result)
        self.assertEqual(
            active_strategy.authority.participant(active_ref).preparation_status,
            PreparationStatus.PREPARING,
        )
        active_strategy.authority.abandon_preparation(destructive_job)
        self.assertEqual(
            active_strategy.authority.participant(active_ref).preparation_status,
            PreparationStatus.INVALID,
        )

        failed_strategy = contract.establish(single_denoiser_definition(), "failed-destructive-run")
        failed_ref = failed_strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "failed-denoiser"))
        failed_runtime = TrainerRuntime("failed-destructive-run", "failed-destructive-run/process-1")
        older_result = FakeBackend().prepare(failed_strategy.prepare())
        failed_job = failed_strategy.prepare(
            behavior=PreparationBehavior.IN_PLACE,
            runtime=failed_runtime,
        )
        with self.assertRaises(BackendFailure):
            FakeBackend("destructive").prepare(failed_job, fail=True)
        failed_strategy.authority.abandon_preparation(failed_job)

        with self.assertRaises(StalePreparationResult):
            publish_preparation(failed_strategy.authority, failed_runtime, older_result)
        self.assertEqual(
            failed_strategy.authority.participant(failed_ref).preparation_status,
            PreparationStatus.INVALID,
        )
        with self.assertRaises(TransitionRejected):
            failed_strategy.prepare()

        retry = failed_strategy.prepare(
            behavior=PreparationBehavior.IN_PLACE,
            runtime=failed_runtime,
        )
        publish_preparation(
            failed_strategy.authority,
            failed_runtime,
            FakeBackend("destructive-retry").prepare(retry),
        )
        self.assertEqual(
            failed_strategy.authority.participant(failed_ref).preparation_status,
            PreparationStatus.PREPARED,
        )

    def test_incompatible_replacement_requires_new_incarnation(self) -> None:
        _, strategy, runtime, old_ref = prepare_single_denoiser()
        before = strategy.authority.participant(old_ref)

        with self.assertRaises(ContractViolation):
            strategy.authority.replace_binding(
                old_ref,
                candidate("decoder-only", "not-a-denoiser"),
            )
        self.assertEqual(strategy.authority.participant(old_ref), before)

        retire_participant(strategy.authority, runtime, old_ref)
        successor_spec = ParticipantSpec(
            "model.denoiser",
            frozenset({ParticipantUse.PERSIST}),
        )
        new_ref = strategy.redeclare_successor(old_ref, successor_spec)
        strategy.materialize("model.denoiser", candidate("decoder-only", "new-semantic-participant"))

        self.assertNotEqual(old_ref, new_ref)
        self.assertEqual(
            strategy.authority.participant(old_ref).lifecycle,
            ParticipantLifecycle.RETIRED,
        )
        self.assertEqual(strategy.ref("model.denoiser"), new_ref)
        self.assertEqual(
            strategy.authority.participant(new_ref).lineage[0].kind,
            LineageKind.SUCCESSOR_OF,
        )

    def test_exact_resume_preserves_logical_identity_but_not_python_objects(self) -> None:
        contract, strategy, runtime, ref = prepare_single_denoiser()
        old_route = strategy.authority.execution_route(ref)
        old_participant = strategy.authority.participant(ref)
        snapshot = capture_resume(strategy, runtime)

        restored_strategy, restored_runtime = restore_resume(
            contract,
            snapshot,
            execution_session_id="run-A/process-2",
        )

        self.assertEqual(restored_strategy.ref("model.denoiser"), ref)
        self.assertEqual(restored_strategy.authority.participant(ref), old_participant)
        self.assertIsNot(restored_strategy.authority.execution_route(ref), old_route)
        self.assertEqual(restored_strategy.authority.authority_id, strategy.authority.authority_id)
        self.assertEqual(restored_runtime.revision, runtime.revision)
        self.assertNotEqual(restored_runtime.execution_session_id, runtime.execution_session_id)

    def test_artifact_capture_requires_declared_persistence(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec(
                    "model.ephemeral",
                    frozenset({ParticipantUse.EXECUTE}),
                ),
            )
        )
        strategy = contract.establish(definition, "ephemeral-run")
        ref = strategy.materialize(
            "model.ephemeral",
            candidate("teacher", "ephemeral"),
        )

        with self.assertRaises(ContractViolation):
            strategy.capture_artifact("artifact/invalid", (ref,))

    def test_new_run_from_artifact_gets_new_ref_with_artifact_lineage(self) -> None:
        contract, producing_strategy, _, producing_ref = prepare_single_denoiser()
        artifact = producing_strategy.capture_artifact("artifact/run-A/final", (producing_ref,))
        new_strategy = contract.establish(single_denoiser_definition(), "run-B")
        new_ref = new_strategy.materialize_from_artifact(
            "model.denoiser",
            candidate("sdxl.denoiser", "run-B-denoiser", source_revision=artifact.ref.value),
            artifact,
        )

        self.assertNotEqual(producing_ref, new_ref)
        self.assertEqual(new_ref.authority_id, "run-B")
        lineage = new_strategy.authority.participant(new_ref).lineage
        self.assertEqual(lineage[0].source, ArtifactRef("artifact/run-A/final"))
        self.assertEqual(lineage[0].kind, LineageKind.DERIVED_FROM_ARTIFACT)
        with self.assertRaises(TransitionRejected):
            new_strategy.authority.participant(producing_ref)

    def test_lineage_kind_and_source_type_are_enforced(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.source", frozenset({ParticipantUse.PERSIST})),
                ParticipantSpec("model.target", frozenset({ParticipantUse.PERSIST})),
            )
        )
        strategy = contract.establish(definition, "lineage-run")
        source_ref = strategy.materialize("model.source", candidate("decoder-only", "source"))
        target_ref = strategy.ref("model.target")
        target_candidate = candidate("decoder-only", "target")

        with self.assertRaises(ContractViolation):
            strategy.authority.materialize(
                target_ref,
                target_candidate,
                lineage_source=ArtifactRef("artifact/source"),
                lineage_kind=LineageKind.SUCCESSOR_OF,
            )
        with self.assertRaises(ContractViolation):
            strategy.authority.materialize(
                target_ref,
                target_candidate,
                lineage_source=source_ref,
                lineage_kind=LineageKind.DERIVED_FROM_ARTIFACT,
            )
        with self.assertRaises(TransitionRejected):
            strategy.authority.materialize(
                target_ref,
                target_candidate,
                lineage_source=source_ref,
                lineage_kind=LineageKind.SUCCESSOR_OF,
            )

    def test_malformed_result_is_rejected_before_either_owner_publishes(self) -> None:
        contract = make_contract()
        strategy = contract.establish(single_denoiser_definition(), "atomic-run")
        strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "denoiser"))
        runtime = TrainerRuntime("atomic-run", "atomic-run/process-1")
        job = strategy.prepare()
        good_result = FakeBackend().prepare(job)
        malformed = PreparationResult(
            job=job,
            participants=(),
            backend_state=good_result.backend_state,
        )
        authority_revision = strategy.authority.revision
        runtime_revision = runtime.revision

        with self.assertRaises(TransitionRejected):
            publish_preparation(strategy.authority, runtime, malformed)

        self.assertEqual(strategy.authority.revision, authority_revision)
        self.assertEqual(runtime.revision, runtime_revision)
        publish_preparation(strategy.authority, runtime, good_result)
        self.assertIn(job.attempt, runtime.published_attempts)

        authority_revision = strategy.authority.revision
        runtime_revision = runtime.revision
        with self.assertRaises(TransitionRejected):
            publish_preparation(strategy.authority, runtime, good_result)
        self.assertEqual(strategy.authority.revision, authority_revision)
        self.assertEqual(runtime.revision, runtime_revision)

    def test_prepared_state_cannot_publish_into_another_authoritys_trainer(self) -> None:
        contract = make_contract()
        strategy = contract.establish(single_denoiser_definition(), "source-run")
        strategy.materialize("model.denoiser", candidate("sdxl.denoiser", "denoiser"))
        result = FakeBackend().prepare(strategy.prepare())
        wrong_runtime = TrainerRuntime("other-run", "other-run/process-1")
        authority_revision = strategy.authority.revision

        with self.assertRaises(TransitionRejected):
            publish_preparation(strategy.authority, wrong_runtime, result)

        self.assertEqual(strategy.authority.revision, authority_revision)
        self.assertEqual(wrong_runtime.revision, 0)

    def test_narrow_result_cannot_orphan_an_existing_joint_backend_group(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.first", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("model.second", frozenset({ParticipantUse.EXECUTE})),
            )
        )
        strategy = contract.establish(definition, "joint-run")
        first_ref = strategy.materialize("model.first", candidate("teacher", "first"))
        second_ref = strategy.materialize("model.second", candidate("teacher", "second"))
        runtime = TrainerRuntime("joint-run", "joint-run/process-1")
        joint_result = FakeBackend("deepspeed", ranks=2).prepare(strategy.prepare())
        publish_preparation(strategy.authority, runtime, joint_result)

        narrow_result = FakeBackend().prepare(strategy.prepare((first_ref,)))
        authority_revision = strategy.authority.revision
        runtime_revision = runtime.revision
        with self.assertRaises(TransitionRejected):
            publish_preparation(strategy.authority, runtime, narrow_result)

        self.assertEqual(strategy.authority.revision, authority_revision)
        self.assertEqual(runtime.revision, runtime_revision)
        self.assertIsInstance(
            runtime.backend_state_for(strategy.authority, (first_ref, second_ref)),
            BackendState,
        )

        retire_participant(strategy.authority, runtime, first_ref)

        with self.assertRaises(CurrentStateUnavailable):
            runtime.backend_state_for(strategy.authority, (first_ref, second_ref))
        with self.assertRaises(CurrentStateUnavailable):
            strategy.authority.execution_route(second_ref)
        self.assertEqual(
            strategy.authority.participant(first_ref).lifecycle,
            ParticipantLifecycle.RETIRED,
        )

        rebuilt_result = FakeBackend().prepare(strategy.prepare((second_ref,)))
        publish_preparation(strategy.authority, runtime, rebuilt_result)
        self.assertIsInstance(
            runtime.backend_state_for(strategy.authority, (second_ref,)),
            BackendState,
        )

    def test_destructive_job_cannot_partially_cover_a_joint_backend_group(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.first", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("model.second", frozenset({ParticipantUse.EXECUTE})),
            )
        )
        strategy = contract.establish(definition, "destructive-joint-run")
        first_ref = strategy.materialize("model.first", candidate("teacher", "first"))
        second_ref = strategy.materialize("model.second", candidate("teacher", "second"))
        runtime = TrainerRuntime(
            "destructive-joint-run",
            "destructive-joint-run/process-1",
        )
        publish_preparation(
            strategy.authority,
            runtime,
            FakeBackend("deepspeed", ranks=2).prepare(strategy.prepare()),
        )
        first_route = strategy.authority.execution_route(first_ref)
        second_route = strategy.authority.execution_route(second_ref)

        with self.assertRaises(TransitionRejected):
            strategy.prepare(
                (first_ref,),
                behavior=PreparationBehavior.IN_PLACE,
                runtime=runtime,
            )

        self.assertIs(strategy.authority.execution_route(first_ref), first_route)
        self.assertIs(strategy.authority.execution_route(second_ref), second_route)
        self.assertEqual(
            strategy.authority.participant(first_ref).preparation_status,
            PreparationStatus.PREPARED,
        )
        self.assertEqual(
            strategy.authority.participant(second_ref).preparation_status,
            PreparationStatus.PREPARED,
        )

    def test_disjoint_backend_groups_can_coexist(self) -> None:
        contract = make_contract()
        definition = StrategyDefinition(
            participants=(
                ParticipantSpec("model.first", frozenset({ParticipantUse.EXECUTE})),
                ParticipantSpec("model.second", frozenset({ParticipantUse.EXECUTE})),
            )
        )
        strategy = contract.establish(definition, "disjoint-run")
        first_ref = strategy.materialize("model.first", candidate("teacher", "first"))
        second_ref = strategy.materialize("model.second", candidate("teacher", "second"))
        runtime = TrainerRuntime("disjoint-run", "disjoint-run/process-1")

        publish_preparation(
            strategy.authority,
            runtime,
            FakeBackend("backend-A").prepare(strategy.prepare((first_ref,))),
        )
        publish_preparation(
            strategy.authority,
            runtime,
            FakeBackend("backend-B").prepare(strategy.prepare((second_ref,))),
        )

        self.assertEqual(len(runtime.backend_groups()), 2)
        self.assertIsInstance(
            runtime.backend_state_for(strategy.authority, (first_ref,)),
            BackendState,
        )
        self.assertIsInstance(
            runtime.backend_state_for(strategy.authority, (second_ref,)),
            BackendState,
        )
        self.assertIsInstance(strategy.authority.execution_route(first_ref), PreparedWrapper)
        self.assertIsInstance(strategy.authority.execution_route(second_ref), PreparedWrapper)

        merged_result = FakeBackend("merged-backend").prepare(strategy.prepare())
        publish_preparation(strategy.authority, runtime, merged_result)

        self.assertEqual(len(runtime.backend_groups()), 1)
        with self.assertRaises(CurrentStateUnavailable):
            runtime.backend_state_for(strategy.authority, (first_ref,))
        self.assertIsInstance(
            runtime.backend_state_for(strategy.authority, (first_ref, second_ref)),
            BackendState,
        )

    def test_contract_rejects_unknown_relationship_endpoint_before_trainer(self) -> None:
        contract = make_contract()
        invalid = StrategyDefinition(
            participants=(ParticipantSpec("adaptation.main", frozenset({ParticipantUse.OPTIMIZE})),),
            relationships=(
                RelationshipSpec(
                    "missing-host",
                    "adaptation.main",
                    "model.denoiser",
                    RelationshipKind.ATTACHMENT,
                ),
            ),
        )

        with self.assertRaises(ContractViolation):
            contract.establish(invalid, "invalid-run")


if __name__ == "__main__":
    unittest.main(verbosity=2)
