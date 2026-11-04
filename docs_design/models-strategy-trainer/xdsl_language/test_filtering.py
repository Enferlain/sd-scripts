"""Torch-free cardinality, effect-driven rewrite and independent oracle checks."""

from __future__ import annotations

import random

from dataclasses import replace

import pytest

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

from . import filter_case, filter_oracle
from .contracts import DecisionSemantics, ItemSemantics, RankRule, RegionSemantics, Reproducibility
from .dialects import CallOp, ConstantOp, FusedLocalOp, OnceOp, OptionalPayloadType, ProgramOp, dump, parse
from .filter_checks import EXPECTED, check_conservation, check_products, check_settlement
from .filter_contracts import INITIAL, MANIFEST
from .fixtures import SCALAR, body, call, end, node, props, structured
from .local_rewrite import barrier, fuse
from .normalize import normalize, source_index
from .observations import compare, observe
from .prepare import Artifact, BindingSpec, CorrespondenceError, Policy, load, prepare
from .reference import CallResult, Reference, ReferenceError, State
from .verify import SemanticError, TargetLimit, verify


def reference(*, fused=False, opaque=False, reject_all=False, history=True):
    plan = prepare(filter_case.program(opaque=opaque), MANIFEST, Policy(workers=3, fuse_local=fused))
    ref = Reference(
        plan.verification.normalized, INITIAL, filter_oracle.bindings(reject_all=reject_all), schemas=filter_oracle.SCHEMAS, history=history
    )
    return ref


def advance(ref, predicate, bound=10000):
    for _ in range(bound):
        if predicate():
            return
        if not ref.step():
            break
    raise AssertionError("reference did not reach the requested external control boundary")


def check_reference(ref):
    assert ref.settled and not ref.errors
    check_products(ref.resources["products"].value)
    check_settlement(ref.streams["raw"].settled, ref.streams["kept"].settled)
    for stream in ref.streams.values():
        check_conservation(
            selected=stream.next_key,
            settled=stream.settled,
            active=len(stream.outstanding),
            ready=len(stream.ready),
            high_water=stream.high_water,
            horizon=3,
        )
    assert ref.handlers["more"].dispositions == {0: "completed"}
    assert all(not cap.active for cap in ref.capabilities.values())


def consumer_facts(items):
    return [item for item in items if any(name == "consumers" for name, _ in item.invocation)]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("opaque", [False, True])
def test_filter_cardinality_roundtrip_independent_values_and_settlement(fused, opaque):
    authored = filter_case.program(opaque=opaque)
    plan = prepare(parse(dump(authored)), MANIFEST, Policy(workers=3, fuse_local=fused))
    cold = load(Artifact.from_json(plan.artifact.to_json()), MANIFEST, plan.policy)
    ref = Reference(cold.verification.normalized, INITIAL, filter_oracle.bindings(), schemas=filter_oracle.SCHEMAS)
    ref.run()
    check_reference(ref)
    filt = source_index(cold.verification.normalized.module)["filter"]
    assert isinstance(filt.results[0].type, OptionalPayloadType)
    items = filt.get_trait(ItemSemantics)
    assert items is not None and items.describe(filt).forwards == (0,)


def test_reference_fusion_preserves_every_consumer_stage_observation():
    plain, grouped = reference(), reference(fused=True)
    plain.run()
    grouped.run()
    compare(observe(consumer_facts(plain.facts)), observe(consumer_facts(grouped.facts)))
    check_reference(grouped)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("fused", [False, True])
def test_bounded_reference_interleavings(seed, fused):
    ref, scheduler = reference(fused=fused), random.Random(seed)
    for _ in range(12000):
        enabled = ref.enabled()
        if not enabled:
            break
        ref.step(scheduler.choice(enabled))
    check_reference(ref)


@pytest.mark.parametrize("fused", [False, True])
def test_reference_pending_request_survives_dropped_input(fused):
    ref = reference(fused=fused)
    ref.held.add(("produce", 2))
    advance(
        ref,
        lambda: bool(ref.handlers["more"].active or ref.handlers["more"].pending) and ref.streams["raw"].terminal.get(1) == "filtered_out",
    )
    ref.run()
    assert not ref.settled
    assert ref.resources["products"].value == ((0, 2),)
    assert not any(value.owner == "consumers" and value.state == State.TERMINAL for value in ref.outcomes.values())
    assert not ref.stopped
    ref.held.clear()
    ref.run()
    check_reference(ref)


@pytest.mark.parametrize("fused", [False, True])
def test_reference_orderly_requested_stop_finishes_the_admitted_iteration(fused):
    ref = reference(fused=fused)
    ref.held.add(("next.offset", 0))
    advance(ref, lambda: ref.call_counts.get("next.offset") == 1)
    ref.request_stop("requested")
    ref.held.clear()
    ref.run()
    assert ref.settled and not ref.errors
    check_products(ref.resources["products"].value, EXPECTED[:2])
    check_settlement(ref.streams["raw"].settled, ref.streams["kept"].settled, EXPECTED[:2])
    assert ref.handlers["more"].dispositions == {0: "completed"}
    assert ref.call_counts["next.double"] == 1


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_reference_cancelled_filter_retains_claim_until_entered_work_finishes(fused, failure):
    ref = reference(fused=fused)
    ref.held.add(("predicate", 1))
    if failure:
        ref.failures["predicate", 1] = "physical filter failure"
    advance(ref, lambda: ref.call_counts.get("predicate", 0) >= 2)
    ref.cancelled = True
    ref.run()
    assert not ref.settled and 1 in ref.streams["raw"].outstanding
    ref.held.clear()
    ref.run()
    assert ref.settled and all(not stream.outstanding for stream in ref.streams.values())
    if failure:
        assert any("physical filter failure" in error for error in ref.errors)


@pytest.mark.parametrize("fused", [False, True])
def test_reference_completed_consumers_survive_late_physical_source_failure(fused):
    ref = reference(fused=fused)
    ref.held.update({("predicate", 5), ("produce", 6), ("next.publish", 1)})
    ref.failures["produce", 6] = "late physical source failure"
    advance(
        ref,
        lambda: (
            ref.call_counts.get("predicate", 0) >= 6
            and ref.call_counts.get("produce", 0) >= 7
            and ref.call_counts.get("next.publish", 0) == 2
        ),
    )
    ref.held.remove(("next.publish", 1))
    ref.run()
    assert not ref.settled
    assert any(
        value.owner == "consumers" and value.state == State.TERMINAL and value.status == "succeeded" for value in ref.outcomes.values()
    )
    check_products(ref.resources["products"].value)
    ref.held.remove(("predicate", 5))
    ref.run()
    assert not ref.settled
    ref.held.clear()
    ref.run()
    assert ref.settled and any("late physical source failure" in item for item in ref.errors)
    assert all(not stream.outstanding for stream in ref.streams.values())
    check_products(ref.resources["products"].value)
    check_settlement(ref.streams["raw"].settled, ref.streams["kept"].settled)


@pytest.mark.parametrize("fused", [False, True])
def test_reference_cancellation_cannot_enter_the_next_local_stage(fused):
    ref = reference(fused=fused)
    ref.held.add(("next.extract", 0))
    advance(ref, lambda: ref.call_counts.get("next.extract", 0) == 1)
    ref.cancelled = True
    ref.held.clear()
    ref.run()
    assert ref.settled and not ref.call_counts.get("next.offset") and not ref.call_counts.get("next.double")
    check_products(ref.resources["products"].value, EXPECTED[:1])


def test_all_drops_leave_demand_pending_and_cancellation_drains():
    ref = reference(reject_all=True, history=False)
    advance(ref, lambda: "raw" in ref.streams and len(ref.streams["raw"].settled) >= 128)
    assert ref.resources["products"].value == () and not ref.settled
    assert all(item.disposition == "filtered_out" for item in ref.streams["raw"].settled.values())
    ref.cancelled = True
    ref.run()
    assert ref.settled and all(not stream.outstanding for stream in ref.streams.values())
    assert not ref.facts


@pytest.mark.parametrize("fused", [False, True])
def test_invalid_intermediate_output_prevents_later_stage_entry(fused):
    ref = reference(fused=fused)
    old = ref.bindings["filter_offset"]
    ref.bindings["filter_offset"] = replace(old, invoke=lambda args: CallResult(("wrong",)))
    ref.run()
    assert ref.settled and any("output contract" in item for item in ref.errors)
    assert not any(name.endswith(".double") for name in ref.call_counts)
    assert all(not stream.outstanding for stream in ref.streams.values())


def test_valid_opaque_stage_is_an_executable_legality_barrier_and_forged_group_rejects():
    plan = prepare(filter_case.program(opaque=True), MANIFEST, Policy(workers=3, fuse_local=True))
    ops = source_index(plan.verification.normalized.module)
    groups = [op for op in ops.values() if isinstance(op, FusedLocalOp)]
    assert len(groups) == 2
    assert all(
        tuple(item.data for item in group.members) in (("first.extract", "first.offset"), ("next.extract", "next.offset"))
        for group in groups
    )
    contracts = {spec.name: spec.contract for spec in MANIFEST.bindings}
    reason = barrier(ops["first.opaque"], contracts)
    assert reason is not None and "unknown effects" in reason
    dishonest = {spec.name: replace(spec, unknown_effects=False).contract for spec in MANIFEST.bindings}
    forged = fuse(normalize(filter_case.program(opaque=True)), dishonest)
    with pytest.raises(VerifyException, match=r"cannot fuse first.opaque.*unknown effects"):
        load(replace(plan.artifact, normalized=forged.text), MANIFEST, plan.policy)
    ref = reference(fused=True, opaque=True)
    ref.run()
    check_reference(ref)


def test_unknown_legacy_effects_never_become_purity_and_resource_grants_remain_barriers():
    plan = prepare(filter_case.program(), MANIFEST, Policy())
    ops = source_index(plan.verification.normalized.module)
    contracts = {spec.name: spec.contract for spec in MANIFEST.bindings}
    legacy = BindingSpec("filter_offset", ("scalar",), ("scalar",))
    contracts[legacy.name] = legacy.contract
    assert barrier(ops["first.offset"], contracts) is not None
    assert barrier(ops["produce"], contracts) is not None
    assert barrier(ops["first.publish"], contracts) is not None
    assert barrier(ops["first.write"], contracts) is not None


def test_forwarded_versions_must_match_the_actual_upstream_owners():
    authored = filter_case.program()
    source_index(authored)["kept"].properties["owners"] = ArrayAttr([])
    with pytest.raises(SemanticError, match="production_versions"):
        verify(normalize(authored))


def test_filter_predicate_must_be_boolean_and_cannot_return_an_ordinary_payload():
    authored = filter_case.program()
    predicate = source_index(authored)["predicate"]
    predicate.results[0]._type = source_index(authored)["first.offset"].results[0].type  # Deliberate malformed IR.
    with pytest.raises(VerifyException, match="entry/yield signature"):
        normalize(authored)


def test_physical_binding_cannot_fabricate_a_cardinality_provenance_envelope():
    authored = filter_case.program()
    source_index(authored)["first.offset"].results[0]._type = OptionalPayloadType(filter_case.RECORD)  # Deliberately forged result type.
    with pytest.raises((SemanticError, VerifyException)):
        verify(normalize(authored))


def test_fusion_preserves_external_uses_and_never_crosses_a_region_boundary():
    authored = filter_case.program()
    ops = source_index(authored)
    first = ops["first.extract"]
    extra = first.clone()
    extra.properties["origin"] = StringAttr("independent.use")
    extra.properties["binding"] = ops["first.offset"].properties["binding"]
    extra.operands = (first.results[0],)
    block = first.parent_block()
    assert block is not None and block.last_op is not None
    block.insert_op_before(extra, block.last_op)
    plan = prepare(authored, MANIFEST, Policy(fuse_local=True))
    groups = [op for op in plan.verification.normalized.module.walk() if isinstance(op, FusedLocalOp)]
    assert not any("first.extract" in tuple(item.data for item in group.members) for group in groups)
    for group in groups:
        parent = group.parent_op()
        assert parent is not None and parent.name == "lang.once"


def test_invalid_fused_wiring_is_rejected_independently_of_saved_summaries():
    plan = prepare(filter_case.program(), MANIFEST, Policy(fuse_local=True))
    cold = parse(plan.artifact.normalized)
    group = next(op for op in cold.walk() if isinstance(op, FusedLocalOp))
    stages = [op for op in group.body.block.ops if isinstance(op, CallOp)]
    stages[-1].operands = (stages[0].results[0],)
    with pytest.raises(VerifyException, match=r"additional uses|SSA chain"):
        load(replace(plan.artifact, normalized=dump(cold)), MANIFEST, plan.policy)


def test_cold_group_source_and_wiring_corruption_rejects():
    plan = prepare(filter_case.program(), MANIFEST, Policy(fuse_local=True))
    module = parse(plan.artifact.normalized)
    group = next(op for op in module.walk() if isinstance(op, FusedLocalOp))
    group.properties["members"] = ArrayAttr([StringAttr("forged"), StringAttr("source")])
    with pytest.raises(VerifyException, match="member/source"):
        load(replace(plan.artifact, normalized=dump(module)), MANIFEST, plan.policy)
    with pytest.raises(CorrespondenceError, match="derived"):
        prepare(parse(plan.artifact.normalized), MANIFEST, plan.policy)


def test_independent_oracle_rejects_wrong_values_and_forwarding_receipts():
    ref = reference()
    ref.run()
    with pytest.raises(ReferenceError, match="independent expectation"):
        check_products(((0, 0), (2, 0), (4, 0)))
    altered = dict(ref.streams["kept"].settled)
    altered[2] = replace(altered[2], provenance=(("raw", 3),))
    with pytest.raises(ReferenceError, match="provenance"):
        check_settlement(ref.streams["raw"].settled, altered)
    assert ref.resources["products"].value == EXPECTED


def deterministic_math():
    value = node(ConstantOp, "input", outputs=[SCALAR], literal=StringAttr("1"))
    local = body(value.results)
    offset = call(local, "offset", "filter_offset", local.args, [SCALAR])
    doubled = call(local, "double", "filter_double", offset.results, [SCALAR])
    end(local, "local", doubled.results[0])
    work = structured(OnceOp, "local", value.results, [local], outputs=[SCALAR])
    root = ProgramOp.create(properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([value, work]))])
    module = ModuleOp([root])
    for op in module.walk():
        if op.get_trait(RegionSemantics) is not None or op.get_trait(DecisionSemantics) is not None:
            op.properties["reproducibility"] = StringAttr(Reproducibility.INDEPENDENT)
            op.properties["rank_rule"] = StringAttr(RankRule.REPLICATED)
    return module


def test_transparent_fusion_preserves_reproducibility_contributors_and_rank_claims():
    module = normalize(deterministic_math())
    before = verify(module, replicated=True)
    after = verify(fuse(module, {spec.name: spec.contract for spec in MANIFEST.bindings}), replicated=True)
    assert after.reproducibility == before.reproducibility == Reproducibility.INDEPENDENT
    assert after.contributors == before.contributors


def test_transparent_fusion_does_not_hide_an_unknown_original_stage_decision():
    module = deterministic_math()
    stage = source_index(module)["offset"]
    stage.properties["reproducibility"] = StringAttr(Reproducibility.UNKNOWN)
    stage.properties["rank_rule"] = StringAttr(RankRule.UNKNOWN)
    after = fuse(normalize(module), {spec.name: spec.contract for spec in MANIFEST.bindings})
    assert verify(after).reproducibility == Reproducibility.UNKNOWN
    with pytest.raises(TargetLimit) as failed:
        verify(after, replicated=True)
    assert failed.value.source == "offset"


@pytest.mark.parametrize("fused", [False, True])
def test_reference_sibling_failure_discards_the_interrupted_filter_input(fused):
    ref = reference(fused=fused)
    ref.bindings["filter_offset"] = replace(ref.bindings["filter_offset"], invoke=lambda args: CallResult(("invalid",)))
    ref.held.update({("predicate", 1), ("first.offset", 0)})
    advance(ref, lambda: ref.call_counts.get("predicate", 0) >= 2 and ref.call_counts.get("first.offset", 0) == 1)
    ref.held.remove(("first.offset", 0))
    ref.run()
    assert not ref.settled and not ref.cancelled and 1 in ref.streams["raw"].outstanding
    ref.held.clear()
    ref.run()
    assert ref.settled and ref.streams["raw"].terminal[1] == "discarded"
    check_settlement(ref.streams["raw"].settled, ref.streams["kept"].settled, expected=None)


def test_reference_duplicate_or_impostor_settlement_cannot_replace_the_owned_receipt():
    ref = reference()
    ref.held.add(("predicate", 1))
    advance(ref, lambda: ref.call_counts.get("predicate", 0) >= 2)
    stream = ref.streams["raw"]
    ticket = stream.outstanding[1]
    with pytest.raises(ReferenceError, match="current reservation"):
        ref._dispose(replace(ticket), "forged", ())
    assert stream.outstanding[1] is ticket and 1 not in stream.settled
    ref.cancelled = True
    ref.held.clear()
    ref.run()
    original = stream.settled[1]
    with pytest.raises(ReferenceError, match="duplicate item disposition"):
        ref._dispose(ticket, "forged", ())
    assert stream.settled[1] == original
