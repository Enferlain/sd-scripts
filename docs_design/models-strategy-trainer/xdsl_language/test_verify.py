"""Semantic rejection, independent lifecycle witnesses and version-policy checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Operation, Region
from xdsl.irdl import irdl_op_definition, traits_def
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.traits import IsolatedFromAbove, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException

from .contracts import (
    AccessMode,
    Admission,
    ItemSemantics,
    DecisionSemantics,
    FlowSemantics,
    Freshness,
    RankRule,
    RealizationSemantics,
    RegionSemantics,
    Reproducibility,
    Protocol,
    Version,
    VersionPolicy,
    origin,
)
from .dialects import (
    AccessOp,
    BorrowOp,
    BranchingOp,
    CapabilityType,
    ChoiceOp,
    ConstantOp,
    CaptureShape,
    bool_type,
    index_type,
    integer,
    dump,
    OnceOp,
    ParallelOp,
    ProducerOp,
    ProgramOp,
    RequestOp,
    RequestType,
    RequestsOp,
    ResourceType,
    RetireOp,
    SequenceOp,
    SourceShape,
    StopAfterOp,
    StreamOp,
    StructuredOp,
    YieldOp,
)
from .fixtures import SCALAR, RECORD, access, body, call, data_only, end, lifetime, node, props, resource, retire, structured, repeat, take
from .normalize import normalize, source_index
from .observations import compare, observe
from .reference import CallResult, Capability, Reference, ReferenceError, StandIn, State, Update, checked
from .standins import bindings, schemas
from .verify import Comparison, Cut, REFERENCE_TARGET, SemanticError, TargetLimit, evaluate_admission, verify


def get[T: Operation](module: Operation, kind: type[T], name: str | None = None) -> T:
    return next(op for op in module.walk() if isinstance(op, kind) and (name is None or origin(op) == name))


def known_decisions(module: ModuleOp) -> None:
    for op in module.walk():
        if any(op.get_trait(kind) is not None for kind in (RegionSemantics, FlowSemantics, DecisionSemantics)):
            op.properties.update(
                reproducibility=StringAttr(Reproducibility.INDEPENDENT),
                rank_rule=StringAttr(RankRule.REPLICATED),
                assumptions=ArrayAttr([]),
            )


def reference(author: ModuleOp, *, model: bool = True) -> Reference:
    return Reference(normalize(author), {"encoder": 2.0, "model": 0.8} if model else {"policy": 1.0}, bindings(), schemas=schemas())


def advance_until(ref: Reference, predicate: Callable[[], bool], limit: int = 4000) -> None:
    for _ in range(limit):
        if predicate():
            return
        assert ref.step(), "oracle blocked before the expected witness"
    pytest.fail("oracle did not reach witness within the exploration bound")


@pytest.mark.parametrize("build", [lifetime, data_only, lambda: lifetime(extra_scope=True)])
def test_all_compositions_share_verification_and_reference_closure(build: Callable[[], ModuleOp]) -> None:
    author = build()
    report = verify(normalize(author), target=REFERENCE_TARGET)
    assert report.inputs and report.obligations and report.waits
    ref = reference(author, model=build is not data_only)
    ref.run()
    assert ref.settled and not ref.errors
    assert all(not stream.outstanding for stream in ref.streams.values())
    assert all(not h.pending and not h.active for h in ref.handlers.values())
    assert all(not c.active for c in ref.capabilities.values())
    assert report.reproducibility == Reproducibility.UNKNOWN


def test_static_diagnostic_origins_do_not_select_rules_or_alias_resource_symbols() -> None:
    author = lifetime()
    for op in source_index(author).values():
        op.properties["origin"] = StringAttr("source." + origin(op))
    report = verify(normalize(author))
    assert report.inputs[0].producer == "source.producer"
    assert report.inputs[0].required_initial_owners == ("encoder",)


def test_training_only_stop_rejects_with_an_independent_late_input_counterexample() -> None:
    author = lifetime()
    get(author, StopAfterOp).properties["source"] = SymbolRefAttr("training")
    with pytest.raises(SemanticError, match=r"stop.*stop_coverage.*evaluation.take"):
        verify(normalize(author))
    ref = reference(author)
    ref.held.add(("encode", 4))
    advance_until(ref, lambda: bool(ref.stopped))
    ref.held.clear()
    ref.run()
    assert ref.settled and ref.errors
    assert any("exhausted" in error for error in ref.errors)
    assert not any(f.owner == "measure" and f.kind == "physical_completed" for f in ref.facts)


@dataclass(frozen=True)
class DriversOnly(RegionSemantics):
    def describe(self, op: Operation):
        return replace(super().describe(op), completion="drivers_only")


@irdl_op_definition
class IncompleteJoin(BranchingOp):
    name = "verify_test.incomplete_join"
    traits = traits_def(SourceShape(), CaptureShape(), DriversOnly(Protocol.PARALLEL), SymbolOpInterface(), IsolatedFromAbove())


def test_omitted_pending_coverage_rejects_against_the_oracles_pending_debt() -> None:
    author = lifetime()
    old = get(author, ParallelOp, "consumers")
    regions = []
    for region in old.regions:
        replacement = Region()
        region.clone_into(replacement)
        regions.append(replacement)
    changed = IncompleteJoin.create(operands=old.operands, regions=regions, properties=dict(old.properties))
    PatternRewriter(old).replace(old, changed)
    with pytest.raises(SemanticError, match=r"consumers.*pending_coverage"):
        verify(normalize(author))
    ref = reference(author)
    ref.held.add(("encode", 4))
    advance_until(ref, lambda: any(h.pending for h in ref.handlers.values()))
    assert not ref.settled
    assert all(o.state != State.TERMINAL for o in ref.outcomes.values() if o.owner == "consumers")
    ref.cancelled = True
    ref.held.clear()
    ref.run()
    assert ref.settled


@pytest.mark.parametrize("kind", ["source_outside", "registration_boundary", "circular_stop", "circular_handler"])
def test_invalid_completion_relationships_have_source_attributed_errors(kind: str) -> None:
    author = lifetime()
    if kind == "source_outside":
        get(author, RequestOp).properties["source"] = SymbolRefAttr("producer")
        get(author, RequestsOp).properties["sources"] = ArrayAttr([SymbolRefAttr("producer")])
        pattern = "request.*request_source"
    elif kind == "registration_boundary":
        get(author, RequestOp).properties["boundary"] = StringAttr("logical_success")
        pattern = "request.*request_registration"
    elif kind == "circular_stop":
        get(author, StopAfterOp).properties["source"] = SymbolRefAttr("activities")
        pattern = "stop.*settlement_cycle"
    else:
        get(author, RequestsOp).properties["sources"] = ArrayAttr([SymbolRefAttr("consumers"), SymbolRefAttr("training")])
        pattern = "evaluation.*settlement_cycle"
    with pytest.raises(SemanticError, match=pattern):
        verify(normalize(author))
    if kind.startswith("circular"):
        with pytest.raises(ReferenceError, match="circular"):
            reference(author)
    else:
        ref = reference(author)
        ref.run()
        assert ref.errors


def test_an_unrequested_handler_creates_no_fictional_admission() -> None:
    author = lifetime(windows=1)
    report = verify(normalize(author))
    assert any(o.kind == "request_closure" for o in report.obligations)
    ref = reference(author)
    ref.run()
    assert ref.settled and not ref.errors
    assert not any(o.invocation_source == "accepted_request" for o in ref.outcomes.values())


@pytest.mark.parametrize("admitted", [False, True])
def test_verified_orderly_stop_preserves_only_admitted_iteration_promises(admitted: bool) -> None:
    author = lifetime()
    verify(normalize(author))
    ref = reference(author)

    def selected() -> bool:
        windows = [o for o in ref.outcomes.values() if o.owner == "training" and o.invocation_source == "repeat_policy"]
        return len(windows) == 2 and (not admitted or windows[-1].state == State.RUNNING)

    advance_until(ref, selected)
    ref.request_stop("training")
    ref.run()
    assert ref.settled
    requests = [f for f in ref.facts if f.kind == "request_pending"]
    assert len(requests) == int(admitted)


def test_completion_and_usability_remain_separate_from_static_acceptance() -> None:
    author = lifetime()
    report = verify(normalize(author))
    assert any("physical completion" in evidence for evidence in report.deferred)
    assert any("usability" in evidence for evidence in report.deferred)
    ref = reference(author)
    ref.held.add(("measure", 0))
    advance_until(ref, lambda: any(f.owner == "measure" and f.kind == "physical_entered" for f in ref.facts))
    assert not ref.settled and ref.resources["model"].readers
    ref.cancelled = True
    ref.run()
    assert not ref.settled and ref.resources["model"].readers
    ref.failures[("measure", 0)] = "late physical failure"
    ref.held.clear()
    ref.run()
    assert ref.settled and not ref.resources["model"].readers
    assert any("late physical failure" in e for e in ref.errors)


def test_value_and_physical_contracts_are_not_inferred_from_binding_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    describe = RealizationSemantics.describe
    monkeypatch.setattr(RealizationSemantics, "describe", lambda self, op: replace(describe(self, op), completion="host_return"))
    with pytest.raises(SemanticError, match="physical_boundary"):
        verify(normalize(data_only()))


@pytest.mark.parametrize("defect", ["duplicate", "missing", "borrow_after"])
def test_affine_claim_errors_have_independent_reference_witnesses(defect: str) -> None:
    author = data_only(count=1)
    old = get(author, RetireOp)
    block, claim = old.parent_block(), old.claim
    assert block is not None
    if defect == "missing":
        block.erase_op(old)
    elif defect == "duplicate":
        block.insert_op_before(node(RetireOp, "duplicate.retire", inputs=[claim], disposition=StringAttr("consumed")), old)
    else:
        block.insert_op_after(node(BorrowOp, "late.borrow", inputs=[claim], outputs=[RECORD]), old)
    with pytest.raises(SemanticError, match=r"affine_claim|claim_after_consume"):
        verify(normalize(author))
    ref = reference(author, model=False)
    ref.run()
    assert ref.settled
    if defect == "missing":
        assert any(f.kind == "disposition" and f.values[1] == "unconsumed" for f in ref.facts)
    else:
        assert ref.errors


@pytest.mark.parametrize("both", [True, False])
def test_choice_claim_ownership_depends_on_exclusive_paths_not_ssa_use_totals(both: bool) -> None:
    author = data_only(count=1)
    old = get(author, RetireOp)
    block, claim = old.parent_block(), old.claim
    assert block is not None
    predicate = node(ConstantOp, "predicate", outputs=[bool_type()], literal=StringAttr("true"))
    yes, no = body([claim]), body([claim])
    retire(yes, "yes.retire", yes.args[0])
    if both:
        retire(no, "no.retire", no.args[0])
    end(yes, "yes")
    end(no, "no")
    block.insert_op_before(predicate, old)
    PatternRewriter(old).replace(old, structured(ChoiceOp, "claim.choice", [predicate.value, claim], [yes, no]))
    if not both:
        with pytest.raises(SemanticError, match="affine_claim"):
            verify(normalize(author))
        return
    verify(normalize(author))
    ref = reference(author, model=False)
    ref.run()
    assert ref.settled and not ref.errors
    assert sum(f.kind == "disposition" and f.values[1] == "consumed" for f in ref.facts) == 1


def test_explicit_claim_transfer_through_a_synchronous_region_preserves_ownership() -> None:
    author = data_only(count=1)
    old = get(author, RetireOp)
    block, claim = old.parent_block(), old.claim
    assert block is not None
    nested = body([claim])
    end(nested, "transfer.body", nested.args[0])
    forwarded = structured(OnceOp, "transfer", [claim], [nested], outputs=[claim.type])
    block.insert_op_before(forwarded, old)
    old.operands = forwarded.results
    verify(normalize(author))
    ref = reference(author, model=False)
    ref.run()
    assert ref.settled and not ref.errors


@pytest.mark.parametrize("count", [0, 2])
def test_captured_claim_cannot_be_implicitly_reused_by_repetition(count: int) -> None:
    author = data_only(count=1)
    old = get(author, RetireOp)
    nested = body([old.claim], index_type())
    retire(nested, "nested.retire", nested.args[0])
    end(nested, "nested")
    PatternRewriter(old).replace(old, repeat("nested.loop", [old.claim], nested, count))
    with pytest.raises(SemanticError, match=r"nested.loop.*repeated_claim"):
        verify(normalize(author))


@pytest.mark.parametrize("count", [0, 2])
def test_explicit_affine_loop_carry_returns_the_claim_for_one_final_disposition(count: int) -> None:
    author = data_only(count=1)
    old = get(author, RetireOp)
    destination, claim = old.parent_block(), old.claim
    assert destination is not None
    nested = Block(arg_types=[index_type(), claim.type])
    end(nested, "carry.body", nested.args[1])
    loop = repeat("carry", [claim], nested, count, carries=1, outputs=[claim.type])
    destination.insert_op_before(loop, old)
    old.operands = loop.results
    verify(normalize(author))
    ref = reference(author, model=False)
    ref.run()
    assert ref.settled and not ref.errors
    assert sum(f.kind == "disposition" and f.values[1] == "consumed" for f in ref.facts) == 1


def test_parallel_capture_does_not_implicitly_fork_a_claim() -> None:
    author = data_only(count=1)
    old = get(author, RetireOp)
    branches = [body([old.claim]), body([old.claim])]
    for i, block in enumerate(branches):
        retire(block, f"parallel.retire.{i}", block.args[0])
        end(block, f"parallel.branch.{i}")
    PatternRewriter(old).replace(old, structured(ParallelOp, "bad.fork", [old.claim], branches))
    with pytest.raises(SemanticError, match=r"bad.fork.*repeated_claim"):
        verify(normalize(author))


def test_sequential_placement_cannot_make_stop_wait_for_its_own_producer() -> None:
    author = data_only(count=1)
    producer, stop = get(author, ProducerOp), get(author, StopAfterOp)
    block = stop.parent_block()
    assert block is not None
    block.detach_op(stop)
    destination = producer.parent_block()
    assert destination is not None and destination.last_op is not None
    destination.insert_op_before(stop, destination.last_op)
    with pytest.raises(SemanticError, match="settlement_cycle"):
        verify(normalize(author))
    ref = reference(author, model=False)
    ref.run()
    assert not ref.settled and not ref.enabled()
    ref.cancelled = True
    ref.run()
    assert ref.settled


def transferred_claim(*, wait_for_producer: bool) -> ModuleOp:
    author = data_only(count=1)
    old = source_index(author)["consumer"]
    destination = old.parent_block()
    assert destination is not None
    work = body(old.operands)
    claim, _ = take(work, "transferred.take", work.args[0])
    end(work, "transferred.body", claim)
    consumer = structured(OnceOp, "consumer", old.operands, [work], outputs=[claim.type])
    destination.insert_op_before(consumer, old)
    if wait_for_producer:
        waiting = body([], RequestType(index_type()))
        end(waiting, "waiting.body")
        wait = structured(
            RequestsOp,
            "wait.producer",
            [],
            [waiting],
            sources=ArrayAttr([SymbolRefAttr("producer")]),
            request_schema=index_type(),
            max_outstanding=integer(1),
            max_active=integer(1),
        )
        destination.insert_op_before(wait, old)
    disposition = node(RetireOp, "transferred.retire", inputs=consumer.results, disposition=StringAttr("consumed"))
    destination.insert_op_before(disposition, old)
    destination.erase_op(old)
    return author


def test_transferred_claim_drain_cycles_reject_and_cancellation_cleans_branch_owned_locals() -> None:
    author = transferred_claim(wait_for_producer=True)
    with pytest.raises(SemanticError, match="settlement_cycle"):
        verify(normalize(author))
    ref = reference(author, model=False)
    ref.run()
    assert not ref.settled and not ref.enabled() and not ref.errors
    assert any(t.owner for stream in ref.streams.values() for t in stream.outstanding.values())
    ref.cancelled = True
    ref.run()
    assert ref.settled and all(not stream.outstanding for stream in ref.streams.values())
    observed = observe(ref.facts)
    dispositions = [key for key in observed.values if key.kind == "disposition" and observed.values[key][1] == "cancelled"]
    parent = next(key for key in observed.values if key.kind == "outcome" and key.owner == "activities")
    assert dispositions and any((key, parent) in observed.required_before for key in dispositions)
    key = next(key for key in dispositions if (key, parent) in observed.required_before)
    with pytest.raises(ReferenceError, match="causal"):
        compare(observed, replace(observed, positions={**observed.positions, key: observed.positions[parent] + 1}))


@pytest.mark.parametrize("cancel", [False, True])
def test_transferred_claim_disposition_is_an_explicit_producer_drain_requirement(cancel: bool) -> None:
    author = transferred_claim(wait_for_producer=False)
    report = verify(normalize(author))
    assert any(o.kind == "producer_drain" and o.required == ("transferred.retire",) for o in report.obligations)
    ref = reference(author, model=False)
    if cancel:
        advance_until(ref, lambda: any(o.owner == "consumer" and o.state == State.TERMINAL for o in ref.outcomes.values()))
        ref.cancelled = True
    ref.run()
    assert ref.settled and not ref.errors
    assert all(not stream.outstanding for stream in ref.streams.values())


def test_unused_handler_template_does_not_add_a_fictional_stop_or_drain_obligation() -> None:
    author = lifetime()
    request = get(author, RequestOp)
    parent = request.parent_block()
    assert parent is not None
    parent.erase_op(request)
    get(author, StopAfterOp).properties["source"] = SymbolRefAttr("training")
    report = verify(normalize(author))
    assert report.inputs[0].consumers == ("training.take",)
    ref = reference(author)
    ref.run()
    assert ref.settled and not ref.errors
    assert not any(o.invocation_source == "accepted_request" for o in ref.outcomes.values())


def test_borrow_result_detachment_is_required_by_the_actual_item_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    describe = ItemSemantics.describe
    monkeypatch.setattr(ItemSemantics, "describe", lambda self, op: replace(describe(self, op), detached_outputs=()))
    with pytest.raises(SemanticError, match="result_lifetime"):
        verify(normalize(data_only()))


def test_root_cannot_transfer_an_owner_reference_without_an_outgoing_contract() -> None:
    author = counter_program()
    root = get(author, ProgramOp)
    owner = source_index(author)["state"]
    root.body.block.add_op(node(YieldOp, "escape.root", inputs=owner.results))
    with pytest.raises(SemanticError, match="root_transfer"):
        verify(normalize(author))
    ref = Reference(normalize(author), {"state": 0}, counter_bindings())
    ref.run()
    assert ref.settled and any("outgoing" in error for error in ref.errors)


def counter_program(*, parallel: bool = False, mode: AccessMode = AccessMode.PROTECTED_CURRENT) -> ModuleOp:
    owner = resource("state", "counter")
    branches = [body([owner.value]) for _ in range(2)] if parallel else [body([owner.value])]
    for index, binding in enumerate(("bump", "read")):
        destination = branches[index] if parallel else branches[0]
        chosen = AccessMode.OWNER_UPDATE if binding == "bump" else mode
        inner = Block(arg_types=[CapabilityType(owner.value.type, StringAttr(chosen))])
        result = call(inner, binding, binding, inner.args, [SCALAR])
        end(inner, binding + ".body", *result.results)
        use = access(binding + ".access", destination.args, inner, chosen, [SCALAR])
        if chosen == AccessMode.SNAPSHOT:
            use.properties.update(freshness=StringAttr(Freshness.CURRENT_AT_START), version_selector=StringAttr("current"))
        destination.add_op(use)
    for i, block in enumerate(branches):
        end(block, "branch." + str(i))
    work = structured(ParallelOp if parallel else SequenceOp, "work", [owner.value], branches)
    root = ProgramOp.create(properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([owner, work]))])
    module = ModuleOp([root])
    known_decisions(module)
    return module


def counter_bindings() -> dict[str, StandIn]:
    def read(args: tuple[object, ...]) -> CallResult:
        return CallResult((checked(args[0], Capability).value,))

    def bump(args: tuple[object, ...]) -> CallResult:
        value = checked(checked(args[0], Capability).value, int) + 1
        return CallResult((value,), (Update(0, value),))

    return {"read": StandIn(("capability",), ("scalar",), read), "bump": StandIn(("capability",), ("scalar",), bump, updates=(0,))}


def nested_access(outer_mode: AccessMode, inner_mode: AccessMode, *, same: bool = True, published: bool = False) -> ModuleOp:
    first, second = resource("first", "counter"), resource("second", "counter")
    work = Block(arg_types=[CapabilityType(first.value.type, StringAttr(outer_mode)), first.value.type])
    inner = Block(arg_types=[CapabilityType(first.value.type, StringAttr(inner_mode))])
    binding = "bump" if inner_mode == AccessMode.OWNER_UPDATE else "read"
    value = call(inner, "inner.call", binding, inner.args, [SCALAR])
    end(inner, "inner.body", *value.results)
    use = access("inner.access", [work.args[1]], inner, inner_mode, [SCALAR])
    if inner_mode == AccessMode.SNAPSHOT and not published:
        use.properties.update(freshness=StringAttr(Freshness.CURRENT_AT_START), version_selector=StringAttr("current"))
    work.add_op(use)
    end(work, "outer.body", *use.results)
    outer = access("outer.access", [first.value, first.value if same else second.value], work, outer_mode, [SCALAR])
    if outer_mode == AccessMode.SNAPSHOT and not published:
        outer.properties.update(freshness=StringAttr(Freshness.CURRENT_AT_START), version_selector=StringAttr("current"))
    root = ProgramOp.create(properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([first, second, outer]))])
    author = ModuleOp([root])
    known_decisions(author)
    return author


@pytest.mark.parametrize(
    "outer,inner,same,published,blocked",
    [
        (AccessMode.PROTECTED_CURRENT, AccessMode.OWNER_UPDATE, True, False, True),
        (AccessMode.OWNER_UPDATE, AccessMode.OWNER_UPDATE, True, False, True),
        (AccessMode.OWNER_UPDATE, AccessMode.PROTECTED_CURRENT, True, False, True),
        (AccessMode.OWNER_UPDATE, AccessMode.SNAPSHOT, True, False, True),
        (AccessMode.PROTECTED_CURRENT, AccessMode.PROTECTED_CURRENT, True, False, False),
        (AccessMode.SNAPSHOT, AccessMode.OWNER_UPDATE, True, False, False),
        (AccessMode.SNAPSHOT, AccessMode.OWNER_UPDATE, True, True, False),
        (AccessMode.OWNER_UPDATE, AccessMode.SNAPSHOT, True, True, False),
        (AccessMode.PROTECTED_CURRENT, AccessMode.OWNER_UPDATE, False, False, False),
        (AccessMode.OWNER_UPDATE, AccessMode.PROTECTED_CURRENT, False, False, False),
        (AccessMode.OWNER_UPDATE, AccessMode.OWNER_UPDATE, False, False, False),
    ],
)
def test_nested_access_compatibility_retains_live_protection_without_inventing_snapshot_locks(
    outer: AccessMode, inner: AccessMode, same: bool, published: bool, blocked: bool
) -> None:
    author = nested_access(outer, inner, same=same, published=published)
    if blocked:
        with pytest.raises(SemanticError, match=r"inner.access.*nested_access_cycle"):
            verify(normalize(author), target=REFERENCE_TARGET)
    else:
        verify(normalize(author), target=REFERENCE_TARGET)
    ref = Reference(normalize(author), {"first": 0, "second": 10}, counter_bindings())
    ref.run()
    assert not ref.errors
    if blocked:
        assert not ref.settled and not ref.enabled()
        assert ref.resources["first"].readers or ref.resources["first"].writer is not None
        ref.cancelled = True
        ref.run()
    assert ref.settled and all(not owner.readers and owner.writer is None for owner in ref.resources.values())


def owner_carry(*, count: int = 2, passthrough: bool = False, read_result: bool = False) -> ModuleOp:
    first, second = resource("first", "counter"), resource("second", "counter")
    reader, writer = body([first.value, second.value]), body([first.value, second.value])
    iteration = Block(arg_types=[first.value.type, index_type(), first.value.type])

    def reading(destination: Block, owner) -> None:
        inner = Block(arg_types=[CapabilityType(first.value.type, StringAttr(AccessMode.PROTECTED_CURRENT))])
        value = call(inner, "read", "read", inner.args, [SCALAR])
        end(inner, "read.body", *value.results)
        destination.add_op(access("read.access", [owner], inner, AccessMode.PROTECTED_CURRENT, [SCALAR]))

    if not read_result:
        reading(iteration, iteration.args[2])
    end(iteration, "loop.body", iteration.args[2] if passthrough else iteration.args[0])
    loop = repeat("loop", [reader.args[1], reader.args[0]], iteration, count, carries=1, outputs=[first.value.type])
    reader.add_op(loop)
    if read_result:
        reading(reader, loop.results[0])
    end(reader, "reader")
    change = Block(arg_types=[CapabilityType(second.value.type, StringAttr(AccessMode.OWNER_UPDATE))])
    updated = call(change, "bump", "bump", change.args, [SCALAR])
    end(change, "bump.body", *updated.results)
    writer.add_op(access("bump.access", [writer.args[1]], change, AccessMode.OWNER_UPDATE, [SCALAR]))
    end(writer, "writer")
    work = structured(ParallelOp, "work", [first.value, second.value], [reader, writer])
    root = ProgramOp.create(properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([first, second, work]))])
    author = ModuleOp([root])
    known_decisions(author)
    return author


def preferred_schedule(ref: Reference, preferred: str) -> list[tuple[str, int]]:
    for _ in range(700):
        enabled = ref.enabled()
        if not enabled:
            break
        options = [i for i in enabled if (command := ref.tasks[i].command) is not None and preferred in origin(command.op)]
        ref.step(options[0] if options else enabled[0])
    assert ref.settled and not ref.errors
    return [
        (checked(f.values[0], Version).owner, checked(f.values[0], Version).ordinal)
        for f in ref.facts
        if f.owner == "read.access" and f.kind == "protected_version"
    ]


def test_changing_owner_carry_retains_later_iteration_provenance_and_schedule_risk() -> None:
    author = owner_carry()
    report = verify(normalize(author))
    assert report.reproducibility == Reproducibility.MAY_DEPEND
    assert next(o for o in report.obligations if o.source == "read.access").required[:2] == ("first", "second")
    outcomes = [
        preferred_schedule(Reference(normalize(author), {"first": 0, "second": 10}, counter_bindings()), preferred)
        for preferred in ("read", "bump")
    ]
    assert outcomes == [[("first", 0), ("second", 0)], [("first", 0), ("second", 1)]]


@pytest.mark.parametrize("count,passthrough,expected", [(1, False, [("first", 0)]), (2, True, [("first", 0), ("first", 0)])])
def test_single_iteration_and_initialized_pass_through_carries_do_not_invent_owner_changes(
    count: int, passthrough: bool, expected: list[tuple[str, int]]
) -> None:
    author = owner_carry(count=count, passthrough=passthrough)
    report = verify(normalize(author))
    assert report.reproducibility == Reproducibility.INDEPENDENT
    assert next(o for o in report.obligations if o.source == "read.access").required[0] == "first"
    for preferred in ("read", "bump"):
        assert preferred_schedule(Reference(normalize(author), {"first": 0, "second": 10}, counter_bindings()), preferred) == expected


@pytest.mark.parametrize("count,stopped", [(0, False), (2, True), (2, False)])
def test_repeat_reference_results_include_zero_iteration_and_orderly_stop_initializers(count: int, stopped: bool) -> None:
    author = owner_carry(count=count, read_result=True)
    report = verify(normalize(author))
    evidence = next(o for o in report.obligations if o.source == "read.access")
    assert evidence.required[:1] == ("first",)
    if count:
        assert evidence.required[:2] == ("first", "second")
    else:
        assert report.reproducibility == Reproducibility.INDEPENDENT
    ref = Reference(normalize(author), {"first": 0, "second": 10}, counter_bindings())
    if stopped:
        ref.request_stop("loop")
    actual = preferred_schedule(ref, "bump")
    assert actual == ([("first", 0)] if count == 0 or stopped else [("second", 1)])


@pytest.mark.parametrize("nested_capture", [False, True])
def test_protected_capability_cannot_escape_by_yield_or_a_later_nested_capture(nested_capture: bool) -> None:
    author = counter_program()
    use = get(author, AccessOp, "read.access")
    cap = use.body.block.args[0]
    terminator = use.body.block.last_op
    assert terminator is not None
    terminator.operands = [cap]
    use.results[0]._type = cap.type
    destination = use.parent_block()
    assert destination is not None and destination.last_op is not None
    if nested_capture:
        nested = body(use.results)
        result = call(nested, "escaped.read", "read", nested.args, [SCALAR])
        end(nested, "escaped.body", *result.results)
        escape = structured(OnceOp, "escaped", use.results, [nested], outputs=[SCALAR])
        destination.insert_op_before(escape, destination.last_op)
    else:
        outside = node(
            type(source_index(author)["read"]),
            "escaped.read",
            inputs=use.results,
            outputs=[SCALAR],
            binding=SymbolRefAttr("read"),
            binding_version=StringAttr("1"),
        )
        destination.insert_op_before(outside, destination.last_op)
    with pytest.raises(SemanticError, match="capability_escape"):
        verify(normalize(author))
    ref = Reference(normalize(author), {"state": 0}, counter_bindings())
    ref.run()
    assert ref.settled and any("input mismatch" in e for e in ref.errors)


def test_bare_resource_operand_cannot_replace_an_access_realization() -> None:
    author = counter_program()
    work = get(author, SequenceOp)
    bare = node(
        type(source_index(author)["read"]),
        "bare.read",
        inputs=work.body.block.args,
        outputs=[SCALAR],
        binding=SymbolRefAttr("read"),
        binding_version=StringAttr("1"),
    )
    terminator = work.body.block.last_op
    assert terminator is not None
    work.body.block.insert_op_before(bare, terminator)
    with pytest.raises(SemanticError, match=r"bare.read.*unscoped_resource"):
        verify(normalize(author))


@dataclass(frozen=True)
class Reordered(RegionSemantics):
    def describe(self, op: Operation):
        original = super().describe(op)
        return replace(
            original,
            roles=tuple(replace(role, captures=tuple(replace(c, operand=1 - c.operand) for c in role.captures)) for role in original.roles),
        )


@irdl_op_definition
class ReorderedOnce(StructuredOp):
    name = "verify_test.reordered_once"
    traits = traits_def(SourceShape(), CaptureShape(), Reordered(Protocol.ONCE), SymbolOpInterface(), IsolatedFromAbove())


def test_registered_capture_mapping_governs_analysis_and_reference_entry() -> None:
    first, second = resource("first", "counter"), resource("second", "counter")
    work = body([first.value, second.value])
    inner = Block(arg_types=[CapabilityType(first.value.type, StringAttr(AccessMode.PROTECTED_CURRENT))])
    read = call(inner, "read", "read", inner.args, [SCALAR])
    end(inner, "read.body", *read.results)
    work.add_op(access("read.access", [work.args[0]], inner, AccessMode.PROTECTED_CURRENT, [SCALAR]))
    end(work, "work")
    operation = structured(ReorderedOnce, "work", [first.value, second.value], [work])
    root = ProgramOp.create(
        properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([first, second, operation]))]
    )
    author = ModuleOp([root])
    report = verify(normalize(author))
    assert any(o.source == "read.access" and o.required[0] == "second" for o in report.obligations)
    ref = Reference(normalize(author), {"first": 10, "second": 20}, counter_bindings())
    ref.run()
    assert ref.settled and not ref.errors
    assert any(f.owner == "read" and f.kind == "physical_completed" and f.values[1] == (20,) for f in ref.facts)


@pytest.mark.parametrize("escape", [False, True])
def test_nested_owner_scope_is_checked_separately_from_root_only_reference_target(escape: bool) -> None:
    local = resource("local", "counter")
    scope = Block([local])
    inner = Block(arg_types=[CapabilityType(local.value.type, StringAttr(AccessMode.PROTECTED_CURRENT))])
    result = call(inner, "local.read", "read", inner.args, [SCALAR])
    end(inner, "local.read.body", *result.results)
    local_read = access("local.access", [local.value], inner, AccessMode.PROTECTED_CURRENT, [SCALAR])
    scope.add_op(local_read)
    end(scope, "local.scope", *([local.value] if escape else []))
    owner = structured(OnceOp, "local.scope", [], [scope], outputs=[local.value.type] if escape else [])
    root = ProgramOp.create(properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([owner]))])
    author = ModuleOp([root])
    if escape:
        with pytest.raises(SemanticError, match="owner_lifetime"):
            verify(normalize(author))
    else:
        verify(normalize(author))
        with pytest.raises(TargetLimit, match=r"local.*root ownership"):
            verify(normalize(author), target=REFERENCE_TARGET)


@pytest.mark.parametrize("mode", [AccessMode.PROTECTED_CURRENT, AccessMode.SNAPSHOT])
def test_ordered_current_read_has_fixed_version_without_equating_mode_with_reproducibility(mode: AccessMode) -> None:
    author = counter_program(mode=mode)
    report = verify(normalize(author))
    assert report.reproducibility == Reproducibility.INDEPENDENT
    ref = Reference(normalize(author), {"state": 0}, counter_bindings())
    ref.run()
    assert ref.settled and not ref.errors
    assert [checked(f.values[0], Version).ordinal for f in ref.facts if f.owner == "read.access" and f.kind == "protected_version"] == [1]


@pytest.mark.parametrize("mode", [AccessMode.PROTECTED_CURRENT, AccessMode.SNAPSHOT])
def test_concurrent_current_selection_is_reported_as_a_schedule_risk(mode: AccessMode) -> None:
    author = counter_program(parallel=True, mode=mode)
    report = verify(normalize(author))
    assert report.reproducibility == Reproducibility.MAY_DEPEND
    assert any(c.source == "read.access" and "race" in c.reason for c in report.contributors)
    seen = set()
    for preferred in ("read", "bump"):
        ref = Reference(normalize(author), {"state": 0}, counter_bindings())
        for _ in range(500):
            enabled = ref.enabled()
            if not enabled:
                break
            options = [i for i in enabled if (command := ref.tasks[i].command) is not None and preferred in origin(command.op)]
            ref.step(options[0] if options else enabled[0])
        assert ref.settled and not ref.errors
        seen.update(checked(f.values[0], Version).ordinal for f in ref.facts if f.owner == "read.access" and f.kind == "protected_version")
    assert seen == {0, 1}


def test_missing_decision_declaration_is_unknown_not_reproducible() -> None:
    author = counter_program()
    get(author, AccessOp, "read.access").properties.pop("reproducibility")
    report = verify(normalize(author))
    assert report.reproducibility == Reproducibility.UNKNOWN
    assert any(c.source == "read.access" and c.classification == Reproducibility.UNKNOWN for c in report.contributors)


def test_quiescence_claims_use_declared_boundaries_and_do_not_publish_cut_readiness() -> None:
    author = counter_program()
    with pytest.raises(TargetLimit, match=r"quiescence.*no declared boundary"):
        verify(normalize(author), cuts=[Cut("work", "settled")])
    get(author, SequenceOp).properties["quiescence"] = ArrayAttr([StringAttr("settled")])
    report = verify(normalize(author), cuts=[Cut("work", "settled")])
    assert any(o.kind == "declared_cut" and "target evidence" in o.required[1] for o in report.obligations)
    get(author, SequenceOp).properties["quiescence"] = ArrayAttr([StringAttr("arbitrary_suspension")])
    with pytest.raises(SemanticError, match="quiescence_contract"):
        verify(normalize(author))


def test_rank_local_randomness_is_valid_when_decided_once_and_distributed() -> None:
    author = counter_program()
    read = source_index(author)["read"]
    read.properties.update(rank_rule=StringAttr(RankRule.LOCAL), reproducibility=StringAttr(Reproducibility.MAY_DEPEND))
    verify(normalize(author))
    with pytest.raises(TargetLimit, match=r"read.*replicated decisions"):
        verify(normalize(author), replicated=True)
    read.properties["rank_rule"] = StringAttr(RankRule.DISTRIBUTED)
    report = verify(normalize(author), replicated=True)
    assert any(o.source == "read" and o.kind == "rank_distribution" for o in report.obligations)


def test_valid_unavailable_access_is_distinct_from_invalid_access_semantics() -> None:
    author = counter_program()
    use = get(author, AccessOp, "read.access")
    use.properties.update(
        mode=StringAttr(AccessMode.GRANT), freshness=StringAttr(Freshness.OWNER_PROTOCOL), profile=StringAttr("delegated_read")
    )
    use.body.block.args[0]._type = CapabilityType(checked(use.operands[0].type, ResourceType), StringAttr(AccessMode.GRANT))
    verify(normalize(author))
    with pytest.raises(TargetLimit, match=r"read.access.*reference.*grant"):
        verify(normalize(author), target=REFERENCE_TARGET)
    ref = Reference(normalize(author), {"state": 0}, counter_bindings())
    ref.run()
    assert any("grant profile" in e for e in ref.errors)
    use.properties["freshness"] = StringAttr(Freshness.PUBLISHED)
    with pytest.raises(VerifyException, match="freshness"):
        verify(normalize(author))


def test_production_capture_substitution_is_checked_against_declared_version_owners() -> None:
    author = data_only()
    other = resource("other", "policy")
    root = get(author, ProgramOp)
    root.body.block.insert_op_before(other, get(author, StreamOp))
    activities = get(author, ParallelOp)
    activities.operands = [activities.operands[0], other.value]
    with pytest.raises(SemanticError, match=r"records.*production_versions"):
        verify(normalize(author))
    ref = Reference(normalize(author), {"policy": 1.0, "other": 1.0}, bindings(), schemas=schemas())
    advance_until(ref, lambda: any(f.kind == "item_rejected" for f in ref.facts))
    assert next(f for f in ref.facts if f.kind == "item_rejected").values[3] == "missing_or_extra_provenance"
    ref.cancelled = True
    ref.run()
    assert ref.settled


def version_flow(admission: Admission, *, lag: int = 0, current_snapshot: bool = False) -> ModuleOp:
    """A fixed snapshot producer, a bounded consumer and an independent writer."""
    author = data_only(count=1)
    get(author, StreamOp).properties.update(admission=StringAttr(admission), max_lag=integer(lag))
    selection = get(author, AccessOp, "select.access")
    selection.properties.update(
        mode=StringAttr(AccessMode.SNAPSHOT),
        freshness=StringAttr(Freshness.CURRENT_AT_START if current_snapshot else Freshness.PUBLISHED),
        version_selector=StringAttr("current" if current_snapshot else "initial"),
    )
    selection.body.block.args[0]._type = CapabilityType(checked(selection.operands[0].type, ResourceType), StringAttr(AccessMode.SNAPSHOT))
    feedback = get(author, AccessOp, "feedback.access")
    parent = feedback.parent_block()
    assert parent is not None
    parent.erase_op(feedback)
    activities = get(author, ParallelOp)
    updater = body(activities.operands)
    change = Block(arg_types=[CapabilityType(checked(updater.args[1].type, ResourceType), StringAttr(AccessMode.OWNER_UPDATE))])
    updated = call(change, "bump", "bump", change.args, [SCALAR])
    end(change, "bump.body", *updated.results)
    updater.add_op(access("bump.access", [updater.args[1]], change, AccessMode.OWNER_UPDATE, [SCALAR]))
    end(updater, "updater")
    activities.add_region(Region(updater))
    known_decisions(author)
    return author


def finish_update_first(ref: Reference) -> None:
    for _ in range(100):
        if ref.resources["policy"].ordinal == 1:
            return
        options = [i for i in ref.enabled() if (command := ref.tasks[i].command) is not None and origin(command.op).startswith("bump")]
        assert ref.step(options[0] if options else None)
    pytest.fail("independent update did not complete")


@pytest.mark.parametrize(
    "admission,lag,accepted",
    [
        (Admission.EXACT_CURRENT, 0, False),
        (Admission.BOUNDED_LAG, 0, False),
        (Admission.BOUNDED_LAG, 1, True),
        (Admission.ANY_WITH_PROVENANCE, 0, True),
    ],
)
def test_reference_admission_retains_actual_snapshot_version_and_rejection_disposition(
    admission: Admission, lag: int, accepted: bool
) -> None:
    author = version_flow(admission, lag=lag)
    report = verify(normalize(author))
    if admission != Admission.ANY_WITH_PROVENANCE:
        assert report.reproducibility == Reproducibility.MAY_DEPEND
        assert any(c.source == "records" and "admission" in c.reason for c in report.contributors)
    ref = Reference(normalize(author), {"policy": 1}, {**bindings(), **counter_bindings()}, schemas=schemas())
    finish_update_first(ref)
    advance_until(ref, lambda: any(f.kind == "admission_checked" for f in ref.facts))
    fact = next(f for f in ref.facts if f.kind == "admission_checked")
    actual = checked(fact.values[1], tuple)
    compared = checked(fact.values[3], tuple)
    assert actual[0].ordinal == 0 and compared[0].ordinal == 1
    flow = get(author, StreamOp)
    policy = FlowSemantics().describe(flow).versions
    result = evaluate_admission(policy, actual, Comparison(str(fact.values[5]), compared))
    assert result.accepted is accepted
    assert result.disposition == fact.values[4]
    if not accepted:
        assert any(f.kind == "disposition" and f.values[1] == "rejected_stale" for f in ref.facts)
        assert not any(f.kind == "item_admitted" for f in ref.facts)
        # Substituting current/requested provenance would conceal this defect.
        assert evaluate_admission(policy, compared, Comparison("item_admission", compared)).accepted
        ref.cancelled = True
    ref.run()
    assert ref.settled


def test_work_selected_before_an_update_can_observe_the_new_version_during_production() -> None:
    author = version_flow(Admission.EXACT_CURRENT, current_snapshot=True)
    verify(normalize(author))
    ref = Reference(normalize(author), {"policy": 1}, {**bindings(), **counter_bindings()}, schemas=schemas())
    advance_until(ref, lambda: any(f.kind == "selected" for f in ref.facts))
    assert not any(f.owner == "select.access" and f.kind == "protected_version" for f in ref.facts)
    finish_update_first(ref)
    ref.run()
    assert ref.settled and not ref.errors
    fact = next(f for f in ref.facts if f.kind == "item_admitted")
    assert checked(fact.values[1], tuple)[0].ordinal == 1


def test_multi_owner_production_is_checked_from_actual_captures_and_oracle_versions() -> None:
    author = lifetime()
    producer = get(author, ProducerOp)
    branch = producer.parent_block()
    assert branch is not None
    producer.operands = [*producer.operands, branch.args[2]]
    production = producer.body.block
    model = production.insert_arg(branch.args[2].type, 1)
    peek = Block(arg_types=[CapabilityType(checked(model.type, ResourceType), StringAttr(AccessMode.PROTECTED_CURRENT))])
    read = call(peek, "peek", "read", peek.args, [SCALAR])
    end(peek, "peek.body", *read.results)
    first = production.first_op
    assert first is not None
    production.insert_op_before(access("peek.access", [model], peek, AccessMode.PROTECTED_CURRENT, [SCALAR]), first)
    get(author, StreamOp).properties["owners"] = ArrayAttr([SymbolRefAttr("encoder"), SymbolRefAttr("model")])
    report = verify(normalize(author))
    assert report.inputs[0].required_initial_owners == ("encoder", "model")
    ref = Reference(
        normalize(author), {"encoder": 2.0, "model": 0.8}, {**bindings(), "read": counter_bindings()["read"]}, schemas=schemas()
    )
    ref.run()
    assert ref.settled and not ref.errors
    policy = FlowSemantics().describe(get(author, StreamOp)).versions
    for fact in (f for f in ref.facts if f.kind == "admission_checked"):
        actual, current = checked(fact.values[1], tuple), checked(fact.values[3], tuple)
        assert {v.owner for v in actual} == {"encoder", "model"}
        assert evaluate_admission(policy, actual, Comparison("item_admission", current)).disposition == fact.values[4]


def test_published_snapshot_remains_available_during_an_incompatible_live_update() -> None:
    author = counter_program(parallel=True, mode=AccessMode.SNAPSHOT)
    use = get(author, AccessOp, "read.access")
    use.properties.update(freshness=StringAttr(Freshness.PUBLISHED), version_selector=StringAttr("initial"))
    assert verify(normalize(author)).reproducibility == Reproducibility.INDEPENDENT
    ref = Reference(normalize(author), {"state": 0}, counter_bindings())
    ref.held.add(("bump", 0))
    advance_until(ref, lambda: any(f.owner == "bump" and f.kind == "physical_entered" for f in ref.facts))
    ref.run()
    assert not ref.settled
    assert any(f.owner == "read" and f.kind == "physical_completed" and f.values[1] == (0,) for f in ref.facts)
    assert ref.resources["state"].writer is not None and not ref.resources["state"].readers
    ref.held.clear()
    ref.run()
    assert ref.settled and ref.resources["state"].value == 1


@pytest.mark.parametrize("history", [True, False])
def test_version_rejection_progress_and_dispositions_do_not_consult_diagnostics(history: bool) -> None:
    author = version_flow(Admission.EXACT_CURRENT)
    verify(normalize(author))
    ref = Reference(normalize(author), {"policy": 1}, {**bindings(), **counter_bindings()}, schemas=schemas(), history=history)
    finish_update_first(ref)
    advance_until(ref, lambda: "rejected_stale" in ref.streams["records"].terminal.values())
    assert all(not items for items in ref.input_uses.values())
    assert ref.streams["records"].high_water <= 3
    ref.cancelled = True
    ref.run()
    assert ref.settled and not ref.streams["records"].outstanding
    assert bool(ref.facts) is history


def policy(admission: Admission, *, owners: tuple[str, ...] = ("a",), lag: int = 0) -> VersionPolicy:
    return VersionPolicy(admission, owners, "owner_update_ordinal_within_lineage", "item_admission", lag)


@pytest.mark.parametrize(
    "admission,old,now,lag,accepted",
    [
        (Admission.EXACT_CURRENT, 2, 2, 0, True),
        (Admission.EXACT_CURRENT, 1, 2, 0, False),
        (Admission.BOUNDED_LAG, 1, 3, 2, True),
        (Admission.BOUNDED_LAG, 0, 3, 2, False),
        (Admission.ANY_WITH_PROVENANCE, 0, 50, 0, True),
    ],
)
def test_admission_uses_actual_observed_version_and_independent_expected_values(
    admission: Admission, old: int, now: int, lag: int, accepted: bool
) -> None:
    production = Version("a", 1, "line", old)
    compared = Comparison("item_admission", (Version("a", 1, "line", now),))
    result = evaluate_admission(policy(admission, lag=lag), [production], compared)
    assert result.accepted is accepted
    assert result.observed == (production,) and result.comparison == compared


@pytest.mark.parametrize("defect", ["owner", "incarnation", "lineage", "future", "missing", "ambiguous", "boundary"])
def test_incomparable_multi_owner_maps_are_not_compared_via_a_global_step(defect: str) -> None:
    observed = (Version("a", 3, "a_line", 7), Version("b", 8, "b_line", 91))
    current = list(observed)
    boundary = "item_admission"
    if defect == "owner":
        current[1] = replace(current[1], owner="c")
    elif defect == "incarnation":
        current[1] = replace(current[1], incarnation=9)
    elif defect == "lineage":
        current[1] = replace(current[1], lineage="restored")
    elif defect == "future":
        current[1] = replace(current[1], ordinal=90)
    elif defect == "missing":
        current.pop()
    elif defect == "ambiguous":
        current.append(replace(current[1], ordinal=92))
    else:
        boundary = "execution_entry"
    result = evaluate_admission(policy(Admission.EXACT_CURRENT, owners=("a", "b")), observed, Comparison(boundary, tuple(current)))
    assert not result.accepted
    assert result.disposition in ("incomparable_provenance", "wrong_comparison_boundary")


def test_multi_owner_lag_is_per_lineage_and_admission_is_not_entry_freshness() -> None:
    observed = (Version("a", 1, "x", 2), Version("b", 3, "y", 100))
    current = (Version("a", 1, "x", 3), Version("b", 3, "y", 102))
    result = evaluate_admission(policy(Admission.BOUNDED_LAG, owners=("a", "b"), lag=2), observed, Comparison("item_admission", current))
    assert result.accepted and result.lags == (("a", 1), ("b", 2))
    current = tuple(replace(v, ordinal=v.ordinal + 1) for v in observed)
    admitted = evaluate_admission(policy(Admission.EXACT_CURRENT, owners=("a", "b")), observed, Comparison("item_admission", observed))
    assert admitted.accepted
    assert not evaluate_admission(
        policy(Admission.EXACT_CURRENT, owners=("a", "b")), observed, Comparison("execution_entry", current)
    ).accepted


def test_static_verification_does_not_read_runtime_bindings_or_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("verification invoked execution/observations")

    monkeypatch.setattr(Reference, "__init__", forbidden)
    monkeypatch.setattr(Reference, "run", forbidden)
    verify(normalize(lifetime()))


def test_verification_is_detached_and_recomputed_after_ir_changes() -> None:
    author = lifetime()
    report = verify(normalize(author))
    get(author, StopAfterOp).properties["source"] = SymbolRefAttr("training")
    assert report.normalized.structure == report.structure
    get(report.normalized.module, StopAfterOp).properties["source"] = SymbolRefAttr("training")
    with pytest.raises(SemanticError, match="stop_coverage"):
        verify(report.normalized)


@pytest.mark.parametrize("optimized", [False, True])
def test_cold_verification_and_rejection_with_authoring_absent_and_assertions_disabled(tmp_path: Path, optimized: bool) -> None:
    path = tmp_path / "program.mlir"
    path.write_text(dump(normalize(lifetime()).module))
    script = "\n".join(
        (
            "import sys, importlib, json",
            "before = sys._is_gil_enabled()",
            'name = "docs_design.models-strategy-trainer.xdsl_language"',
            'd = importlib.import_module(name + ".dialects")',
            'n = importlib.import_module(name + ".normalize")',
            'v = importlib.import_module(name + ".verify")',
            'b = importlib.import_module("xdsl.dialects.builtin")',
            "m = d.parse(open(sys.argv[1]).read())",
            "report = v.verify(n.normalize(m))",
            "op = next(op for op in m.walk() if isinstance(op, d.StopAfterOp))",
            'op.properties["source"] = b.SymbolRefAttr("training")',
            "try:",
            "    v.verify(n.normalize(m))",
            "except v.SemanticError as error:",
            '    if error.code != "stop_coverage": raise',
            "else:",
            '    raise RuntimeError("invalid stop accepted")',
            'if name + ".fixtures" in sys.modules or "torch" in sys.modules:',
            '    raise RuntimeError("authoring or Torch imported")',
            'print(json.dumps({"before":before,"after":sys._is_gil_enabled(),"class":report.reproducibility}))',
        )
    )
    assert "PYTHON_GIL" not in os.environ
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert json.loads(result.stdout) == {"before": False, "after": False, "class": "unknown"}
