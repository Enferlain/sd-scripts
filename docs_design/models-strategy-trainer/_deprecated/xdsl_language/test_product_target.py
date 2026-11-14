"""Real storage/derivative retention with the existing owner and worker services."""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import weakref

from dataclasses import replace

import pytest


pytest.importorskip("torch", reason="owner engine and actual autograd require Torch")
from . import product_bindings, product_case, product_reference
from xdsl.dialects.builtin import StringAttr
from .contracts import AccessMode, Freshness
from .dialects import PayloadType, CapabilityType, ResourceType
from .fixtures import call
from .normalize import source_index
from .prepare import Artifact, CorrespondenceError, Policy, load, prepare
from .target import Controls, ExecutionError, ResourceBinding, Runtime, graph, backend
from .verify import SemanticError


async def until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.001)


def runtime_for(kind="buffer", *, derived=False, fused=False, controls=None, history=True, wrong=False, update="none"):
    selected = product_bindings.installed(kind, derived=derived, wrong=wrong)
    plan = prepare(product_case.program(kind, derived=derived, update=update), selected.manifest, Policy(fuse_local=fused))
    cold = load(Artifact.from_json(plan.artifact.to_json()), selected.manifest, plan.policy)
    return Runtime(cold, selected, product_case.initial(kind), controls=controls, history=history)


def drained(runtime):
    assert not runtime.retained and not runtime.state.state_locks
    assert runtime.workers.active == 0 and runtime.workers.executor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["buffer", "autograd"])
@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("fused", [False, True])
async def test_actual_views_graphs_and_reference_agree_after_cold_loading(kind, derived, fused):
    runtime = runtime_for(kind, derived=derived, fused=fused)
    await asyncio.wait_for(runtime.run(), 10)
    expected = product_reference.expected(kind, derived)
    assert runtime.value("result") == expected
    oracle = product_reference.reference(kind, derived=derived)
    oracle.run()
    assert oracle.resources["result"].value == expected
    assert runtime.product_receipts == oracle.product_receipts
    assert len(runtime.product_receipts) == (2 if derived else 1)
    assert not any(key.startswith("$fused") for key in runtime.local_dispatches)
    completed = [item.owner for item in runtime.facts if item.kind == "physical_completed"]
    assert completed == (["make", "derive", "consume", "publish"] if derived else ["make", "consume", "publish"])
    drained(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["buffer", "autograd"])
@pytest.mark.parametrize("late_failure", [False, True])
@pytest.mark.parametrize("erased", [False, True])
async def test_entered_consumer_holds_product_and_storage_through_repeated_cancellation(kind, late_failure, erased):
    gate = threading.Event()
    controls = Controls(gates={("consume", 0): gate}, failures={("consume", 0): "late retained consumer failure"} if late_failure else {})
    runtime = runtime_for(kind, derived=True, controls=controls, history=not erased)
    if erased:

        async def checkpoint(point):
            runtime.facts.clear()
            runtime.outcomes.clear()

        controls.checkpoint = checkpoint
    task = asyncio.create_task(runtime.run())
    acquired = asyncio.Event()
    internal = runtime.resource_names["storage"]

    async def writer():
        async with runtime.state.acquire((), (internal,)):
            acquired.set()

    competing = None
    try:
        await until(lambda: controls.entered.get(("consume", 0), threading.Event()).is_set())
        product = runtime.retained["derive", 0]
        assert product.borrowers == 1 and internal in runtime.state.state_locks
        competing = asyncio.create_task(writer())
        runtime.cancel(task)
        for _ in range(3):
            await asyncio.sleep(0)
            runtime.cancel(task)
        assert not task.done() and not acquired.is_set() and runtime.workers.active
        with pytest.raises(ExecutionError, match="dependent physical"):
            runtime._release_product(product, "forged")
        with pytest.raises(ExecutionError, match="dependent physical"):
            runtime._release_product(runtime.retained["make", 0], "forged")
        assert not runtime.product_receipts and product.active
        gate.set()
        outcome = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        assert isinstance(outcome[0], BaseException)
        if late_failure:
            with pytest.raises(graph.NotReady):
                await asyncio.wait_for(competing, 5)
            assert internal in runtime.state.unusable_states
        else:
            await asyncio.wait_for(competing, 5)
            assert acquired.is_set()
        assert runtime.value("result") == 0
        assert all(status == "cancelled" for status, _ in runtime.product_receipts.values())
        assert not product.active and product.value is None and not product.parents and not product.capabilities
        if erased:
            assert not runtime.facts and not runtime.outcomes
        else:
            assert any(
                item.kind == ("physical_failed" if late_failure else "physical_completed") and item.owner == "consume"
                for item in runtime.facts
            )
        drained(runtime)
    finally:
        gate.set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)
        if competing is not None:
            await asyncio.gather(competing, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["buffer", "autograd"])
async def test_wrong_result_is_rejected_by_independent_expectation(kind):
    runtime = runtime_for(kind, wrong=True, history=False)
    await runtime.run()
    assert runtime.value("result") != product_reference.expected(kind)
    drained(runtime)


def test_known_reuse_and_cold_corruption_reject_before_factories():
    selected = product_bindings.installed()
    factories = []
    selected = replace(
        selected,
        resources=tuple(ResourceBinding(item.spec, lambda value: factories.append(value), item.validate) for item in selected.resources),
    )
    with pytest.raises(SemanticError, match="retained_update"):
        Runtime(prepare(product_case.program(update="before"), selected.manifest, Policy()), selected, product_case.initial())
    plan = prepare(product_case.program(), selected.manifest, Policy())
    recipe = json.loads(plan.artifact.prepared)
    next(fragment for fragment in recipe["fragments"] if fragment["source"] == "make")["realization"]["retains"] = []
    forged = replace(plan, artifact=replace(plan.artifact, prepared=json.dumps(recipe)))
    with pytest.raises(CorrespondenceError):
        Runtime(forged, selected, product_case.initial())
    assert factories == []


@pytest.mark.asyncio
async def test_release_permits_a_later_owner_replacement():
    runtime = runtime_for(update="after")
    await runtime.run()
    assert runtime.value("result") == product_reference.expected("buffer")
    assert runtime._version("storage").ordinal == 1
    drained(runtime)


@pytest.mark.asyncio
async def test_an_unmarked_nonleaf_tensor_cannot_be_silently_treated_as_detached():

    selected = product_bindings.installed("autograd")
    module = product_case.program("autograd")
    index = source_index(module)
    index["make"].results[0]._type = PayloadType(StringAttr("autograd"))
    del index["make"].properties["retains"]
    block = index["release"].parent_block()
    assert block is not None
    block.erase_op(index["release"])
    selected = replace(
        selected,
        bindings=tuple(
            replace(binding, spec=replace(binding.spec, outputs=("autograd",), retains=()))
            if binding.spec.name == "make_autograd"
            else replace(binding, spec=replace(binding.spec, inputs=("autograd", "capability")))
            if binding.spec.name == "consume_autograd"
            else binding
            for binding in selected.bindings
        ),
    )
    runtime = Runtime(prepare(module, selected.manifest, Policy()), selected, product_case.initial("autograd"), history=False)
    with pytest.raises(backend.WorkFailed, match="detached result isolation failed"):
        await runtime.run()
    assert not runtime.call_counts.get("consume") and not runtime.retained
    drained(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["buffer", "autograd"])
async def test_multiple_borrows_preserve_the_product_until_explicit_release_and_drop_references(kind):

    selected = product_bindings.installed(kind)
    references = []
    original = next(binding for binding in selected.bindings if binding.spec.name == "make_" + kind)

    def make(args):
        result = original.invoke(args)
        references.append(weakref.ref(result.values[0]))
        return result

    selected = replace(
        selected, bindings=tuple(replace(binding, invoke=make) if binding is original else binding for binding in selected.bindings)
    )
    module = product_case.program(kind)
    index = source_index(module)
    block = index["make"].parent_block()
    assert block is not None
    extra = call(block, "consume.again", "consume_" + kind, index["consume"].operands, [index["consume"].results[0].type])
    block.detach_op(extra)
    block.insert_op_before(extra, index["release"])
    runtime = Runtime(prepare(module, selected.manifest, Policy()), selected, product_case.initial(kind))
    await runtime.run()
    assert runtime.value("result") == product_reference.expected(kind)
    assert runtime.call_counts["consume"] == runtime.call_counts["consume.again"] == 1
    gc.collect()
    assert references and all(reference() is None for reference in references)
    drained(runtime)


@pytest.mark.asyncio
async def test_snapshot_product_keeps_its_observed_version_while_live_owner_advances():

    selected = product_bindings.installed()
    module = product_case.program()
    use = source_index(module)["use"]
    use.properties.update(
        mode=StringAttr(AccessMode.SNAPSHOT), freshness=StringAttr(Freshness.CURRENT_AT_START), version_selector=StringAttr("current")
    )
    resource_type = use.operands[0].type
    assert isinstance(resource_type, ResourceType)
    use.regions[0].block.args[0]._type = CapabilityType(resource_type, StringAttr(AccessMode.SNAPSHOT))
    gate = threading.Event()
    controls = Controls(gates={("consume", 0): gate})
    runtime = Runtime(prepare(module, selected.manifest, Policy()), selected, product_case.initial(), controls=controls)
    task = asyncio.create_task(runtime.run())
    try:
        await until(lambda: controls.entered.get(("consume", 0), threading.Event()).is_set())
        async with runtime._access("storage", AccessMode.OWNER_UPDATE, Freshness.OWNER_PROTOCOL):
            cell = runtime.state.owners[runtime.resource_names["storage"]]
            cell.value, cell.ordinal = bytearray(b"uvwxyz"), cell.ordinal + 1
        assert runtime._version("storage").ordinal == 1
        assert runtime.retained["make", 0].capabilities[0].version.ordinal == 0
        gate.set()
        await asyncio.wait_for(task, 10)
        assert runtime.value("result") == 297
        assert runtime.product_receipts["make", 0][1][0].ordinal == 0
        drained(runtime)
    finally:
        gate.set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", ["cancel", "failure"])
async def test_closing_at_physical_completion_cannot_admit_a_retained_result(closing):
    runtime = runtime_for()
    task = None

    async def checkpoint(point):
        if point.source == "make" and point.phase == "call_complete":
            if closing == "cancel":
                runtime.cancel(task)
            else:
                runtime.failed = True

    runtime.controls.checkpoint = checkpoint
    task = asyncio.create_task(runtime.run())
    result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
    assert isinstance(result[0], BaseException)
    assert not runtime.product_receipts
    assert not any(item.kind == "product_retained" for item in runtime.facts)
    assert not runtime.call_counts.get("consume") and runtime.value("result") == 0
    drained(runtime)
