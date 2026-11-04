"""Owner-integrated live/cold execution and independent lifecycle witnesses."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading

from dataclasses import replace
from pathlib import Path

import pytest

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Block, Region


pytest.importorskip("torch", reason="existing CPU owner target imports Torch")
from . import fixtures, standins
from .contracts import AccessMode, Admission, Protocol, Reproducibility, origin, text
from .dialects import CapabilityType, ParallelOp, ResourceType
from .normalize import normalize, source_index
from .observations import compare, observe
from .prepare import Artifact, BindingSpec, CorrespondenceError, Policy, ResourceSpec, load, prepare
from .reference import CallResult, Reference, Request, StandIn, State, Ticket, Update
from .target import Binding, Controls, Effect, ExecutionError, Installed, ResourceBinding, Result, Runtime, backend, graph, owners
from .target_bindings import capability, installed, model
from .test_verify import counter_program, version_flow
from .verify import SemanticError, TargetLimit


async def until(predicate, *, timeout=10):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


def facts(runtime, kind, owner=None):
    return [fact for fact in runtime.facts if fact.kind == kind and (owner is None or fact.owner == owner)]


def finished(runtime, owner):
    return [outcome for outcome in runtime.outcomes.values() if outcome.owner == owner and outcome.state == State.TERMINAL]


def assert_drained(runtime):
    assert not runtime.state.running
    assert runtime.workers.active == 0
    assert runtime.workers.executor is None
    assert not runtime.state.readers and not runtime.state.writers and not runtime.state.state_locks
    assert not runtime.state.active_retentions and not runtime.state.active_producers
    assert all(not flow.active for flow in runtime.flows.values())
    assert all(not feed.ready for feed in runtime.state.feeds.values())


def assert_lifetime(runtime, prefix="", *, cause=("training", 1)):
    assert runtime.value(prefix + "model") == pytest.approx(0.635, abs=1e-12)
    assert [fact.values[1] for fact in facts(runtime, "updated", prefix + "model")] == pytest.approx([0.77, 0.635])
    measurements = facts(runtime, "physical_completed", prefix + "measure")
    assert len(measurements) == 1
    result = measurements[0].values[1][0]
    assert result == pytest.approx({"key": 4, "epoch": 2, "weight": 0.635, "prediction": 6.35, "loss": 0.91125}, abs=1e-12)
    admitted = facts(runtime, "item_admitted", prefix + "training.take")
    assert [item.values[0] for item in admitted] == [0, 1, 2, 3]
    assert [item.values[0] for item in facts(runtime, "item_admitted", prefix + "evaluation.take")] == [4]
    requests = facts(runtime, "request_registered", prefix + "request")
    assert len(requests) == 1
    assert requests[0].values[0].cause[-1] == (prefix + cause[0], cause[1])
    assert measurements[0].values[2] == requests[0].values[0]
    assert measurements[0].values[3] == ((prefix + "records", 4),)
    group = finished(runtime, prefix + "consumers")[0]
    producer = next(outcome for outcome in finished(runtime, prefix + "producer") if outcome.invocation_source == "parent_entry")
    assert group.status == producer.status == "succeeded"
    group_index = next(i for i, fact in enumerate(runtime.facts) if fact.kind == "outcome" and fact.invocation == group.invocation)
    stop_index = next(i for i, fact in enumerate(runtime.facts) if fact.kind == "stop_requested")
    producer_index = next(i for i, fact in enumerate(runtime.facts) if fact.kind == "outcome" and fact.invocation == producer.invocation)
    assert group_index < stop_index < producer_index
    assert runtime.flows[prefix + "records"].high_water <= 3
    assert runtime.state.feeds[prefix + "records"].high_water <= 2
    assert_drained(runtime)


def consumer_observation(items, prefix=""):
    return observe(
        [
            item
            for item in items
            if not any(owner == prefix + "producer" for owner, _ in item.invocation)
            and item.owner not in {prefix + "run", prefix + "activities"}
        ]
    )


def assert_recorded_causality(items):
    observation = observe(items)
    for before, after in observation.required_before:
        assert observation.positions[before] < observation.positions[after], (before, after)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_scope,prefix", [(False, ""), (True, "nested.")])
@pytest.mark.parametrize("cold", [False, True])
async def test_model_and_rearranged_scope_live_cold(extra_scope, prefix, cold):
    bindings = installed()
    plan = prepare(fixtures.lifetime(extra_scope=extra_scope, prefix=prefix), bindings.manifest, Policy())
    if cold:
        artifact = Artifact.from_json(plan.artifact.to_json())
        plan = load(artifact, bindings.manifest, Policy())
    runtime = Runtime(plan, bindings, {prefix + "encoder": 2.0, prefix + "model": 0.8})
    await asyncio.wait_for(runtime.run(), 15)
    assert_lifetime(runtime, prefix)
    oracle = Reference(
        normalize(fixtures.lifetime(extra_scope=extra_scope, prefix=prefix)),
        {prefix + "encoder": 2.0, prefix + "model": 0.8},
        standins.bindings(),
        schemas=standins.schemas(),
    )
    oracle.run()
    assert oracle.settled and not oracle.errors

    compare(consumer_observation(oracle.facts, prefix), consumer_observation(runtime.facts, prefix))
    assert_recorded_causality(runtime.facts)
    assert next(outcome for outcome in finished(runtime, prefix + "producer") if outcome.invocation_source == "parent_entry").stop_requested
    assert any("TaskGroup" in value for value in runtime.owner.source.values())
    assert any("selected policy" in value for value in runtime.owner.source.values())
    assert any("requests from" in value for value in runtime.owner.source.values())
    assert len(runtime.state.image.units) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cold", [False, True])
async def test_data_only_has_no_dummy_model_or_optimizer_and_matches_reference(cold):
    authored = fixtures.data_only(horizon=1)
    bindings = installed()
    plan = prepare(authored, bindings.manifest, Policy())
    if cold:
        plan = load(Artifact.from_json(plan.artifact.to_json()), bindings.manifest, Policy())
    runtime = Runtime(plan, bindings, {"policy": 0})
    await asyncio.wait_for(runtime.run(), 15)
    oracle = Reference(normalize(authored), {"policy": 0}, standins.bindings(), schemas=standins.schemas())
    oracle.run()
    assert oracle.settled and not oracle.errors
    assert runtime.value("policy") == oracle.resources["policy"].value == 4
    assert not runtime.state.image.bindings and not runtime.state.image.units
    for kind, owner in [
        ("physical_completed", "score"),
        ("updated", "policy"),
        ("item_admitted", "consume.take"),
    ]:
        assert [fact.values for fact in facts(runtime, kind, owner)] == [fact.values for fact in facts(oracle, kind, owner)]
    # The horizon of one permits only one extra production read. No other
    # consumer/update surplus is tolerated above.
    reads = [facts(execution, "protected_version", "select.access") for execution in (runtime, oracle)]
    assert [fact.values for fact in reads[0][:4]] == [fact.values for fact in reads[1][:4]]
    for execution, selected in zip((runtime, oracle), reads, strict=True):
        assert len(selected) in (4, 5)
        for extra in selected[4:]:
            assert extra.values[0].owner == "policy" and extra.values[0].ordinal == 4
            assert extra.invocation[-2][0] == "producer"
            assert any(f.values[0] == 4 for f in facts(execution, "selected", "records"))
    assert [fact.values[1][0] for fact in facts(runtime, "physical_completed", "score")] == [1, 25, 25, 49]
    assert_drained(runtime)


@pytest.mark.asyncio
async def test_late_input_keeps_pending_request_and_group_alive():
    controls = Controls(before={("encode", 4): asyncio.Event()})
    bindings = installed()
    authored = fixtures.lifetime()
    runtime = Runtime(prepare(authored, bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8}, controls=controls)
    task = asyncio.create_task(runtime.run())
    try:
        await until(lambda: bool(facts(runtime, "request_registered")))
        assert runtime.value("model") == pytest.approx(0.635)
        assert not finished(runtime, "consumers")
        assert not facts(runtime, "stop_requested")
        assert not facts(runtime, "physical_entered", "measure")
        controls.before["encode", 4].set()
        await asyncio.wait_for(task, 15)
        assert_lifetime(runtime)
    finally:
        controls.before["encode", 4].set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_orderly_stop_after_second_window_admission_retains_request():
    controls = Controls(before={("gradient", 2): asyncio.Event()})
    bindings = installed()
    authored = fixtures.lifetime(windows=3)
    runtime = Runtime(prepare(authored, bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8}, controls=controls)
    task = asyncio.create_task(runtime.run())
    await until(lambda: ("gradient", 2) in controls.reached)
    runtime.request_stop("training")
    controls.before["gradient", 2].set()
    await asyncio.wait_for(task, 15)
    assert_lifetime(runtime)
    assert (
        len(
            [
                outcome
                for outcome in runtime.outcomes.values()
                if outcome.owner == "training" and outcome.invocation_source == "repeat_policy"
            ]
        )
        == 2
    )
    oracle = Reference(normalize(authored), {"encoder": 2.0, "model": 0.8}, standins.bindings(), schemas=standins.schemas())
    for _ in range(5000):
        if oracle.call_counts["gradient"] == 3:
            break
        assert oracle.step()
    assert oracle.call_counts["gradient"] == 3
    oracle.request_stop("training")
    oracle.run()
    assert oracle.settled and not oracle.errors
    compare(consumer_observation(oracle.facts), consumer_observation(runtime.facts))


@pytest.mark.asyncio
@pytest.mark.parametrize("late_failure", [False, True])
async def test_cancellation_retains_physical_protection_and_failure(late_failure):
    gate = threading.Event()
    controls = Controls(gates={("measure", 0): gate}, failures={("measure", 0): "late physical failure"} if late_failure else {})
    bindings = installed()
    runtime = Runtime(
        prepare(fixtures.lifetime(), bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8}, controls=controls
    )
    task = asyncio.create_task(runtime.run())
    writer_entered = asyncio.Event()

    async def competing_write():
        async with runtime.state.use((graph.Use(runtime.resource_names["model"], "write"),)):
            writer_entered.set()

    writer = None
    try:
        await until(lambda: controls.entered.get(("measure", 0), threading.Event()).is_set())
        assert runtime.state.readers[runtime.resource_names["model"]] == 1
        writer = asyncio.create_task(competing_write())
        runtime.cancel(task)
        await asyncio.sleep(0.01)
        assert not task.done() and not writer_entered.is_set()
        assert runtime.workers.active >= 1
        gate.set()
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 15)
        assert isinstance(result[0], BaseException)
        if late_failure:
            assert "late physical failure" in repr(result[0]) or facts(runtime, "physical_failed", "measure")
        if late_failure:
            with pytest.raises(graph.NotReady):
                await asyncio.wait_for(writer, 5)
            assert runtime.resource_names["model"] in runtime.state.unusable
        else:
            await asyncio.wait_for(writer, 5)
            assert writer_entered.is_set()
        assert runtime.value("model") == pytest.approx(0.635)
        completed = [
            i for i, fact in enumerate(runtime.facts) if fact.owner == "measure" and fact.kind in ("physical_completed", "physical_failed")
        ]
        disposition = next(i for i, fact in enumerate(runtime.facts) if fact.kind == "disposition" and fact.values[0] == 4)
        assert completed and max(completed) < disposition
        assert_drained(runtime)
    finally:
        gate.set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)
        if writer is not None:
            await asyncio.gather(writer, return_exceptions=True)


@pytest.mark.asyncio
async def test_failure_before_request_stops_and_drains_producer():
    controls = Controls(failures={("gradient", 2): "training failed"})
    bindings = installed()
    runtime = Runtime(
        prepare(fixtures.lifetime(), bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8}, controls=controls
    )
    with pytest.raises(backend.WorkFailed):
        await asyncio.wait_for(runtime.run(), 15)
    assert runtime.value("model") == pytest.approx(0.77)
    assert not facts(runtime, "request_registered")
    assert not facts(runtime, "physical_entered", "measure")
    assert_drained(runtime)


def test_forged_artifact_rejects_before_resource_factory():
    bindings = installed()
    plan = prepare(fixtures.lifetime(), bindings.manifest, Policy())
    tree = json.loads(plan.artifact.prepared)
    tree["fragments"][0]["source"] = "forged"
    plan = replace(plan, artifact=replace(plan.artifact, prepared=json.dumps(tree)))
    calls = []
    resources = tuple(replace(item, initialize=lambda value: calls.append(value)) for item in bindings.resources)
    with pytest.raises(CorrespondenceError):
        Runtime(plan, replace(bindings, resources=resources), {"encoder": 2.0, "model": 0.8})
    assert not calls


@pytest.mark.asyncio
async def test_wrong_selected_computation_is_detected_by_independent_expectation():
    bindings = installed()
    changed = tuple(Binding(item.spec, lambda _args: Result((0.0,))) if item.spec.name == "score" else item for item in bindings.bindings)
    bindings = replace(bindings, bindings=changed)
    runtime = Runtime(prepare(fixtures.data_only(horizon=1), bindings.manifest, Policy()), bindings, {"policy": 0})
    await asyncio.wait_for(runtime.run(), 15)
    assert [fact.values[1][0] for fact in facts(runtime, "physical_completed", "score")] != [1, 25, 25, 49]
    assert runtime.value("policy") != 4


@pytest.mark.asyncio
async def test_unrequested_template_does_not_invent_an_invocation():
    authored = fixtures.lifetime()
    request = source_index(authored)["request"]
    assert request.parent is not None
    request.parent.detach_op(request)
    bindings = installed()
    runtime = Runtime(prepare(authored, bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8})
    await asyncio.wait_for(runtime.run(), 15)
    assert not facts(runtime, "request_registered")
    assert not facts(runtime, "physical_entered", "measure")
    assert all(outcome.invocation_source != "accepted_request" for outcome in runtime.outcomes.values())
    assert_drained(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [AccessMode.SNAPSHOT, AccessMode.PROTECTED_CURRENT])
async def test_bounded_lag_observed_versions_against_independent_external_schedule(mode):
    authored = version_flow(Admission.BOUNDED_LAG, lag=1, current_snapshot=True)
    index = source_index(authored)
    selection = index["select.access"]
    selection.properties["mode"] = StringAttr(mode)
    resource_type = selection.operands[0].type
    assert isinstance(resource_type, ResourceType)
    selection.regions[0].block.args[0]._type = CapabilityType(resource_type, StringAttr(mode))
    first = index["bump.access"]
    second = first.clone()
    for operation in second.walk():
        operation.properties["origin"] = StringAttr(origin(operation) + ".2")
        if "sym_name" in operation.properties:
            operation.properties["sym_name"] = StringAttr(text(operation, "sym_name") + ".2")
    assert first.parent is not None
    first.parent.insert_op_after(second, first)
    base = installed()

    def bump(args):
        return Result((args[0].value + 1,), (Effect(0, "assign", args[0].value + 1),))

    selected = replace(base, bindings=(*base.bindings, Binding(BindingSpec("bump", ("capability",), ("scalar",), updates=(0,)), bump)))
    plan = prepare(authored, selected.manifest, Policy())
    assert plan.verification.reproducibility == Reproducibility.MAY_DEPEND
    controls = Controls(before={key: asyncio.Event() for key in (("bump.access", 0), ("select.access", 2), ("consume.take", 0))})
    runtime = Runtime(plan, selected, {"policy": 0}, controls=controls)
    task = asyncio.create_task(runtime.run())
    try:
        await until(lambda: len(facts(runtime, "published")) == 2)
        controls.before["bump.access", 0].set()
        await until(lambda: runtime._version("policy").ordinal == 2)
        controls.before["select.access", 2].set()
        controls.before["consume.take", 0].set()
        await asyncio.wait_for(task, 15)
    finally:
        for event in controls.before.values():
            event.set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)

    oracle_bindings = {
        **standins.bindings(),
        "bump": StandIn(
            ("capability",),
            ("scalar",),
            lambda args: CallResult(
                (standins.numeric(standins.capability(args[0]).value) + 1,),
                (Update(0, standins.numeric(standins.capability(args[0]).value) + 1),),
            ),
            updates=(0,),
        ),
    }
    oracle = Reference(normalize(authored), {"policy": 0}, oracle_bindings, schemas=standins.schemas())
    # Supply the same phases independently: old prefill, two completed writes,
    # then ordinary consumer/producer progress.
    for _ in range(1000):
        if len(facts(oracle, "published")) == 2:
            break
        choices = [
            i
            for i in oracle.enabled()
            if (command := oracle.tasks[i].command) is not None and not origin(command.op).startswith("bump") and command.kind != "take"
        ]
        assert choices
        oracle.step(choices[0])
    assert len(facts(oracle, "published")) == 2
    for _ in range(1000):
        if oracle.resources["policy"].ordinal == 2:
            break
        choices = [
            i for i in oracle.enabled() if (command := oracle.tasks[i].command) is not None and origin(command.op).startswith("bump")
        ]
        assert choices
        oracle.step(choices[0])
    assert oracle.resources["policy"].ordinal == 2
    oracle.run()
    assert oracle.settled and not oracle.errors
    for kind in ("admission_checked", "item_rejected", "item_admitted"):
        assert [item.values for item in facts(runtime, kind)] == [item.values for item in facts(oracle, kind)]
    assert [item.values[0] for item in facts(runtime, "item_rejected")] == [0, 1]
    assert [item.values[0] for item in facts(runtime, "item_admitted")] == [2]
    assert runtime.flows["records"].terminal[0] == runtime.flows["records"].terminal[1] == "rejected_stale"
    assert_drained(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [AccessMode.SNAPSHOT, AccessMode.PROTECTED_CURRENT])
async def test_snapshot_isolates_while_current_access_excludes_writer_through_physical_use(mode):
    authored = version_flow(Admission.ANY_WITH_PROVENANCE, current_snapshot=True)
    selection = source_index(authored)["select.access"]
    selection.properties["mode"] = StringAttr(mode)
    kind = selection.operands[0].type
    assert isinstance(kind, ResourceType)
    selection.regions[0].block.args[0]._type = CapabilityType(kind, StringAttr(mode))
    base = installed()

    def bump(args):
        return Result((args[0].value + 1,), (Effect(0, "assign", args[0].value + 1),))

    bindings = replace(base, bindings=(*base.bindings, Binding(BindingSpec("bump", ("capability",), ("scalar",), updates=(0,)), bump)))
    gate = threading.Event()
    controls = Controls(gates={("select", 0): gate}, before={("bump.access", 0): asyncio.Event()})
    runtime = Runtime(prepare(authored, bindings.manifest, Policy()), bindings, {"policy": 0}, controls=controls)
    task = asyncio.create_task(runtime.run())
    try:
        await until(lambda: controls.entered.get(("select", 0), threading.Event()).is_set())
        controls.before["bump.access", 0].set()
        if mode == AccessMode.SNAPSHOT:
            await until(lambda: runtime._version("policy").ordinal == 1)
            assert not gate.is_set()
        else:
            await asyncio.sleep(0.01)
            assert runtime._version("policy").ordinal == 0
            assert runtime.resource_names["policy"] in runtime.state.state_locks
        gate.set()
        await asyncio.wait_for(task, 15)
    finally:
        gate.set()
        controls.before["bump.access", 0].set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)
    oracle_binding = StandIn(
        ("capability",),
        ("scalar",),
        lambda args: CallResult(
            (standins.numeric(standins.capability(args[0]).value) + 1,),
            (Update(0, standins.numeric(standins.capability(args[0]).value) + 1),),
        ),
        updates=(0,),
    )
    oracle = Reference(normalize(authored), {"policy": 0}, {**standins.bindings(), "bump": oracle_binding}, schemas=standins.schemas())
    oracle.held.add(("select", 0))
    for _ in range(1000):
        if facts(oracle, "physical_entered", "select"):
            break
        choices = [
            i for i in oracle.enabled() if (command := oracle.tasks[i].command) is not None and not origin(command.op).startswith("bump")
        ]
        assert choices
        oracle.step(choices[0])
    assert facts(oracle, "physical_entered", "select")
    oracle.run()
    assert oracle.resources["policy"].ordinal == (1 if mode == AccessMode.SNAPSHOT else 0)
    oracle.held.clear()
    oracle.run()
    assert oracle.settled and not oracle.errors
    assert runtime.value("policy") == oracle.resources["policy"].value == 1
    assert facts(runtime, "item_admitted")[0].values == facts(oracle, "item_admitted")[0].values
    assert facts(runtime, "physical_completed", "score")[0].values[1] == facts(oracle, "physical_completed", "score")[0].values[1] == (1.0,)
    assert_drained(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["training", "inner"])
async def test_nested_group_preserves_outer_handler_pending_work(source):
    authored = fixtures.lifetime()
    group = source_index(authored)["consumers"]
    detached = group.detach_region(0)
    body = detached.detach_block(0)
    wrapper = Block(arg_types=[value.type for value in body.args])
    wrapper.add_op(fixtures.structured(ParallelOp, "inner", wrapper.args, [body]))
    fixtures.end(wrapper, "inner.branch")
    group.add_region(Region(wrapper))
    group.regions = (group.regions[-1], *group.regions[:-1])
    if source == "inner":
        from xdsl.dialects.builtin import ArrayAttr, SymbolRefAttr

        index = source_index(authored)
        index["request"].properties["source"] = SymbolRefAttr("inner")
        index["evaluation"].properties["sources"] = ArrayAttr([SymbolRefAttr("inner")])
    bindings = installed()
    controls = Controls(before={("evaluation", 0): asyncio.Event()})
    if source == "inner":
        with pytest.raises(SemanticError, match="settlement_cycle"):
            prepare(authored, bindings.manifest, Policy())
        return
    runtime = Runtime(prepare(authored, bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8}, controls=controls)
    task = asyncio.create_task(runtime.run())
    try:
        await until(lambda: bool(facts(runtime, "request_pending")))
        if source == "training":
            await until(lambda: any(outcome.owner == "inner" and outcome.state == State.SETTLING for outcome in runtime.outcomes.values()))
            assert not finished(runtime, "inner")
        assert not facts(runtime, "request_disposed")
        assert not finished(runtime, "consumers")
        controls.before["evaluation", 0].set()
        await asyncio.wait_for(task, 15)
        assert_lifetime(runtime, cause=(source, 1 if source == "training" else 0))
    finally:
        controls.before["evaluation", 0].set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)
    oracle = Reference(normalize(authored), {"encoder": 2.0, "model": 0.8}, standins.bindings(), schemas=standins.schemas())
    for _ in range(5000):
        choices = [i for i in oracle.enabled() if (command := oracle.tasks[i].command) is not None and origin(command.op) != "evaluation"]
        if not choices:
            break
        oracle.step(choices[0])
    assert facts(oracle, "request_pending") and not facts(oracle, "request_disposed")
    if source == "training":
        assert not finished(oracle, "inner")
    oracle.run()
    assert oracle.settled and not oracle.errors
    compare(consumer_observation(oracle.facts), consumer_observation(runtime.facts))


@pytest.mark.asyncio
async def test_live_only_binding_does_not_materialize_an_undeclared_snapshot():
    import torch

    authored = counter_program()
    bump = source_index(authored)["bump.access"]
    assert bump.parent is not None
    bump.parent.erase_op(bump)
    copies = []

    class LiveOnly(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(float(value), dtype=torch.float64))

        def __deepcopy__(self, memo):
            copies.append(True)
            raise RuntimeError("copying is not supported")

    bindings = Installed(
        (
            Binding(
                BindingSpec("read", ("capability",), ("scalar",)),
                lambda args: Result((float(model(capability(args[0]).value).get_parameter("weight").detach()),)),
            ),
        ),
        (ResourceBinding(ResourceSpec("counter", "participant", parameter="weight"), LiveOnly, lambda value: None),),
        {},
    )
    runtime = Runtime(prepare(authored, bindings.manifest, Policy()), bindings, {"state": 3.0})
    await asyncio.wait_for(runtime.run(), 10)
    assert not copies and not runtime.initial_snapshots
    assert facts(runtime, "physical_completed", "read")[0].values[1] == (3.0,)
    assert_drained(runtime)


@pytest.mark.asyncio
async def test_disposed_ready_product_cannot_be_admitted_after_close():
    bindings = installed()
    runtime = Runtime(prepare(fixtures.data_only(), bindings.manifest, Policy()), bindings, {"policy": 0})
    flow, feed = runtime.flows["records"], runtime.state.feeds["records"]
    await flow.capacity.acquire()
    ticket = Ticket("records", 0, (), versions=(runtime._version("policy"),))
    flow.active[0] = ticket
    await feed.publish(owners.Product(0, ticket, ()))
    runtime._dispose(ticket, "discarded", ())
    await feed.close()
    with pytest.raises(ExecutionError, match="exhausted"):
        await runtime._take(source_index(runtime.module)["consume.take"], "records", ())
    assert not facts(runtime, "item_admitted") and not feed.ready and not flow.active


@pytest.mark.asyncio
async def test_disposed_request_packet_is_declined_without_reentry():
    bindings = installed()
    runtime = Runtime(prepare(fixtures.lifetime(), bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8})
    op = source_index(runtime.module)["evaluation"]
    scope = (("consumers", 0),)
    slot = runtime._slot(op, scope)
    request = Request("evaluation", 0, scope, 1)
    await slot.capacity.acquire()
    slot.pending[0] = request
    queue = runtime.owner.request_queues[slot.name]
    queue.put_nowait((graph.Completion("source", "body", 0, 0, 1), None))
    queue.put_nowait(None)
    runtime.cancelled = True
    await runtime._settle_pending(scope)
    await runtime._protocol(op, Protocol.REQUESTS, ("records", "model"), (*scope, ("evaluation", 0)))
    assert slot.dispositions == {0: "cancelled_unfulfilled"}
    assert not slot.pending and slot.current is None
    assert len(facts(runtime, "request_disposed")) == 1
    assert not facts(runtime, "physical_entered")


@pytest.mark.asyncio
@pytest.mark.parametrize("name,prefix", [("lifetime", ""), ("extra-scope", "nested."), ("data-only", "")])
async def test_checked_in_artifacts_hashes_cold_load_and_execute(name, prefix):
    folder = Path(__file__).parent / "evidence_prepared"
    record = json.loads((folder / "qualification.json").read_text())
    for filename, expected in record["artifact_hashes"].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected
    selected = folder / name
    artifact = Artifact(
        (selected / "accepted.mlir").read_text(), (selected / "normalized.mlir").read_text(), (selected / "prepared.json").read_text()
    )
    bindings = installed()
    plan = load(artifact, bindings.manifest, Policy())
    initial = {"policy": 0} if name == "data-only" else {prefix + "encoder": 2.0, prefix + "model": 0.8}
    runtime = Runtime(plan, bindings, initial)
    await asyncio.wait_for(runtime.run(), 15)
    if name == "data-only":
        assert not runtime.state.image.bindings and not runtime.state.image.units
        assert len(facts(runtime, "item_admitted")) == 4
        assert_drained(runtime)
    else:
        assert_lifetime(runtime, prefix)


@pytest.mark.asyncio
async def test_invalid_selected_output_retains_physical_return_and_validation_rejection():
    base = installed()
    changed = tuple(
        Binding(item.spec, lambda args: Result(("invalid scalar",))) if item.spec.name == "score" else item for item in base.bindings
    )
    bindings = replace(base, bindings=changed)
    runtime = Runtime(prepare(fixtures.data_only(horizon=1), bindings.manifest, Policy()), bindings, {"policy": 0})
    with pytest.raises(backend.WorkFailed):
        await asyncio.wait_for(runtime.run(), 15)
    assert facts(runtime, "physical_completed", "score")[0].values[1] == ("invalid scalar",)
    assert len(facts(runtime, "validation_rejected", "score")) == 1
    assert not facts(runtime, "physical_entered", "feedback")
    assert runtime.value("policy") == 0
    assert_drained(runtime)


@pytest.mark.parametrize("name", ["model", "consumers"])
def test_external_stop_rejects_unsupported_protocol_without_changing_admission(name):
    bindings = installed()
    runtime = Runtime(prepare(fixtures.lifetime(), bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8})
    with pytest.raises(TargetLimit, match="repeat or producer"):
        runtime.request_stop(name)
    assert not runtime.stopped


@pytest.mark.asyncio
async def test_producer_stop_wakes_capacity_wait_without_new_production():
    authored = fixtures.data_only(horizon=1)
    bindings = installed()
    controls = Controls(before={("consume.take", 0): asyncio.Event()})
    runtime = Runtime(prepare(authored, bindings.manifest, Policy()), bindings, {"policy": 0}, controls=controls)
    task = asyncio.create_task(runtime.run())
    try:
        await until(lambda: bool(runtime.state.feeds["records"].ready))
        runtime.request_stop("producer")
        await until(lambda: any(outcome.invocation_source == "parent_entry" for outcome in finished(runtime, "producer")))
        assert not task.done()
        assert [fact.values[0] for fact in facts(runtime, "selected")] == [0]
        assert not runtime.flows["records"].active and not runtime.state.feeds["records"].ready
        controls.before["consume.take", 0].set()
        with pytest.raises(backend.WorkFailed):
            await asyncio.wait_for(task, 10)
        assert [fact.values[0] for fact in facts(runtime, "selected")] == [0]
        assert runtime.call_counts["select"] == 1 and runtime.value("policy") == 0
        assert_drained(runtime)
    finally:
        controls.before["consume.take", 0].set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)
    oracle = Reference(normalize(authored), {"policy": 0}, standins.bindings(), schemas=standins.schemas())
    for _ in range(1000):
        if facts(oracle, "published"):
            break
        choices = [i for i in oracle.enabled() if (command := oracle.tasks[i].command) is not None and command.kind != "take"]
        assert choices
        oracle.step(choices[0])
    assert facts(oracle, "published")
    oracle.request_stop("producer")
    for _ in range(1000):
        if any(outcome.invocation_source == "parent_entry" for outcome in finished(oracle, "producer")):
            break
        choices = [i for i in oracle.enabled() if (command := oracle.tasks[i].command) is not None and origin(command.op) == "producer"]
        assert choices
        oracle.step(choices[0])
    oracle.run()
    assert oracle.settled and oracle.errors
    assert [fact.values for fact in facts(runtime, "selected")] == [fact.values for fact in facts(oracle, "selected")]
    assert [fact.values for fact in facts(runtime, "disposition")] == [fact.values for fact in facts(oracle, "disposition")]


@pytest.mark.asyncio
async def test_producer_stop_retains_claim_through_entered_consumer_use():
    authored = fixtures.data_only(horizon=1)
    bindings = installed()
    gate = threading.Event()
    controls = Controls(gates={("score", 0): gate})
    runtime = Runtime(prepare(authored, bindings.manifest, Policy()), bindings, {"policy": 0}, controls=controls)
    task = asyncio.create_task(runtime.run())
    try:
        await until(lambda: controls.entered.get(("score", 0), threading.Event()).is_set())
        runtime.request_stop("producer")
        await asyncio.sleep(0.01)
        assert 0 in runtime.flows["records"].active
        assert not any(outcome.invocation_source == "parent_entry" for outcome in finished(runtime, "producer"))
        assert not facts(runtime, "disposition")
        gate.set()
        with pytest.raises(backend.WorkFailed):
            await asyncio.wait_for(task, 10)
        assert [fact.values[0] for fact in facts(runtime, "selected")] == [0]
        assert runtime.call_counts["select"] == 1 and runtime.value("policy") == 1
        completed = next(i for i, fact in enumerate(runtime.facts) if fact.kind == "physical_completed" and fact.owner == "score")
        disposed = next(i for i, fact in enumerate(runtime.facts) if fact.kind == "disposition" and fact.values[0] == 0)
        assert completed < disposed
        assert_drained(runtime)
    finally:
        gate.set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)
    oracle = Reference(normalize(authored), {"policy": 0}, standins.bindings(), schemas=standins.schemas())
    oracle.held.add(("score", 0))
    for _ in range(1000):
        if facts(oracle, "physical_entered", "score"):
            break
        assert oracle.step()
    assert facts(oracle, "physical_entered", "score")
    oracle.request_stop("producer")
    oracle.run()
    assert not oracle.settled and 0 in oracle.streams["records"].outstanding
    oracle.held.clear()
    oracle.run()
    assert oracle.settled and oracle.errors
    assert runtime.value("policy") == oracle.resources["policy"].value
    for kind, owner in (("selected", "records"), ("updated", "policy"), ("physical_completed", "score")):
        assert [fact.values for fact in facts(runtime, kind, owner)] == [fact.values for fact in facts(oracle, kind, owner)]


@pytest.mark.asyncio
async def test_producer_stop_before_service_entry_never_selects_work():
    bindings = installed()
    runtime = Runtime(prepare(fixtures.data_only(horizon=1), bindings.manifest, Policy()), bindings, {"policy": 0})
    runtime.request_stop("producer")
    with pytest.raises(backend.WorkFailed):
        await asyncio.wait_for(runtime.run(), 10)
    assert not facts(runtime, "selected") and not facts(runtime, "physical_entered")
    assert runtime.value("policy") == 0
    assert_drained(runtime)


@pytest.mark.asyncio
async def test_cancellation_disposes_request_before_handler_admission():
    authored = fixtures.lifetime()
    bindings = installed()
    controls = Controls(before={("evaluation", 0): asyncio.Event()})
    runtime = Runtime(prepare(authored, bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8}, controls=controls)
    task = asyncio.create_task(runtime.run())
    try:
        await until(lambda: bool(facts(runtime, "request_pending")))
        runtime.cancel(task)
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 15)
        assert not any(outcome.invocation_source == "accepted_request" for outcome in runtime.outcomes.values())
        assert not any(slot.pending or slot.current for slot in runtime.handlers.values())
        dispositions = facts(runtime, "request_disposed")
        assert len(dispositions) == 1 and dispositions[0].values[1] == "cancelled_unfulfilled"
        oracle = Reference(normalize(authored), {"encoder": 2.0, "model": 0.8}, standins.bindings(), schemas=standins.schemas())
        for _ in range(5000):
            if facts(oracle, "request_pending"):
                break
            assert oracle.step()
        assert facts(oracle, "request_pending")
        oracle.cancelled = True
        oracle.run()
        assert oracle.settled
        assert [fact.values for fact in facts(oracle, "request_disposed")] == [fact.values for fact in dispositions]
        assert_drained(runtime)
    finally:
        controls.before["evaluation", 0].set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_partial_optimizer_effect_retains_owner_uncertainty_and_prior_update():
    import torch

    bindings = installed()
    runtime = Runtime(prepare(fixtures.lifetime(), bindings.manifest, Policy()), bindings, {"encoder": 2.0, "model": 0.8})
    unit = runtime.state.image.units[runtime.resource_names["model"]]
    original = unit.optimizer.step
    calls = 0

    def step():
        nonlocal calls
        calls += 1
        if calls == 2:
            with torch.no_grad():
                unit.parameters[0].fill_(123.0)
            raise RuntimeError("optimizer failed after effect")
        original()

    unit.optimizer.step = step
    with pytest.raises(backend.WorkFailed):
        await asyncio.wait_for(runtime.run(), 15)
    assert runtime.value("model") == 123.0
    assert unit.steps == 1 and runtime._version("model").ordinal == 1
    assert [(outcome.optimizer, outcome.reset) for outcome in unit.outcomes] == [("returned", "returned"), ("uncertain", "not_attempted")]
    assert runtime.resource_names["model"] in runtime.state.unusable
    assert not facts(runtime, "request_pending")
    assert_drained(runtime)
