"""An independently installed payload/operation composes without workflow cases."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

from .conformance import check_b
from .dialects import CallOp, ComposeOp, FlowType, FusedOp, LocalSignatureOp, PlanOp, integer, parse
from .extension_conformance import check_extension
from .fixture import execute
from .fusion import PROFILE
from .local_contract import ResourceUse
from .pair_extension import COEFFICIENTS, PAIR, USE, PackContract, PackOp, PairSchema, build_extension, extension_registry
from .payloads import FrozenResource, LocalOutcomeError
from .prepare import NotReady, Rejected, accept, local_maps, origin, prepare
from .runtime import ScalarSession

assert __package__ is not None
preparation = importlib.import_module(__package__ + ".prepare")


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.asyncio
async def test_extension_uses_common_workflow_and_local_interfaces(fuse) -> None:
    registry = extension_registry()
    accepted = accept(build_extension())
    assert accept(parse(accepted.text, extensions=registry.dialects)) == accepted
    prepared = prepare(accepted, registry, fuse=fuse)
    report = asdict((await execute(prepared)).report)
    check_extension(report)
    assert prepared.local.source_groups == (
        (("data.double", "data.offset"), ("pair.pack",), ("pair.first",))
        if fuse
        else (("data.double",), ("data.offset",), ("pair.pack",), ("pair.first",))
    )
    assert report["local_dispatches"] == len(report["tickets"]) * (3 if fuse else 4)
    metadata = json.loads(prepared.manifest)
    resource = next(item for item in metadata["local_contracts"] if item["origin"] == "pair.pack")["resources"]
    assert resource == [{"symbol": USE.symbol, "access": "read_frozen", "version": "1", "digest": COEFFICIENTS.digest}]


def test_payload_type_rejects_invalid_width_at_construction() -> None:
    with pytest.raises(VerifyException, match="positive width"):
        PairSchema(integer(0))


@pytest.mark.parametrize("case", ["effect", "cardinality", "chain"])
def test_operation_schema_and_local_contract_fail_semantically(case, monkeypatch) -> None:
    module = build_extension()
    pack = next(op for op in module.walk() if isinstance(op, PackOp))
    if case == "effect":
        pack.properties["effect"] = StringAttr("write_unknown_state")
    elif case == "cardinality":
        original = PackContract.describe

        def describe(self, op):
            value = original(self, op)
            assert value is not None
            return replace(value, cardinality="zero-or-one")

        monkeypatch.setattr(PackContract, "describe", describe)
    else:
        pack.operands = [pack.output]
    with pytest.raises(Rejected):
        accept(module)


@pytest.mark.parametrize(
    "case", ["codec", "dialect", "lowering", "resource", "version", "digest", "access", "outcomes", "binding", "lower_result"]
)
def test_missing_installed_support_is_a_target_failure(case) -> None:
    registry = extension_registry()
    bindings = dict(registry.bindings)
    if case == "codec":
        registry = replace(registry, codecs={key: value for key, value in registry.codecs.items() if key != PAIR})
    elif case == "dialect":
        registry = replace(registry, dialects=())
    elif case == "resource":
        registry = replace(registry, resources={})
    elif case == "version":
        registry = replace(registry, resources={"pair_coefficients": FrozenResource("2", (1, 2))})
    elif case == "digest":
        registry = replace(registry, resources={"pair_coefficients": FrozenResource("1", (1, 3))})
    elif case == "binding":
        bindings.pop("pack_pair_v1")
    else:
        original = bindings["pack_pair_v1"]
        if case == "lowering":
            original = replace(original, lower=None)
        elif case == "access":
            original = replace(original, resources=(ResourceUse(USE.symbol, "write_live", "1", USE.digest),))
        elif case == "lower_result":
            original = replace(original, lower=lambda resources: None)
        else:
            original = replace(original, outcomes=("returned",))
        bindings["pack_pair_v1"] = original
    registry = replace(registry, bindings=bindings)
    accepted = accept(build_extension())
    with pytest.raises(NotReady):
        prepare(accepted, registry, fuse=True)


def test_positive_new_schema_can_be_accepted_before_target_codec_support() -> None:
    module = build_extension()
    next(op for op in module.walk() if isinstance(op, PackOp)).output._type = FlowType(PairSchema(integer(3)))
    accepted = accept(module)
    with pytest.raises(NotReady, match="payload codec"):
        prepare(accepted, extension_registry())


@pytest.mark.parametrize(
    "case",
    [
        "input_type",
        "output_type",
        "input_codec",
        "output_codec",
        "resources",
        "outcomes",
        "source",
        "missing",
        "duplicate",
        "illegal_fusion",
    ],
)
def test_decoded_local_contract_corruption_precedes_lowering(case, monkeypatch) -> None:
    registry = extension_registry()
    lower_calls = []
    bindings = dict(registry.bindings)
    original_binding = bindings["pack_pair_v1"]
    original_lower = original_binding.lower
    assert original_lower is not None

    def lower(resources):
        lower_calls.append(resources)
        return original_lower(resources)

    bindings["pack_pair_v1"] = replace(original_binding, lower=lower)
    registry = replace(registry, bindings=bindings)
    original = preparation.parse

    def corrupt(text, **options):
        module = original(text, **options)
        root = next((op for op in module.walk() if isinstance(op, PlanOp)), None)
        if root is None:
            return module
        signature = next(op for op in root.body.block.ops if isinstance(op, LocalSignatureOp) and op.origin.data == "pair.pack")
        if case in ("input_type", "output_type"):
            signature.properties[case] = FlowType(StringAttr("feedback"))
        elif case in ("input_codec", "output_codec", "source"):
            signature.properties["origin" if case == "source" else case] = StringAttr("different")
        elif case in ("resources", "outcomes"):
            signature.properties[case] = ArrayAttr([])
        elif case == "missing":
            root.body.block.erase_op(signature)
        elif case == "duplicate":
            root.body.block.add_op(signature.clone())
        else:
            pack = next(op for op in root.walk() if isinstance(op, CallOp) and op.origin.data == "pair.pack")
            first = next(op for op in root.walk() if isinstance(op, CallOp) and op.origin.data == "pair.first")
            block = pack.parent_block()
            group = FusedOp(
                properties={"profile": StringAttr(PROFILE), "origins": ArrayAttr([pack.origin, first.origin])},
                regions=[Region(Block([pack.clone(), first.clone()]))],
            )
            block.insert_op_before(group, pack)
            block.erase_op(pack)
            block.erase_op(first)
        return module

    monkeypatch.setattr(preparation, "parse", corrupt)
    with pytest.raises(NotReady):
        prepare(accept(build_extension()), registry, fuse=True)
    assert not lower_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("case", ["invalid_payload", "declared_failure", "undeclared_failure"])
async def test_extension_outcome_prevents_later_entry_and_keeps_release(fuse, case) -> None:
    registry = extension_registry()
    bindings = dict(registry.bindings)
    original = bindings["pack_pair_v1"]
    original_lower = original.lower
    assert original_lower is not None
    if case == "declared_failure":

        async def negative(value):
            return -10

        bindings["double_v1"] = replace(bindings["double_v1"], invoke=negative)
    else:

        def lower(resources):
            async def invoke(value):
                if case == "invalid_payload":
                    return [value, value * 2]  # Codec requires immutable tuple.
                raise LocalOutcomeError("retry", "undeclared outcome")

            return invoke

        bindings["pack_pair_v1"] = replace(original, lower=lower)
    report = asdict((await execute(prepare(accept(build_extension()), replace(registry, bindings=bindings), fuse=fuse))).report)
    assert report["status"] == "failed" and report["errors"]
    failed = next(ticket for ticket in report["tickets"] if ticket["terminal"] == "failed")
    assert [call["origin"] for call in failed["calls"]] == ["data.double", "data.offset", "pair.pack"]
    call = failed["calls"][-1]
    assert (
        call["outcome"]
        == {"invalid_payload": "invalid_output", "declared_failure": "negative_input", "undeclared_failure": "undeclared_outcome"}[case]
    )
    assert call["entered"] < call["physical_end"] < call["released"] < failed["retired"] < report["released"]
    assert failed["feedback_ack"] is None and failed["publication_ack"] is None


@pytest.mark.asyncio
async def test_payload_oracle_detects_shared_wrong_hidden_lane() -> None:
    registry = extension_registry()
    bindings = dict(registry.bindings)

    def lower(resources):
        async def invoke(value):
            return (value, 3 * value)  # First lane and final scalar still correct.

        return invoke

    bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], lower=lower)
    for fuse in (False, True):
        report = asdict((await execute(prepare(accept(build_extension()), replace(registry, bindings=bindings), fuse=fuse))).report)
        projected = copy.deepcopy(report)
        for ticket in projected["tickets"]:
            ticket["calls"] = [item for item in ticket["calls"] if item["origin"] not in ("pair.pack", "pair.first")]
        check_b(projected)  # Deliberately insufficient without the payload oracle.
        with pytest.raises(AssertionError, match="pair resource/schema arithmetic"):
            check_extension(report)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_resource_binding_physical_return_stays_inside_generic_stage(cancel) -> None:
    entered, release, ended = (threading.Event() for _ in range(3))
    registry = extension_registry()
    bindings = dict(registry.bindings)
    original = bindings["pack_pair_v1"]
    original_lower = original.lower
    assert original_lower is not None

    def lower(resources):
        selected = original_lower(resources)

        async def invoke(value):
            def physical():
                entered.set()
                try:
                    if not release.wait(10):
                        raise TimeoutError("extension physical gate timed out")
                finally:
                    ended.set()

            await asyncio.to_thread(physical)
            return await selected(value)

        return invoke

    bindings["pack_pair_v1"] = replace(original, lower=lower)
    session = ScalarSession(prepare(accept(build_extension()), replace(registry, bindings=bindings), fuse=True), diagnostic_limit=0)
    task = asyncio.create_task(session.run())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        session.stop()
        if cancel:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
        assert not task.done() and session.active_calls == 1 and not ended.is_set()
    finally:
        release.set()
        session.stop()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        report = session.report
    else:
        report = await asyncio.wait_for(task, 5)
    assert ended.is_set() and session.active_calls == session.outstanding == 0
    assert report is not None
    check_extension(asdict(report))


def test_registry_resource_snapshots_are_frozen() -> None:
    registry = extension_registry()
    with pytest.raises(TypeError):
        registry.resources["pair_coefficients"] = FrozenResource("2", (2, 3))  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        registry.resources["pair_coefficients"].values = (2, 3)  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError):
        FrozenResource("1", json.loads("[1,2]"))


@pytest.mark.parametrize("fuse", [False, True])
def test_extension_reconstructs_with_builder_disabled_in_fresh_process(tmp_path: Path, fuse) -> None:
    prepared = prepare(accept(build_extension()), extension_registry(), fuse=fuse)
    (tmp_path / "accepted.mlir").write_text(prepared.accepted.text)
    (tmp_path / "bindings.json").write_text(prepared.manifest)
    script = "import importlib; m=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.extension_demo'); m.build_extension=lambda: (_ for _ in ()).throw(AssertionError('author called')); m.main()"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--load",
            str(tmp_path / "accepted.mlir"),
            "--manifest",
            str(tmp_path / "bindings.json"),
            *(["--fuse"] if fuse else []),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=os.environ | {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    check_extension(json.loads(result.stdout)["report"])


def test_common_modules_do_not_import_or_dispatch_on_the_extension() -> None:
    root = Path(__file__).resolve().parent
    for name in ("prepare.py", "dialects.py", "runtime.py", "fusion.py", "payloads.py", "local_runtime.py"):
        text = (root / name).read_text()
        assert "pairing.pack" not in text and "pair_extension" not in text and "PackOp" not in text
    module = build_extension()
    compose = next(op for op in module.walk() if isinstance(op, ComposeOp))
    assert [origin(op) for op in local_maps(compose)] == ["data.double", "data.offset", "pair.pack", "pair.first"]


def test_lowering_only_receives_declared_immutable_snapshots() -> None:
    registry = extension_registry()
    bindings = dict(registry.bindings)
    original = bindings["pack_pair_v1"].lower
    assert original is not None
    scopes = []

    def lower(resources):
        scopes.append(tuple(resources))
        with pytest.raises(KeyError):
            resources["unrelated"]
        return original(resources)

    bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], lower=lower)
    registry = replace(registry, bindings=bindings, resources=dict(registry.resources) | {"unrelated": FrozenResource("1", (8, 9))})
    prepare(accept(build_extension()), registry)
    assert scopes == [("pair_coefficients",)]


@pytest.mark.parametrize("field_name", ["resources", "outcomes", "lower", "invoke"])
def test_nonlocal_scalar_calls_reject_unsupported_new_binding_fields(field_name) -> None:
    registry = extension_registry()
    bindings = dict(registry.bindings)
    value = {"resources": (USE,), "outcomes": ("returned",), "lower": lambda resources: None, "invoke": json.loads("null")}[field_name]
    bindings["square_v1"] = replace(bindings["square_v1"], **{field_name: value})
    with pytest.raises(NotReady, match="nonlocal binding"):
        prepare(accept(build_extension()), replace(registry, bindings=bindings))


@pytest.mark.parametrize("case", ["missing", "wrong_type"])
def test_local_resolution_rejects_missing_source_identity_explicitly(case) -> None:
    from .payloads import resolve_local

    registry = extension_registry()
    pack = next(op for op in build_extension().walk() if isinstance(op, PackOp))
    if case == "missing":
        pack.properties.pop("source_id")
    else:
        pack.properties["source_id"] = integer(1)
    with pytest.raises(ValueError, match=r"pairing\.pack.*local source identity"):
        resolve_local(pack, registry.bindings["pack_pair_v1"], registry.codecs, registry.resources)


@pytest.mark.asyncio
@pytest.mark.parametrize("reserved", ["returned", "invalid_output", ["returned"]])
async def test_domain_failure_cannot_impersonate_a_runtime_outcome(reserved) -> None:
    registry = extension_registry()
    bindings = dict(registry.bindings)

    def lower(resources):
        async def invoke(value):
            raise LocalOutcomeError(reserved, "reserved label used as a domain failure")

        return invoke

    bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], lower=lower)
    report = (await execute(prepare(accept(build_extension()), replace(registry, bindings=bindings), fuse=True))).report
    assert report.status == "failed"
    failed = next(ticket for ticket in report.tickets if ticket.terminal == "failed")
    assert failed.calls[-1].origin == "pair.pack" and failed.calls[-1].outcome == "undeclared_outcome"
    assert failed.calls[-1].output is None and failed.publication_ack is None and failed.feedback_ack is None


def test_domain_contract_cannot_declare_a_reserved_runtime_failure_label(monkeypatch) -> None:
    original = PackContract.describe

    def describe(self, op):
        description = original(self, op)
        assert description is not None
        return replace(description, outcomes=(*description.outcomes, "entered_failed"))

    monkeypatch.setattr(PackContract, "describe", describe)
    registry = extension_registry()
    bindings = dict(registry.bindings)
    binding = bindings["pack_pair_v1"]
    bindings["pack_pair_v1"] = replace(binding, outcomes=(*binding.outcomes, "entered_failed"))
    with pytest.raises(NotReady, match="reserved runtime outcome"):
        prepare(accept(build_extension()), replace(registry, bindings=bindings))


@pytest.mark.parametrize("outcomes", [("negative_input",), ("returned", "negative_input"), ("invalid_output", "raised", "negative_input")])
def test_operation_and_binding_cannot_jointly_omit_baseline_outcomes(monkeypatch, outcomes) -> None:
    from .payloads import resolve_local

    original = PackContract.describe

    def describe(self, op):
        description = original(self, op)
        assert description is not None
        return replace(description, outcomes=outcomes)

    monkeypatch.setattr(PackContract, "describe", describe)
    registry = extension_registry()
    bindings = dict(registry.bindings)
    bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], outcomes=outcomes)
    module = build_extension()
    pack = next(op for op in module.walk() if isinstance(op, PackOp))
    with pytest.raises(Rejected, match="mandatory baseline local outcomes"):
        accept(module)
    # Cold resolution defends its boundary even for a caller bypassing accept.
    with pytest.raises(ValueError, match="mandatory baseline local outcomes"):
        resolve_local(pack, bindings["pack_pair_v1"], registry.codecs, registry.resources)
