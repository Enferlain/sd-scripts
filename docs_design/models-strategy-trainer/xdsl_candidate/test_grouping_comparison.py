"""Matched frozen-read grouping, cold legality and analysis dependencies."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Block, Region

from . import pair_extension as pair
from . import prepare as preparation
from .dialects import CallOp, ChainOp, ComposeOp, FusedOp, dump, parse
from .extension_conformance import check_extension
from .fixture import execute
from .fusion import PROFILE, LocalAnalyses, dispatch_groups, rewrite
from .payloads import FrozenResource

_prefix = "docs_design.models-strategy-trainer"
native_tests = importlib.import_module(_prefix + ".pliron_candidate.test_b")
native_pair = importlib.import_module(_prefix + ".pliron_candidate.pair_extension")
native = native_tests.native


def variant(native, framework, *, opaque=False, fuse=False, horizon=2, selected=None):
    effects = []
    if framework == "xdsl":
        selected = pair.extension_registry(opaque=opaque, effects=effects) if selected is None else selected
        accepted = preparation.accept(pair.build_extension(opaque=opaque, horizon=horizon))
        ready = preparation.prepare(accepted, selected, fuse=fuse)
        artifact = {"accepted": accepted.text, "prepared": ready.text, "manifest": json.loads(ready.manifest)}
    else:
        selected = native_pair.registry(opaque=opaque, effects=effects) if selected is None else selected
        ready, artifact = native.prepare(native.accept(native_pair.build(opaque=opaque, horizon=horizon)), selected, fuse=fuse)
    return ready, artifact, selected, effects


def cold(native, framework, artifact, selected, *, fuse=True):
    if framework == "xdsl":
        accepted = preparation.accept(parse(artifact["accepted"], extensions=selected.dialects))
        return preparation.prepare(accepted, selected, fuse=fuse, saved_text=artifact["prepared"], saved_manifest=artifact["manifest"])
    return native.load(artifact["accepted"], artifact["prepared"], artifact["manifest"], selected)[0]


def forge_group(framework, artifact, selected):
    if framework == "xdsl":
        module = parse(artifact["prepared"], extensions=selected.dialects)
        producer = next(op for op in module.walk() if isinstance(op, ChainOp) and op.role.data == "producer")
        calls = tuple(producer.body.block.ops)
        assert len(calls) == 4 and all(isinstance(call, CallOp) for call in calls)
        group = FusedOp(
            properties={"profile": StringAttr(PROFILE), "origins": ArrayAttr(call.origin for call in calls)},
            regions=[Region(Block(call.clone() for call in calls))],
        )
        for call in calls:
            producer.body.block.erase_op(call)
        producer.body.block.add_op(group)
        return dump(module)
    text = artifact["prepared"]
    calls = list(re.finditer(r"(?m)^\s+exec.call[^\n]+", text))[:4]
    assert len(calls) == 4
    body = "\n".join(call.group().rstrip(";") + (";" if index < 3 else "") for index, call in enumerate(calls))
    group = (
        'exec.fused () [] [protocol: builtin.string "preserve_each_stage_guard_and_completion"] : <() -> ()> {\n^forged():\n' + body + "\n}"
    )
    return text[: calls[0].start()] + "\n" + group + text[calls[-1].end() :]


@pytest.mark.parametrize("framework", ["xdsl", "pliron"])
@pytest.mark.parametrize("opaque", [False, True])
@pytest.mark.parametrize("horizon", [1, 2])
def test_matched_execution_and_resource_decisions(native, framework, opaque, horizon):
    for fuse in (False, True):
        ready, artifact, _, effects = variant(native, framework, opaque=opaque, fuse=fuse, horizon=horizon)
        report = asdict(asyncio.run(execute(ready, diagnostic_limit=0)).report)
        check_extension(report)
        expected = (
            (("data.double",), ("data.offset",), ("pair.pack", "pair.first"))
            if opaque
            else (("data.double", "data.offset", "pair.pack", "pair.first"),)
        )
        groups = expected if fuse else tuple((name,) for name in ("data.double", "data.offset", "pair.pack", "pair.first"))
        assert ready.local.source_groups == groups
        assert report["local_dispatches"] == len(report["tickets"]) * len(groups)
        if opaque:
            assert effects == [ticket["calls"][1]["input"] for ticket in report["tickets"]]
        if framework == "xdsl":
            evidence = ready.local_analysis
            assert evidence is not None and evidence.summary.reads == (pair.USE,)
            assert evidence.summary.unknown_sources == (("data.offset",) if opaque else ())
            assert [(len(row.groups), row.dispatch_computations, row.resource_computations) for row in evidence.observations] == [
                (4, 1, 1),
                (len(groups), 2 if fuse else 1, 1),
            ]
        else:
            assert artifact["resource_analysis"]["reads"] == [asdict(native_pair.USE)]
            assert artifact["resource_analysis"]["unknown_sources"] == (["data.offset"] if opaque else [])
            assert artifact["resource_analysis"]["computations"] == 1
            assert artifact["analysis"] == [[4, 1, 1], [len(groups), 2 if fuse else 1, 1]]


@pytest.mark.parametrize("framework", ["xdsl", "pliron"])
def test_valid_opaque_run_then_forged_analysis_and_group_cannot_authorize_lowering(native, framework):
    ready, artifact, selected, _ = variant(native, framework, opaque=True)
    check_extension(asdict(asyncio.run(execute(ready)).report))
    lowered = []
    bindings = dict(selected.bindings)
    original = bindings["pack_pair_v1"]
    bindings["pack_pair_v1"] = replace(original, lower=lambda resources: lowered.append(resources) or original.lower(resources))
    selected = replace(selected, bindings=bindings)
    artifact["prepared"] = forge_group(framework, artifact, selected)
    artifact["analysis"] = {"groupable": True, "unknown_sources": []}
    error = preparation.NotReady if framework == "xdsl" else native_tests.Rejected
    with pytest.raises(error, match=r"effect/schema/resource|ineligible fused stage"):
        cold(native, framework, artifact, selected)
    assert not lowered


def source_map(accepted, selected):
    module = parse(accepted.text, extensions=selected.dialects)
    compose = next(op for op in module.walk() if isinstance(op, ComposeOp))
    return {preparation.origin(op): op for op in preparation.local_maps(compose)}


@pytest.mark.parametrize("change", ["effect", "resource_version", "resource_digest"])
def test_project_cache_reuses_grouping_and_recomputes_changed_dependencies(change):
    selected = pair.extension_registry()
    accepted = preparation.accept(pair.build_extension())
    ready = preparation.prepare(accepted, selected)
    module = parse(ready.text, extensions=selected.dialects)
    sources = source_map(accepted, selected)
    analyses = LocalAnalyses()
    analyses.observe(module, sources)
    before = analyses.evidence().summary
    transformed = rewrite(module, sources, analyses=analyses)
    analyses.observe(transformed, sources)
    assert analyses.evidence().summary is before
    assert (analyses.resource_computations, analyses.dispatch_computations) == (1, 2)
    # New accepted documents and target contracts, rather than editing an
    # observation dictionary, supply the changed dependency snapshot.
    authored = pair.build_extension(opaque=change == "effect")
    selected = pair.extension_registry(opaque=change == "effect")
    if change != "effect":
        resource = FrozenResource("2", (1, 2)) if change == "resource_version" else FrozenResource("1", (1, 3))
        use = replace(pair.USE, version=resource.version, digest=resource.digest)
        pack = next(op for op in authored.walk() if isinstance(op, pair.PackOp))
        pack.properties["resource_version"] = StringAttr(use.version)
        pack.properties["resource_digest"] = StringAttr(use.digest)
        bindings = dict(selected.bindings)
        bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], resources=(use,))
        selected = replace(selected, resources={use.symbol: resource}, bindings=bindings)
    accepted = preparation.accept(authored)
    ready = preparation.prepare(accepted, selected)
    changed = parse(ready.text, extensions=selected.dialects)
    changed_sources = source_map(accepted, selected)
    analyses.observe(changed, changed_sources)
    summary = analyses.evidence().summary
    assert analyses.resource_computations == 2 and summary is not before
    if change == "effect":
        assert not summary.permits("data.offset") and summary.unknown_sources == ("data.offset",)
    else:
        assert summary.reads != before.reads and summary.permits("pair.pack")
    rewritten = rewrite(changed, changed_sources, analyses=analyses)
    analyses.observe(rewritten, changed_sources)
    assert analyses.resource_computations == 2
    assert len(dispatch_groups(rewritten)) == (3 if change == "effect" else 1)
    assert analyses.evidence().summary is summary


@pytest.mark.parametrize("framework", ["xdsl", "pliron"])
@pytest.mark.parametrize("opaque", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_fresh_aligned_cold_load_without_authoring(native, tmp_path, framework, opaque, optimized):
    _, artifact, _, _ = variant(native, framework, opaque=opaque, fuse=True)
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps(artifact))
    script = r"""
import asyncio,importlib,importlib.abc,json,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,sys.argv[1]);before=sys._is_gil_enabled()
p='docs_design.models-strategy-trainer';effects=[]
def forbidden(*a,**k):raise RuntimeError('author or other frontend IR machinery entered cold path')
class NoOtherFrontend(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if sys.argv[3]=='xdsl' and '.pliron_candidate' in fullname:raise RuntimeError('Pliron entered xDSL cold path')
sys.meta_path.insert(0,NoOtherFrontend())
fixture=importlib.import_module(p+'.xdsl_candidate.fixture');fixture.build=forbidden
api=importlib.import_module(p+'.xdsl_candidate.prepare')
data=json.loads(Path(sys.argv[2]).read_text());opaque=sys.argv[4]=='1'
if sys.argv[3]=='xdsl':
 pair=importlib.import_module(p+'.xdsl_candidate.pair_extension');pair.build_extension=forbidden
 selected=pair.extension_registry(opaque=opaque,effects=effects)
 accepted=api.accept(importlib.import_module(p+'.xdsl_candidate.dialects').parse(data['accepted'],extensions=selected.dialects))
 ready=api.prepare(accepted,selected,fuse=True,saved_text=data['prepared'],saved_manifest=data['manifest'])
else:
 pair=importlib.import_module(p+'.pliron_candidate.pair_extension');pair.build=forbidden
 importlib.import_module(p+'.pliron_candidate.author').build=forbidden
 bridge=importlib.import_module(p+'.pliron_candidate.bridge');bridge.Native.accept=forbidden;bridge.Native.prepare=forbidden
 api.parse=forbidden;api.prepare=forbidden;api.accept=forbidden
 importlib.import_module(p+'.xdsl_candidate.fusion').rewrite=forbidden
 ready,_=bridge.Native(Path(sys.argv[5])).load(data['accepted'],data['prepared'],data['manifest'],pair.registry(opaque=opaque,effects=effects))
report=asdict(asyncio.run(fixture.execute(ready,diagnostic_limit=0)).report)
print(json.dumps({'report':report,'groups':ready.local.source_groups,'effects':effects,
 'gil_before':before,'gil_after':sys._is_gil_enabled(),'optimized':not __debug__}))
"""
    result = subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            "-c",
            script,
            str(Path(__file__).resolve().parents[3]),
            str(saved),
            framework,
            str(int(opaque)),
            str(native.binary),
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["optimized"] is optimized and not observed["gil_before"] and not observed["gil_after"]
    assert len(observed["groups"]) == (3 if opaque else 1)
    check_extension(observed["report"])
    if opaque:
        assert observed["effects"] == [ticket["calls"][1]["input"] for ticket in observed["report"]["tickets"]]
