"""Retain the fixed invalid-program sample at the actual frontend boundaries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import traceback
from dataclasses import replace
from pathlib import Path

from .cases import Case, module
from .measure import ROOT, provenance


SAMPLES = [
    ("missing_binding", "B", "prepare"),
    ("incompatible_payload", "extension", "accept"),
    ("resource_identity", "extension", "prepare"),
    ("resource_access", "extension", "prepare"),
    ("unsupported_target", "A", "prepare"),
    ("forged_correspondence", "A", "cold"),
    ("invalid_authority", "C", "accept"),
    ("invalid_lifetime", "lifetime", "accept"),
]


def example(label, case, phase):
    framework = case.framework
    if label == "missing_binding":
        bindings = dict(case.selected.bindings)
        del bindings["double_v1"]
        case.selected = replace(case.selected, bindings=bindings)
        return lambda: case.prepare()
    if label == "incompatible_payload":
        authored = case.build()
        if framework == "xdsl":
            pack = next(op for op in authored.walk() if op.name == "pairing.pack")
            pack.output._type = module("xdsl_candidate.dialects").FlowType(module("xdsl_candidate.prepare").StringAttr("record"))
        else:
            authored = authored.replace("-> (pairing.pair)>", "-> (wf.record)>")
        return lambda: case.accept(authored)
    if label in ("resource_identity", "resource_access"):
        if label == "resource_identity":
            resources = dict(case.selected.resources)
            original = resources["pair_coefficients"]
            resources["pair_coefficients"] = replace(original, version="other")
            case.selected = replace(case.selected, resources=resources)
        else:
            bindings = dict(case.selected.bindings)
            binding = bindings["pack_pair_v1"]
            bindings["pack_pair_v1"] = replace(binding, resources=tuple(replace(use, access="write") for use in binding.resources))
            case.selected = replace(case.selected, bindings=bindings)
        return lambda: case.prepare()
    if label == "unsupported_target":
        case.accepted = case.accept(case.build(count=1))
        return lambda: case.prepare()
    if label == "forged_correspondence":
        saved = copy.deepcopy(case.prepare())
        needle = "protected_physical_return_then_validate"
        assert needle in saved["prepared"]
        saved["prepared"] = saved["prepared"].replace(needle, "release_before_physical_return")
        return lambda: case.cold(saved)
    if label == "invalid_authority":
        authored = case.build()
        if framework == "xdsl":
            op = next(op for op in authored.walk() if op.name == "grant.phase" and op.phase.data == "restore")
            op.properties["phase"] = module("xdsl_candidate.prepare").StringAttr("temporary_edit")
        else:
            authored = authored.replace('phase: builtin.string "restore"', 'phase: builtin.string "temporary_edit"')
        return lambda: case.accept(authored)
    if label == "invalid_lifetime":
        authored = case.build()
        if framework == "xdsl":
            op = next(op for op in authored.walk() if getattr(op, "source_id", None) is not None and op.source_id.data == "consumers")
            builtin = __import__("xdsl.dialects.builtin", fromlist=["DictionaryAttr", "StringAttr"])
            op.properties["policy"] = builtin.DictionaryAttr(dict(op.policy.data) | {"pending": builtin.StringAttr("executing_only")})
        else:
            assert "include_requested" in authored
            authored = authored.replace("include_requested", "executing_only")
        return lambda: case.accept(authored)
    raise ValueError(label)


def capture(binaries):
    result = []
    for label, name, phase in SAMPLES:
        for framework in ("xdsl", "pliron"):
            case = Case(name, framework, binaries)
            row = {"sample": label, "case": name, "framework": framework, "boundary": phase, "status": "unexpected_acceptance"}
            try:
                invoke = example(label, case, phase)
            except BaseException as error:
                row.update(status="probe_error", type=type(error).__name__, message=str(error), trace=traceback.format_exc())
                result.append(row)
                continue
            try:
                invoke()
            except BaseException as error:
                row.update(status="rejected", type=type(error).__name__, message=str(error), trace=traceback.format_exc())
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or ROOT
    output.mkdir(parents=True, exist_ok=True)
    path = output / "diagnostics.json"
    sidecar = output / "diagnostics_provenance.json"
    if path.exists() or sidecar.exists():
        raise RuntimeError("refusing to overwrite retained diagnostic evidence")
    context = {
        **provenance(args.binaries),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("diagnostics.py", "cases.py", "measure.py")
        },
    }
    rows = capture(args.binaries)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    sidecar.write_text(json.dumps(context, indent=2) + "\n")
    for row in rows:
        print(row["sample"], row["framework"], row["status"], row.get("message", "")[:180])


if __name__ == "__main__":
    main()
