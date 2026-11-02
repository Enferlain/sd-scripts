"""Guard measurement coverage, retained failures, actual receipts and opt-in CLI."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from .measure import ROOT, summary

PREFIX = "docs_design.models-strategy-trainer"


def test_complete_frozen_protocol_and_import_state():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    data = json.loads((ROOT / "measurements.json").read_text())
    assert data["protocol_sha256"] == hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    assert not data["failures"] and not data["sample_failures"]
    assert len(data["warm"]) == 6 and len(data["fresh"]) == 120
    for entry in data["warm"]:
        case = entry["case"]
        result = entry["result"]
        meta = result["metadata"]
        assert entry["exit_code"] == 0 and meta["gil"] is False and meta["cuda_initialized"] is False and not meta["triton_native_imported"]
        assert meta["torch_imported"] == (case not in ("B", "extension"))
        assert Counter((row["framework"], row["fuse"], row["phase"]) for row in result["samples"]) == Counter(
            {
                (framework, fuse, phase): 20
                for framework in ("xdsl", "pliron")
                for fuse in protocol["rewrite"][case]
                for phase in protocol["phases"]
            }
        )
        assert len(result["warmups"]) == 3 * 2 * len(protocol["rewrite"][case]) * len(protocol["phases"])
        for row in result["samples"]:
            assert row["status"] == "passed" and row["elapsed_ns"] > 0
            expected_first = "xdsl" if (row["index"] + protocol["warmups"]) % 2 == 0 else "pliron"
            assert row["order"] == int(row["framework"] != expected_first)
        fresh = [row for row in data["fresh"] if row["case"] == case]
        assert Counter(row["framework"] for row in fresh) == {"xdsl": 10, "pliron": 10}
        for row in fresh:
            assert row["status"] == "passed" and row["result"]["metadata"]["gil"] is False
            assert row["result"]["metadata"]["peak_parent_rss_kib"] > 0
    assert sum(len(entry["result"]["samples"]) for entry in data["warm"]) == 2800
    assert summary([row for entry in data["warm"] for row in entry["result"]["samples"]] + data["fresh"]) == data["summary"]


def test_measured_sources_remain_available_after_style_cleanup():
    data = json.loads((ROOT / "measurements.json").read_text())
    for name, digest in data["harness_sha256"].items():
        assert hashlib.sha256((ROOT / "measured_sources" / (name + ".source")).read_bytes()).hexdigest() == digest


def test_summary_retains_failure_counts():
    rows = [
        {"case": "B", "framework": "xdsl", "fuse": False, "phase": "accept", "status": status, "elapsed_ns": value}
        for status, value in [("passed", 10), ("failed", 999), ("passed", 30)]
    ]
    group = summary(rows)["B|xdsl|False|accept"]
    assert (group["attempted"], group["passed"], group["failed"]) == (3, 2, 1)


def test_every_saved_report_passes_its_independent_oracle():
    names = {
        "B": ("xdsl_candidate.conformance", "check_b"),
        "extension": ("xdsl_candidate.extension_conformance", "check_extension"),
        "A": ("xdsl_candidate.model_conformance", "check_a"),
        "C": ("xdsl_candidate.grant_conformance", "check_c"),
        "mixed": ("xdsl_candidate.mixed_conformance", "check_mixed"),
        "lifetime": ("pliron_candidate.composition_conformance", "check"),
    }
    count = 0
    for case, (name, fn) in names.items():
        check = getattr(importlib.import_module(PREFIX + "." + name), fn)
        for path in (ROOT / "cases" / case).glob("*/report.json"):
            check(json.loads(path.read_text()))
            count += 1
    assert count == 20


def test_fixed_diagnostics_are_actual_rejections():
    rows = json.loads((ROOT / "diagnostics.json").read_text())
    assert len(rows) == 16 and {row["framework"] for row in rows} == {"xdsl", "pliron"}
    assert all(row["status"] == "rejected" and row["type"] in ("Rejected", "NotReady") and row["message"] for row in rows)
    initial = json.loads((ROOT / "diagnostics_probe_initial.json").read_text())
    assert any(row.get("type") == "AttributeError" for row in initial)


@pytest.fixture(scope="session")
def binaries():
    target = Path(os.environ.get("CARGO_TARGET_DIR", "/tmp/sd-scripts-pliron-b-target"))
    manifest = ROOT.parent / "pliron_candidate/native/Cargo.toml"
    result = subprocess.run(
        ["cargo", "build", "--release", "--locked", "--manifest-path", str(manifest), "--bins"],
        env=os.environ | {"CARGO_TARGET_DIR": str(target)},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return target / "release"


@pytest.mark.parametrize(
    "binary,author",
    [
        ("rmst-pliron-b", "author"),
        ("rmst-pliron-composition", "composition_author"),
        ("rmst-pliron-correlation", "correlation_author"),
        ("rmst-pliron-grant", "grant_author"),
        ("rmst-pliron-mixed", "mixed_author"),
    ],
)
def test_opt_in_spans_preserve_normal_result(binaries, binary, author):
    build = importlib.import_module(PREFIX + ".pliron_candidate." + author).build
    request = json.dumps({"action": "accept", "text": build()})

    def invoke(options):
        return subprocess.run([str(binaries / binary), *options], input=request, capture_output=True, text=True, timeout=30, check=False)

    normal, measured = invoke([]), invoke(["--measure"])
    assert normal.returncode == measured.returncode == 0
    value = json.loads(measured.stdout)
    assert value["outcome"]["Ok"] == json.loads(normal.stdout)
    assert "timing" not in json.loads(normal.stdout)
    assert value["timing"]["native_request_body_ns"] > 0 and value["timing"]["native_serialize_ns"] > 0
    assert value["timing"]["native_peak_rss_kib"] > 0


def test_failed_measured_request_still_fails(binaries):
    result = subprocess.run(
        [str(binaries / "rmst-pliron-b"), "--measure"], input="invalid JSON", capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 2 and not result.stdout
    error = json.loads(result.stderr)
    assert error["error"] and error["timing"]["native_request_body_ns"] > 0


def test_natural_endpoint_coverage_sources_and_native_boundaries():
    protocol = json.loads((ROOT / "endpoint_protocol.json").read_text())
    data = json.loads((ROOT / "natural_measurements.json").read_text())
    assert data["protocol_sha256"] == hashlib.sha256((ROOT / "endpoint_protocol.json").read_bytes()).hexdigest()
    assert not data["failures"] and len(data["warm"]) == 5 and len(data["fresh"]) == 100
    assert sum(len(entry["result"]["rows"]) for entry in data["warm"]) == 360
    for name, digest in data["source_sha256"].items():
        assert hashlib.sha256((ROOT / "natural_sources" / (name + ".source")).read_bytes()).hexdigest() == digest
    for entry in data["warm"]:
        assert entry["exit_code"] == 0 and entry["result"]["metadata"]["gil"] is False
        for row in entry["result"]["rows"]:
            assert row["status"] == "passed"
            expected = (
                []
                if row["framework"] == "xdsl"
                else ["accept", "prepare"]
                if row["case"] in ("B", "extension")
                else ["accept", "prepare", "load"]
            )
            assert [event["action"] for event in row["native"]] == expected
    assert Counter((row["case"], row["framework"]) for row in data["fresh"]) == Counter(
        {(case, framework): 10 for case in protocol["cases"] for framework in ("xdsl", "pliron")}
    )


def test_subprocess_timeout_preserves_partial_output():
    import sys
    from .measure import run_process

    result = run_process(
        [sys.executable, "-u", "-c", "import time; print('partial',flush=True); time.sleep(5)"], env=os.environ, timeout=0.3
    )
    assert result.returncode == 124 and result.timed_out and "partial" in result.stdout
    assert "TimeoutExpired" in result.stderr and result.exception


@pytest.mark.parametrize("collector", ["measure", "natural"])
@pytest.mark.parametrize("failure", ["timeout", "launch"])
def test_collectors_persist_failed_attempts(tmp_path, monkeypatch, collector, failure):
    import sys
    from types import SimpleNamespace
    from . import measure, natural

    api = measure if collector == "measure" else natural
    fixture = tmp_path / "config"
    fixture.mkdir()
    output = tmp_path / "output"
    base = {
        "cases": ["B"],
        "warmups": 0,
        "measured_pairs": 1,
        "fresh_starts_per_candidate_case": 1,
        "rewrite": {"B": [False]},
        "phases": ["author"],
    }
    for name in ("protocol.json", "endpoint_protocol.json"):
        (fixture / name).write_text(json.dumps(base))
    for name in ("measure.py", "cases.py", "natural.py"):
        (fixture / name).write_text("test source")
    monkeypatch.setattr(api, "ROOT", fixture)
    monkeypatch.setattr(api, "provenance", lambda binaries: {"test_fixture": True})
    code = 124 if failure == "timeout" else 127
    observation = measure.ProcessObservation(
        code, "partial output", "retained failure", "TimeoutExpired" if failure == "timeout" else "FileNotFoundError", failure == "timeout"
    )
    monkeypatch.setattr(api, "run_process", lambda *args, **kwargs: observation)
    if collector == "measure":
        args = SimpleNamespace(out=output, binaries=tmp_path, torch_free_python=sys.executable, owner_python=sys.executable)
        with pytest.raises(RuntimeError, match="failures"):
            measure.driver(args, base)
        name = "measurements.json"
    else:
        monkeypatch.setattr(sys, "argv", ["natural", "--binaries", str(tmp_path), "--out", str(output)])
        with pytest.raises(RuntimeError, match="failures"):
            api.main()
        name = "natural_measurements.json"
    data = json.loads((output / name).read_text())
    assert len(data["warm"]) == 1 and len(data["fresh"]) == 2 and len(data["failures"]) == 3
    assert all(entry["exception"] == observation.exception and entry["timed_out"] == observation.timed_out for entry in data["failures"])
    assert all(entry.get("stdout") == "partial output" for entry in data["failures"])


@pytest.mark.parametrize("existing", ["diagnostics.json", "diagnostics_provenance.json"])
def test_diagnostics_refuses_overwrite_before_capture(tmp_path, monkeypatch, existing):
    import sys
    from . import diagnostics

    path = tmp_path / existing
    path.write_bytes(b"preserved")
    monkeypatch.setattr(sys, "argv", ["diagnostics", "--binaries", str(tmp_path), "--out", str(tmp_path)])
    monkeypatch.setattr(diagnostics, "capture", lambda *args: pytest.fail("capture ran before overwrite guard"))
    with pytest.raises(RuntimeError, match="overwrite"):
        diagnostics.main()
    assert path.read_bytes() == b"preserved"


def test_natural_worker_retains_rows_on_gil_regression(monkeypatch):
    import sys
    from types import SimpleNamespace
    from . import natural

    class FakeCase:
        def __init__(self, *args, **kwargs):
            pass

        def natural_invocation(self, fuse):
            return {}

        def check(self, report):
            pass

    fake = SimpleNamespace(Case=FakeCase, NATIVE_EVENTS=[])
    monkeypatch.setattr(natural.importlib, "import_module", lambda *args: fake)
    monkeypatch.setattr(natural, "threads", lambda: None)
    monkeypatch.setattr(natural, "metadata", lambda: {"gil": True})
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True)
    args = SimpleNamespace(fresh=False, case="B", binaries=ROOT, framework=None)
    result = natural.worker(args, {"warmups": 1, "measured_pairs": 2}, {"rewrite": {"B": [False]}})
    assert len(result["rows"]) == 4 and len(result["warmups"]) == 2
    assert all(row["status"] == "failed" and row["error"] == "GIL enabled" for row in (*result["rows"], *result["warmups"]))


def test_phase_collector_rejects_warmup_only_failure(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from . import measure

    base = {
        "cases": ["B"],
        "warmups": 1,
        "measured_pairs": 1,
        "fresh_starts_per_candidate_case": 0,
        "rewrite": {"B": [False]},
        "phases": ["author"],
    }
    (tmp_path / "protocol.json").write_text(json.dumps(base))
    row = {"case": "B", "framework": "xdsl", "fuse": False, "phase": "author", "status": "passed", "elapsed_ns": 1}
    payload = {"samples": [row], "warmups": [dict(row, status="failed")], "artifacts": {}}
    monkeypatch.setattr(measure, "ROOT", tmp_path)
    monkeypatch.setattr(measure, "provenance", lambda *args: {})
    monkeypatch.setattr(measure, "run_process", lambda *args, **kwargs: measure.ProcessObservation(0, json.dumps(payload), ""))
    args = SimpleNamespace(out=tmp_path / "output", binaries=tmp_path, torch_free_python=sys.executable, owner_python=sys.executable)
    with pytest.raises(RuntimeError, match="failures"):
        measure.driver(args, base)
    result = json.loads((args.out / "measurements.json").read_text())
    assert not result["sample_failures"] and len(result["warmup_failures"]) == 1
