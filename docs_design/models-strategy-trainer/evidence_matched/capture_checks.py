"""Retain one declared pytest gate, its failures and ordinary-import GIL state."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pytest


class Outcomes:
    def __init__(self) -> None:
        self.results: dict[str, str] = {}

    def pytest_collectreport(self, report) -> None:
        if report.skipped or report.failed:
            self.results[report.nodeid] = report.outcome

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" or report.failed or report.skipped:
            self.results[report.nodeid] = report.outcome


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: capture_checks.py OUTPUT.json PYTEST_ARGS...")
    output, arguments = Path(sys.argv[1]), sys.argv[2:]
    before, started = sys._is_gil_enabled(), time.perf_counter()
    outcomes, stream = Outcomes(), io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        exit_code = int(pytest.main(arguments, plugins=[outcomes]))
    after = sys._is_gil_enabled()
    torch = sys.modules.get("torch")
    result = {
        "exit_code": exit_code,
        "counts": dict(Counter(outcomes.results.values())),
        "gil_before": before,
        "gil_after": after,
        "python": sys.version,
        "xdsl": version("xdsl"),
        "args": arguments,
        "seconds": time.perf_counter() - started,
        "duration_scope": "quality gate elapsed time; not matched framework performance evidence",
        "torch_imported": torch is not None,
        "output": stream.getvalue(),
    }
    if torch is not None:
        result.update(torch=torch.__version__, cuda_initialized=torch.cuda.is_initialized())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("exit_code", "counts", "gil_before", "gil_after", "seconds")}))
    if exit_code:
        print(stream.getvalue())
    raise SystemExit(exit_code or int(before or after))


if __name__ == "__main__":
    main()
