"""G3.7's numerical regression gates for the IR runtime (design G3.7; tasks 14.2 and 18.5).

Run as ``uv run python -m tests.unit.iro.runtime.cost_gate --out <file>``.
It is not a unit test: G3.7 keeps noisy wall-clock thresholds out of ordinary
tests. It runs, in alternating order, three separate processes of this
runtime's cost probe (:mod:`tests.unit.iro.runtime.cost_probe`) and three of
the retained reference experiment's probe
(``tests/unit/training/execution_cost_probe.py``) in the same environment,
and applies the gates the design states:

- **Cheap action** (1, 4 and 16 operations): the runtime's median excess over
  direct Python may exceed the reference's median excess by at most the
  greater of 1 microsecond per action or 20% of the reference excess.
- **Dormant work** (0, 256 and 1,024 dormant handlers): the cost per round
  may not grow with the arrangement; an increase over the zero-dormant
  median beyond the greater of 10% of it or three times its within-run
  median absolute deviation fails.

The exit status is 0 when every gate passes. The record keeps every
process's report, so the numbers can be checked again.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OURS = "tests.unit.iro.runtime.cost_probe"
REFERENCE = "tests.unit.training.execution_cost_probe"
CHEAP = (1, 4, 16)
DORMANT = (256, 1024)


def run(module: str, *arguments: str) -> dict:
    done = subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        raise RuntimeError(f"{module} failed:\n{done.stderr}")
    return json.loads(done.stdout[done.stdout.index("{") :])


def ours(scratch: Path, index: int) -> dict:
    out = scratch / f"runtime_{index}.json"
    run(OURS, "--repeats", "7", "--out", str(out))
    return json.loads(out.read_text())


def reference() -> dict:
    return run(REFERENCE, "--iterations", "10000", "--repeats", "7")


def mad(values: list[float]) -> float:
    middle = statistics.median(values)
    return statistics.median(abs(v - middle) for v in values)


def gates(runtime: list[dict], references: list[dict]) -> dict:
    """The gate results from three processes of each probe."""
    results: dict[str, dict] = {}
    for count in CHEAP:
        excess = statistics.median(
            r["median_us_per_iteration"][f"cheap_{count}"]["runtime_us"] - r["median_us_per_iteration"][f"cheap_{count}"]["direct_us"]
            for r in runtime
        )
        base = statistics.median(r["paired_candidate_minus_direct_us"][f"{count}_cheap"]["median"] for r in references)
        allowed = base + max(1.0, 0.2 * base)
        results[f"cheap_{count}"] = {
            "runtime_excess_us": round(excess, 2),
            "reference_excess_us": round(base, 2),
            "allowed_us": round(allowed, 2),
            "passes": excess <= allowed,
        }
    zero = statistics.median(r["median_us_per_iteration"]["dormant_0"]["per_round_us"] for r in runtime)
    spread = statistics.median(mad(r["samples_us"]["dormant_0"]["per_round_us"]) for r in runtime)
    allowed_increase = max(0.1 * zero, 3 * spread)
    for count in DORMANT:
        per_round = statistics.median(r["median_us_per_iteration"][f"dormant_{count}"]["per_round_us"] for r in runtime)
        results[f"dormant_{count}"] = {
            "per_round_us": round(per_round, 2),
            "zero_dormant_us": round(zero, 2),
            "allowed_increase_us": round(allowed_increase, 2),
            "passes": per_round - zero <= allowed_increase,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()
    out = Path(options.out)
    scratch = out.parent
    runtime, references = [], []
    for index in range(options.runs):
        # Alternate which probe goes first, so neither always runs on a warmer machine.
        if index % 2 == 0:
            runtime.append(ours(scratch, index))
            references.append(reference())
        else:
            references.append(reference())
            runtime.append(ours(scratch, index))
    results = gates(runtime, references)
    record = {
        "python": sys.version,
        "gil_enabled": [r["gil_enabled"] for r in runtime],
        "platform": platform.platform(),
        "runs": options.runs,
        "gates": results,
        "runtime_reports": runtime,
        "reference_reports": references,
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    width = max(map(len, results))
    for name, result in results.items():
        shown = ", ".join(f"{k} {v}" for k, v in result.items() if k != "passes")
        print(f"{name:{width}}  {'pass' if result['passes'] else 'FAIL'}  {shown}")
    sys.exit(0 if all(r["passes"] for r in results.values()) else 1)


if __name__ == "__main__":
    main()
