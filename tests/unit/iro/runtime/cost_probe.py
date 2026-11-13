"""G3.7 cost cases for the IR runtime, measured against direct Python (14.2).

Run as ``uv run python -m tests.unit.iro.runtime.cost_probe --repeats 7 --out <file>``
three times, as separate processes. It is not a unit test: G3.7 keeps noisy
wall-clock thresholds out of ordinary tests.

- **Cheap action** with 1, 4 and 16 selected operations: one repeat iteration
  calling a fused chain of cheap host implementations, against a direct
  Python loop calling the same functions.
- **Dormant work**: the same 4-operation action beside 0, 256 and 1,024
  handlers that wait for requests that never come. Startup may scale with
  the strategy and dispatch may not (G3.7), so besides the cost per round
  of the whole run, each repeat also runs three times as many rounds: the
  difference gives the cost per round alone, and what is left the fixed
  startup and teardown.
- **Admission and lookup**: one access per iteration over 1, 4 and 16
  resources, timing the lease acquisition and release the runtime does.
"""

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time

from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.frontend import implementation
from library.iro.prepare import Target, prepare
from library.iro.runtime import Coordinator
from tests.unit.iro.accepted_runs import accepted_run
from tests.unit.iro.runtime.harness import Memory

ITERATIONS = 2000


def _op(name: str):
    return implementation(version=1, inputs=("int",), outputs=("int",), name=name, concurrency_safe=True)(lambda x: x + 1)


def cheap(operations: int, dormant: int = 0, rounds: int = ITERATIONS) -> str:
    names = [f"op{i}" for i in range(operations)]
    calls = "\n".join(f"      %v{i + 1} = iro.call @{name}(%v{i}) -> int" for i, name in enumerate(names))
    handlers = "".join(
        f"    iro.handler @idle{d} sources [@loop] max 1 {{\n    ^(%x{d}: int):\n      %y{d} = iro.call @op0(%x{d}) -> int\n    }}\n"
        for d in range(dormant)
    )
    uses = ", ".join(f"@{n} v1" for n in names)
    return (
        f"iro.strategy @cost uses({uses}) {{\n"
        "  iro.parallel @all {\n"
        f"    iro.repeat @loop {rounds} times {{\n"
        "    ^(%v0: int):\n"
        f"{calls}\n"
        "    }\n"
        f"{handlers}"
        "  }\n"
        "}\n"
    )


def leases(resources: int) -> str:
    names = [f"r{i}" for i in range(resources)]
    declared = "\n".join(f"  %{n} = iro.resource @{n} : participant" for n in names)
    uses = ", ".join(f"%{n} protected_current" for n in names)
    arguments = ", ".join(f"%l{i}" for i in range(resources))
    return (
        "iro.strategy @cost uses(@touch v1) {\n"
        f"{declared}\n"
        f"  iro.repeat @loop {ITERATIONS} times capture({', '.join('%' + n for n in names)}) {{\n"
        "  ^(%i: int):\n"
        f"    iro.access {uses} {{\n"
        f"    ^({arguments}):\n"
        "      %x = iro.call @touch(%i) -> int\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def runtime_seconds(text: str, installed: dict, resources: dict | None = None) -> float:
    run = accepted_run(text, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, Target(workers=2))
    owner = Memory(resources or {})
    coordinator = Coordinator(prepared, dict.fromkeys(resources or {}, owner), installed, {"host": Host()})
    started = time.perf_counter()
    outcome = asyncio.run(coordinator.run())
    elapsed = time.perf_counter() - started
    assert outcome.status == "succeeded", outcome.error
    return elapsed


def direct_seconds(operations: int) -> float:
    functions = [lambda x: x + 1 for _ in range(operations)]
    started = time.perf_counter()
    for i in range(ITERATIONS):
        value = i
        for function in functions:
            value = function(value)
    return time.perf_counter() - started


def measure(repeats: int) -> dict:
    results: dict[str, dict] = {}
    for operations in (1, 4, 16):
        installed = {f"op{i}": _op(f"op{i}") for i in range(operations)}
        text = cheap(operations)
        samples, direct = [], []
        for repeat in range(repeats):
            # Alternate which goes first, so neither always runs on a warmer process.
            if repeat % 2 == 0:
                run = runtime_seconds(text, installed)
                ref = direct_seconds(operations)
            else:
                ref = direct_seconds(operations)
                run = runtime_seconds(text, installed)
            samples.append(run / ITERATIONS * 1e6)
            direct.append(ref / ITERATIONS * 1e6)
        results[f"cheap_{operations}"] = {"runtime_us": samples, "direct_us": direct}
    installed = {f"op{i}": _op(f"op{i}") for i in range(4)}
    for dormant in (0, 256, 1024):
        text, longer = cheap(4, dormant), cheap(4, dormant, 3 * ITERATIONS)
        whole, marginal, fixed = [], [], []
        for _ in range(repeats):
            short, long = runtime_seconds(text, installed), runtime_seconds(longer, installed)
            per_round = (long - short) / (2 * ITERATIONS)
            whole.append(short / ITERATIONS * 1e6)
            marginal.append(per_round * 1e6)
            fixed.append((short - per_round * ITERATIONS) * 1e3)
        results[f"dormant_{dormant}"] = {"runtime_us": whole, "per_round_us": marginal, "fixed_ms": fixed}
    touch = {"touch": _op("touch")}
    for count in (1, 4, 16):
        resources = {f"r{i}": 0 for i in range(count)}
        results[f"leases_{count}"] = {
            "runtime_us": [runtime_seconds(leases(count), touch, resources) / ITERATIONS * 1e6 for _ in range(repeats)]
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out")
    options = parser.parse_args()
    results = measure(options.repeats)
    summary = {key: {k: round(statistics.median(v), 2) for k, v in value.items()} for key, value in results.items()}
    report = {
        "python": sys.version,
        "gil_enabled": sys._is_gil_enabled(),
        "platform": platform.platform(),
        "iterations": ITERATIONS,
        "repeats": options.repeats,
        "median_us_per_iteration": summary,
        "samples_us": results,
    }
    text = json.dumps(report, indent=1)
    if options.out:
        with open(options.out, "w") as file:
            file.write(text + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
