"""CPU-only cache ownership/cost evidence, independent of candidate worker policy.

No native patch, compiler, optimizer object or dispatch instrumentation is used.
The explicit-cast case is a functional Linear realization, not an arbitrary
module adapter: it never swaps parameters on a shared module. All casts and
autograd graphs are new per region, before any possible weight update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import torch

type Policy = Literal["cached", "uncached", "explicit"]
type Source = Literal["fp32_trainable", "fp32_frozen", "bf16_trainable"]
POLICIES: tuple[Policy, ...] = ("cached", "uncached", "explicit")


def forward(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    reuses: int,
    policy: Policy,
    dtype: torch.dtype = torch.bfloat16,
    before_last: Callable[[], None] | None = None,
) -> tuple[torch.Tensor, ...]:
    """One unchanged-weight region; explicit reuse lasts only this invocation."""
    values = []
    with torch.autocast("cpu", dtype=dtype, enabled=policy != "explicit", cache_enabled=policy == "cached"):
        selected = weight.to(dtype) if policy == "explicit" else weight
        for index in range(reuses):
            if index == reuses - 1 and before_last is not None:
                before_last()
            values.append(torch.nn.functional.linear(inputs, selected))
    return tuple(values)


def cast_nodes(values: tuple[torch.Tensor, ...]) -> int:
    """Count distinct differentiable casts without installing a dispatch mode.

    Fixtures differentiate only one source: weight for trainable-weight cases,
    input for frozen-weight cases. Consequently this is a weight-cast count
    only when the fixture's weight requires grad. It is not an allocation count.
    """
    nodes = set()
    pending = [value.grad_fn for value in values]
    while pending:
        node = pending.pop()
        if node is None or node in nodes:
            continue
        nodes.add(node)
        pending.extend(child for child, _ in node.next_functions)
    return sum(type(node).__name__.startswith("ToCopyBackward") for node in nodes)


def gradient(values: tuple[torch.Tensor, ...], target: torch.Tensor) -> torch.Tensor:
    # Match ordinary AMP usage: backward is outside the forward autocast scope.
    return torch.autograd.grad(tuple(value.float().sum() for value in values), target)[0]


def overlap(first: Policy, second: Policy, dtype: torch.dtype) -> dict:
    """Hold the first region open while the other enters AND exits its scope."""
    weight = torch.nn.Parameter(torch.full((4, 4), 0.5))
    filled, finished = threading.Event(), threading.Event()
    other_dtype = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16

    def wait_for_second():
        filled.set()
        assert finished.wait(10), "second region failed to finish"

    def run_first():
        try:
            return forward(weight, torch.ones(2, 4, dtype=dtype), 3, first, dtype, wait_for_second), threading.get_ident()
        finally:
            filled.set()

    def run_second():
        try:
            assert filled.wait(10), "first region failed to start"
            return forward(weight, torch.ones(2, 4, dtype=other_dtype), 2, second, other_dtype), threading.get_ident()
        finally:
            finished.set()

    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(run_first), pool.submit(run_second))
        threads = []
        for name, future, expected_dtype, reuses in zip(("first", "second"), futures, (dtype, other_dtype), (3, 2), strict=True):
            try:
                values, thread = future.result(timeout=15)
            except RuntimeError as error:
                # Keep the unsafe negative's failure evidence; reject other failures.
                if first != "cached" or second != "cached" or "same dtype" not in str(error):
                    raise
                results[name] = {"error": str(error)}
                continue
            threads.append(thread)
            for value in values:
                assert value.dtype == expected_dtype and torch.equal(value, torch.full_like(value, 2))
            nodes = cast_nodes(values)
            actual = gradient(values, weight)
            assert torch.equal(actual, torch.full_like(weight, 2 * reuses))
            results[name] = {"dtype": str(expected_dtype), "weight_cast_nodes": nodes, "values_and_gradient": "exact"}
        if len(threads) == 2:
            assert threads[0] != threads[1], "the region lifetimes must really overlap"
    return {"policies": [first, second], "first_dtype": str(dtype), "regions": results}


def fixture(source: Source, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.bfloat16 if source == "bf16_trainable" else torch.float32
    trainable = source != "fp32_frozen"
    # Exactly representable nonzero values make all compared results/gradients
    # bit-identical, including low-precision shared vs separate cast gradients.
    weight = torch.full((width, width), 0.03125, dtype=dtype, requires_grad=trainable)
    inputs = torch.full((8, width), 0.125, dtype=torch.bfloat16, requires_grad=not trainable)
    return weight, inputs, weight if trainable else inputs


def cost(source: Source, width: int, reuses: int, samples: int, loops: int) -> dict:
    weight, inputs, target = fixture(source, width)
    evidence, timings = {}, {policy: [] for policy in POLICIES}
    reference = None
    for policy in POLICIES:
        values = forward(weight, inputs, reuses, policy)
        count = cast_nodes(values) if weight.requires_grad else None
        actual = gradient(values, target)
        if reference is None:
            reference = (values, actual)
        else:
            assert all(torch.equal(left, right) for left, right in zip(reference[0], values, strict=True))
            assert torch.equal(reference[1], actual)
        evidence[policy] = {"weight_cast_nodes": count, "values_and_gradient": "exact"}

    def invoke(policy):
        values = forward(weight, inputs, reuses, policy)
        gradient(values, target)

    # Warm all paths; rotate their order rather than always timing cached first.
    for _ in range(3):
        for policy in POLICIES:
            invoke(policy)
    for sample in range(samples):
        order = POLICIES[sample % 3 :] + POLICIES[: sample % 3]
        for policy in order:
            start = time.perf_counter_ns()
            for _ in range(loops):
                invoke(policy)
            timings[policy].append((time.perf_counter_ns() - start) / loops / 1000)
    return {
        "source": source,
        "width": width,
        "batch": 8,
        "input_dtype": "torch.bfloat16",
        "weight_uses_per_region": reuses,
        "correctness": evidence,
        "region_us": timings,
        "median_us": {policy: statistics.median(values) for policy, values in timings.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--loops", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.samples < 1 or arguments.loops < 1:
        parser.error("samples and loops must be positive")
    assert "PYTHON_GIL" not in os.environ, "do not force the GIL off for this qualification"
    assert not sys._is_gil_enabled(), "the ordinary import must preserve free threading"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "platform": platform.platform(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "gil_enabled": sys._is_gil_enabled(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "warmup_regions_per_policy": 3,
        "samples": arguments.samples,
        "regions_per_sample": arguments.loops,
        "overlap": [
            overlap(first, second, dtype)
            for first, second in (("cached", "cached"), ("cached", "uncached"), ("uncached", "cached"), ("explicit", "explicit"))
            for dtype in (torch.bfloat16, torch.float16)
        ],
        "cost": [
            cost(source, width, reuses, arguments.samples, arguments.loops)
            for source, widths in (("fp32_trainable", (512, 1024)), ("fp32_frozen", (512,)), ("bf16_trainable", (512,)))
            for width in widths
            for reuses in (1, 8)
        ],
    }
    assert not sys._is_gil_enabled() and "torch._dynamo" not in sys.modules and "triton" not in sys.modules
    result["environment"]["gil_enabled_at_end"] = sys._is_gil_enabled()
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "median_us": [row["median_us"] for row in result["cost"]]}))


if __name__ == "__main__":
    main()
