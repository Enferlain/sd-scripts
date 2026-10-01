"""G3.7 measurement fixtures; run with ``uv run python -m tests.unit.training.execution_cost_probe``.

The direct reference specializes a unary action already checked at construction.
It retains runtime input checks, exception boundaries, addressed outputs and
observation results. It is a comparison fixture, not a production lowering.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

from library.training.execution import (
    Action,
    ActionResult,
    AdvancementResult,
    AttemptReport,
    DueWork,
    InputActivity,
    InputDelivery,
    InputFeed,
    InputRequest,
    InvalidExecution,
    Operation,
    OperationFailed,
    OptimizationInput,
    OptimizationValue,
    PreparedAction,
    RunCoordinator,
    RunDescription,
    UnitOutcome,
    Value,
    compile_run,
    prepare_workflow,
)


@dataclass(frozen=True)
class DirectAction:
    """Direct Python for the same accepted one-input, unary-operation fixture."""

    input: Value
    source: Value
    operations: tuple[tuple[str, Callable[[object], object]], ...]

    def execute(self, inputs: Mapping[Value, object]) -> ActionResult:
        if set(inputs) != {self.input}:
            raise InvalidExecution("direct fixture input mismatch")
        result = inputs[self.input]
        for name, operation in self.operations:
            try:
                result = operation(result)
            except Exception as error:
                raise OperationFailed(name) from error
        return ActionResult(
            "train",
            (OptimizationValue("main", self.source, result),),
            MappingProxyType({self.source: result}),
        )


def increment(value: object) -> object:
    assert isinstance(value, int)
    return value + 1


def compute(value: object) -> object:
    """Same selected CPU work for both paths; no model/backend integration."""
    assert isinstance(value, int)
    return value + sum(range(1000))


def action_pair(count: int, operation: Callable[[object], object] = increment) -> tuple[PreparedAction, DirectAction]:
    if count < 1:
        raise ValueError("the cost fixture needs at least one operation")
    values = tuple(Value(f"value_{index}") for index in range(count + 1))
    operations = tuple(Operation(f"op_{index}", operation, (values[index],), (values[index + 1],)) for index in range(count))
    action = Action("train", (values[0],), operations, (OptimizationInput("main", values[-1]),), (values[-1],))
    prepared = compile_run((action,), due=lambda _: ("train",)).actions["train"]
    direct = DirectAction(values[0], values[-1], tuple((item.name, operation) for item in operations))
    return prepared, direct


def current_loop_dispatch(direct: DirectAction) -> tuple[Callable[[object], object], str, str]:
    """Execute the actual process_batch call expression, with a lightweight facade.

    This measures its argument dispatch only; it excludes the live Trainer,
    objective, model, optimization, monitoring and phase implementation.
    """
    path = Path(__file__).resolve().parents[3] / "library/training/phases/training_loop.py"
    source = path.read_text()
    tree = ast.parse(source)
    epoch = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_epoch_steps")
    calls = [
        node
        for node in ast.walk(epoch)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "process_batch"
    ]
    if len(calls) != 1:
        raise ValueError("current loop dispatch scope changed; inspect and update the probe")
    expression = calls[0]

    class Facade:
        def process_batch(self, batch: object, *_args: object, **_kwargs: object) -> ActionResult:
            return direct.execute({direct.input: batch})

    # Compile a callable once, avoiding eval/compilation inside the measurement.
    function = ast.FunctionDef(
        name="dispatch",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="batch")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Return(expression)],
        decorator_list=[],
    )
    trainer = SimpleNamespace(
        **dict.fromkeys(
            (
                "text_encoders",
                "denoiser",
                "trainable_model",
                "vae",
                "objective_runtime",
                "vae_dtype",
                "weight_dtype",
                "_train_text_encoder",
                "_train_denoiser",
            )
        ),
        global_step=0,
    )
    namespace: dict[str, object] = {"strategies": Facade(), "trainer": trainer, "accelerator": None, "cfg": None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    dispatch = namespace["dispatch"]
    assert callable(dispatch)
    return cast(Callable[[object], object], dispatch), ast.unparse(expression), hashlib.sha256(source.encode()).hexdigest()


class ReadyProvider:
    """Already-ready delivery; no worker compute or waiting is timed."""

    def __init__(self, input_value: Value) -> None:
        self.values = {input_value: 1}
        self.dependencies = {"encoder": 1, "caption": 2, "source": 3}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def take(self, work: Hashable) -> InputDelivery:
        return InputDelivery(work, self.values, self.dependencies)


def workflow_probe(
    dormant_actions: int = 0, unit_count: int = 1
) -> tuple[
    RunCoordinator,
    Callable[[], tuple[AttemptReport, ...]],
    Callable[[], tuple[PreparedAction, InputActivity]],
    Callable[[], bool],
    Callable[[], AttemptReport],
]:
    if dormant_actions < 0 or unit_count < 1:
        raise ValueError("invalid cost-fixture size")
    input_value, output = Value("input"), Value("output")
    provider = ReadyProvider(input_value)
    units = tuple(f"unit_{index}" for index in range(unit_count))
    action = Action(
        "train",
        (input_value,),
        (Operation("increment", increment, (input_value,), (output,)),),
        tuple(OptimizationInput(unit, output) for unit in units),
        (output,),
    )
    activity = InputActivity(
        "producer",
        (input_value,),
        tuple(provider.dependencies),
        provider,
        lambda request, delivery: dict(request.dependencies) == dict(delivery.dependencies),
    )
    request = InputRequest(0, provider.dependencies)
    due = DueWork("train", request)
    actions = (action,) + tuple(
        Action(f"dormant_{index}", action.inputs, action.operations, action.optimization, action.observations)
        for index in range(dormant_actions)
    )
    workflow = prepare_workflow(
        RunDescription(
            actions,
            (activity,),
            tuple(InputFeed(item.name, "producer") for item in actions),
            lambda coordinates: (DueWork("train", InputRequest(coordinates["turn"], provider.dependencies)),),
        )
    )
    outcome = tuple(UnitOutcome(unit, "contribution_recorded") for unit in units)
    advance = AdvancementResult(outcome, can_continue=True)
    run = RunCoordinator(workflow, lambda _: advance)
    run.start()
    turn = 0

    def tick() -> tuple[AttemptReport, ...]:
        nonlocal turn
        turn += 1
        report = run.tick({"turn": turn})
        run.reports.clear()  # Model bounded report draining, equally for every sample.
        return report

    def lookup() -> tuple[PreparedAction, InputActivity]:
        return workflow.actions["train"], workflow.feeds["train"]

    delivery = provider.take(0)

    def admission() -> bool:
        return activity.admit(request, delivery)

    def correlation() -> AttemptReport:
        report = run._report(
            0, due, activity, "completed", outcomes=outcome, observations={output: 2}, produced_dependencies=delivery.dependencies
        )
        run.reports.clear()
        return report

    return run, tick, lookup, admission, correlation


def allocation_probe(call: Callable[[], object], iterations: int = 100) -> dict[str, int]:
    """Peak extra traced bytes, not an estimate of cumulative allocation rate."""
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    for _ in range(iterations):
        call()
    current, peak = tracemalloc.get_traced_memory()
    gc.collect()
    after_gc, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "batch_calls": iterations,
        "peak_extra_bytes": peak - before,
        "retained_extra_bytes_before_gc": current - before,
        "retained_extra_bytes_after_gc": after_gc - before,
    }


def measure(iterations: int, repeats: int) -> dict[str, object]:
    if iterations < 1 or repeats < 3:
        raise ValueError("use positive iterations and at least three timing repeats")
    cases: dict[str, Callable[[], object]] = {"harness_floor": lambda: None}
    for count in (1, 4, 16):
        prepared, direct = action_pair(count)
        inputs = {direct.input: 1}
        assert prepared.execute(inputs) == direct.execute(inputs)
        cases[f"candidate_{count}_cheap"] = lambda prepared=prepared, inputs=inputs: prepared.execute(inputs)
        cases[f"direct_{count}_cheap"] = lambda direct=direct, inputs=inputs: direct.execute(inputs)
    prepared, direct = action_pair(4, compute)
    inputs = {direct.input: 1}
    assert prepared.execute(inputs) == direct.execute(inputs)
    cases["candidate_4_compute"] = lambda: prepared.execute(inputs)
    cases["direct_4_compute"] = lambda: direct.execute(inputs)
    dispatch, expression, source_hash = current_loop_dispatch(action_pair(4)[1])
    assert dispatch(1) == action_pair(4)[1].execute({Value("value_0"): 1})
    cases["current_loop_4_cheap_dispatch"] = lambda: dispatch(1)
    run, tick, lookup, admission, correlation = workflow_probe()
    cases.update(workflow_tick=tick, workflow_lookup=lookup, input_admission=admission, attempt_correlation=correlation)
    runs = [run]
    for dormant in (256, 1024):
        extra_run, extra_tick, *_ = workflow_probe(dormant_actions=dormant)
        runs.append(extra_run)
        cases[f"workflow_tick_dormant_{dormant}"] = extra_tick
    for unit_count in (2, 16):
        extra_run, extra_tick, *_ = workflow_probe(unit_count=unit_count)
        runs.append(extra_run)
        cases[f"workflow_tick_units_{unit_count}"] = extra_tick
    samples: dict[str, list[float]] = {name: [] for name in cases}
    try:
        for call in cases.values():
            for _ in range(min(iterations, 1000)):
                call()
        for repeat in range(repeats):
            # Reverse order each repeat so one path is not always warmed first.
            names = tuple(cases) if repeat % 2 == 0 else tuple(reversed(cases))
            for name in names:
                call = cases[name]
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    call()
                samples[name].append((time.perf_counter_ns() - start) / iterations / 1000)
        allocations = {name: allocation_probe(call) for name, call in cases.items() if name != "harness_floor"}
    finally:
        for selected_run in runs:
            selected_run.stop()
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "gc_enabled": gc.isenabled(),
            "clock": "perf_counter_ns",
            "iterations": iterations,
            "repeats": repeats,
        },
        "scope": "CPU fixtures only; compilation/startup excluded; workflow uses ready input and reported contribution, no optimizer/model compute. Component timings overlap and are not additive. Reports are drained after each tick.",
        "current_loop": {
            "source_sha256": source_hash,
            "expression": expression,
            "scope": "actual call expression with direct fixture facade; argument dispatch only",
        },
        "timing_us_per_call": {
            name: {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}
            for name, values in samples.items()
        },
        "allocations": allocations,
        "paired_candidate_minus_direct_us": {
            suffix: {
                "median": statistics.median(
                    samples[f"candidate_{suffix}"][index] - samples[f"direct_{suffix}"][index] for index in range(repeats)
                )
            }
            for suffix in ("1_cheap", "4_cheap", "16_cheap", "4_compute")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(measure(args.iterations, args.repeats), indent=2))


if __name__ == "__main__":
    main()
