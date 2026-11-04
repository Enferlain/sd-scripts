"""Cold-capable qualification driver; external choices, not an IR scheduler."""

from __future__ import annotations

import asyncio
import random
import threading

from collections import Counter
from dataclasses import asdict
from typing import cast

from .contracts import text
from .prepare import Prepared, data
from .qualification_target import selected
from .target import Controls, Point, Runtime


async def until(predicate, *, seconds: float = 10) -> None:
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.001)


def erase(runtime: Runtime) -> None:
    """Drop projections, retaining live authority, products and owner receipts."""
    runtime.facts.clear()
    runtime.outcomes.clear()
    runtime.state.trace.clear()
    runtime.owner.source.clear()
    for feed in runtime.state.feeds.values():
        feed.handed_off.clear()
        feed.discarded.clear()


def failure_messages(error: BaseException | None) -> list[str]:
    seen: set[int] = set()

    def visit(error: BaseException | None) -> list[str]:
        if error is None or id(error) in seen:
            return []
        seen.add(id(error))
        messages = [f"{type(error).__name__}: {error}"]
        children = error.exceptions if isinstance(error, BaseExceptionGroup) else ()
        children = (
            *children,
            error.__cause__,
            getattr(error, "cause", None),
            getattr(error, "primary", None),
            getattr(error, "cancellation", None),
            *getattr(error, "cleanup_failures", ()),
        )
        for child in children:
            if isinstance(child, BaseException):
                messages.extend(visit(child))
        return messages

    return visit(error)


def summary(runtime: Runtime) -> dict:
    """Read surviving products and owner ledgers, never observation archives."""
    resources = {name: {"value": runtime.value(name), "epoch": runtime._version(name).ordinal} for name in runtime.declarations}
    units = {
        name: {
            "steps": unit.steps,
            "unfinished": unit.unfinished,
            "outcomes": [
                {
                    "contribution": item.contribution,
                    "optimizer": item.optimizer,
                    "reset": item.reset,
                    "inputs": [asdict(link) for link in item.input_handoffs],
                }
                for item in unit.outcomes
            ],
        }
        for name, unit in runtime.state.image.units.items()
    }
    requests = [
        {
            "handler": text(slot.operation, "sym_name"),
            "issued": {ordinal: asdict(request) for ordinal, request in slot.issued.items()},
            "dispositions": slot.dispositions,
            "pending": len(slot.pending),
            "active": slot.current is not None,
            "permits": slot.capacity._value,
        }
        for slot in runtime.handlers.values()
    ]
    flows = {
        name: {
            "terminal": dict(flow.terminal),
            "active": len(flow.active),
            "selected": flow.next_key,
            "high_water": flow.high_water,
            "ready": len(runtime.state.feeds[name].ready),
            "ready_high_water": runtime.state.feeds[name].high_water,
            "permits": flow.capacity._value,
        }
        for name, flow in runtime.flows.items()
    }
    return cast(
        dict,
        data(
            {
                "resources": resources,
                "units": units,
                "requests": requests,
                "flows": flows,
                "source_receipts": [asdict(receipt) for receipts in runtime.closed_sources.values() for receipt in receipts],
                "completed": runtime.state.completed,
                "cancel_requested": runtime.cancelled,
                "unusable": sorted(name for name, internal in runtime.resource_names.items() if internal in runtime.state.unusable),
                "failures": failure_messages(runtime.failure),
                "physical_entered": sorted(key for key, event in runtime.controls.entered.items() if event.is_set()),
                "drained": not (
                    runtime.state.running
                    or runtime.workers.active
                    or runtime.workers.executor is not None
                    or runtime._live
                    or runtime.state.readers
                    or runtime.state.writers
                    or runtime.state.state_locks
                    or runtime.state.active_retentions
                    or runtime.state.active_producers
                    or any(flow.active for flow in runtime.flows.values())
                    or any(feed.ready for feed in runtime.state.feeds.values())
                    or any(slot.pending or slot.current is not None for slot in runtime.handlers.values())
                ),
            }
        ),
    )


async def execute(
    plan: Prepared,
    initial: dict,
    *,
    prefix: str = "",
    seed: int = 0,
    mode: str = "normal",
    erased: bool = False,
    wrong_score: bool = False,
) -> tuple[Runtime, dict]:
    """Each phase condition is fixed independently of a runtime admission trace."""
    rng = random.Random(seed)
    points: Counter[str] = Counter()
    blocked, release = asyncio.Event(), asyncio.Event()
    erasures = 0
    runtime: Runtime

    async def checkpoint(point: Point) -> None:
        nonlocal erasures
        points[point.phase] += 1
        if erased:
            erase(runtime)
            erasures += 1
        # The exclusion witness must rely on native progress, not test yields.
        if mode != "rejections":
            for _ in range(rng.randrange(3)):
                await asyncio.sleep(0)
        before_second = (
            mode == "orderly-before"
            and point.source == prefix + "training"
            and point.phase == "admit"
            and point.invocation[-1] == (prefix + "training", 1)
            and sum(name == prefix + "training" for name, _ in point.invocation) == 2
        )
        before_request = (
            mode == "pending-cancel"
            and point.source == prefix + "evaluation"
            and point.phase == "admit"
            and sum(name == prefix + "evaluation" for name, _ in point.invocation) == 2
        )
        if before_second or before_request:
            blocked.set()
            await release.wait()

    controls = Controls(checkpoint=checkpoint)
    if mode == "late":
        controls.before[prefix + "encode", 4] = asyncio.Event()
    if mode in ("prefetched", "orderly-after"):
        controls.before[prefix + "gradient", 2] = asyncio.Event()
    if mode in ("held-cancel", "held-failure"):
        controls.gates[prefix + "measure", 0] = threading.Event()
        if mode == "held-failure":
            controls.failures[prefix + "measure", 0] = "late physical failure"
    if mode == "late-producer-failure":
        controls.before[prefix + "measure", 0] = asyncio.Event()
        controls.gates[prefix + "encode", 6] = threading.Event()
        controls.failures[prefix + "encode", 6] = "late producer failure"
    if mode == "rejections":
        controls.before["consume.take", 0] = asyncio.Event()
    if mode not in {
        "normal",
        "late",
        "prefetched",
        "orderly-before",
        "orderly-after",
        "pending-cancel",
        "held-cancel",
        "held-failure",
        "late-producer-failure",
        "rejections",
    }:
        raise ValueError("unknown qualification choice")
    runtime = Runtime(plan, selected(wrong_score=wrong_score), initial, controls=controls, history=not erased)
    task = asyncio.create_task(runtime.run())
    witness: dict = {"seed": seed, "mode": mode}
    try:
        if mode == "late":
            await until(lambda: any(slot.issued for slot in runtime.handlers.values()))
            witness["pending_keeps_group_open"] = not runtime.closed_sources and not any(
                event.is_set() for event in runtime.production_stops.values()
            )
            witness["measurement_not_entered"] = (prefix + "measure", 0) not in controls.entered
            controls.before[prefix + "encode", 4].set()
        elif mode in ("prefetched", "orderly-after"):
            await until(lambda: (prefix + "gradient", 2) in controls.reached)
            if mode == "prefetched":
                await until(lambda: 4 in runtime.state.feeds[prefix + "records"].ready)
                witness["evaluation_input_prefetched"] = True
            else:
                runtime.request_stop(prefix + "training")
            controls.before[prefix + "gradient", 2].set()
        elif mode in ("orderly-before", "pending-cancel"):
            await asyncio.wait_for(blocked.wait(), 10)
            if mode == "orderly-before":
                runtime.request_stop(prefix + "training")
            else:
                runtime.cancel(task)
            release.set()
        elif mode in ("held-cancel", "held-failure"):
            await until(lambda: controls.entered.get((prefix + "measure", 0), threading.Event()).is_set())
            internal = runtime.resource_names[prefix + "model"]
            runtime.cancel(task)
            await asyncio.sleep(0.01)
            witness["protected_while_cancelled"] = runtime.state.readers[internal] == 1 and not task.done() and runtime.workers.active > 0
            controls.gates[prefix + "measure", 0].set()
        elif mode == "late-producer-failure":
            await until(lambda: (prefix + "measure", 0) in controls.reached)
            await until(lambda: controls.entered.get((prefix + "encode", 6), threading.Event()).is_set())
            controls.before[prefix + "measure", 0].set()
            await until(lambda: bool(runtime.closed_sources) and any(event.is_set() for event in runtime.production_stops.values()))
            witness["completed_before_failure"] = bool(runtime.value(prefix + "products")) and not task.done()
            controls.gates[prefix + "encode", 6].set()
        elif mode == "rejections":
            await until(lambda: runtime._version("policy").ordinal == 2)
            controls.before["consume.take", 0].set()
            heartbeats = 0

            async def heartbeat():
                nonlocal heartbeats
                while True:
                    heartbeats += 1
                    await asyncio.sleep(0)

            pulse = asyncio.create_task(heartbeat())
            try:
                await until(lambda: sum(status == "rejected_stale" for status in runtime.flows["records"].terminal.values()) >= 128)
                witness["heartbeats_during_exclusions"] = heartbeats
                runtime.cancel(task)
            finally:
                pulse.cancel()
                await asyncio.gather(pulse, return_exceptions=True)
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 15)
        witness["returned_error"] = failure_messages(result[0]) if isinstance(result[0], BaseException) else []
    finally:
        release.set()
        for event in controls.before.values():
            event.set()
        for event in controls.gates.values():
            event.set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)
    witness.update(points=dict(points), erasures=erasures)
    if erased:
        erase(runtime)
        witness["archives_empty"] = not runtime.facts and not runtime.outcomes and not runtime.state.trace and not runtime.owner.source
    return runtime, {"owner": summary(runtime), "witness": witness}
