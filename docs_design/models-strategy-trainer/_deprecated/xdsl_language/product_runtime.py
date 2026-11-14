"""External pressure choices and owner observations for product qualification."""

from __future__ import annotations

import asyncio
import threading

from .prepare import Prepared, data
from .product_bindings import installed
from .product_contracts import initial
from .target import Controls, Runtime


async def execute(plan: Prepared, kind: str, derived: bool, scenario: str = "normal", *, wrong: bool = False) -> dict:
    gate = threading.Event()
    held = "make" if scenario == "cancel_make" else "consume"
    completion_closure = scenario in ("cancel_at_completion", "failure_at_completion")
    interrupted = scenario != "normal" and not completion_closure
    late_failure = scenario in ("cancel_failure", "failure")
    controls = Controls(
        gates={(held, 0): gate} if interrupted else {}, failures={(held, 0): "late retained consumer failure"} if late_failure else {}
    )
    runtime = Runtime(plan, installed(kind, derived=derived, wrong=wrong), initial(kind), controls=controls, history=False)
    choices = []

    async def checkpoint(point):
        runtime.facts.clear()
        runtime.outcomes.clear()
        choices.append((point.source, point.phase))
        if completion_closure and point.source == "make" and point.phase == "call_complete":
            conditions["completion_closed_before_admission"] = True
            if scenario == "cancel_at_completion":
                runtime.cancel(task)
            else:
                runtime.failed = True
            return
        await asyncio.sleep(0)

    controls.checkpoint = checkpoint
    task = asyncio.create_task(runtime.run())
    competing = None
    conditions = {}
    acquired = asyncio.Event()
    internal = runtime.resource_names["storage"]

    async def writer():
        async with runtime.state.acquire((), (internal,)):
            acquired.set()

    try:
        if interrupted:
            async with asyncio.timeout(15):
                while not controls.entered.get((held, 0), threading.Event()).is_set():
                    if task.done():
                        await task
                        raise RuntimeError("run completed without physical pressure boundary")
                    await asyncio.sleep(0.001)
            conditions["physical_entered"] = bool(runtime.workers.active)
            conditions["storage_protected"] = internal in runtime.state.state_locks
            conditions["products_before_settlement"] = len(runtime.retained)
            competing = asyncio.create_task(writer())
            if scenario != "failure":
                runtime.cancel(task)
                for _ in range(3):
                    await asyncio.sleep(0)
                    runtime.cancel(task)
            else:
                await asyncio.sleep(0)
            conditions["run_waits_for_physical"] = not task.done()
            conditions["writer_blocked"] = not acquired.is_set()
            conditions["no_early_receipts"] = not runtime.product_receipts
            gate.set()
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 20)
        raised = isinstance(result[0], BaseException)
        if competing is not None:
            await asyncio.wait_for(asyncio.gather(competing, return_exceptions=True), 5)
        receipts = [
            {"source": source, "ordinal": ordinal, "disposition": status, "versions": data(versions)}
            for (source, ordinal), (status, versions) in runtime.product_receipts.items()
        ]
        return {
            "kind": kind,
            "derived": derived,
            "scenario": scenario,
            "result": runtime.value("result"),
            "raised": raised,
            "failure": str(result[0]) if raised else "",
            "conditions": conditions,
            "receipts": receipts,
            "live_products": len(runtime.retained),
            "drained": runtime.workers.active == 0 and runtime.workers.executor is None and not runtime.state.state_locks,
            "writer_acquired": acquired.is_set(),
            "storage_unusable": internal in runtime.state.unusable_states,
            "history_empty": not runtime.facts and not runtime.outcomes,
            "consumer_calls": runtime.call_counts.get("consume", 0),
            "choices": choices,
        }
    finally:
        gate.set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)
        if competing is not None:
            await asyncio.gather(competing, return_exceptions=True)
