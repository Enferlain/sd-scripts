"""External qualification choices for prepared filtering; no authoring import."""

from __future__ import annotations

import asyncio
import random
import threading

from .filter_checks import EXPECTED, check_conservation
from .filter_contracts import INITIAL
from .filter_target import installed
from .prepare import Prepared, data
from .qualification_runtime import erase, summary, until
from .target import Binding, Controls, Result, Runtime


SCENARIOS = (
    "normal",
    "late_input",
    "prefetched",
    "held_shutdown",
    "late_source_failure",
    "cancel_filter",
    "fail_filter",
    "cancel_then_fail_filter",
    "orderly_requested",
    "cancel_local",
    "invalid_stage",
    "all_drops",
)


async def execute(
    plan: Prepared, scenario: str = "normal", *, erase_history: bool = False, seed: int | None = None, wrong_double: bool = False
):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown external filter scenario: {scenario}")
    controls, choices, conditions = Controls(), [], {}
    generator = random.Random(seed)
    selected = installed(reject_all=scenario == "all_drops", wrong_double=wrong_double)
    if scenario == "invalid_stage":
        selected = type(selected)(
            tuple(
                Binding(item.spec, lambda args: Result(("invalid",))) if item.spec.name == "filter_offset" else item
                for item in selected.bindings
            ),
            selected.resources,
            selected.schemas,
        )
        controls.before["first.offset", 0] = asyncio.Event()
        controls.gates["predicate", 1] = threading.Event()
    if scenario == "late_input":
        controls.before["produce", 2] = asyncio.Event()
    if scenario == "prefetched":
        controls.before["first.take", 0] = asyncio.Event()
    if scenario in ("orderly_requested", "cancel_local"):
        controls.before["next.offset", 0] = asyncio.Event()
    if scenario in ("held_shutdown", "late_source_failure"):
        controls.before["next.publish", 1] = asyncio.Event()
        controls.gates["predicate", 5] = threading.Event()
        controls.gates["produce", 6] = threading.Event()
        if scenario == "late_source_failure":
            controls.failures["produce", 6] = "late physical source failure"
    if scenario in ("cancel_filter", "fail_filter", "cancel_then_fail_filter"):
        controls.gates["predicate", 1] = threading.Event()
        if scenario != "cancel_filter":
            controls.failures["predicate", 1] = "physical filter failure"

    async def checkpoint(point):
        choices.append((point.source, point.phase, point.ordinal))
        if seed is not None:
            for _ in range(generator.randrange(3)):
                await asyncio.sleep(0)
        if erase_history:
            erase(runtime)

    controls.checkpoint = checkpoint
    runtime = Runtime(plan, selected, INITIAL, controls=controls, history=not erase_history)
    task = asyncio.create_task(runtime.run())

    def entered(key):
        return key in controls.entered and controls.entered[key].is_set()

    def source_closed(name):
        return any(key[1] == name for key in runtime.closed_sources)

    try:
        if scenario == "invalid_stage":
            await until(lambda: entered(("predicate", 1)) and ("first.offset", 0) in controls.reached)
            controls.before["first.offset", 0].set()
            await until(lambda: runtime.failed)
            conditions["claim_retained_during_sibling_failure"] = (
                not runtime.cancelled and not task.done() and 1 in runtime.flows["raw"].active
            )
            controls.gates["predicate", 1].set()
        elif scenario == "late_input":
            await until(
                lambda: runtime.flows["raw"].terminal.get(1) == "filtered_out" and any(slot.issued for slot in runtime.handlers.values())
            )
            conditions["demand_survived_drop"] = not source_closed("consumers") and runtime.value("products") == ((0, 2),)
            controls.before["produce", 2].set()
        elif scenario == "prefetched":
            await until(lambda: len(runtime.flows["kept"].active) == 3 and len(runtime.state.feeds["kept"].ready) == 2)
            conditions["aggregate_bound_exhausted"] = runtime.flows["kept"].capacity._value == 0 and not entered(("first.extract", 0))
            controls.before["first.take", 0].set()
        elif scenario in ("orderly_requested", "cancel_local"):
            await until(lambda: ("next.offset", 0) in controls.reached)
            conditions["entered_iteration_owns_input"] = (
                runtime.flows["kept"].active[2].owner is not None and entered(("next.extract", 0)) and not entered(("next.offset", 0))
            )
            if scenario == "orderly_requested":
                runtime.request_stop("requested")
                controls.before["next.offset", 0].set()
            else:
                runtime.cancel(task)
        elif scenario in ("held_shutdown", "late_source_failure"):
            await until(lambda: entered(("predicate", 5)) and entered(("produce", 6)) and ("next.publish", 1) in controls.reached)
            controls.before["next.publish", 1].set()
            await until(lambda: source_closed("consumers"))
            conditions["group_finished_before_producers"] = (
                runtime.value("products") == EXPECTED and not source_closed("filtering") and not task.done() and runtime.workers.active >= 2
            )
            controls.gates["predicate", 5].set()
            await until(lambda: source_closed("filtering"))
            conditions["filter_closed_before_source"] = not task.done() and entered(("produce", 6))
            controls.gates["produce", 6].set()
        elif scenario in ("cancel_filter", "fail_filter", "cancel_then_fail_filter"):
            await until(lambda: entered(("predicate", 1)))
            if scenario != "fail_filter":
                runtime.cancel(task)
                await asyncio.sleep(0)
            conditions["claim_protected_through_physical_filter"] = (
                not task.done() and 1 in runtime.flows["raw"].active and 1 in runtime.flows["kept"].active and runtime.workers.active >= 1
            )
            controls.gates["predicate", 1].set()
        elif scenario == "all_drops":
            heartbeat = asyncio.Event()
            asyncio.get_running_loop().call_soon(heartbeat.set)
            await until(lambda: len(runtime.flows["raw"].settled) >= 128)
            conditions["progress_during_exclusions"] = (
                heartbeat.is_set() and not source_closed("consumers") and runtime.value("products") == ()
            )
            runtime.cancel(task)
        results = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 20)
        conditions["run_raised"] = isinstance(results[0], BaseException)
    finally:
        for event in (*controls.gates.values(), *controls.before.values()):
            event.set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)
    if erase_history:
        erase(runtime)
    facts = tuple(runtime.facts)
    observed = summary(runtime)
    observed["settlements"] = data({name: flow.settled for name, flow in runtime.flows.items()})
    observed["local_dispatches"] = dict(runtime.local_dispatches)
    observed["conditions"] = conditions
    if not observed["drained"]:
        raise AssertionError("filtering did not settle physical work and live ownership")
    for flow in runtime.flows.values():
        check_conservation(
            selected=flow.next_key,
            settled=flow.settled,
            active=len(flow.active),
            ready=len(runtime.state.feeds[flow.declaration.sym_name.data].ready),
            high_water=flow.high_water,
            horizon=flow.declaration.horizon.value.data,
        )
    return observed, facts, choices
