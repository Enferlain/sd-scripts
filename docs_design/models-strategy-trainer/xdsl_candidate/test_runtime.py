"""Positive, failure, cancellation, control-progress and shared-fault evidence."""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace

import pytest

from .conformance import check_b
from . import fixture
from .dialects import parse
from .fixture import build, execute, registry
from .prepare import Prepared, Registry, accept, prepare
from .runtime import ScalarSession


def with_call(symbol: str, invoke: Callable[[int], Awaitable[object]]) -> Prepared:
    bindings = dict(registry().bindings)
    bindings[symbol] = replace(bindings[symbol], invoke=invoke)
    return prepare(accept(build()), Registry("test-registry", bindings))


@pytest.mark.asyncio
@pytest.mark.parametrize("horizon", [1, 2])
@pytest.mark.parametrize("history", [0, 1, 32])
async def test_data_only_result_and_associations_survive_diagnostic_eviction(horizon: int, history: int) -> None:
    result = await asyncio.wait_for(execute(prepare(accept(build(horizon=horizon)), registry()), diagnostic_limit=history), 3)
    report = asdict(result.report)
    check_b(report, horizon=horizon)
    assert report["status"] == "stopped" and len(report["tickets"]) >= 4
    assert len(report["diagnostics"]) <= history
    assert any(ticket["policy_revision"] > 0 for ticket in report["tickets"])
    if horizon == 2:
        assert report["tickets"][1]["policy_revision"] == 0  # Actual preselected policy, never relabelled.


@pytest.mark.asyncio
async def test_feedback_progress_while_publication_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    allow_receipt = asyncio.Event()

    async def slow_receipt(value: int) -> int:
        await allow_receipt.wait()
        return value

    session = ScalarSession(with_call("receipt_v1", slow_receipt), diagnostic_limit=1000)
    original = session._apply_feedback
    progress: list[tuple[int, int]] = []

    def observe(ticket, score):
        if not progress:
            assert session.buffered == 1 and session.publication_waiting == {1}
            original(ticket, score)
            # Feedback applied with the producer still blocked and the output
            # slot retained. No draining is needed to make control progress.
            assert session.buffered == 1 and session.publication_waiting == {1}
            progress.append((session.feedback_count, session.outstanding))
            session.stop()
        else:
            original(ticket, score)

    monkeypatch.setattr(session, "_apply_feedback", observe)
    running = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(session.wait_for(lambda: session.publication_waiting == {1}), 3)
        assert session.feedback_count == 0 and session.outstanding == 2 and session.buffered == 1
        allow_receipt.set()
        report = await asyncio.wait_for(running, 3)
    finally:
        allow_receipt.set()
        session.stop()
        if not running.done():
            await running
    assert progress == [(1, 1)]
    assert len(report.tickets) == 2 and report.high_water == 2
    check_b(asdict(report))


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["double_v1", "offset_v1", "square_v1", "receipt_v1"])
@pytest.mark.parametrize("failure", ["raise", "cancel_error", "invalid", "bool"])
async def test_entered_failure_and_invalid_output_are_honest(symbol: str, failure: str) -> None:
    effects: list[int] = []

    async def selected(value: int) -> object:
        await asyncio.sleep(0)
        effects.append(value)
        if failure == "raise":
            raise RuntimeError("selected failure with possible effects")
        if failure == "cancel_error":
            raise asyncio.CancelledError("selected work raised this, not coordinator cancellation")
        return True if failure == "bool" else None

    result = await asyncio.wait_for(execute(with_call(symbol, selected)), 3)
    report = asdict(result.report)
    assert report["status"] == "failed" and effects
    failed = next(ticket for ticket in report["tickets"] if ticket["terminal"] == "failed")
    assert failed["calls"][-1]["outcome"] == ("entered_failed" if failure in ("raise", "cancel_error") else "invalid_output")
    assert failed["publication_ack"] is None and failed["feedback_ack"] is None
    check_b(report)


@pytest.mark.asyncio
async def test_wrong_publication_receipt_cannot_authorize_feedback() -> None:
    async def wrong(value: int) -> int:
        return value + 1

    result = await execute(with_call("receipt_v1", wrong))
    check_b(asdict(result.report))
    assert result.report.status == "failed" and not any(d.kind == "feedback" for d in result.report.decisions)


@pytest.mark.asyncio
async def test_partial_publication_failure_keeps_prior_acknowledged_tickets() -> None:
    count = 0

    async def selected(value: int) -> int:
        nonlocal count
        count += 1
        await asyncio.sleep(0)
        if count == 3:
            raise RuntimeError("third publication entered and failed")
        return value

    result = await asyncio.wait_for(execute(with_call("receipt_v1", selected), stop_after=10, diagnostic_limit=0), 3)
    check_b(asdict(result.report))
    assert [ticket.terminal for ticket in result.report.tickets[:3]] == ["acknowledged", "acknowledged", "failed"]
    assert len([decision for decision in result.report.decisions if decision.kind == "feedback"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["stop", "cancel_twice"])
@pytest.mark.parametrize("symbol", ["double_v1", "square_v1", "receipt_v1"])
async def test_stop_and_cancellation_hold_physical_uses_until_thread_completion(symbol: str, interrupt: str) -> None:
    entered, release, physically_finished = threading.Event(), threading.Event(), threading.Event()
    original = registry().bindings[symbol].invoke

    def physical() -> None:
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release physical worker")
        physically_finished.set()

    async def selected(value: int) -> object:
        await asyncio.to_thread(physical)
        return await original(value)

    session = ScalarSession(with_call(symbol, selected), diagnostic_limit=0)
    running = asyncio.create_task(session.run())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if interrupt == "stop":
            session.stop()
        else:
            running.cancel()
            await asyncio.sleep(0)
            running.cancel()
        await asyncio.sleep(0)
        assert not physically_finished.is_set() and not running.done()
        assert session.active_calls > 0 and session.outstanding > 0 and session.report is None
    finally:
        session.stop()
        release.set()
    if interrupt == "stop":
        report = await asyncio.wait_for(running, 3)
        assert report.status == "stopped"
    else:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 3)
        report = session.report
        assert report is not None and report.status == "cancelled"
    assert physically_finished.is_set() and session.active_calls == session.outstanding == 0
    check_b(asdict(report))


@pytest.mark.asyncio
async def test_duplicate_or_premature_feedback_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original_registry = registry()
    gate = asyncio.Event()

    async def waiting(value: int) -> object:
        await gate.wait()
        return await original_registry.bindings["double_v1"].invoke(value)

    session = ScalarSession(with_call("double_v1", waiting))
    running = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(session.wait_for(lambda: session.outstanding == 1), 3)
        ticket = session._active[0]
        with pytest.raises(ValueError, match="feedback lacks"):
            session._apply_feedback(ticket, 1)
        assert session.feedback_count == 0 and session.outstanding == 1
        original = session._apply_feedback

        def duplicate_control(ticket, score):
            original(ticket, score)
            revision = session.feedback_count
            with pytest.raises(ValueError, match="feedback lacks"):
                original(ticket, score)
            assert session.feedback_count == revision
            session.stop()

        monkeypatch.setattr(session, "_apply_feedback", duplicate_control)
        gate.set()
        result = await asyncio.wait_for(running, 3)
    finally:
        gate.set()
        session.stop()
        if not running.done():
            await running
    check_b(asdict(result))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "number",
        "policy",
        "key",
        "source",
        "association",
        "feedback",
        "mapping",
        "outcome",
        "release",
        "stop",
        "bound",
        "missing_decision",
    ],
)
async def test_independent_checker_rejects_corrupted_facts(corruption: str) -> None:
    report = asdict((await execute(prepare(accept(build()), registry()))).report)
    bad = copy.deepcopy(report)
    ticket = bad["tickets"][0]
    if corruption == "number":
        ticket["calls"][0]["output"] += 1
    elif corruption == "policy":
        bad["tickets"][1]["policy_revision"] += 1
    elif corruption == "key":
        ticket["key"] = 99
    elif corruption == "source":
        ticket["source_revision"] = 1
    elif corruption == "association":
        ticket["calls"][1]["input"] += 1
    elif corruption == "feedback":
        ticket["feedback_ack"] = ticket["publication_ack"] - 1
    elif corruption == "mapping":
        ticket["calls"][0]["origin"] = "another-stage"
    elif corruption == "outcome":
        ticket["terminal"] = "abandoned"
    elif corruption == "release":
        ticket["calls"][0]["released"] = ticket["calls"][0]["entered"]
    elif corruption == "stop":
        bad["stop_clock"] = ticket["selected"]
    elif corruption == "bound":
        bad["high_water"] = 1
    else:
        bad["decisions"] = bad["decisions"][1:]
    with pytest.raises(AssertionError):
        check_b(bad)


@pytest.mark.asyncio
async def test_shared_wrong_implementation_passes_equivalence_but_fails_independent_oracle() -> None:
    async def wrong(value: int) -> int:
        await asyncio.sleep(0)
        return value  # Both fresh reconstructions share the same defect.

    first = with_call("double_v1", wrong)
    bindings = dict(registry().bindings)
    bindings["double_v1"] = replace(bindings["double_v1"], invoke=wrong)
    second = prepare(accept(parse(first.accepted.text)), Registry("test-registry", bindings))
    reports = [asdict((await execute(plan)).report) for plan in (first, second)]
    projections = [[call["output"] for ticket in report["tickets"] for call in ticket["calls"]] for report in reports]
    assert projections[0] == projections[1]
    for report in reports:
        with pytest.raises(AssertionError, match=r"arithmetic|input"):
            check_b(report)


@pytest.mark.asyncio
async def test_stop_before_selection_and_single_lifetime() -> None:
    session = ScalarSession(prepare(accept(build()), registry()))
    session.stop()
    result = await asyncio.wait_for(session.run(), 3)
    assert result.status == "stopped" and not result.tickets and session.outstanding == 0
    with pytest.raises(RuntimeError, match="one lifetime"):
        await session.run()


@pytest.mark.asyncio
@pytest.mark.parametrize("driver", [False, True])
@pytest.mark.parametrize("failure", ["raise", "invalid"])
async def test_cancellation_retains_physical_failure_and_driver_drains(
    driver: bool,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def physical() -> None:
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test worker release missing")
        finished.set()

    async def selected(value: int) -> object:
        await asyncio.to_thread(physical)
        if failure == "raise":
            raise RuntimeError("physical work failed after interruption")
        return None

    prepared = with_call("receipt_v1", selected)
    session = ScalarSession(prepared, diagnostic_limit=0)
    if driver:
        monkeypatch.setattr(fixture, "ScalarSession", lambda *args, **kwargs: session)
    running = asyncio.create_task(fixture.execute(prepared) if driver else session.run())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        running.cancel()
        await asyncio.sleep(0)
        running.cancel()
        await asyncio.sleep(0)
        assert not running.done() and session.active_calls > 0 and session.report is None
    finally:
        session.stop()
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 3)
    report = session.report
    assert finished.is_set() and report is not None
    assert report.status == "cancelled" and report.errors and session.active_calls == session.outstanding == 0
    check_b(asdict(report))


@pytest.mark.asyncio
@pytest.mark.parametrize("yields", [0, 2, 5])
async def test_slow_scoring_does_not_change_the_selected_algorithm(yields: int) -> None:
    async def slow(value: int) -> int:
        for _ in range(yields):
            await asyncio.sleep(0)
        return value * value

    result = await asyncio.wait_for(execute(with_call("square_v1", slow), stop_after=8, diagnostic_limit=1), 3)
    check_b(asdict(result.report))
    assert result.report.high_water == 2


@pytest.mark.parametrize("extension", [False, True])
def test_actual_session_completes_with_python_assertions_disabled(extension: bool) -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = """
import asyncio,importlib,json,sys
from dataclasses import asdict
base="docs_design.models-strategy-trainer.xdsl_candidate."
p=importlib.import_module(base+"prepare")
f=importlib.import_module(base+"fixture")
if sys.argv[1]=="extension":
    e=importlib.import_module(base+"pair_extension")
    builder,registry=e.build_extension,e.extension_registry
else:
    builder,registry=f.build,f.registry
runtime=importlib.import_module(base+"runtime")
async def main():
    session=runtime.ScalarSession(p.prepare(p.accept(builder()),registry(),fuse=True))
    task=asyncio.create_task(session.run())
    await session.wait_for(lambda:session.feedback_count>=4)
    session.stop()
    report=await task
    print(json.dumps({"report":asdict(report),"outstanding":session.outstanding,"active_calls":session.active_calls,"optimized":not __debug__}))
asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script, "extension" if extension else "base"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["optimized"] and result["outstanding"] == result["active_calls"] == 0
    assert result["report"]["status"] == "stopped" and len(result["report"]["tickets"]) >= 4
    if extension:
        from .extension_conformance import check_extension

        check_extension(result["report"])
    else:
        check_b(result["report"])
