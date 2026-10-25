"""Semantic controls for the parallel-input probe; no speed thresholds."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import sysconfig
import threading

import pytest

from .graph import Producer, Together
from . import parallel_work_probe as probe
from .parallel_work_probe import ProviderStats, arrangement, corpus, process_text, trial
from .test_candidate import leaves
from .workers import PhysicalFailed, Workers


def test_processing_preserves_sample_boundaries_when_packing_and_normalizing():
    packed, timing = process_text(9, ((1, "A b b!"), (2, "C,d")), 2, threading.Event())
    normalized, _ = process_text(9, ((1, "a B B"), (2, "c d")), 2, threading.Event())
    assert packed == normalized
    assert packed.sample_ids == (1, 2) and packed.token_count == 5
    assert tuple(map(len, packed.packs)) == (2, 2, 1)
    assert packed.segments == (((1, 0, 2),), ((1, 2, 1), (2, 0, 1)), ((2, 1, 1),))
    assert timing.end_ns >= timing.start_ns and timing.cpu_ns > 0


def test_private_processing_honors_stop_and_rejects_zero_budget():
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(InterruptedError):
        process_text(0, ((1, "unfinished"),), 2, stopped)
    with pytest.raises(ValueError, match="positive"):
        process_text(0, ((1, "input"),), 0, threading.Event())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_worker_count_preserves_actual_pipeline_outcomes_and_bounds(lowering):
    async def check():
        first = await trial(1, 6, 8, 5, lowering)
        second = await trial(2, 6, 8, 5, lowering)
        assert first["outcome"] == second["outcome"]
        assert first["outcome"]["steps"] == 6
        assert len(first["outcome"]["inputs"]) == 6
        assert first["max_ready"] <= 2 and second["max_ready"] <= 2
        assert first["max_private_pending"] <= 4 and second["max_private_pending"] <= 4

    asyncio.run(check())


@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="Requires a free-threaded build")
@pytest.mark.parametrize("gil", ("on", "off"))
def test_measurement_process_checks_actual_gil_state_and_results(gil):
    command = [
        sys.executable,
        *(("-X", "gil=1") if gil == "on" else ()),
        "-m",
        "docs_design.models-strategy-trainer.execution_candidate.parallel_work_probe",
        "--measure",
        "--gil",
        gil,
        "--workers",
        "2",
        "--jobs",
        "2",
        "--records",
        "4",
        "--budget",
        "5",
    ]
    environment = os.environ.copy()
    environment.pop("PYTHON_GIL", None)
    result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    measured = json.loads(result.stdout)
    assert measured["environment"]["gil_enabled_after"] == (gil == "on")
    assert measured["environment"]["python_gil_override"] is None
    assert measured["outcome"]["steps"] == 2


@pytest.mark.parametrize("ending", ("returned", "stopped", "failed"))
def test_cancelled_processing_is_drained_without_publication_and_retains_real_failure(ending, monkeypatch):
    async def check():
        entered, release, physically_returned = asyncio.Event(), threading.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        stops, pools, published = [], [], []
        original = probe.process_text

        def tail(key, records, budget, stop):
            if key != 0:
                return original(key, records, budget, stop)
            result = original(key, records, budget, stop) if ending != "stopped" else None
            if key == 0:
                stops.append(stop)
                loop.call_soon_threadsafe(entered.set)
                try:
                    if not release.wait(5):
                        raise TimeoutError("Test did not release completed processing")
                    if ending == "stopped":
                        return original(key, records, budget, stop)
                    if ending == "failed":
                        raise ValueError("Actual processing failure during stop")
                finally:
                    physically_returned.set()
            return result

        class ObservedWorkers(Workers):
            def __init__(self, limit):
                super().__init__(limit)
                pools.append(self)

        class RecordingContext:
            def __init__(self):
                self.states = {"provider": ProviderStats()}

            async def publish(self, channel, product):
                published.append(product.key)

        monkeypatch.setattr(probe, "process_text", tail)
        monkeypatch.setattr(probe, "Workers", ObservedWorkers)
        filing = arrangement(corpus(2, 1), 1, 4)
        assert isinstance(filing.root, Together)
        source = filing.root.children[0]
        assert isinstance(source, Producer)

        async def invoke():
            await source.implementation(RecordingContext())

        task = asyncio.create_task(invoke())
        try:
            async with asyncio.timeout(5):
                await entered.wait()
                task.cancel()
                while not stops[0].is_set():
                    await asyncio.sleep(0)
                release.set()
                if ending == "failed":
                    with pytest.raises(ExceptionGroup) as failure:
                        await task
                    actual = [error for error in leaves(failure.value) if isinstance(error, PhysicalFailed)]
                    assert any(isinstance(error.cause, ValueError) and error.cancellation is not None for error in actual)
                else:
                    with pytest.raises(asyncio.CancelledError):
                        await task
            assert not published
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert physically_returned.is_set()
        assert pools[0].active == 0 and pools[0].executor is None

    asyncio.run(check())
