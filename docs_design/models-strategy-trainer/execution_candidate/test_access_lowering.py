"""Access specialization preserves the shared protection and completion rules."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from typing import Literal

import pytest
import torch

from .examples import Scale, adversarial, diffusion, joint, producer_continuity, replacement, routed_views, scale, two_pass
from .graph import (
    Advance,
    Block,
    Call,
    Choice,
    Contract,
    Differentiate,
    NotReady,
    Output,
    OwnedState,
    Producer,
    Repeat,
    Run,
    Together,
    Unit,
    Use,
)
from .prepare import Engine, WorkFailed, prepare
from .state import Replacement
from .test_candidate import run
from .test_specialized_lowering import observable

type AccessKind = Literal["general", "prepared"]
type LoweringKind = Literal["calls", "inline"]


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("case", ("adaptive", "alternating", "joint", "two-pass", "replacement", "retained", "isolated"))
def test_access_realizations_preserve_the_same_accepted_arrangement(lowering: LoweringKind, case):
    build = {
        "adaptive": diffusion,
        "alternating": adversarial,
        "joint": joint,
        "two-pass": two_pass,
        "replacement": replacement,
        "retained": routed_views,
        "isolated": lambda: routed_views(isolated=True, recompute=True),
    }[case]
    accepted = Contract(two_pass=case == "two-pass").accept(build())
    baseline = prepare(accepted, lowering=lowering).publish()
    specialized = prepare(accepted, lowering=lowering, access="prepared").publish()
    assert baseline.access == "general" and specialized.access == "prepared"
    run(Engine(baseline).run())
    run(Engine(specialized).run())
    assert observable(baseline) == observable(specialized)
    assert not specialized.state.progress.leases and not specialized.state.progress.demands
    if case in ("retained", "isolated"):
        assert baseline.source == specialized.source
        assert not specialized.state.active_retentions


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
def test_independent_producer_fixture_preserves_provenance_and_demand_end(lowering: LoweringKind, access: AccessKind):
    # Each fixture has private staging Events. They are not reusable run state.
    ready = prepare(Contract().accept(producer_continuity()), lowering=lowering, access=access).publish()
    run(Engine(ready).run())
    state = ready.state
    assert state.image is not None and state.image.units["weights"].steps == 1
    feed = state.feeds["condition"]
    assert [key for key, _, _ in feed.handed_off] == [1]
    assert feed.handed_off[0][2] == (state.stamp("encoder"),)
    assert feed.closed and not feed.ready and not state.active_producers
    assert not state.progress.leases and not state.progress.demands
    state.capture()


@pytest.mark.parametrize("access", ("general", "prepared"))
def test_empty_input_demand_is_private_bookkeeping_but_owner_protection_remains(access: AccessKind, monkeypatch):
    seen = []

    def inspect(context):
        context.states["owner"].append("ran")
        seen.append((len(state.progress.demands), state.state_locks, len(state.progress.leases)))

    ready = prepare(
        Contract().accept(Run(Block("work", (Call("inspect", inspect, states=("owner",)),)), states=(OwnedState("owner", list),))),
        access=access,
    ).publish()
    state = ready.state
    borrowed_regions = []
    original = state._region_sources

    def observe(region):
        borrowed_regions.append(region)
        return original(region)

    monkeypatch.setattr(state, "_region_sources", observe)
    run(Engine(ready).run())
    assert seen == [(int(access == "general"), {"owner"}, 1)]
    assert borrowed_regions == ([None] if access == "general" else [])
    assert state.owners["owner"] == ["ran"] and state.completed


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
def test_live_producer_read_still_delays_a_conflicting_training_block(lowering: LoweringKind, access: AccessKind, monkeypatch):
    async def scenario():
        held, release, blocked = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def produce(context):
            async with context.use_source() as source:
                assert float(source.models["model"].get_parameter("weight").detach()) == 1.0
                held.set()
                await release.wait()

        async def wait(context):
            await held.wait()

        def objective(context):
            return (context.models["model"](torch.tensor(2.0)) - 1).square()

        waiting = Block("wait", (Call("await-source", wait),))
        learning = Block(
            "learn",
            (
                Call("loss", objective, outputs=(Output("loss", torch.Tensor),), uses=(Use("model"),)),
                Differentiate("backward", "loss", ("unit",), "gradient"),
                Advance("update", "gradient", ("unit",)),
            ),
        )
        filing = Run(
            Together(
                "run",
                (
                    Producer("source", (), produce, (Use("model"),)),
                    Repeat(
                        "learning",
                        (waiting, learning),
                        lambda counts, state: Choice("wait") if not counts else Choice("learn") if not counts.get("learn") else None,
                        "policy",
                    ),
                ),
            ),
            (scale("model", 1.0),),
            (Unit("unit", (("model", "weight"),), learning_rate=0.1),),
            (OwnedState("policy", lambda: None),),
        )
        ready = prepare(Contract().accept(filing), lowering=lowering, access=access).publish()
        state = ready.state
        original = state.progress.waiting

        @contextmanager
        def observe(description, mode, dependencies):
            if description.startswith("access:participant/model:write"):
                blocked.set()
            with original(description, mode, dependencies):
                yield

        monkeypatch.setattr(state.progress, "waiting", observe)
        task = asyncio.create_task(Engine(ready).run())
        try:
            await blocked.wait()
            assert held.is_set() and state.readers == {"model": 1} and not state.writers
            assert state.image is not None and state.image.units["unit"].steps == 0
            release.set()
            await task
            assert state.image.units["unit"].steps == 1
            assert float(state.image.bindings["model"].model.get_parameter("weight").detach()) == pytest.approx(0.6)
            assert not state.progress.leases and not state.active_producers
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("access", ("general", "prepared"))
@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_unshared_block_still_protects_publication_and_resolves_fresh_state(access: AccessKind, lowering: LoweringKind):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        observed = []

        async def inspect(context):
            observed.append(float(context.models["model"].get_parameter("weight").detach()))
            context.states["owner"].append(observed[-1])
            entered.set()
            await release.wait()

        filing = Run(
            Block("inspect", (Call("inspect", inspect, uses=(Use("model"),), states=("owner",)),)),
            (scale("model", 1.0),),
            states=(OwnedState("owner", list),),
        )
        ready = prepare(Contract().accept(filing), access=access, lowering=lowering).publish()
        state = ready.state
        task = asyncio.create_task(Engine(ready).run())
        change = None
        try:
            await entered.wait()
            candidate = state.candidate({"model": Scale(4.0)}, replacing="model")
            with pytest.raises(NotReady, match="protected current use"):
                state.install(candidate, state.generation)
            with pytest.raises(NotReady, match="quiescent"):
                state.capture()
            change = asyncio.create_task(state.replace(Replacement("model", 1, Scale(9.0)), "model"))
            await asyncio.sleep(0)
            assert not change.done() and state.readers == {"model": 1} and state.state_locks == {"owner"}
            release.set()
            await task
            await change
            await Engine(ready).run()
            assert observed == [1.0, 9.0] and state.owners["owner"] == observed
            assert state.stamp("model").binding == 2 and not state.progress.leases
            state.unusable.add("model")
            with pytest.raises(NotReady, match="unavailable"):
                await Engine(ready).run()
            assert observed == [1.0, 9.0]
        finally:
            release.set()
            await asyncio.gather(task, *((change,) if change is not None else ()), return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("access", ("general", "prepared"))
def test_worker_cancellation_keeps_the_same_physical_lifetime(lowering: LoweringKind, access: AccessKind):
    async def scenario():
        loop, entered, finish = asyncio.get_running_loop(), asyncio.Event(), threading.Event()

        def selected(context):
            context.states["effects"].append("entered")
            loop.call_soon_threadsafe(entered.set)
            assert finish.wait(4), "worker was not released"
            assert context.stop_requested is not None and context.stop_requested()

        filing = Run(
            Block("work", (Call("work", selected, uses=(Use("model", "write"),), states=("effects",)),)),
            (scale("model", 1.0),),
            states=(OwnedState("effects", list),),
        )
        ready = prepare(Contract().accept(filing), access=access, lowering=lowering, worker_calls=("work",)).publish()
        state = ready.state
        task = asyncio.create_task(Engine(ready).run())
        try:
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and state.writers == {"model"} and state.state_locks == {"effects"}
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert state.owners["effects"] == ["entered"]
            assert state.unusable == {"model"} and state.unusable_states == {"effects"}
            assert not state.progress.leases and not state.completed
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario())


@pytest.mark.parametrize("access", ("general", "prepared"))
@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_partial_unit_failure_still_preserves_known_outcomes(access: AccessKind, lowering: LoweringKind, monkeypatch):
    ready = prepare(Contract().accept(joint()), access=access, lowering=lowering).publish()
    assert ready.state.image is not None
    unit = ready.state.image.units["B"]

    def fail():
        with torch.no_grad():
            unit.parameters[0].add_(1)
        raise ValueError("lost backend confirmation")

    monkeypatch.setattr(unit.optimizer, "step", fail)
    with pytest.raises(WorkFailed) as caught:
        run(Engine(ready).run())
    assert [outcome.optimizer for outcome in caught.value.unit_outcomes] == ["returned", "uncertain"]
    assert [outcome.reset for outcome in caught.value.unit_outcomes] == ["returned", "not_attempted"]
    assert ready.state.unusable == {"a", "b"} and not ready.state.progress.leases
    with pytest.raises(NotReady):
        ready.state.capture()


def test_unknown_access_realization_rejects_before_loading():
    with pytest.raises(NotReady, match="access realization"):
        prepare(Contract().accept(joint()), access="unsafe")  # ty: ignore[invalid-argument-type]


def test_cost_probe_checks_matched_execution_in_a_fresh_gil_disabled_process():
    environment = dict(os.environ)
    environment.pop("PYTHON_GIL", None)
    environment.pop("PYTHONOPTIMIZE", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "docs_design.models-strategy-trainer.execution_candidate.access_cost_probe",
            "--worker",
            "--iterations",
            "4",
            "--samples",
            "2",
            "--warmup",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert not result["environment"]["gil_enabled"] and not result["gil_enabled_after"]
    assert not result["environment"]["dynamo_imported"] and not result["environment"]["triton_imported"]
    assert result["imports_after"] == {"dynamo_imported": False, "triton_imported": False}
    assert result["environment"]["free_threaded_build"] and result["environment"]["pid"] != os.getpid()
    assert result["started_utc"] < result["finished_utc"]
    for case in result["cases"].values():
        assert case["access_realizations"] == {"general": "general", "prepared": "prepared"}
        assert case["order"] == [["general", "prepared"], ["prepared", "general"]]
        assert all(len(values) == 2 and min(values) > 0 for values in case["ns_per_invocation"].values())
        assert len(case["paired_saving_ns"]) == 2
