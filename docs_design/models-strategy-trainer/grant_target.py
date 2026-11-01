"""Existing C owner target, shared only after independent IR verification.

This module consumes target declarations. It defines no semantic IR, parser,
prepared correspondence or second grant/optimization owner.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

import torch

_prefix = "docs_design.models-strategy-trainer.execution_candidate"
graph = importlib.import_module(_prefix + ".graph")
backend = importlib.import_module(_prefix + ".prepare")
examples = importlib.import_module(_prefix + ".examples")


@dataclass(frozen=True)
class GrantSpec:
    session: str
    model: str
    unit: str
    parameter: str
    initial: float
    learning_rate: float
    clip_norm: float | None
    grant: str
    advance: str
    phases: tuple[tuple[str, str], ...]


@dataclass
class GrantFacts:
    """Observations; live grant and optimization owners decide readiness."""

    entered: bool = False
    selected_returned: bool = False
    initial_weight: float | None = None
    selected_end_weight: float | None = None
    selected_gradient: float | None = None
    restored: bool = False
    phases: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PreparedTarget:
    attempt: Any
    facts: GrantFacts
    model: str
    unit: str
    parameter: str

    def publish(self) -> GrantRun:
        ready = self.attempt.publish()
        return GrantRun(self, ready, backend.Engine(ready))


def causes(error: BaseException | None) -> tuple[BaseException, ...]:
    if error is None:
        return ()
    if isinstance(error, BaseExceptionGroup):
        return tuple(item for nested in error.exceptions for item in causes(nested))
    nested = getattr(error, "cause", None)
    cancellation = getattr(error, "cancellation", None)
    return (
        error,
        *(causes(nested) if isinstance(nested, BaseException) else ()),
        *(causes(cancellation) if isinstance(cancellation, BaseException) else ()),
    )


@dataclass
class GrantRun:
    prepared: Any
    ready: Any
    engine: Any
    last_report: dict[str, Any] | None = None
    started: bool = False
    interrupted: bool = False

    async def run(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("grant run has one owner lifetime")
        self.started = True
        task = asyncio.current_task()
        cancellations = task.cancelling() if task is not None else 0
        failure: BaseException | None = None
        try:
            await self.engine.run()
        except BaseException as error:
            failure = error
            self.interrupted = isinstance(error, asyncio.CancelledError) or (task is not None and task.cancelling() > cancellations)
        self.last_report = self.report(failure)
        if failure is not None and self.interrupted:
            raise failure
        return self.last_report

    def report(self, failure: BaseException | None = None) -> dict[str, Any]:
        failure = failure or self.engine.failure
        state = self.ready.state
        if state.image is None:
            raise RuntimeError("grant owner has no materialized state")
        runtime = state.image.units[self.prepared.unit]
        parameter = state.image.bindings[self.prepared.model].model.get_parameter(self.prepared.parameter)
        evidence = {record.invocation: record for record in (*runtime.outcomes, *runtime.active_outcomes.values())}
        records = [
            {
                "contribution": item.contribution,
                "optimizer": item.optimizer,
                "reset": item.reset,
                "invocation": item.invocation,
                "advance_invocation": item.advancement_invocation,
                "work": item.work,
                "input_count": len(item.input_handoffs),
            }
            for item in evidence.values()
        ]
        return {
            "status": "stopped"
            if state.completed
            else "cancelled"
            if self.interrupted or isinstance(failure, asyncio.CancelledError)
            else "failed",
            "errors": [{"type": type(item).__name__, "message": str(item)} for item in causes(failure)],
            "selected": asdict(self.prepared.facts),
            "final_weight": float(parameter.detach()),
            "final_gradient": None if parameter.grad is None else float(parameter.grad.detach()),
            "epoch": state.numerical_epochs[self.prepared.model],
            "steps": runtime.steps,
            "unit_outcomes": records,
            "pending": runtime.unfinished,
            "handback_verified": bool(records) and all(item["contribution"] == "granted_ready" for item in records),
            "unusable": sorted(state.unusable),
            "released": not state.running
            and not state.readers
            and not state.writers
            and not state.state_locks
            and not state.active_producers
            and not state.active_retentions,
            "diagnostics": len(state.trace),
        }


def scale(initial: float, trainable: bool):
    model = examples.Scale(initial)
    model.weight.requires_grad_(trainable)
    return model


def functions() -> dict[str, Callable[..., Any]]:
    selected = examples.two_pass().root.work[0].implementation
    return {"scale_v1": scale, "two_pass_v1": selected}


def lower(
    spec: GrantSpec,
    implementations: Mapping[str, Callable[..., Any]],
    *,
    offered: bool = True,
    lowering: Literal["calls", "inline"] = "calls",
    history_limit: int = 256,
) -> PreparedTarget:
    # No authored graph or other frontend's prepared objects enter this target.
    implementations = MappingProxyType(dict(implementations))
    facts = GrantFacts()
    model_name, unit_name = spec.model, spec.unit
    selected = implementations["grant"]

    async def observed(access: Any, models: Any) -> None:
        parameter = models[model_name].get_parameter(spec.parameter)
        facts.entered = True
        facts.initial_weight = float(parameter.detach())
        try:
            result = selected(access, MappingProxyType({"model": models[model_name]}))
            if inspect.isawaitable(result):
                await result
            facts.selected_returned = True
        finally:
            # Selected execution must join its physical work before returning.
            # These observations never certify handback themselves.
            facts.selected_end_weight = float(parameter.detach())
            facts.selected_gradient = None if parameter.grad is None else float(parameter.grad.detach())
            facts.restored = access.restored
            facts.phases = [
                {"phase": phase, "origin": spec.phases[index][0] if index < len(spec.phases) and phase == spec.phases[index][1] else None}
                for index, phase in enumerate(access.phases)
            ]

    base = examples.scale(model_name, spec.initial)
    loader = implementations["model"]

    def verify(concrete: Any) -> None:
        base.verify(concrete)
        parameter = concrete.get_parameter(spec.parameter)
        if (
            parameter.device.type != "cpu"
            or parameter.dtype != torch.float32
            or not parameter.requires_grad
            or not bool(torch.isfinite(parameter))
        ):
            raise ValueError("C scalar target requires finite trainable CPU float32 state")

    participant = replace(base, load=lambda: loader(spec.initial, True), verify=verify)
    filing = graph.Run(
        graph.Block(
            spec.session,
            (
                graph.TwoPass(spec.grant, observed, (), unit_name, "verified-handback", (graph.Use(model_name, "write"),)),
                graph.Advance(spec.advance, "verified-handback", (unit_name,)),
            ),
        ),
        (participant,),
        (graph.Unit(unit_name, ((model_name, spec.parameter),), learning_rate=spec.learning_rate, clip_norm=spec.clip_norm),),
    )
    checked = graph.Contract(two_pass=offered).accept(filing)
    attempt = backend.prepare(checked, lowering=lowering)
    attempt.state.trace = deque(maxlen=history_limit)
    return PreparedTarget(attempt, facts, model_name, unit_name, spec.parameter)
