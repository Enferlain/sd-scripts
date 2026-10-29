"""Tiny Python authoring and separately versioned scalar implementations."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter

from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Region

from .dialects import ComposeOp, EntityOp, FlowOp, FlowType, LocalOp, MapOp, PublishOp, ScoreOp, SelectOp, SessionOp, YieldOp, integer
from .prepare import CONTRACT, Binding, Prepared, Registry
from .runtime import Report, ScalarSession, drain


def build(*, horizon: int = 2) -> ModuleOp:
    record = FlowType(StringAttr("record"))
    entities = [
        EntityOp(properties={"sym_name": StringAttr(name), "source_id": StringAttr("owner." + name), "kind": StringAttr(name)})
        for name in ("cursor", "policy", "publication")
    ]
    flows = [
        FlowOp(properties={"source_id": StringAttr(name)}, result_types=[FlowType(StringAttr(payload))])
        for name, payload in (("selected", "record"), ("mapped", "record"), ("scored", "score"), ("feedback", "feedback"))
    ]
    source, mapped, scored, feedback = [flow.endpoint for flow in flows]
    select = SelectOp(
        operands=[feedback, source],
        properties={
            "source_id": StringAttr("selector"),
            "cursor": SymbolRefAttr("cursor"),
            "policy": SymbolRefAttr("policy"),
            "application": StringAttr("ack_before_next_selection"),
        },
    )
    local = Block(arg_types=[record])
    double = MapOp(
        operands=[local.args[0]],
        result_types=[record],
        properties={
            "source_id": StringAttr("data.double"),
            "implementation": SymbolRefAttr("double_v1"),
            "effect": StringAttr("pure"),
        },
    )
    nested = Block(arg_types=[record])
    offset = MapOp(
        operands=[nested.args[0]],
        result_types=[record],
        properties={
            "source_id": StringAttr("data.offset"),
            "implementation": SymbolRefAttr("offset_v1"),
            "effect": StringAttr("pure"),
        },
    )
    nested.add_ops([offset, YieldOp(operands=[offset.output])])
    group = LocalOp(
        operands=[double.output], result_types=[record], properties={"source_id": StringAttr("data.nested")}, regions=[Region(nested)]
    )
    local.add_ops([double, group, YieldOp(operands=[group.output])])
    compose = ComposeOp(operands=[source, mapped], properties={"source_id": StringAttr("data.composed")}, regions=[Region(local)])
    score = ScoreOp(operands=[mapped, scored], properties={"source_id": StringAttr("scorer"), "implementation": SymbolRefAttr("square_v1")})
    publish = PublishOp(
        operands=[scored, feedback],
        properties={
            "source_id": StringAttr("publisher"),
            "implementation": SymbolRefAttr("receipt_v1"),
            "owner": SymbolRefAttr("publication"),
            "acknowledgement": StringAttr("returned_receipt"),
        },
    )
    return ModuleOp(
        [
            SessionOp(
                properties={
                    "source_id": StringAttr("data-session"),
                    "contract": StringAttr(CONTRACT),
                    "version": integer(1),
                    "horizon": integer(horizon),
                    "stop_rule": StringAttr("settle_selected"),
                },
                regions=[Region(Block([*entities, *flows, select, compose, score, publish]))],
            )
        ]
    )


async def double(value: int) -> int:
    await asyncio.sleep(0)
    return value * 2


async def offset(value: int) -> int:
    await asyncio.sleep(0)
    return value + 1


async def square(value: int) -> int:
    await asyncio.sleep(0)
    return value * value


async def receipt(value: int) -> int:
    await asyncio.sleep(0)
    return value


def registry() -> Registry:
    return Registry(
        "scalar-fixtures-v1",
        {
            "double_v1": Binding("1", "map", "pure", double),
            "offset_v1": Binding("1", "map", "pure", offset),
            "square_v1": Binding("1", "score", "pure", square),
            "receipt_v1": Binding("1", "publish", "publication", receipt),
        },
    )


@dataclass(frozen=True)
class Execution:
    report: Report
    seconds: float


async def execute(prepared: Prepared, *, stop_after: int = 4, diagnostic_limit: int = 32) -> Execution:
    """External driver requests stop; the semantic producer itself is open."""
    if stop_after < 1:
        raise ValueError("stop_after must be positive")
    session = ScalarSession(prepared, diagnostic_limit=diagnostic_limit)
    started = perf_counter()
    running = asyncio.create_task(session.run())
    threshold = asyncio.create_task(session.wait_for(lambda: session.feedback_count >= stop_after))
    try:
        await asyncio.wait((running, threshold), return_when=asyncio.FIRST_COMPLETED)
        session.stop()
        report, cancellation = await drain(running, session.stop)
        if cancellation:
            raise cancellation
    except asyncio.CancelledError as cancellation:
        session.stop()
        running.cancel()
        with suppress(asyncio.CancelledError):
            await drain(running, session.stop)
        # The owned session has physically settled and reported.
        raise cancellation
    finally:
        threshold.cancel()
        await asyncio.gather(threshold, return_exceptions=True)
    return Execution(report, perf_counter() - started)
