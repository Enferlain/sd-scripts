"""Independent pair authoring, payload codec and frozen-resource bindings."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import replace

from .author import attributes, build as build_b
from .bridge import TargetRegistry, payloads

_prefix = "docs_design.models-strategy-trainer.xdsl_candidate"
fixture = importlib.import_module(_prefix + ".fixture")
local_contract = importlib.import_module(_prefix + ".local_contract")

COEFFICIENTS = payloads.FrozenResource("1", (1, 2))
USE = local_contract.ResourceUse("pair_coefficients", "read_frozen", COEFFICIENTS.version, COEFFICIENTS.digest)
PAIR = payloads.PayloadCodec(
    "pair-int2-v1", lambda v: type(v) is tuple and len(v) == 2 and all(type(x) is int for x in v), lambda v: list(v)
)


def lower_pack(resources):
    coefficients = resources[USE.symbol].values

    async def invoke(value):
        if value < 0:
            raise payloads.LocalOutcomeError("negative_input", "pair pack rejected negative input")
        await asyncio.sleep(0)
        return tuple(value * c for c in coefficients)

    return invoke


async def needs_lowering(value):
    raise AssertionError("verified resource-aware lowering required")


async def first_lane(value):
    await asyncio.sleep(0)
    return value[0]


def registry(*, opaque: bool = False, effects: list[int] | None = None) -> TargetRegistry:
    bindings = dict(fixture.registry().bindings)
    bindings["pack_pair_v1"] = fixture.Binding(
        "1",
        "map",
        "read_frozen",
        needs_lowering,
        resources=(USE,),
        outcomes=(*payloads.DEFAULT_OUTCOMES, "negative_input"),
        lower=lower_pack,
    )
    bindings["first_lane_v1"] = fixture.Binding("1", "map", "pure", first_lane)
    if opaque:

        async def offset(value):
            if effects is not None:
                effects.append(value)
            await asyncio.sleep(0)
            return value + 1

        bindings["offset_v1"] = replace(bindings["offset_v1"], effect="may_effect", invoke=offset)
    return TargetRegistry(
        "independent-pair-fixture-v1",
        bindings,
        codecs={"record": payloads.INTEGER, "pair-int2": PAIR},
        resources={USE.symbol: COEFFICIENTS},
        signatures={"pack_pair_v1": ("record", "pair-int2"), "first_lane_v1": ("pair-int2", "record")},
    )


def build(*, horizon: int = 2, opaque: bool = False) -> str:
    text = build_b(horizon=horizon)
    pack = f"pair = pairing.pack (offset) [] {attributes(source_id='pair.pack', implementation='pack_pair_v1', effect='read_frozen', resource=USE.symbol, resource_version=USE.version, resource_digest=USE.digest)} : <(wf.record) -> (pairing.pair)>"
    first = f"first = pairing.first (pair) [] {attributes(source_id='pair.first', implementation='first_lane_v1', effect='pure')} : <(pairing.pair) -> (wf.record)>"
    text = text.replace(
        "wf.yield (offset) [] [] : <(wf.record) -> ()>",
        pack + ";\n        " + first + ";\n        wf.yield (first) [] [] : <(wf.record) -> ()>",
    )
    if opaque:
        text = text.replace(
            'implementation: builtin.string "offset_v1", effect: builtin.string "pure"',
            'implementation: builtin.string "offset_v1", effect: builtin.string "may_effect"',
        )
    return text
