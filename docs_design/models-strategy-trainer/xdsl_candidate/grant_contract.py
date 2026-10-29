"""C's offered phase contract and author, independent of numerical realization."""

from __future__ import annotations

import math

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, ModuleOp, StringAttr, SymbolRefAttr, f64
from xdsl.ir import Block, Region

from .dialects import EntityOp, SessionOp, integer
from .grant_dialect import AdvanceGrantedOp, ContributionType, GrantOp, PhaseOp
from .model_contract import one
from .model_dialect import ModelOp
from .prepare import Accepted, ContractExtension, accept, ref, require

CONTRACT_C = "granted-model-v1"
PROFILE = "restored-two-pass-v1"
PHASES = ("backward", "temporary_edit", "intermediate_reset", "backward", "restore")
DELEGATED = ("backward", "temporary_edit", "intermediate_reset", "restore")
RETAINED = ("clipping", "final_advancement", "final_reset")


def verify_grant(session: SessionOp) -> None:
    require(session.horizon is None, session, "model-only contract has no flow horizon")
    require(session.stop_rule.data == "drain_entered_preserve_uncertain", session, "unknown grant failure disposition")
    children = tuple(session.body.block.ops)
    require(
        all(isinstance(op, (EntityOp, ModelOp, GrantOp, AdvanceGrantedOp)) for op in children), session, "unknown grant-scope operation"
    )
    entities = [op for op in children if isinstance(op, EntityOp)]
    symbols = {op.sym_name.data: op.kind.data for op in entities}
    require(
        len(entities) == len(symbols) == 2 and set(symbols.values()) == {"participant", "unit"} and all(symbols),
        session,
        "one participant and one unit identity required",
    )
    model, grant, advance = one(session, ModelOp), one(session, GrantOp), one(session, AdvanceGrantedOp)
    require(symbols.get(ref(model.entity)) == "participant" and model.treatment.data == "trainable", model, "invalid training subject")
    require(
        math.isfinite(model.initial.value.data) and bool(ref(model.implementation)), model, "invalid model implementation or initial state"
    )
    require(ref(grant.model) == ref(model.entity) and symbols.get(ref(grant.unit)) == "unit", grant, "invalid grant subject/unit")
    require(grant.profile.data == PROFILE and bool(ref(grant.implementation)), grant, "unprovided semantic grant profile")
    require(
        tuple(item.data for item in grant.delegated) == DELEGATED and tuple(item.data for item in grant.retained) == RETAINED,
        grant,
        "illegal delegated/retained authority split",
    )
    phases = tuple(grant.body.block.ops)
    require(not grant.body.block.args and all(isinstance(op, PhaseOp) for op in phases), grant, "grant body is a closed phase contract")
    require(tuple(op.phase.data for op in phases if isinstance(op, PhaseOp)) == PHASES, grant, "incomplete or reordered phase protocol")
    require(
        isinstance(grant.handback.type, ContributionType) and grant.handback.type.unit == grant.unit, grant, "handback unit type differs"
    )
    require(
        advance.handback is grant.handback and advance.unit == grant.unit and len(tuple(grant.handback.uses)) == 1,
        advance,
        "advancement requires this grant's unique typed handback",
    )
    require(
        bool(advance.parameter.data) and math.isfinite(advance.learning_rate.value.data) and advance.learning_rate.value.data > 0,
        advance,
        "invalid parameter/learning rate",
    )
    if advance.clip_norm is not None:
        require(
            math.isfinite(advance.clip_norm.value.data) and advance.clip_norm.value.data > 0, advance, "invalid retained clipping bound"
        )


EXTENSION_C = ContractExtension(
    CONTRACT_C,
    1,
    verify_grant,
    (
        "match an offered concrete restored two-pass authority profile",
        "resolve versioned numerical implementations and parameter correspondence",
        "verify actual phase history, restoration and final gradient before handback",
        "standard owner alone consumes verified handback for final advancement/reset",
        "drain entered uses and preserve uncertainty on failure or cancellation",
    ),
)


def accept_grant(module: ModuleOp) -> Accepted:
    return accept(module, extension=EXTENSION_C)


def build_grant(*, initial: float = 0.8, learning_rate: float = 0.05, clip_norm: float | None = None) -> ModuleOp:
    entities = [
        EntityOp(properties={"sym_name": StringAttr(name), "source_id": StringAttr("owner." + name), "kind": StringAttr(kind)})
        for name, kind in (("model", "participant"), ("weights", "unit"))
    ]
    model = ModelOp(
        properties={
            "source_id": StringAttr("model.binding"),
            "entity": SymbolRefAttr("model"),
            "implementation": SymbolRefAttr("scale_v1"),
            "initial": FloatAttr(initial, f64),
            "treatment": StringAttr("trainable"),
        }
    )
    phases = [
        PhaseOp(properties={"source_id": StringAttr(f"phase.{index}.{phase}"), "phase": StringAttr(phase)})
        for index, phase in enumerate(PHASES)
    ]
    grant = GrantOp(
        properties={
            "source_id": StringAttr("selected-two-pass"),
            "model": SymbolRefAttr("model"),
            "unit": SymbolRefAttr("weights"),
            "implementation": SymbolRefAttr("two_pass_v1"),
            "profile": StringAttr(PROFILE),
            "delegated": ArrayAttr(StringAttr(name) for name in DELEGATED),
            "retained": ArrayAttr(StringAttr(name) for name in RETAINED),
        },
        result_types=[ContributionType(SymbolRefAttr("weights"))],
        regions=[Region(Block(phases))],
    )
    advance = AdvanceGrantedOp(
        operands=[grant.handback],
        properties={
            "source_id": StringAttr("retained-final-step"),
            "unit": SymbolRefAttr("weights"),
            "parameter": StringAttr("weight"),
            "learning_rate": FloatAttr(learning_rate, f64),
            "clip_norm": FloatAttr(clip_norm, f64) if clip_norm is not None else None,
        },
    )
    session = SessionOp(
        properties={
            "source_id": StringAttr("grant-session"),
            "contract": StringAttr(CONTRACT_C),
            "version": integer(1),
            "stop_rule": StringAttr("drain_entered_preserve_uncertain"),
        },
        regions=[Region(Block([*entities, model, grant, advance]))],
    )
    return ModuleOp([session])
