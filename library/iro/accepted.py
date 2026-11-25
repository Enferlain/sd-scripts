"""The accepted run and its authority seed (D17).

An accepted run says that a run is meaningful and permitted, and what must be
shown before each use. It holds the accepted strategy in its written form,
evidence of the code it uses, the obligations verification derived, and the
authority seed: the declared participants and relationships, still unbound.
It records the definitions its strategy uses and the implementations it
selects, each by its declaration and its source (D16). The strategy is the source;
the other parts are derived from it and kept for convenience, never as
independent authority. An accepted run is immutable and holds no live
objects.

Only fulfillment produces one on the normal path; tests may build one with
the test-only builder under ``tests/`` (D1). A saved run goes back through the
same checks when it is loaded: :func:`cold_load` re-derives every part from
the written form under the definitions it recorded and refuses a run whose
parts do not correspond, before any effect.
"""

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import Attribute, Operation

from library.iro.composition import Composition, core
from library.iro.diagnostics import Problems, Stage
from library.iro.evidence import CodeEvidence, source_evidence
from library.iro.normalize import Normalized, normalize
from library.iro.operation_contracts import Reproducibility, origin, sym_name, symbol, symbols
from library.iro.seams import Implementation
from library.iro.text import parse, print_strategy
from library.iro.verify import Obligations, Verification, verify
from library.iro.words.model import RelationshipOp, ResourceOp
from library.iro.words.structure import StrategyOp
from library.iro.words.types import RelationshipType, ResourceType


@dataclass(frozen=True)
class DeclaredParticipant:
    """A participant the strategy declares: its name, kind and how it materializes.

    ``reads`` are the participants whose views its materializer reads, as
    ``(name, view)`` pairs; ``settings`` the values its materializer receives
    after them, as Python values.
    """

    name: str
    kind: str
    materializer: str | None = None
    reads: tuple[tuple[str, str], ...] = ()
    origin: str = ""
    settings: tuple[str | int | float | bool, ...] = ()


@dataclass(frozen=True)
class DeclaredRelationship:
    """A relationship the strategy declares between two participants."""

    name: str
    kind: str
    source: str
    target: str
    origin: str = ""


@dataclass(frozen=True)
class AuthoritySeed:
    """The declared participants and relationships, unbound, under the strategy and obligation revisions.

    ``strategy`` is the strategy's revision, a digest of its written form
    without source locations; ``obligations`` the obligation set's revision.
    """

    strategy: str
    obligations: str
    participants: tuple[DeclaredParticipant, ...]
    relationships: tuple[DeclaredRelationship, ...]


@dataclass(frozen=True)
class AcceptedRun:
    """The accepted strategy with the evidence of its code, its obligations and its authority seed.

    ``written`` is the accepted strategy's written form with source
    locations; ``normalized`` its normalized form, saved for inspection.
    ``definitions`` records each definition the strategy uses, and
    ``implementations`` each implementation it selects, by its declaration
    and its source; or by its name alone when none was
    installed at acceptance, which a host refuses (D16). ``contract`` names
    the composition it was accepted under, for reading only.
    ``replicated`` says it was accepted to run on several ranks, which adds
    rank agreement to its obligations.
    """

    written: str
    normalized: str
    contract: str
    definitions: tuple[CodeEvidence, ...]
    implementations: tuple[CodeEvidence, ...]
    obligations: Obligations
    reproducibility: Reproducibility
    seed: AuthoritySeed
    replicated: bool = False

    @property
    def identity(self) -> str:
        """The content identity of the strategy's meaning and of the code it uses.

        Two runs of the same strategy share it; the run's own identity comes
        from the run authority it seeds (D6). It is built from the strategy's
        revision, which leaves out where each statement was authored, so a
        strategy edited only in its source layout, such as a moved line,
        keeps its identity; the written form keeps those locations for
        diagnostics.
        """
        code = "\n".join(record.digest for record in (*self.definitions, *self.implementations))
        listing = f"replicated={self.replicated}\n{code}\n{self.seed.strategy}"
        return hashlib.blake2s(listing.encode(), digest_size=16).hexdigest()


def accept(
    strategy: StrategyOp,
    composition: Composition | None = None,
    *,
    installed: Mapping[str, Implementation] | None = None,
    replicated: bool = False,
) -> AcceptedRun:
    """Normalizes and verifies an accepted strategy and derives its accepted run.

    Fulfillment calls this once it has read the strategy, with the installed
    implementations; problems are reported as a rejection. Every
    implementation the strategy uses must then be installed, and the run
    records the evidence of each one's code (D16).
    """
    contract = composition or core()
    if installed is not None:
        problems = Problems(Stage.FULFILL)
        for name in used_implementations(strategy):
            if name not in installed:
                problems.add("iro-run-003", f"@{name} is used and not installed", fix=_CODE_GONE)
        problems.reject_if_any()
    verification = verify(normalize(strategy), replicated=replicated)
    evidence = {name: implementation_evidence(name, implementation) for name, implementation in (installed or {}).items()}
    return _derive(verification, contract, replicated, evidence)


def implementation_evidence(name: str, implementation: Implementation) -> CodeEvidence:
    """The record of an installed implementation under the name a strategy uses: its declaration and its source (D16)."""
    return CodeEvidence(name, implementation.declaration.text, source_evidence(implementation))


def _derive(verification: Verification, contract: Composition, replicated: bool, evidence: Mapping[str, CodeEvidence]) -> AcceptedRun:
    accepted = verification.normalized.accepted
    written = print_strategy(accepted, contract, locations=True)
    obligations = verification.obligations
    # The strategy's revision is its meaning: the written form without where each statement was authored.
    revision = hashlib.blake2s(print_strategy(accepted, contract).encode(), digest_size=16).hexdigest()
    used = {*used_definitions(accepted, contract), *used_definitions(verification.normalized.strategy, contract)}
    return AcceptedRun(
        written=written,
        normalized=verification.normalized.text(contract, locations=True),
        contract=contract.name,
        definitions=contract.evidence(used),
        implementations=tuple(evidence.get(name, CodeEvidence(name)) for name in used_implementations(accepted)),
        obligations=obligations,
        reproducibility=verification.reproducibility,
        seed=seed(accepted, revision, obligations.revision),
        replicated=replicated,
    )


def used_implementations(strategy: StrategyOp) -> tuple[str, ...]:
    """The implementations the strategy's uses header names, in its order."""
    return tuple(symbols(strategy, "uses"))


def used_definitions(strategy: StrategyOp, contract: Composition) -> set[str]:
    """The names of the contract's definitions a strategy uses: its words and the types and attributes they carry."""
    used: set[str] = set()
    for op in strategy.walk():
        used.add(op.name)
        for attribute in _attributes(op):
            if attribute.name in contract:
                used.add(attribute.name)
    return used & set(contract.definitions)


def _attributes(op: Operation) -> Iterator[Attribute]:
    pending: list[Attribute] = [
        *(v.type for v in op.operands),
        *(r.type for r in op.results),
        *(a.type for region in op.regions for block in region.blocks for a in block.args),
        *op.properties.values(),
        *op.attributes.values(),
    ]
    while pending:
        attribute = pending.pop()
        yield attribute
        if isinstance(attribute, ArrayAttr):
            pending.extend(attribute.data)
        else:
            pending.extend(p for p in getattr(attribute, "parameters", ()) if isinstance(p, Attribute))


def seed(strategy: StrategyOp, strategy_revision: str, obligation_revision: str) -> AuthoritySeed:
    """The authority seed: every declared participant and relationship, unbound."""
    names = {op.handle: sym_name(op) for op in strategy.walk() if isinstance(op, ResourceOp)}
    participants = []
    relationships = []
    for op in strategy.walk():
        if isinstance(op, ResourceOp):
            kind = op.handle.type.kind.data if isinstance(op.handle.type, ResourceType) else ""
            if kind != "participant":
                continue
            views = [v.data for v in op.read_views.data] if op.read_views is not None else []
            participants.append(
                DeclaredParticipant(
                    op.sym_name.data,
                    kind,
                    symbol(op.materialize) if op.materialize is not None else None,
                    tuple((names.get(read) or "", view) for read, view in zip(op.reads, views, strict=True)),
                    origin(op) or "",
                    tuple(_setting(value) for value in op.settings.data) if op.settings is not None else (),
                )
            )
        elif isinstance(op, RelationshipOp):
            relationships.append(
                DeclaredRelationship(
                    op.sym_name.data,
                    op.handle.type.kind.data if isinstance(op.handle.type, RelationshipType) else "",
                    names.get(op.source) or "",
                    names.get(op.target) or "",
                    origin(op) or "",
                )
            )
    return AuthoritySeed(strategy_revision, obligation_revision, tuple(participants), tuple(relationships))


def _setting(value: Attribute) -> str | int | float | bool:
    """A materializer setting as the Python value its implementation receives."""
    if isinstance(value, IntegerAttr) and isinstance(value.type, IntegerType) and value.type.width.data == 1:
        return bool(value.value.data)
    if isinstance(value, IntegerAttr):
        return int(value.value.data)
    if isinstance(value, FloatAttr):
        return float(value.value.data)
    if isinstance(value, StringAttr):
        return value.data
    raise ValueError(f"a materializer setting is a literal, not {value}")


@dataclass(frozen=True)
class Loaded:
    """A cold-loaded run: the accepted strategy re-read and re-verified under the definitions it recorded."""

    run: AcceptedRun
    strategy: StrategyOp
    normalized: Normalized
    verification: Verification


def cold_load(run: AcceptedRun, composition: Composition | None = None, *, installed: Mapping[str, Implementation] | None = None) -> Loaded:
    """Re-derives every part of a saved run and refuses one whose parts do not correspond.

    Every definition the run recorded must be installed with the declaration
    and source it recorded; definitions it does not use may differ. With ``installed``,
    every implementation it recorded must be installed and give the
    evidence it recorded. Reading, normalizing and verifying the written form
    must give the run's normalized form, obligations and authority seed
    exactly.
    """
    contract = composition or core()
    problems = Problems(Stage.FULFILL)
    for recorded in run.definitions:
        if recorded.name not in contract:
            problems.add("iro-run-001", f"the run uses {recorded.name}, which the installed contract does not define", fix=_CODE_GONE)
        elif contract[recorded.name].evidence != recorded:
            problems.add(
                "iro-run-001",
                f"{recorded.name} is installed with {' and '.join(recorded.differences(contract[recorded.name].evidence))} changed since the run recorded it",
                fix=_CODE_GONE,
            )
    problems.reject_if_any()
    if installed is not None:
        check_implementations(run, installed)
    strategy = parse(run.written, contract, source="<accepted run>")
    verification = verify(normalize(strategy), replicated=run.replicated)
    derived = _derive(verification, contract, run.replicated, {record.name: record for record in run.implementations})
    # Implementation evidence comes from the saved run here; check_implementations compares it with installed code.
    for part, saved, found in (
        ("definitions", run.definitions, derived.definitions),
        ("implementations", run.implementations, derived.implementations),
        ("normalized form", run.normalized, derived.normalized),
        ("obligations", run.obligations.revision, derived.obligations.revision),
        ("reproducibility class", run.reproducibility, derived.reproducibility),
        ("authority seed", run.seed, derived.seed),
    ):
        if saved != found:
            problems.add(
                "iro-run-002",
                f"what the run saved as its {part} does not follow from its strategy",
                fix="fulfill the strategy again; a saved run's parts are derived, never edited",
            )
    if derived.written != run.written:
        problems.add("iro-run-002", "the run's written strategy does not print back the same", fix="fulfill the strategy again")
    problems.reject_if_any()
    return Loaded(run, strategy, verification.normalized, verification)


_CODE_GONE = "resume from a revision that has the code the run recorded, or start a new run from its products (D16)"


def check_implementations(run: AcceptedRun, installed: Mapping[str, Implementation]) -> None:
    """A host's check before it runs a run, as the Trainer's is: every implementation the run records is installed and gives the evidence the run recorded (D16, D17).

    A record without a declaration or a source, from a run accepted without
    installed implementations, never passes: no part alone says that two
    pieces are the same code.
    """
    problems = Problems(Stage.FULFILL)
    for recorded in run.implementations:
        found = installed.get(recorded.name)
        if found is None:
            problems.add("iro-run-003", f"@{recorded.name} is used and not installed", fix=_CODE_GONE)
        elif not recorded.declaration or recorded.source is None:
            problems.add(
                "iro-run-003",
                f"the run holds no complete evidence of @{recorded.name}: it was accepted without the implementation installed",
                fix="accept the strategy with its implementations installed",
            )
        else:
            changed = recorded.differences(implementation_evidence(recorded.name, found))
            if changed:
                problems.add(
                    "iro-run-003",
                    f"@{recorded.name} is installed with {' and '.join(changed)} changed since the run recorded it",
                    fix=_CODE_GONE,
                )
    problems.reject_if_any()


__all__ = [
    "AcceptedRun",
    "AuthoritySeed",
    "DeclaredParticipant",
    "DeclaredRelationship",
    "Loaded",
    "accept",
    "check_implementations",
    "cold_load",
    "implementation_evidence",
    "seed",
    "used_definitions",
    "used_implementations",
]
