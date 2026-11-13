"""The accepted run and its authority seed (D17).

An accepted run says that a run is meaningful and permitted, and what must be
shown before each use. It holds the accepted strategy in its written form,
what it pins, the obligations verification derived under those pins, and the
authority seed: the declared participants and relationships, still unbound.
It pins the definitions its strategy uses and the implementations it
selects, each with its version and a digest of its declaration (D16). The strategy is the source;
the other parts are derived from it and kept for convenience, never as
independent authority. An accepted run is immutable and holds no live
objects.

Only fulfillment produces one on the normal path; tests may build one with
the test-only builder under ``tests/`` (D1). A saved run goes back through the
same checks when it is loaded: :func:`cold_load` re-derives every part from
the written form under the pinned versions and refuses a run whose parts do
not correspond, before any effect.
"""

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr
from xdsl.ir import Attribute, Operation

from library.iro.composition import Composition, core
from library.iro.diagnostics import Problems, Stage
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
    ``(name, view)`` pairs.
    """

    name: str
    kind: str
    materializer: str | None = None
    reads: tuple[tuple[str, str], ...] = ()
    origin: str = ""


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
    """The declared participants and relationships, unbound, under the strategy and obligation revisions."""

    strategy: str
    obligations: str
    participants: tuple[DeclaredParticipant, ...]
    relationships: tuple[DeclaredRelationship, ...]


@dataclass(frozen=True)
class AcceptedRun:
    """The accepted strategy with its pinned versions, obligations and authority seed.

    ``written`` is the accepted strategy's written form with source
    locations; ``normalized`` its normalized form, saved for inspection.
    ``definitions`` pins each definition the strategy uses as name,
    definition version and digest; ``implementations`` pins each selected
    implementation as name, implementation version and declaration digest,
    with an empty digest when none was installed at acceptance, which the
    Trainer refuses. ``contract`` names the composition it was accepted
    under, for reading only. ``replicated`` says it was accepted to run on
    several ranks, which adds rank agreement to its obligations.
    """

    written: str
    normalized: str
    contract: str
    definitions: tuple[tuple[str, int, str], ...]
    implementations: tuple[tuple[str, str, str], ...]
    obligations: Obligations
    reproducibility: Reproducibility
    seed: AuthoritySeed
    replicated: bool = False

    @property
    def identity(self) -> str:
        """The content identity of the strategy and the versions it pins.

        Two runs of the same strategy share it; the run's own identity comes
        from the run authority it seeds (D6).
        """
        pinned = "\n".join(f"{name} v{version} {digest}" for name, version, digest in (*self.definitions, *self.implementations))
        listing = f"replicated={self.replicated}\n{pinned}\n{self.written}"
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
    implementation the strategy pins must then be installed at its pinned
    version, and the pins record the installed declarations' digests.
    """
    contract = composition or core()
    if installed is not None:
        problems = Problems(Stage.FULFILL)
        for name, version in pinned(strategy):
            _check_installed(problems, name, version, None, installed)
        problems.reject_if_any()
    verification = verify(normalize(strategy), replicated=replicated)
    digests = {name: implementation.declaration.digest for name, implementation in (installed or {}).items()}
    return _derive(verification, contract, replicated, digests)


def _derive(verification: Verification, contract: Composition, replicated: bool, digests: Mapping[str, str]) -> AcceptedRun:
    accepted = verification.normalized.accepted
    written = print_strategy(accepted, contract, locations=True)
    obligations = verification.obligations
    revision = hashlib.blake2s(written.encode(), digest_size=16).hexdigest()
    used = {*used_definitions(accepted, contract), *used_definitions(verification.normalized.strategy, contract)}
    return AcceptedRun(
        written=written,
        normalized=verification.normalized.text(contract, locations=True),
        contract=contract.name,
        definitions=contract.pins(used),
        implementations=tuple((name, version, digests.get(name, "")) for name, version in pinned(accepted)),
        obligations=obligations,
        reproducibility=verification.reproducibility,
        seed=seed(accepted, revision, obligations.revision),
        replicated=replicated,
    )


def pinned(strategy: StrategyOp) -> tuple[tuple[str, str], ...]:
    """The implementations the strategy's uses header pins, with their implementation versions."""
    versions = [v.data for v in strategy.use_versions or ()]
    return tuple(zip(symbols(strategy, "uses"), versions, strict=True))


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


@dataclass(frozen=True)
class Loaded:
    """A cold-loaded run: the accepted strategy re-read and re-verified under its pinned versions."""

    run: AcceptedRun
    strategy: StrategyOp
    normalized: Normalized
    verification: Verification


def cold_load(run: AcceptedRun, composition: Composition | None = None, *, installed: Mapping[str, Implementation] | None = None) -> Loaded:
    """Re-derives every part of a saved run and refuses one whose parts do not correspond.

    Every definition the run pins must be installed at its definition version
    with the same declaration; definitions it does not use may differ. With
    ``installed``, so must every implementation it pins. Reading, normalizing
    and verifying the written form must give the run's normalized form,
    obligations and authority seed exactly.
    """
    contract = composition or core()
    problems = Problems(Stage.FULFILL)
    for name, version, digest in run.definitions:
        if name not in contract:
            problems.add("iro-run-001", f"the run uses {name}, which the installed contract does not define", fix=_PINS_GONE)
        elif contract[name].version != version:
            problems.add("iro-run-001", f"the run pins {name} v{version}, and v{contract[name].version} is installed", fix=_PINS_GONE)
        elif contract[name].digest != digest:
            problems.add(
                "iro-run-001",
                f"{name} v{version} is installed with a declaration that differs from the one the run pinned",
                fix="a definition whose declaration changes needs a new definition version (D16)",
            )
    problems.reject_if_any()
    if installed is not None:
        check_implementations(run, installed)
    strategy = parse(run.written, contract, source="<accepted run>")
    verification = verify(normalize(strategy), replicated=run.replicated)
    derived = _derive(verification, contract, run.replicated, {name: digest for name, _, digest in run.implementations})
    # Implementation digests come from the saved run here; check_implementations compares them with installed code.
    for part, saved, found in (
        ("pinned definitions", run.definitions, derived.definitions),
        ("pinned implementations", run.implementations, derived.implementations),
        ("normalized form", run.normalized, derived.normalized),
        ("obligations", run.obligations.revision, derived.obligations.revision),
        ("reproducibility class", run.reproducibility, derived.reproducibility),
        ("authority seed", run.seed, derived.seed),
    ):
        if saved != found:
            problems.add(
                "iro-run-002",
                f"the run's {part} does not follow from its strategy under the pinned versions",
                fix="fulfill the strategy again; a saved run's parts are derived, never edited",
            )
    if derived.written != run.written:
        problems.add("iro-run-002", "the run's written strategy does not print back the same", fix="fulfill the strategy again")
    problems.reject_if_any()
    return Loaded(run, strategy, verification.normalized, verification)


_PINS_GONE = "resume from a revision that has what the run pins, or start a new run from its products (D16)"


def _check_installed(problems: Problems, name: str, version: str, digest: str | None, installed: Mapping[str, Implementation]) -> None:
    found = installed.get(name)
    if found is None:
        problems.add("iro-run-003", f"@{name} v{version} is pinned and not installed", fix=_PINS_GONE)
    elif found.declaration.version != version:
        problems.add("iro-run-003", f"@{name} v{version} is pinned, and v{found.declaration.version} is installed", fix=_PINS_GONE)
    elif digest is not None and found.declaration.digest != digest:
        problems.add(
            "iro-run-003",
            f"@{name} v{version} is installed with a declaration that differs from the one the run pinned",
            fix="an implementation whose declaration changes needs a new implementation version (D16)",
        )


def check_implementations(run: AcceptedRun, installed: Mapping[str, Implementation]) -> None:
    """The Trainer's check: every pinned implementation is installed at its version and declaration (D17).

    A pin without a digest, from a run accepted without installed
    implementations, never passes.
    """
    problems = Problems(Stage.FULFILL)
    for name, version, digest in run.implementations:
        _check_installed(problems, name, version, digest, installed)
    problems.reject_if_any()


def implementation_versions(run: AcceptedRun) -> dict[str, str]:
    """Each pinned implementation's implementation version."""
    return {name: version for name, version, _ in run.implementations}


__all__ = [
    "AcceptedRun",
    "AuthoritySeed",
    "DeclaredParticipant",
    "DeclaredRelationship",
    "Loaded",
    "accept",
    "check_implementations",
    "cold_load",
    "implementation_versions",
    "pinned",
    "seed",
    "used_definitions",
]
