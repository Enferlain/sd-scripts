"""The accepted run and its authority seed (D17).

An accepted run says that a run is meaningful and permitted, and what must be
shown before each use. It holds the accepted strategy in its written form,
the contract and implementation versions it pins, the obligations
verification derived under them, and the authority seed: the declared
participants and relationships, still unbound. The strategy is the source;
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
from dataclasses import dataclass

from library.iro.composition import Composition, core
from library.iro.diagnostics import Problems, Stage
from library.iro.normalize import Normalized, normalize
from library.iro.operation_contracts import Reproducibility, origin, sym_name, symbol, symbols
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
    ``replicated`` says it was accepted to run on several ranks, which adds
    rank agreement to its obligations.
    """

    written: str
    normalized: str
    contract: str
    contract_version: str
    implementations: tuple[tuple[str, str], ...]
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
        pinned = "\n".join(f"{name} v{version}" for name, version in self.implementations)
        listing = f"{self.contract} {self.contract_version} replicated={self.replicated}\n{pinned}\n{self.written}"
        return hashlib.blake2s(listing.encode(), digest_size=16).hexdigest()


def accept(strategy: StrategyOp, composition: Composition | None = None, *, replicated: bool = False) -> AcceptedRun:
    """Normalizes and verifies an accepted strategy and derives its accepted run.

    Fulfillment calls this once it has read the strategy; problems are
    reported as a rejection.
    """
    contract = composition or core()
    verification = verify(normalize(strategy), replicated=replicated)
    return _derive(verification, contract, replicated)


def _derive(verification: Verification, contract: Composition, replicated: bool) -> AcceptedRun:
    accepted = verification.normalized.accepted
    written = print_strategy(accepted, contract, locations=True)
    obligations = verification.obligations
    revision = hashlib.blake2s(written.encode(), digest_size=16).hexdigest()
    return AcceptedRun(
        written=written,
        normalized=verification.normalized.text(contract, locations=True),
        contract=contract.name,
        contract_version=contract.version,
        implementations=pinned(accepted),
        obligations=obligations,
        reproducibility=verification.reproducibility,
        seed=seed(accepted, revision, obligations.revision),
        replicated=replicated,
    )


def pinned(strategy: StrategyOp) -> tuple[tuple[str, str], ...]:
    """The implementations the strategy's uses header pins, with their versions."""
    versions = [v.data for v in strategy.use_versions or ()]
    return tuple(zip(symbols(strategy, "uses"), versions, strict=True))


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


def cold_load(run: AcceptedRun, composition: Composition | None = None) -> Loaded:
    """Re-derives every part of a saved run and refuses one whose parts do not correspond.

    The installed contract must be the one the run pins. Reading, normalizing
    and verifying the written form must give the run's normalized form,
    obligations and authority seed exactly.
    """
    contract = composition or core()
    problems = Problems(Stage.FULFILL)
    if (run.contract, run.contract_version) != (contract.name, contract.version):
        problems.add(
            "iro-run-001",
            f"the run pins contract {run.contract} {run.contract_version}, and {contract.name} {contract.version} is installed",
            fix="install the contract version the run pins, or fulfill the strategy again",
        )
        problems.reject_if_any()
    strategy = parse(run.written, contract, source="<accepted run>")
    verification = verify(normalize(strategy), replicated=run.replicated)
    derived = _derive(verification, contract, run.replicated)
    for part, saved, found in (
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


def implementation_versions(run: AcceptedRun) -> dict[str, str]:
    """What the Trainer checks its installed implementations against."""
    return dict(run.implementations)


__all__ = [
    "AcceptedRun",
    "AuthoritySeed",
    "DeclaredParticipant",
    "DeclaredRelationship",
    "Loaded",
    "accept",
    "cold_load",
    "implementation_versions",
    "pinned",
    "seed",
]
