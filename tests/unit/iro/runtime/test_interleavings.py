"""The runtime under explored interleavings, checked trace by trace against the reference semantics (D16, 14.2).

The runtime runs each fixture under perturbed schedules: its coordinator
yields at random at every wait and submission, and its selected calls take
random time on the worker threads. For every run, ``refinement.explains``
searches for a schedule of the reference semantics under which every owner
observes exactly the runtime's facts in the runtime's order, with the same
invocations, values and versions. Tampered traces serve as negative
controls: they must not be explained.
"""

import asyncio
import random
from pathlib import Path
import time
from collections import Counter

import pytest

from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.normalize import normalize
from library.iro.prepare import Target, prepare
from library.iro.reference import Reference
from library.iro.runtime import Coordinator, Fact
from library.iro.text import parse
from tests.unit.iro.accepted_runs import accepted_run
from tests.unit.iro.conformance import standins
from tests.unit.iro.conformance.test_conformance import CASES, fixture
from tests.unit.iro.runtime.harness import RUNTIME_ONLY, Memory, comparable, implementation
from tests.unit.iro.runtime.refinement import explains

SEMANTIC = frozenset(
    {"call_completed", "call_failed", "published", "nothing_produced", "admission", "disposition", "updated", "outcome"}
    | {"request_pending", "request_settled", "request_disposed", "stop_requested", "product_released"}
)


def summary(facts) -> frozenset:
    """What owners observe, without identities: a multiset of owner, kind and values."""
    kept = Counter()
    for fact in facts:
        if fact.kind in SEMANTIC and fact.kind not in RUNTIME_ONLY:
            owner, _, kind, values = comparable(fact)
            kept[(owner, kind, values)] += 1
    return frozenset(kept.items())


def runtime_facts(name: str, seed: int, checked: bool = True) -> list[Fact]:
    resources, made = CASES[name]
    rng = random.Random(seed)
    installed = {}
    for key, standin in made().items():
        declared = implementation(key, standin)

        def jittered(*args, _run=declared.function, _delay=rng.random() * 0.002):
            time.sleep(_delay)
            return _run(*args)

        installed[key] = type(declared)(declared.declaration, jittered)
    run = accepted_run(fixture(name), installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, Target(workers=3))
    owner = Memory(dict(resources))
    facts: list[Fact] = []
    coordinator = Coordinator(
        prepared,
        dict.fromkeys(resources, owner),
        installed,
        {"host": Host()},
        observe=facts.append,
        explore=random.Random(seed),
        checked=checked,
    )
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    return facts


def make_reference(name: str):
    resources, made = CASES[name]
    normalized = normalize(parse(fixture(name)))
    return lambda: Reference(normalized, resources, made(), validators=standins.validators())


def check(name: str, seeds: range, keep: Path, checked: bool = True) -> None:
    make = make_reference(name)
    for seed in seeds:
        facts = runtime_facts(name, seed, checked)
        verdict = explains(facts, make)
        if not verdict.explained:
            saved = keep / f"{name}-{seed}.facts"
            saved.write_text("\n".join(repr(f) for f in facts))
            raise AssertionError(f"{name}, seed {seed}: {verdict.detail} after {verdict.replays} replays; trace saved to {saved}")


@pytest.mark.parametrize("checked", [True, False], ids=["checked", "unchecked"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_every_runtime_trace_is_a_trace_of_the_reference(name: str, checked: bool, tmp_path: Path) -> None:
    check(name, range(12), tmp_path, checked)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CASES))
def test_every_runtime_trace_under_many_schedules_is_a_trace_of_the_reference(name: str, tmp_path: Path) -> None:
    # Half the schedules with every narrowed check in full, half as a run executes by default.
    check(name, range(12, 112), tmp_path, checked=True)
    check(name, range(112, 212), tmp_path, checked=False)


def _tampered(facts, how: str):
    facts = list(facts)
    owner_facts = [i for i, f in enumerate(facts) if f.owner == facts[-1].owner or f.kind == "leased"]
    if how == "swap":
        i, j = [k for k, f in enumerate(facts) if f.kind == "call_completed"][:2]
        facts[i], facts[j] = facts[j], facts[i]
    elif how == "version":
        k = next(k for k, f in enumerate(facts) if f.kind == "leased")
        name, mode, version = facts[k].values
        facts[k] = Fact(
            facts[k].owner,
            facts[k].invocation,
            "leased",
            (name, mode, type(version)(version.owner, version.incarnation, version.lineage, version.ordinal + 7)),
        )
    elif how == "drop":
        del facts[next(k for k, f in enumerate(facts) if f.kind == "disposition")]
    elif how == "extra":
        k = next(k for k, f in enumerate(facts) if f.kind == "call_completed")
        facts.insert(k, facts[k])
    assert owner_facts
    return facts


@pytest.mark.parametrize("how", ["swap", "version", "drop", "extra"])
def test_a_tampered_trace_is_not_explained(how: str) -> None:
    facts = runtime_facts("lifetime", 0)
    verdict = explains(_tampered(facts, how), make_reference("lifetime"), max_replays=200)
    assert not verdict.explained


def test_the_perturbation_reaches_more_than_one_outcome_where_the_schedule_matters() -> None:
    assert len({summary(runtime_facts("data_only", seed)) for seed in range(8)}) > 1
