"""Training's deriver keeps the derivations it was recorded with under its version (D16).

A derivation is part of how the prepared run came to be, as a declaration is,
and what it derives can change while no declaration does. So its outputs on
small fixture strategies are recorded under the deriver's version, as a light
implementation's results are: a change in what it derives under a recorded
version fails here; give the deriver a new version. New versions are
recorded with ``IRO_RECORD_VERSIONS=1``, which only adds entries.
"""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

from library.iro.accepted import cold_load
from library.iro.prepare import Target, prepare
from library.training.units import TrainingDerivation
from library.training.words import contract
from tests.unit.iro.accepted_runs import accepted_run
from tests.unit.training_core.test_units import ALSO_SCORED, SAMPLED, STEP, implementations, on_host, sampling

RECORDED = Path(__file__).parent / "recorded_derivations.json"

FIXTURES: dict[str, Callable[[], tuple[str, dict]]] = {
    "step": lambda: (STEP, implementations({"encoder": [], "network": []})),
    "sampled": lambda: (SAMPLED.replace("JUDGED", ALSO_SCORED), {**sampling(("protected_current lease", "prediction")), **on_host()}),
}


def derived(text: str, installed: dict, deriver: TrainingDerivation) -> str:
    """A digest of what the deriver gave each group: its calls, setups and annotations."""
    run = accepted_run(text, contract(), installed=installed)
    prepared = prepare(cold_load(run, contract(), installed=installed), installed, Target(), [deriver])
    lines = [f"{g.calls} {[(s.name, s.version) for s in g.setups]} {g.annotations}" for g in prepared.groups]
    return hashlib.blake2s("\n".join(lines).encode(), digest_size=8).hexdigest()


def test_no_derivation_changes_under_a_recorded_deriver_version() -> None:
    deriver = TrainingDerivation({"learner": frozenset({"net"})})
    recorded = json.loads(RECORDED.read_text()) if RECORDED.exists() else {}
    versions = recorded.setdefault(deriver.name, {}).setdefault(deriver.version, {})
    changed, unrecorded = [], []
    for name, fixture in FIXTURES.items():
        digest = derived(*fixture(), deriver)
        if name not in versions:
            unrecorded.append(name)
            versions[name] = digest
        elif versions[name] != digest:
            changed.append(name)
    assert not changed, (
        f"@{deriver.name} v{deriver.version} derives differently for {changed}: give it a new version (D16), "
        "or, if a fixture strategy in test_units.py changed, record the fixture again under a new name"
    )
    if unrecorded and os.environ.get("IRO_RECORD_VERSIONS") == "1":
        RECORDED.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
        return
    assert not unrecorded, f"unrecorded derivations, record them with IRO_RECORD_VERSIONS=1: {unrecorded}"
