"""Training's deriver keeps deriving what was recorded for it on fixture strategies (D16).

A derivation is part of how the prepared run came to be, as a declaration is,
and what it derives can change while no declaration does. So its outputs on
small fixture strategies are recorded, as a light implementation's fixture
outputs are: a change in what it derives fails here, and the developer
confirms the change is meant by recording the new outputs with
``IRO_RECORD_FIXTURES=1``.
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
    lines = [f"{g.calls} {[s.name for s in g.setups]} {g.annotations}" for g in prepared.groups]
    return hashlib.blake2s("\n".join(lines).encode(), digest_size=8).hexdigest()


def test_the_deriver_derives_what_was_recorded_on_its_fixture_strategies() -> None:
    deriver = TrainingDerivation({"learner": frozenset({"net"})})
    recorded = json.loads(RECORDED.read_text()) if RECORDED.exists() else {}
    outputs = recorded.setdefault(deriver.name, {})
    found = {name: derived(*fixture(), deriver) for name, fixture in FIXTURES.items()}
    changed = sorted(name for name in found if name in outputs and outputs[name] != found[name])
    unrecorded = sorted(name for name in found if name not in outputs)
    if (changed or unrecorded) and os.environ.get("IRO_RECORD_FIXTURES") == "1":
        outputs |= found
        RECORDED.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
        return
    assert not changed, (
        f"@{deriver.name} derives differently for {changed} (D16): if the change is meant, record it with IRO_RECORD_FIXTURES=1; "
        "if a fixture strategy in test_units.py changed, record that fixture again under a new name"
    )
    assert not unrecorded, f"unrecorded derivations, record them with IRO_RECORD_FIXTURES=1: {unrecorded}"
