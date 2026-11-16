"""Training's definition and implementation versions keep the declarations they were recorded with (D16).

Training records its own vocabulary and implementations beside the IR's
record, so the IR's tests never depend on a domain. A declaration that changes
under a recorded version fails here: give it a new version. New versions are
recorded with ``IRO_RECORD_VERSIONS=1``, which only adds entries.
"""

import importlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

from library.iro.frontend import Declared
from library.training.words import definitions
from tests.unit.iro.test_versions import compare

RECORDED = Path(__file__).parent / "recorded_versions.json"

IMPLEMENTATION_MODULES = ("library.training.units.standard",)


def _definitions() -> Iterator[tuple[str, str, str]]:
    for entry in definitions():
        yield entry.name, str(entry.version), entry.digest


def _implementations() -> Iterator[tuple[str, str, str]]:
    for module in IMPLEMENTATION_MODULES:
        for value in vars(importlib.import_module(module)).values():
            if isinstance(value, Declared):
                yield value.declaration.name, value.declaration.version, value.declaration.digest


def test_no_training_declaration_changes_under_a_recorded_version() -> None:
    recorded = json.loads(RECORDED.read_text()) if RECORDED.exists() else {}
    changed, unrecorded = compare(recorded, {"definitions": list(_definitions()), "implementations": list(_implementations())})
    assert not changed, f"declarations changed without a new version (D16): {changed}"
    if unrecorded and os.environ.get("IRO_RECORD_VERSIONS") == "1":
        RECORDED.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
        return
    assert not unrecorded, f"unrecorded versions, record them with IRO_RECORD_VERSIONS=1: {unrecorded}"
