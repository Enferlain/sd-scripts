"""Every definition and implementation version keeps the declaration it was recorded with (D16).

``recorded_versions.json`` holds the declaration digest of each definition
version and implementation version in the repository. A declaration that
changes under a recorded version fails here: give it a new version. A new
version is recorded with ``IRO_RECORD_VERSIONS=1``, which only adds entries,
so it can never re-record a changed declaration under an old version.

A section that adds a vocabulary or implementations lists them below.
"""

import importlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

from library.iro.composition import Composition, core
from library.iro.frontend import Declared

RECORDED = Path(__file__).parent / "recorded_versions.json"

VOCABULARIES: tuple[Callable[[], Composition], ...] = (core,)
IMPLEMENTATION_MODULES: tuple[str, ...] = ()


def _definitions() -> Iterator[tuple[str, str, str]]:
    for vocabulary in VOCABULARIES:
        for entry in vocabulary().definitions.values():
            yield entry.name, str(entry.version), entry.digest


def _implementations() -> Iterator[tuple[str, str, str]]:
    for module in IMPLEMENTATION_MODULES:
        for value in vars(importlib.import_module(module)).values():
            if isinstance(value, Declared):
                yield value.declaration.name, value.declaration.version, value.declaration.digest


def compare(recorded: dict, current: dict[str, list[tuple[str, str, str]]]) -> tuple[list[str], list[str]]:
    """Changed and unrecorded versions; unrecorded ones are added to ``recorded``."""
    changed, unrecorded = [], []
    for kind, entries in current.items():
        table = recorded.setdefault(kind, {})
        for name, version, digest in entries:
            known = table.setdefault(name, {})
            if version not in known:
                unrecorded.append(f"{kind[:-1]} {name} v{version}")
                known[version] = digest
            elif known[version] != digest:
                changed.append(f"{kind[:-1]} {name} v{version}")
    return changed, unrecorded


def test_no_declaration_changes_under_a_recorded_version() -> None:
    recorded = json.loads(RECORDED.read_text()) if RECORDED.exists() else {}
    changed, unrecorded = compare(recorded, {"definitions": list(_definitions()), "implementations": list(_implementations())})
    assert not changed, (
        f"declarations changed without a new version (D16): {changed}. "
        "If every definition changed at once, an xDSL upgrade changed how definitions print."
    )
    if unrecorded and os.environ.get("IRO_RECORD_VERSIONS") == "1":
        RECORDED.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
        return
    assert not unrecorded, f"unrecorded versions, record them with IRO_RECORD_VERSIONS=1: {unrecorded}"


def test_a_changed_declaration_under_a_recorded_version_is_reported_and_a_new_version_is_added() -> None:
    recorded = {"implementations": {"denoise": {"1": "aaaa"}}}
    changed, unrecorded = compare(recorded, {"implementations": [("denoise", "1", "bbbb"), ("denoise", "2", "bbbb")]})
    assert changed == ["implementation denoise v1"] and unrecorded == ["implementation denoise v2"]
    assert recorded["implementations"]["denoise"] == {"1": "aaaa", "2": "bbbb"}, "recording never rewrites an old version"
