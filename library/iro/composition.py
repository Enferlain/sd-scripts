"""The contract: the composed, versioned vocabulary a strategy may use (D18, D21).

A composition holds definitions: each word's operation or type, the area of
the vocabulary it belongs to and its definition version. It is built by
explicit composition (D2). The core vocabulary is one composition; a domain
vocabulary, training first, extends it with its own definitions. Nothing
registers itself on import.

The contract's version identifies exactly which definitions at which versions
it holds, so an accepted run can pin it.
"""

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.ir import Attribute, Dialect, Operation

AREAS = ("structure", "lifetimes", "data", "access", "calls", "model")


@dataclass(frozen=True)
class Definition:
    """One contract entry: a word's operation or type, its area and its definition version.

    ``form`` is how the word prints and parses, for words outside the core
    vocabulary, whose forms the written-form module holds itself.
    """

    definition: type[Operation] | type[Attribute]
    area: str
    version: int = 1
    form: object | None = None

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def meaning(self) -> str:
        doc = (self.definition.__doc__ or "").strip()
        return doc.splitlines()[0] if doc else ""

    @property
    def is_operation(self) -> bool:
        return issubclass(self.definition, Operation)


class Composition:
    """An immutable set of definitions with queries, a version and a readable printout."""

    def __init__(self, definitions: Iterable[Definition], *, name: str = "kuroiro"):
        entries: dict[str, Definition] = {}
        for entry in definitions:
            if entry.version < 1:
                raise ValueError(f"{entry.name}: definition versions start at 1")
            if entry.name in entries:
                raise ValueError(f"{entry.name} is defined twice")
            entries[entry.name] = entry
        self.name = name
        self._entries = MappingProxyType(entries)

    @property
    def definitions(self) -> Mapping[str, Definition]:
        return self._entries

    @property
    def version(self) -> str:
        """A digest of every definition's name, area and version."""
        listing = "\n".join(f"{e.name} {e.area} {e.version}" for e in sorted(self._entries.values(), key=lambda e: e.name))
        return hashlib.blake2s(listing.encode(), digest_size=8).hexdigest()

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> Definition:
        return self._entries[name]

    def operation(self, name: str) -> type[Operation] | None:
        entry = self._entries.get(name)
        found = entry.definition if entry is not None else None
        return found if isinstance(found, type) and issubclass(found, Operation) else None

    def words(self, area: str | None = None) -> tuple[Definition, ...]:
        return tuple(e for e in self._entries.values() if e.is_operation and (area is None or e.area == area))

    def extend(self, definitions: Iterable[Definition], *, name: str | None = None) -> Composition:
        """A new composition with further definitions, such as a domain vocabulary."""
        return Composition([*self._entries.values(), *definitions], name=name or self.name)

    def context(self) -> Context:
        """An xDSL context that knows exactly these definitions."""
        context = Context()
        context.load_dialect(Builtin)
        defined = [e.definition for e in self._entries.values()]
        operations = [d for d in defined if issubclass(d, Operation)]
        attributes = [d for d in defined if issubclass(d, Attribute)]
        context.load_dialect(Dialect("iro", operations, attributes))
        return context

    def printout(self) -> str:
        """The readable contract: every word by area, with its version and meaning."""
        lines = [f"# {self.name} contract {self.version}"]
        areas = list(AREAS) + sorted({e.area for e in self._entries.values()} - set(AREAS))
        for area in areas:
            words = sorted((e for e in self.words(area)), key=lambda e: e.name)
            if not words:
                continue
            lines += ["", f"## {area}", ""]
            width = max(len(e.name) for e in words)
            lines += [f"{e.name.ljust(width)}  v{e.version}  {e.meaning}" for e in words]
        return "\n".join(lines) + "\n"


def core() -> Composition:
    """The core vocabulary: Kuroiro's own words and types."""
    from library.iro.operation_contracts import ResourceUseAttr
    from library.iro.words import access, calls, data, lifetimes, model, structure
    from library.iro.words.types import TYPES

    areas = {
        "structure": structure.OPERATIONS,
        "lifetimes": lifetimes.OPERATIONS,
        "data": data.OPERATIONS,
        "access": access.OPERATIONS,
        "calls": calls.OPERATIONS,
        "model": model.OPERATIONS,
    }
    entries = [Definition(op, area) for area, ops in areas.items() for op in ops]
    entries += [Definition(t, "types") for t in (*TYPES, ResourceUseAttr)]
    return Composition(entries)


__all__ = ["AREAS", "Composition", "Definition", "core"]
