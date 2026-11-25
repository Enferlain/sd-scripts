"""The contract: the composed vocabulary a strategy may use (D18, D21).

A composition holds definitions: each word's operation or type and the area
of the vocabulary it belongs to. It is built by
explicit composition (D2). The core vocabulary is one composition; a domain
vocabulary, training first, extends it with its own definitions. Nothing
registers itself on import.

An accepted run records evidence of only the definitions its strategy uses,
each one's declaration and source (D16), so a definition added elsewhere does not stop a
saved run from loading. The composition's digest identifies the whole
vocabulary, for the printout.
"""

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.ir import Attribute, Dialect, Operation, ParametrizedAttribute
from xdsl.irdl import IRDLOperation

from library.iro.evidence import CodeEvidence, source_evidence

AREAS = ("structure", "lifetimes", "data", "access", "calls", "model")


@dataclass(frozen=True)
class Definition:
    """One contract entry: a word's operation or type and its area.

    ``form`` is how the word prints and parses, for words outside the core
    vocabulary, whose forms the written-form module holds itself.
    """

    definition: type[Operation] | type[Attribute]
    area: str
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

    @property
    def text(self) -> str:
        """The definition's canonical declaration, which a run's evidence of the word records (D16).

        For an operation it covers the operands, results, properties, regions
        and traits; for a type its parameters; and the written form's code,
        by name, for a word outside the core vocabulary. A change in how a
        word prints or parses, or in a derived word's expansion, is not in
        it: cold loading derives a saved run again and refuses one whose
        forms differ, and the conformance fixtures show it in tests. The text
        comes from xDSL's definition objects, so an xDSL upgrade that prints
        them differently changes every definition's text, and saved runs
        refuse to load until they are fulfilled again.
        """
        if self.is_operation:
            irdl = cast(type[IRDLOperation], self.definition).get_irdl_definition()
            parts = [
                f"operands {[(n, repr(d)) for n, d in irdl.operands]}",
                f"results {[(n, repr(d)) for n, d in irdl.results]}",
                f"properties {sorted((n, repr(d)) for n, d in irdl.properties.items())}",
                f"attributes {sorted((n, repr(d)) for n, d in irdl.attributes.items())}",
                f"regions {[(n, repr(d)) for n, d in irdl.regions]}",
                f"successors {[(n, repr(d)) for n, d in irdl.successors]}",
                f"traits {sorted(_stable(t) for t in irdl.traits)}",
            ]
        else:
            parameters = cast(type[ParametrizedAttribute], self.definition).get_irdl_definition().parameters
            parts = [f"parameters {[(n, repr(d)) for n, d in parameters]}"]
        if self.form is not None:
            parts.append(f"form {_stable(self.form)}")
        return f"{self.name}\n" + "\n".join(parts)

    @property
    def digest(self) -> str:
        """A short name for the definition's canonical declaration."""
        return hashlib.blake2s(self.text.encode(), digest_size=8).hexdigest()

    @property
    def evidence(self) -> CodeEvidence:
        """The run's record of this definition: its declaration and the source of its word and its written form (D16).

        The source covers what the declaration does not show, such as the
        word's verifier and expansion, until words carry fixtures of their
        own; it is strict, so any edit to the module that defines the word
        or its written form counts as a change.
        """
        source = source_evidence(self.definition)
        if self.form is not None:
            source = source.joined(source_evidence(self.form), "written form")
        return CodeEvidence(self.name, self.text, source)


def _stable(value: object) -> str:
    """A printout that is the same in every process: code by module and name, never by address."""
    if isinstance(value, type) or (callable(value) and hasattr(value, "__qualname__")):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', '')}"
    if dataclasses.is_dataclass(value):
        fields = ", ".join(f"{f.name}={_stable(getattr(value, f.name))}" for f in dataclasses.fields(value))
        return f"{type(value).__qualname__}({fields})"
    if type(value).__repr__ is object.__repr__:
        return f"{type(value).__module__}.{type(value).__qualname__}()"  # the default repr holds an address
    return repr(value)


class Composition:
    """An immutable set of definitions with queries, a digest and a readable printout."""

    def __init__(self, definitions: Iterable[Definition], *, name: str = "kuroiro"):
        entries: dict[str, Definition] = {}
        for entry in definitions:
            if entry.name in entries:
                raise ValueError(f"{entry.name} is defined twice")
            entries[entry.name] = entry
        self.name = name
        self._entries = MappingProxyType(entries)

    @property
    def definitions(self) -> Mapping[str, Definition]:
        return self._entries

    @property
    def digest(self) -> str:
        """A digest of every definition's name, area and declaration."""
        listing = "\n".join(f"{e.name} {e.area} {e.digest}" for e in sorted(self._entries.values(), key=lambda e: e.name))
        return hashlib.blake2s(listing.encode(), digest_size=8).hexdigest()

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> Definition:
        return self._entries[name]

    def evidence(self, names: Iterable[str]) -> tuple[CodeEvidence, ...]:
        """The run's records of the named definitions, by name (D16)."""
        return tuple(self._entries[n].evidence for n in sorted(set(names)))

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
        """The readable contract: every word by area, with its meaning."""
        lines = [f"# {self.name} contract {self.digest}"]
        areas = list(AREAS) + sorted({e.area for e in self._entries.values()} - set(AREAS))
        for area in areas:
            words = sorted((e for e in self.words(area)), key=lambda e: e.name)
            if not words:
                continue
            lines += ["", f"## {area}", ""]
            width = max(len(e.name) for e in words)
            lines += [f"{e.name.ljust(width)}  {e.meaning}" for e in words]
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
