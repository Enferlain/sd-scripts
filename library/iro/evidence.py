"""Evidence of what a piece of code is (D16).

A run records, for each piece of code it uses, evidence of what that code is,
and a later check compares it with the installed code's: what the piece
declares it takes and returns, as its declaration's canonical text, and the
source of the module that defines it with its qualified name. The source
stands in for the outputs of a piece's own fixtures until pieces carry them,
and it is strict: any edit to the defining module changes it. It covers only
that module, so a change that reaches a piece through helpers in other
modules shows only in fixture outputs, once pieces carry them.

A record keeps its parts, not only their digest, so it prints in full, and a
mismatch is reported part by part. No part alone says that two pieces are the
same code: two records match only when every part matches, and the checks
that compare them refuse a record that lacks a part its kind needs.
"""

import functools
import hashlib
import importlib.metadata
import inspect
import marshal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


def _digest(data: bytes) -> str:
    return hashlib.blake2s(data, digest_size=8).hexdigest()


@dataclass(frozen=True)
class SourceEvidence:
    """Where a piece's code is defined: its module, its qualified name and a digest of the module's source with that name.

    The name tells apart two pieces defined in one module. ``covers`` names
    what else the digest covers: each layer a wrapped function goes through,
    the values a partial binds, a definition's written form, or, for compiled
    code without source, the library release it comes from. Code that has
    none of these gets a digest of this one object, so it never matches in
    another process.
    """

    module: str
    qualname: str
    digest: str
    covers: tuple[str, ...] = ()

    def printout(self) -> str:
        line = f"{self.module}.{self.qualname} source {self.digest}"
        return line + (f" (with {'; '.join(self.covers)})" if self.covers else "")

    def joined(self, other: SourceEvidence, label: str) -> SourceEvidence:
        """This evidence extended by another piece of code it relies on, named by ``label``."""
        digest = _digest(f"{self.digest} {other.digest}".encode())
        return SourceEvidence(self.module, self.qualname, digest, (*self.covers, f"{label} {other.module}.{other.qualname}", *other.covers))


@dataclass(frozen=True)
class CodeEvidence:
    """What one piece of code a run uses is (D16).

    ``declaration`` is the canonical text of what the piece declares it takes
    and returns, empty for code that declares nothing, such as a setup.
    ``source`` is where its code is defined, None only in a record that holds
    a name alone, as a run accepted without installed code does. A word's
    definition records its declaration and the source of its operation or
    type and of its written form; cold loading also derives its written form
    and expansion again (D14).
    """

    name: str
    declaration: str = ""
    source: SourceEvidence | None = None

    @property
    def digest(self) -> str:
        """The record's identity: a short name for its printout."""
        return _digest(self.printout().encode())

    def differences(self, installed: CodeEvidence) -> tuple[str, ...]:
        """What differs between this record and the installed code's, part by part, in words."""
        found = []
        if self.name != installed.name:
            found.append(f"its name (@{installed.name})")
        if self.declaration != installed.declaration:
            found.append("what it takes or returns")
        if self.source != installed.source:
            where = installed.source or self.source
            found.append(f"its source in {where.module}.{where.qualname}" if where else "its source")
        return tuple(found)

    def printout(self) -> str:
        """The record in full: its name, each line of its declaration and its source."""
        lines = [f"@{self.name}"]
        lines += [f"  declares {line}" for line in self.declaration.splitlines()]
        if self.source is not None:
            lines.append(f"  {self.source.printout()}")
        return "\n".join(lines)


def source_evidence(code: object) -> SourceEvidence:
    """The source evidence of a function, of an implementation that wraps one, or of an object's class.

    A function's evidence is its own, through every layer that wraps it; an
    implementation's is its function's; a partial's is its function's with
    the values it binds; any other object's, such as a setup's or a
    deriver's, is its class's, since what an object is configured with is
    recorded with the settings that configure it (D16). Computed once per
    function or class.
    """
    function = getattr(code, "function", code)
    bound: list[str] = []
    while True:
        if isinstance(function, functools.partial):
            bound.append(_bound(function))
            function = function.func
        elif inspect.ismethod(function):
            function = function.__func__
        else:
            break
    target = function if inspect.isroutine(function) or inspect.isclass(function) else type(function)
    found = _source_of(target)
    if bound:
        values = "; ".join(reversed(bound))
        found = SourceEvidence(
            found.module, found.qualname, _digest(f"{found.digest} {values}".encode()), (*found.covers, f"bound {values}")
        )
    return found


_LITERALS = (type(None), bool, int, float, complex, str, bytes)


def _literal(value: object) -> str | None:
    """A value's text when it is the same in every process, as literals and containers of them are; else None."""
    if isinstance(value, _LITERALS):
        return repr(value)
    if isinstance(value, tuple | list):
        items = [_literal(v) for v in value]
        return None if None in items else f"{type(value).__name__}({', '.join(i for i in items if i is not None)})"
    if isinstance(value, dict):
        pairs = [(_literal(k), _literal(v)) for k, v in value.items()]
        return None if any(None in p for p in pairs) else "{" + ", ".join(f"{k}: {v}" for k, v in sorted(pairs)) + "}"
    return None


def _bound(partial: functools.partial) -> str:
    """The values a partial binds, as text; a value with no process-independent text is named by this object, so it never matches elsewhere."""
    parts = [_literal(v) or f"<{type(v).__qualname__} {id(v)}>" for v in partial.args]
    parts += [f"{key}={_literal(v) or f'<{type(v).__qualname__} {id(v)}>'}" for key, v in sorted(partial.keywords.items())]
    return f"({', '.join(parts)})"


def _layers(target: object) -> list[object]:
    """The target and every function it wraps, outermost first."""
    layers = [target]
    while (inner := getattr(layers[-1], "__wrapped__", None)) is not None and all(inner is not seen for seen in layers):
        layers.append(inner)
    return layers


def _file_bytes(layer: object) -> bytes | None:
    try:
        path = inspect.getsourcefile(layer)  # ty: ignore[invalid-argument-type]
        return Path(path).read_bytes() if path else None
    except TypeError, OSError:
        return None


def _release(module: str) -> str | None:
    """The release compiled code comes from: Python's for its own modules, else the installed library's."""
    top = module.split(".")[0]
    if top in sys.stdlib_module_names or top == "builtins":
        return f"python {sys.version.split()[0]}"
    try:
        return f"{top} {importlib.metadata.version(top)}"
    except importlib.metadata.PackageNotFoundError:
        version = getattr(sys.modules.get(top), "__version__", None)
        return f"{top} {version}" if isinstance(version, str) else None


@functools.cache
def _source_of(target: object) -> SourceEvidence:
    layers = _layers(target)
    inner = layers[-1]
    module = getattr(inner, "__module__", None) or ""
    qualname = getattr(inner, "__qualname__", None) or type(inner).__qualname__
    covers: list[str] = []
    data = f"{module}.{qualname}".encode()
    for depth, layer in enumerate(layers):
        if depth < len(layers) - 1:
            code = getattr(layer, "__code__", None)
            covers.append(f"wrapped by {code.co_qualname if code is not None else type(layer).__qualname__}")
        found = _file_bytes(layer)
        if found is None:
            code = getattr(layer, "__code__", None)
            if code is not None:
                found = marshal.dumps(code)
            elif (release := _release(module)) is not None:
                found = release.encode()
                covers.append(f"compiled, {release}")
            else:
                found = f"unknown {id(layer)}".encode()
        data += b"\n" + found
    return SourceEvidence(module, qualname, _digest(data), tuple(covers))


class Named(Protocol):
    """Code that a run records by a name it carries, as a setup or a deriver does."""

    @property
    def name(self) -> str: ...


def setup_evidence(setup: Named) -> CodeEvidence:
    """The record of a setup or a deriver: its name and the evidence of its source (D16)."""
    return CodeEvidence(setup.name, source=source_evidence(setup))


def setup_digest(setup: object) -> str:
    """The digest of a setup's or deriver's source, cheap enough to compare on every call: computed once per class."""
    return _source_of(type(setup)).digest


__all__ = ["CodeEvidence", "SourceEvidence", "setup_digest", "setup_evidence", "source_evidence"]
