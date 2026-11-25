"""Evidence of what a piece of code is (D16).

A run records, for each piece of code it uses, evidence of what that code is,
and a later check compares it with the installed code's: what the piece
declares it takes and returns, as its declaration's canonical text, and the
source of the module that defines it with its qualified name. The source
stands in for the outputs of a piece's own fixtures until pieces carry them,
and it is strict: any edit to the defining module changes it. It covers only
that module, so a change that reaches a piece through helpers in other
modules shows only in fixture outputs, once pieces carry them, and so does
state a function reads from its module at run time. Compiled code without
source is named by the release it comes from, which does not tell apart two
builds of one release, so a selected implementation that is itself compiled
is refused until the binary can be hashed; code it calls or captures is
named that way.

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
    the values its closures capture or a partial binds, a definition's
    written form, or, for compiled code without source, the library release
    it comes from. ``compiled`` says the piece itself is compiled code
    without source, whose code the system cannot read. ``opaque`` says why
    the code's identity cannot be established from what it captures or
    binds, such as a list or a model object, which can change while staying
    the same object; empty when it can. A selected implementation that is
    compiled or opaque is refused (D16).
    """

    module: str
    qualname: str
    digest: str
    covers: tuple[str, ...] = ()
    compiled: bool = False
    opaque: str = ""

    def printout(self) -> str:
        line = f"{self.module}.{self.qualname} source {self.digest}"
        line += f" (with {'; '.join(self.covers)})" if self.covers else ""
        return line + (f" (cannot be identified: it {self.opaque})" if self.opaque else "")

    def joined(self, other: SourceEvidence, label: str) -> SourceEvidence:
        """This evidence extended by another piece of code it relies on, named by ``label``."""
        digest = _digest(f"{self.digest} {other.digest}".encode())
        covers = (*self.covers, f"{label} {other.module}.{other.qualname}", *other.covers)
        return SourceEvidence(self.module, self.qualname, digest, covers, self.compiled, self.opaque or other.opaque)


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


def source_evidence(code: object, *, selected: bool = False) -> SourceEvidence:
    """The source evidence of a function or of an object's class.

    A function's evidence is its own, through every layer that wraps it and
    with the values its closures capture; a partial's is its function's with
    the values it binds; any other object's, such as a setup's or a
    deriver's, is its class's, since what an object is configured with is
    recorded with the settings that configure it (D16).

    For code a strategy ``selected`` as an implementation, every piece of
    state bound into it must be represented in its evidence, so a callable
    object or a method bound to an object, whose state the class's source
    does not show, makes the evidence opaque. Captured and bound values are
    held to the same rule: a value that cannot change and reads the same in
    every program, such as a number, a string or a tuple of them, is written
    out; captured or bound code is included by its own evidence; anything
    else, such as a list, a dict or a model object, can change while staying
    the same object and makes the evidence opaque. The source part is
    computed once per code; captured and bound values are read
    each time.
    """
    return _evidence(code, frozenset(), selected)


def _evidence(code: object, seen: frozenset[int], strict: bool) -> SourceEvidence:
    bound: list[tuple[str, str]] = []
    problem = ""
    function = code
    while True:
        if isinstance(function, functools.partial):
            bound.append(_bound(function, seen))
            function = function.func
        elif inspect.ismethod(function):
            if strict and not inspect.isclass(function.__self__):
                problem = problem or _bound_to(function.__self__)
            function = function.__func__
        else:
            break
    owner = getattr(function, "__self__", None)
    if strict and inspect.isbuiltin(function) and owner is not None and not (inspect.ismodule(owner) or inspect.isclass(owner)):
        problem = problem or _bound_to(owner)
    if inspect.isroutine(function) or inspect.isclass(function):
        target = function
    else:
        target = type(function)
        if strict:
            problem = problem or f"is a callable {target.__qualname__} object, whose state the system cannot see"
    found = _source_of(target)
    if id(target) in seen:  # code that refers back to itself: its source is covered already
        return _with_problem(found, problem)
    captured, why = _captured(target, seen | {id(target)})
    problem = problem or why
    extra = [f"closes over ({captured})"] if captured else []
    if bound:
        extra.append(f"bound {'; '.join(text for text, _ in reversed(bound))}")
        problem = problem or next((why for _, why in bound if why), "")
    if not extra:
        return _with_problem(found, problem)
    text = "; ".join(extra)
    digest = _digest(f"{found.digest} {text}".encode())
    return SourceEvidence(found.module, found.qualname, digest, (*found.covers, *extra), found.compiled, problem or found.opaque)


def _bound_to(owner: object) -> str:
    return f"is a method bound to {_a(type(owner).__qualname__)}, whose state the system cannot see"


def _a(name: str) -> str:
    """A type's name with its article."""
    return f"{'an' if name[:1].lower() in 'aeiou' else 'a'} {name}"


def _with_problem(found: SourceEvidence, problem: str) -> SourceEvidence:
    """The evidence with the reason state bound into the code leaves it open, which says more than a reason from its source."""
    if not problem:
        return found
    return SourceEvidence(found.module, found.qualname, found.digest, found.covers, found.compiled, problem)


_LITERALS = (type(None), bool, int, float, complex, str, bytes)


def _value(value: object, seen: frozenset[int]) -> tuple[str, str]:
    """A captured or bound value as text, and, when it leaves the code's identity open, what it is, as a phrase; else nothing."""
    if isinstance(value, _LITERALS):
        return repr(value), ""
    if isinstance(value, tuple | frozenset):
        items = [_value(v, seen) for v in value]
        texts = sorted(t for t, _ in items) if isinstance(value, frozenset) else [t for t, _ in items]
        return f"{type(value).__name__}({', '.join(texts)})", next((why for _, why in items if why), "")
    if callable(value):
        nested = _evidence(value, seen, True)
        return f"code {nested.digest}", f"code that {nested.opaque}" if nested.opaque else ""
    return f"<{type(value).__qualname__}>", f"{_a(type(value).__qualname__)}, which can change while staying the same object"


def _bound(partial: functools.partial, seen: frozenset[int]) -> tuple[str, str]:
    """The values a partial binds, as text, and why they leave the code's identity open, as a clause, or nothing."""
    parts = [_value(v, seen) for v in partial.args]
    parts += [(f"{key}={text}", why) for key, (text, why) in ((k, _value(v, seen)) for k, v in sorted(partial.keywords.items()))]
    return f"({', '.join(t for t, _ in parts)})", next((f"binds {why}" for _, why in parts if why), "")


def _captured(target: object, seen: frozenset[int]) -> tuple[str, str]:
    """The values the closures of a function and of every layer it wraps capture, as text or nothing, and why they leave the code's identity open, as a clause, or nothing."""
    parts: list[tuple[str, str]] = []
    for layer in _layers(target):
        code, cells = getattr(layer, "__code__", None), getattr(layer, "__closure__", None)
        for name, cell in zip(code.co_freevars if code is not None else (), cells or (), strict=False):
            try:
                contents = cell.cell_contents
            except ValueError:  # a cell not yet filled, such as a function's own name before its definition ends
                parts.append((f"{name}=<empty>", ""))
                continue
            text, why = _value(contents, seen)
            parts.append((f"{name}={text}", f"closes over {name}, {why}" if why else ""))
    return ", ".join(t for t, _ in parts), next((why for _, why in parts if why), "")


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
    if not top:
        return None
    if top in sys.stdlib_module_names or top == "builtins":
        return f"python {sys.version.split()[0]}"
    try:
        return f"{top} {importlib.metadata.version(top)}"
    except importlib.metadata.PackageNotFoundError:
        version = getattr(sys.modules.get(top), "__version__", None)
        return f"{top} {version}" if isinstance(version, str) else None


_SOURCES: dict[tuple[object, ...], SourceEvidence] = {}


def _source_of(target: object) -> SourceEvidence:
    """The source evidence of a function or class through every layer that wraps it, computed once per code.

    Keyed by each layer's code object, or by the layer itself where it has
    none, such as a class: closures made from one definition share an entry,
    and the cache keeps no closure, nor anything it captures, alive.
    """
    layers = _layers(target)
    key = tuple(getattr(layer, "__code__", layer) for layer in layers)
    found = _SOURCES.get(key)
    return found if found is not None else _SOURCES.setdefault(key, _computed_source(layers))


def _computed_source(layers: list[object]) -> SourceEvidence:
    inner = layers[-1]
    module = getattr(inner, "__module__", None) or ""
    qualname = getattr(inner, "__qualname__", None) or type(inner).__qualname__
    covers: list[str] = []
    compiled, opaque = False, ""
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
            else:
                compiled = compiled or depth == 0
                if (release := _release(module)) is not None:
                    found = release.encode()
                    covers.append(f"compiled, {release}")
                else:
                    found, opaque = f"unknown {module}.{qualname}".encode(), "is compiled code from a release the system cannot name"
        data += b"\n" + found
    return SourceEvidence(module, qualname, _digest(data), tuple(covers), compiled, opaque)


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
