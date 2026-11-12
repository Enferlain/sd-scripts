"""The shared diagnostic: one source-attributed problem report (D18).

Every stage of the IR system reports problems the same way: parsing a written
form, normalization, verification, fulfillment and preparation. A rejection
reports every independent problem it finds. A diagnostic says whether the
meaning is invalid or whether a valid strategy hits a limit of the selected
target, reported at the earliest point where that is known (D14).

A rendered diagnostic reads:

    iro verify [iro-life-004] strategies/sdxl/finetune.py:88
      stop @data after @training does not cover @validation's pending
      request; @validation still takes from @batches.
      fix: stop @data after @consumers, or give @validation its own stream.
"""

import re
import textwrap
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

_CODE = re.compile(r"iro-[a-z]+-\d{3}")
_WIDTH = 72


class Stage(StrEnum):
    """The stage that found a problem."""

    PARSE = "parse"
    READ = "read"
    NORMALIZE = "normalize"
    VERIFY = "verify"
    FULFILL = "fulfill"
    PREPARE = "prepare"


class Kind(StrEnum):
    """Invalid meaning, or a valid strategy the selected target cannot realize."""

    SEMANTIC_ERROR = "semantic_error"
    TARGET_LIMIT = "target_limit"


@dataclass(frozen=True)
class Diagnostic:
    """One problem: its rule code, the authored names involved, where it was written, and a fix.

    ``source`` is where the offending element was authored, such as
    ``strategies/sdxl/finetune.py:88``, or a written form's own line when the
    problem was found while parsing one. ``fragment`` is the element as
    printed. ``path`` is the operation's structural identity, for tools.
    """

    stage: Stage
    code: str
    message: str
    kind: Kind = Kind.SEMANTIC_ERROR
    names: tuple[str, ...] = ()
    source: str | None = None
    fragment: str | None = None
    fix: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if not _CODE.fullmatch(self.code):
            raise ValueError(f"rule code {self.code!r} must look like iro-<area>-<nnn>")
        if not self.message:
            raise ValueError("a diagnostic needs a message")

    def render(self) -> str:
        header = f"iro {self.stage} [{self.code}]"
        if self.source:
            header += f" {self.source}"
        if self.kind == Kind.TARGET_LIMIT:
            header += " (target limit)"
        lines = [header]
        lines.extend(textwrap.wrap(self.message, _WIDTH, initial_indent="  ", subsequent_indent="  "))
        if self.fragment:
            lines.extend("    " + line for line in self.fragment.splitlines())
        if self.fix:
            lines.extend(textwrap.wrap(f"fix: {self.fix}", _WIDTH, initial_indent="  ", subsequent_indent="    "))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


class Rejection(Exception):
    """A rejection carrying every independent problem found, in the order found."""

    def __init__(self, diagnostics: Iterable[Diagnostic]):
        self.diagnostics: tuple[Diagnostic, ...] = tuple(diagnostics)
        if not self.diagnostics:
            raise ValueError("a rejection needs at least one diagnostic")
        super().__init__("\n\n".join(d.render() for d in self.diagnostics))

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(d.code for d in self.diagnostics)

    @property
    def target_limits_only(self) -> bool:
        """True when the strategy is valid and only the selected target cannot realize it."""
        return all(d.kind == Kind.TARGET_LIMIT for d in self.diagnostics)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self.diagnostics)


class Problems:
    """Collects a stage's independent problems and rejects once at the end."""

    def __init__(self, stage: Stage):
        self.stage = stage
        self._found: list[Diagnostic] = []

    def add(
        self,
        code: str,
        message: str,
        *,
        kind: Kind = Kind.SEMANTIC_ERROR,
        names: Iterable[str] = (),
        source: str | None = None,
        fragment: str | None = None,
        fix: str | None = None,
        path: str | None = None,
    ) -> Diagnostic:
        found = Diagnostic(self.stage, code, message, kind, tuple(names), source, fragment, fix, path)
        if found not in self._found:
            self._found.append(found)
        return found

    def __bool__(self) -> bool:
        return bool(self._found)

    def __len__(self) -> int:
        return len(self._found)

    @property
    def found(self) -> tuple[Diagnostic, ...]:
        return tuple(self._found)

    def reject_if_any(self) -> None:
        if self._found:
            raise Rejection(self._found)


__all__ = ["Diagnostic", "Kind", "Problems", "Rejection", "Stage"]
