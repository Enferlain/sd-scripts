"""Keeping the GIL off for a run (D15).

The runtime relies on running without the GIL: its coordinator, producers and
device workers run Python in parallel, which the GIL would turn into taking
turns. So nothing turns the GIL on unless a run names the selection that
needs it, and a run checks rather than trusts that:

- :func:`keep_off`, as a run starts in its process, fails if the GIL is
  already on, keeps each module known to turn it on unavailable, so that
  libraries which only look for it take their other paths, and makes
  CPython's warning that it enabled the GIL an error, so an import that turns
  it on fails at once, naming the module;
- :func:`check` reads the GIL state, which the host does after each of the
  run's loads and the runtime does periodically, off the per-call path,
  since code may swallow the import's error.

CPython enables the GIL before it warns, so a failed import still leaves the
GIL on; the run fails either way. No selection names a module that turns the
GIL on yet: the first piece that needs one, such as Triton for
``torch.compile``'s GPU code generation, brings the declaration that names it.
"""

import sys
import warnings

KNOWN = frozenset({"triton"})
"""Modules known to turn the GIL on when they load."""

WARNING = r"The global interpreter lock \(GIL\) has been enabled"
"""The start of CPython's warning when an import enables the GIL."""


GOING_AHEAD = (
    "a run goes ahead with the GIL on only by declaring the selection that needs it, and no selection declares one yet, "
    "so remove the import that turned it on, or run the module that needs it outside the run"
)
"""The declared way to go ahead, which a refusal names (D15)."""


class GilTurnedOn(RuntimeError):
    """The GIL is on, and nothing the run names turns it on (D15)."""


def enabled() -> bool | None:
    """Whether the GIL is on; ``None`` on a CPython built with the GIL, where it is always on."""
    probe = getattr(sys, "_is_gil_enabled", None)
    return probe() if probe is not None else None


def state() -> str:
    """The GIL state a run reports: ``off``, ``on``, or ``built in`` for a CPython without the free-threaded build."""
    found = enabled()
    return "built in" if found is None else ("on" if found else "off")


def check(where: str) -> None:
    """Raises :class:`GilTurnedOn` if the GIL is on, saying where it was found."""
    if enabled():
        raise GilTurnedOn(f"the GIL is on {where}, and nothing the run names turns it on (D15); {GOING_AHEAD}")


def keep_off(where: str = "as the run starts") -> None:
    """Prepares this process for a run without the GIL; raises :class:`GilTurnedOn` if the GIL is already on, saying where it was found.

    A module known to turn the GIL on that is not loaded yet becomes
    unavailable for the rest of the process: importing it fails, and
    ``importlib.util.find_spec`` reports it missing. CPython's warning that
    it enabled the GIL becomes an error, in every thread.
    """
    if enabled() is None:
        raise GilTurnedOn("this CPython is built with the GIL, and a run needs the free-threaded build (D15)")
    check(where)
    for name in KNOWN:
        if name not in sys.modules:
            sys.modules[name] = None  # ty: ignore[invalid-assignment]  # the import system's marker for a module that is not available
    warnings.filterwarnings("error", message=WARNING, category=RuntimeWarning)


__all__ = ["GOING_AHEAD", "KNOWN", "WARNING", "GilTurnedOn", "check", "enabled", "keep_off", "state"]
