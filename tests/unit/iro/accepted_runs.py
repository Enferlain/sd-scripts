"""Test-only accepted runs (D1).

Builds an accepted run straight from a written form, without the Python
front end, so the training side can be built and tested against explicit
accepted inputs. It is not a production acceptance bypass: no production
module imports it, which the repository's dependency test checks.
"""

from library.iro.accepted import AcceptedRun, accept
from library.iro.composition import Composition
from library.iro.text import parse


def accepted_run(text: str, composition: Composition | None = None, *, source: str = "<test>", replicated: bool = False) -> AcceptedRun:
    """The accepted run of a written strategy, or a rejection naming every problem."""
    return accept(parse(text, composition, source=source), composition, replicated=replicated)


__all__ = ["accepted_run"]
