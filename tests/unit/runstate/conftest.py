"""The run state's tests: like the IR system's, their pytest header reports the GIL state they run with.

uv run pytest tests/unit/runstate --confcutdir=tests/unit/runstate
"""

from tests.unit.iro.conftest import pytest_report_header

__all__ = ["pytest_report_header"]
