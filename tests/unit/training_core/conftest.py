"""The new training core's tests: like the IR system's, their header reports the GIL state they run with.

uv run pytest tests/unit/training_core --confcutdir=tests/unit/training_core
"""

from tests.unit.iro.conftest import pytest_report_header

__all__ = ["pytest_report_header"]
