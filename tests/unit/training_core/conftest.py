"""The new training core's tests: like the IR system's, their header reports the GIL state they run with.

uv run pytest tests/unit/training_core --confcutdir=tests/unit/training_core
"""

import pytest

from library.iro import gil
from tests.unit.iro.conftest import pytest_report_header


def pytest_configure(config: pytest.Config) -> None:
    """A session without the GIL keeps Triton unavailable from its start, so no test loads it and turns the GIL on for the tests after it.

    Only that half of :func:`gil.keep_off` lasts the session: pytest restores
    the warning filters around each test, and each Trainer applies the whole
    rule again as its run starts. It applies only where the session runs
    without the GIL, which needs ``--confcutdir``: in a full ``tests/unit``
    session, ``tests/conftest.py``'s imports have turned the GIL on already.
    """
    if gil.enabled() is False:
        gil.keep_off()


__all__ = ["pytest_configure", "pytest_report_header"]
