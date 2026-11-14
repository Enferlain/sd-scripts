"""The new IR system's tests: their pytest header reports the GIL state they actually run with.

Run them without the repository-wide ``tests/conftest.py``, whose imports of
today's code load torchvision and, through it, Triton, which turns the GIL on
for the whole session:

    uv run pytest tests/unit/iro --confcutdir=tests/unit/iro
"""

import sys
import sysconfig


def pytest_report_header(config) -> str:
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    enabled = sys._is_gil_enabled()
    line = f"free-threaded build: {free_threaded}; GIL enabled: {enabled}"
    if free_threaded and enabled:
        line += " (this session's imports turned it on; run with --confcutdir=tests/unit/iro)"
    return line
