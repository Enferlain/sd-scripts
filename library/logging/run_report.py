"""Compatibility wrapper for benchmark/run report helpers.

New code should import from ``library.logging.reports``.
"""

from library.logging.reports import is_benchmark_report_enabled, write_run_report

__all__ = ["is_benchmark_report_enabled", "write_run_report"]
