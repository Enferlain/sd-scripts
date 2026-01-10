"""
Resource tracking utilities for benchmarking.

Provides decorators and context managers to track:
- GPU memory (allocated, reserved, peak)
- CPU RAM usage
- Execution time

Usage:
    from library.utils.resource_tracker import track_resources, ResourceTracker

    # As decorator:
    @track_resources("my_function")
    def my_function():
        ...

    # As context manager:
    with ResourceTracker("my_block") as tracker:
        ...
    print(tracker.summary())

    # Manual tracking:
    tracker = ResourceTracker("manual")
    tracker.start()
    ... do work ...
    tracker.stop()
    print(tracker.summary())
"""

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement."""

    timestamp: float = 0.0
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    cpu_ram_mb: float = 0.0
    cpu_percent: float = 0.0


@dataclass
class ResourceStats:
    """Aggregated resource statistics."""

    name: str
    duration_sec: float = 0.0

    # GPU stats
    gpu_allocated_start_mb: float = 0.0
    gpu_allocated_end_mb: float = 0.0
    gpu_allocated_peak_mb: float = 0.0
    gpu_reserved_start_mb: float = 0.0
    gpu_reserved_end_mb: float = 0.0
    gpu_reserved_peak_mb: float = 0.0

    # CPU stats
    cpu_ram_start_mb: float = 0.0
    cpu_ram_end_mb: float = 0.0
    cpu_ram_peak_mb: float = 0.0

    def summary(self) -> str:
        """Format stats as a readable summary."""
        lines = [
            f"=== {self.name} ===",
            f"Duration: {self.duration_sec:.2f}s",
            "",
            "GPU Memory:",
            f"  Allocated: {self.gpu_allocated_start_mb:.0f} → {self.gpu_allocated_end_mb:.0f} MB (peak: {self.gpu_allocated_peak_mb:.0f} MB)",
            f"  Reserved:  {self.gpu_reserved_start_mb:.0f} → {self.gpu_reserved_end_mb:.0f} MB (peak: {self.gpu_reserved_peak_mb:.0f} MB)",
            "",
            "CPU RAM:",
            f"  Used: {self.cpu_ram_start_mb:.0f} → {self.cpu_ram_end_mb:.0f} MB (peak: {self.cpu_ram_peak_mb:.0f} MB)",
        ]
        return "\n".join(lines)

    def log(self, level: int = logging.INFO) -> None:
        """Log the summary."""
        logger.log(level, self.summary())


class ResourceTracker:
    """
    Track GPU and CPU resource usage.

    Can be used as a context manager or manually with start()/stop().
    """

    def __init__(self, name: str = "unnamed", reset_peak: bool = True):
        """
        Initialize the tracker.

        Args:
            name: Label for this tracking session.
            reset_peak: If True, reset CUDA peak memory stats at start.
        """
        self.name = name
        self.reset_peak = reset_peak
        self._start_snapshot: ResourceSnapshot | None = None
        self._end_snapshot: ResourceSnapshot | None = None
        self._stats: ResourceStats | None = None
        self._process = psutil.Process()

    def _take_snapshot(self) -> ResourceSnapshot:
        """Capture current resource state."""
        snapshot = ResourceSnapshot(timestamp=time.perf_counter())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            snapshot.gpu_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            snapshot.gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024

        snapshot.cpu_ram_mb = self._process.memory_info().rss / 1024 / 1024
        snapshot.cpu_percent = psutil.cpu_percent(interval=None)

        return snapshot

    def start(self) -> "ResourceTracker":
        """Start tracking."""
        if self.reset_peak and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._start_snapshot = self._take_snapshot()
        return self

    def stop(self) -> ResourceStats:
        """Stop tracking and compute stats."""
        self._end_snapshot = self._take_snapshot()

        # Ensure start was called
        assert self._start_snapshot is not None, "Must call start() before stop()"
        start = self._start_snapshot
        end = self._end_snapshot

        # Get peak values
        gpu_allocated_peak = 0.0
        gpu_reserved_peak = 0.0
        if torch.cuda.is_available():
            gpu_allocated_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            gpu_reserved_peak = torch.cuda.max_memory_reserved() / 1024 / 1024

        self._stats = ResourceStats(
            name=self.name,
            duration_sec=end.timestamp - start.timestamp,
            gpu_allocated_start_mb=start.gpu_allocated_mb,
            gpu_allocated_end_mb=end.gpu_allocated_mb,
            gpu_allocated_peak_mb=gpu_allocated_peak,
            gpu_reserved_start_mb=start.gpu_reserved_mb,
            gpu_reserved_end_mb=end.gpu_reserved_mb,
            gpu_reserved_peak_mb=gpu_reserved_peak,
            cpu_ram_start_mb=start.cpu_ram_mb,
            cpu_ram_end_mb=end.cpu_ram_mb,
            cpu_ram_peak_mb=max(start.cpu_ram_mb, end.cpu_ram_mb),  # Approximate
        )

        return self._stats

    @property
    def stats(self) -> ResourceStats | None:
        """Get computed stats (available after stop())."""
        return self._stats

    def summary(self) -> str:
        """Get summary string (available after stop())."""
        if self._stats is None:
            return f"=== {self.name} === (not stopped yet)"
        return self._stats.summary()

    def __enter__(self) -> "ResourceTracker":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def track_resources(name: str | None = None, log_result: bool = True) -> Callable:
    """
    Decorator to track resources during function execution.

    Args:
        name: Label for tracking (defaults to function name).
        log_result: If True, log the summary after execution.

    Example:
        @track_resources("VAE Caching")
        def cache_latents():
            ...
    """

    def decorator(func: Callable) -> Callable:
        label = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            tracker = ResourceTracker(label)
            tracker.start()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                stats = tracker.stop()
                if log_result:
                    stats.log()

        return wrapper

    return decorator


# Convenience function for quick tracking
def get_current_resources() -> dict:
    """Get current resource usage as a dict."""
    result = {
        "cpu_ram_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_percent": psutil.cpu_percent(interval=None),
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        result["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    return result
