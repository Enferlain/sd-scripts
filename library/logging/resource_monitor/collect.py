from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Snapshot:
    """Point-in-time resource snapshot."""

    gpu_allocated_mb: float | None
    gpu_allocated_by_device_mb: dict[str, float] | None
    gpu_reserved_mb: float | None
    gpu_reserved_by_device_mb: dict[str, float] | None
    gpu_peak_allocated_mb: float | None
    gpu_peak_allocated_by_device_mb: dict[str, float] | None
    cpu_rss_mb: float | None
    cpu_vms_mb: float | None


@dataclass(frozen=True)
class _SampledMetrics:
    """A single background-sampler datapoint."""

    ts: float
    gpu_used_mb: float | None
    gpu_used_by_device_mb: dict[str, float] | None
    cpu_rss_mb: float | None
    cpu_vms_mb: float | None
    collection_ms: float | None


@dataclass(frozen=True)
class _DeepCounters:
    """Deep allocator diagnostics collected from torch.cuda.memory_stats()."""

    alloc_retries: int | None
    ooms: int | None
    active_mb: float | None
    reserved_mb: float | None
    inactive_split_mb: float | None
    collection_ms: float | None


class ResourceCollectionMixin:
    def _is_cuda_visible(self) -> bool:
        return torch.cuda.is_available()

    def _snapshot_device_indices(self) -> list[int]:
        if not self._is_cuda_visible():
            return []
        if self._device_scope == "all_visible":
            return list(range(torch.cuda.device_count()))
        return [torch.cuda.current_device()]

    def _format_device_map_mb(self, values_by_device: Mapping[int, float]) -> dict[str, float] | None:
        if not values_by_device:
            return None
        return {str(device): float(value) for device, value in sorted(values_by_device.items())}

    def _collect_snapshot(self, *, reset_peak: bool = False) -> _Snapshot:
        gpu_allocated_mb: float | None = None
        gpu_allocated_by_device_mb: dict[str, float] | None = None
        gpu_reserved_mb: float | None = None
        gpu_reserved_by_device_mb: dict[str, float] | None = None
        gpu_peak_allocated_mb: float | None = None
        gpu_peak_allocated_by_device_mb: dict[str, float] | None = None

        if self._is_cuda_visible():
            try:
                allocated_by_device: dict[int, float] = {}
                reserved_by_device: dict[int, float] = {}
                peak_by_device: dict[int, float] = {}
                for device_index in self._snapshot_device_indices():
                    if reset_peak:
                        torch.cuda.reset_peak_memory_stats(device_index)
                    allocated_by_device[device_index] = torch.cuda.memory_allocated(device_index) / (1024 * 1024)
                    reserved_by_device[device_index] = torch.cuda.memory_reserved(device_index) / (1024 * 1024)
                    peak_by_device[device_index] = torch.cuda.max_memory_allocated(device_index) / (1024 * 1024)
                gpu_allocated_by_device_mb = self._format_device_map_mb(allocated_by_device)
                gpu_reserved_by_device_mb = self._format_device_map_mb(reserved_by_device)
                gpu_peak_allocated_by_device_mb = self._format_device_map_mb(peak_by_device)
                gpu_allocated_mb = sum(allocated_by_device.values(), start=0.0) if allocated_by_device else None
                gpu_reserved_mb = sum(reserved_by_device.values(), start=0.0) if reserved_by_device else None
                gpu_peak_allocated_mb = sum(peak_by_device.values(), start=0.0) if peak_by_device else None
            except Exception as exc:  # pragma: no cover - backend-specific failure path
                self._warn_once("snapshot_cuda", "resource monitor CUDA snapshot failed: %s", exc)

        cpu_rss_mb: float | None = None
        cpu_vms_mb: float | None = None
        try:
            memory_info = self._process.memory_info()
            cpu_rss_mb = memory_info.rss / (1024 * 1024)
            cpu_vms_mb = memory_info.vms / (1024 * 1024)
        except Exception as exc:  # pragma: no cover - platform-specific failure path
            self._warn_once("snapshot_cpu", "resource monitor CPU memory snapshot failed: %s", exc)

        return _Snapshot(
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_allocated_by_device_mb=gpu_allocated_by_device_mb,
            gpu_reserved_mb=gpu_reserved_mb,
            gpu_reserved_by_device_mb=gpu_reserved_by_device_mb,
            gpu_peak_allocated_mb=gpu_peak_allocated_mb,
            gpu_peak_allocated_by_device_mb=gpu_peak_allocated_by_device_mb,
            cpu_rss_mb=cpu_rss_mb,
            cpu_vms_mb=cpu_vms_mb,
        )

    def _collect_gpu_used_via_nvml_mb(self) -> tuple[float | None, dict[str, float] | None]:
        if not self._is_cuda_visible() or self._nvml_unavailable:
            return None, None

        if self._nvml_module is None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml_module = pynvml
            except Exception as exc:
                self._nvml_unavailable = True
                self._warn_once("nvml_unavailable", "resource monitor NVML unavailable; using fallback metrics (%s)", exc)
                return None, None

        assert self._nvml_module is not None

        try:
            if self._device_scope == "all_visible":
                indices = range(torch.cuda.device_count())
            else:
                indices = [torch.cuda.current_device()]
            used_bytes_by_device: dict[int, int] = {}
            for index in indices:
                handle = self._nvml_module.nvmlDeviceGetHandleByIndex(index)
                info = self._nvml_module.nvmlDeviceGetMemoryInfo(handle)
                used_bytes_by_device[int(index)] = int(info.used)
            device_map = self._format_device_map_mb(
                {device: used_bytes / (1024 * 1024) for device, used_bytes in used_bytes_by_device.items()}
            )
            used_mb = sum(device_map.values(), start=0.0) if device_map else None
            return used_mb, device_map
        except Exception as exc:
            self._warn_once("nvml_collect", "resource monitor NVML collection failed; using fallback metrics (%s)", exc)
            return None, None

    def _collect_gpu_used_via_torch_mb(self) -> tuple[float | None, dict[str, float] | None]:
        if not self._is_cuda_visible():
            return None, None

        def _used_for_device(index: int) -> int:
            try:
                free, total = torch.cuda.mem_get_info(index)
            except TypeError:
                with torch.cuda.device(index):
                    free, total = torch.cuda.mem_get_info()
            return max(int(total - free), 0)

        try:
            if self._device_scope == "all_visible":
                used_bytes_by_device = {index: _used_for_device(index) for index in range(torch.cuda.device_count())}
            else:
                current = torch.cuda.current_device()
                used_bytes_by_device = {current: _used_for_device(current)}
            device_map = self._format_device_map_mb(
                {device: used_bytes / (1024 * 1024) for device, used_bytes in used_bytes_by_device.items()}
            )
            used_mb = sum(device_map.values(), start=0.0) if device_map else None
            return used_mb, device_map
        except Exception as exc:  # pragma: no cover - backend-specific
            self._warn_once("torch_mem_get_info", "resource monitor torch mem_get_info fallback failed: %s", exc)
            return None, None

    def _collect_sample_metrics(self) -> _SampledMetrics:
        started = time.perf_counter()

        gpu_used_mb, gpu_used_by_device_mb = self._collect_gpu_used_via_nvml_mb()
        if gpu_used_mb is None:
            gpu_used_mb, gpu_used_by_device_mb = self._collect_gpu_used_via_torch_mb()

        cpu_rss_mb: float | None = None
        cpu_vms_mb: float | None = None
        try:
            memory_info = self._process.memory_info()
            cpu_rss_mb = memory_info.rss / (1024 * 1024)
            cpu_vms_mb = memory_info.vms / (1024 * 1024)
        except Exception as exc:  # pragma: no cover - platform specific
            self._warn_once("sample_cpu", "resource monitor sampled CPU memory collection failed: %s", exc)

        collection_ms = (time.perf_counter() - started) * 1000
        return _SampledMetrics(
            ts=time.time(),
            gpu_used_mb=gpu_used_mb,
            gpu_used_by_device_mb=gpu_used_by_device_mb,
            cpu_rss_mb=cpu_rss_mb,
            cpu_vms_mb=cpu_vms_mb,
            collection_ms=collection_ms,
        )

    def _bytes_to_mb(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value) / (1024 * 1024)
        except (TypeError, ValueError):
            return None

    def _collect_deep_counters(self, global_step: int) -> tuple[_DeepCounters | None, bool]:
        if self._mode != "deep":
            self._latest_deep_counters = None
            self._latest_deep_window_active = None
            return None, False

        self._ensure_deep_window_started(global_step)
        window_active = self._is_deep_window_active(global_step)
        self._latest_deep_window_active = window_active
        if not window_active:
            self._latest_deep_counters = None
            self._emit_deep_window_summary(reason="window_limit")
            return None, False

        if not self._is_cuda_visible():
            self._latest_deep_counters = None
            return None, True

        started = time.perf_counter()
        try:
            stats = torch.cuda.memory_stats()
        except Exception as exc:  # pragma: no cover - backend-specific
            self._warn_once("deep_stats", "resource monitor deep memory_stats collection failed: %s", exc)
            return None, True

        counters = _DeepCounters(
            alloc_retries=int(stats.get("num_alloc_retries", 0)),
            ooms=int(stats.get("num_ooms", 0)),
            active_mb=self._bytes_to_mb(stats.get("active_bytes.all.current")),
            reserved_mb=self._bytes_to_mb(stats.get("reserved_bytes.all.current")),
            inactive_split_mb=self._bytes_to_mb(stats.get("inactive_split_bytes.all.current")),
            collection_ms=(time.perf_counter() - started) * 1000,
        )
        self._latest_deep_counters = counters
        self._enforce_collection_budget(counters.collection_ms, source="deep")

        self._deep_window_sample_count += 1
        if self._deep_window_baseline_alloc_retries is None:
            self._deep_window_baseline_alloc_retries = counters.alloc_retries
        if self._deep_window_baseline_ooms is None:
            self._deep_window_baseline_ooms = counters.ooms
        self._deep_window_last_alloc_retries = counters.alloc_retries
        self._deep_window_last_ooms = counters.ooms
        if counters.inactive_split_mb is not None and (
            self._deep_window_peak_inactive_split_mb is None or counters.inactive_split_mb > self._deep_window_peak_inactive_split_mb
        ):
            self._deep_window_peak_inactive_split_mb = counters.inactive_split_mb

        if self._log_every_n_steps > 0 and global_step % self._log_every_n_steps == 0:
            self._log_info_external(
                "Resource deep[%s]: alloc_retries=%s, ooms=%s, active=%s, reserved=%s, inactive_split=%s",
                global_step,
                counters.alloc_retries,
                counters.ooms,
                self._format_gpu(counters.active_mb),
                self._format_gpu(counters.reserved_mb),
                self._format_gpu(counters.inactive_split_mb),
            )
        return counters, True
