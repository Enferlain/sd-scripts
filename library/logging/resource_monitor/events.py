from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from library.metadata.dataclasses.observability import ResourceMonitorFacts

from .collect import _DeepCounters, _Snapshot


logger = logging.getLogger(__name__)


class ResourceEventMixin:
    def _resolve_flush_mode(self) -> str:
        if self._jsonl_flush_mode != "auto":
            return self._jsonl_flush_mode
        return "line" if self._mode in {"sampled", "deep"} else "batch"

    def _open_jsonl_stream(self) -> None:
        if self._jsonl_file is not None or self._jsonl_path is None:
            return
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = self._jsonl_path.open("a", encoding="utf-8")
        except Exception as exc:
            self._jsonl_file = None
            self._warn_once("jsonl_open", "resource monitor JSONL disabled; failed to open %s (%s)", self._jsonl_path, exc)

    def _close_jsonl_stream(self) -> None:
        if self._jsonl_file is None:
            return
        try:
            self._jsonl_file.flush()
            self._jsonl_file.close()
        except Exception as exc:  # pragma: no cover - filesystem specific
            self._warn_once("jsonl_close", "resource monitor JSONL close failed: %s", exc)
        finally:
            self._jsonl_file = None

    def _write_jsonl_event(self, event: dict[str, Any], *, force_flush: bool = False) -> None:
        if self._jsonl_file is None:
            return

        try:
            self._jsonl_file.write(json.dumps(event, ensure_ascii=True) + "\n")
            self._jsonl_events_since_flush += 1

            flush_mode = self._resolve_flush_mode()
            should_flush = force_flush or flush_mode == "line"
            if flush_mode == "batch" and self._jsonl_events_since_flush >= self._jsonl_flush_every_n_events:
                should_flush = True

            if should_flush:
                self._jsonl_file.flush()
                self._jsonl_events_since_flush = 0
        except Exception as exc:  # pragma: no cover - filesystem specific
            self._warn_once("jsonl_write", "resource monitor JSONL write failed: %s", exc)

    def _build_event(
        self,
        *,
        event: str,
        global_step: int | None,
        epoch: int | None,
        phase: str | None,
        duration_ms: float | None,
        steps_per_sec: float | None,
        snapshot: _Snapshot | None,
        gpu_used_mb: float | None,
        gpu_used_by_device_mb: dict[str, float] | None,
        cpu_rss_mb: float | None,
        cpu_vms_mb: float | None,
        dropped_samples: int | None,
        collection_ms: float | None,
        deep_alloc_retries: int | None,
        deep_ooms: int | None,
        deep_active_mb: float | None,
        deep_reserved_mb: float | None,
        deep_inactive_split_mb: float | None,
        deep_window_active: bool | None,
        ts: float | None = None,
    ) -> dict[str, Any]:
        return {
            "ts": time.time() if ts is None else ts,
            "event": event,
            "rank": self._rank,
            "world_size": self._world_size,
            "mode": self._mode,
            "device_scope": self._device_scope,
            "run_identifier": self._run_identifier,
            "config_name": self._config_name,
            "git_sha": self._git_sha,
            "git_dirty": self._git_dirty,
            "global_step": global_step,
            "epoch": epoch,
            "phase": phase,
            "duration_ms": duration_ms,
            "gpu_allocated_mb": None if snapshot is None else snapshot.gpu_allocated_mb,
            "gpu_allocated_by_device_mb": None if snapshot is None else snapshot.gpu_allocated_by_device_mb,
            "gpu_reserved_mb": None if snapshot is None else snapshot.gpu_reserved_mb,
            "gpu_reserved_by_device_mb": None if snapshot is None else snapshot.gpu_reserved_by_device_mb,
            "gpu_peak_allocated_mb": None if snapshot is None else snapshot.gpu_peak_allocated_mb,
            "gpu_peak_allocated_by_device_mb": None if snapshot is None else snapshot.gpu_peak_allocated_by_device_mb,
            "gpu_used_mb": gpu_used_mb,
            "gpu_used_by_device_mb": gpu_used_by_device_mb,
            "cpu_rss_mb": cpu_rss_mb,
            "cpu_vms_mb": cpu_vms_mb,
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": None,
            "dropped_samples": dropped_samples,
            "collection_ms": collection_ms,
            "deep_alloc_retries": deep_alloc_retries,
            "deep_ooms": deep_ooms,
            "deep_active_mb": deep_active_mb,
            "deep_reserved_mb": deep_reserved_mb,
            "deep_inactive_split_mb": deep_inactive_split_mb,
            "deep_window_active": deep_window_active,
        }

    def _emit_event(
        self,
        *,
        event: str,
        global_step: int | None = None,
        epoch: int | None = None,
        phase: str | None = None,
        duration_ms: float | None = None,
        steps_per_sec: float | None = None,
        snapshot: _Snapshot | None = None,
        gpu_used_mb: float | None = None,
        gpu_used_by_device_mb: dict[str, float] | None = None,
        cpu_rss_mb: float | None = None,
        cpu_vms_mb: float | None = None,
        dropped_samples: int | None = None,
        collection_ms: float | None = None,
        deep_counters: _DeepCounters | None = None,
        deep_window_active: bool | None = None,
        force_flush: bool = False,
        ts: float | None = None,
    ) -> None:
        counters = self._latest_deep_counters if deep_counters is None else deep_counters
        window_active = self._latest_deep_window_active if deep_window_active is None else deep_window_active

        event_payload = self._build_event(
            event=event,
            global_step=global_step,
            epoch=epoch,
            phase=phase,
            duration_ms=duration_ms,
            steps_per_sec=steps_per_sec,
            snapshot=snapshot,
            gpu_used_mb=self._latest_gpu_used_mb if gpu_used_mb is None else gpu_used_mb,
            gpu_used_by_device_mb=self._latest_gpu_used_by_device_mb if gpu_used_by_device_mb is None else gpu_used_by_device_mb,
            cpu_rss_mb=(snapshot.cpu_rss_mb if snapshot is not None else cpu_rss_mb),
            cpu_vms_mb=(snapshot.cpu_vms_mb if snapshot is not None else cpu_vms_mb),
            dropped_samples=dropped_samples,
            collection_ms=self._latest_collection_ms if collection_ms is None else collection_ms,
            deep_alloc_retries=None if counters is None else counters.alloc_retries,
            deep_ooms=None if counters is None else counters.ooms,
            deep_active_mb=None if counters is None else counters.active_mb,
            deep_reserved_mb=None if counters is None else counters.reserved_mb,
            deep_inactive_split_mb=None if counters is None else counters.inactive_split_mb,
            deep_window_active=window_active,
            ts=ts,
        )
        self._file_metadata_event(event_payload)
        if self._jsonl_file is None:
            return
        self._write_jsonl_event(event_payload, force_flush=force_flush)

    def _file_metadata_event(self, event_payload: Mapping[str, Any]) -> None:
        if self._metadata_runtime is None or self._run_identifier is None:
            return
        self._metadata_runtime.file(
            ResourceMonitorFacts(
                run_identifier=self._run_identifier,
                event_name=str(event_payload["event"]),
                ts=float(event_payload["ts"]),
                rank=event_payload["rank"],
                world_size=event_payload["world_size"],
                mode=event_payload["mode"],
                device_scope=event_payload["device_scope"],
                config_name=event_payload["config_name"],
                git_sha=event_payload["git_sha"],
                git_dirty=event_payload["git_dirty"],
                global_step=event_payload["global_step"],
                epoch=event_payload["epoch"],
                phase=event_payload["phase"],
                duration_ms=event_payload["duration_ms"],
                gpu_allocated_mb=event_payload["gpu_allocated_mb"],
                gpu_allocated_by_device_mb=event_payload["gpu_allocated_by_device_mb"],
                gpu_reserved_mb=event_payload["gpu_reserved_mb"],
                gpu_reserved_by_device_mb=event_payload["gpu_reserved_by_device_mb"],
                gpu_peak_allocated_mb=event_payload["gpu_peak_allocated_mb"],
                gpu_peak_allocated_by_device_mb=event_payload["gpu_peak_allocated_by_device_mb"],
                gpu_used_mb=event_payload["gpu_used_mb"],
                gpu_used_by_device_mb=event_payload["gpu_used_by_device_mb"],
                cpu_rss_mb=event_payload["cpu_rss_mb"],
                cpu_vms_mb=event_payload["cpu_vms_mb"],
                steps_per_sec=event_payload["steps_per_sec"],
                samples_per_sec=event_payload["samples_per_sec"],
                dropped_samples=event_payload["dropped_samples"],
                collection_ms=event_payload["collection_ms"],
                deep_alloc_retries=event_payload["deep_alloc_retries"],
                deep_ooms=event_payload["deep_ooms"],
                deep_active_mb=event_payload["deep_active_mb"],
                deep_reserved_mb=event_payload["deep_reserved_mb"],
                deep_inactive_split_mb=event_payload["deep_inactive_split_mb"],
                deep_window_active=event_payload["deep_window_active"],
            )
        )
