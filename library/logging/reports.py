from __future__ import annotations

import json
import logging
import platform
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from library.logging.phase_tags import is_training_epoch_phase
from library.logging.summaries import build_trainer_diagnostic_rows, diagnostic_rows_to_memory_rows
from library.metadata.dataclasses.observability import AnalyticsSnapshotFacts, RunReportFacts
from library.utils.common_utils import resolve_hydra_runtime_context

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional import safety
    OmegaConf = None


logger = logging.getLogger(__name__)


REPORT_KEY_CONFIG_PATHS: tuple[str, ...] = (
    "mode",
    "model.model_type",
    "output.saving.output_dir",
    "output.saving.output_name",
    "training.train_batch_size",
    "training.gradient_accumulation_steps",
    "training.max_train_steps",
    "training.max_train_epochs",
    "data.preprocessing.resolution",
    "data.caching.cache_latents",
    "data.caching.cache_latents_to_disk",
    "data.caching.cache_text_encoder_outputs",
    "data.caching.cache_text_encoder_outputs_to_disk",
    "data.caching.vae_batch_size",
    "data.caching.te_batch_size",
    "data.loader.num_workers",
    "data.loader.prefetch_factor",
    "data.loader.persistent_workers",
    "data.loader.pin_memory",
    "performance.precision.mixed_precision",
    "performance.precision.no_half_vae",
    "performance.memory.gradient_checkpointing",
    "performance.memory.offload_text_encoders",
    "performance.attention.xformers",
    "performance.deepspeed.deepspeed",
    "objective.prediction",
    "optimizer.optimizer_type",
    "optimizer.learning_rates.base",
    "optimizer.learning_rates.denoiser",
    "optimizer.learning_rates.text_encoders",
    "optimizer.learning_rates.groups_file",
    "output.logging.benchmark_report.enabled",
    "output.logging.resource_monitor.enabled",
    "output.logging.resource_monitor.mode",
    "output.logging.resource_monitor.output_jsonl",
    "output.logging.logging_dir",
)


@dataclass(frozen=True, slots=True)
class RunReportContext:
    """Trainer facts needed by benchmark report payload construction."""

    cfg: Any
    resource_jsonl_path: Path | None
    session_id: Any
    training_started_at: float
    mode_name: str
    strategy_name: str
    optimizer_name: str | None
    global_step: int | None
    num_train_epochs: int | None
    component_memory_estimates: list[dict[str, Any]]
    runtime_trace: dict[str, Any] | None


def _safe_get(obj: Any, dotted_path: str, default: Any = None) -> Any:
    current = obj
    for part in dotted_path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
            continue
        with suppress(Exception):
            current = getattr(current, part)
            continue
        return default
    return current


def _to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]
    if OmegaConf is not None:
        with suppress(Exception):
            return OmegaConf.to_container(value, resolve=True)
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def _render_config_yaml(cfg: Any) -> str:
    if OmegaConf is not None:
        with suppress(Exception):
            return OmegaConf.to_yaml(cfg, resolve=True)
    plain = _to_plain_data(cfg)
    return yaml.safe_dump(plain, sort_keys=False, allow_unicode=False)


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}" if value.is_integer() else f"{value:g}"
    if isinstance(value, (list, tuple)):
        return json.dumps(_to_plain_data(value))
    return str(value)


def _format_mb(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.0f} MB"


def _format_device_map_mb(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "N/A"
    formatted_items: list[str] = []
    for device, amount in sorted(value.items()):
        if not isinstance(device, str) or not isinstance(amount, (int, float)):
            continue
        formatted_items.append(f"{device}: {_format_mb(amount)}")
    return ", ".join(formatted_items) if formatted_items else "N/A"


def _format_seconds(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f}s"


def is_benchmark_report_enabled(cfg: Any) -> bool:
    """Return whether benchmark-style run reports are enabled for this config."""
    return _safe_get(cfg, "output.logging.benchmark_report.enabled", False) is True


def _gpu_info() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        device_name = torch.cuda.get_device_name(0)
        total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        return f"{device_name}, {total_memory_mb:.0f} MiB"
    except Exception:
        return "CUDA device present"


def _estimate_component_memory_rows(trainer: Any) -> list[dict[str, Any]]:
    try:
        rows, _aliases = build_trainer_diagnostic_rows(trainer)
    except Exception:
        return []

    optimizer_name = getattr(trainer, "optimizer_name", "") or ""
    return diagnostic_rows_to_memory_rows(rows, optimizer_name)


def _normalize_path(path: Any) -> Path | None:
    if path is None:
        return None
    if isinstance(path, Path):
        return path
    return Path(str(path))


def _build_run_report_context(trainer: Any) -> RunReportContext:
    resource_monitor = getattr(trainer, "_resource_monitor", None)
    runtime_trace = getattr(trainer, "runtime_trace", None)
    return RunReportContext(
        cfg=trainer.cfg,
        resource_jsonl_path=_normalize_path(getattr(resource_monitor, "jsonl_path", None)),
        session_id=getattr(trainer, "session_id", None),
        training_started_at=getattr(trainer, "training_started_at", time.time()),
        mode_name=type(trainer.mode).__name__,
        strategy_name=type(trainer.strategies).__name__,
        optimizer_name=getattr(trainer, "optimizer_name", None),
        global_step=getattr(trainer, "global_step", None),
        num_train_epochs=getattr(trainer, "num_train_epochs", None),
        component_memory_estimates=_estimate_component_memory_rows(trainer),
        runtime_trace=runtime_trace.summary() if runtime_trace is not None else None,
    )


def _load_resource_events(jsonl_path: Path | None) -> list[dict[str, Any]]:
    if jsonl_path is None or not jsonl_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        with suppress(json.JSONDecodeError):
            events.append(json.loads(stripped))
    return events


def _events_for_run(events: list[dict[str, Any]], run_identifier: Any) -> list[dict[str, Any]]:
    normalized_run_identifier = _format_scalar(run_identifier)
    run_events = [event for event in events if _format_scalar(event.get("run_identifier")) == normalized_run_identifier]
    return run_events or events


def _pair_phase_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_starts: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    phase_rows: list[dict[str, Any]] = []
    for event in events:
        phase_name = event.get("phase")
        if not phase_name:
            continue
        if event.get("event") == "phase_start":
            phase_starts[phase_name].append(event)
            continue
        if event.get("event") != "phase_end":
            continue
        start = phase_starts[phase_name].pop() if phase_starts[phase_name] else {}
        start_ts = start.get("ts")
        end_ts = event.get("ts")
        phase_samples = [
            sample
            for sample in events
            if sample.get("event") == "step_sample"
            and sample.get("phase") is None
            and sample.get("gpu_used_mb") is not None
            and isinstance(sample.get("ts"), (int, float))
            and isinstance(start_ts, (int, float))
            and isinstance(end_ts, (int, float))
            and start_ts <= sample["ts"] <= end_ts
        ]
        gpu_used_peak_mb = max((float(sample["gpu_used_mb"]) for sample in phase_samples), default=None)
        gpu_used_peak_by_device_mb = _peak_device_map(phase_samples, "gpu_used_by_device_mb")
        phase_rows.append(
            {
                "phase": phase_name,
                "duration_s": (event.get("duration_ms") or 0.0) / 1000.0 if event.get("duration_ms") is not None else None,
                "gpu_allocated_start_mb": start.get("gpu_allocated_mb"),
                "gpu_allocated_start_by_device_mb": start.get("gpu_allocated_by_device_mb"),
                "gpu_allocated_end_mb": event.get("gpu_allocated_mb"),
                "gpu_allocated_end_by_device_mb": event.get("gpu_allocated_by_device_mb"),
                "gpu_reserved_start_mb": start.get("gpu_reserved_mb"),
                "gpu_reserved_start_by_device_mb": start.get("gpu_reserved_by_device_mb"),
                "gpu_reserved_end_mb": event.get("gpu_reserved_mb"),
                "gpu_reserved_end_by_device_mb": event.get("gpu_reserved_by_device_mb"),
                "gpu_peak_allocated_mb": event.get("gpu_peak_allocated_mb"),
                "gpu_peak_allocated_by_device_mb": event.get("gpu_peak_allocated_by_device_mb"),
                "gpu_used_peak_mb": gpu_used_peak_mb,
                "gpu_used_peak_by_device_mb": gpu_used_peak_by_device_mb,
                "cpu_rss_start_mb": start.get("cpu_rss_mb"),
                "cpu_rss_end_mb": event.get("cpu_rss_mb"),
                "cpu_vms_start_mb": start.get("cpu_vms_mb"),
                "cpu_vms_end_mb": event.get("cpu_vms_mb"),
            }
        )
    return phase_rows


def _peak_device_map(events: list[dict[str, Any]], field_name: str) -> dict[str, float] | None:
    peaks: dict[str, float] = {}
    for event in events:
        raw_map = event.get(field_name)
        if not isinstance(raw_map, dict):
            continue
        for device, value in raw_map.items():
            if not isinstance(device, str) or not isinstance(value, (int, float)):
                continue
            numeric_value = float(value)
            previous = peaks.get(device)
            if previous is None or numeric_value > previous:
                peaks[device] = numeric_value
    return peaks or None


def _device_map_to_rows(device_map: dict[str, float] | None, *, value_key: str) -> list[dict[str, Any]]:
    if not isinstance(device_map, dict):
        return []
    rows: list[dict[str, Any]] = []
    for device, value in sorted(device_map.items()):
        if not isinstance(device, str) or not isinstance(value, (int, float)):
            continue
        rows.append({"device": device, value_key: float(value)})
    return rows


def _build_resource_debug_summary(
    *,
    session_start: dict[str, Any] | None,
    session_end: dict[str, Any] | None,
    gpu_used_peak_session_by_device_mb: dict[str, float] | None,
    phase_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "surface_split": {
            "live_metadata_fields": [
                "gpu_allocated_mb",
                "gpu_allocated_by_device_mb",
                "gpu_reserved_mb",
                "gpu_reserved_by_device_mb",
                "gpu_peak_allocated_mb",
                "gpu_peak_allocated_by_device_mb",
                "gpu_used_mb",
                "gpu_used_by_device_mb",
                "cpu_rss_mb",
                "cpu_vms_mb",
            ],
            "report_debug_only_fields": [
                "session_gpu_used_peak_by_device_rows",
                "phase_device_rows",
            ],
        },
        "session_gpu_used_peak_by_device_rows": _device_map_to_rows(
            gpu_used_peak_session_by_device_mb,
            value_key="gpu_used_peak_mb",
        ),
        "phase_device_rows": [
            {
                "phase": row.get("phase"),
                "gpu_allocated_start_by_device_mb": row.get("gpu_allocated_start_by_device_mb"),
                "gpu_allocated_end_by_device_mb": row.get("gpu_allocated_end_by_device_mb"),
                "gpu_reserved_start_by_device_mb": row.get("gpu_reserved_start_by_device_mb"),
                "gpu_reserved_end_by_device_mb": row.get("gpu_reserved_end_by_device_mb"),
                "gpu_peak_allocated_by_device_mb": row.get("gpu_peak_allocated_by_device_mb"),
                "gpu_used_peak_by_device_mb": row.get("gpu_used_peak_by_device_mb"),
            }
            for row in phase_rows
            if any(
                row.get(field) is not None
                for field in (
                    "gpu_allocated_start_by_device_mb",
                    "gpu_allocated_end_by_device_mb",
                    "gpu_reserved_start_by_device_mb",
                    "gpu_reserved_end_by_device_mb",
                    "gpu_peak_allocated_by_device_mb",
                    "gpu_used_peak_by_device_mb",
                )
            )
        ],
        "session_allocator_by_device": {
            "gpu_allocated_start_by_device_mb": None if session_start is None else session_start.get("gpu_allocated_by_device_mb"),
            "gpu_allocated_end_by_device_mb": None if session_end is None else session_end.get("gpu_allocated_by_device_mb"),
            "gpu_reserved_start_by_device_mb": None if session_start is None else session_start.get("gpu_reserved_by_device_mb"),
            "gpu_reserved_end_by_device_mb": None if session_end is None else session_end.get("gpu_reserved_by_device_mb"),
            "gpu_peak_allocated_end_by_device_mb": None
            if session_end is None
            else session_end.get("gpu_peak_allocated_by_device_mb"),
        },
    }


def _collect_trace_only_phase_rows(
    runtime_trace: dict[str, Any] | None,
    resource_phase_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(runtime_trace, dict):
        return []

    trace_phase_rows = runtime_trace.get("phases")
    if not isinstance(trace_phase_rows, list):
        return []

    resource_phase_names = {
        str(row["phase"]) for row in resource_phase_rows if isinstance(row, dict) and row.get("phase") is not None
    }

    trace_only_rows: list[dict[str, Any]] = []
    for row in trace_phase_rows:
        if not isinstance(row, dict):
            continue
        tag = row.get("tag")
        if not isinstance(tag, str) or tag in resource_phase_names:
            continue
        trace_only_rows.append(
            {
                "tag": tag,
                "start_offset_s": row.get("start_offset_s"),
                "end_offset_s": row.get("end_offset_s"),
                "duration_s": row.get("duration_s"),
                "coverage_note": "Trace-only span with no matching resource-monitor phase row.",
            }
        )
    return trace_only_rows


def _collect_key_config_rows(cfg: Any, *, hydra_config_name: str | None, hydra_overrides: list[str]) -> list[tuple[str, Any]]:
    key_paths = [("hydra.config_name", hydra_config_name)]
    key_paths.extend((path, _safe_get(cfg, path)) for path in REPORT_KEY_CONFIG_PATHS)
    rows = [(key, value) for key, value in key_paths if value is not None]
    if hydra_overrides:
        rows.append(("hydra.task_overrides", hydra_overrides))
    return rows


def _build_report_payload(trainer: Any, *, succeeded: bool, error_message: str | None) -> dict[str, Any]:
    return _build_report_payload_from_context(
        _build_run_report_context(trainer),
        succeeded=succeeded,
        error_message=error_message,
    )


def _build_report_payload_from_context(
    context: RunReportContext,
    *,
    succeeded: bool,
    error_message: str | None,
) -> dict[str, Any]:
    hydra_config_name, hydra_overrides = resolve_hydra_runtime_context()
    jsonl_path = context.resource_jsonl_path
    all_events = _load_resource_events(jsonl_path)
    events = _events_for_run(all_events, context.session_id)
    phase_rows = _pair_phase_events(events)
    runtime_trace = context.runtime_trace
    if isinstance(context.runtime_trace, dict):
        runtime_trace = dict(context.runtime_trace)
        runtime_trace["trace_only_phases"] = _collect_trace_only_phase_rows(runtime_trace, phase_rows)
    session_start = next((event for event in events if event.get("event") == "session_start"), None)
    session_end = next((event for event in reversed(events) if event.get("event") == "session_end"), None)
    finished_at = time.time()
    gpu_used_peak_session_mb = max(
        (float(event["gpu_used_mb"]) for event in events if event.get("gpu_used_mb") is not None),
        default=None,
    )
    gpu_used_peak_session_by_device_mb = _peak_device_map(events, "gpu_used_by_device_mb")

    training_phases = [row for row in phase_rows if is_training_epoch_phase(str(row["phase"]))]
    training_duration_s = sum(row["duration_s"] or 0.0 for row in training_phases)
    optimization_steps = context.global_step or _safe_get(context.cfg, "training.max_train_steps", 0) or 0
    train_seconds_per_step = training_duration_s / optimization_steps if optimization_steps else None

    return {
        "status": "succeeded" if succeeded else "failed",
        "error_message": error_message,
        "generated_at": finished_at,
        "training_started_at": context.training_started_at,
        "total_duration_s": finished_at - context.training_started_at,
        "session_id": context.session_id,
        "hydra_config_name": hydra_config_name,
        "hydra_task_overrides": hydra_overrides,
        "output_name": _safe_get(context.cfg, "output.saving.output_name"),
        "output_dir": _safe_get(context.cfg, "output.saving.output_dir"),
        "mode_name": context.mode_name,
        "strategy_name": context.strategy_name,
        "optimizer_name": context.optimizer_name,
        "global_step": context.global_step,
        "num_train_epochs": context.num_train_epochs,
        "train_seconds_per_step": train_seconds_per_step,
        "system": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "gpu": _gpu_info(),
            "cuda_available": torch.cuda.is_available(),
        },
        "runtime_trace": runtime_trace,
        "resource_monitor": {
            "jsonl_path": None if jsonl_path is None else str(jsonl_path),
            "event_count": len(events),
            "total_jsonl_event_count": len(all_events),
            "session_start": session_start,
            "session_end": session_end,
            "gpu_used_peak_session_mb": gpu_used_peak_session_mb,
            "gpu_used_peak_session_by_device_mb": gpu_used_peak_session_by_device_mb,
            "phases": phase_rows,
            "debug": _build_resource_debug_summary(
                session_start=session_start,
                session_end=session_end,
                gpu_used_peak_session_by_device_mb=gpu_used_peak_session_by_device_mb,
                phase_rows=phase_rows,
            ),
        },
        "component_memory_estimates": context.component_memory_estimates,
        "key_config": _collect_key_config_rows(context.cfg, hydra_config_name=hydra_config_name, hydra_overrides=hydra_overrides),
        "include_full_config": _safe_get(context.cfg, "output.logging.benchmark_report.include_full_config", True) is not False,
        "full_config_yaml": _render_config_yaml(context.cfg),
    }


def _render_report_markdown(payload: dict[str, Any]) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(payload["generated_at"]))
    output_name = payload.get("output_name") or "unnamed_run"
    hydra_config_name = payload.get("hydra_config_name") or "unknown"
    training_speed = "N/A"
    if payload.get("train_seconds_per_step") is not None:
        training_speed = f"{_format_seconds(payload.get('train_seconds_per_step'))} / it"

    lines = [
        "# Benchmark Report",
        "",
        f"**Date:** {generated_at}",
        f"**Status:** {payload['status']}",
        f"**Config:** {hydra_config_name}",
        f"**Output Name:** {output_name}",
        f"**Total Time:** {_format_seconds(payload.get('total_duration_s'))}",
    ]
    if payload.get("error_message"):
        lines.extend(["", f"**Error:** `{payload['error_message']}`"])

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Key | Value |",
            "|-----|-------|",
            f"| Status | {payload['status']} |",
            f"| Config | {hydra_config_name} |",
            f"| Output Name | {output_name} |",
            f"| Total Time | {_format_seconds(payload.get('total_duration_s'))} |",
            "",
            "## Runtime",
            "",
            "| Key | Value |",
            "|-----|-------|",
            f"| Mode | {payload.get('mode_name')} |",
            f"| Strategy | {payload.get('strategy_name')} |",
            f"| Optimizer | {payload.get('optimizer_name')} |",
            f"| Global Step | {_format_scalar(payload.get('global_step'))} |",
            f"| Num Epochs | {_format_scalar(payload.get('num_train_epochs'))} |",
            f"| Avg Train Step Time | {_format_seconds(payload.get('train_seconds_per_step'))} |",
            f"| Session ID | {_format_scalar(payload.get('session_id'))} |",
            f"| Output Dir | {_format_scalar(payload.get('output_dir'))} |",
        ]
    )

    lines.extend(
        [
            "",
            "## System",
            "",
            "| Key | Value |",
            "|-----|-------|",
            f"| GPU | {payload['system']['gpu']} |",
            f"| Python | {payload['system']['python']} |",
            f"| PyTorch | {payload['system']['pytorch']} |",
        ]
    )

    lines.extend(
        [
            "",
            "## Key Configuration",
            "",
            "| YAML Key | Value |",
            "|----------|-------|",
        ]
    )
    for key, value in payload["key_config"]:
        lines.append(f"| `{key}` | {_format_scalar(value)} |")

    lines.extend(
        [
            "",
            "## Speed Results",
            "",
            "| Phase | Final Speed |",
            "|-------|-------------|",
            "| Latent Caching | N/A |",
            "| TE Caching | N/A |",
            f"| Training | {training_speed} |",
        ]
    )

    runtime_trace = payload.get("runtime_trace")
    if isinstance(runtime_trace, dict):
        milestone_rows = [
            ("Launch -> Progress Bar", runtime_trace.get("milestones", {}).get("time_to_progress_bar_s")),
            ("Launch -> First Step Started", runtime_trace.get("milestones", {}).get("time_to_first_step_started_s")),
            ("Launch -> First Synced Step", runtime_trace.get("milestones", {}).get("time_to_first_synced_step_s")),
            ("Launch -> Run Ended", runtime_trace.get("milestones", {}).get("time_to_run_ended_s")),
        ]
        non_null_milestones = [(label, value) for label, value in milestone_rows if value is not None]
        phase_totals = runtime_trace.get("phase_totals")
        trace_only_phase_rows = runtime_trace.get("trace_only_phases")
        slowest_phase_rows = []
        if isinstance(phase_totals, dict):
            slowest_phase_rows = sorted(
                ((str(tag), total) for tag, total in phase_totals.items() if isinstance(total, (int, float))),
                key=lambda item: item[1],
                reverse=True,
            )[:5]

        if non_null_milestones:
            lines.extend(
                [
                    "",
                    "## Runtime Trace",
                    "",
                    "| Milestone | Time |",
                    "|-----------|------|",
                ]
            )
            for label, value in non_null_milestones:
                lines.append(f"| {label} | {_format_seconds(value)} |")

        if slowest_phase_rows:
            lines.extend(
                [
                    "",
                    "| Slowest Phase | Total Time |",
                    "|---------------|------------|",
                ]
            )
            for tag, total in slowest_phase_rows:
                lines.append(f"| `{tag}` | {_format_seconds(total)} |")

        if isinstance(trace_only_phase_rows, list) and trace_only_phase_rows:
            lines.extend(
                [
                    "",
                    "## Trace-only Phases",
                    "",
                    "| Phase | Duration | Note |",
                    "|-------|----------|------|",
                ]
            )
            for row in trace_only_phase_rows:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| `{row.get('tag')}` | {_format_seconds(row.get('duration_s'))} | {row.get('coverage_note', 'Trace-only span')} |"
                )

    component_rows = payload.get("component_memory_estimates") or []
    if component_rows:
        lines.extend(
            [
                "",
                "## Startup Estimates",
                "",
                "Optimizer state memory is estimated from optimizer family and trainable parameter size.",
                "",
                "| Component | Modules Trainable / Total | Params Trainable / Total | Param MB | Grad MB Est | Optimizer State MB Est |",
                "|-----------|---------------------------|---------------------------|----------|-------------|------------------------|",
            ]
        )
        for row in component_rows:
            lines.append(
                f"| `{row['name']}` | {row['modules_trainable']:,} / {row['modules_total']:,} | "
                f"{row['params_trainable']:,} / {row['params_total']:,} | "
                f"{row['param_mb']:.1f} | {row['gradient_mb_est']:.1f} | {row['optimizer_state_mb_est']:.1f} |"
            )

    resource = payload["resource_monitor"]
    session_end = resource.get("session_end")
    if session_end is not None:
        lines.extend(
            [
                "",
                "## Session Summary",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Duration | {_format_seconds((session_end.get('duration_ms') or 0.0) / 1000.0 if session_end.get('duration_ms') is not None else None)} |",
                f"| GPU Allocated | {_format_mb(session_end.get('gpu_allocated_mb'))} |",
                f"| GPU Reserved | {_format_mb(session_end.get('gpu_reserved_mb'))} |",
                f"| GPU Peak Allocated | {_format_mb(session_end.get('gpu_peak_allocated_mb'))} |",
                f"| GPU Used | {_format_mb(session_end.get('gpu_used_mb'))} |",
                f"| GPU Used Peak | {_format_mb(resource.get('gpu_used_peak_session_mb'))} |",
                f"| CPU RSS | {_format_mb(session_end.get('cpu_rss_mb'))} |",
                f"| CPU VMS | {_format_mb(session_end.get('cpu_vms_mb'))} |",
                f"| Resource Events | {_format_scalar(resource.get('event_count'))} |",
                f"| Resource JSONL | {_format_scalar(resource.get('jsonl_path'))} |",
            ]
        )

    phases = resource.get("phases") or []
    if phases:
        lines.extend(
            [
                "",
                "## Phase Resource Summary",
                "",
                "| Phase | Duration | GPU Allocated | GPU Reserved | GPU Peak Allocated | GPU Used Peak | CPU RSS | CPU VMS |",
                "|-------|----------|---------------|--------------|--------------------|---------------|---------|---------|",
            ]
        )
        for row in phases:
            lines.append(
                f"| `{row['phase']}` | {_format_seconds(row['duration_s'])} | "
                f"{_format_mb(row['gpu_allocated_start_mb'])} -> {_format_mb(row['gpu_allocated_end_mb'])} | "
                f"{_format_mb(row['gpu_reserved_start_mb'])} -> {_format_mb(row['gpu_reserved_end_mb'])} | "
                f"{_format_mb(row['gpu_peak_allocated_mb'])} | "
                f"{_format_mb(row['gpu_used_peak_mb'])} | "
                f"{_format_mb(row['cpu_rss_start_mb'])} -> {_format_mb(row['cpu_rss_end_mb'])} | "
                f"{_format_mb(row['cpu_vms_start_mb'])} -> {_format_mb(row['cpu_vms_end_mb'])} |"
            )

    resource_debug = resource.get("debug")
    if isinstance(resource_debug, dict):
        session_device_rows = resource_debug.get("session_gpu_used_peak_by_device_rows") or []
        if session_device_rows:
            lines.extend(
                [
                    "",
                    "## Per-Device GPU Session Peaks",
                    "",
                    "| Device | GPU Used Peak |",
                    "|--------|---------------|",
                ]
            )
            for row in session_device_rows:
                if not isinstance(row, dict):
                    continue
                lines.append(f"| `{row.get('device')}` | {_format_mb(row.get('gpu_used_peak_mb'))} |")

        phase_device_rows = resource_debug.get("phase_device_rows") or []
        if phase_device_rows:
            lines.extend(
                [
                    "",
                    "## Per-Device GPU Phase Details",
                    "",
                    "| Phase | GPU Used Peak by Device | GPU Peak Allocated by Device |",
                    "|-------|-------------------------|------------------------------|",
                ]
            )
            for row in phase_device_rows:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| `{row.get('phase')}` | "
                    f"{_format_device_map_mb(row.get('gpu_used_peak_by_device_mb'))} | "
                    f"{_format_device_map_mb(row.get('gpu_peak_allocated_by_device_mb'))} |"
                )

    if payload.get("include_full_config", True):
        lines.extend(
            [
                "",
                "## Full Composed Config",
                "",
                "The full resolved Hydra config is included here so the report is self-contained.",
                "",
                "```yaml",
                payload["full_config_yaml"].rstrip(),
                "```",
            ]
        )

    return "\n".join(lines) + "\n"


def write_run_report(trainer: Any, *, succeeded: bool, error_message: str | None = None) -> Path | None:
    output_dir = _safe_get(trainer.cfg, "output.logging.benchmark_report.output_dir") or _safe_get(trainer.cfg, "output.saving.output_dir")
    if not output_dir:
        return None

    output_path = Path(str(output_dir))
    output_path.mkdir(parents=True, exist_ok=True)

    payload = _build_report_payload(trainer, succeeded=succeeded, error_message=error_message)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(payload["generated_at"]))
    filename_prefix = _safe_get(trainer.cfg, "output.logging.benchmark_report.filename_prefix", "benchmark_report")
    if not isinstance(filename_prefix, str) or not filename_prefix.strip():
        filename_prefix = "benchmark_report"
    report_base = output_path / f"{filename_prefix.strip()}_{timestamp}"

    markdown_path = report_base.with_suffix(".md")
    json_path = report_base.with_suffix(".json")

    markdown_path.write_text(_render_report_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(_to_plain_data(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    _file_run_report_metadata(trainer, markdown_path=markdown_path, payload=payload)
    return markdown_path


def _file_run_report_metadata(trainer: Any, *, markdown_path: Path, payload: dict[str, Any]) -> None:
    observer = getattr(trainer, "_observer", None)
    metadata_runtime = getattr(observer, "metadata_runtime", None)
    if metadata_runtime is None:
        return

    run_identifier = _resolve_report_run_identifier(trainer, observer, payload)
    if run_identifier is None:
        logger.warning("Skipping run-report metadata filing because no run identifier is available.")
        return

    generated_at_value = payload.get("generated_at")
    generated_at = float(generated_at_value) if generated_at_value is not None else time.time()

    metadata_runtime.file(_build_run_report_facts(markdown_path, payload, run_identifier=str(run_identifier), generated_at=generated_at))
    metadata_runtime.file(
        _build_run_report_snapshot_facts(markdown_path, payload, run_identifier=str(run_identifier), generated_at=generated_at)
    )


def _resolve_report_run_identifier(trainer: Any, observer: Any, payload: dict[str, Any]) -> str | None:
    run_identifier = payload.get("session_id")
    if run_identifier is None:
        run_identifier = getattr(trainer, "session_id", None)
    if run_identifier is None:
        run_identifier = getattr(observer, "run_identifier", None)
    if run_identifier is None:
        run_identifier = payload.get("output_name")
    return str(run_identifier) if run_identifier is not None else None


def _extract_report_resource_fields(payload: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    resource_monitor = payload.get("resource_monitor")
    if not isinstance(resource_monitor, dict):
        return None, None, None

    event_count = resource_monitor.get("event_count")
    resource_event_count = event_count if isinstance(event_count, int) else None

    phases = resource_monitor.get("phases")
    phase_count = len(phases) if isinstance(phases, list) else None

    jsonl_path = resource_monitor.get("jsonl_path")
    resource_jsonl_path = jsonl_path if isinstance(jsonl_path, str) else None
    return resource_event_count, phase_count, resource_jsonl_path


def _build_run_report_facts(
    markdown_path: Path,
    payload: dict[str, Any],
    *,
    run_identifier: str,
    generated_at: float,
) -> RunReportFacts:
    resource_event_count, phase_count, resource_jsonl_path = _extract_report_resource_fields(payload)
    return RunReportFacts(
        report_identifier=str(markdown_path),
        run_identifier=run_identifier,
        status=str(payload.get("status") or "unknown"),
        generated_at=generated_at,
        output_name=payload.get("output_name") if isinstance(payload.get("output_name"), str) else None,
        output_dir=payload.get("output_dir") if isinstance(payload.get("output_dir"), str) else None,
        mode_name=payload.get("mode_name") if isinstance(payload.get("mode_name"), str) else None,
        strategy_name=payload.get("strategy_name") if isinstance(payload.get("strategy_name"), str) else None,
        optimizer_name=payload.get("optimizer_name") if isinstance(payload.get("optimizer_name"), str) else None,
        global_step=payload.get("global_step") if isinstance(payload.get("global_step"), int) else None,
        num_train_epochs=payload.get("num_train_epochs") if isinstance(payload.get("num_train_epochs"), int) else None,
        resource_event_count=resource_event_count,
        phase_count=phase_count,
        include_full_config=payload.get("include_full_config") if isinstance(payload.get("include_full_config"), bool) else None,
        resource_jsonl_path=resource_jsonl_path,
    )


def _build_run_report_snapshot_facts(
    markdown_path: Path,
    payload: dict[str, Any],
    *,
    run_identifier: str,
    generated_at: float,
) -> AnalyticsSnapshotFacts:
    return AnalyticsSnapshotFacts(
        snapshot_identifier=f"{markdown_path.with_suffix('.json')}#payload",
        snapshot_kind="benchmark_report_payload",
        source="library.logging.reports.write_run_report",
        payload=_build_report_analytics_snapshot_payload(payload),
        generated_at=generated_at,
        run_identifier=run_identifier,
    )


def _build_report_analytics_snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot_payload = {
        "status": payload.get("status"),
        "error_message": payload.get("error_message"),
        "generated_at": payload.get("generated_at"),
        "training_started_at": payload.get("training_started_at"),
        "total_duration_s": payload.get("total_duration_s"),
        "session_id": payload.get("session_id"),
        "hydra_config_name": payload.get("hydra_config_name"),
        "hydra_task_overrides": payload.get("hydra_task_overrides"),
        "output_name": payload.get("output_name"),
        "output_dir": payload.get("output_dir"),
        "mode_name": payload.get("mode_name"),
        "strategy_name": payload.get("strategy_name"),
        "optimizer_name": payload.get("optimizer_name"),
        "global_step": payload.get("global_step"),
        "num_train_epochs": payload.get("num_train_epochs"),
        "train_seconds_per_step": payload.get("train_seconds_per_step"),
        "system": payload.get("system"),
        "resource_monitor": payload.get("resource_monitor"),
        "component_memory_estimates": payload.get("component_memory_estimates"),
        "runtime_trace": payload.get("runtime_trace"),
        "key_config": payload.get("key_config"),
        "include_full_config": payload.get("include_full_config"),
    }
    return {key: value for key, value in snapshot_payload.items() if value is not None}
