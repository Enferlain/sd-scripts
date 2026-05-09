from __future__ import annotations

import json
import platform
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from library.logging.summaries import build_trainer_diagnostic_rows, diagnostic_rows_to_memory_rows

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional import safety
    OmegaConf = None


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


def _resolve_hydra_context() -> tuple[str | None, list[str]]:
    with suppress(Exception):
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            hydra_cfg = HydraConfig.get()
            config_name = getattr(hydra_cfg.job, "config_name", None)
            task_overrides = list(getattr(hydra_cfg.overrides, "task", []) or [])
            return config_name, task_overrides
    return None, []


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


def _events_for_run(events: list[dict[str, Any]], run_id: Any) -> list[dict[str, Any]]:
    normalized_run_id = _format_scalar(run_id)
    run_events = [event for event in events if _format_scalar(event.get("run_id")) == normalized_run_id]
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
        phase_rows.append(
            {
                "phase": phase_name,
                "duration_s": (event.get("duration_ms") or 0.0) / 1000.0 if event.get("duration_ms") is not None else None,
                "gpu_allocated_start_mb": start.get("gpu_allocated_mb"),
                "gpu_allocated_end_mb": event.get("gpu_allocated_mb"),
                "gpu_reserved_start_mb": start.get("gpu_reserved_mb"),
                "gpu_reserved_end_mb": event.get("gpu_reserved_mb"),
                "gpu_peak_allocated_mb": event.get("gpu_peak_allocated_mb"),
                "gpu_used_peak_mb": gpu_used_peak_mb,
                "cpu_rss_start_mb": start.get("cpu_rss_mb"),
                "cpu_rss_end_mb": event.get("cpu_rss_mb"),
            }
        )
    return phase_rows


def _collect_key_config_rows(trainer: Any, *, hydra_config_name: str | None, hydra_overrides: list[str]) -> list[tuple[str, Any]]:
    key_paths = [
        ("hydra.config_name", hydra_config_name),
        ("mode", _safe_get(trainer.cfg, "mode")),
        ("model.model_type", _safe_get(trainer.cfg, "model.model_type")),
        ("output.saving.output_dir", _safe_get(trainer.cfg, "output.saving.output_dir")),
        ("output.saving.output_name", _safe_get(trainer.cfg, "output.saving.output_name")),
        ("training.train_batch_size", _safe_get(trainer.cfg, "training.train_batch_size")),
        ("training.gradient_accumulation_steps", _safe_get(trainer.cfg, "training.gradient_accumulation_steps")),
        ("training.max_train_steps", _safe_get(trainer.cfg, "training.max_train_steps")),
        ("training.max_train_epochs", _safe_get(trainer.cfg, "training.max_train_epochs")),
        ("data.preprocessing.resolution", _safe_get(trainer.cfg, "data.preprocessing.resolution")),
        ("data.caching.cache_latents", _safe_get(trainer.cfg, "data.caching.cache_latents")),
        ("data.caching.cache_latents_to_disk", _safe_get(trainer.cfg, "data.caching.cache_latents_to_disk")),
        ("data.caching.cache_text_encoder_outputs", _safe_get(trainer.cfg, "data.caching.cache_text_encoder_outputs")),
        ("data.caching.cache_text_encoder_outputs_to_disk", _safe_get(trainer.cfg, "data.caching.cache_text_encoder_outputs_to_disk")),
        ("data.caching.vae_batch_size", _safe_get(trainer.cfg, "data.caching.vae_batch_size")),
        ("data.caching.te_batch_size", _safe_get(trainer.cfg, "data.caching.te_batch_size")),
        ("data.loader.num_workers", _safe_get(trainer.cfg, "data.loader.num_workers")),
        ("data.loader.prefetch_factor", _safe_get(trainer.cfg, "data.loader.prefetch_factor")),
        ("data.loader.persistent_workers", _safe_get(trainer.cfg, "data.loader.persistent_workers")),
        ("data.loader.pin_memory", _safe_get(trainer.cfg, "data.loader.pin_memory")),
        ("performance.precision.mixed_precision", _safe_get(trainer.cfg, "performance.precision.mixed_precision")),
        ("performance.precision.no_half_vae", _safe_get(trainer.cfg, "performance.precision.no_half_vae")),
        ("performance.memory.gradient_checkpointing", _safe_get(trainer.cfg, "performance.memory.gradient_checkpointing")),
        ("performance.memory.offload_text_encoders", _safe_get(trainer.cfg, "performance.memory.offload_text_encoders")),
        ("performance.attention.xformers", _safe_get(trainer.cfg, "performance.attention.xformers")),
        ("performance.deepspeed.deepspeed", _safe_get(trainer.cfg, "performance.deepspeed.deepspeed")),
        ("objective.prediction", _safe_get(trainer.cfg, "objective.prediction")),
        ("optimizer.optimizer_type", _safe_get(trainer.cfg, "optimizer.optimizer_type")),
        ("optimizer.learning_rates.base", _safe_get(trainer.cfg, "optimizer.learning_rates.base")),
        ("optimizer.learning_rates.denoiser", _safe_get(trainer.cfg, "optimizer.learning_rates.denoiser")),
        ("optimizer.learning_rates.text_encoders", _safe_get(trainer.cfg, "optimizer.learning_rates.text_encoders")),
        ("optimizer.learning_rates.groups_file", _safe_get(trainer.cfg, "optimizer.learning_rates.groups_file")),
        ("adapter.peft.lora", _safe_get(trainer.cfg, "adapter.peft.lora")),
        ("adapter.peft.loha", _safe_get(trainer.cfg, "adapter.peft.loha")),
        ("output.logging.benchmark_report.enabled", _safe_get(trainer.cfg, "output.logging.benchmark_report.enabled")),
        ("output.logging.resource_monitor.enabled", _safe_get(trainer.cfg, "output.logging.resource_monitor.enabled")),
        ("output.logging.resource_monitor.mode", _safe_get(trainer.cfg, "output.logging.resource_monitor.mode")),
        ("output.logging.resource_monitor.output_jsonl", _safe_get(trainer.cfg, "output.logging.resource_monitor.output_jsonl")),
        ("output.logging.logging_dir", _safe_get(trainer.cfg, "output.logging.logging_dir")),
    ]
    rows = [(key, value) for key, value in key_paths if value is not None]
    if hydra_overrides:
        rows.append(("hydra.task_overrides", hydra_overrides))
    return rows


def _build_report_payload(trainer: Any, *, succeeded: bool, error_message: str | None) -> dict[str, Any]:
    hydra_config_name, hydra_overrides = _resolve_hydra_context()
    jsonl_path = getattr(trainer._resource_monitor, "jsonl_path", None)
    all_events = _load_resource_events(jsonl_path)
    events = _events_for_run(all_events, getattr(trainer, "session_id", None))
    phase_rows = _pair_phase_events(events)
    session_start = next((event for event in events if event.get("event") == "session_start"), None)
    session_end = next((event for event in reversed(events) if event.get("event") == "session_end"), None)
    finished_at = time.time()
    gpu_used_peak_session_mb = max(
        (float(event["gpu_used_mb"]) for event in events if event.get("gpu_used_mb") is not None),
        default=None,
    )

    training_phases = [row for row in phase_rows if str(row["phase"]).startswith("training_epoch_")]
    training_duration_s = sum(row["duration_s"] or 0.0 for row in training_phases)
    optimization_steps = getattr(trainer, "global_step", 0) or _safe_get(trainer.cfg, "training.max_train_steps", 0) or 0
    train_seconds_per_step = training_duration_s / optimization_steps if optimization_steps else None

    return {
        "status": "succeeded" if succeeded else "failed",
        "error_message": error_message,
        "generated_at": finished_at,
        "training_started_at": getattr(trainer, "training_started_at", finished_at),
        "total_duration_s": finished_at - getattr(trainer, "training_started_at", finished_at),
        "session_id": getattr(trainer, "session_id", None),
        "hydra_config_name": hydra_config_name,
        "hydra_task_overrides": hydra_overrides,
        "output_name": _safe_get(trainer.cfg, "output.saving.output_name"),
        "output_dir": _safe_get(trainer.cfg, "output.saving.output_dir"),
        "mode_name": type(trainer.mode).__name__,
        "strategy_name": type(trainer.strategies).__name__,
        "optimizer_name": getattr(trainer, "optimizer_name", None),
        "global_step": getattr(trainer, "global_step", None),
        "num_train_epochs": getattr(trainer, "num_train_epochs", None),
        "train_seconds_per_step": train_seconds_per_step,
        "system": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "gpu": _gpu_info(),
            "cuda_available": torch.cuda.is_available(),
        },
        "resource_monitor": {
            "jsonl_path": None if jsonl_path is None else str(jsonl_path),
            "event_count": len(events),
            "total_jsonl_event_count": len(all_events),
            "session_start": session_start,
            "session_end": session_end,
            "gpu_used_peak_session_mb": gpu_used_peak_session_mb,
            "phases": phase_rows,
        },
        "component_memory_estimates": _estimate_component_memory_rows(trainer),
        "key_config": _collect_key_config_rows(trainer, hydra_config_name=hydra_config_name, hydra_overrides=hydra_overrides),
        "include_full_config": _safe_get(trainer.cfg, "output.logging.benchmark_report.include_full_config", True) is not False,
        "full_config_yaml": _render_config_yaml(trainer.cfg),
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
                "| Phase | Duration | GPU Allocated | GPU Reserved | GPU Peak Allocated | GPU Used Peak | CPU RSS |",
                "|-------|----------|---------------|--------------|--------------------|---------------|---------|",
            ]
        )
        for row in phases:
            lines.append(
                f"| `{row['phase']}` | {_format_seconds(row['duration_s'])} | "
                f"{_format_mb(row['gpu_allocated_start_mb'])} -> {_format_mb(row['gpu_allocated_end_mb'])} | "
                f"{_format_mb(row['gpu_reserved_start_mb'])} -> {_format_mb(row['gpu_reserved_end_mb'])} | "
                f"{_format_mb(row['gpu_peak_allocated_mb'])} | "
                f"{_format_mb(row['gpu_used_peak_mb'])} | "
                f"{_format_mb(row['cpu_rss_start_mb'])} -> {_format_mb(row['cpu_rss_end_mb'])} |"
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
    return markdown_path
