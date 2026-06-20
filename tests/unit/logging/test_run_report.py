"""Unit tests for benchmark-style Python run reports."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from library.adapters.shared.trainables import AdapterTrainableParameterRef
import torch
import torch.nn as nn

from library.logging.phase_tags import PHASE_CACHE_LATENTS, training_epoch_phase
from library.logging.reports import (
    RunReportContext,
    _build_report_payload_from_context,
    _collect_key_config_rows,
    _render_report_markdown,
    is_benchmark_report_enabled,
    write_run_report,
)
from library.logging.resource_monitor.fact_production import build_resource_monitor_produced_facts
from library.metadata import (
    METADATA_PAYLOAD_VERSION,
    MetadataRuntime,
    ResourceObservationFrameFacts,
    ResourceRunView,
    project_resource_run_compatibility_events,
)


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def _canonical_resource_frame(
    event: dict,
    *,
    sequence: int,
    run_identifier: str = "run-1",
) -> ResourceObservationFrameFacts:
    payload = {
        "ts": float(sequence),
        "event": "step_sample",
        "rank": 0,
        "world_size": 1,
        "mode": "sampled",
        "device_scope": "all",
        "config_name": "tests",
        "git_sha": None,
        "git_dirty": None,
        "global_step": None,
        "epoch": None,
        "phase": None,
        "duration_ms": None,
        "gpu_allocated_mb": None,
        "gpu_allocated_by_device_mb": None,
        "gpu_reserved_mb": None,
        "gpu_reserved_by_device_mb": None,
        "gpu_peak_allocated_mb": None,
        "gpu_peak_allocated_by_device_mb": None,
        "gpu_used_mb": None,
        "gpu_used_by_device_mb": None,
        "cpu_rss_mb": None,
        "cpu_vms_mb": None,
        "steps_per_sec": None,
        "samples_per_sec": None,
        "dropped_samples": None,
        "collection_ms": None,
        "deep_alloc_retries": None,
        "deep_ooms": None,
        "deep_active_mb": None,
        "deep_reserved_mb": None,
        "deep_inactive_split_mb": None,
        "deep_window_active": None,
    }
    payload.update(event)
    produced = build_resource_monitor_produced_facts(
        payload,
        run_identifier=run_identifier,
        sequence=sequence,
    )
    assert produced.observation_frame is not None
    return produced.observation_frame


def test_benchmark_report_enabled_requires_explicit_true():
    assert is_benchmark_report_enabled({"output": {"logging": {"benchmark_report": {"enabled": True}}}})
    assert not is_benchmark_report_enabled({"output": {"logging": {"benchmark_report": {"enabled": False}}}})
    assert not is_benchmark_report_enabled({})


def test_collect_key_config_rows_uses_declared_config_paths():
    rows = _collect_key_config_rows(
        {
            "mode": "adapter",
            "model": {"model_type": "sdxl"},
            "output": {"saving": {"output_name": "unit"}, "logging": {"logging_dir": "logs"}},
        },
        hydra_config_name="presets/sdxl",
        hydra_overrides=["training.max_train_steps=10"],
    )

    row_map = dict(rows)
    assert row_map["hydra.config_name"] == "presets/sdxl"
    assert row_map["mode"] == "adapter"
    assert row_map["model.model_type"] == "sdxl"
    assert row_map["output.saving.output_name"] == "unit"
    assert row_map["output.logging.logging_dir"] == "logs"
    assert row_map["hydra.task_overrides"] == ["training.max_train_steps=10"]
    assert "training.max_train_steps" not in row_map


def test_build_report_payload_from_context_uses_explicit_run_facts(tmp_path):
    cfg = {
        "output": {
            "saving": {"output_dir": str(tmp_path), "output_name": "context_run"},
            "logging": {"benchmark_report": {"include_full_config": False}},
        },
        "training": {"max_train_steps": 20},
    }
    context = RunReportContext(
        cfg=cfg,
        resource_jsonl_path=None,
        session_id="run-1",
        training_started_at=100.0,
        mode_name="AdapterMode",
        strategy_name="SdxlTrainingStrategy",
        optimizer_name="AdamW8bit",
        global_step=5,
        num_train_epochs=2,
        component_memory_estimates=[{"name": "unet"}],
        runtime_trace={"phases": [], "events": [], "phase_totals": {"startup.metadata": 1.2}, "milestones": {"time_to_progress_bar_s": 2.0}, "open_phases": []},
    )

    payload = _build_report_payload_from_context(context, succeeded=True, error_message=None)

    assert payload["status"] == "succeeded"
    assert payload["session_id"] == "run-1"
    assert payload["mode_name"] == "AdapterMode"
    assert payload["strategy_name"] == "SdxlTrainingStrategy"
    assert payload["optimizer_name"] == "AdamW8bit"
    assert payload["global_step"] == 5
    assert payload["num_train_epochs"] == 2
    assert payload["component_memory_estimates"] == [{"name": "unet"}]
    assert payload["runtime_trace"]["phase_totals"]["startup.metadata"] == 1.2
    assert payload["resource_monitor"]["event_count"] == 0
    assert payload["resource_monitor"]["input_source"] == "none"
    assert payload["resource_monitor"]["total_resource_event_count"] == 0
    assert payload["resource_monitor"]["total_jsonl_event_count"] == 0
    assert payload["include_full_config"] is False


def test_report_resource_payload_prefers_metadata_view_and_preserves_jsonl_summary(tmp_path):
    events = [
        {
            "ts": 1.0,
            "event": "session_start",
            "gpu_allocated_mb": 100.0,
            "gpu_allocated_by_device_mb": {"0": 60.0, "1": 40.0},
            "gpu_reserved_mb": 120.0,
            "gpu_reserved_by_device_mb": {"0": 70.0, "1": 50.0},
            "gpu_peak_allocated_mb": 100.0,
            "gpu_peak_allocated_by_device_mb": {"0": 60.0, "1": 40.0},
            "gpu_used_mb": 140.0,
            "gpu_used_by_device_mb": {"0": 90.0, "1": 50.0},
            "cpu_rss_mb": 200.0,
            "cpu_vms_mb": 300.0,
        },
        {
            "ts": 2.0,
            "event": "phase_start",
            "phase": training_epoch_phase(0),
            "gpu_allocated_mb": 110.0,
            "gpu_reserved_mb": 130.0,
            "cpu_rss_mb": 210.0,
            "cpu_vms_mb": 310.0,
        },
        {
            "ts": 2.5,
            "event": "step_sample",
            "gpu_used_mb": 180.0,
            "gpu_used_by_device_mb": {"0": 100.0, "1": 80.0},
        },
        {
            "ts": 3.0,
            "event": "phase_end",
            "phase": training_epoch_phase(0),
            "duration_ms": 1000.0,
            "gpu_allocated_mb": 130.0,
            "gpu_reserved_mb": 150.0,
            "gpu_peak_allocated_mb": 160.0,
            "cpu_rss_mb": 230.0,
            "cpu_vms_mb": 330.0,
        },
        {
            "ts": 4.0,
            "event": "session_end",
            "duration_ms": 3000.0,
            "gpu_allocated_mb": 140.0,
            "gpu_reserved_mb": 170.0,
            "gpu_peak_allocated_mb": 170.0,
            "gpu_used_mb": 160.0,
            "cpu_rss_mb": 240.0,
            "cpu_vms_mb": 340.0,
        },
    ]
    runtime = MetadataRuntime()
    runtime.file_many(
        tuple(_canonical_resource_frame(event, sequence=index) for index, event in enumerate(events))
    )
    resource_view = ResourceRunView.from_snapshot(runtime.snapshot(), run_identifier="run-1")
    canonical_events = list(project_resource_run_compatibility_events(resource_view))
    jsonl_path = tmp_path / "resource.jsonl"
    _write_jsonl(jsonl_path, canonical_events)

    jsonl_payload = _build_report_payload_from_context(
        RunReportContext(
            cfg={"output": {"saving": {"output_dir": str(tmp_path)}}, "training": {"max_train_steps": 1}},
            resource_jsonl_path=jsonl_path,
            session_id="run-1",
            training_started_at=100.0,
            mode_name="AdapterMode",
            strategy_name="TestStrategy",
            optimizer_name="AdamW",
            global_step=1,
            num_train_epochs=1,
            component_memory_estimates=[],
            runtime_trace=None,
        ),
        succeeded=True,
        error_message=None,
    )

    _write_jsonl(jsonl_path, [{"event": "session_end", "gpu_used_mb": 9999.0}])
    view_payload = _build_report_payload_from_context(
        RunReportContext(
            cfg={"output": {"saving": {"output_dir": str(tmp_path)}}, "training": {"max_train_steps": 1}},
            resource_jsonl_path=jsonl_path,
            session_id="run-1",
            training_started_at=100.0,
            mode_name="AdapterMode",
            strategy_name="TestStrategy",
            optimizer_name="AdamW",
            global_step=1,
            num_train_epochs=1,
            component_memory_estimates=[],
            runtime_trace=None,
            resource_view=resource_view,
        ),
        succeeded=True,
        error_message=None,
    )

    for key in (
        "event_count",
        "session_start",
        "session_end",
        "gpu_used_peak_session_mb",
        "gpu_used_peak_session_by_device_mb",
        "phases",
        "debug",
    ):
        assert view_payload["resource_monitor"][key] == jsonl_payload["resource_monitor"][key]
    assert view_payload["resource_monitor"]["input_source"] == "metadata_view"
    assert view_payload["resource_monitor"]["total_resource_event_count"] == len(events)
    assert view_payload["resource_monitor"]["total_jsonl_event_count"] is None
    assert jsonl_payload["resource_monitor"]["input_source"] == "jsonl_compatibility"


def test_report_resource_payload_falls_back_when_resource_view_has_no_observation_frames(tmp_path):
    jsonl_path = tmp_path / "resource.jsonl"
    _write_jsonl(
        jsonl_path,
        [{"ts": 1.0, "event": "session_end", "run_identifier": "run-1", "gpu_used_mb": 100.0}],
    )
    empty_view = ResourceRunView.from_snapshot(MetadataRuntime().snapshot(), run_identifier="run-1")

    payload = _build_report_payload_from_context(
        RunReportContext(
            cfg={"output": {"saving": {"output_dir": str(tmp_path)}}},
            resource_jsonl_path=jsonl_path,
            session_id="run-1",
            training_started_at=100.0,
            mode_name="AdapterMode",
            strategy_name="TestStrategy",
            optimizer_name="AdamW",
            global_step=1,
            num_train_epochs=1,
            component_memory_estimates=[],
            runtime_trace=None,
            resource_view=empty_view,
        ),
        succeeded=True,
        error_message=None,
    )

    assert payload["resource_monitor"]["input_source"] == "jsonl_compatibility"
    assert payload["resource_monitor"]["event_count"] == 1
    assert payload["resource_monitor"]["total_resource_event_count"] == 1
    assert payload["resource_monitor"]["total_jsonl_event_count"] == 1


def test_render_report_markdown_skips_empty_runtime_trace_milestone_table():
    payload = {
        "generated_at": 0.0,
        "status": "failed",
        "hydra_config_name": "test_cfg",
        "output_name": "test_run",
        "total_duration_s": 1.0,
        "mode_name": "AdapterMode",
        "strategy_name": "SdxlTrainingStrategy",
        "optimizer_name": None,
        "global_step": 0,
        "num_train_epochs": 0,
        "output_dir": "/tmp/out",
        "system": {"gpu": "CUDA unavailable", "python": "3.11", "pytorch": "2.x"},
        "key_config": [],
        "resource_monitor": {"phases": []},
        "component_memory_estimates": [],
        "include_full_config": False,
        "runtime_trace": {
            "phases": [],
            "events": [],
            "phase_totals": {},
            "milestones": {},
            "open_phases": [],
        },
    }

    markdown = _render_report_markdown(payload)
    assert "## Runtime Trace" not in markdown


def test_write_run_report_emits_markdown_and_json_with_phase_peaks(tmp_path):
    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "resource_monitor.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {
                "ts": 1.0,
                "event": "session_start",
                "gpu_allocated_mb": 100.0,
                "gpu_allocated_by_device_mb": {"0": 60.0, "1": 40.0},
                "gpu_reserved_mb": 120.0,
                "gpu_reserved_by_device_mb": {"0": 70.0, "1": 50.0},
                "gpu_peak_allocated_mb": 100.0,
                "gpu_peak_allocated_by_device_mb": {"0": 60.0, "1": 40.0},
                "cpu_rss_mb": 200.0,
                "cpu_vms_mb": 300.0,
                "gpu_used_mb": 140.0,
                "gpu_used_by_device_mb": {"0": 90.0, "1": 50.0},
            },
            {
                "ts": 2.0,
                "event": "phase_start",
                "phase": PHASE_CACHE_LATENTS,
                "gpu_allocated_mb": 100.0,
                "gpu_allocated_by_device_mb": {"0": 60.0, "1": 40.0},
                "gpu_reserved_mb": 120.0,
                "gpu_reserved_by_device_mb": {"0": 70.0, "1": 50.0},
                "gpu_peak_allocated_mb": 100.0,
                "gpu_peak_allocated_by_device_mb": {"0": 60.0, "1": 40.0},
                "cpu_rss_mb": 200.0,
                "cpu_vms_mb": 300.0,
            },
            {"ts": 2.2, "event": "step_sample", "phase": None, "gpu_used_mb": 180.0, "gpu_used_by_device_mb": {"0": 100.0, "1": 80.0}},
            {
                "ts": 3.0,
                "event": "phase_end",
                "phase": PHASE_CACHE_LATENTS,
                "duration_ms": 1000.0,
                "gpu_allocated_mb": 110.0,
                "gpu_allocated_by_device_mb": {"0": 65.0, "1": 45.0},
                "gpu_reserved_mb": 130.0,
                "gpu_reserved_by_device_mb": {"0": 75.0, "1": 55.0},
                "gpu_peak_allocated_mb": 150.0,
                "gpu_peak_allocated_by_device_mb": {"0": 80.0, "1": 70.0},
                "cpu_rss_mb": 210.0,
                "cpu_vms_mb": 310.0,
                "gpu_used_mb": 170.0,
                "gpu_used_by_device_mb": {"0": 95.0, "1": 75.0},
            },
            {
                "ts": 4.0,
                "event": "phase_start",
                "phase": training_epoch_phase(0),
                "gpu_allocated_mb": 150.0,
                "gpu_allocated_by_device_mb": {"0": 90.0, "1": 60.0},
                "gpu_reserved_mb": 170.0,
                "gpu_reserved_by_device_mb": {"0": 100.0, "1": 70.0},
                "gpu_peak_allocated_mb": 150.0,
                "gpu_peak_allocated_by_device_mb": {"0": 90.0, "1": 60.0},
                "cpu_rss_mb": 220.0,
                "cpu_vms_mb": 320.0,
            },
            {"ts": 4.5, "event": "step_sample", "phase": None, "gpu_used_mb": 240.0, "gpu_used_by_device_mb": {"0": 150.0, "1": 90.0}},
            {"ts": 5.0, "event": "step_sample", "phase": None, "gpu_used_mb": 260.0, "gpu_used_by_device_mb": {"0": 170.0, "1": 90.0}},
            {
                "ts": 6.0,
                "event": "phase_end",
                "phase": training_epoch_phase(0),
                "duration_ms": 2000.0,
                "gpu_allocated_mb": 180.0,
                "gpu_allocated_by_device_mb": {"0": 110.0, "1": 70.0},
                "gpu_reserved_mb": 210.0,
                "gpu_reserved_by_device_mb": {"0": 125.0, "1": 85.0},
                "gpu_peak_allocated_mb": 220.0,
                "gpu_peak_allocated_by_device_mb": {"0": 130.0, "1": 90.0},
                "cpu_rss_mb": 260.0,
                "cpu_vms_mb": 360.0,
                "gpu_used_mb": 250.0,
                "gpu_used_by_device_mb": {"0": 160.0, "1": 90.0},
            },
            {
                "ts": 7.0,
                "event": "session_end",
                "duration_ms": 5000.0,
                "gpu_allocated_mb": 190.0,
                "gpu_allocated_by_device_mb": {"0": 120.0, "1": 70.0},
                "gpu_reserved_mb": 220.0,
                "gpu_reserved_by_device_mb": {"0": 130.0, "1": 90.0},
                "gpu_peak_allocated_mb": 225.0,
                "gpu_peak_allocated_by_device_mb": {"0": 135.0, "1": 90.0},
                "cpu_rss_mb": 270.0,
                "cpu_vms_mb": 370.0,
                "gpu_used_mb": 245.0,
                "gpu_used_by_device_mb": {"0": 155.0, "1": 90.0},
            },
        ],
    )

    trainer = SimpleNamespace(
        cfg={
            "mode": "adapter",
            "model": {"model_type": "sdxl"},
            "output": {
                "saving": {"output_dir": str(tmp_path / "models"), "output_name": "unit_test_run"},
                "logging": {
                    "benchmark_report": {
                        "enabled": True,
                        "output_dir": str(output_dir),
                        "filename_prefix": "unit_report",
                        "include_full_config": True,
                    },
                    "resource_monitor": {"enabled": True, "mode": "sampled", "output_jsonl": str(jsonl_path)},
                },
            },
            "training": {"train_batch_size": 2, "gradient_accumulation_steps": 1, "max_train_steps": 10},
            "data": {"loader": {"pin_memory": True}},
            "optimizer": {"optimizer_type": "adamw8bit", "learning_rates": {"base": 1e-4, "denoiser": 1e-4}},
            "adapter": {"peft": {"loha": {"rank": 16}}},
        },
        mode=MagicMock(),
        strategies=MagicMock(),
        optimizer_name="AdamW8bit",
        global_step=10,
        num_train_epochs=1,
        session_id=12345,
        training_started_at=100.0,
        _resource_monitor=SimpleNamespace(jsonl_path=jsonl_path),
        runtime_trace=SimpleNamespace(
            summary=lambda: {
                "phases": [
                    {"tag": "startup.accelerator", "start_offset_s": 0.0, "end_offset_s": 0.1, "duration_s": 0.1},
                    {"tag": "training.epoch.0", "start_offset_s": 0.0, "end_offset_s": 2.0, "duration_s": 2.0},
                ],
                "events": [{"tag": "training.progress_bar.started", "offset_s": 0.5}],
                "phase_totals": {"training.epoch.0": 2.0, "startup.metadata": 1.0},
                "milestones": {
                    "time_to_progress_bar_s": 0.5,
                    "time_to_first_step_started_s": 0.8,
                    "time_to_first_synced_step_s": 1.2,
                    "time_to_run_ended_s": 5.0,
                },
                "open_phases": [],
            }
        ),
    )
    trainer.mode.get_diagnostics_components.return_value = [("adapter", nn.Linear(4, 4))]

    markdown_path = write_run_report(trainer, succeeded=True)

    assert markdown_path is not None
    assert markdown_path.exists()
    assert markdown_path.name.startswith("unit_report_")
    assert markdown_path.with_suffix(".json").exists()

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Summary" in markdown
    assert "## Speed Results" in markdown
    assert "## Runtime Trace" in markdown
    assert "## Trace-only Phases" in markdown
    assert "`startup.accelerator`" in markdown
    assert "## Full Composed Config" in markdown
    assert "GPU Used Peak" in markdown
    assert "CPU VMS" in markdown
    assert "## Resource Interpretation" in markdown
    assert "They do not claim component-level ownership or causality." in markdown
    assert "## Per-Device GPU Session Peaks" in markdown
    assert "## Per-Device GPU Phase Details" in markdown
    assert "unit_test_run" in markdown
    assert "`data.loader.pin_memory`" in markdown

    payload = json.loads(markdown_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["runtime_trace"]["milestones"]["time_to_progress_bar_s"] == 0.5
    assert payload["runtime_trace"]["milestones"]["time_to_run_ended_s"] == 5.0
    assert "time_to_training_finished_s" not in payload["runtime_trace"]["milestones"]
    assert payload["runtime_trace"]["trace_only_phases"][0]["tag"] == "startup.accelerator"
    assert payload["resource_monitor"]["gpu_used_peak_session_mb"] == 260.0
    assert payload["resource_monitor"]["gpu_used_peak_session_by_device_mb"] == {"0": 170.0, "1": 90.0}
    assert payload["resource_monitor"]["phases"][0]["gpu_used_peak_mb"] == 180.0
    assert payload["resource_monitor"]["phases"][0]["gpu_used_peak_by_device_mb"] == {"0": 100.0, "1": 80.0}
    assert payload["resource_monitor"]["phases"][1]["gpu_used_peak_mb"] == 260.0
    assert payload["resource_monitor"]["phases"][1]["gpu_peak_allocated_by_device_mb"] == {"0": 130.0, "1": 90.0}
    assert payload["resource_monitor"]["session_end"]["cpu_vms_mb"] == 370.0
    assert payload["resource_monitor"]["session_end"]["gpu_allocated_by_device_mb"] == {"0": 120.0, "1": 70.0}
    assert payload["resource_monitor"]["phases"][0]["cpu_vms_start_mb"] == 300.0
    assert payload["resource_monitor"]["phases"][0]["cpu_vms_end_mb"] == 310.0
    assert payload["resource_monitor"]["debug"]["surface_split"]["live_metadata_fields"][0] == "gpu_allocated_mb"
    assert payload["resource_monitor"]["debug"]["surface_split"]["report_debug_only_fields"] == [
        "session_gpu_used_peak_by_device_rows",
        "phase_device_rows",
        "session_change_summary",
        "phase_change_rows",
    ]
    assert payload["resource_monitor"]["debug"]["session_change_summary"]["gpu_allocated_delta_mb"] == 90.0
    assert payload["resource_monitor"]["debug"]["session_change_summary"]["cpu_vms_delta_mb"] == 70.0
    assert (
        payload["resource_monitor"]["debug"]["session_change_summary"]["observations"][0]
        == "GPU allocated ended 90 MB higher than it started."
    )
    assert payload["resource_monitor"]["debug"]["session_gpu_used_peak_by_device_rows"] == [
        {"device": "0", "gpu_used_peak_mb": 170.0},
        {"device": "1", "gpu_used_peak_mb": 90.0},
    ]
    assert payload["resource_monitor"]["debug"]["phase_device_rows"][0]["phase"] == PHASE_CACHE_LATENTS
    assert payload["resource_monitor"]["debug"]["phase_change_rows"][0]["phase"] == PHASE_CACHE_LATENTS
    assert payload["resource_monitor"]["debug"]["phase_change_rows"][0]["gpu_allocated_delta_mb"] == 10.0
    assert payload["resource_monitor"]["debug"]["phase_change_rows"][1]["gpu_reserved_delta_mb"] == 40.0
    assert (
        "GPU reserved grew more than GPU allocated, so allocator capacity expanded beyond the end-of-window live allocation level."
        in payload["resource_monitor"]["debug"]["phase_change_rows"][1]["observations"]
    )


def test_write_run_report_files_metadata_record_when_observer_runtime_exists(tmp_path):
    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_runtime = MetadataRuntime()
    metadata_runtime.file_many(
        (
            _canonical_resource_frame(
                {"ts": 1.0, "event": "session_start", "gpu_allocated_mb": 10.0},
                sequence=0,
                run_identifier="abc123",
            ),
            _canonical_resource_frame(
                {
                    "ts": 2.0,
                    "event": "phase_start",
                    "phase": training_epoch_phase(0),
                    "gpu_allocated_mb": 20.0,
                },
                sequence=1,
                run_identifier="abc123",
            ),
            _canonical_resource_frame(
                {
                    "ts": 3.0,
                    "event": "phase_end",
                    "phase": training_epoch_phase(0),
                    "duration_ms": 1000.0,
                    "gpu_allocated_mb": 30.0,
                },
                sequence=2,
                run_identifier="abc123",
            ),
            _canonical_resource_frame(
                {
                    "ts": 4.0,
                    "event": "session_end",
                    "duration_ms": 3000.0,
                    "gpu_allocated_mb": 40.0,
                },
                sequence=3,
                run_identifier="abc123",
            ),
        )
    )
    trainer = SimpleNamespace(
        cfg={
            "output": {
                "saving": {"output_dir": str(tmp_path / "models"), "output_name": "report_runtime"},
                "logging": {
                    "benchmark_report": {
                        "enabled": True,
                        "output_dir": str(output_dir),
                        "filename_prefix": "metadata_report",
                        "include_full_config": False,
                    }
                },
            }
        },
        mode=MagicMock(),
        strategies=MagicMock(),
        optimizer_name="AdamW8bit",
        global_step=7,
        num_train_epochs=1,
        session_id="abc123",
        training_started_at=100.0,
        _resource_monitor=SimpleNamespace(jsonl_path=None),
        _observer=SimpleNamespace(
            metadata_runtime=metadata_runtime,
            metadata_snapshot=metadata_runtime.snapshot,
            run_identifier="abc123",
        ),
        runtime_trace=SimpleNamespace(
            summary=lambda: {
                "phases": [{"tag": "training.epoch.0", "start_offset_s": 0.0, "end_offset_s": 1.0, "duration_s": 1.0}],
                "events": [{"tag": "training.progress_bar.started", "offset_s": 0.5}],
                "phase_totals": {"training.epoch.0": 1.0},
                "milestones": {"time_to_progress_bar_s": 0.5},
                "open_phases": [],
            }
        ),
    )
    trainer.mode.get_diagnostics_components.return_value = []

    markdown_path = write_run_report(trainer, succeeded=True)

    assert markdown_path is not None
    snapshot = metadata_runtime.snapshot()

    report_record = next(record for record in snapshot.records if record.facts.get("kind") == "benchmark_report")
    assert report_record.identity.identifier == str(markdown_path)
    assert report_record.schema_version == METADATA_PAYLOAD_VERSION
    assert report_record.identity.schema_version == METADATA_PAYLOAD_VERSION
    assert report_record.facts["run_identifier"] == "abc123"
    assert report_record.facts["resource_event_count"] == 4
    assert report_record.facts["phase_count"] == 1
    assert report_record.facts["include_full_config"] is False

    analytics_record = next(record for record in snapshot.records if record.facts.get("kind") == "benchmark_report_payload")
    assert analytics_record.identity.identifier == f"{markdown_path.with_suffix('.json')}#payload"
    assert analytics_record.schema_version == METADATA_PAYLOAD_VERSION
    assert analytics_record.identity.schema_version == METADATA_PAYLOAD_VERSION
    assert analytics_record.facts["run_identifier"] == "abc123"
    assert analytics_record.facts["source"] == "library.logging.reports.write_run_report"
    analytics_payload = analytics_record.facts["payload"]
    assert isinstance(analytics_payload, dict)
    analytics_payload = cast(dict[str, Any], analytics_payload)
    runtime_trace_payload = cast(dict[str, Any], analytics_payload["runtime_trace"])
    resource_monitor_payload = cast(dict[str, Any], analytics_payload["resource_monitor"])
    assert analytics_payload["status"] == "succeeded"
    assert runtime_trace_payload["phase_totals"]["training.epoch.0"] == 1.0
    assert resource_monitor_payload["input_source"] == "metadata_view"
    assert resource_monitor_payload["event_count"] == 4
    assert analytics_payload["include_full_config"] is False


def test_write_run_report_handles_missing_jsonl_gracefully(tmp_path):
    output_dir = tmp_path / "reports"
    trainer = SimpleNamespace(
        cfg={"output": {"saving": {"output_dir": str(output_dir), "output_name": "missing_jsonl"}}},
        mode=MagicMock(),
        strategies=MagicMock(),
        optimizer_name="AdamW8bit",
        global_step=0,
        num_train_epochs=0,
        session_id=999,
        training_started_at=100.0,
        _resource_monitor=SimpleNamespace(jsonl_path=output_dir / "missing.jsonl"),
        runtime_trace=SimpleNamespace(
            summary=lambda: {"phases": [], "events": [], "phase_totals": {}, "milestones": {}, "open_phases": []}
        ),
    )
    trainer.mode.get_diagnostics_components.return_value = []

    markdown_path = write_run_report(trainer, succeeded=False, error_message="boom")

    assert markdown_path is not None
    assert markdown_path.exists()
    payload = json.loads(markdown_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["resource_monitor"]["event_count"] == 0
    assert payload["resource_monitor"]["input_source"] == "none"
    assert payload["resource_monitor"]["total_resource_event_count"] == 0
    assert payload["resource_monitor"]["total_jsonl_event_count"] == 0
    assert payload["resource_monitor"]["phases"] == []


def test_write_run_report_filters_appended_jsonl_to_current_session(tmp_path):
    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "resource_monitor.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"ts": 1.0, "event": "session_start", "run_identifier": "old-run", "gpu_used_mb": 100.0},
            {"ts": 2.0, "event": "phase_start", "run_identifier": "old-run", "phase": training_epoch_phase(0), "gpu_allocated_mb": 10.0},
            {
                "ts": 3.0,
                "event": "phase_end",
                "run_identifier": "old-run",
                "phase": training_epoch_phase(0),
                "duration_ms": 1000.0,
                "gpu_allocated_mb": 20.0,
            },
            {"ts": 4.0, "event": "session_end", "run_identifier": "old-run", "duration_ms": 3000.0, "gpu_used_mb": 150.0},
            {"ts": 5.0, "event": "session_start", "run_identifier": "12345", "gpu_used_mb": 200.0},
            {"ts": 6.0, "event": "phase_start", "run_identifier": "12345", "phase": training_epoch_phase(0), "gpu_allocated_mb": 30.0},
            {"ts": 7.0, "event": "step_sample", "run_identifier": "12345", "phase": None, "gpu_used_mb": 300.0},
            {
                "ts": 8.0,
                "event": "phase_end",
                "run_identifier": "12345",
                "phase": training_epoch_phase(0),
                "duration_ms": 2000.0,
                "gpu_allocated_mb": 40.0,
            },
            {"ts": 9.0, "event": "session_end", "run_identifier": "12345", "duration_ms": 5000.0, "gpu_used_mb": 250.0},
        ],
    )

    def failed_metadata_snapshot():
        raise RuntimeError("metadata snapshot unavailable")

    trainer = SimpleNamespace(
        cfg={"output": {"saving": {"output_dir": str(output_dir), "output_name": "current_run"}}},
        mode=MagicMock(),
        strategies=MagicMock(),
        optimizer_name="AdamW8bit",
        global_step=1,
        num_train_epochs=1,
        session_id=12345,
        training_started_at=100.0,
        _resource_monitor=SimpleNamespace(jsonl_path=jsonl_path),
        _observer=SimpleNamespace(metadata_snapshot=failed_metadata_snapshot, run_identifier="12345"),
        runtime_trace=SimpleNamespace(
            summary=lambda: {"phases": [], "events": [], "phase_totals": {}, "milestones": {}, "open_phases": []}
        ),
    )
    trainer.mode.get_diagnostics_components.return_value = []

    markdown_path = write_run_report(trainer, succeeded=True)

    assert markdown_path is not None
    payload = json.loads(markdown_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["resource_monitor"]["input_source"] == "jsonl_compatibility"
    assert payload["resource_monitor"]["event_count"] == 5
    assert payload["resource_monitor"]["total_resource_event_count"] == 5
    assert payload["resource_monitor"]["total_jsonl_event_count"] == 9
    assert payload["resource_monitor"]["session_start"]["run_identifier"] == "12345"
    assert len(payload["resource_monitor"]["phases"]) == 1
    assert payload["resource_monitor"]["phases"][0]["duration_s"] == 2.0
    assert payload["resource_monitor"]["gpu_used_peak_session_mb"] == 300.0


def test_write_run_report_uses_adapter_provenance_for_component_memory_rows(tmp_path):
    output_dir = tmp_path / "reports"
    trainer = SimpleNamespace(
        cfg={"output": {"saving": {"output_dir": str(output_dir), "output_name": "adapter_provenance"}}},
        mode=MagicMock(),
        strategies=MagicMock(),
        optimizer_name="AdamW8bit",
        global_step=0,
        num_train_epochs=0,
        session_id=1,
        training_started_at=100.0,
        _resource_monitor=SimpleNamespace(jsonl_path=output_dir / "missing.jsonl"),
        runtime_trace=SimpleNamespace(
            summary=lambda: {"phases": [], "events": [], "phase_totals": {}, "milestones": {}, "open_phases": []}
        ),
    )

    adapter = SimpleNamespace()
    adapter.describe_trainable_parameter_refs = lambda: [
        AdapterTrainableParameterRef(
            param=nn.Parameter(torch.ones(4)),
            name="lora_unet.block.up",
            algorithm="lora",
            component="unet",
            component_key="denoiser",
            target_path="unet.block",
            adapter_module_path="lora_unet.block.lora_up",
        )
    ]
    trainer.adapter = adapter
    trainer.text_encoders = []
    trainer.mode.get_diagnostics_components.return_value = [("adapter", nn.Linear(4, 4))]

    markdown_path = write_run_report(trainer, succeeded=True)

    assert markdown_path is not None
    payload = json.loads(markdown_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["component_memory_estimates"][0]["name"] == "unet"
    assert payload["component_memory_estimates"][0]["adapter_modules_total"] == 1
