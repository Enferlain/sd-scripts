"""Unit tests for library.logging.resource_monitor."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from library.logging.phase_tags import PHASE_CACHE_LATENTS, training_epoch_phase
from library.logging.summaries import DiagnosticRow
from library.logging.resource_monitor import (
    BasicResourceMonitor,
    NoOpResourceMonitor,
    SampledResourceMonitor,
    create_resource_monitor,
)
from library.logging.resource_monitor.collect import (
    _DeepCounters,
    _SampledMetrics,
    _Snapshot,
    build_resource_collection_policy,
)
from library.logging.resource_monitor.console_summary import (
    build_phase_resource_console_summary,
    build_session_resource_console_summary,
    build_step_resource_console_summary,
    render_phase_resource_console_summary,
    render_session_resource_console_summary,
    render_step_resource_console_summary,
)
from library.logging.resource_monitor.fact_production import (
    build_resource_monitor_produced_facts,
    project_resource_monitor_jsonl_event,
)
from library.metadata import (
    MetadataRuntime,
    project_resource_run_compatibility_events,
    ResourceRunView,
)


def _make_cfg(**overrides):
    base = {
        "enabled": True,
        "mode": "basic",
        "rank_scope": "main",
        "phase_summary": "verbose",
        "component_breakdown": True,
        "log_every_n_steps": 1,
        "device_scope": "local",
        "sample_interval_sec": 0.01,
        "output_jsonl": None,
        "jsonl_flush_mode": "auto",
        "jsonl_flush_every_n_events": 50,
        "queue_maxsize": 1024,
        "drop_policy": "drop_oldest",
        "max_collection_ms": 0.0,
        "deep_window_steps": 0,
        "deep_window_seconds": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.unit
class TestResourceMonitorFactory:
    def test_create_resource_monitor_disabled_returns_noop(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        cfg = _make_cfg(enabled=False)

        monitor = create_resource_monitor(accelerator=accelerator, resource_monitor_config=cfg)

        assert isinstance(monitor, NoOpResourceMonitor)

    def test_create_resource_monitor_non_bool_enabled_defaults_to_noop(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        cfg = MagicMock()
        cfg.enabled = MagicMock()  # Not a bool
        cfg.mode = "basic"

        monitor = create_resource_monitor(accelerator=accelerator, resource_monitor_config=cfg)

        assert isinstance(monitor, NoOpResourceMonitor)

    def test_create_resource_monitor_basic_returns_basic_monitor(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True

        monitor = create_resource_monitor(accelerator=accelerator, resource_monitor_config=_make_cfg(mode="basic"))

        assert isinstance(monitor, BasicResourceMonitor)

    def test_create_resource_monitor_sampled_returns_sampled_monitor(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True

        monitor = create_resource_monitor(accelerator=accelerator, resource_monitor_config=_make_cfg(mode="sampled"))

        assert isinstance(monitor, SampledResourceMonitor)

    def test_collection_policy_maps_current_modes_to_capabilities(self):
        off_policy = build_resource_collection_policy("off")
        basic_policy = build_resource_collection_policy("basic")
        sampled_policy = build_resource_collection_policy("sampled")
        deep_policy = build_resource_collection_policy("deep")

        assert off_policy.capabilities == ()
        assert basic_policy.includes("resource_monitor.process_memory")
        assert basic_policy.includes("resource_monitor.cuda_allocator")
        assert not basic_policy.includes("resource_monitor.nvml_gpu_used")
        assert sampled_policy.includes("resource_monitor.sampler")
        assert sampled_policy.includes("resource_monitor.nvml_gpu_used")
        assert sampled_policy.includes("resource_monitor.torch_gpu_used")
        assert not sampled_policy.includes("resource_monitor.deep_allocator")
        assert deep_policy.includes("resource_monitor.deep_allocator")


@pytest.mark.unit
class TestBasicResourceMonitorBehavior:
    def test_console_summary_renderers_preserve_existing_message_shapes(self):
        start_snapshot = _Snapshot(
            gpu_allocated_mb=10.0,
            gpu_allocated_by_device_mb=None,
            gpu_reserved_mb=20.0,
            gpu_reserved_by_device_mb=None,
            gpu_peak_allocated_mb=30.0,
            gpu_peak_allocated_by_device_mb=None,
            cpu_rss_mb=40.0,
            cpu_vms_mb=50.0,
        )
        end_snapshot = _Snapshot(
            gpu_allocated_mb=11.0,
            gpu_allocated_by_device_mb=None,
            gpu_reserved_mb=21.0,
            gpu_reserved_by_device_mb=None,
            gpu_peak_allocated_mb=31.0,
            gpu_peak_allocated_by_device_mb=None,
            cpu_rss_mb=41.0,
            cpu_vms_mb=51.0,
        )

        session_message = render_session_resource_console_summary(
            build_session_resource_console_summary(duration_s=1.25, snapshot=end_snapshot, gpu_used_mb=99.0)
        )
        phase_message = render_phase_resource_console_summary(
            build_phase_resource_console_summary(
                phase_name=training_epoch_phase(0),
                duration_s=2.5,
                start_snapshot=start_snapshot,
                end_snapshot=end_snapshot,
                sampled_peak_gpu_used_mb=88.0,
            )
        )
        step_message = render_step_resource_console_summary(
            build_step_resource_console_summary(
                global_step=3,
                epoch=1,
                snapshot=end_snapshot,
                gpu_used_mb=None,
                steps_per_sec=4.5,
            )
        )
        first_step_message = render_step_resource_console_summary(
            build_step_resource_console_summary(
                global_step=1,
                epoch=1,
                snapshot=end_snapshot,
                gpu_used_mb=77.0,
                steps_per_sec=None,
            )
        )

        assert session_message.message.startswith("Resource session summary:")
        assert session_message.args == (1.25, "11MB", "21MB", "31MB", "99MB", "41MB")
        assert phase_message.message.startswith("Resource phase[%s]:")
        assert phase_message.args == (training_epoch_phase(0), 2.5, "10MB", "11MB", "20MB", "21MB", "31MB", ", gpu_used_peak=88MB", "40MB", "41MB")
        assert step_message.message.startswith("Resource step[%s|epoch=%s]: %.2f steps/s")
        assert step_message.args == (3, 1, 4.5, "11MB", "21MB", "n/a", "41MB")
        assert first_step_message.message.startswith("Resource step[%s|epoch=%s]: gpu_allocated=")
        assert first_step_message.args == (1, 1, "11MB", "21MB", "77MB", "41MB")

    def test_phase_and_step_logging_paths_do_not_raise(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        with patch("library.logging.resource_monitor.monitor.logger") as mock_logger:
            monitor.start_session()
            monitor.phase_start(training_epoch_phase(0))
            monitor.step_end(global_step=1, epoch=1)
            monitor.phase_end(training_epoch_phase(0))
            monitor.end_session()

            assert mock_logger.info.call_count >= 3

    def test_session_phase_and_step_console_uses_resource_domain_summaries(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        with (
            patch(
                "library.logging.resource_monitor.monitor.build_session_resource_console_summary",
                wraps=build_session_resource_console_summary,
            ) as mock_session_summary,
            patch(
                "library.logging.resource_monitor.monitor.build_phase_resource_console_summary",
                wraps=build_phase_resource_console_summary,
            ) as mock_phase_summary,
            patch(
                "library.logging.resource_monitor.monitor.build_step_resource_console_summary",
                wraps=build_step_resource_console_summary,
            ) as mock_step_summary,
        ):
            monitor.start_session()
            monitor.phase_start(training_epoch_phase(0))
            monitor.step_end(global_step=1, epoch=1)
            monitor.phase_end(training_epoch_phase(0))
            monitor.end_session()

        mock_step_summary.assert_called_once()
        mock_phase_summary.assert_called_once()
        mock_session_summary.assert_called_once()

    def test_phase_and_step_logging_uses_external_write_mode(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        with (
            patch("library.logging.resource_monitor.monitor.logger") as mock_logger,
            patch("library.logging.resource_monitor.monitor.tqdm.external_write_mode", return_value=nullcontext()) as mock_external,
        ):
            monitor.start_session()
            monitor.phase_start(training_epoch_phase(0))
            monitor.step_end(global_step=1, epoch=1)
            monitor.phase_end(training_epoch_phase(0))
            monitor.end_session()

            assert mock_external.call_count >= 4
            assert mock_logger.info.call_count >= 3

    def test_phase_events_still_emit_when_phase_summary_is_off(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        metadata_runtime = MetadataRuntime()
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic", phase_summary="off"),
            output_jsonl_path=None,
            run_identifier="run-1",
            metadata_runtime=metadata_runtime,
        )

        with patch.object(monitor, "_log_info_external") as mock_log_info_external:
            monitor.start_session()
            monitor.phase_start(training_epoch_phase(0))
            monitor.phase_end(training_epoch_phase(0))
            monitor.end_session()

        event_names = [
            frame.facts["event_name"]
            for frame in metadata_runtime.snapshot().records_for(
                entity_type="resource_observation_frame"
            )
        ]
        assert "phase_start" in event_names
        assert "phase_end" in event_names
        assert not any(
            call.args and isinstance(call.args[0], str) and call.args[0].startswith("Resource phase[")
            for call in mock_log_info_external.call_args_list
        )
        assert training_epoch_phase(0) not in monitor._phase_states

    def test_default_phase_summary_logs_only_curated_console_phases(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        metadata_runtime = MetadataRuntime()
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic", phase_summary="default"),
            output_jsonl_path=None,
            run_identifier="run-1",
            metadata_runtime=metadata_runtime,
        )

        with patch.object(monitor, "_log_info_external") as mock_log_info_external:
            monitor.start_session()
            monitor.phase_start("training.prep.lr_scheduler")
            monitor.phase_end("training.prep.lr_scheduler")
            monitor.phase_start("startup.metadata")
            monitor.phase_end("startup.metadata")
            monitor.end_session()

        phase_summary_calls = [
            call for call in mock_log_info_external.call_args_list if call.args and isinstance(call.args[0], str) and call.args[0].startswith("Resource phase[")
        ]
        assert len(phase_summary_calls) == 1
        assert phase_summary_calls[0].args[1] == "startup.metadata"

        snapshot = metadata_runtime.snapshot()
        phase_events = [
            frame.facts["phase"]
            for frame in snapshot.records_for(entity_type="resource_observation_frame")
            if frame.facts["event_name"] in {"phase_start", "phase_end"}
        ]
        assert "training.prep.lr_scheduler" in phase_events
        assert "startup.metadata" in phase_events

    def test_default_phase_summary_uses_canonical_epoch_phase_matching(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic", phase_summary="default"),
            output_jsonl_path=None,
        )

        assert monitor._should_log_phase_summary(training_epoch_phase(0)) is True
        assert monitor._should_log_phase_summary("training.epoch.setup") is False

    def test_resource_monitor_files_metadata_without_jsonl_output(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1
        metadata_runtime = MetadataRuntime()
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
            run_identifier="run-1",
            config_name="unit-test",
            metadata_runtime=metadata_runtime,
        )
        monitor.start_session()
        monitor.phase_start(training_epoch_phase(0))
        monitor.step_end(global_step=1, epoch=1)
        monitor.phase_end(training_epoch_phase(0))
        monitor.end_session()

        snapshot = metadata_runtime.snapshot()

        frames = snapshot.records_for(entity_type="resource_observation_frame")
        assert snapshot.events == ()
        assert [frame.facts["event_name"] for frame in frames] == [
            "session_start",
            "phase_start",
            "step_sample",
            "phase_end",
            "session_end",
        ]
        assert frames[2].facts["global_step"] == 1
        assert frames[3].facts["phase"] == training_epoch_phase(0)
        assert len(frames) == 5
        assert len(snapshot.records_for(entity_type="resource_observation")) >= 10

    def test_resource_monitor_routes_metadata_through_resource_fact_production_seam(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        metadata_runtime = MetadataRuntime()
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
            run_identifier="run-1",
            metadata_runtime=metadata_runtime,
        )

        with patch.object(monitor, "_produce_resource_facts", wraps=monitor._produce_resource_facts) as producer:
            monitor.start_session()
            monitor.end_session()

        assert [call.args[0]["event"] for call in producer.call_args_list] == [
            "session_start",
            "session_end",
        ]
        snapshot = metadata_runtime.snapshot()
        assert snapshot.events == ()
        assert [frame.facts["event_name"] for frame in snapshot.records_for(entity_type="resource_observation_frame")] == [
            "session_start",
            "session_end",
        ]

    def test_resource_monitor_produces_canonical_observation_frames_once(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1
        metadata_runtime = MetadataRuntime()
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
            run_identifier="run-1",
            metadata_runtime=metadata_runtime,
        )
        collected_snapshot = _Snapshot(
            gpu_allocated_mb=128.0,
            gpu_allocated_by_device_mb={"0": 128.0},
            gpu_reserved_mb=160.0,
            gpu_reserved_by_device_mb={"0": 160.0},
            gpu_peak_allocated_mb=192.0,
            gpu_peak_allocated_by_device_mb={"0": 192.0},
            cpu_rss_mb=256.0,
            cpu_vms_mb=512.0,
        )

        with patch.object(monitor, "_collect_snapshot", return_value=collected_snapshot):
            monitor.start_session()
            monitor.phase_start(training_epoch_phase(0))
            monitor.step_end(global_step=1, epoch=1)
            monitor.phase_end(training_epoch_phase(0))
            monitor.end_session()

        snapshot = metadata_runtime.snapshot()
        frames = snapshot.records_for(entity_type="resource_observation_frame")
        measurements = snapshot.records_for(entity_type="resource_observation")
        step_frame = next(frame for frame in frames if frame.facts["event_name"] == "step_sample")
        step_frame_metadata = step_frame.facts["metadata"]

        assert step_frame.facts["global_step"] == 1
        assert step_frame.facts["rank"] == 0
        assert isinstance(step_frame_metadata, dict)
        assert "device_scope" in step_frame_metadata
        assert any(record.facts["measurement_kind"] == "rss" for record in measurements)
        assert any(record.facts["measurement_kind"] == "allocated" for record in measurements)
        assert all(record.facts["semantic_class"] == "observation" for record in frames)

    def test_emit_startup_component_memory_logs_estimate(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        components = {"denoiser": torch.nn.Linear(4, 4)}
        with patch("builtins.print") as mock_print:
            monitor.emit_startup_component_memory(components, "AdamW")
            mock_print.assert_called_once()
            logged = mock_print.call_args.args[0]
            assert "loaded model weights:" in logged
            assert "training state:" in logged
            assert "component |" in logged
            assert "loaded |" in logged
            assert "frozen |" in logged
            assert "share" in logged
            assert "total" in logged

    def test_emit_startup_component_memory_accepts_component_pair_list(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        components = [("adapter", torch.nn.Linear(4, 4))]
        with patch("builtins.print") as mock_print:
            monitor.emit_startup_component_memory(components, "AdamW")
            mock_print.assert_called_once()
            logged = mock_print.call_args.args[0]
            assert "loaded model weights:" in logged

    def test_emit_startup_component_memory_ignores_malformed_component_entries(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        components = [("adapter", torch.nn.Linear(4, 4)), ("bad_only_name",), 123]
        with patch("builtins.print") as mock_print:
            monitor.emit_startup_component_memory(components, "AdamW")
            mock_print.assert_called_once()

    def test_emit_startup_component_memory_includes_deepspeed_partitioning_caveat(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        components = {"denoiser": torch.nn.Linear(4, 4)}
        with patch("builtins.print") as mock_print:
            monitor.emit_startup_component_memory(
                components,
                "AdamW",
                deepspeed_enabled=True,
                deepspeed_zero_stage=2,
            )

            logged = mock_print.call_args.args[0]
            assert "DeepSpeed/ZeRO caveat" in logged
            assert "zero_stage=2" in logged

    def test_emit_startup_component_memory_uses_total_loaded_share(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        components = [("small", torch.nn.Linear(1, 1, bias=False)), ("large", torch.nn.Linear(1, 3, bias=False))]
        with patch("builtins.print") as mock_print:
            monitor.emit_startup_component_memory(components, "SGD")

            logged = mock_print.call_args.args[0]
            assert "25.0%" in logged
            assert "75.0%" in logged
            assert "100.0%" in logged

    def test_emit_startup_component_memory_uses_diagnostic_trainable_bytes(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        rows = [
            DiagnosticRow(
                label="unet",
                component_key="denoiser",
                modules_trainable=1,
                modules_total=1,
                params_trainable=1,
                params_total=1,
                param_bytes_trainable=2 * 1024 * 1024,
                param_bytes_total=10 * 1024 * 1024,
            )
        ]
        with patch("builtins.print") as mock_print:
            monitor.emit_startup_component_memory(rows, "AdamW")

            logged = mock_print.call_args.args[0]
            assert "gradients (est): 2.0MB" in logged
            assert "optimizer_state (est): 4.0MB" in logged
            assert "total (est): 16.0MB" in logged


@pytest.mark.unit
class TestResourceMonitorJsonl:
    def test_jsonl_emits_schema_and_core_events(self, tmp_path):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1

        cfg = _make_cfg(
            mode="basic",
            output_jsonl="resource/monitor.jsonl",
            jsonl_flush_mode="line",
            log_every_n_steps=1,
        )

        with patch(
            "library.logging.resource_monitor.events.project_resource_monitor_jsonl_event",
            wraps=project_resource_monitor_jsonl_event,
        ) as mock_project_jsonl:
            monitor = create_resource_monitor(
                accelerator=accelerator,
                resource_monitor_config=cfg,
                output_dir=tmp_path,
                run_identifier="run-123",
                config_name="test_peft_resource_sampled",
                git_sha="abc123def",
                git_dirty=True,
            )

            monitor.start_session()
            monitor.phase_start(training_epoch_phase(0))
            monitor.step_end(global_step=1, epoch=1)
            monitor.phase_end(training_epoch_phase(0))
            monitor.end_session()

        assert mock_project_jsonl.call_count >= 5

        jsonl_path = tmp_path / "resource" / "monitor.jsonl"
        assert jsonl_path.exists()

        events = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        event_names = {event["event"] for event in events}

        assert {"session_start", "phase_start", "phase_end", "step_sample", "session_end"}.issubset(event_names)

        required_keys = {
            "ts",
            "event",
            "rank",
            "world_size",
            "mode",
            "device_scope",
            "run_identifier",
            "config_name",
            "git_sha",
            "git_dirty",
            "global_step",
            "epoch",
            "phase",
            "duration_ms",
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
            "steps_per_sec",
            "samples_per_sec",
            "dropped_samples",
            "collection_ms",
            "deep_alloc_retries",
            "deep_ooms",
            "deep_active_mb",
            "deep_reserved_mb",
            "deep_inactive_split_mb",
            "deep_window_active",
        }
        for event in events:
            assert required_keys.issubset(event.keys())

        session_start = next(event for event in events if event["event"] == "session_start")
        assert session_start["run_identifier"] == "run-123"
        assert session_start["config_name"] == "test_peft_resource_sampled"
        assert session_start["git_sha"] == "abc123def"
        assert "cpu_vms_mb" in session_start
        assert "gpu_allocated_by_device_mb" in session_start
        assert session_start["git_dirty"] is True

    def test_jsonl_projection_recreates_existing_flat_shape_from_canonical_frame(self):
        event_payload = {
            "ts": 123.0,
            "event": "step_sample",
            "rank": 0,
            "world_size": 2,
            "mode": "deep",
            "device_scope": "all_visible",
            "run_identifier": "run-123",
            "config_name": "test-config",
            "git_sha": "abc123def",
            "git_dirty": True,
            "global_step": 4,
            "epoch": 1,
            "phase": training_epoch_phase(0),
            "duration_ms": 12.5,
            "gpu_allocated_mb": 100.0,
            "gpu_allocated_by_device_mb": {"0": 40.0, "1": 60.0},
            "gpu_reserved_mb": 200.0,
            "gpu_reserved_by_device_mb": {"0": 80.0, "1": 120.0},
            "gpu_peak_allocated_mb": 250.0,
            "gpu_peak_allocated_by_device_mb": {"0": 90.0, "1": 160.0},
            "gpu_used_mb": 300.0,
            "gpu_used_by_device_mb": {"10": 190.0, "2": 110.0},
            "cpu_rss_mb": 512.0,
            "cpu_vms_mb": 1024.0,
            "steps_per_sec": 1.25,
            "samples_per_sec": 2.5,
            "dropped_samples": 3,
            "collection_ms": 1.5,
            "deep_alloc_retries": 7,
            "deep_ooms": 1,
            "deep_active_mb": 64.0,
            "deep_reserved_mb": 128.0,
            "deep_inactive_split_mb": 16.0,
            "deep_window_active": True,
        }

        produced_facts = build_resource_monitor_produced_facts(event_payload, run_identifier="run-123", sequence=1)
        projected = project_resource_monitor_jsonl_event(produced_facts)

        assert produced_facts.observation_frame is not None
        assert produced_facts.compatibility is None
        assert produced_facts.as_metadata_items() == (produced_facts.observation_frame,)
        assert projected == event_payload
        assert list(projected["gpu_used_by_device_mb"]) == ["2", "10"]

    def test_jsonl_projection_retains_compatibility_fallback_when_no_frame_is_produced(self):
        event_payload = {
            "ts": 123.0,
            "event": "session_start",
            "rank": 0,
            "world_size": 1,
            "mode": "basic",
            "device_scope": "local",
            "run_identifier": "run-123",
            "config_name": "test-config",
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
            "dropped_samples": 0,
            "collection_ms": None,
            "deep_alloc_retries": None,
            "deep_ooms": None,
            "deep_active_mb": None,
            "deep_reserved_mb": None,
            "deep_inactive_split_mb": None,
            "deep_window_active": False,
        }

        produced_facts = build_resource_monitor_produced_facts(
            event_payload,
            run_identifier="run-123",
            sequence=1,
        )

        assert produced_facts.observation_frame is None
        assert produced_facts.compatibility is not None
        assert produced_facts.as_metadata_items() == (produced_facts.compatibility,)
        assert project_resource_monitor_jsonl_event(produced_facts) == event_payload

    def test_phase_end_forces_flush_in_batch_mode(self, tmp_path):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1

        cfg = _make_cfg(
            mode="basic",
            output_jsonl="resource/batch_flush.jsonl",
            jsonl_flush_mode="batch",
            jsonl_flush_every_n_events=999,
            log_every_n_steps=0,
        )

        monitor = create_resource_monitor(
            accelerator=accelerator,
            resource_monitor_config=cfg,
            output_dir=tmp_path,
        )

        monitor.start_session()
        monitor.phase_start(PHASE_CACHE_LATENTS)
        monitor.phase_end(PHASE_CACHE_LATENTS)

        jsonl_path = tmp_path / "resource" / "batch_flush.jsonl"
        assert jsonl_path.exists()
        lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) >= 3  # session_start + phase_start + forced-flush phase_end

        monitor.end_session()


@pytest.mark.unit
class TestResourceMonitorMigrationEquivalence:
    def test_jsonl_metadata_and_console_surfaces_stay_aligned_during_fact_migration(self, tmp_path):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1
        metadata_runtime = MetadataRuntime()
        cfg = _make_cfg(
            mode="basic",
            output_jsonl="resource/equivalence.jsonl",
            jsonl_flush_mode="line",
            log_every_n_steps=1,
            phase_summary="verbose",
        )

        monitor = create_resource_monitor(
            accelerator=accelerator,
            resource_monitor_config=cfg,
            output_dir=tmp_path,
            run_identifier="run-equivalence",
            config_name="equivalence-config",
            git_sha="abc123def",
            git_dirty=False,
            metadata_runtime=metadata_runtime,
        )
        assert isinstance(monitor, BasicResourceMonitor)

        with patch.object(monitor, "_log_info_external", wraps=monitor._log_info_external) as mock_log_info_external:
            monitor.start_session()
            monitor.phase_start(training_epoch_phase(0))
            monitor.step_end(global_step=1, epoch=1)
            monitor.phase_end(training_epoch_phase(0))
            monitor.end_session()

        jsonl_path = tmp_path / "resource" / "equivalence.jsonl"
        jsonl_events = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        metadata_snapshot = metadata_runtime.snapshot()
        frame_records = metadata_snapshot.records_for(entity_type="resource_observation_frame")
        measurement_records = metadata_snapshot.records_for(entity_type="resource_observation")
        resource_view = ResourceRunView.from_snapshot(
            metadata_snapshot,
            run_identifier="run-equivalence",
        )
        projected_events = list(project_resource_run_compatibility_events(resource_view))

        expected_events = ["session_start", "phase_start", "step_sample", "phase_end", "session_end"]
        assert [event["event"] for event in jsonl_events] == expected_events
        assert projected_events == jsonl_events
        assert metadata_snapshot.events == ()
        assert [record.facts["event_name"] for record in frame_records] == expected_events

        jsonl_by_event = {event["event"]: event for event in jsonl_events}
        projected_by_event = {event["event"]: event for event in projected_events}
        frame_by_event = {record.facts["event_name"]: record for record in frame_records}

        for event_name in expected_events:
            jsonl_event = jsonl_by_event[event_name]
            projected_event = projected_by_event[event_name]
            frame_record = frame_by_event[event_name]

            assert jsonl_event["run_identifier"] == "run-equivalence"
            assert projected_event["run_identifier"] == "run-equivalence"
            assert frame_record.facts["run_identifier"] == "run-equivalence"
            assert frame_record.facts["collection_policy"] == jsonl_event["mode"]
            assert frame_record.facts["rank"] == jsonl_event["rank"]
            assert frame_record.facts["world_size"] == jsonl_event["world_size"]

        step_jsonl = jsonl_by_event["step_sample"]
        step_projected = projected_by_event["step_sample"]
        step_frame = frame_by_event["step_sample"]
        phase_jsonl = jsonl_by_event["phase_end"]
        phase_projected = projected_by_event["phase_end"]
        phase_frame = frame_by_event["phase_end"]

        assert step_jsonl["global_step"] == 1
        assert step_projected["global_step"] == 1
        assert step_frame.facts["global_step"] == 1
        assert step_jsonl["epoch"] == 1
        assert step_projected["epoch"] == 1
        assert step_frame.facts["epoch"] == 1
        assert phase_jsonl["phase"] == training_epoch_phase(0)
        assert phase_projected["phase"] == training_epoch_phase(0)
        assert phase_frame.facts["phase"] == training_epoch_phase(0)

        step_measurements = [record for record in measurement_records if record.facts["frame_identifier"] == step_frame.identity.identifier]
        assert {record.facts["measurement_kind"] for record in step_measurements} >= {"rss", "vms"}
        assert all(record.facts["semantic_class"] == "observation" for record in frame_records)
        assert all(record.facts["semantic_class"] == "observation" for record in measurement_records)

        console_messages = [call.args[0] for call in mock_log_info_external.call_args_list]
        assert any(message.startswith("Resource monitor started:") for message in console_messages)
        assert any(message.startswith("Resource step[") for message in console_messages)
        assert any(message.startswith("Resource phase[") for message in console_messages)
        assert any(message.startswith("Resource session summary:") for message in console_messages)
        assert all("run-equivalence" not in message for message in console_messages)


@pytest.mark.unit
class TestSampledResourceMonitor:
    def test_sampler_thread_lifecycle(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1

        monitor = create_resource_monitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="sampled", sample_interval_sec=0.01),
        )

        assert isinstance(monitor, SampledResourceMonitor)

        monitor.start_session()
        assert monitor._sampler_thread is not None
        assert monitor._sampler_thread.daemon is True

        monitor.end_session()
        assert monitor._sampler_thread is None

    def test_drop_policy_drop_newest_counts_drops(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True

        monitor = SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="sampled", queue_maxsize=1, drop_policy="drop_newest"),
            output_jsonl_path=None,
        )

        sample = SimpleNamespace(
            ts=0.0,
            gpu_used_mb=None,
            gpu_used_by_device_mb=None,
            cpu_rss_mb=None,
            cpu_vms_mb=None,
            collection_ms=None,
        )
        monitor._enqueue_sample(sample)
        monitor._enqueue_sample(sample)

        assert monitor._dropped_samples == 1

    def test_sampled_updates_produce_canonical_observation_frames(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1
        metadata_runtime = MetadataRuntime()
        monitor = SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="sampled"),
            output_jsonl_path=None,
            run_identifier="run-1",
            metadata_runtime=metadata_runtime,
        )
        sample = _SampledMetrics(
            ts=123.0,
            gpu_used_mb=8.0,
            gpu_used_by_device_mb={"0": 8.0},
            gpu_used_source="nvml",
            gpu_used_quality=None,
            cpu_rss_mb=512.0,
            cpu_vms_mb=1024.0,
            collection_ms=1.5,
        )

        monitor._enqueue_sample(sample)
        monitor._process_sampler_updates(max_items=1)

        snapshot = metadata_runtime.snapshot()
        frame = next(
            record
            for record in snapshot.records_for(entity_type="resource_observation_frame")
            if record.facts["collector_id"] == "resource_monitor.sampler"
        )
        measurements = snapshot.records_for(entity_type="resource_observation")

        assert frame.facts["event_name"] == "step_sample"
        assert frame.facts["collector_id"] == "resource_monitor.sampler"
        assert "global_step" not in frame.facts
        used_visible = next(record for record in measurements if record.facts["measurement_kind"] == "used_visible")
        assert used_visible.facts["source"] == "nvml"
        assert used_visible.facts["collector_id"] == "resource_monitor.nvml_gpu_used"
        assert any(record.facts["measurement_kind"] == "collection_duration" for record in measurements)

    def test_sampler_budget_breach_records_collector_status(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1
        metadata_runtime = MetadataRuntime()
        monitor = SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="sampled", max_collection_ms=1.0),
            output_jsonl_path=None,
            run_identifier="run-1",
            metadata_runtime=metadata_runtime,
        )
        sample = _SampledMetrics(
            ts=123.0,
            gpu_used_mb=None,
            gpu_used_by_device_mb=None,
            gpu_used_source=None,
            gpu_used_quality=None,
            cpu_rss_mb=512.0,
            cpu_vms_mb=1024.0,
            collection_ms=2.5,
        )

        monitor._enqueue_sample(sample)
        monitor._process_sampler_updates(max_items=1)

        status = metadata_runtime.snapshot().records_for(entity_type="resource_collector_status")[0]
        assert status.facts["collector_id"] == "resource_monitor.sampler"
        assert status.facts["status"] == "budget_exceeded"
        assert status.facts["degraded"] is True
        assert status.facts["reason"] == "collection_budget_exceeded"

    def test_nvml_failure_records_fallback_status_and_torch_source(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1
        metadata_runtime = MetadataRuntime()
        monitor = SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="sampled", device_scope="local"),
            output_jsonl_path=None,
            run_identifier="run-1",
            metadata_runtime=metadata_runtime,
        )
        monitor._nvml_module = MagicMock()
        monitor._nvml_module.nvmlDeviceGetHandleByIndex.return_value = object()
        monitor._nvml_module.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("nvml boom")

        with (
            patch.object(monitor, "_is_cuda_visible", return_value=True),
            patch("library.logging.resource_monitor.collect.torch.cuda.current_device", return_value=0),
            patch("library.logging.resource_monitor.collect.torch.cuda.mem_get_info", return_value=(3 * 1024 * 1024, 10 * 1024 * 1024)),
        ):
            sample = monitor._collect_sample_metrics()

        status = metadata_runtime.snapshot().records_for(entity_type="resource_collector_status")[0]
        assert sample.gpu_used_mb == 7.0
        assert sample.gpu_used_source == "torch_cuda_mem_get_info"
        assert sample.gpu_used_quality == "fallback"
        assert status.facts["collector_id"] == "resource_monitor.nvml_gpu_used"
        assert status.facts["status"] == "failed"
        assert status.facts["fallback_collector_id"] == "resource_monitor.torch_gpu_used"

    def test_collect_sample_metrics_local_device_records_one_device_map(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True

        monitor = SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="sampled", device_scope="local"),
            output_jsonl_path=None,
        )

        with (
            patch.object(monitor, "_is_cuda_visible", return_value=True),
            patch("library.logging.resource_monitor.collect.torch.cuda.current_device", return_value=2),
            patch("library.logging.resource_monitor.collect.torch.cuda.mem_get_info", return_value=(3 * 1024 * 1024, 10 * 1024 * 1024)),
        ):
            sample = monitor._collect_sample_metrics()

        assert sample.gpu_used_mb == 7.0
        assert sample.gpu_used_by_device_mb == {"2": 7.0}
        assert sample.gpu_used_source == "torch_cuda_mem_get_info"
        assert sample.gpu_used_quality == "fallback"

    def test_collect_sample_metrics_all_visible_sums_device_map(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True

        monitor = SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="sampled", device_scope="all_visible"),
            output_jsonl_path=None,
        )

        def _mem_get_info(index):
            by_device = {
                0: (7 * 1024 * 1024, 10 * 1024 * 1024),
                1: (5 * 1024 * 1024, 10 * 1024 * 1024),
            }
            return by_device[index]

        with (
            patch.object(monitor, "_is_cuda_visible", return_value=True),
            patch("library.logging.resource_monitor.collect.torch.cuda.device_count", return_value=2),
            patch("library.logging.resource_monitor.collect.torch.cuda.mem_get_info", side_effect=_mem_get_info),
        ):
            sample = monitor._collect_sample_metrics()

        assert sample.gpu_used_mb == 8.0
        assert sample.gpu_used_by_device_mb == {"0": 3.0, "1": 5.0}
        assert sample.gpu_used_source == "torch_cuda_mem_get_info"
        assert sample.gpu_used_quality == "fallback"

    def test_deep_step_event_includes_allocator_counters(self, tmp_path):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1

        cfg = _make_cfg(
            mode="deep",
            output_jsonl="resource/deep.jsonl",
            jsonl_flush_mode="line",
            log_every_n_steps=1,
            deep_window_steps=0,
            deep_window_seconds=0.0,
        )

        with patch.object(SampledResourceMonitor, "_start_sampler", autospec=True):
            monitor = create_resource_monitor(
                accelerator=accelerator,
                resource_monitor_config=cfg,
                output_dir=tmp_path,
            )
            assert isinstance(monitor, SampledResourceMonitor)

            with (
                patch.object(monitor, "_is_cuda_visible", return_value=True),
                patch(
                    "library.logging.resource_monitor.collect.torch.cuda.memory_stats",
                    return_value={
                        "num_alloc_retries": 3,
                        "num_ooms": 1,
                        "active_bytes.all.current": 64 * 1024 * 1024,
                        "reserved_bytes.all.current": 128 * 1024 * 1024,
                        "inactive_split_bytes.all.current": 16 * 1024 * 1024,
                    },
                ),
            ):
                monitor.start_session()
                monitor.step_end(global_step=1, epoch=1)
                monitor.end_session()

        jsonl_path = tmp_path / "resource" / "deep.jsonl"
        events = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        deep_step = next(event for event in events if event["event"] == "step_sample" and event["global_step"] == 1)

        assert deep_step["deep_alloc_retries"] == 3
        assert deep_step["deep_ooms"] == 1
        assert deep_step["deep_active_mb"] == 64
        assert deep_step["deep_reserved_mb"] == 128
        assert deep_step["deep_inactive_split_mb"] == 16
        assert deep_step["deep_window_active"] is True

    def test_deep_step_produces_canonical_diagnostic_measurements(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1
        metadata_runtime = MetadataRuntime()
        monitor = SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="deep", log_every_n_steps=1),
            output_jsonl_path=None,
            run_identifier="run-1",
            metadata_runtime=metadata_runtime,
        )
        deep_counters = _DeepCounters(
            alloc_retries=3,
            ooms=1,
            active_mb=64.0,
            reserved_mb=128.0,
            inactive_split_mb=16.0,
            collection_ms=None,
        )

        with patch.object(monitor, "_collect_deep_counters", return_value=(deep_counters, True)):
            monitor.step_end(global_step=1, epoch=0)

        snapshot = metadata_runtime.snapshot()
        frame = next(
            record
            for record in snapshot.records_for(entity_type="resource_observation_frame")
            if record.facts["collector_id"] == "resource_monitor.deep"
        )
        measurements = snapshot.records_for(entity_type="resource_observation")
        frame_metadata = frame.facts["metadata"]

        assert frame.facts["collector_id"] == "resource_monitor.deep"
        assert isinstance(frame_metadata, dict)
        assert frame_metadata["deep_window_active"] is True
        alloc_retries = next(record for record in measurements if record.facts["measurement_kind"] == "alloc_retries")
        inactive_split = next(record for record in measurements if record.facts["measurement_kind"] == "deep_inactive_split")
        assert alloc_retries.facts["collector_id"] == "resource_monitor.deep_allocator"
        assert inactive_split.facts["collector_id"] == "resource_monitor.deep_allocator"

    def test_deep_window_steps_limits_counter_collection(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True

        monitor = SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="deep", log_every_n_steps=1, deep_window_steps=1),
            output_jsonl_path=None,
        )

        with (
            patch.object(monitor, "_is_cuda_visible", return_value=True),
            patch(
                "library.logging.resource_monitor.collect.torch.cuda.memory_stats",
                return_value={
                    "num_alloc_retries": 2,
                    "num_ooms": 0,
                    "active_bytes.all.current": 32 * 1024 * 1024,
                    "reserved_bytes.all.current": 64 * 1024 * 1024,
                    "inactive_split_bytes.all.current": 8 * 1024 * 1024,
                },
            ) as mock_memory_stats,
        ):
            monitor.step_end(global_step=1, epoch=1)
            monitor.step_end(global_step=2, epoch=1)

        assert mock_memory_stats.call_count == 1
        assert monitor._latest_deep_window_active is False

    def test_deep_allocator_failure_records_collector_status(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.process_index = 0
        accelerator.num_processes = 1
        metadata_runtime = MetadataRuntime()
        monitor = SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="deep", log_every_n_steps=1),
            output_jsonl_path=None,
            run_identifier="run-1",
            metadata_runtime=metadata_runtime,
        )

        with (
            patch.object(monitor, "_is_cuda_visible", return_value=True),
            patch("library.logging.resource_monitor.collect.torch.cuda.memory_stats", side_effect=RuntimeError("stats boom")),
        ):
            deep_counters, deep_window_active = monitor._collect_deep_counters(global_step=1)

        status = metadata_runtime.snapshot().records_for(entity_type="resource_collector_status")[0]
        assert deep_counters is None
        assert deep_window_active is True
        assert status.facts["collector_id"] == "resource_monitor.deep_allocator"
        assert status.facts["status"] == "failed"
        assert status.facts["reason"] == "collector_failed"
