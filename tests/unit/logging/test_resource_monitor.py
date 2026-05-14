"""Unit tests for library.logging.resource_monitor."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from library.logging.summaries import DiagnosticRow
from library.logging.resource_monitor import (
    BasicResourceMonitor,
    NoOpResourceMonitor,
    SampledResourceMonitor,
    create_resource_monitor,
)


def _make_cfg(**overrides):
    base = {
        "enabled": True,
        "mode": "basic",
        "rank_scope": "main",
        "phase_summary": True,
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


@pytest.mark.unit
class TestBasicResourceMonitorBehavior:
    def test_phase_and_step_logging_paths_do_not_raise(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        with patch("library.logging.resource_monitor.logger") as mock_logger:
            monitor.start_session()
            monitor.phase_start("training_epoch_1")
            monitor.step_end(global_step=1, epoch=1)
            monitor.phase_end("training_epoch_1")
            monitor.end_session()

            assert mock_logger.info.call_count >= 3

    def test_phase_and_step_logging_uses_external_write_mode(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=_make_cfg(mode="basic"),
            output_jsonl_path=None,
        )

        with (
            patch("library.logging.resource_monitor.logger") as mock_logger,
            patch("library.logging.resource_monitor.tqdm.external_write_mode", return_value=nullcontext()) as mock_external,
        ):
            monitor.start_session()
            monitor.phase_start("training_epoch_1")
            monitor.step_end(global_step=1, epoch=1)
            monitor.phase_end("training_epoch_1")
            monitor.end_session()

            assert mock_external.call_count >= 4
            assert mock_logger.info.call_count >= 3

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

        monitor = create_resource_monitor(
            accelerator=accelerator,
            resource_monitor_config=cfg,
            output_dir=tmp_path,
            run_id="run-123",
            config_name="test_peft_resource_sampled",
            git_sha="abc123def",
            git_dirty=True,
        )

        monitor.start_session()
        monitor.phase_start("training_epoch_1")
        monitor.step_end(global_step=1, epoch=1)
        monitor.phase_end("training_epoch_1")
        monitor.end_session()

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
            "run_id",
            "config_name",
            "git_sha",
            "git_dirty",
            "global_step",
            "epoch",
            "phase",
            "duration_ms",
            "gpu_allocated_mb",
            "gpu_reserved_mb",
            "gpu_peak_allocated_mb",
            "gpu_used_mb",
            "cpu_rss_mb",
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
        assert session_start["run_id"] == "run-123"
        assert session_start["config_name"] == "test_peft_resource_sampled"
        assert session_start["git_sha"] == "abc123def"
        assert session_start["git_dirty"] is True

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
        monitor.phase_start("latent_caching")
        monitor.phase_end("latent_caching")

        jsonl_path = tmp_path / "resource" / "batch_flush.jsonl"
        assert jsonl_path.exists()
        lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) >= 3  # session_start + phase_start + forced-flush phase_end

        monitor.end_session()


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

        sample = SimpleNamespace(ts=0.0, gpu_used_mb=None, cpu_rss_mb=None, collection_ms=None)
        monitor._enqueue_sample(sample)
        monitor._enqueue_sample(sample)

        assert monitor._dropped_samples == 1

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
                    "library.logging.resource_monitor.torch.cuda.memory_stats",
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
                "library.logging.resource_monitor.torch.cuda.memory_stats",
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
