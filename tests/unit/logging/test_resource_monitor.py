"""Unit tests for library.logging.resource_monitor."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from library.logging.resource_monitor import BasicResourceMonitor, NoOpResourceMonitor, create_resource_monitor


@pytest.mark.unit
class TestResourceMonitorFactory:
    def test_create_resource_monitor_disabled_returns_noop(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        cfg = SimpleNamespace(enabled=False, mode="basic")

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
        cfg = SimpleNamespace(
            enabled=True,
            mode="basic",
            rank_scope="main",
            phase_summary=True,
            component_breakdown=True,
            log_every_n_steps=1,
        )

        monitor = create_resource_monitor(accelerator=accelerator, resource_monitor_config=cfg)

        assert isinstance(monitor, BasicResourceMonitor)


@pytest.mark.unit
class TestBasicResourceMonitorBehavior:
    def _make_cfg(self):
        return SimpleNamespace(
            enabled=True,
            mode="basic",
            rank_scope="main",
            phase_summary=True,
            component_breakdown=True,
            log_every_n_steps=1,
        )

    def test_phase_and_step_logging_paths_do_not_raise(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(accelerator=accelerator, resource_monitor_config=self._make_cfg())

        with patch("library.logging.resource_monitor.logger") as mock_logger:
            monitor.start_session()
            monitor.phase_start("training_epoch_1")
            monitor.step_end(global_step=1, epoch=1)
            monitor.phase_end("training_epoch_1")
            monitor.end_session()

            assert mock_logger.info.call_count >= 3

    def test_emit_startup_component_memory_logs_estimate(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(accelerator=accelerator, resource_monitor_config=self._make_cfg())

        components = {"unet": torch.nn.Linear(4, 4)}
        with patch("library.logging.resource_monitor.logger") as mock_logger:
            monitor.emit_startup_component_memory(components, "AdamW")
            mock_logger.info.assert_called()

    def test_emit_startup_component_memory_accepts_component_pair_list(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(accelerator=accelerator, resource_monitor_config=self._make_cfg())

        components = [("adapter", torch.nn.Linear(4, 4))]
        with patch("library.logging.resource_monitor.logger") as mock_logger:
            monitor.emit_startup_component_memory(components, "AdamW")
            mock_logger.info.assert_called()

    def test_emit_startup_component_memory_ignores_malformed_component_entries(self):
        accelerator = MagicMock()
        accelerator.is_main_process = True
        monitor = BasicResourceMonitor(accelerator=accelerator, resource_monitor_config=self._make_cfg())

        components = [("adapter", torch.nn.Linear(4, 4)), ("bad_only_name",), 123]
        with patch("library.logging.resource_monitor.logger") as mock_logger:
            monitor.emit_startup_component_memory(components, "AdamW")
            mock_logger.info.assert_called()
