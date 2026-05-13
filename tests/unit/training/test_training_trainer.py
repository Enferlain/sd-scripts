import logging
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from library.logging.console import MainProcessConsole
from library.training.runners.trainer import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.cfg = MagicMock()
        self.cfg.objective.path = "ddpm"
        self.strategies = MagicMock()
        self.mode = MagicMock()
        self.trainer = Trainer(self.cfg, self.strategies, self.mode)

    def test_accelerator_access_before_init_raises_error(self):
        """Test that accessing accelerator before setup raises RuntimeError."""
        with self.assertRaises(RuntimeError) as cm:
            _ = self.trainer.accelerator
        self.assertIn("Accelerator not initialized", str(cm.exception))

    def test_accelerator_access_after_setup(self):
        """Test that accessing accelerator after assignment works."""
        mock_accelerator = MagicMock()
        # Simulate setup setting the private attribute directly
        self.trainer._accelerator = mock_accelerator

        # Should not raise
        acc = self.trainer.accelerator
        self.assertEqual(acc, mock_accelerator)

    def test_is_main_process_property(self):
        """Test is_main_process property logic."""
        # Case 1: Not initialized -> returns True (default safety)
        self.assertTrue(self.trainer.is_main_process)

        # Case 2: Initialized and is main
        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True
        self.trainer._accelerator = mock_accelerator
        self.assertTrue(self.trainer.is_main_process)

        # Case 3: Initialized and is NOT main
        mock_accelerator.is_main_process = False
        self.trainer._accelerator = mock_accelerator
        self.assertFalse(self.trainer.is_main_process)

    def test_save_checkpoint_passes_target_model_to_mode(self):
        """Test that save_checkpoint passes unwrapped_adapter through as target_model.

        Regression test: EDM2 loss weight checkpoints pass a non-adapter
        model (e.g. edm2.model). The mode must receive this as
        target_model so it saves the correct weights.
        """
        mock_accelerator = MagicMock()
        self.trainer._accelerator = mock_accelerator

        # Set up required trainer state for save_checkpoint
        self.cfg.output.saving.no_metadata = True
        self.trainer._minimum_metadata = {"ss_adapter_module": "test"}
        self.trainer._metadata = {}
        self.strategies.get_model_metadata.return_value = {}

        edm2_model = MagicMock(name="edm2_loss_weights")
        self.trainer.save_checkpoint(
            "edm2_weights.safetensors",
            edm2_model,
            step=100,
            epoch=1,
            dtype_override=None,
        )

        # Verify mode.save_checkpoint received the EDM2 model, not the adapter
        self.mode.save_checkpoint.assert_called_once()
        call_kwargs = self.mode.save_checkpoint.call_args
        self.assertIs(call_kwargs.kwargs["target_model"], edm2_model)

    def test_save_checkpoint_passes_adapter_as_target_model(self):
        """Test that standard adapter checkpoints pass the adapter through."""
        mock_accelerator = MagicMock()
        self.trainer._accelerator = mock_accelerator

        self.cfg.output.saving.no_metadata = False
        self.trainer._minimum_metadata = {}
        self.trainer._metadata = {"ss_adapter_module": "test"}
        self.strategies.get_model_metadata.return_value = {}

        mock_adapter = MagicMock(name="adapter")
        self.trainer.save_checkpoint(
            "adapter.safetensors",
            mock_adapter,
            step=50,
            epoch=1,
        )

        call_kwargs = self.mode.save_checkpoint.call_args
        self.assertIs(call_kwargs.kwargs["target_model"], mock_adapter)

    def test_train_ends_resource_monitor_session_when_training_fails(self):
        """Resource monitor session should close even if training aborts mid-run."""
        mock_monitor = MagicMock()
        mock_progress_bar = MagicMock()

        self.trainer._resource_monitor = None
        self.trainer._progress_bar = mock_progress_bar
        self.trainer.setup = MagicMock()
        self.trainer.run_caching = MagicMock()
        self.trainer.prepare_models = MagicMock()
        self.trainer.prepare_optimizer = MagicMock()
        self.trainer._initialize_training_run_state = MagicMock()
        self.trainer._run_startup_eval_actions = MagicMock()
        self.trainer.run_training_loop = MagicMock(side_effect=RuntimeError("training failed"))
        self.trainer._finalize_training = MagicMock()

        def assign_monitor():
            self.trainer._resource_monitor = mock_monitor

        self.trainer.setup.side_effect = assign_monitor

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.trainer.train()

        mock_progress_bar.close.assert_called_once()
        self.assertIsNone(self.trainer._progress_bar)
        mock_monitor.end_session.assert_called_once()
        self.trainer._finalize_training.assert_not_called()

    @patch("library.training.runners.trainer.write_run_report")
    def test_train_writes_enabled_benchmark_report_after_monitor_closes(self, mock_write_run_report):
        """Benchmark reports should be written after final resource events are flushed."""
        mock_monitor = MagicMock()
        call_order: list[str] = []
        mock_monitor.end_session.side_effect = lambda: call_order.append("end_session")
        mock_write_run_report.side_effect = lambda *_args, **_kwargs: call_order.append("write_run_report")
        self.cfg.output.logging.benchmark_report.enabled = True

        self.trainer._resource_monitor = None
        self.trainer.setup = MagicMock()
        self.trainer.run_caching = MagicMock()
        self.trainer.prepare_models = MagicMock()
        self.trainer.prepare_optimizer = MagicMock()
        self.trainer._initialize_training_run_state = MagicMock()
        self.trainer._run_startup_eval_actions = MagicMock()
        self.trainer.run_training_loop = MagicMock()
        self.trainer._finalize_training = MagicMock()

        def assign_monitor():
            self.trainer._resource_monitor = mock_monitor

        self.trainer.setup.side_effect = assign_monitor

        self.trainer.train()

        mock_monitor.end_session.assert_called_once()
        mock_write_run_report.assert_called_once_with(self.trainer, succeeded=True, error_message=None)
        self.assertEqual(call_order, ["end_session", "write_run_report"])

    @patch("library.training.runners.trainer.write_run_report")
    def test_train_logs_benchmark_report_artifacts_when_report_is_written(self, mock_write_run_report):
        """Generated benchmark report files should be registered as observer artifacts."""
        mock_monitor = MagicMock()
        self.cfg.output.logging.benchmark_report.enabled = True

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path = Path(temp_dir) / "benchmark_report.md"
            json_path = markdown_path.with_suffix(".json")
            markdown_path.write_text("# report\n", encoding="utf-8")
            json_path.write_text("{}\n", encoding="utf-8")
            mock_write_run_report.return_value = markdown_path

            self.trainer._resource_monitor = None
            self.trainer._observer = MagicMock()
            self.trainer.setup = MagicMock()
            self.trainer.run_caching = MagicMock()
            self.trainer.prepare_models = MagicMock()
            self.trainer.prepare_optimizer = MagicMock()
            self.trainer._initialize_training_run_state = MagicMock()
            self.trainer._run_startup_eval_actions = MagicMock()
            self.trainer.run_training_loop = MagicMock()
            self.trainer._finalize_training = MagicMock()

            def assign_monitor():
                self.trainer._resource_monitor = mock_monitor

            self.trainer.setup.side_effect = assign_monitor

            self.trainer.train()

            self.trainer._observer.log_artifact.assert_any_call(
                str(markdown_path),
                kind="benchmark_report",
                metadata={"format": "markdown"},
            )
            self.trainer._observer.log_artifact.assert_any_call(
                str(json_path),
                kind="benchmark_report",
                metadata={"format": "json"},
            )
            self.assertEqual(self.trainer._observer.log_artifact.call_count, 2)
            self.trainer._observer.finish_run.assert_called_once()
            self.assertEqual(
                [call[0] for call in self.trainer._observer.method_calls],
                ["log_artifact", "log_artifact", "finish_run"],
            )

    @patch("library.training.runners.trainer.write_run_report")
    def test_train_writes_failure_benchmark_report_when_enabled(self, mock_write_run_report):
        """Enabled benchmark reports should capture the failure state without swallowing the exception."""
        mock_monitor = MagicMock()
        self.cfg.output.logging.benchmark_report.enabled = True

        self.trainer._resource_monitor = None
        self.trainer.setup = MagicMock()
        self.trainer.run_caching = MagicMock()
        self.trainer.prepare_models = MagicMock()
        self.trainer.prepare_optimizer = MagicMock()
        self.trainer._initialize_training_run_state = MagicMock()
        self.trainer._run_startup_eval_actions = MagicMock()
        self.trainer.run_training_loop = MagicMock(side_effect=RuntimeError("training failed"))
        self.trainer._finalize_training = MagicMock()

        def assign_monitor():
            self.trainer._resource_monitor = mock_monitor

        self.trainer.setup.side_effect = assign_monitor

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.trainer.train()

        mock_monitor.end_session.assert_called_once()
        mock_write_run_report.assert_called_once_with(self.trainer, succeeded=False, error_message="training failed")
        self.trainer._finalize_training.assert_not_called()

    @patch("library.utils.device_utils.clean_memory_on_device")
    @patch("library.training.phases.validation.ValidationScheduler")
    @patch("library.losses.loss.EMARecorder")
    @patch("library.logging.metrics.build_tracker_config", return_value={"console_log_level": "INFO"})
    @patch("library.logging.metrics.resolve_tracker_name", return_value="unit-training")
    @patch("library.logging.metrics.init_trackers")
    def test_initialize_tracking_state_starts_observer_run(
        self,
        mock_init_trackers,
        mock_resolve_tracker_name,
        mock_build_tracker_config,
        mock_ema_recorder,
        mock_validation_scheduler,
        mock_clean_memory,
    ):
        """Tracking initialization should bracket later metrics/artifacts with a run lifecycle."""
        del mock_ema_recorder, mock_validation_scheduler, mock_clean_memory
        self.trainer._accelerator = MagicMock()
        self.trainer._accelerator.trackers = []
        self.trainer._accelerator.device = "cpu"
        self.trainer._accelerator.is_local_main_process = False
        self.trainer.max_train_steps = 1
        self.trainer._initial_step = 0
        self.trainer._observer = MagicMock()

        self.trainer._initialize_tracking_state()

        mock_init_trackers.assert_called_once_with(self.trainer.accelerator, self.cfg.output.logging, "training")
        mock_resolve_tracker_name.assert_called_once_with(self.cfg.output.logging, "training")
        mock_build_tracker_config.assert_called_once_with(self.cfg.output.logging)
        self.trainer._observer.start_run.assert_called_once_with("unit-training", {"console_log_level": "INFO"})

    def test_finalize_training_closes_progress_bar_before_final_save_work(self):
        """Final checkpoint logging should happen after the progress bar is out of the way."""
        self.trainer._accelerator = MagicMock()
        self.trainer._progress_bar = MagicMock()
        progress_bar = self.trainer._progress_bar
        self.trainer._metadata = {}
        self.trainer.optimizer = MagicMock()
        self.trainer.optimization_plan = MagicMock()
        self.trainer._save_final_state_if_enabled = MagicMock()
        self.trainer._save_final_checkpoint_artifacts = MagicMock()

        self.trainer._finalize_training()

        self.trainer.accelerator.end_training.assert_called_once()
        progress_bar.close.assert_called_once()
        self.assertIsNone(self.trainer._progress_bar)
        self.trainer._save_final_state_if_enabled.assert_called_once()
        self.trainer._save_final_checkpoint_artifacts.assert_called_once()

    def test_log_progress_message_uses_console_when_available(self):
        """Progress-safe lifecycle logs should use the console transport when present."""
        self.trainer._console = MagicMock()

        self.trainer.log_progress_message("prepared epoch 0", tag="epoch", stacklevel=4)

        self.trainer._console.log_external.assert_called_once_with(
            "prepared epoch 0",
            tag="epoch",
            level="info",
            stacklevel=5,
        )

    @patch("library.training.runners.trainer.logger")
    def test_log_progress_message_falls_back_to_logger(self, mock_logger):
        """Without a console seam, progress-safe lifecycle logs should fall back to the module logger."""
        self.trainer._console = None

        self.trainer.log_progress_message("prepared epoch 0", tag="epoch", stacklevel=4)

        mock_logger.log.assert_called_once_with(
            unittest.mock.ANY,
            "[epoch] prepared epoch 0",
            stacklevel=4,
        )

    def test_print_progress_message_uses_console_when_available(self):
        """Progress-safe plain lines should use the console transport when present."""
        self.trainer._console = MagicMock()

        self.trainer.print_progress_message("Epoch 1/1")

        self.trainer._console.print_external.assert_called_once_with("Epoch 1/1")

    def test_print_progress_message_falls_back_to_accelerator(self):
        """Without a console seam, progress-safe plain lines should fall back to accelerator printing."""
        self.trainer._console = None
        self.trainer._accelerator = MagicMock()

        self.trainer.print_progress_message("Epoch 1/1")

        self.trainer.accelerator.print.assert_called_once_with("Epoch 1/1")

    def test_log_progress_message_preserves_caller_file_and_line_with_console(self):
        """Progress-safe console logging should still attribute records to the real caller."""

        class _CaptureHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.record = None

            def emit(self, record):
                self.record = record

        test_logger = logging.getLogger("tests.unit.training.test_training_trainer.progress")
        test_logger.handlers = []
        test_logger.propagate = False
        test_logger.setLevel(logging.INFO)
        capture = _CaptureHandler()
        test_logger.addHandler(capture)

        self.trainer._console = MainProcessConsole(is_main_process=True, logger=test_logger)

        def _emit():
            expected_lineno = _emit.__code__.co_firstlineno + 2
            self.trainer.log_progress_message("prepared epoch 0", tag="epoch", stacklevel=2)
            return expected_lineno

        with patch("library.logging.console.tqdm.external_write_mode", return_value=nullcontext()):
            expected_lineno = _emit()

        assert capture.record is not None
        assert os.path.basename(capture.record.pathname) == "test_training_trainer.py"
        assert capture.record.lineno == expected_lineno

    def test_order_memory_components_follows_diagnostic_row_order(self):
        """Startup resource rows should follow the same component order as the startup diagnostics table."""
        unet = MagicMock()
        clip_l = MagicMock()
        clip_g = MagicMock()
        vae = MagicMock()
        components = [("clip_l", clip_l), ("clip_g", clip_g), ("vae", vae), ("unet", unet)]
        component_rows = [
            SimpleNamespace(label="unet"),
            SimpleNamespace(label="clip_l"),
            SimpleNamespace(label="clip_g"),
            SimpleNamespace(label="vae"),
        ]

        ordered = self.trainer._order_memory_components(components, component_rows)

        assert [label for label, _module in ordered] == ["unet", "clip_l", "clip_g", "vae"]

    def test_run_startup_eval_actions_validation_only_skips_sampling(self):
        """Startup validation should not force startup sampling."""
        self.trainer._accelerator = MagicMock()
        self.trainer._accelerator.device = "cpu"
        self.trainer._validation_scheduler = MagicMock()
        self.trainer._validation_scheduler.should_run.return_value = True
        self.trainer._val_dataloader = MagicMock()
        self.trainer._cyclic_val_dataloader = MagicMock()
        self.trainer._val_loss_recorder = MagicMock()
        self.trainer._primary_trainable = MagicMock()
        self.trainer.text_encoders = [MagicMock()]
        self.trainer.denoiser = MagicMock()
        self.trainer.vae = MagicMock()
        self.trainer.tokenizers = [MagicMock()]
        self.trainer._text_encoder = self.trainer.text_encoders
        self.trainer.vae_dtype = MagicMock()
        self.trainer.weight_dtype = MagicMock()
        self.trainer.objective_runtime = SimpleNamespace(
            name="ddpm",
            num_train_timesteps=1000,
            timestep_runtime=None,
            loss_modifier=MagicMock(),
            advance_to_step=MagicMock(return_value=None),
            update_from_batch=MagicMock(),
        )
        self.trainer.num_batches_per_epoch = 5
        self.trainer._train_text_encoder = False
        self.trainer.strategies.calculate_val_loss.return_value = (0.5, 0.5)

        with patch("library.training.sample_generation.sample_images_check", return_value=False):
            self.trainer._run_startup_eval_actions()

        self.trainer.mode.set_eval.assert_called_once_with(self.trainer)
        self.trainer.mode.set_train.assert_called_once_with(self.trainer)
        self.trainer.strategies.sample_images.assert_not_called()
        self.trainer.strategies.calculate_val_loss.assert_called_once()

    def test_run_startup_eval_actions_sampling_only_skips_validation(self):
        """Startup sampling should not force startup validation."""
        self.trainer._accelerator = MagicMock()
        self.trainer._accelerator.device = "cpu"
        self.trainer._validation_scheduler = MagicMock()
        self.trainer._validation_scheduler.should_run.return_value = False
        self.trainer._val_dataloader = None
        self.trainer._cyclic_val_dataloader = None
        self.trainer._val_loss_recorder = MagicMock()
        self.trainer._primary_trainable = MagicMock()
        self.trainer.text_encoders = [MagicMock()]
        self.trainer.denoiser = MagicMock()
        self.trainer.vae = MagicMock()
        self.trainer.tokenizers = [MagicMock()]
        self.trainer._text_encoder = self.trainer.text_encoders
        self.trainer.vae_dtype = MagicMock()
        self.trainer.weight_dtype = MagicMock()
        self.trainer.objective_runtime = SimpleNamespace(
            name="ddpm",
            num_train_timesteps=1000,
            timestep_runtime=None,
            loss_modifier=MagicMock(),
            advance_to_step=MagicMock(return_value=None),
            update_from_batch=MagicMock(),
        )
        self.trainer.num_batches_per_epoch = 5
        self.trainer._train_text_encoder = False

        with patch("library.training.sample_generation.sample_images_check", return_value=True):
            self.trainer._run_startup_eval_actions()

        self.trainer.mode.set_eval.assert_called_once_with(self.trainer)
        self.trainer.mode.set_train.assert_called_once_with(self.trainer)
        self.trainer.strategies.sample_images.assert_called_once()
        self.trainer.strategies.calculate_val_loss.assert_not_called()
