import logging
import tempfile
import unittest

from contextlib import nullcontext
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from library.config.dataclasses.output import MetadataConfig
from library.logging.console import MainProcessConsole
from library.logging.runtime_trace import RuntimeTrace
from library.metadata.dataclasses import RunMetadataFacts
from library.metadata import (
    build_model_realization_state,
    MetadataRuntime,
    ModelArtifactFacts,
    ModelFamilyMetadataContribution,
    ModelFamilyMetadataField,
)
from library.models import LoadedModelComponent
from library.strategies.base.features import ModelFamilyMetadataStrategy
from library.training.metadata import TrainingMetadataState
from library.training.runners.trainer import Trainer

from library.metadata.exports.resource import (
    RESOURCE_EXPORT_SCHEMA_VERSION,
    RESOURCE_MONITOR_JSONL_SCHEMA,
    RESOURCE_REPORT_EXPORT_SCHEMA,
)


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

    def test_file_model_realization_metadata_files_once_and_reuses_identities(self):
        self.cfg.model.model_type = "sd15"
        self.trainer.session_id = 42
        self.trainer._model_version = "sd_v1"
        self.trainer.loaded_components = (
            LoadedModelComponent(
                key="text_encoder1",
                public_name="clip_l",
                module=object(),
                roles=("text_encoder",),
            ),
            LoadedModelComponent(
                key="text_encoder2",
                public_name="clip_g",
                module=None,
                roles=("text_encoder",),
            ),
            LoadedModelComponent(
                key="denoiser",
                public_name="unet",
                module=object(),
                roles=("denoiser",),
            ),
        )
        runtime = MetadataRuntime()
        self.trainer._observer = SimpleNamespace(metadata_runtime=runtime)

        state = self.trainer._file_model_realization_metadata()
        first_snapshot = runtime.snapshot()
        repeated_state = self.trainer._file_model_realization_metadata()
        second_snapshot = runtime.snapshot()

        self.assertIs(repeated_state, state)
        self.assertEqual(state.realization.family.family_identifier, "sd")
        self.assertEqual(
            [record.identity.entity_type for record in first_snapshot.records],
            ["model_realization", "model_component", "model_component", "model_component"],
        )
        self.assertEqual(len(second_snapshot.records), len(first_snapshot.records))
        self.assertEqual(self.trainer.model_realization_identifier, "run/42/model/training-target")
        self.assertEqual(
            self.trainer.get_model_component_identifier("denoiser"),
            "run/42/model/training-target/component/denoiser",
        )
        self.assertIsNone(self.trainer.get_model_component_identifier("missing"))
        component_records = first_snapshot.records_for(entity_type="model_component")
        self.assertEqual(
            [record.facts["component_key"] for record in component_records],
            ["text_encoder1", "text_encoder2", "denoiser"],
        )
        self.assertEqual(
            [record.facts["present"] for record in component_records],
            [True, False, True],
        )

    def test_file_model_realization_metadata_includes_optional_family_contribution(self):
        class FamilyMetadataStrategy(ModelFamilyMetadataStrategy):
            def resolve_model_family_metadata(
                self,
                cfg,
                *,
                run_identifier: str,
                realization_identifier: str,
            ) -> ModelFamilyMetadataContribution:
                del cfg
                return ModelFamilyMetadataContribution.for_realization(
                    run_identifier=run_identifier,
                    realization_identifier=realization_identifier,
                    contribution_namespace="future.runtime",
                    contribution_version="1",
                    fields=(ModelFamilyMetadataField(name="custom_runtime_flag", value=True),),
                )

        self.cfg.model.model_type = "future"
        self.trainer.strategies = FamilyMetadataStrategy()
        self.trainer.session_id = 7
        self.trainer._model_version = "v1"
        self.trainer.loaded_components = ()
        runtime = MetadataRuntime()
        self.trainer._observer = SimpleNamespace(metadata_runtime=runtime)

        self.trainer._file_model_realization_metadata()

        self.assertEqual(
            [record.identity.entity_type for record in runtime.snapshot().records],
            ["model_realization", "model_family_facts"],
        )

    def test_file_model_realization_metadata_rejects_unavailable_identity(self):
        self.cfg.model.model_type = "sdxl"
        self.trainer.session_id = None
        self.trainer._model_version = "sdxl"
        self.trainer._observer = SimpleNamespace(metadata_runtime=MetadataRuntime())

        with self.assertRaisesRegex(ValueError, "integer trainer session identifier"):
            self.trainer._file_model_realization_metadata()

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
        self.trainer._metadata_state = TrainingMetadataState(
            full=RunMetadataFacts(run_identifier="test", metadata={}),
        )
        self.trainer._build_checkpoint_metadata = MagicMock(return_value={})

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
        self.trainer._metadata_state = TrainingMetadataState(
            full=RunMetadataFacts(run_identifier="test", metadata={"adapter_method": "test"}),
        )
        self.trainer._build_checkpoint_metadata = MagicMock(return_value={})

        mock_adapter = MagicMock(name="adapter")
        self.trainer.save_checkpoint(
            "adapter.safetensors",
            mock_adapter,
            step=50,
            epoch=1,
        )

        call_kwargs = self.mode.save_checkpoint.call_args
        self.assertIs(call_kwargs.kwargs["target_model"], mock_adapter)

    def test_save_checkpoint_logs_canonical_phase_message_from_trainer(self):
        """Trainer should own the canonical checkpoint save-start lifecycle log."""
        self.trainer._accelerator = MagicMock()
        self.cfg.output.saving.output_dir = "/tmp/out"
        self.trainer._metadata_state = TrainingMetadataState(
            full=RunMetadataFacts(run_identifier="test", metadata={}),
        )
        self.trainer._resource_monitor = MagicMock()
        self.trainer.runtime_trace = MagicMock()
        self.mode.save_checkpoint = MagicMock()
        self.trainer._build_checkpoint_metadata = MagicMock(return_value={})

        with patch("library.training.runners.trainer.logger") as mock_logger:
            self.trainer.save_checkpoint(
                "adapter.safetensors",
                MagicMock(name="adapter"),
                step=50,
                epoch=1,
            )

        logged_paths = [str(call.args[2]).replace("\\", "/") for call in mock_logger.info.call_args_list if len(call.args) >= 3]
        self.assertIn("/tmp/out/adapter.safetensors", logged_paths)

    def test_save_checkpoint_normalizes_windows_like_output_path_in_log_message(self):
        """Checkpoint lifecycle logs should avoid mixed path separators in rendered paths."""
        self.trainer._accelerator = MagicMock()
        self.cfg.output.saving.output_dir = "D:/Projects/sd-scripts/tests/test_output"
        self.trainer._metadata_state = TrainingMetadataState(
            full=RunMetadataFacts(run_identifier="test", metadata={}),
        )
        self.trainer._resource_monitor = MagicMock()
        self.trainer.runtime_trace = MagicMock()
        self.mode.save_checkpoint = MagicMock()
        self.trainer._build_checkpoint_metadata = MagicMock(return_value={})

        with patch("library.training.runners.trainer.logger") as mock_logger:
            self.trainer.save_checkpoint(
                "adapter.safetensors",
                MagicMock(name="adapter"),
                step=50,
                epoch=1,
            )

        logged_paths = [str(call.args[2]).replace("\\", "/") for call in mock_logger.info.call_args_list if len(call.args) >= 3]
        self.assertIn("D:/Projects/sd-scripts/tests/test_output/adapter.safetensors", logged_paths)

    def test_save_checkpoint_records_saved_event_before_runtime_trace_phase_end(self):
        """Checkpoint completion event should land before the enclosing save phase closes."""

        class _IncrementingClock:
            def __init__(self):
                self.value = 0.0

            def __call__(self):
                current = self.value
                self.value += 1.0
                return current

        self.trainer._accelerator = MagicMock()
        self.cfg.output.saving.output_dir = "/tmp/out"
        self.trainer._metadata_state = TrainingMetadataState(
            full=RunMetadataFacts(run_identifier="test", metadata={}),
        )
        self.trainer._resource_monitor = MagicMock()
        self.trainer.runtime_trace = RuntimeTrace(launched_perf=0.0, clock=_IncrementingClock())
        self.mode.save_checkpoint = MagicMock()
        self.trainer._build_checkpoint_metadata = MagicMock(return_value={})

        self.trainer.save_checkpoint(
            "adapter.safetensors",
            MagicMock(name="adapter"),
            step=50,
            epoch=1,
        )

        summary = self.trainer.runtime_trace.summary()
        checkpoint_phase = next(phase for phase in summary["phases"] if phase["tag"] == "checkpoint.save")
        saved_event = next(event for event in summary["events"] if event["tag"] == "checkpoint.saved")

        self.assertLess(checkpoint_phase["start_offset_s"], saved_event["offset_s"])
        self.assertLess(saved_event["offset_s"], checkpoint_phase["end_offset_s"])

    def test_build_checkpoint_metadata_files_typed_artifact_linked_to_accepted_realization(self):
        runtime = MetadataRuntime()
        self.trainer._observer = SimpleNamespace(metadata_runtime=runtime)
        self.trainer._metadata_state = TrainingMetadataState(
            full=RunMetadataFacts(
                run_identifier="42",
                metadata={"seed": "7", "sd_scripts_commit_hash": "abc123"},
            ),
        )
        self.trainer._model_version = "sdxl_base_v1-0"
        self.trainer._model_realization_state = build_model_realization_state(
            run_identifier="42",
            realization_key="training-target",
            family_identifier="sdxl",
            model_version=self.trainer._model_version,
            loaded_components=(),
        )
        runtime.file(self.trainer._model_realization_state.realization)
        self.trainer.objective = SimpleNamespace(name="ddpm")
        self.cfg.model.model_type = "sdxl"
        self.cfg.objective.prediction = "epsilon"
        self.cfg.output.metadata = MetadataConfig(metadata_title="Typed Adapter")
        self.cfg.output.saving.no_metadata = False
        self.cfg.data.preprocessing.resolution = "1024,768"
        self.cfg.timestep.min_timestep = 10
        self.cfg.timestep.max_timestep = 900
        self.cfg.training.clip_skip = 2
        self.mode.checkpoint_artifact_role = "adapter"
        self.mode.resolve_checkpoint_artifact_format.return_value = "safetensors"

        def resolve_facts(context):
            return ModelArtifactFacts.from_resolution_context(
                context,
                architecture="stable-diffusion-xl-v1-base/lora",
                implementation="https://github.com/Stability-AI/generative-models",
                default_title="Adapter",
            )

        self.strategies.resolve_model_artifact_facts.side_effect = resolve_facts

        metadata = Trainer._build_checkpoint_metadata(
            self.trainer,
            ckpt_name="adapter.safetensors",
            step=12,
            epoch=3,
        )

        context = self.strategies.resolve_model_artifact_facts.call_args.args[0]
        self.assertEqual(context.artifact_identifier, "adapter.safetensors")
        self.assertEqual(context.artifact_role, "adapter")
        self.assertEqual(context.serialization_format, "safetensors")
        self.assertEqual(context.resolution, (1024, 768))
        self.assertEqual(context.realization_identifier, "run/42/model/training-target")
        self.assertEqual(context.implementation_version, "sd-scripts/abc123")
        artifact_record = runtime.snapshot().record_for(
            entity_type="model_artifact",
            identifier="adapter.safetensors",
        )
        self.assertIsNotNone(artifact_record)
        self.assertTrue(
            any(
                edge.source == artifact_record.identity
                and edge.target.identifier == "run/42/model/training-target"
                and edge.relationship == "derived_from"
                for edge in runtime.snapshot().edges
            )
        )
        self.assertEqual(metadata["modelspec.title"], "Typed Adapter")
        self.assertEqual(metadata["kuro.model.artifact.artifact_role"], "adapter")

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

    def test_resource_cleanup_failure_does_not_mask_original_training_failure(self):
        """Best-effort resource cleanup must preserve the original exception."""
        mock_monitor = MagicMock()
        mock_monitor.end_session.side_effect = RuntimeError("resource cleanup OOM")

        self.trainer._resource_monitor = mock_monitor
        self.trainer.setup = MagicMock()
        self.trainer.run_caching = MagicMock()
        self.trainer.prepare_models = MagicMock()
        self.trainer.prepare_optimizer = MagicMock()
        self.trainer._initialize_training_run_state = MagicMock()
        self.trainer._run_startup_eval_actions = MagicMock()
        self.trainer.run_training_loop = MagicMock(side_effect=RuntimeError("original training OOM"))
        self.trainer._finalize_training = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "original training OOM"):
            self.trainer.train()

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
            resource_jsonl_path = Path(temp_dir) / "resource_monitor.jsonl"
            markdown_path.write_text("# report\n", encoding="utf-8")
            json_path.write_text("{}\n", encoding="utf-8")
            resource_jsonl_path.write_text("{}\n", encoding="utf-8")
            mock_write_run_report.return_value = markdown_path
            mock_monitor.jsonl_path = resource_jsonl_path

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
                str(resource_jsonl_path),
                kind="resource_monitor",
                metadata={
                    "format": "jsonl",
                    "schema_name": RESOURCE_MONITOR_JSONL_SCHEMA,
                    "schema_version": RESOURCE_EXPORT_SCHEMA_VERSION,
                },
            )
            self.trainer._observer.log_artifact.assert_any_call(
                str(markdown_path),
                kind="benchmark_report",
                metadata={
                    "format": "markdown",
                    "schema_name": RESOURCE_REPORT_EXPORT_SCHEMA,
                    "schema_version": RESOURCE_EXPORT_SCHEMA_VERSION,
                },
            )
            self.trainer._observer.log_artifact.assert_any_call(
                str(json_path),
                kind="benchmark_report",
                metadata={
                    "format": "json",
                    "schema_name": RESOURCE_REPORT_EXPORT_SCHEMA,
                    "schema_version": RESOURCE_EXPORT_SCHEMA_VERSION,
                },
            )
            self.assertEqual(self.trainer._observer.log_artifact.call_count, 3)
            self.trainer._observer.finish_run.assert_called_once()
            self.assertEqual(
                [call[0] for call in self.trainer._observer.method_calls],
                ["log_artifact", "log_artifact", "log_artifact", "finish_run"],
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

    @patch("library.training.runners.trainer.write_run_report")
    def test_train_writes_failure_report_for_startup_phase_exception(self, mock_write_run_report):
        """Startup-phase failures should still finalize observability surfaces cleanly."""
        mock_monitor = MagicMock()
        self.cfg.output.logging.benchmark_report.enabled = True
        self.trainer.runtime_trace = MagicMock()

        self.trainer._resource_monitor = None
        self.trainer.setup = MagicMock()
        self.trainer.run_caching = MagicMock()
        self.trainer.prepare_models = MagicMock()
        self.trainer.prepare_optimizer = MagicMock()
        self.trainer._initialize_training_run_state = MagicMock(side_effect=RuntimeError("startup failed"))
        self.trainer._run_startup_eval_actions = MagicMock()
        self.trainer.run_training_loop = MagicMock()
        self.trainer._finalize_training = MagicMock()

        def assign_monitor():
            self.trainer._resource_monitor = mock_monitor

        self.trainer.setup.side_effect = assign_monitor

        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            self.trainer.train()

        mock_monitor.end_session.assert_called_once()
        self.trainer.runtime_trace.finish.assert_called_once()
        mock_write_run_report.assert_called_once_with(self.trainer, succeeded=False, error_message="startup failed")
        self.trainer._run_startup_eval_actions.assert_not_called()
        self.trainer.run_training_loop.assert_not_called()
        self.trainer._finalize_training.assert_not_called()

    @patch("library.training.runners.trainer.write_run_report")
    def test_train_writes_failure_report_for_startup_eval_exception(self, mock_write_run_report):
        """Startup-eval failures should still close the monitor and write a failure report."""
        mock_monitor = MagicMock()
        self.cfg.output.logging.benchmark_report.enabled = True
        self.trainer.runtime_trace = MagicMock()

        self.trainer._resource_monitor = None
        self.trainer.setup = MagicMock()
        self.trainer.run_caching = MagicMock()
        self.trainer.prepare_models = MagicMock()
        self.trainer.prepare_optimizer = MagicMock()
        self.trainer._initialize_training_run_state = MagicMock()
        self.trainer._run_startup_eval_actions = MagicMock(side_effect=RuntimeError("startup eval failed"))
        self.trainer.run_training_loop = MagicMock()
        self.trainer._finalize_training = MagicMock()

        def assign_monitor():
            self.trainer._resource_monitor = mock_monitor

        self.trainer.setup.side_effect = assign_monitor

        with self.assertRaisesRegex(RuntimeError, "startup eval failed"):
            self.trainer.train()

        mock_monitor.end_session.assert_called_once()
        self.trainer.runtime_trace.finish.assert_called_once()
        mock_write_run_report.assert_called_once_with(self.trainer, succeeded=False, error_message="startup eval failed")
        self.trainer.run_training_loop.assert_not_called()
        self.trainer._finalize_training.assert_not_called()

    @patch("library.utils.device_utils.clean_memory_on_device")
    @patch("library.training.phases.validation.ValidationScheduler")
    @patch("library.losses.loss.EMARecorder")
    @patch("library.logging.metrics.build_tracker_config", return_value={"console_log_level": "INFO"})
    @patch("library.logging.metrics.resolve_tracker_name", return_value="unit-training")
    @patch("library.utils.common_utils.resolve_hydra_runtime_context", return_value=("presets/unit_training", []))
    @patch("library.logging.metrics.init_trackers")
    def test_initialize_tracking_state_starts_observer_run(
        self,
        mock_init_trackers,
        mock_resolve_hydra_runtime_context,
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
        mock_resolve_hydra_runtime_context.assert_called_once_with()
        mock_build_tracker_config.assert_called_once_with(self.cfg.output.logging)
        self.trainer._observer.start_run.assert_called_once_with(
            "unit-training",
            {"console_log_level": "INFO"},
            run_identifier=str(self.trainer.session_id),
            mode_name=type(self.trainer.mode).__name__,
            strategy_name=type(self.trainer.strategies).__name__,
            optimizer_name=None,
            config_name="presets/unit_training",
            global_step=0,
            epoch=0,
        )

    def test_initialize_training_run_state_starts_tracking_before_startup_summary(self):
        call_order: list[str] = []
        self.trainer._initialize_tracking_state = MagicMock(side_effect=lambda: call_order.append("tracking"))
        self.trainer._emit_training_startup_summary = MagicMock(side_effect=lambda: call_order.append("summary"))
        self.trainer._initialize_training_metadata = MagicMock(side_effect=lambda **_: call_order.append("metadata"))
        self.trainer._initialize_training_runtime = MagicMock(side_effect=lambda: call_order.append("runtime"))
        self.trainer._compute_total_batch_size = MagicMock(return_value=16)

        self.trainer._initialize_training_run_state()

        assert call_order == ["tracking", "summary", "metadata", "runtime"]
        self.trainer._initialize_training_metadata.assert_called_once_with(total_batch_size=16)

    def test_finalize_training_closes_progress_bar_before_final_save_work(self):
        """Final checkpoint logging should happen after the progress bar is out of the way."""
        self.trainer._accelerator = MagicMock()
        self.trainer._progress_bar = MagicMock()
        progress_bar = self.trainer._progress_bar
        self.trainer._metadata_state = TrainingMetadataState(
            full=RunMetadataFacts(run_identifier="test", metadata={}),
        )
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
        normalized_basename = (
            PureWindowsPath(capture.record.pathname).name if "\\" in capture.record.pathname else Path(capture.record.pathname).name
        )
        assert normalized_basename == "test_training_trainer.py"
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
