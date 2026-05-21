"""Unit tests for library.logging.metrics.

Tests correctness of LR metric emission and tracker initialization,
specifically the bugs fixed in Phase 0:
1. Duplicate LR keys (removed dead fallback code; plan metadata can now supply labels)
2. wandb_run_name overwritten by log_tracker_config
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Lightweight stubs - avoid importing heavy deps (torch, accelerate)
# ---------------------------------------------------------------------------


@dataclass
class _OptimizerConfig:
    optimizer_type: str = "AdamW"


@dataclass
class _TimestepConfig:
    timestep_sampling: str = "uniform"


@dataclass
class _StubCfg:
    optimizer: _OptimizerConfig = field(default_factory=_OptimizerConfig)
    timestep: _TimestepConfig = field(default_factory=_TimestepConfig)


@dataclass
class _OptimizationPlanStub:
    lr_descriptions: list[str]


@dataclass
class _SamplerStub:
    bin_loss_ema: torch.Tensor
    last_entropy_ratio: float
    num_bins: int
    T: int


@dataclass
class _TimestepRuntimeStub:
    sampler: _SamplerStub


class _TrackerStub:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[tuple[dict, int | None]] = []

    def log(self, payload: dict, step: int | None = None):
        self.calls.append((dict(payload), step))


class _AcceleratorStub:
    def __init__(self, trackers: list[_TrackerStub]):
        self.trackers = trackers
        self._trackers_by_name = {tracker.name: tracker for tracker in trackers}

    def get_tracker(self, name: str):
        return self._trackers_by_name[name]


def _make_lr_scheduler(lrs: list[float]):
    """Create a minimal LR scheduler mock returning *lrs* from get_last_lr."""
    sched = MagicMock()
    sched.get_last_lr.return_value = lrs
    return sched


# ===================================================================
# Step metric event and flat rendering
# ===================================================================


class TestStepMetricsEvent:
    """Assert step logs are built from a typed event and rendered compatibly."""

    def test_base_loss_metrics_are_stored_on_event_and_rendered_flat(self):
        from library.logging.metrics import StepMetricsEvent, build_step_metrics_event, render_step_metrics_event

        event = build_step_metrics_event(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4]),
            lr_descriptions=["denoiser"],
        )

        assert isinstance(event, StepMetricsEvent)
        assert event.current_loss == 0.1
        assert event.average_loss == 0.2
        assert event.learning_rates[0].label == "denoiser"
        assert render_step_metrics_event(event) == {
            "loss/current": 0.1,
            "loss/average": 0.2,
            "lr/denoiser": 1e-4,
        }

    def test_scaled_loss_and_modifier_lr_metrics_are_rendered(self):
        from library.logging.metrics import build_step_metrics_event, render_step_metrics_event

        event = build_step_metrics_event(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4]),
            lr_descriptions=["denoiser"],
            current_loss_scaled=0.3,
            average_loss_scaled=0.4,
            modifier_lrs={"min_snr": 2e-5},
        )
        logs = render_step_metrics_event(event)

        assert event.modifier_lrs == (("min_snr", 2e-5),)
        assert logs["loss/current_scaled"] == 0.3
        assert logs["loss/average_scaled"] == 0.4
        assert logs["lr/min_snr"] == 2e-5

    def test_validation_and_norm_metrics_are_rendered(self):
        from library.logging.metrics import generate_step_logs

        logs = generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4]),
            lr_descriptions=["denoiser"],
            keys_scaled=3,
            maximum_norm=1.5,
            mean_norm=0.75,
            mean_grad_norm=0.25,
            mean_combined_norm=0.5,
            current_val_loss=0.11,
            average_val_loss=0.22,
        )

        assert logs["max_norm/keys_scaled"] == 3
        assert logs["max_norm/max_key_norm"] == 1.5
        assert logs["norm/avg_key_norm"] == 0.75
        assert logs["norm/avg_grad_norm"] == 0.25
        assert logs["norm/avg_combined_norm"] == 0.5
        assert logs["loss/current_val_loss"] == 0.11
        assert logs["loss/average_val_loss"] == 0.22

    def test_sampler_metrics_are_rendered(self):
        from library.logging.metrics import build_step_metrics_event, render_step_metrics_event

        event = build_step_metrics_event(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4]),
            lr_descriptions=["denoiser"],
            timestep_runtime=_TimestepRuntimeStub(
                sampler=_SamplerStub(
                    bin_loss_ema=torch.tensor([1.0, 3.0]),
                    last_entropy_ratio=0.8,
                    num_bins=2,
                    T=10,
                )
            ),
            timesteps=torch.tensor([0, 2, 6, 9]),
        )
        logs = render_step_metrics_event(event)

        assert event.sampler is not None
        assert event.sampler.ema_loss_bins == (1.0, 3.0)
        assert logs["sampler/ema_loss_mean"] == pytest.approx(2.0)
        assert logs["sampler/ema_loss_std"] == pytest.approx(torch.tensor([1.0, 3.0]).std().item())
        assert logs["sampler_ema_loss_bins/bin_0"] == 1.0
        assert logs["sampler_ema_loss_bins/bin_1"] == 3.0
        assert logs["sampler/entropy_ratio"] == 0.8
        assert logs["sampler_timestep_hist/bin_0"] == 2.0
        assert logs["sampler_timestep_hist/bin_1"] == 2.0


# ===================================================================
# generate_step_logs – LR key correctness
# ===================================================================


class TestGenerateStepLogsLrKeys:
    """Assert LR keys match the resolved trainer-facing group labels with no duplicates."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from library.logging.metrics import generate_step_logs

        self.generate_step_logs = generate_step_logs

    def test_single_group(self):
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4]),
            lr_descriptions=["denoiser"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/denoiser"]

    def test_two_groups_te_and_unet(self):
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([5e-5, 1e-4]),
            lr_descriptions=["textencoder", "denoiser"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/textencoder", "lr/denoiser"]
        assert len(lr_keys) == len(set(lr_keys)), f"Duplicate LR keys: {lr_keys}"

    def test_multiple_text_encoders(self):
        """SDXL-style: TE1, TE2, UNet each with own LR."""
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([5e-5, 3e-5, 1e-4]),
            lr_descriptions=["text_encoder1", "text_encoder2", "denoiser"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/text_encoder1", "lr/text_encoder2", "lr/denoiser"]
        assert len(lr_keys) == len(set(lr_keys)), f"Duplicate LR keys: {lr_keys}"

    def test_custom_descriptions(self):
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4, 2e-4]),
            lr_descriptions=["my_unet", "my_te"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/my_unet", "lr/my_te"]

    def test_optimization_plan_descriptions_take_priority(self):
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4, 2e-4]),
            lr_descriptions=["legacy_a", "legacy_b"],
            optimization_plan=_OptimizationPlanStub(["denoiser", "text_encoder1"]),
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/denoiser", "lr/text_encoder1"]

    def test_optimization_plan_does_not_require_legacy_descriptions(self):
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4, 2e-4]),
            optimization_plan=_OptimizationPlanStub(["denoiser", "text_encoder1"]),
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/denoiser", "lr/text_encoder1"]

    def test_no_duplicate_keys(self):
        """Ensure exactly one key per param group, no extra keys from stale fallback logic."""
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([3e-4]),
            lr_descriptions=["denoiser"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert len(lr_keys) == 1, f"Expected exactly 1 LR key, got {lr_keys}"

    def test_mismatched_label_count_raises_value_error(self):
        with pytest.raises(ValueError, match="learning-rate labels has 1 elements but lr_scheduler has 2 learning rates"):
            self.generate_step_logs(
                cfg=_StubCfg(),
                current_loss=0.1,
                avr_loss=0.2,
                lr_scheduler=_make_lr_scheduler([1e-4, 2e-4]),
                lr_descriptions=["denoiser"],
            )

    @pytest.mark.parametrize("optimizer_type", ["DAdaptAdam", "Prodigy"])
    def test_dadapt_and_prodigy_derived_lrs_use_validated_labels(self, optimizer_type: str):
        scheduler = _make_lr_scheduler([1e-4, 2e-4])
        scheduler.optimizers = [
            MagicMock(
                param_groups=[
                    {"d": 2.0, "lr": 0.1},
                    {"d": 3.0, "lr": 0.2},
                ]
            )
        ]

        logs = self.generate_step_logs(
            cfg=_StubCfg(optimizer=_OptimizerConfig(optimizer_type=optimizer_type)),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=scheduler,
            lr_descriptions=["denoiser", "text_encoder"],
        )

        assert logs["lr/d*lr/denoiser"] == pytest.approx(0.2)
        assert logs["lr/d*lr/text_encoder"] == pytest.approx(0.6)

    def test_prodigy_plus_schedule_free_derived_lr_is_preserved(self):
        optimizer = MagicMock(param_groups=[{"d": 4.0, "lr": 0.5}])

        logs = self.generate_step_logs(
            cfg=_StubCfg(optimizer=_OptimizerConfig(optimizer_type="ProdigyPlusScheduleFree")),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4]),
            lr_descriptions=["denoiser"],
            optimizer=optimizer,
        )

        assert logs["lr/d*lr"] == pytest.approx(2.0)


# ===================================================================
# tracker routing
# ===================================================================


class TestTrackerRouting:
    """Assert metric payloads route to trackers with explicit backend semantics."""

    def test_step_logging_uses_global_step_as_tracker_step(self):
        from library.logging import metrics

        accelerator = MagicMock()
        logs = {"loss/current": 0.1}

        with pytest.MonkeyPatch.context() as monkeypatch:
            routed = MagicMock()
            monkeypatch.setattr(metrics, "log_metrics_to_trackers", routed)
            metrics.step_logging(accelerator, logs, global_step=12, epoch=3)

        routed.assert_called_once_with(accelerator, logs, 12, 12, 3)

    def test_epoch_logging_uses_epoch_as_tracker_step(self):
        from library.logging import metrics

        accelerator = MagicMock()
        logs = {"loss/current": 0.1}

        with pytest.MonkeyPatch.context() as monkeypatch:
            routed = MagicMock()
            monkeypatch.setattr(metrics, "log_metrics_to_trackers", routed)
            metrics.epoch_logging(accelerator, logs, global_step=12, epoch=3)

        routed.assert_called_once_with(accelerator, logs, 3, 12, 3)

    def test_backend_routing_copies_wandb_payload_without_mutating_source(self):
        from library.logging.metrics import log_metrics_to_trackers

        tensorboard = _TrackerStub("tensorboard")
        wandb = _TrackerStub("wandb")
        generic = _TrackerStub("mlflow")
        accelerator = _AcceleratorStub([tensorboard, wandb, generic])
        logs = {"loss/current": 0.1}

        log_metrics_to_trackers(accelerator, logs, step_value=7, global_step=12, epoch=3)

        assert tensorboard.calls == [({"loss/current": 0.1}, 7)]
        assert wandb.calls == [({"loss/current": 0.1, "global_step": 12, "epoch": 3}, None)]
        assert generic.calls == [({"loss/current": 0.1}, 7)]
        assert logs == {"loss/current": 0.1}


# ===================================================================
# init_trackers – W&B run name retention
# ===================================================================


@dataclass
class _LoggingConfigStub:
    """Minimal LoggingConfig for init_trackers tests."""

    wandb_run_name: str | None = None
    log_tracker_config: dict | None = field(default_factory=dict)
    log_tracker_name: str | None = None
    console_log_level: str | None = None
    console_log_file: str | None = None
    console_log_simple: bool = False
    wandb_api_key: str | None = None


class TestInitTrackers:
    """Assert init_trackers preserves wandb_run_name through deep-merge."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from library.logging.metrics import init_trackers

        self.init_trackers = init_trackers

    def _make_accelerator(self):
        acc = MagicMock()
        acc.is_main_process = True
        return acc

    def test_only_wandb_run_name(self):
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(wandb_run_name="my-run")
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        init_kwargs = call_kwargs.kwargs.get("init_kwargs") or call_kwargs[1].get("init_kwargs")
        assert init_kwargs["wandb"]["name"] == "my-run"

    def test_only_log_tracker_config(self):
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(
            log_tracker_config={"wandb": {"entity": "my-team"}},
        )
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        init_kwargs = call_kwargs.kwargs.get("init_kwargs") or call_kwargs[1].get("init_kwargs")
        assert init_kwargs["wandb"]["entity"] == "my-team"
        # No name key since wandb_run_name is not set
        assert "name" not in init_kwargs["wandb"]

    def test_both_wandb_run_name_and_tracker_config_preserves_name(self):
        """wandb_run_name should survive when log_tracker_config also has wandb keys."""
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(
            wandb_run_name="my-run",
            log_tracker_config={"wandb": {"entity": "my-team", "tags": ["test"]}},
        )
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        init_kwargs = call_kwargs.kwargs.get("init_kwargs") or call_kwargs[1].get("init_kwargs")
        # Both name and entity should exist
        assert init_kwargs["wandb"]["name"] == "my-run", "wandb_run_name was overwritten!"
        assert init_kwargs["wandb"]["entity"] == "my-team"
        assert init_kwargs["wandb"]["tags"] == ["test"]

    def test_tracker_config_without_wandb_key(self):
        """Non-wandb tracker config should not affect wandb run name."""
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(
            wandb_run_name="my-run",
            log_tracker_config={"tensorboard": {"log_dir": "/tmp/tb"}},
        )
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        init_kwargs = call_kwargs.kwargs.get("init_kwargs") or call_kwargs[1].get("init_kwargs")
        assert init_kwargs["wandb"]["name"] == "my-run"
        assert init_kwargs["tensorboard"]["log_dir"] == "/tmp/tb"

    def test_custom_tracker_name(self):
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(log_tracker_name="my-project")
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        # First positional arg is tracker_name
        assert call_kwargs[0][0] == "my-project"

    def test_default_tracker_name_when_none(self):
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub()
        self.init_trackers(acc, cfg, "fallback_name")

        call_kwargs = acc.init_trackers.call_args
        assert call_kwargs[0][0] == "fallback_name"

    def test_not_called_on_non_main_process(self):
        acc = self._make_accelerator()
        acc.is_main_process = False
        cfg = _LoggingConfigStub(wandb_run_name="my-run")
        self.init_trackers(acc, cfg, "default_project")

        acc.init_trackers.assert_not_called()


class TestLoggingTrainingObserver:
    """Assert observer-level run and artifact registration remains structured."""

    def test_start_and_finish_run_forward_to_sink_once(self):
        from library.logging.metrics import LoggingTrainingObserver

        sink = MagicMock()
        observer = LoggingTrainingObserver(console=MagicMock(), metrics_sink=sink)
        config = {"wandb_api_key": "*****"}

        observer.start_run("training", config)
        config["wandb_api_key"] = "mutated"
        observer.finish_run()
        observer.finish_run()

        sink.start_run.assert_called_once_with("training", {"wandb_api_key": "*****"})
        sink.finish_run.assert_called_once()
        assert observer.run_name == "training"
        assert observer.run_config == {"wandb_api_key": "*****"}

    def test_log_artifact_records_path_kind_and_metadata(self):
        from library.logging.metrics import LoggingTrainingObserver
        from library.metadata import METADATA_PAYLOAD_VERSION
        from library.metadata.emitters import build_logged_artifact_metadata

        observer = LoggingTrainingObserver(console=MagicMock())

        observer.log_artifact("/tmp/report.md", kind="benchmark_report", metadata={"format": "markdown"})

        assert len(observer.artifacts) == 1
        artifact = observer.artifacts[0]
        assert artifact.path == "/tmp/report.md"
        assert artifact.kind == "benchmark_report"
        assert artifact.metadata == {"format": "markdown"}
        result = build_logged_artifact_metadata(artifact)

        assert len(result.events) == 1
        assert result.events[0].event_type == "artifact_registered"
        assert result.events[0].facts["metadata"] == {"format": "markdown"}
        assert result.events[0].schema_version == METADATA_PAYLOAD_VERSION

    def test_metadata_snapshot_collects_run_and_artifact_boundaries(self):
        from library.logging.metrics import LoggingTrainingObserver

        observer = LoggingTrainingObserver(console=MagicMock(), metrics_sink=MagicMock())

        observer.start_run("training", {"wandb_api_key": "*****"})
        observer.log_artifact("/tmp/report.md", kind="benchmark_report", metadata={"format": "markdown"})
        observer.finish_run()

        snapshot = observer.metadata_snapshot()

        assert [event.event_type for event in snapshot.events] == [
            "run_started",
            "artifact_registered",
            "run_finished",
        ]
        assert snapshot.events[0].identity.identifier == "training"
        assert snapshot.events[1].facts["metadata"] == {"format": "markdown"}
        assert snapshot.events[2].facts["status"] == "finished"
