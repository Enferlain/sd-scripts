"""
Unit tests for library/training/modes/finetune_mode.py

Tests FineTuneMode protocol hook implementations with mocked trainer state.
Verifies strategy delegation (no SDXL-specific logic in mode).
"""

import pytest
from unittest.mock import MagicMock, PropertyMock, patch
from types import SimpleNamespace

import torch
from torch import nn

from library.optimization.types import OptimizerBuildResult
from library.training.modes.finetune_mode import FineTuneMode


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mode():
    """Create a FineTuneMode instance."""
    return FineTuneMode()


@pytest.fixture
def mock_cfg():
    """Create a mock config for fine-tune testing."""
    cfg = MagicMock()

    # optimizer.learning_rates
    cfg.optimizer.learning_rates.base = 1e-5
    cfg.optimizer.learning_rates.denoiser = None  # Default: use base LR
    cfg.optimizer.learning_rates.text_encoders = None  # Default: train TEs with base LR
    cfg.optimizer.optimizer_type = "AdamW"
    cfg.optimizer.optimizer_args = None
    cfg.optimizer.scheduler = "constant"

    # performance
    cfg.performance.precision.full_fp16 = False
    cfg.performance.precision.full_bf16 = False
    cfg.performance.deepspeed = SimpleNamespace(deepspeed=False)

    # output
    cfg.output.saving.output_dir = "/tmp/output"
    cfg.output.saving.save_model_as = "safetensors"
    cfg.output.huggingface = None
    cfg.output.metadata = MagicMock()

    # loss
    cfg.loss.v_parameterization = False

    # model
    cfg.model.pretrained_model_name_or_path = "/models/sdxl-base"

    return cfg


@pytest.fixture
def mock_accelerator():
    """Create a mock Accelerator."""
    accelerator = MagicMock()
    accelerator.device = torch.device("cpu")
    accelerator.is_main_process = True
    accelerator.unwrap_model = MagicMock(side_effect=lambda x: x)
    accelerator.prepare = MagicMock(side_effect=lambda *args: args if len(args) > 1 else args[0])
    accelerator.print = MagicMock()
    return accelerator


@pytest.fixture
def mock_denoiser():
    """Create a mock denoiser."""
    denoiser = MagicMock(spec=nn.Module)
    denoiser.parameters = MagicMock(return_value=[nn.Parameter(torch.randn(4, 4))])
    return denoiser


@pytest.fixture
def mock_text_encoders():
    """Create mock SDXL text encoders with TE1 having text_model structure."""
    te1 = MagicMock(spec=nn.Module)
    te1.parameters = MagicMock(return_value=[nn.Parameter(torch.randn(2, 2))])
    te1.text_model = MagicMock()
    te1.text_model.encoder = MagicMock()
    te1.text_model.encoder.layers = [MagicMock() for _ in range(12)]
    te1.text_model.final_layer_norm = MagicMock()

    te2 = MagicMock(spec=nn.Module)
    te2.parameters = MagicMock(return_value=[nn.Parameter(torch.randn(2, 2))])

    return [te1, te2]


@pytest.fixture
def mock_strategies():
    """Create a mock strategies object with strategy methods."""
    strategies = MagicMock()
    strategies.post_process_trainable = MagicMock()
    strategies.save_model_checkpoint = MagicMock()
    return strategies


@pytest.fixture
def mock_trainer(mock_cfg, mock_accelerator, mock_denoiser, mock_text_encoders, mock_strategies):
    """Create a mock Trainer for FineTuneMode testing."""
    trainer = MagicMock()
    trainer.cfg = mock_cfg
    type(trainer).accelerator = PropertyMock(return_value=mock_accelerator)
    type(trainer).is_main_process = PropertyMock(return_value=True)

    trainer.denoiser = mock_denoiser
    trainer.vae = MagicMock()
    trainer.text_encoders = mock_text_encoders
    trainer._text_encoder = mock_text_encoders
    trainer.strategies = mock_strategies

    trainer.weight_dtype = torch.float16
    trainer.save_dtype = torch.float16

    trainer.optimizer = MagicMock()
    trainer.lr_scheduler = MagicMock()

    trainer._current_epoch_state = SimpleNamespace(value=0)
    trainer._current_step_state = SimpleNamespace(value=0)

    trainer._train_denoiser = True
    trainer._train_text_encoder = False
    trainer._primary_trainable = None
    trainer._grad_sync_handle = None

    return trainer


# =============================================================================
# Test: prepare_trainables — strategy delegation
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestPrepareTrainables:
    """Test prepare_trainables uses shared trainability helpers + strategy hooks."""

    def test_sets_denoiser_train_flag_from_learning_rates(self, mode, mock_trainer):
        """Denoiser train flag follows the configured learning rates."""
        mode.prepare_trainables(mock_trainer)

        assert mock_trainer._train_denoiser is True

    def test_sets_text_encoder_train_flag_from_learning_rates(self, mode, mock_trainer):
        """TE train flag follows the configured learning rates."""
        mode.prepare_trainables(mock_trainer)

        assert mock_trainer._train_text_encoder is True

    def test_delegates_post_process_to_strategy(self, mode, mock_trainer):
        """post_process_trainable is called on strategy after unfreezing."""
        mode.prepare_trainables(mock_trainer)

        mock_trainer.strategies.post_process_trainable.assert_called_once()

    def test_uses_trainability_helper_te_flags_when_no_lr_override(self, mode, mock_trainer):
        """When text_encoders LR is None, all TEs train with the base LR."""
        mock_trainer.cfg.optimizer.learning_rates.text_encoders = None

        mode.prepare_trainables(mock_trainer)

        assert mode._te_train_flags == [True, True]

    def test_per_te_lr_list_controls_te_flags(self, mode, mock_trainer):
        """Per-TE LR list drives per-encoder training flags directly."""
        mock_trainer.cfg.optimizer.learning_rates.text_encoders = [1e-5, 0.0]

        mode.prepare_trainables(mock_trainer)

        assert mode._te_train_flags == [True, False]

    def test_zero_te_lr_disables_all_tes(self, mode, mock_trainer):
        """Zero text_encoders LR disables both TEs."""
        mock_trainer.cfg.optimizer.learning_rates.text_encoders = 0.0

        mode.prepare_trainables(mock_trainer)

        assert mode._te_train_flags == [False, False]
        assert mock_trainer._train_text_encoder is False

    def test_sets_primary_trainable(self, mode, mock_trainer):
        """Primary trainable is set to the denoiser."""
        mode.prepare_trainables(mock_trainer)

        assert mock_trainer._primary_trainable is mock_trainer.denoiser


# =============================================================================
# Test: configure_trainable_precision
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestConfigureTrainablePrecision:
    """Test configure_trainable_precision hook."""

    def test_full_fp16_casts_denoiser_and_tes(self, mode, mock_trainer):
        """Full fp16 casts the denoiser and trained TEs to weight_dtype."""
        mock_trainer.cfg.performance.precision.full_fp16 = True
        mode._te_train_flags = [True, True]

        mode.configure_trainable_precision(mock_trainer)

        mock_trainer.denoiser.to.assert_called_with(torch.float16)
        for te in mock_trainer.text_encoders:
            te.to.assert_called()

    def test_non_trained_tes_cast_to_weight_dtype(self, mode, mock_trainer):
        """Non-trained TEs are cast to weight_dtype for efficiency."""
        mode._te_train_flags = [True, False]

        mode.configure_trainable_precision(mock_trainer)

        mock_trainer.text_encoders[1].to.assert_called_with(torch.float16)


# =============================================================================
# Test: build_optimizer_params
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestBuildOptimizerParams:
    """Test build_optimizer_params hook."""

    def test_returns_optimizer_build_result(self, mode, mock_trainer):
        """Returns a normalized build result with logical-group metadata."""
        mode._te_train_flags = [True, False]
        mock_trainer._train_denoiser = True

        with (
            patch("library.training.modes.finetune_mode.build_finetune_grouping") as mock_grouping,
            patch("library.training.modes.finetune_mode.get_optimizer") as mock_get_opt,
            patch("library.training.modes.finetune_mode.get_optimizer_train_eval_fn") as mock_get_fn,
        ):
            mock_grouping.return_value.execution_groups = [MagicMock()]
            mock_grouping.return_value.logical_groups = [
                SimpleNamespace(metric_name="denoiser"),
                SimpleNamespace(metric_name="text_encoder1"),
            ]
            mock_get_opt.return_value = ("AdamW", {}, MagicMock())
            mock_get_fn.return_value = (lambda: None, lambda: None)

            result = mode.build_optimizer_params(mock_trainer)

        mock_grouping.assert_called_once_with(
            denoiser=mock_trainer.denoiser,
            train_denoiser=True,
            text_encoders=mock_trainer.text_encoders,
            te_train_flags=[True, False],
            learning_rates=mock_trainer.cfg.optimizer.learning_rates,
        )
        assert isinstance(result, OptimizerBuildResult)
        assert result.optimizer_name == "AdamW"
        assert any(group.metric_name == "denoiser" for group in result.optimization_plan.logical_groups)
        assert result.optimization_plan.execution_groups == mock_grouping.return_value.execution_groups
        assert result.optimization_plan.parameter_groups == mock_grouping.return_value.execution_groups
        assert result.lr_descriptions == ["denoiser", "text_encoder1"]

    def test_rejects_block_lr(self, mode, mock_trainer):
        """Block LR in optimizer_args raises NotImplementedError."""
        mock_trainer.cfg.optimizer.optimizer_args = ["block_lr=0.01"]
        mode._te_train_flags = [False, False]

        with pytest.raises(NotImplementedError, match="Block-level learning rates"):
            mode.build_optimizer_params(mock_trainer)


# =============================================================================
# Test: prepare_with_accelerator — DeepSpeed dynamic kwargs
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestPrepareWithAccelerator:
    """Test prepare_with_accelerator hook."""

    def test_sets_grad_sync_handle_and_primary_trainable(self, mode, mock_trainer):
        """_grad_sync_handle and _primary_trainable are set to the denoiser."""
        mode._te_train_flags = [False, False]
        mock_trainer._train_denoiser = True

        mode.prepare_with_accelerator(mock_trainer)

        assert mock_trainer._grad_sync_handle is mock_trainer.denoiser
        assert mock_trainer._primary_trainable is mock_trainer.denoiser

    def test_non_deepspeed_prepares_denoiser_directly_with_accelerator(self, mode, mock_trainer):
        """Non-DeepSpeed fine-tune now prepares the denoiser directly."""
        mode._te_train_flags = [False, False]
        mock_trainer._train_denoiser = True

        mode.prepare_with_accelerator(mock_trainer)

        mock_trainer.accelerator.prepare.assert_any_call(mock_trainer.denoiser)

    def test_deepspeed_uses_dynamic_kwargs(self, mode, mock_trainer):
        """DeepSpeed path builds dynamic kwargs, no fixed TE count."""
        mock_trainer.cfg.performance.deepspeed = SimpleNamespace(deepspeed=True)
        mode._te_train_flags = [True, False]
        mock_trainer._train_denoiser = True

        with patch("library.training.modes.finetune_mode.deepspeed_utils") as mock_ds:
            mock_ds_model = MagicMock()
            mock_ds.prepare_deepspeed_model.return_value = mock_ds_model
            mock_trainer.accelerator.prepare.return_value = (mock_ds_model, MagicMock(), MagicMock())

            mode.prepare_with_accelerator(mock_trainer)

            # Verify dynamic kwargs — only text_encoder1 is passed (flag[1] is False)
            mock_ds.prepare_deepspeed_model.assert_called_once()
            call_kwargs = mock_ds.prepare_deepspeed_model.call_args
            assert "denoiser" in call_kwargs.kwargs
            assert "text_encoder1" in call_kwargs.kwargs
            assert "text_encoder2" not in call_kwargs.kwargs
            # No adapter kwarg for fine-tune
            assert "adapter" not in call_kwargs.kwargs

    def test_deepspeed_supports_3_encoders(self, mode, mock_trainer):
        """DeepSpeed path supports variable number of text encoders."""
        mock_trainer.cfg.performance.deepspeed = SimpleNamespace(deepspeed=True)
        te3 = MagicMock(spec=nn.Module)
        mock_trainer.text_encoders.append(te3)
        mode._te_train_flags = [True, True, True]
        mock_trainer._train_denoiser = True

        with patch("library.training.modes.finetune_mode.deepspeed_utils") as mock_ds:
            mock_ds_model = MagicMock()
            mock_ds.prepare_deepspeed_model.return_value = mock_ds_model
            mock_trainer.accelerator.prepare.return_value = (mock_ds_model, MagicMock(), MagicMock())

            mode.prepare_with_accelerator(mock_trainer)

            call_kwargs = mock_ds.prepare_deepspeed_model.call_args
            assert "text_encoder1" in call_kwargs.kwargs
            assert "text_encoder2" in call_kwargs.kwargs
            assert "text_encoder3" in call_kwargs.kwargs


# =============================================================================
# Test: register_state_hooks
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestRegisterStateHooks:
    """Test register_state_hooks hook."""

    def test_returns_resume_state(self, mode, mock_trainer):
        """Returns an explicit ResumeState object."""
        mode._te_train_flags = [False, False]

        result = mode.register_state_hooks(mock_trainer)

        assert result.epoch is None
        assert result.step is None


# =============================================================================
# Test: on_epoch_start / on_step_start / on_step_end
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestCallbacks:
    """Test per-epoch / per-step callbacks."""

    def test_on_epoch_start_calls_train(self, mode, mock_trainer):
        """on_epoch_start sets training models to train mode."""
        mock_trainer._train_denoiser = True
        mode._te_train_flags = [True, False]

        mode.on_epoch_start(mock_trainer)

        mock_trainer.denoiser.train.assert_called_once()
        mock_trainer.text_encoders[0].train.assert_called()
        mock_trainer.text_encoders[1].train.assert_not_called()

    def test_on_step_end_returns_empty(self, mode, mock_trainer):
        """on_step_end returns empty dict (no regularization)."""
        assert mode.on_step_end(mock_trainer) == {}


# =============================================================================
# Test: get_trainable_params / set_eval / set_train
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestEvalTrain:
    """Test eval/train transitions and param retrieval."""

    def test_get_trainable_params_returns_all(self, mode, mock_trainer):
        """Returns denoiser + trained TE params."""
        mock_trainer._train_denoiser = True
        mode._te_train_flags = [True, False]

        denoiser_params = [nn.Parameter(torch.randn(4, 4))]
        te1_params = [nn.Parameter(torch.randn(2, 2))]
        mock_trainer.denoiser.parameters.return_value = denoiser_params
        mock_trainer.text_encoders[0].parameters.return_value = te1_params

        params = mode.get_trainable_params(mock_trainer)

        assert len(params) == 2
        assert params[0] is denoiser_params[0]
        assert params[1] is te1_params[0]

    def test_set_eval(self, mode, mock_trainer):
        """set_eval calls .eval() on trained models."""
        mock_trainer._train_denoiser = True
        mode._te_train_flags = [True, False]

        mode.set_eval(mock_trainer)

        mock_trainer.denoiser.eval.assert_called_once()
        mock_trainer.text_encoders[0].eval.assert_called()
        mock_trainer.text_encoders[1].eval.assert_not_called()

    def test_set_train(self, mode, mock_trainer):
        """set_train calls .train() on trained models."""
        mock_trainer._train_denoiser = True
        mode._te_train_flags = [True, False]

        mode.set_train(mock_trainer)

        mock_trainer.denoiser.train.assert_called()
        mock_trainer.text_encoders[0].train.assert_called()
        mock_trainer.text_encoders[1].train.assert_not_called()


# =============================================================================
# Test: save_checkpoint — strategy delegation
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestSaveCheckpoint:
    """Test save_checkpoint delegates to strategy."""

    def test_delegates_to_strategy_save_model_checkpoint(self, mode, mock_trainer, tmp_path):
        """Normal checkpoint delegates to strategies.save_model_checkpoint."""
        mode._te_train_flags = [True, True]
        mock_trainer.cfg.output.saving.output_dir = str(tmp_path)

        mode.save_checkpoint(
            mock_trainer,
            ckpt_name="test-epoch1.safetensors",
            step=100,
            epoch=1,
            metadata={"ss_key": "value"},
        )

        # Verify strategy was called
        mock_trainer.strategies.save_model_checkpoint.assert_called_once_with(
            trainer=mock_trainer,
            ckpt_name="test-epoch1.safetensors",
            step=100,
            epoch=1,
            metadata={"ss_key": "value"},
            save_dtype=torch.float16,
            force_sync_upload=False,
        )

    def test_no_sdxl_imports_in_mode(self, mode, mock_trainer, tmp_path):
        """Mode save_checkpoint does not import any SDXL conversion modules."""
        import library.training.modes.finetune_mode as ft_module

        with open(ft_module.__file__) as f:
            source = f.read()
        # No direct imports from SDXL model/conversion packages
        assert "from library.models.sdxl" not in source
        assert "import library.models.sdxl" not in source
        # No getattr for logit_scale or ckpt_info (strategy-specific)
        assert "logit_scale" not in source
        assert "ckpt_info" not in source

    def test_edm2_side_artifact_uses_save_weights(self, mode, mock_trainer, tmp_path):
        """EDM2 side-save delegates to target_model.save_weights."""
        mode._te_train_flags = [True, True]
        mock_trainer.cfg.output.saving.output_dir = str(tmp_path)

        target_model = MagicMock()

        mode.save_checkpoint(
            mock_trainer,
            ckpt_name="test_edm2_loss_weights.safetensors",
            step=100,
            epoch=1,
            metadata={"ss_key": "value"},
            target_model=target_model,
            dtype_override=torch.float32,
        )

        # Strategy save should NOT be called for EDM2 side-saves
        mock_trainer.strategies.save_model_checkpoint.assert_not_called()
        # Target model save_weights should be called
        target_model.save_weights.assert_called_once()

    def test_edm2_deferred_features_still_fail_fast(self, mode, mock_trainer):
        """Block LR fails fast even in refactored mode."""
        mock_trainer.cfg.optimizer.optimizer_args = ["block_lr=0.01"]
        mode._te_train_flags = [False, False]

        with pytest.raises(NotImplementedError, match="Block-level learning rates"):
            mode.build_optimizer_params(mock_trainer)


# =============================================================================
# Test: Trainer.remove_checkpoint hardening
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestRemoveCheckpoint:
    """Test Trainer.remove_checkpoint file/directory handling."""

    def test_removes_file(self, tmp_path):
        """Removes a checkpoint file."""
        from library.training.runners.trainer import Trainer

        ckpt_file = tmp_path / "test.safetensors"
        ckpt_file.write_text("dummy")

        trainer = MagicMock(spec=Trainer)
        trainer.cfg = MagicMock()
        trainer.cfg.output.saving.output_dir = str(tmp_path)
        type(trainer).accelerator = PropertyMock(return_value=MagicMock())

        Trainer.remove_checkpoint(trainer, "test.safetensors")

        assert not ckpt_file.exists()

    def test_removes_directory(self, tmp_path):
        """Removes a diffusers-format checkpoint directory."""
        from library.training.runners.trainer import Trainer

        ckpt_dir = tmp_path / "test-diffusers"
        ckpt_dir.mkdir()
        (ckpt_dir / "model_index.json").write_text("{}")

        trainer = MagicMock(spec=Trainer)
        trainer.cfg = MagicMock()
        trainer.cfg.output.saving.output_dir = str(tmp_path)
        type(trainer).accelerator = PropertyMock(return_value=MagicMock())

        Trainer.remove_checkpoint(trainer, "test-diffusers")

        assert not ckpt_dir.exists()

    def test_no_op_when_missing(self, tmp_path):
        """No error raised when path doesn't exist."""
        from library.training.runners.trainer import Trainer

        trainer = MagicMock(spec=Trainer)
        trainer.cfg = MagicMock()
        trainer.cfg.output.saving.output_dir = str(tmp_path)
        type(trainer).accelerator = PropertyMock(return_value=MagicMock())

        Trainer.remove_checkpoint(trainer, "nonexistent.safetensors")
