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
    cfg.optimizer.learning_rates.unet = None  # Default: use base LR
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
def mock_unet():
    """Create a mock UNet."""
    unet = MagicMock(spec=nn.Module)
    unet.parameters = MagicMock(return_value=[nn.Parameter(torch.randn(4, 4))])
    return unet


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
    strategies.is_train_unet = MagicMock(return_value=True)
    strategies.is_train_text_encoder = MagicMock(return_value=True)
    strategies.get_text_encoders_train_flags = MagicMock(return_value=[True, True])
    strategies.post_process_trainable = MagicMock()
    strategies.prepare_unet_with_accelerator = MagicMock(side_effect=lambda cfg, acc, u: u)
    strategies.save_model_checkpoint = MagicMock()
    return strategies


@pytest.fixture
def mock_trainer(mock_cfg, mock_accelerator, mock_unet, mock_text_encoders, mock_strategies):
    """Create a mock Trainer for FineTuneMode testing."""
    trainer = MagicMock()
    trainer.cfg = mock_cfg
    type(trainer).accelerator = PropertyMock(return_value=mock_accelerator)
    type(trainer).is_main_process = PropertyMock(return_value=True)

    trainer.unet = mock_unet
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

    trainer._train_unet = True
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
    """Test prepare_trainables uses strategy methods."""

    def test_uses_strategy_is_train_unet(self, mode, mock_trainer):
        """Train flag is set via strategy.is_train_unet(cfg)."""
        mode.prepare_trainables(mock_trainer)

        mock_trainer.strategies.is_train_unet.assert_called_once_with(mock_trainer.cfg)
        assert mock_trainer._train_unet is True

    def test_uses_strategy_is_train_text_encoder(self, mode, mock_trainer):
        """TE training flag is set via strategy."""
        mode.prepare_trainables(mock_trainer)

        mock_trainer.strategies.is_train_text_encoder.assert_called_once_with(mock_trainer.cfg)

    def test_delegates_post_process_to_strategy(self, mode, mock_trainer):
        """post_process_trainable is called on strategy after unfreezing."""
        mode.prepare_trainables(mock_trainer)

        mock_trainer.strategies.post_process_trainable.assert_called_once()

    def test_uses_strategy_te_flags_when_no_lr_override(self, mode, mock_trainer):
        """When text_encoders LR is None, uses strategy-provided flags."""
        mock_trainer.cfg.optimizer.learning_rates.text_encoders = None
        mock_trainer.strategies.get_text_encoders_train_flags.return_value = [True, False]

        mode.prepare_trainables(mock_trainer)

        mock_trainer.strategies.get_text_encoders_train_flags.assert_called_once()
        assert mode._te_train_flags == [True, False]

    def test_per_te_lr_overrides_strategy_flags(self, mode, mock_trainer):
        """Per-TE LR list overrides strategy flags."""
        mock_trainer.cfg.optimizer.learning_rates.text_encoders = [1e-5, 0.0]

        mode.prepare_trainables(mock_trainer)

        # Strategy flags not used when explicit LR is given
        mock_trainer.strategies.get_text_encoders_train_flags.assert_not_called()
        assert mode._te_train_flags == [True, False]

    def test_zero_te_lr_disables_all_tes(self, mode, mock_trainer):
        """Zero text_encoders LR disables both TEs."""
        mock_trainer.cfg.optimizer.learning_rates.text_encoders = 0.0

        mode.prepare_trainables(mock_trainer)

        assert mode._te_train_flags == [False, False]
        assert mock_trainer._train_text_encoder is False

    def test_sets_primary_trainable(self, mode, mock_trainer):
        """Primary trainable is set to UNet."""
        mode.prepare_trainables(mock_trainer)

        assert mock_trainer._primary_trainable is mock_trainer.unet


# =============================================================================
# Test: configure_trainable_precision
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestConfigureTrainablePrecision:
    """Test configure_trainable_precision hook."""

    def test_full_fp16_casts_unet_and_tes(self, mode, mock_trainer):
        """Full fp16 casts UNet and trained TEs to weight_dtype."""
        mock_trainer.cfg.performance.precision.full_fp16 = True
        mode._te_train_flags = [True, True]

        mode.configure_trainable_precision(mock_trainer)

        mock_trainer.unet.to.assert_called_with(torch.float16)
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

    def test_returns_6_tuple(self, mode, mock_trainer):
        """Returns the standard 6-tuple of optimizer components."""
        mode._te_train_flags = [True, False]
        mock_trainer._train_unet = True

        with patch("library.training.modes.finetune_mode.get_optimizer") as mock_get_opt, \
             patch("library.training.modes.finetune_mode.get_optimizer_train_eval_fn") as mock_get_fn:
            mock_get_opt.return_value = ("AdamW", {}, MagicMock())
            mock_get_fn.return_value = (lambda: None, lambda: None)

            result = mode.build_optimizer_params(mock_trainer)

        assert len(result) == 6
        name, args, optimizer, train_fn, eval_fn, lr_descs = result
        assert name == "AdamW"
        assert any("unet" in d for d in lr_descs)

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
        """_grad_sync_handle and _primary_trainable are set to UNet."""
        mode._te_train_flags = [False, False]
        mock_trainer._train_unet = True

        mode.prepare_with_accelerator(mock_trainer)

        assert mock_trainer._grad_sync_handle is mock_trainer.unet
        assert mock_trainer._primary_trainable is mock_trainer.unet

    def test_deepspeed_uses_dynamic_kwargs(self, mode, mock_trainer):
        """DeepSpeed path builds dynamic kwargs, no fixed TE count."""
        mock_trainer.cfg.performance.deepspeed = SimpleNamespace(deepspeed=True)
        mode._te_train_flags = [True, False]
        mock_trainer._train_unet = True

        with patch("library.training.modes.finetune_mode.deepspeed_utils") as mock_ds:
            mock_ds_model = MagicMock()
            mock_ds.prepare_deepspeed_model.return_value = mock_ds_model
            mock_trainer.accelerator.prepare.return_value = (mock_ds_model, MagicMock(), MagicMock())

            mode.prepare_with_accelerator(mock_trainer)

            # Verify dynamic kwargs — only text_encoder1 is passed (flag[1] is False)
            mock_ds.prepare_deepspeed_model.assert_called_once()
            call_kwargs = mock_ds.prepare_deepspeed_model.call_args
            assert "unet" in call_kwargs.kwargs
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
        mock_trainer._train_unet = True

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

    def test_returns_callable(self, mode, mock_trainer):
        """Returns a callable get_steps_from_state function."""
        mode._te_train_flags = [False, False]

        result = mode.register_state_hooks(mock_trainer)

        assert callable(result)
        assert result() is None  # No state loaded yet


# =============================================================================
# Test: on_epoch_start / on_step_start / on_step_end
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestCallbacks:
    """Test per-epoch / per-step callbacks."""

    def test_on_epoch_start_calls_train(self, mode, mock_trainer):
        """on_epoch_start sets training models to train mode."""
        mock_trainer._train_unet = True
        mode._te_train_flags = [True, False]

        mode.on_epoch_start(mock_trainer)

        mock_trainer.unet.train.assert_called_once()
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
        """Returns UNet + trained TE params."""
        mock_trainer._train_unet = True
        mode._te_train_flags = [True, False]

        unet_params = [nn.Parameter(torch.randn(4, 4))]
        te1_params = [nn.Parameter(torch.randn(2, 2))]
        mock_trainer.unet.parameters.return_value = unet_params
        mock_trainer.text_encoders[0].parameters.return_value = te1_params

        params = mode.get_trainable_params(mock_trainer)

        assert len(params) == 2
        assert params[0] is unet_params[0]
        assert params[1] is te1_params[0]

    def test_set_eval(self, mode, mock_trainer):
        """set_eval calls .eval() on trained models."""
        mock_trainer._train_unet = True
        mode._te_train_flags = [True, False]

        mode.set_eval(mock_trainer)

        mock_trainer.unet.eval.assert_called_once()
        mock_trainer.text_encoders[0].eval.assert_called()
        mock_trainer.text_encoders[1].eval.assert_not_called()

    def test_set_train(self, mode, mock_trainer):
        """set_train calls .train() on trained models."""
        mock_trainer._train_unet = True
        mode._te_train_flags = [True, False]

        mode.set_train(mock_trainer)

        mock_trainer.unet.train.assert_called()
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
