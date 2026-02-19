"""
Integration tests for the training loop (library/training/phases/training_loop.py).

Exercises the real ``run_training_loop`` function with a mock trainer to verify:
- global_step advances correctly
- step-based checkpoint triggers fire at the right steps
- epoch-based checkpoint triggers fire at the right epoch boundaries
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import torch

from library.training.phases.training_loop import run_training_loop
from library.training.checkpointing import get_step_ckpt_name, get_epoch_ckpt_name


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_batch():
    """Create a minimal mock batch dict that process_batch can consume."""
    return {
        "latents": torch.randn(1, 4, 8, 8),
        "captions": ["test"],
        "input_ids": torch.randint(0, 100, (1, 77)),
    }


def _make_mock_trainer(
    *,
    num_epochs: int = 2,
    batches_per_epoch: int = 5,
    max_train_steps: int | None = None,
    save_every_n_steps: int | None = None,
    save_every_n_epochs: int | None = None,
    save_model_as: str = "safetensors",
    save_state: bool = False,
    gradient_accumulation_steps: int = 1,
):
    """Build a mock trainer with enough fidelity for run_training_loop."""
    trainer = MagicMock()

    # ---- Config ----
    cfg = MagicMock()

    # data.caption
    cfg.data.caption.shuffle_caption = False
    cfg.data.caption.keep_tokens = 0
    cfg.data.caption.caption_dropout_rate = 0.0
    cfg.data.caption.caption_tag_dropout_rate = 0.0
    cfg.data.caption.enable_wildcard = False
    cfg.data.caption.caption_separator = ","
    cfg.data.caption.secondary_separator = None
    cfg.data.caption.keep_tokens_separator = None
    cfg.data.caption.token_warmup_min = 1
    cfg.data.caption.token_warmup_step = 0

    # data.caching
    cfg.data.caching.cache_tokens_per_epoch = False

    # data.preprocessing
    cfg.data.preprocessing.flip_aug = False

    # data.loader
    cfg.data.loader.prefetch_factor = 2
    cfg.data.loader.pin_memory = True
    cfg.data.loader.persistent_workers = False

    # training
    cfg.training.seed = 42
    cfg.training.train_batch_size = 1
    cfg.training.max_train_epochs = num_epochs
    total_steps = max_train_steps if max_train_steps else num_epochs * batches_per_epoch
    cfg.training.max_train_steps = total_steps
    cfg.training.gradient_accumulation_steps = gradient_accumulation_steps

    # performance
    cfg.performance.precision.mixed_precision = "no"
    cfg.performance.precision.full_fp16 = False
    cfg.performance.precision.full_bf16 = False
    cfg.performance.deepspeed = False

    # optimizer
    cfg.optimizer.max_grad_norm = 0.0

    # peft
    cfg.peft.scale_weight_norms = False

    # output.saving
    cfg.output.saving.output_dir = "/tmp/test_output"
    cfg.output.saving.output_name = "test_model"
    cfg.output.saving.save_model_as = save_model_as
    cfg.output.saving.save_every_n_steps = save_every_n_steps
    cfg.output.saving.save_every_n_epochs = save_every_n_epochs
    cfg.output.saving.save_state = save_state
    cfg.output.saving.save_last_n_steps = None
    cfg.output.saving.save_last_n_epochs = None
    cfg.output.saving.save_n_epoch_ratio = None

    # output.sampling
    cfg.output.sampling.sample_every_n_steps = None
    cfg.output.sampling.sample_every_n_epochs = None

    # output.logging
    cfg.output.logging.log_timestep_distribution_every_n_steps = None

    # output.huggingface
    cfg.output.huggingface = None

    # loss
    cfg.loss.prior_loss_weight = 1.0
    cfg.loss.edm2.edm2_loss_weighting = False

    # validation
    cfg.validation.validate_every_n_steps = None
    cfg.validation.validation_seed = 42

    trainer.cfg = cfg

    # ---- Accelerator ----
    accelerator = MagicMock()
    accelerator.device = torch.device("cpu")
    accelerator.process_index = 0
    accelerator.num_processes = 1
    accelerator.is_main_process = True
    accelerator.is_local_main_process = True
    accelerator.sync_gradients = True
    accelerator.gradient_accumulation_steps = gradient_accumulation_steps
    accelerator.trackers = []
    accelerator.wait_for_everyone = MagicMock()
    accelerator.unwrap_model = MagicMock(side_effect=lambda x: x)
    accelerator.backward = MagicMock()
    accelerator.clip_grad_norm_ = MagicMock()
    accelerator.print = MagicMock()
    accelerator.log = MagicMock()

    trainer._accelerator = accelerator
    type(trainer).accelerator = PropertyMock(return_value=accelerator)
    type(trainer).is_main_process = PropertyMock(return_value=True)

    # ---- Strategies ----
    strategies = MagicMock()
    strategies.process_batch = MagicMock(return_value=(torch.tensor(0.5), torch.tensor(0.5), None, torch.tensor([500])))
    strategies.on_step_start = MagicMock()
    strategies.all_reduce_trainable = MagicMock()
    strategies.sample_images = MagicMock()
    strategies.calculate_val_loss = MagicMock(return_value=(None, None))
    strategies.la_sampler = None
    strategies.live_plotter_process = None
    trainer.strategies = strategies

    # ---- Models ----
    trainer.vae = MagicMock()
    trainer.unet = MagicMock()
    trainer.text_encoders = [MagicMock()]
    trainer._text_encoder = trainer.text_encoders
    trainer.tokenizers = [MagicMock()]

    # ---- Adapter ----
    adapter = MagicMock()
    adapter.on_epoch_start = MagicMock()
    adapter.get_trainable_params = MagicMock(return_value=[torch.nn.Parameter(torch.randn(10))])
    adapter.train = MagicMock()
    adapter.eval = MagicMock()
    trainer.adapter = adapter
    trainer._grad_sync_handle = adapter

    # trainable_model property (returns adapter for PEFT)
    type(trainer).trainable_model = PropertyMock(return_value=adapter)

    # Training mode (TrainingMode protocol)
    trainer.mode = MagicMock()
    trainer.mode.on_epoch_start = MagicMock()
    trainer.mode.on_step_start = MagicMock()
    trainer.mode.on_step_end = MagicMock(return_value={})
    trainer.mode.get_trainable_params = MagicMock(return_value=[torch.nn.Parameter(torch.randn(10))])
    trainer.mode.set_eval = MagicMock()
    trainer.mode.set_train = MagicMock()
    trainer.mode.save_checkpoint = MagicMock()

    # ---- Dtypes ----
    trainer.weight_dtype = torch.float32
    trainer.vae_dtype = torch.float32

    # ---- Optimizer / Scheduler ----
    trainer.optimizer = MagicMock()
    trainer.lr_scheduler = MagicMock()
    trainer.lr_descriptions = ["unet"]
    trainer.optimizer_train_fn = MagicMock()
    trainer.optimizer_eval_fn = MagicMock()

    # ---- Training state ----
    trainer.global_step = 0
    trainer.epoch_to_start = 0
    trainer.num_train_epochs = num_epochs
    trainer.max_train_steps = total_steps
    trainer.num_batches_per_epoch = batches_per_epoch
    trainer._initial_step = 0
    trainer._accumulation_counter = 0

    # ---- Epoch/step state ----
    trainer._current_epoch_state = SimpleNamespace(value=0)
    trainer._current_step_state = SimpleNamespace(value=0)

    # ---- Loss tracking ----
    trainer._loss_recorder = MagicMock()
    trainer._loss_recorder.add = MagicMock()
    trainer._loss_recorder.average = 0.5
    trainer._val_loss_recorder = None
    trainer._loss_scaled_recorder = None
    trainer._current_global_step_loss = 0.0
    trainer._current_global_step_loss_scaled = None
    trainer._current_val_loss = None
    trainer._average_val_loss = None

    # ---- Progress bar ----
    trainer._progress_bar = MagicMock()
    trainer._progress_bar.update = MagicMock()
    trainer._progress_bar.set_postfix = MagicMock()
    trainer._progress_bar.unpause = MagicMock()

    # ---- Misc ----
    trainer._is_tracking = False
    trainer._metadata = {}
    trainer._cache_dir = "/tmp/cache"
    trainer._n_workers = 0
    trainer._val_dataloader = None
    trainer._cyclic_val_dataloader = None
    trainer._edm2_model = None
    trainer._edm2_optimizer = None
    trainer._edm2_lr_scheduler = None
    trainer._timestep_counts = None
    trainer._plotter_settings = None
    trainer._dynamic_timestep_schedule = []
    trainer._current_min_timestep = 0
    trainer._current_max_timestep = 1000
    trainer.noise_scheduler = MagicMock()

    # ---- train_manifest ----
    trainer.train_manifest = MagicMock()
    trainer.train_manifest.entries = {}

    # ---- Checkpoint methods ----
    trainer.save_checkpoint = MagicMock()
    trainer.remove_checkpoint = MagicMock()

    return trainer


def _run_loop_with_mock_dataloader(trainer, batches_per_epoch: int):
    """Patch data pipeline functions and run the training loop."""
    mock_batches = [_make_mock_batch() for _ in range(batches_per_epoch)]

    with patch("library.training.phases.training_loop.prepare_epoch") as mock_prepare:
        mock_prepare.return_value = MagicMock()
        with patch("library.training.phases.training_loop.create_training_dataloader") as mock_dl:
            mock_dl.return_value = mock_batches
            run_training_loop(trainer)


# =============================================================================
# Test 1: Step Advancement
# =============================================================================


class TestTrainingLoopStepAdvancement:
    """Verify global_step advances correctly through epochs."""

    def test_global_step_advances_over_epochs(self):
        """global_step should equal total batches processed."""
        num_epochs = 3
        batches_per_epoch = 4
        trainer = _make_mock_trainer(
            num_epochs=num_epochs,
            batches_per_epoch=batches_per_epoch,
        )

        _run_loop_with_mock_dataloader(trainer, batches_per_epoch)

        assert trainer.global_step == num_epochs * batches_per_epoch

    def test_process_batch_call_count(self):
        """strategies.process_batch should be called once per step."""
        num_epochs = 2
        batches_per_epoch = 5
        trainer = _make_mock_trainer(
            num_epochs=num_epochs,
            batches_per_epoch=batches_per_epoch,
        )

        _run_loop_with_mock_dataloader(trainer, batches_per_epoch)

        assert trainer.strategies.process_batch.call_count == num_epochs * batches_per_epoch

    def test_progress_bar_updates(self):
        """Progress bar should be updated on each step."""
        num_epochs = 2
        batches_per_epoch = 3
        trainer = _make_mock_trainer(
            num_epochs=num_epochs,
            batches_per_epoch=batches_per_epoch,
        )

        _run_loop_with_mock_dataloader(trainer, batches_per_epoch)

        assert trainer._progress_bar.update.call_count == num_epochs * batches_per_epoch

    def test_stops_at_max_train_steps(self):
        """Training stops cleanly when global_step >= max_train_steps.

        With 2 epochs, 5 batches/epoch, and max_train_steps=3:
          epoch 0 → processes 3 batches then breaks (global_step=3)
          epoch 1 → skipped by outer-loop guard
        """
        trainer = _make_mock_trainer(
            num_epochs=2,
            batches_per_epoch=5,
            max_train_steps=3,
        )

        _run_loop_with_mock_dataloader(trainer, 5)

        assert trainer.global_step == 3


# =============================================================================
# Test 2: Checkpoint Triggers on Step
# =============================================================================


class TestCheckpointTriggersOnStep:
    """Verify checkpoint saving is triggered at correct step intervals."""

    def test_save_at_every_n_steps(self):
        """save_checkpoint should be called at multiples of save_every_n_steps."""
        trainer = _make_mock_trainer(
            num_epochs=1,
            batches_per_epoch=10,
            save_every_n_steps=3,
        )

        _run_loop_with_mock_dataloader(trainer, 10)

        # Should fire at steps 3, 6, 9
        save_calls = trainer.save_checkpoint.call_args_list
        saved_steps = [c.args[2] for c in save_calls]  # arg[2] is step
        assert saved_steps == [3, 6, 9]

    def test_step_checkpoint_naming(self):
        """Checkpoint names should use step-based format."""
        trainer = _make_mock_trainer(
            num_epochs=1,
            batches_per_epoch=5,
            save_every_n_steps=5,
        )

        _run_loop_with_mock_dataloader(trainer, 5)

        save_calls = trainer.save_checkpoint.call_args_list
        assert len(save_calls) == 1
        ckpt_name = save_calls[0].args[0]  # arg[0] is ckpt_name
        expected = get_step_ckpt_name(trainer.cfg.output.saving, ".safetensors", 5)
        assert ckpt_name == expected

    def test_no_save_when_not_configured(self):
        """No step checkpoints when save_every_n_steps is None."""
        trainer = _make_mock_trainer(
            num_epochs=1,
            batches_per_epoch=10,
            save_every_n_steps=None,
            save_every_n_epochs=None,
        )

        _run_loop_with_mock_dataloader(trainer, 10)

        trainer.save_checkpoint.assert_not_called()


# =============================================================================
# Test 3: Checkpoint Triggers on Epoch End
# =============================================================================


class TestCheckpointTriggersOnEpochEnd:
    """Verify checkpoint saving at epoch boundaries."""

    def test_save_every_epoch(self):
        """save_checkpoint fires at the end of every epoch (except last)."""
        trainer = _make_mock_trainer(
            num_epochs=4,
            batches_per_epoch=3,
            save_every_n_epochs=1,
        )

        _run_loop_with_mock_dataloader(trainer, 3)

        # Epochs 1, 2, 3 (not 4, because epoch 4 == num_train_epochs)
        save_calls = trainer.save_checkpoint.call_args_list
        saved_epochs = [c.args[3] for c in save_calls]  # arg[3] is epoch
        assert saved_epochs == [1, 2, 3]

    def test_save_every_2_epochs(self):
        """save_checkpoint fires at every 2nd epoch end (except last)."""
        trainer = _make_mock_trainer(
            num_epochs=6,
            batches_per_epoch=2,
            save_every_n_epochs=2,
        )

        _run_loop_with_mock_dataloader(trainer, 2)

        save_calls = trainer.save_checkpoint.call_args_list
        saved_epochs = [c.args[3] for c in save_calls]
        # Epochs 2, 4 (not 6 because it's the last)
        assert saved_epochs == [2, 4]

    def test_epoch_checkpoint_naming(self):
        """Checkpoint names should use epoch-based format."""
        trainer = _make_mock_trainer(
            num_epochs=3,
            batches_per_epoch=2,
            save_every_n_epochs=1,
        )

        _run_loop_with_mock_dataloader(trainer, 2)

        save_calls = trainer.save_checkpoint.call_args_list
        first_ckpt_name = save_calls[0].args[0]
        expected = get_epoch_ckpt_name(trainer.cfg.output.saving, ".safetensors", 1)
        assert first_ckpt_name == expected
