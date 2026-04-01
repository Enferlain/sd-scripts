import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
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
        self.trainer._finalize_training.assert_not_called()

    def test_run_startup_eval_actions_validation_only_skips_sampling(self):
        """Startup validation should not force startup sampling."""
        self.trainer._accelerator = MagicMock()
        self.trainer._accelerator.device = "cpu"
        self.trainer._validation_scheduler = MagicMock()
        self.trainer._validation_scheduler.should_run.return_value = True
        self.trainer.optimizer_eval_fn = MagicMock()
        self.trainer.optimizer_train_fn = MagicMock()
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
        self.trainer.optimizer_eval_fn = MagicMock()
        self.trainer.optimizer_train_fn = MagicMock()
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
