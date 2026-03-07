import unittest
from unittest.mock import MagicMock
from library.training.runners.trainer import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.cfg = MagicMock()
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
        model (e.g. _edm2_model).  The mode must receive this as
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
        self.trainer._log_training_info = MagicMock()
        self.trainer._maybe_sample_at_start = MagicMock()
        self.trainer.run_training_loop = MagicMock(side_effect=RuntimeError("training failed"))
        self.trainer._finalize_training = MagicMock()

        def assign_monitor():
            self.trainer._resource_monitor = mock_monitor

        self.trainer.setup.side_effect = assign_monitor

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.trainer.train()

        mock_monitor.end_session.assert_called_once()
        self.trainer._finalize_training.assert_not_called()
