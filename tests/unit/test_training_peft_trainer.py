import unittest
from unittest.mock import MagicMock
from library.training.runners.peft_trainer import PeftTrainer


class TestPeftTrainer(unittest.TestCase):
    def setUp(self):
        self.cfg = MagicMock()
        self.strategies = MagicMock()
        self.mode = MagicMock()
        self.trainer = PeftTrainer(self.cfg, self.strategies, self.mode)

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
