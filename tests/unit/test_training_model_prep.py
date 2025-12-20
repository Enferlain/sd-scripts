import unittest
from unittest.mock import MagicMock, patch
import pytest
import torch
from library.training import model_prep

class TestModelPrep(unittest.TestCase):
    def test_set_padding_mode_for_vae_conv2d_modules(self):
        # Create a mock VAE with Conv2d layers
        class MockVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=(0, 0)) # Zero padding
                self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=(1, 2))

        vae = MockVAE()
        
        # Verify initial states
        self.assertEqual(vae.conv1.padding_mode, 'zeros')
        self.assertEqual(vae.conv2.padding_mode, 'zeros')

        # Apply padding mode change
        model_prep.set_padding_mode_for_vae_conv2d_modules(vae, 'reflect')

        # Verify changes
        self.assertEqual(vae.conv1.padding_mode, 'reflect')
        self.assertEqual(vae.conv2.padding_mode, 'zeros') # Should remain zeros as padding was 0
        self.assertEqual(vae.conv3.padding_mode, 'reflect')

    @patch("library.training.model_prep.logger")
    def test_replace_unet_modules_mem_eff(self, mock_logger):
        # Mock UNet
        mock_unet = MagicMock()
        
        # Test memory efficient attention
        model_prep.replace_unet_modules(mock_unet, mem_eff_attn=True, xformers=False, sdpa=False)
        
        mock_unet.set_use_memory_efficient_attention.assert_called_with(False, True)
        mock_logger.info.assert_called_with("Enable memory efficient attention for U-Net")

    @patch("library.training.model_prep.logger")
    def test_replace_unet_modules_sdpa(self, mock_logger):
        # Mock UNet
        mock_unet = MagicMock()
        
        # Test SDPA
        model_prep.replace_unet_modules(mock_unet, mem_eff_attn=False, xformers=False, sdpa=True)
        
        mock_unet.set_use_sdpa.assert_called_with(True)
        # Check logs if needed, though exact string matching might be brittle
        mock_logger.info.assert_called_with("Enable SDPA for U-Net")

    @patch("library.training.model_prep.logger")
    def test_replace_unet_modules_xformers(self, mock_logger):
        # Mock UNet
        mock_unet = MagicMock()
        
        # Test Xformers (simulating import success)
        with patch.dict("sys.modules", {"xformers.ops": MagicMock()}):
            model_prep.replace_unet_modules(mock_unet, mem_eff_attn=False, xformers=True, sdpa=False)
        
        mock_unet.set_use_memory_efficient_attention.assert_called_with(True, False)
        mock_logger.info.assert_called_with("Enable xformers for U-Net")

    def test_replace_unet_modules_xformers_import_error(self):
         # Mock UNet
        mock_unet = MagicMock()
        
        # Test Xformers import failure
        with patch.dict("sys.modules", {"xformers.ops": None}): # Ensure it fails
            # We need to make sure import inside the function fails
            with patch("builtins.__import__", side_effect=ImportError("No xformers")):
                 with self.assertRaises(ImportError):
                    model_prep.replace_unet_modules(mock_unet, mem_eff_attn=False, xformers=True, sdpa=False)

    @patch("library.training.model_prep.logger")
    def test_patch_accelerator_for_fp16_training(self, mock_logger):
        mock_accelerator = MagicMock()
        from accelerate import DistributedType
        mock_accelerator.distributed_type = DistributedType.NO
        
        # Mock scaler
        mock_scaler = MagicMock()
        # Mock _unscale_grads_
        original_unscale = MagicMock()
        mock_scaler._unscale_grads_ = original_unscale
        mock_accelerator.scaler = mock_scaler

        model_prep.patch_accelerator_for_fp16_training(mock_accelerator)
        
        # Verify that _unscale_grads_ was replaced
        self.assertNotEqual(mock_scaler._unscale_grads_, original_unscale)
        
        # Execute the new function
        mock_optimizer = MagicMock()
        mock_scaler._unscale_grads_(mock_optimizer, inv_scale=1.0, found_inf=False, allow_fp16=False)
        
        # Verify it called original with True
        original_unscale.assert_called_with(mock_optimizer, 1.0, False, True)

    @patch("library.training.model_prep.logger")
    def test_patch_accelerator_deepspeed_skip(self, mock_logger):
        mock_accelerator = MagicMock()
        from accelerate import DistributedType
        mock_accelerator.distributed_type = DistributedType.DEEPSPEED
        
        mock_scaler = MagicMock()
        original_unscale = MagicMock()
        mock_scaler._unscale_grads_ = original_unscale
        mock_accelerator.scaler = mock_scaler
        
        model_prep.patch_accelerator_for_fp16_training(mock_accelerator)
        
        # Should not change anything
        self.assertEqual(mock_scaler._unscale_grads_, original_unscale)
