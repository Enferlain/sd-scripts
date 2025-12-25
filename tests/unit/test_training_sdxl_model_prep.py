
import unittest
from unittest.mock import MagicMock, patch, ANY
import torch
import pytest
import os

# Import the module under test
import library.training.sdxl_model_prep as sdxl_model_prep
from library.config.dataclasses.sdxl_finetune import SDXLFineTuneConfig
from library.config.dataclasses.deepspeed import DeepSpeedConfig

@pytest.mark.unit
class TestSDXLModelPrep(unittest.TestCase):
    def setUp(self):
        # Create a mock config
        self.cfg = MagicMock(spec=SDXLFineTuneConfig)
        self.cfg.training = MagicMock()
        self.cfg.sd_models = MagicMock()
        self.cfg.performance = MagicMock()
        self.cfg.sdxl = MagicMock()
        
        # Default config values
        self.cfg.sd_models.pretrained_model_name_or_path = "model/path"
        self.cfg.sd_models.vae = None
        self.cfg.sd_models.vae_conv2d_padding_mode = None
        self.cfg.performance.memory.lowram = False
        self.cfg.performance.caching.disable_mmap_load_safetensors = False
        self.cfg.training.mixed_precision = "fp16"

    @patch("library.training.sdxl_model_prep.match_mixed_precision")
    @patch("library.training.sdxl_model_prep._load_target_model")
    @patch("library.training.sdxl_model_prep.clean_memory_on_device")
    def test_load_target_model_main_process(self, mock_clean, mock_load_internal, mock_match_mp):
        """Test load_target_model on the main process (index 0)."""
        # Mock accelerator
        accelerator = MagicMock()
        accelerator.state.num_processes = 2
        accelerator.state.local_process_index = 0
        accelerator.device = "cuda:0"
        
        # Mock return values
        mock_match_mp.return_value = torch.float16
        
        # Mock return from _load_target_model
        # (load_stable_diffusion_format, te1, te2, vae, unet, logit_scale, ckpt_info)
        te1 = MagicMock()
        te2 = MagicMock()
        vae = MagicMock()
        unet = MagicMock()
        return_tuple = (True, te1, te2, vae, unet, 1.0, None)
        mock_load_internal.return_value = return_tuple
        
        # Execute
        result = sdxl_model_prep.load_target_model(self.cfg, accelerator, "v1", torch.float16)
        
        # Assertions
        mock_load_internal.assert_called_once_with(
            self.cfg,
            "model/path",
            None,
            "v1",
            torch.float16,
            "cpu", # Default device passed when lowram is False
            torch.float16,
            False
        )
        # Called twice because num_processes=2 and it's called inside the loop
        self.assertEqual(accelerator.wait_for_everyone.call_count, 2)
        mock_clean.assert_called_once_with("cuda:0")
        
        # Check return
        self.assertEqual(result, return_tuple)

    @patch("library.training.sdxl_model_prep.match_mixed_precision")
    @patch("library.training.sdxl_model_prep._load_target_model")
    def test_load_target_model_secondary_process(self, mock_load_internal, mock_match_mp):
        """Test load_target_model on a secondary process (index 1)."""
        accelerator = MagicMock()
        accelerator.state.num_processes = 2
        accelerator.state.local_process_index = 1
        
        # Set return value to prevent unpacking error
        te1 = MagicMock()
        mock_load_internal.return_value = (True, te1, MagicMock(), MagicMock(), MagicMock(), 1.0, None)
        
        result = sdxl_model_prep.load_target_model(self.cfg, accelerator, "v1", torch.float16)
        
        # It should call load when pi matches local_process_index (which is 1)
        mock_load_internal.assert_called_once()
        self.assertEqual(accelerator.wait_for_everyone.call_count, 2)


    @patch("library.training.sdxl_model_prep.match_mixed_precision")
    @patch("library.training.sdxl_model_prep._load_target_model")
    @patch("library.training.sdxl_model_prep.clean_memory_on_device")
    def test_load_target_model_iterates_processes(self, mock_clean, mock_load_internal, mock_match_mp):
        """Test that the loop correctly triggers loading when it matches local index."""
        accelerator = MagicMock()
        accelerator.state.num_processes = 2
        accelerator.state.local_process_index = 1 # I am process 1
        accelerator.device = "cuda:1"
        
        te1 = MagicMock()
        mock_load_internal.return_value = (True, te1, MagicMock(), MagicMock(), MagicMock(), 1.0, None)
        
        sdxl_model_prep.load_target_model(self.cfg, accelerator, "v1", torch.float16)
        
        # It should have called wait_for_everyone twice (once for pi=0, once for pi=1)
        self.assertEqual(accelerator.wait_for_everyone.call_count, 2)
        
        # load should be called exactly once
        mock_load_internal.assert_called_once()

    @patch("library.training.sdxl_model_prep.match_mixed_precision")
    @patch("library.training.sdxl_model_prep._load_target_model")
    @patch("library.training.sdxl_model_prep.clean_memory_on_device")
    def test_load_target_model_lowram(self, mock_clean, mock_load_internal, mock_match_mp):
        """Test lowram behavior moving models to device."""
        self.cfg.performance.memory.lowram = True
        accelerator = MagicMock()
        accelerator.state.num_processes = 1
        accelerator.state.local_process_index = 0
        accelerator.device = "cuda:0"
        
        te1 = MagicMock()
        te2 = MagicMock()
        vae = MagicMock()
        unet = MagicMock()
        
        mock_load_internal.return_value = (True, te1, te2, vae, unet, 1.0, None)
        
        sdxl_model_prep.load_target_model(self.cfg, accelerator, "v1", torch.float16)
        
        # Verify passed device was 'cuda:0' because lowram=True
        args, _ = mock_load_internal.call_args
        self.assertEqual(args[5], "cuda:0") # device arg
        
        # Verify .to(device) calls
        te1.to.assert_called_with("cuda:0")
        te2.to.assert_called_with("cuda:0")
        vae.to.assert_called_with("cuda:0")
        unet.to.assert_called_with("cuda:0")

    # =========================================================================
    # _load_target_model Tests
    # =========================================================================

    @patch("os.path.isfile")
    @patch("os.path.islink")
    @patch("library.models.sdxl_model_util.load_models_from_sdxl_checkpoint")
    def test_load_internal_from_checkpoint_file(self, mock_load_ckpt, mock_islink, mock_isfile):
        """Test loading from a single checkpoint file (safetensors/ckpt)."""
        mock_islink.return_value = False
        mock_isfile.return_value = True # It IS a file
        
        # Setup mock return
        mock_load_ckpt.return_value = ("te1", "te2", "vae", "unet", "logit", "info")
        
        result = sdxl_model_prep._load_target_model(
            self.cfg, "my_model.safetensors", None, "v1", torch.float16, "cpu"
        )
        
        # Assertions
        mock_load_ckpt.assert_called_once_with(
            "v1", "my_model.safetensors", "cpu", None, False
        )
        # load_stable_diffusion_format should be True
        self.assertEqual(result[0], True)
        self.assertEqual(result[1:], ("te1", "te2", "vae", "unet", "logit", "info"))

    @patch("os.path.isfile")
    @patch("os.path.islink")
    @patch("library.models.sdxl_model_util.load_models_from_sdxl_checkpoint")
    @patch("library.models.model_util.load_vae")
    def test_load_internal_with_vae(self, mock_load_vae, mock_load_ckpt, mock_islink, mock_isfile):
        """Test loading checkpoint with an external VAE."""
        mock_islink.return_value = False
        mock_isfile.return_value = True
        mock_load_ckpt.return_value = ("te1", "te2", "old_vae", "unet", "logit", "info")
        mock_load_vae.return_value = "new_vae"
        
        result = sdxl_model_prep._load_target_model(
            self.cfg, "my_model.safetensors", "vae_path.pt", "v1", torch.float16, "cpu"
        )
        
        mock_load_vae.assert_called_once_with("vae_path.pt", torch.float16)
        # check that VAE was replaced
        self.assertEqual(result[3], "new_vae")

    @patch("os.path.isfile")
    @patch("library.training.sdxl_model_prep.set_padding_mode_for_vae_conv2d_modules")
    @patch("library.models.sdxl_model_util.load_models_from_sdxl_checkpoint")
    def test_load_internal_vae_padding(self, mock_load_ckpt, mock_set_padding, mock_isfile):
        """Test that VAE padding mode is applied if configured."""
        mock_isfile.return_value = True
        mock_load_ckpt.return_value = ("te1", "te2", "vae", "unet", "logit", "info")
        
        self.cfg.sd_models.vae_conv2d_padding_mode = "reflect"
        
        sdxl_model_prep._load_target_model(
            self.cfg, "my_model.safetensors", None, "v1", torch.float16
        )
        
        mock_set_padding.assert_called_once_with("vae", "reflect")

    @patch("os.path.isfile")
    @patch("diffusers.StableDiffusionXLPipeline")
    @patch("library.models.sdxl_model_util.convert_diffusers_unet_state_dict_to_sdxl")
    @patch("library.models.sdxl_original_unet.SdxlUNet2DConditionModel")
    @patch("library.models.sdxl_model_util._load_state_dict_on_device")
    @patch("library.training.sdxl_model_prep.init_empty_weights")
    def test_load_internal_diffusers_folder(self, mock_init_empty, mock_load_on_device, 
                                          mock_unet_class, mock_convert_sd, mock_pipeline, mock_isfile):
        """Test loading from a Diffusers folder/repo."""
        mock_isfile.return_value = False # Not a file, so Diffusers path
        
        # Mock Pipeline
        pipe = MagicMock()
        pipe.text_encoder = MagicMock()
        pipe.text_encoder.dtype = torch.float32
        pipe.text_encoder_2 = MagicMock()
        pipe.text_encoder_2.dtype = torch.float32
        pipe.vae = "diffusers_vae"
        pipe.unet = MagicMock()
        pipe.unet.state_dict.return_value = {"unet_sd": 1}
        mock_pipeline.from_pretrained.return_value = pipe
        
        # Mock UNet conversion
        mock_convert_sd.return_value = {"converted_sd": 2}
        
        # Mock Target UNet
        target_unet = MagicMock()
        mock_unet_class.return_value = target_unet
        
        result = sdxl_model_prep._load_target_model(
            self.cfg, "user/repo", None, "v1", torch.float16, "cpu"
        )
        
        # Assertions
        mock_pipeline.from_pretrained.assert_called_once()
        args, kwargs = mock_pipeline.from_pretrained.call_args
        self.assertEqual(args[0], "user/repo")
        self.assertEqual(kwargs["variant"], "fp16") # float16 passed in
        self.assertEqual(kwargs["tokenizer"], None)
        
        mock_convert_sd.assert_called_once()
        mock_unet_class.assert_called_once()
        
        # Verify _load_state_dict_on_device called to load weights into original UNet
        mock_load_on_device.assert_called_once_with(
            target_unet, {"converted_sd": 2}, device="cpu", dtype=None
        )
        
        # result: (load_stable_diffusion_format=False, te1, te2, vae, unet, logit_scale=None, ckpt_info=None)
        self.assertEqual(result[0], False)
        self.assertEqual(result[3], "diffusers_vae")
        self.assertEqual(result[4], target_unet)

    @patch("os.path.isfile")
    @patch("diffusers.StableDiffusionXLPipeline")
    @patch("library.models.sdxl_model_util.convert_diffusers_unet_state_dict_to_sdxl")
    @patch("library.models.sdxl_original_unet.SdxlUNet2DConditionModel")
    @patch("library.models.sdxl_model_util._load_state_dict_on_device")
    @patch("library.training.sdxl_model_prep.init_empty_weights")
    def test_load_internal_diffusers_fallback_fp32(self, mock_init, mock_load, mock_unet_cls, 
                                                 mock_conv, mock_pipeline, mock_isfile):
        """Test fallback to fp32/non-variant if variant load fails."""
        mock_isfile.return_value = False
        
        # First call raises EnvironmentError
        mock_pipeline.from_pretrained.side_effect = [
            OSError("fp16 not found"), # First call fails
            MagicMock() # Second call succeeds
        ]
        
        # Run
        sdxl_model_prep._load_target_model(
            self.cfg, "user/repo", None, "v1", torch.float16
        )
        
        self.assertEqual(mock_pipeline.from_pretrained.call_count, 2)
        # Second call should have variant=None
        _, kwargs2 = mock_pipeline.from_pretrained.call_args_list[1]
        self.assertEqual(kwargs2["variant"], None)

    @patch("os.path.isfile")
    @patch("diffusers.StableDiffusionXLPipeline")
    def test_load_internal_diffusers_not_found(self, mock_pipeline, mock_isfile):
        """Test error when model is not found at all."""
        mock_isfile.return_value = False
        mock_pipeline.from_pretrained.side_effect = OSError("Not found")
        
        with self.assertRaises(OSError):
           sdxl_model_prep._load_target_model(
                self.cfg, "invalid/path", None, "v1", torch.float32
            ) 
