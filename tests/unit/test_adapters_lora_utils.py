
import torch
from unittest.mock import patch
from library.adapters import lora_utils

class TestLoraUtils:
    
    # === filter_lora_state_dict Tests ===

    def test_filter_lora_state_dict_include(self):
        """Test filtering with include pattern."""
        state_dict = {
            "lora_unet_down_blocks_0_resnets_0_downsup_alpha": torch.tensor(1.0),
            "lora_unet_up_blocks_1_resnets_1_downsup_alpha": torch.tensor(1.0),
            "lora_te_text_model_encoder_layers_0_mlp_fc1_alpha": torch.tensor(1.0),
        }
        
        # Include only unet
        filtered = lora_utils.filter_lora_state_dict(state_dict, include_pattern="unet")
        assert len(filtered) == 2
        assert "lora_te_text_model_encoder_layers_0_mlp_fc1_alpha" not in filtered

    def test_filter_lora_state_dict_exclude(self):
        """Test filtering with exclude pattern."""
        state_dict = {
            "lora_unet_down_blocks_0_resnets_0_downsup_alpha": torch.tensor(1.0),
            "lora_unet_up_blocks_1_resnets_1_downsup_alpha": torch.tensor(1.0),
            "lora_te_text_model_encoder_layers_0_mlp_fc1_alpha": torch.tensor(1.0),
        }
        
        # Exclude text encoder
        filtered = lora_utils.filter_lora_state_dict(state_dict, exclude_pattern="te_")
        assert len(filtered) == 2
        assert "lora_te_text_model_encoder_layers_0_mlp_fc1_alpha" not in filtered

    def test_filter_lora_state_dict_combined(self):
        """Test filtering with both include and exclude patterns."""
        state_dict = {
            "lora_unet_block_1": torch.tensor(1.0),
            "lora_unet_block_2": torch.tensor(1.0),
            "lora_te_block_1": torch.tensor(1.0),
        }
        
        # Include unet, exclude block_2
        filtered = lora_utils.filter_lora_state_dict(
            state_dict, 
            include_pattern="unet", 
            exclude_pattern="block_2"
        )
        assert len(filtered) == 1
        assert "lora_unet_block_1" in filtered

    # === load_safetensors_with_lora_and_fp8 Tests ===

    @patch("library.adapters.lora_utils.load_safetensors_with_fp8_optimization_and_hook")
    def test_load_lora_merge_logic_linear(self, mock_load):
        """
        Test that LoRA weights are merged correctly for Linear layers (2D weights).
        W_prime = W + multiplier * (Up @ Down) * scale
        """
        # Mock dependencies
        calc_device = torch.device("cpu")
        
        # Define base model weight (Linear: out_features=2, in_features=2)
        base_weight = torch.eye(2) # Identity matrix
        
        # Define LoRA weights
        # down: rank=1, in=2 -> shape (1, 2)
        lora_down = torch.tensor([[1.0, 0.0]]) 
        # up: out=2, rank=1 -> shape (2, 1)
        lora_up = torch.tensor([[2.0], [0.0]])
        # alpha = 1.0, rank = 1 -> scale = 1.0/1 = 1.0
        lora_alpha = torch.tensor(1.0)
        
        lora_weights = {
            "lora_unet_layer.lora_down.weight": lora_down,
            "lora_unet_layer.lora_up.weight": lora_up,
            "lora_unet_layer.alpha": lora_alpha
        }
        
        # Setup mock behavior
        # The hook inside lora_utils will be called by our mock logic below?
        # No, we need to inspect how 'load_safetensors_with_fp8_optimization_and_hook' is called
        # and trigger the hook ourselves to test verification.
        # But 'load_safetensors_with_lora_and_fp8' does simpler thing: it passes a hook to the loader.
        # It's better to test the HOOK logic, but the hook is internal.
        # We can simulate the loader behavior by calling the captured hook.
        
        def mock_loader_side_effect(model_files, fp8_optimization, calc_device, move_to_device, dit_weight_dtype, target_keys, exclude_keys, weight_hook):
            # Simulate loading 'layer.weight' which matches 'lora_unet_layer'
            key = "layer.weight"
            val = base_weight.clone() # Pass clone so hook can modify
            if weight_hook:
                 val = weight_hook(key, val, keep_on_calc_device=False)
            return {key: val}
            
        mock_load.side_effect = mock_loader_side_effect
        
        result_sd = lora_utils.load_safetensors_with_lora_and_fp8(
            model_files=["dummy.safetensors"],
            lora_weights_list=[lora_weights],
            lora_multipliers=[1.0],
            fp8_optimization=False,
            calc_device=calc_device
        )
        
        # Expected calculation:
        # Up @ Down = [[2], [0]] @ [[1, 0]] = [[2, 0], [0, 0]]
        # W' = I + 1.0 * [[2, 0], [0, 0]] * 1.0 = [[1, 0], [0, 1]] + [[2, 0], [0, 0]] = [[3, 0], [0, 1]]
        
        expected_weight = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        assert torch.allclose(result_sd["layer.weight"], expected_weight)

    @patch("library.adapters.lora_utils.load_safetensors_with_fp8_optimization_and_hook")
    def test_load_lora_conv2d_1x1(self, mock_load):
        """Test LoRA merge for Conv2d 1x1 layers."""
        calc_device = torch.device("cpu")
        
        # Conv2d 1x1: shape (out_ch, in_ch, 1, 1)
        base_weight = torch.ones((2, 2, 1, 1))
        
        # LoRA for Conv2d 1x1 (rank=1)
        lora_down = torch.ones((1, 2, 1, 1))  # (rank, in_ch, 1, 1)
        lora_up = torch.ones((2, 1, 1, 1))    # (out_ch, rank, 1, 1)
        
        lora_weights = {
            "lora_unet_layer.lora_down.weight": lora_down,
            "lora_unet_layer.lora_up.weight": lora_up,
            "lora_unet_layer.alpha": torch.tensor(1.0)
        }
        
        def mock_loader_side_effect(*args, **kwargs):
            key = "layer.weight"
            val = base_weight.clone()
            hook = kwargs.get('weight_hook')
            if hook:
                val = hook(key, val, keep_on_calc_device=False)
            return {key: val}
        
        mock_load.side_effect = mock_loader_side_effect
        
        result_sd = lora_utils.load_safetensors_with_lora_and_fp8(
            model_files=["dummy"],
            lora_weights_list=[lora_weights],
            lora_multipliers=[1.0],
            fp8_optimization=False,
            calc_device=calc_device
        )
        
        # up.squeeze @ down.squeeze = [, ] @ [[1, 1]] = [[1, 1], [1, 1]]
        # Then unsqueeze back to (2, 2, 1, 1)
        # Result: ones + ones * 1.0 = 2*ones
        expected = torch.full((2, 2, 1, 1), 2.0)
        assert torch.allclose(result_sd["layer.weight"], expected)

    @patch("library.adapters.lora_utils.load_safetensors_with_fp8_optimization_and_hook")
    @patch("library.adapters.lora_utils.logger")
    def test_warns_on_unused_lora_keys(self, mock_logger, mock_load):
        """Test that warning is logged for LoRA keys not matched to any model weight."""
        calc_device = torch.device("cpu")
        
        lora_weights = {
            "lora_unet_nonexistent_layer.lora_down.weight": torch.ones(1, 1),
            "lora_unet_nonexistent_layer.lora_up.weight": torch.ones(1, 1),
            "lora_unet_nonexistent_layer.alpha": torch.tensor(1.0)
        }
        
        def mock_loader_side_effect(*args, **kwargs):
            # Return empty model (no weights to merge LoRA into)
            return {}
        
        mock_load.side_effect = mock_loader_side_effect
        
        lora_utils.load_safetensors_with_lora_and_fp8(
            model_files=["dummy"],
            lora_weights_list=[lora_weights],
            lora_multipliers=[1.0],
            fp8_optimization=False,
            calc_device=calc_device
        )
        
        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert "not all LoRA keys are used" in mock_logger.warning.call_args[0][0]

    @patch("library.adapters.lora_utils.load_safetensors_with_fp8_optimization_and_hook")
    @patch("os.path.exists", return_value=True)
    def test_handles_split_safetensors_files(self, mock_exists, mock_load):
        """Test that split model files (e.g., 00001-of-00003) are correctly discovered."""
        mock_load.return_value = {}
        
        lora_utils.load_safetensors_with_lora_and_fp8(
            model_files=["model-00001-of-00003.safetensors"],
            lora_weights_list=[],
            lora_multipliers=[],
            fp8_optimization=False,
            calc_device=torch.device("cpu")
        )
        
        # Should have called loader with all 3 files
        call_args = mock_load.call_args
        model_files_arg = call_args[0][0]  # First positional arg
        assert len(model_files_arg) == 3
        # Match expected filenames constructed in implementation
        # The implementation constructs: "model-{i+1:05d}-of-{count:05d}.safetensors"
        # It joins with os.path.dirname(model_file). Since input is just filename, dirname is empty string.
        # os.path.join("", "...") -> "..."
        assert "model-00001-of-00003.safetensors" in model_files_arg
        assert "model-00002-of-00003.safetensors" in model_files_arg
        assert "model-00003-of-00003.safetensors" in model_files_arg

    @patch("library.adapters.lora_utils.load_safetensors_with_fp8_optimization_and_hook")
    def test_load_lora_multiple_loras(self, mock_load):
        """Test merging multiple LoRAs."""
        calc_device = torch.device("cpu")
        base_weight = torch.zeros(1, 1)
        
        # LoRA 1: adds 1.0
        lora1 = {
            "lora_unet_layer.lora_down.weight": torch.ones(1, 1),
            "lora_unet_layer.lora_up.weight": torch.ones(1, 1),
            "lora_unet_layer.alpha": torch.tensor(1.0)
        }
        
        # LoRA 2: adds 2.0 (weights=2, scale=0.5 -> 2*2*0.5 = 2)
        lora2 = {
            "lora_unet_layer.lora_down.weight": torch.ones(1, 1) * 2,
            "lora_unet_layer.lora_up.weight": torch.ones(1, 1) * 2,
            "lora_unet_layer.alpha": torch.tensor(0.5) # dim=1, scale=0.5
        }
        
        def mock_loader_side_effect(model_files, fp8_optimization, calc_device, move_to_device, dit_weight_dtype, target_keys, exclude_keys, weight_hook):
            key = "layer.weight"
            val = base_weight.clone()
            if weight_hook:
                 val = weight_hook(key, val)
            return {key: val}
        mock_load.side_effect = mock_loader_side_effect
        
        result_sd = lora_utils.load_safetensors_with_lora_and_fp8(
            model_files=["dummy"],
            lora_weights_list=[lora1, lora2],
            lora_multipliers=[1.0, 1.0],
            fp8_optimization=False,
            calc_device=calc_device
        )
        
        # Expected: 0 + (1*1*1) + (2*2*0.5) = 1 + 2 = 3
        assert result_sd["layer.weight"].item() == 3.0

