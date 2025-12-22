"""
Unit tests for library/models/model_util.py

Tests pure utility functions that don't require model loading.
"""

import pytest

from library.models.model_util import (
    shave_segments,
    is_safetensors,
    create_unet_diffusers_config,
    create_vae_diffusers_config,
)
from library.constants import (
    UNET_PARAMS_IMAGE_SIZE,
    UNET_PARAMS_IN_CHANNELS,
    UNET_PARAMS_OUT_CHANNELS,
    UNET_PARAMS_CONTEXT_DIM,
    UNET_PARAMS_NUM_HEADS,
    V2_UNET_PARAMS_CONTEXT_DIM,
    V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
    VAE_PARAMS_RESOLUTION,
    VAE_PARAMS_IN_CHANNELS,
    VAE_PARAMS_OUT_CH,
    VAE_PARAMS_Z_CHANNELS,
)


# =============================================================================
# shave_segments Tests
# =============================================================================

@pytest.mark.unit
class TestShaveSegments:
    """Test path segment removal utility."""
    
    def test_shave_one_prefix_segment(self):
        """Default: remove first segment."""
        path = "model.encoder.layer1.weight"
        
        result = shave_segments(path, n_shave_prefix_segments=1)
        
        assert result == "encoder.layer1.weight"
    
    def test_shave_two_prefix_segments(self):
        """Remove first two segments."""
        path = "model.encoder.layer1.weight"
        
        result = shave_segments(path, n_shave_prefix_segments=2)
        
        assert result == "layer1.weight"
    
    def test_shave_zero_segments(self):
        """Zero shave should return original."""
        path = "model.encoder.layer1.weight"
        
        result = shave_segments(path, n_shave_prefix_segments=0)
        
        assert result == path
    
    def test_shave_negative_one_suffix(self):
        """Negative value removes last segment."""
        path = "model.encoder.layer1.weight"
        
        result = shave_segments(path, n_shave_prefix_segments=-1)
        
        assert result == "model.encoder.layer1"
    
    def test_shave_negative_two_suffix(self):
        """Remove last two segments."""
        path = "model.encoder.layer1.weight"
        
        result = shave_segments(path, n_shave_prefix_segments=-2)
        
        assert result == "model.encoder"
    
    def test_single_segment(self):
        """Single segment path."""
        path = "weight"
        
        result = shave_segments(path, n_shave_prefix_segments=1)
        
        assert result == ""
    
    def test_empty_path(self):
        """Empty path returns empty."""
        result = shave_segments("", n_shave_prefix_segments=1)
        
        assert result == ""


# =============================================================================
# is_safetensors Tests
# =============================================================================

@pytest.mark.unit
class TestIsSafetensors:
    """Test safetensors file detection."""
    
    def test_safetensors_extension(self):
        """Should detect .safetensors files."""
        assert is_safetensors("model.safetensors") is True
        assert is_safetensors("/path/to/model.safetensors") is True
    
    def test_safetensors_uppercase(self):
        """Should detect uppercase extension."""
        assert is_safetensors("model.SAFETENSORS") is True
        assert is_safetensors("model.Safetensors") is True
    
    def test_non_safetensors(self):
        """Should reject non-safetensors files."""
        assert is_safetensors("model.ckpt") is False
        assert is_safetensors("model.pt") is False
        assert is_safetensors("model.bin") is False
        assert is_safetensors("model.pth") is False
    
    def test_no_extension(self):
        """Files without extension are not safetensors."""
        assert is_safetensors("model") is False
        assert is_safetensors("/path/to/model") is False


# =============================================================================
# create_unet_diffusers_config Tests
# =============================================================================

@pytest.mark.unit
class TestCreateUnetDiffusersConfig:
    """Test UNet config generation for Diffusers."""
    
    def test_v1_config_structure(self):
        """SD 1.x config should have expected keys."""
        config = create_unet_diffusers_config(v2=False)
        
        assert "sample_size" in config
        assert "in_channels" in config
        assert "out_channels" in config
        assert "down_block_types" in config
        assert "up_block_types" in config
        assert "block_out_channels" in config
        assert "layers_per_block" in config
        assert "cross_attention_dim" in config
        assert "attention_head_dim" in config
    
    def test_v1_config_values(self):
        """SD 1.x config should use correct constants."""
        config = create_unet_diffusers_config(v2=False)
        
        assert config["sample_size"] == UNET_PARAMS_IMAGE_SIZE
        assert config["in_channels"] == UNET_PARAMS_IN_CHANNELS
        assert config["out_channels"] == UNET_PARAMS_OUT_CHANNELS
        assert config["cross_attention_dim"] == UNET_PARAMS_CONTEXT_DIM
        assert config["attention_head_dim"] == UNET_PARAMS_NUM_HEADS
    
    def test_v1_block_types(self):
        """SD 1.x should have correct block types."""
        config = create_unet_diffusers_config(v2=False)
        
        # Down blocks: 3 CrossAttn + 1 regular
        assert len(config["down_block_types"]) == 4
        assert "CrossAttnDownBlock2D" in config["down_block_types"]
        assert "DownBlock2D" in config["down_block_types"]
        
        # Up blocks: 1 regular + 3 CrossAttn  
        assert len(config["up_block_types"]) == 4
    
    def test_v2_config_uses_v2_constants(self):
        """SD 2.x config should use V2 constants."""
        config = create_unet_diffusers_config(v2=True)
        
        assert config["cross_attention_dim"] == V2_UNET_PARAMS_CONTEXT_DIM
        assert config["attention_head_dim"] == V2_UNET_PARAMS_ATTENTION_HEAD_DIM
    
    def test_v2_linear_projection_option(self):
        """V2 with use_linear_projection should add the key."""
        config = create_unet_diffusers_config(v2=True, use_linear_projection_in_v2=True)
        
        assert config.get("use_linear_projection") is True
    
    def test_v2_no_linear_projection_by_default(self):
        """V2 without use_linear_projection should not add the key."""
        config = create_unet_diffusers_config(v2=True, use_linear_projection_in_v2=False)
        
        assert "use_linear_projection" not in config


# =============================================================================
# create_vae_diffusers_config Tests
# =============================================================================

@pytest.mark.unit
class TestCreateVaeDiffusersConfig:
    """Test VAE config generation for Diffusers."""
    
    def test_config_structure(self):
        """VAE config should have expected keys."""
        config = create_vae_diffusers_config()
        
        assert "sample_size" in config
        assert "in_channels" in config
        assert "out_channels" in config
        assert "down_block_types" in config
        assert "up_block_types" in config
        assert "block_out_channels" in config
        assert "latent_channels" in config
        assert "layers_per_block" in config
    
    def test_config_values(self):
        """VAE config should use correct constants."""
        config = create_vae_diffusers_config()
        
        assert config["sample_size"] == VAE_PARAMS_RESOLUTION
        assert config["in_channels"] == VAE_PARAMS_IN_CHANNELS
        assert config["out_channels"] == VAE_PARAMS_OUT_CH
        assert config["latent_channels"] == VAE_PARAMS_Z_CHANNELS
    
    def test_block_types_are_encoder_decoder(self):
        """VAE should use encoder/decoder block types."""
        config = create_vae_diffusers_config()
        
        assert all(bt == "DownEncoderBlock2D" for bt in config["down_block_types"])
        assert all(bt == "UpDecoderBlock2D" for bt in config["up_block_types"])
    
    def test_block_out_channels_calculated(self):
        """Block out channels should be computed from CH and CH_MULT."""
        config = create_vae_diffusers_config()
        
        # VAE_PARAMS_CH=128, VAE_PARAMS_CH_MULT=[1,2,4,4]
        expected = (128, 256, 512, 512)
        assert config["block_out_channels"] == expected
