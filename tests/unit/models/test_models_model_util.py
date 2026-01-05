"""
Unit tests for library/models/model_util.py

Tests pure utility functions that don't require model loading.
"""

import pytest

from library.models.model_util import (
    create_vae_diffusers_config,
    reshape_weight_for_sd,
    conv_attn_to_linear,
    renew_attention_paths,
    renew_vae_resnet_paths,
    renew_resnet_paths,
    shave_segments,
)
from library.utils.safetensors_utils import is_safetensors
from library.models.sd_model_util import (
    controlnet_conversion_map,
    get_model_version_str_for_sd1_sd2,
    create_unet_diffusers_config,
    linear_transformer_to_conv,
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


# =============================================================================
# renew_resnet_paths Tests
# =============================================================================


@pytest.mark.unit
class TestRenewResnetPaths:
    """Test resnet path renaming utility."""

    def test_basic_renaming(self):
        """Should rename in_layers/out_layers to norm/conv."""
        old_list = [
            "block.in_layers.0.weight",
            "block.in_layers.2.weight",
            "block.out_layers.0.weight",
            "block.out_layers.3.weight",
        ]
        result = renew_resnet_paths(old_list, n_shave_prefix_segments=0)

        assert result[0]["new"] == "block.norm1.weight"
        assert result[1]["new"] == "block.conv1.weight"
        assert result[2]["new"] == "block.norm2.weight"
        assert result[3]["new"] == "block.conv2.weight"

    def test_emb_layers_and_skip_connection(self):
        """Should rename emb_layers and skip_connection."""
        old_list = [
            "block.emb_layers.1.weight",
            "block.skip_connection.weight",
        ]
        result = renew_resnet_paths(old_list, n_shave_prefix_segments=0)

        assert result[0]["new"] == "block.time_emb_proj.weight"
        assert result[1]["new"] == "block.conv_shortcut.weight"

    def test_empty_list(self):
        """Empty input should return empty output."""
        result = renew_resnet_paths([], n_shave_prefix_segments=0)
        assert result == []


# =============================================================================
# renew_vae_resnet_paths Tests
# =============================================================================


@pytest.mark.unit
class TestRenewVaeResnetPaths:
    """Test VAE resnet path renaming utility."""

    def test_nin_shortcut_renaming(self):
        """Should rename nin_shortcut to conv_shortcut."""
        old_list = ["block.nin_shortcut.weight"]
        result = renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0)

        assert result[0]["old"] == "block.nin_shortcut.weight"
        assert result[0]["new"] == "block.conv_shortcut.weight"

    def test_preserves_other_paths(self):
        """Paths without nin_shortcut should be preserved."""
        old_list = ["block.conv.weight"]
        result = renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0)

        assert result[0]["new"] == "block.conv.weight"


# =============================================================================
# renew_attention_paths Tests
# =============================================================================


@pytest.mark.unit
class TestRenewAttentionPaths:
    """Test attention path renaming utility."""

    def test_returns_mapping_structure(self):
        """Should return list of old/new mappings."""
        old_list = ["attn.proj_in.weight", "attn.proj_out.weight"]
        result = renew_attention_paths(old_list, n_shave_prefix_segments=0)

        assert len(result) == 2
        assert "old" in result[0]
        assert "new" in result[0]

    def test_shave_prefix_segments(self):
        """Note: shave_segments is commented out in current implementation, path is preserved."""
        old_list = ["model.attn.proj_in.weight"]
        result = renew_attention_paths(old_list, n_shave_prefix_segments=1)

        # Current implementation preserves the path (shave_segments is commented out)
        assert result[0]["new"] == "model.attn.proj_in.weight"


# =============================================================================
# get_model_version_str_for_sd1_sd2 Tests
# =============================================================================


@pytest.mark.unit
class TestGetModelVersionStr:
    """Test model version string generation."""

    def test_v1_no_vparam(self):
        """SD v1 without v_parameterization."""
        result = get_model_version_str_for_sd1_sd2(v2=False, v_parameterization=False)
        assert result == "sd_v1"

    def test_v1_with_vparam(self):
        """SD v1 with v_parameterization."""
        result = get_model_version_str_for_sd1_sd2(v2=False, v_parameterization=True)
        assert result == "sd_v1_v"

    def test_v2_no_vparam(self):
        """SD v2 without v_parameterization."""
        result = get_model_version_str_for_sd1_sd2(v2=True, v_parameterization=False)
        assert result == "sd_v2"

    def test_v2_with_vparam(self):
        """SD v2 with v_parameterization."""
        result = get_model_version_str_for_sd1_sd2(v2=True, v_parameterization=True)
        assert result == "sd_v2_v"


# =============================================================================
# conv_attn_to_linear Tests
# =============================================================================

import torch


@pytest.mark.unit
class TestConvAttnToLinear:
    """Test 4D->2D weight conversion for attention."""

    def test_reshapes_query_key_value(self):
        """Should reshape query/key/value weights from 4D to 2D."""
        checkpoint = {
            "attn.query.weight": torch.randn(64, 32, 1, 1),
            "attn.key.weight": torch.randn(64, 32, 1, 1),
            "attn.value.weight": torch.randn(64, 32, 1, 1),
            "other.weight": torch.randn(16, 16),  # Not matching
        }
        conv_attn_to_linear(checkpoint)

        assert checkpoint["attn.query.weight"].shape == (64, 32)
        assert checkpoint["attn.key.weight"].shape == (64, 32)
        assert checkpoint["attn.value.weight"].shape == (64, 32)
        assert checkpoint["other.weight"].shape == (16, 16)  # Unchanged

    def test_reshapes_proj_attn_to_3d(self):
        """proj_attn.weight uses [:,:,0] -> 3D output."""
        checkpoint = {
            "layer.proj_attn.weight": torch.randn(32, 16, 1, 1),
        }
        conv_attn_to_linear(checkpoint)

        # Note: implementation uses [:,:,0] for proj_attn which gives 3D
        assert checkpoint["layer.proj_attn.weight"].shape == (32, 16, 1)


# =============================================================================
# controlnet_conversion_map Tests
# =============================================================================


@pytest.mark.unit
class TestControlnetConversionMap:
    """Test ControlNet conversion map generation."""

    def test_returns_tuple_of_three_lists(self):
        """Should return 3 conversion maps: base, resnet, layer."""
        result = controlnet_conversion_map()

        assert isinstance(result, tuple)
        assert len(result) == 3
        # All three should be lists
        unet_map, resnet_map, layer_map = result
        assert isinstance(unet_map, list)
        assert isinstance(resnet_map, list)
        assert isinstance(layer_map, list)

    def test_contains_input_hint_block(self):
        """Layer map should contain input_hint_block mappings."""
        _, _, layer_map = controlnet_conversion_map()
        keys = [item[0] for item in layer_map]

        input_hint_keys = [k for k in keys if "input_hint_block" in k]
        assert len(input_hint_keys) > 0


# =============================================================================
# reshape_weight_for_sd Tests
# =============================================================================


@pytest.mark.unit
class TestReshapeWeightForSd:
    """Test weight reshaping for SD format."""

    def test_adds_spatial_dimensions(self):
        """Should add 1x1 spatial dimensions to 2D weights."""
        weight = torch.randn(64, 32)

        result = reshape_weight_for_sd(weight)

        assert result.shape == (64, 32, 1, 1)

    def test_preserves_values(self):
        """Reshaped tensor should have same values."""
        weight = torch.randn(16, 8)

        result = reshape_weight_for_sd(weight)

        assert torch.equal(result.squeeze(), weight)


# =============================================================================
# linear_transformer_to_conv Tests
# =============================================================================


@pytest.mark.unit
class TestLinearTransformerToConv:
    """Test 2D->4D weight conversion for transformers."""

    def test_converts_proj_in_weight(self):
        """Should convert 2D proj_in.weight to 4D."""
        checkpoint = {
            "layer.proj_in.weight": torch.randn(64, 32),
        }
        linear_transformer_to_conv(checkpoint)

        assert checkpoint["layer.proj_in.weight"].shape == (64, 32, 1, 1)

    def test_converts_proj_out_weight(self):
        """Should convert 2D proj_out.weight to 4D."""
        checkpoint = {
            "block.proj_out.weight": torch.randn(128, 64),
        }
        linear_transformer_to_conv(checkpoint)

        assert checkpoint["block.proj_out.weight"].shape == (128, 64, 1, 1)

    def test_ignores_already_4d(self):
        """Should not modify already 4D weights."""
        checkpoint = {
            "layer.proj_in.weight": torch.randn(64, 32, 1, 1),
        }
        linear_transformer_to_conv(checkpoint)

        # Should remain unchanged
        assert checkpoint["layer.proj_in.weight"].shape == (64, 32, 1, 1)

    def test_ignores_non_matching_keys(self):
        """Should not modify non-transformer weights."""
        checkpoint = {
            "other.weight": torch.randn(32, 16),
        }
        linear_transformer_to_conv(checkpoint)

        # Should remain 2D
        assert checkpoint["other.weight"].shape == (32, 16)
