"""
Unit tests for library/models/sdxl_model_util.py

Tests pure math functions and conversion maps that don't require model loading.
"""

import pytest
import torch

from library.models.sdxl_model_util import (
    timestep_embedding,
    get_timestep_embedding,
    get_size_embeddings,
    make_unet_conversion_map,
    convert_unet_state_dict,
    convert_diffusers_unet_state_dict_to_sdxl,
    convert_sdxl_unet_state_dict_to_diffusers,
)


# =============================================================================
# timestep_embedding Tests
# =============================================================================


@pytest.mark.unit
class TestTimestepEmbedding:
    """Test sinusoidal timesteps embedding generation."""

    def test_output_shape_even_dim(self):
        """Even dimension should produce correct output shape."""
        timesteps = torch.tensor([0, 100, 500, 999])
        dim = 256

        result = timestep_embedding(timesteps, dim)

        assert result.shape == (4, 256)

    def test_output_shape_odd_dim(self):
        """Odd dimension should zero-pad the last element."""
        timesteps = torch.tensor([0, 100])
        dim = 255  # Odd

        result = timestep_embedding(timesteps, dim)

        assert result.shape == (2, 255)
        # Last column should be zeros due to padding
        assert torch.allclose(result[:, -1], torch.zeros(2))

    def test_different_timesteps_produce_different_embeddings(self):
        """Different timesteps should produce different embeddings."""
        t0 = torch.tensor([0])
        t500 = torch.tensor([500])
        t999 = torch.tensor([999])

        emb0 = timestep_embedding(t0, 128)
        emb500 = timestep_embedding(t500, 128)
        emb999 = timestep_embedding(t999, 128)

        assert not torch.allclose(emb0, emb500)
        assert not torch.allclose(emb500, emb999)
        assert not torch.allclose(emb0, emb999)

    def test_zero_timestep(self):
        """Timestep 0 should produce valid embedding."""
        timesteps = torch.tensor([0.0])

        result = timestep_embedding(timesteps, 64)

        assert result.shape == (1, 64)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_fractional_timesteps(self):
        """Fractional timesteps should work."""
        timesteps = torch.tensor([0.5, 1.5, 999.5])

        result = timestep_embedding(timesteps, 128)

        assert result.shape == (3, 128)
        assert not torch.isnan(result).any()

    def test_values_bounded(self):
        """Sinusoidal embeddings should be bounded by [-1, 1]."""
        timesteps = torch.tensor([0, 250, 500, 750, 999], dtype=torch.float32)

        result = timestep_embedding(timesteps, 256)

        assert result.min() >= -1.0
        assert result.max() <= 1.0


# =============================================================================
# get_timestep_embedding Tests
# =============================================================================


@pytest.mark.unit
class TestGetTimestepEmbedding:
    """Test timesteps embedding wrapper for 2D inputs."""

    def test_2d_input_shape(self):
        """2D input should flatten, embed, and reshape correctly."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        outdim = 64

        result = get_timestep_embedding(x, outdim)

        # Should be (batch, dims * outdim) = (2, 2 * 64) = (2, 128)
        assert result.shape == (2, 128)

    def test_single_element_per_batch(self):
        """Single element per batch item."""
        x = torch.tensor([[100.0], [200.0], [300.0]])  # (3, 1)
        outdim = 256

        result = get_timestep_embedding(x, outdim)

        assert result.shape == (3, 256)


# =============================================================================
# get_size_embeddings Tests
# =============================================================================


@pytest.mark.unit
class TestGetSizeEmbeddings:
    """Test SDXL size conditioning embeddings."""

    def test_output_shape(self):
        """Size embeddings should have correct shape."""
        orig_size = torch.tensor([[1024, 1024]], dtype=torch.float32)
        crop_size = torch.tensor([[0, 0]], dtype=torch.float32)
        target_size = torch.tensor([[1024, 1024]], dtype=torch.float32)

        result = get_size_embeddings(orig_size, crop_size, target_size, "cpu")

        # 3 embeddings of (batch, 2 * 256) = (1, 512) each, concatenated = (1, 1536)
        assert result.shape == (1, 1536)

    def test_different_sizes_produce_different_embeddings(self):
        """Different input sizes should produce different embeddings."""
        orig_a = torch.tensor([[512, 512]], dtype=torch.float32)
        orig_b = torch.tensor([[1024, 1024]], dtype=torch.float32)
        crop = torch.tensor([[0, 0]], dtype=torch.float32)
        target = torch.tensor([[1024, 1024]], dtype=torch.float32)

        emb_a = get_size_embeddings(orig_a, crop, target, "cpu")
        emb_b = get_size_embeddings(orig_b, crop, target, "cpu")

        assert not torch.allclose(emb_a, emb_b)

    def test_batch_support(self):
        """Should support batched inputs."""
        orig_size = torch.tensor([[1024, 1024], [768, 1024]], dtype=torch.float32)
        crop_size = torch.tensor([[0, 0], [128, 0]], dtype=torch.float32)
        target_size = torch.tensor([[1024, 1024], [1024, 1024]], dtype=torch.float32)

        result = get_size_embeddings(orig_size, crop_size, target_size, "cpu")

        assert result.shape == (2, 1536)


# =============================================================================
# make_unet_conversion_map Tests
# =============================================================================


@pytest.mark.unit
class TestMakeUnetConversionMap:
    """Test SDXL UNet state dict conversion map generation."""

    def test_returns_list_of_tuples(self):
        """Should return a list of (sd_key, hf_key) tuples."""
        result = make_unet_conversion_map()

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_contains_expected_mappings(self):
        """Should contain critical mapping entries."""
        result = make_unet_conversion_map()
        mapping_dict = {sd: hf for sd, hf in result}

        # Check some expected mappings
        assert "input_blocks.0.0." in mapping_dict
        assert mapping_dict["input_blocks.0.0."] == "conv_in."

        assert "out.0." in mapping_dict
        assert mapping_dict["out.0."] == "conv_norm_out."

        assert "out.2." in mapping_dict
        assert mapping_dict["out.2."] == "conv_out."

    def test_contains_time_embedding_mappings(self):
        """Should contain time embedding mappings."""
        result = make_unet_conversion_map()
        mapping_dict = {sd: hf for sd, hf in result}

        assert "time_embed.0." in mapping_dict
        assert "time_embed.2." in mapping_dict

    def test_contains_label_embedding_mappings(self):
        """Should contain SDXL label embedding mappings."""
        result = make_unet_conversion_map()
        mapping_dict = {sd: hf for sd, hf in result}

        assert "label_emb.0.0." in mapping_dict
        assert "label_emb.0.2." in mapping_dict

    def test_contains_mid_block_mappings(self):
        """Should contain middle block mappings."""
        result = make_unet_conversion_map()
        mapping_dict = {sd: hf for sd, hf in result}

        assert "middle_block.1." in mapping_dict
        assert mapping_dict["middle_block.1."] == "mid_block.attentions.0."

    def test_mapping_count_reasonable(self):
        """Should have a reasonable number of mappings for SDXL."""
        result = make_unet_conversion_map()

        # SDXL UNet has many layers; should have >100 mappings
        assert len(result) > 100


# =============================================================================
# convert_unet_state_dict Tests
# =============================================================================


@pytest.mark.unit
class TestConvertUnetStateDict:
    """Test state dict key conversion utility."""

    def test_converts_keys_using_map(self):
        """Should convert keys using provided conversion map."""
        src_sd = {
            "input_blocks.0.0.weight": torch.randn(4, 4),
            "input_blocks.0.0.bias": torch.randn(4),
        }
        conversion_map = {"input_blocks.0.0.": "conv_in."}

        result = convert_unet_state_dict(src_sd, conversion_map)

        assert "conv_in.weight" in result
        assert "conv_in.bias" in result
        assert "input_blocks.0.0.weight" not in result

    def test_preserves_tensor_values(self):
        """Converted state dict should have same tensor values."""
        original_tensor = torch.randn(8, 8)
        src_sd = {"old_prefix.weight": original_tensor}
        conversion_map = {"old_prefix.": "new_prefix."}

        result = convert_unet_state_dict(src_sd, conversion_map)

        assert torch.equal(result["new_prefix.weight"], original_tensor)

    def test_raises_on_unmapped_key(self):
        """Should raise AssertionError if key not found in map."""
        src_sd = {"unknown.key.weight": torch.randn(4)}
        conversion_map = {"different.prefix.": "output."}

        with pytest.raises(AssertionError, match="not found in conversion map"):
            convert_unet_state_dict(src_sd, conversion_map)


# =============================================================================
# Bidirectional Conversion Tests
# =============================================================================


@pytest.mark.unit
class TestBidirectionalConversion:
    """Test SDXL <-> Diffusers state dict conversion."""

    def test_sdxl_to_diffusers_converts_conv_in(self):
        """Should convert SDXL input_blocks.0.0 to Diffusers conv_in."""
        sdxl_sd = {
            "input_blocks.0.0.weight": torch.randn(320, 4, 3, 3),
            "input_blocks.0.0.bias": torch.randn(320),
        }

        result = convert_sdxl_unet_state_dict_to_diffusers(sdxl_sd)

        assert "conv_in.weight" in result
        assert "conv_in.bias" in result

    def test_diffusers_to_sdxl_converts_conv_in(self):
        """Should convert Diffusers conv_in to SDXL input_blocks.0.0."""
        diffusers_sd = {
            "conv_in.weight": torch.randn(320, 4, 3, 3),
            "conv_in.bias": torch.randn(320),
        }

        result = convert_diffusers_unet_state_dict_to_sdxl(diffusers_sd)

        assert "input_blocks.0.0.weight" in result
        assert "input_blocks.0.0.bias" in result

    def test_roundtrip_preserves_keys(self):
        """Converting SDXL -> Diffusers -> SDXL should preserve keys."""
        original_sdxl = {
            "input_blocks.0.0.weight": torch.randn(320, 4, 3, 3),
            "out.2.weight": torch.randn(4, 320, 3, 3),
        }

        diffusers = convert_sdxl_unet_state_dict_to_diffusers(original_sdxl)
        back_to_sdxl = convert_diffusers_unet_state_dict_to_sdxl(diffusers)

        assert set(original_sdxl.keys()) == set(back_to_sdxl.keys())

    def test_roundtrip_preserves_values(self):
        """Roundtrip conversion should preserve tensor values."""
        original_tensor = torch.randn(320, 4, 3, 3)
        original_sdxl = {"input_blocks.0.0.weight": original_tensor}

        diffusers = convert_sdxl_unet_state_dict_to_diffusers(original_sdxl)
        back_to_sdxl = convert_diffusers_unet_state_dict_to_sdxl(diffusers)

        assert torch.equal(back_to_sdxl["input_blocks.0.0.weight"], original_tensor)
