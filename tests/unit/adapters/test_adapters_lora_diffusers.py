"""
Unit tests for the pure utility functions in library/adapters/lora_diffusers.py.

Tests the UNet conversion map used to convert between Stability AI and Diffusers layer naming.
"""

from library.adapters.methods.peft.lora.lora_diffusers import make_unet_conversion_map, UNET_CONVERSION_MAP


class TestMakeUnetConversionMap:
    """Tests for make_unet_conversion_map function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = make_unet_conversion_map()
        assert isinstance(result, dict)

    def test_not_empty(self):
        """Conversion map should not be empty."""
        result = make_unet_conversion_map()
        assert len(result) > 0

    def test_keys_are_sd_format(self):
        """Keys should be in Stability AI format (using underscores, no trailing dots)."""
        result = make_unet_conversion_map()
        for key in result.keys():
            assert "." not in key, f"Key {key} should not contain dots"
            assert not key.endswith("_"), f"Key {key} should not end with underscore"

    def test_values_are_hf_format(self):
        """Values should be in HuggingFace/Diffusers format (using underscores, no trailing dots)."""
        result = make_unet_conversion_map()
        for value in result.values():
            assert "." not in value, f"Value {value} should not contain dots"
            assert not value.endswith("_"), f"Value {value} should not end with underscore"

    def test_contains_input_blocks(self):
        """Should contain mappings for input_blocks."""
        result = make_unet_conversion_map()
        input_block_keys = [k for k in result.keys() if k.startswith("input_blocks")]
        assert len(input_block_keys) > 0, "Should have input_blocks mappings"

    def test_contains_output_blocks(self):
        """Should contain mappings for output_blocks."""
        result = make_unet_conversion_map()
        output_block_keys = [k for k in result.keys() if k.startswith("output_blocks")]
        assert len(output_block_keys) > 0, "Should have output_blocks mappings"

    def test_contains_middle_block(self):
        """Should contain mappings for middle_block."""
        result = make_unet_conversion_map()
        middle_block_keys = [k for k in result.keys() if k.startswith("middle_block")]
        assert len(middle_block_keys) > 0, "Should have middle_block mappings"

    def test_contains_time_embed(self):
        """Should contain mappings for time_embed."""
        result = make_unet_conversion_map()
        time_embed_keys = [k for k in result.keys() if k.startswith("time_embed")]
        assert len(time_embed_keys) > 0, "Should have time_embed mappings"

    def test_contains_conv_in(self):
        """Should contain mapping for conv_in (input_blocks.0.0)."""
        result = make_unet_conversion_map()
        # input_blocks.0.0 -> conv_in
        assert "input_blocks_0_0" in result

    def test_contains_conv_out(self):
        """Should contain mapping for conv_out (out.2)."""
        result = make_unet_conversion_map()
        # out.2 -> conv_out
        assert "out_2" in result

    def test_resnet_submappings(self):
        """Resnet blocks should have norm1, conv1, norm2, conv2 submappings."""
        result = make_unet_conversion_map()
        # Check that resnet-related mappings include the expected subcomponents
        norm1_mappings = [k for k in result.keys() if "norm1" in result[k]]
        conv1_mappings = [k for k in result.keys() if "conv1" in result[k]]
        norm2_mappings = [k for k in result.keys() if "norm2" in result[k]]
        conv2_mappings = [k for k in result.keys() if "conv2" in result[k]]

        assert len(norm1_mappings) > 0, "Should have norm1 mappings"
        assert len(conv1_mappings) > 0, "Should have conv1 mappings"
        assert len(norm2_mappings) > 0, "Should have norm2 mappings"
        assert len(conv2_mappings) > 0, "Should have conv2 mappings"

    def test_specific_known_mapping(self):
        """Test a specific known mapping."""
        result = make_unet_conversion_map()
        # middle_block.1 -> mid_block.attentions.0
        assert "middle_block_1" in result
        assert result["middle_block_1"] == "mid_block_attentions_0"

    def test_module_level_constant_matches_function(self):
        """UNET_CONVERSION_MAP constant should match make_unet_conversion_map() output."""
        result = make_unet_conversion_map()
        assert result == UNET_CONVERSION_MAP


class TestUnetConversionMapCoverage:
    """Tests to verify conversion map has complete coverage for SDXL architecture."""

    def test_down_blocks_coverage(self):
        """Should have mappings for all down_blocks (3 blocks x 2 resnets + attentions)."""
        result = make_unet_conversion_map()
        hf_down_blocks = [v for v in result.values() if v.startswith("down_blocks")]
        # Should have multiple down_block mappings
        assert len(hf_down_blocks) >= 6, f"Should have >= 6 down_block mappings, got {len(hf_down_blocks)}"

    def test_up_blocks_coverage(self):
        """Should have mappings for all up_blocks (3 blocks x 3 resnets + attentions)."""
        result = make_unet_conversion_map()
        hf_up_blocks = [v for v in result.values() if v.startswith("up_blocks")]
        # Should have multiple up_block mappings
        assert len(hf_up_blocks) >= 9, f"Should have >= 9 up_block mappings, got {len(hf_up_blocks)}"

    def test_no_duplicate_keys(self):
        """All keys should be unique."""
        result = make_unet_conversion_map()
        keys = list(result.keys())
        assert len(keys) == len(set(keys)), "Duplicate keys found in conversion map"

    def test_downsamplers_mapping_exists(self):
        """Should have downsamplers mappings."""
        result = make_unet_conversion_map()
        downsamplers = [v for v in result.values() if "downsamplers" in v]
        assert len(downsamplers) > 0, "Should have downsamplers mappings"

    def test_upsamplers_mapping_exists(self):
        """Should have upsamplers mappings."""
        result = make_unet_conversion_map()
        upsamplers = [v for v in result.values() if "upsamplers" in v]
        assert len(upsamplers) > 0, "Should have upsamplers mappings"
