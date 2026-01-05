"""
Unit tests for the pure utility functions in library/adapters/lora.py.

These functions handle block learning rate parsing, dims/alphas calculation,
and don't require heavy model mocks.
"""

import pytest

# Import the functions and constants we're testing
from library.adapters.lora import (
    parse_block_lr_kwargs,
    get_block_dims_and_alphas,
    get_block_lr_weight,
    remove_block_dims_and_alphas,
    get_block_index,
    LoRAAdapter,
)


class TestParseBlockLrKwargs:
    """Tests for parse_block_lr_kwargs function."""

    def test_returns_none_when_no_lr_weights_provided(self):
        """Should return None if no down/mid/up LR weights are provided."""
        kwargs = {"some_other_param": 123}
        result = parse_block_lr_kwargs(is_sdxl=False, nw_kwargs=kwargs)
        assert result is None

    def test_parses_comma_separated_down_lr_weight(self):
        """Should parse comma-separated down_lr_weight string."""
        kwargs = {"down_lr_weight": "1.0,0.5,0.0,0.8"}
        # This will call get_block_lr_weight which fills in defaults for mid and up
        result = parse_block_lr_kwargs(is_sdxl=False, nw_kwargs=kwargs)
        assert result is not None
        # For SD1.5: total blocks = 12*2 + 1 = 25
        assert len(result) == LoRAAdapter.NUM_OF_BLOCKS * 2 + LoRAAdapter.NUM_OF_MID_BLOCKS

    def test_parses_mid_lr_weight(self):
        """Should parse mid_lr_weight string."""
        kwargs = {"mid_lr_weight": "0.5"}
        result = parse_block_lr_kwargs(is_sdxl=False, nw_kwargs=kwargs)
        assert result is not None
        assert len(result) == LoRAAdapter.NUM_OF_BLOCKS * 2 + LoRAAdapter.NUM_OF_MID_BLOCKS

    def test_parses_up_lr_weight_comma_separated(self):
        """Should parse comma-separated up_lr_weight string."""
        kwargs = {"up_lr_weight": "0.0,0.5,1.0"}
        result = parse_block_lr_kwargs(is_sdxl=False, nw_kwargs=kwargs)
        assert result is not None

    def test_parses_named_lr_weight_pattern(self):
        """Should parse named patterns like 'cosine', 'sine', 'linear'."""
        kwargs = {"down_lr_weight": "cosine"}
        result = parse_block_lr_kwargs(is_sdxl=False, nw_kwargs=kwargs)
        assert result is not None

    def test_uses_zero_threshold_from_kwargs(self):
        """Should apply block_lr_zero_threshold from kwargs."""
        kwargs = {"down_lr_weight": "0.01,0.5,1.0", "block_lr_zero_threshold": "0.05"}
        result = parse_block_lr_kwargs(is_sdxl=False, nw_kwargs=kwargs)
        assert result is not None
        # Values below threshold should become 0
        # The first value 0.01 is below threshold 0.05, so it should be 0
        assert result[0] == 0  # First block of down weights

    def test_sdxl_total_blocks(self):
        """For SDXL, should have different total blocks count."""
        kwargs = {"down_lr_weight": "cosine"}
        result = parse_block_lr_kwargs(is_sdxl=True, nw_kwargs=kwargs)
        assert result is not None
        # SDXL: 1 + 9*2 + 3 + 1 = 23
        expected_len = 1 + LoRAAdapter.SDXL_NUM_OF_BLOCKS * 2 + LoRAAdapter.SDXL_NUM_OF_MID_BLOCKS + 1
        assert len(result) == expected_len


class TestGetBlockDimsAndAlphas:
    """Tests for get_block_dims_and_alphas function."""

    def test_creates_default_dims_when_block_dims_none(self):
        """Should fill all dims with dim when block_dims is None."""
        adapter_rank = 16
        adapter_alpha = 8.0
        block_dims, block_alphas, conv_block_dims, conv_block_alphas = get_block_dims_and_alphas(
            is_sdxl=False,
            block_dims=None,
            block_alphas=None,
            adapter_rank=adapter_rank,
            adapter_alpha=adapter_alpha,
            conv_block_dims=None,
            conv_block_alphas=None,
            conv_dim=None,
            conv_alpha=None,
        )

        num_blocks = LoRAAdapter.NUM_OF_BLOCKS * 2 + LoRAAdapter.NUM_OF_MID_BLOCKS  # 25
        assert len(block_dims) == num_blocks
        assert all(d == adapter_rank for d in block_dims)
        assert len(block_alphas) == num_blocks
        assert all(a == adapter_alpha for a in block_alphas)
        assert conv_block_dims is None
        assert conv_block_alphas is None

    def test_parses_comma_separated_block_dims(self):
        """Should parse comma-separated block_dims string."""
        num_blocks = LoRAAdapter.NUM_OF_BLOCKS * 2 + LoRAAdapter.NUM_OF_MID_BLOCKS
        block_dims_str = ",".join(str(i) for i in range(num_blocks))
        block_alphas_str = ",".join(str(float(i)) for i in range(num_blocks))

        block_dims, block_alphas, _, _ = get_block_dims_and_alphas(
            is_sdxl=False,
            block_dims=block_dims_str,
            block_alphas=block_alphas_str,
            adapter_rank=4,
            adapter_alpha=1.0,
            conv_block_dims=None,
            conv_block_alphas=None,
            conv_dim=None,
            conv_alpha=None,
        )

        assert block_dims == list(range(num_blocks))
        assert block_alphas == [float(i) for i in range(num_blocks)]

    def test_sdxl_num_blocks(self):
        """SDXL should have 23 blocks."""
        adapter_rank = 8
        adapter_alpha = 4.0

        block_dims, block_alphas, _, _ = get_block_dims_and_alphas(
            is_sdxl=True,
            block_dims=None,
            block_alphas=None,
            adapter_rank=adapter_rank,
            adapter_alpha=adapter_alpha,
            conv_block_dims=None,
            conv_block_alphas=None,
            conv_dim=None,
            conv_alpha=None,
        )

        expected_num_blocks = 1 + LoRAAdapter.SDXL_NUM_OF_BLOCKS * 2 + LoRAAdapter.SDXL_NUM_OF_MID_BLOCKS + 1
        assert len(block_dims) == expected_num_blocks

    def test_sets_conv_block_dims_from_conv_dim(self):
        """When conv_dim is provided but no conv_block_dims, should fill with conv_dim."""
        conv_dim = 32
        conv_alpha = 16.0

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = get_block_dims_and_alphas(
            is_sdxl=False,
            block_dims=None,
            block_alphas=None,
            adapter_rank=8,
            adapter_alpha=4.0,
            conv_block_dims=None,
            conv_block_alphas=None,
            conv_dim=conv_dim,
            conv_alpha=conv_alpha,
        )

        num_blocks = LoRAAdapter.NUM_OF_BLOCKS * 2 + LoRAAdapter.NUM_OF_MID_BLOCKS
        assert len(conv_block_dims) == num_blocks
        assert all(d == conv_dim for d in conv_block_dims)
        assert len(conv_block_alphas) == num_blocks
        assert all(a == conv_alpha for a in conv_block_alphas)

    def test_raises_on_wrong_block_dims_count(self):
        """Should raise AssertionError if block_dims count doesn't match expected."""
        with pytest.raises(AssertionError, match="block_dims must have"):
            get_block_dims_and_alphas(
                is_sdxl=False,
                block_dims="1,2,3",  # Only 3, but need 25
                block_alphas=None,
                adapter_rank=4,
                adapter_alpha=1.0,
                conv_block_dims=None,
                conv_block_alphas=None,
                conv_dim=None,
                conv_alpha=None,
            )


class TestGetBlockLrWeight:
    """Tests for get_block_lr_weight function."""

    def test_returns_none_when_all_weights_none(self):
        """Should return None when no weights are provided."""
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight=None,
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is None

    def test_cosine_pattern_generates_descending_values(self):
        """Cosine pattern should generate values that start high and decrease."""
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight="cosine",
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is not None
        down_portion = result[: LoRAAdapter.NUM_OF_BLOCKS]
        # Cosine starts from 1.0 (at block 0), decreasing towards 0
        assert down_portion[0] > down_portion[-1]

    def test_sine_pattern_generates_ascending_values(self):
        """Sine pattern should generate values that start low and increase."""
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight="sine",
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is not None
        down_portion = result[: LoRAAdapter.NUM_OF_BLOCKS]
        # Sine starts from 0.0 (at block 0), increasing towards 1
        assert down_portion[0] < down_portion[-1]

    def test_linear_pattern(self):
        """Linear pattern should generate linearly increasing values."""
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight="linear",
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is not None
        down_portion = result[: LoRAAdapter.NUM_OF_BLOCKS]
        # First value ~ 0, last value = 1.0
        assert down_portion[0] == pytest.approx(0.0, abs=0.01)
        assert down_portion[-1] == pytest.approx(1.0, abs=0.01)

    def test_reverse_linear_pattern(self):
        """Reverse linear pattern should generate linearly decreasing values."""
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight="reverse_linear",
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is not None
        down_portion = result[: LoRAAdapter.NUM_OF_BLOCKS]
        # First value = 1.0, last value ~ 0
        assert down_portion[0] == pytest.approx(1.0, abs=0.01)
        assert down_portion[-1] == pytest.approx(0.0, abs=0.01)

    def test_zeros_pattern(self):
        """Zeros pattern should generate all zeros."""
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight="zeros",
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is not None
        down_portion = result[: LoRAAdapter.NUM_OF_BLOCKS]
        assert all(v == 0.0 for v in down_portion)

    def test_base_lr_addition(self):
        """Pattern+base_lr should add base_lr to all values."""
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight="zeros+0.5",
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is not None
        down_portion = result[: LoRAAdapter.NUM_OF_BLOCKS]
        assert all(v == 0.5 for v in down_portion)

    def test_list_input_for_weights(self):
        """Should accept list input for weights."""
        down_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight=down_weights,
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is not None
        assert result[:12] == down_weights

    def test_zero_threshold_zeroes_small_values(self):
        """Values below zero_threshold should be set to 0."""
        down_weights = [0.05, 0.5, 1.0]  # 0.05 is below threshold
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight=down_weights,
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.1,
        )
        assert result is not None
        assert result[0] == 0  # 0.05 was below threshold

    def test_sdxl_adds_wrapper_values(self):
        """SDXL should add 1.0 at start and end for emb_layers and out."""
        result = get_block_lr_weight(
            is_sdxl=True,
            down_lr_weight="zeros",
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is not None
        # First and last values should be 1.0 (for emb_layers and out)
        assert result[0] == 1.0
        assert result[-1] == 1.0

    def test_short_weights_padded_with_ones(self):
        """Short weight lists should be padded with 1.0."""
        down_weights = [0.5, 0.5, 0.5]  # Only 3, but need 12
        result = get_block_lr_weight(
            is_sdxl=False,
            down_lr_weight=down_weights,
            mid_lr_weight=None,
            up_lr_weight=None,
            zero_threshold=0.0,
        )
        assert result is not None
        # First 3 should be 0.5, rest should be 1.0
        assert result[0:3] == [0.5, 0.5, 0.5]
        # Entries 3-11 (rest of down) should be 1.0
        assert all(v == 1.0 for v in result[3 : LoRAAdapter.NUM_OF_BLOCKS])


class TestRemoveBlockDimsAndAlphas:
    """Tests for remove_block_dims_and_alphas function."""

    def test_returns_unchanged_when_no_lr_weight(self):
        """Should return dims unchanged when block_lr_weight is None."""
        block_dims = [4, 8, 16]
        block_alphas = [1.0, 2.0, 4.0]

        result = remove_block_dims_and_alphas(
            is_sdxl=False,
            block_dims=block_dims,
            block_alphas=block_alphas,
            conv_block_dims=None,
            conv_block_alphas=None,
            block_lr_weight=None,
        )

        assert result[0] == [4, 8, 16]
        assert result[1] == [1.0, 2.0, 4.0]

    def test_zeroes_dims_for_zero_lr_weight(self):
        """Should set dims to 0 where lr_weight is 0."""
        block_dims = [4, 8, 16]
        block_alphas = [1.0, 2.0, 4.0]
        block_lr_weight = [1.0, 0.0, 1.0]  # Middle block has 0 LR

        result = remove_block_dims_and_alphas(
            is_sdxl=False,
            block_dims=block_dims,
            block_alphas=block_alphas,
            conv_block_dims=None,
            conv_block_alphas=None,
            block_lr_weight=block_lr_weight,
        )

        assert result[0] == [4, 0, 16]  # Middle dim zeroed
        assert result[1] == [1.0, 2.0, 4.0]  # Alphas unchanged

    def test_zeroes_conv_dims_for_zero_lr_weight(self):
        """Should also zero conv_block_dims where lr_weight is 0."""
        block_dims = [4, 8, 16]
        block_alphas = [1.0, 2.0, 4.0]
        conv_block_dims = [2, 4, 8]
        conv_block_alphas = [0.5, 1.0, 2.0]
        block_lr_weight = [1.0, 0.0, 1.0]

        result = remove_block_dims_and_alphas(
            is_sdxl=False,
            block_dims=block_dims,
            block_alphas=block_alphas,
            conv_block_dims=conv_block_dims,
            conv_block_alphas=conv_block_alphas,
            block_lr_weight=block_lr_weight,
        )

        assert result[0] == [4, 0, 16]
        assert result[2] == [2, 0, 8]  # Conv dims also zeroed


class TestGetBlockIndex:
    """Tests for get_block_index function."""

    def test_sd_down_blocks_resnets(self):
        """SD1.5 down_blocks resnets should map to correct indices."""
        # down_blocks_0_resnets_0 -> block 1 (index = 3*0 + 0 + 1 = 1)
        result = get_block_index("lora_unet_down_blocks_0_resnets_0_conv1", is_sdxl=False)
        assert result == 1

        # down_blocks_2_resnets_1 -> block 7 (index = 3*2 + 1 + 1 = 8)
        result = get_block_index("lora_unet_down_blocks_2_resnets_1_conv1", is_sdxl=False)
        assert result == 8

    def test_sd_up_blocks(self):
        """SD1.5 up_blocks should map to indices after down+mid."""
        # up_blocks_0_resnets_0 -> block NUM_OF_BLOCKS + 1 + 0 = 13
        result = get_block_index("lora_unet_up_blocks_0_resnets_0_conv1", is_sdxl=False)
        assert result == 13

    def test_sd_mid_block(self):
        """SD1.5 mid_block should map to index 12."""
        result = get_block_index("lora_unet_mid_block_attentions_0", is_sdxl=False)
        assert result == LoRAAdapter.NUM_OF_BLOCKS  # 12

    def test_sdxl_input_blocks(self):
        """SDXL input_blocks should map to 1-9."""
        result = get_block_index("lora_unet_input_blocks_0_something", is_sdxl=True)
        assert result == 1

        result = get_block_index("lora_unet_input_blocks_5_something", is_sdxl=True)
        assert result == 6

    def test_sdxl_middle_blocks(self):
        """SDXL middle_blocks should map to 10-12."""
        result = get_block_index("lora_unet_middle_block_0_something", is_sdxl=True)
        assert result == 10

        result = get_block_index("lora_unet_middle_block_2_something", is_sdxl=True)
        assert result == 12

    def test_sdxl_output_blocks(self):
        """SDXL output_blocks should map to 13-21."""
        result = get_block_index("lora_unet_output_blocks_0_something", is_sdxl=True)
        assert result == 13

        result = get_block_index("lora_unet_output_blocks_8_something", is_sdxl=True)
        assert result == 21

    def test_sdxl_out_layer(self):
        """SDXL out_ layer should map to 22."""
        result = get_block_index("lora_unet_out_something", is_sdxl=True)
        assert result == 22

    def test_sdxl_time_embed(self):
        """SDXL time_embed should map to 0."""
        result = get_block_index("lora_unet_time_embed_0", is_sdxl=True)
        assert result == 0

    def test_invalid_name_returns_minus_one(self):
        """Invalid LoRA name should return -1."""
        result = get_block_index("some_invalid_lora_name", is_sdxl=False)
        assert result == -1
