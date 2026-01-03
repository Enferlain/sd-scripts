"""
Unit tests for the pure parsing functions in library/pipelines/lpw_stable_diffusion.py
and library/pipelines/sdxl_lpw_stable_diffusion.py.

Tests prompt attention parsing and token padding utilities.
Both SD and SDXL versions share identical implementations for these functions.
"""

import pytest

from library.pipelines.lpw_stable_diffusion import (
    parse_prompt_attention,
    pad_tokens_and_weights,
)
from library.pipelines.sdxl_lpw_stable_diffusion import (
    parse_prompt_attention as sdxl_parse_prompt_attention,
    pad_tokens_and_weights as sdxl_pad_tokens_and_weights,
)


class TestParsePromptAttention:
    """Tests for parse_prompt_attention function."""

    def test_plain_text_returns_weight_one(self):
        """Plain text without any brackets should have weight 1.0."""
        result = parse_prompt_attention("normal text")
        assert result == [["normal text", 1.0]]

    def test_empty_string_returns_empty_with_weight_one(self):
        """Empty string should return empty string with weight 1.0."""
        result = parse_prompt_attention("")
        assert result == [["", 1.0]]

    def test_single_round_brackets_increase_weight(self):
        """Single round brackets should increase weight by 1.1."""
        result = parse_prompt_attention("an (important) word")
        assert len(result) == 3
        assert result[0] == ["an ", 1.0]
        assert result[1][0] == "important"
        assert result[1][1] == pytest.approx(1.1, abs=0.001)
        assert result[2] == [" word", 1.0]

    def test_explicit_weight_in_brackets(self):
        """Explicit weight specification (text:weight) should use that weight."""
        result = parse_prompt_attention("(emphasized:1.5)")
        assert len(result) == 1
        assert result[0][0] == "emphasized"
        assert result[0][1] == pytest.approx(1.5, abs=0.001)

    def test_square_brackets_decrease_weight(self):
        """Square brackets should decrease weight by 1/1.1."""
        result = parse_prompt_attention("[deemphasized]")
        assert len(result) == 1
        assert result[0][0] == "deemphasized"
        assert result[0][1] == pytest.approx(1 / 1.1, abs=0.001)

    def test_nested_round_brackets(self):
        """Nested round brackets should multiply weights."""
        result = parse_prompt_attention("((double))")
        assert len(result) == 1
        assert result[0][0] == "double"
        # 1.1 * 1.1 = 1.21
        assert result[0][1] == pytest.approx(1.21, abs=0.001)

    def test_escaped_brackets(self):
        """Escaped brackets should be treated as literal characters."""
        result = parse_prompt_attention(r"\(literal\)")
        assert len(result) == 1
        assert result[0][0] == "(literal)"
        assert result[0][1] == 1.0

    def test_escaped_square_brackets(self):
        """Escaped square brackets should be treated as literal characters."""
        result = parse_prompt_attention(r"\[literal\]")
        assert len(result) == 1
        assert result[0][0] == "[literal]"
        assert result[0][1] == 1.0

    def test_unbalanced_opening_bracket(self):
        """Unbalanced opening bracket should still apply weight."""
        result = parse_prompt_attention("(unbalanced")
        assert len(result) == 1
        assert result[0][0] == "unbalanced"
        assert result[0][1] == pytest.approx(1.1, abs=0.001)

    def test_complex_nested_example(self):
        """Test complex nested example from docstring."""
        result = parse_prompt_attention("a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).")

        # Find the "house" entry
        house_entry = next((r for r in result if "house" in r[0]), None)
        assert house_entry is not None
        # house:1.3 nested twice more = 1.3 * 1.1 * 1.1 ≈ 1.573
        assert house_entry[1] == pytest.approx(1.573, abs=0.01)

        # Find the "on" entry
        on_entry = next((r for r in result if r[0] == "on"), None)
        assert on_entry is not None
        # [on] decreases weight
        assert on_entry[1] == pytest.approx(1.0, abs=0.01)  # Inside () but also []

        # Find the "hill" entry
        hill_entry = next((r for r in result if "hill" in r[0]), None)
        assert hill_entry is not None
        # hill:0.5 nested once more = 0.5 * 1.1 = 0.55
        assert hill_entry[1] == pytest.approx(0.55, abs=0.01)

        # Find the "sky" entry
        sky_entry = next((r for r in result if "sky" in r[0]), None)
        assert sky_entry is not None
        # ((( ))) = 1.1^3 ≈ 1.331, but inside (), so 1.1^4 ≈ 1.4641
        assert sky_entry[1] == pytest.approx(1.4641, abs=0.01)

    def test_adjacent_brackets_merge(self):
        """Adjacent items with same weight should be merged."""
        result = parse_prompt_attention("(a)(b)")
        # Both have weight 1.1, should be merged
        assert len(result) == 1
        assert result[0][0] == "ab"
        assert result[0][1] == pytest.approx(1.1, abs=0.001)

    def test_mixed_weights_not_merged(self):
        """Items with different weights should not be merged."""
        result = parse_prompt_attention("(a) [b]")
        # "a" has weight 1.1, " " has weight 1.0, "b" has weight 1/1.1
        assert len(result) >= 2  # At least "a" and "b" should be separate

    def test_decimal_weight(self):
        """Decimal weights should be parsed correctly."""
        result = parse_prompt_attention("(test:0.75)")
        assert result[0][1] == pytest.approx(0.75, abs=0.001)

    def test_high_weight(self):
        """High weights should be parsed correctly."""
        result = parse_prompt_attention("(very strong:5.0)")
        assert result[0][1] == pytest.approx(5.0, abs=0.001)

    def test_zero_weight(self):
        """Zero weight should be parsed correctly."""
        result = parse_prompt_attention("(nothing:0.0)")
        assert result[0][1] == pytest.approx(0.0, abs=0.001)


class TestPadTokensAndWeights:
    """Tests for pad_tokens_and_weights function."""

    def test_adds_bos_and_eos(self):
        """Should add BOS at start and pad with EOS to max_length."""
        tokens = [[1, 2, 3]]
        weights = [[1.0, 1.0, 1.0]]
        bos, eos = 49406, 49407  # Typical CLIP tokenizer values
        max_length = 77

        result_tokens, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos)

        assert result_tokens[0][0] == bos
        assert len(result_tokens[0]) == max_length
        # All remaining positions should be EOS
        assert result_tokens[0][-1] == eos

    def test_weight_length_matches_token_length(self):
        """Weights should be padded to match token length."""
        tokens = [[1, 2, 3]]
        weights = [[0.5, 1.0, 1.5]]
        bos, eos = 49406, 49407
        max_length = 77

        result_tokens, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos)

        assert len(result_tokens[0]) == len(result_weights[0])

    def test_bos_weight_is_one(self):
        """BOS token weight should be 1.0."""
        tokens = [[1, 2]]
        weights = [[0.5, 0.5]]
        bos, eos = 49406, 49407
        max_length = 77

        _, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos)

        assert result_weights[0][0] == 1.0

    def test_padding_weight_is_one(self):
        """Padding weights should be 1.0."""
        tokens = [[1, 2, 3]]
        weights = [[0.5, 0.5, 0.5]]
        bos, eos = 49406, 49407
        max_length = 77

        _, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos)

        # After BOS and 3 original tokens, rest should be 1.0
        for i in range(4, max_length):
            assert result_weights[0][i] == 1.0

    def test_preserves_original_weights(self):
        """Original token weights should be preserved."""
        tokens = [[1, 2, 3]]
        weights = [[0.5, 1.0, 1.5]]
        bos, eos = 49406, 49407
        max_length = 77

        _, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos)

        # Positions 1, 2, 3 should have original weights
        assert result_weights[0][1] == 0.5
        assert result_weights[0][2] == 1.0
        assert result_weights[0][3] == 1.5

    def test_multiple_prompts(self):
        """Should handle multiple prompts in batch."""
        tokens = [[1, 2, 3], [4, 5]]
        weights = [[1.0, 1.0, 1.0], [0.5, 0.5]]
        bos, eos = 49406, 49407
        max_length = 77

        result_tokens, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos)

        assert len(result_tokens) == 2
        assert len(result_weights) == 2
        assert len(result_tokens[0]) == max_length
        assert len(result_tokens[1]) == max_length

    def test_empty_tokens(self):
        """Should handle empty token list."""
        tokens = [[]]
        weights = [[]]
        bos, eos = 49406, 49407
        max_length = 77

        result_tokens, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos)

        assert len(result_tokens[0]) == max_length
        assert result_tokens[0][0] == bos

    def test_chunk_length_parameter(self):
        """Should respect chunk_length parameter."""
        tokens = [[1, 2, 3]]
        weights = [[1.0, 1.0, 1.0]]
        bos, eos = 49406, 49407
        max_length = 152  # 2 chunks of 77
        chunk_length = 77

        result_tokens, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos, chunk_length=chunk_length)

        assert len(result_tokens[0]) == max_length


class TestSdxlCrossValidation:
    """Tests to verify SD and SDXL LPW pipelines use identical implementations."""

    def test_parse_prompt_attention_identical_for_simple_text(self):
        """SD and SDXL parse_prompt_attention should produce identical results."""
        test_prompts = [
            "normal text",
            "(emphasized)",
            "[deemphasized]",
            "(custom:1.5)",
            "a (((nested))) prompt",
        ]
        for prompt in test_prompts:
            sd_result = parse_prompt_attention(prompt)
            sdxl_result = sdxl_parse_prompt_attention(prompt)
            assert sd_result == sdxl_result, f"Mismatch for prompt: {prompt}"

    def test_parse_prompt_attention_identical_for_complex_text(self):
        """Both versions should handle complex nested prompts identically."""
        prompt = "a (((house:1.3)) [on] a (hill:0.5), sun, (((sky)))."
        sd_result = parse_prompt_attention(prompt)
        sdxl_result = sdxl_parse_prompt_attention(prompt)
        assert sd_result == sdxl_result

    def test_sdxl_pad_tokens_with_separate_pad(self):
        """SDXL version accepts separate pad token (vs SD which uses eos for padding)."""
        tokens = [[1, 2, 3]]
        weights = [[1.0, 1.0, 1.0]]
        bos, eos, pad = 49406, 49407, 0  # SDXL uses separate pad token
        max_length = 77

        result_tokens, result_weights = sdxl_pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad)

        assert result_tokens[0][0] == bos
        assert len(result_tokens[0]) == max_length
        # SDXL has explicit EOS after tokens, then PAD
        assert result_tokens[0][4] == eos  # EOS after 3 tokens + BOS

    def test_sdxl_preserves_weights(self):
        """SDXL version should also preserve original weights."""
        tokens = [[1, 2, 3]]
        weights = [[0.5, 1.0, 1.5]]
        bos, eos, pad = 49406, 49407, 0
        max_length = 77

        _, result_weights = sdxl_pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad)

        # Positions 1, 2, 3 should have original weights
        assert result_weights[0][1] == 0.5
        assert result_weights[0][2] == 1.0
        assert result_weights[0][3] == 1.5
