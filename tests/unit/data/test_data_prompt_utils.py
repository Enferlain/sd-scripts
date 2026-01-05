"""
Unit tests for library/data/prompt_utils.py

Tests prompt attention parsing and token padding utilities.
"""

import pytest

from library.data.prompt_utils import (
    parse_prompt_attention,
    pad_tokens_and_weights,
)


# =============================================================================
# parse_prompt_attention Tests
# =============================================================================


@pytest.mark.data
@pytest.mark.unit
class TestParsePromptAttention:
    """Test parse_prompt_attention function - pure string parsing logic."""

    def test_normal_text_returns_weight_1(self):
        """Test that normal text without brackets has weight 1.0."""
        result = parse_prompt_attention("normal text")

        assert result == [["normal text", 1.0]]

    def test_empty_text_returns_empty_with_weight_1(self):
        """Test that empty string returns empty string with weight 1.0."""
        result = parse_prompt_attention("")

        assert result == [["", 1.0]]

    def test_single_parentheses_increases_weight(self):
        """Test that (word) increases weight by 1.1."""
        result = parse_prompt_attention("an (important) word")

        assert len(result) == 3
        assert result[0] == ["an ", 1.0]
        assert result[1][0] == "important"
        assert abs(result[1][1] - 1.1) < 0.001  # ~1.1
        assert result[2] == [" word", 1.0]

    def test_single_square_brackets_decreases_weight(self):
        """Test that [word] decreases weight by 1/1.1."""
        result = parse_prompt_attention("[weak]")

        assert len(result) == 1
        assert result[0][0] == "weak"
        assert abs(result[0][1] - (1 / 1.1)) < 0.001  # ~0.909

    def test_explicit_weight_in_parentheses(self):
        """Test that (word:1.5) sets weight to 1.5."""
        result = parse_prompt_attention("(emphasized:1.5)")

        assert len(result) == 1
        assert result[0][0] == "emphasized"
        assert abs(result[0][1] - 1.5) < 0.001

    def test_nested_parentheses_compound_weight(self):
        """Test that nested parentheses multiply weights."""
        result = parse_prompt_attention("((double))")

        assert len(result) == 1
        assert result[0][0] == "double"
        # 1.1 * 1.1 = 1.21
        assert abs(result[0][1] - 1.21) < 0.001

    def test_escaped_brackets_are_literal(self):
        """Test that escaped brackets are treated as literal characters."""
        result = parse_prompt_attention(r"\(literal\]")

        assert result == [["(literal]", 1.0]]

    def test_unbalanced_opening_parenthesis(self):
        """Test that unbalanced opening paren still applies weight."""
        result = parse_prompt_attention("(unbalanced")

        assert len(result) == 1
        assert result[0][0] == "unbalanced"
        assert abs(result[0][1] - 1.1) < 0.001

    def test_adjacent_parentheses_merge(self):
        """Test that adjacent same-weight tokens are merged."""
        result = parse_prompt_attention("(unnecessary)(parens)")

        # Both have weight 1.1, should be merged
        assert len(result) == 1
        assert result[0][0] == "unnecessaryparens"
        assert abs(result[0][1] - 1.1) < 0.001

    def test_complex_mixed_syntax(self):
        """Test complex prompt with mixed syntax from docstring example."""
        result = parse_prompt_attention("a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).")

        # Expected from docstring:
        # [['a ', 1.0], ['house', 1.573...], [' ', 1.1], ['on', 1.0],
        #  [' a ', 1.1], ['hill', 0.55], [', sun, ', 1.1], ['sky', 1.464...], ['.', 1.1]]

        assert result[0] == ["a ", 1.0]
        assert result[1][0] == "house"
        assert abs(result[1][1] - 1.573) < 0.01  # 1.3 * 1.1 * 1.1

    def test_weight_zero(self):
        """Test that (word:0) sets weight to 0."""
        result = parse_prompt_attention("(invisible:0)")

        assert result[0][0] == "invisible"
        assert result[0][1] == 0.0

    def test_fractional_weight(self):
        """Test fractional weights like 0.5."""
        result = parse_prompt_attention("(half:0.5)")

        assert result[0][0] == "half"
        assert abs(result[0][1] - 0.5) < 0.001


# =============================================================================
# pad_tokens_and_weights Tests
# =============================================================================


@pytest.mark.data
@pytest.mark.unit
class TestPadTokensAndWeights:
    """Test pad_tokens_and_weights function - token padding logic."""

    def test_pads_tokens_to_max_length(self):
        """Test that tokens are padded to max_length with BOS/EOS."""
        tokens = [[101, 102, 103]]  # 3 tokens
        weights = [[1.0, 1.0, 1.0]]
        bos, eos = 1, 2
        max_length = 10

        result_tokens, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True)

        # Should be: [BOS, 101, 102, 103, EOS, EOS, EOS, EOS, EOS, EOS]
        assert len(result_tokens[0]) == max_length
        assert result_tokens[0][0] == bos  # starts with BOS
        assert result_tokens[0][1:4] == [101, 102, 103]  # original tokens
        assert result_tokens[0][4:] == [eos] * 6  # rest is EOS

    def test_pads_weights_with_1s(self):
        """Test that weights are padded with 1.0."""
        tokens = [[101, 102]]
        weights = [[1.5, 0.8]]
        bos, eos = 1, 2
        max_length = 6

        result_tokens, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True)

        # Should be: [1.0, 1.5, 0.8, 1.0, 1.0, 1.0]
        assert len(result_weights[0]) == max_length
        assert result_weights[0][0] == 1.0  # BOS weight
        assert result_weights[0][1:3] == [1.5, 0.8]  # original weights
        assert all(w == 1.0 for w in result_weights[0][3:])  # padding

    def test_empty_tokens_get_padded(self):
        """Test that empty token list gets proper padding."""
        tokens = [[]]
        weights = [[]]
        bos, eos = 1, 2
        max_length = 5

        result_tokens, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True)

        assert result_tokens[0][0] == bos
        assert result_tokens[0][1:] == [eos] * 4

    def test_multiple_sequences(self):
        """Test padding of multiple sequences."""
        tokens = [[101], [201, 202]]
        weights = [[1.0], [0.5, 0.5]]
        bos, eos = 1, 2
        max_length = 5

        result_tokens, result_weights = pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True)

        assert len(result_tokens) == 2
        assert len(result_tokens[0]) == 5
        assert len(result_tokens[1]) == 5
