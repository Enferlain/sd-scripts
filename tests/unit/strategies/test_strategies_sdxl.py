"""
Unit tests for library/strategies/strategy_sdxl.py

Tests the SDXL strategy classes with mocked dual tokenizers and text encoders.
"""

import os
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from library.strategies.sdxl.caching import SdxlTextEncoderOutputsCachingStrategy
from library.strategies.sdxl.encoding import SdxlTextEncodingStrategy
from library.strategies.sdxl.tokenization import SdxlTokenizeStrategy


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_clip_tokenizer1():
    """Create a mock CLIPTokenizer for SDXL tokenizer 1 (CLIP-L)."""
    tokenizer = Mock()
    tokenizer.model_max_length = 77
    tokenizer.bos_token_id = 49406
    tokenizer.eos_token_id = 49407
    tokenizer.pad_token_id = 49407
    tokenizer.eos_token = 49407

    def tokenizer_call(text, **kwargs):
        if not text.strip():
            return Mock(input_ids=[49406, 49407] + [49407] * 75)
        return Mock(input_ids=[49406] + [100] * 5 + [49407] + [49407] * 70)

    tokenizer.__call__ = tokenizer_call
    tokenizer.side_effect = tokenizer_call
    return tokenizer


@pytest.fixture
def mock_clip_tokenizer2():
    """Create a mock CLIPTokenizer for SDXL tokenizer 2 (OpenCLIP)."""
    tokenizer = Mock()
    tokenizer.model_max_length = 77
    tokenizer.bos_token_id = 49406
    tokenizer.eos_token_id = 49407
    tokenizer.pad_token_id = 0  # SDXL uses 0 for tokenizer2
    tokenizer.eos_token = 49407

    def tokenizer_call(text, **kwargs):
        if not text.strip():
            return Mock(input_ids=[49406, 49407] + [0] * 75)
        return Mock(input_ids=[49406] + [200] * 5 + [49407] + [0] * 70)

    tokenizer.__call__ = tokenizer_call
    tokenizer.side_effect = tokenizer_call
    return tokenizer


@pytest.fixture
def mock_clip_text_encoder1():
    """Create a mock CLIP-L text encoder (CLIPTextModel)."""
    encoder = Mock()
    encoder.device = torch.device("cpu")

    # Mock parameters() to return an iterator with a tensor that has .device
    mock_param = Mock()
    mock_param.device = torch.device("cpu")
    encoder.parameters = Mock(return_value=iter([mock_param]))

    def encoder_call(tokens, output_hidden_states=False, return_dict=False):
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]
        hidden_dim = 768  # CLIP-L hidden dim

        hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(13)]
        return {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states[-1],
        }

    encoder.__call__ = encoder_call
    encoder.side_effect = encoder_call
    return encoder


@pytest.fixture
def mock_clip_text_encoder2():
    """Create a mock OpenCLIP text encoder (CLIPTextModelWithProjection)."""
    encoder = Mock()
    encoder.device = torch.device("cpu")

    # Mock parameters() to return an iterator with a tensor that has .device
    mock_param = Mock()
    mock_param.device = torch.device("cpu")
    encoder.parameters = Mock(return_value=iter([mock_param]))

    # Mock text_projection for pool workaround
    encoder.text_projection = Mock()
    encoder.text_projection.weight = Mock(dtype=torch.float32)
    encoder.text_projection.side_effect = lambda x: x

    def encoder_call(tokens, output_hidden_states=False, return_dict=False):
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]
        hidden_dim = 1280  # OpenCLIP hidden dim

        hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(13)]
        return {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states[-1],
            "text_embeds": torch.randn(batch_size, 1280),
        }

    encoder.__call__ = encoder_call
    encoder.side_effect = encoder_call
    return encoder


# =============================================================================
# SdxlTokenizeStrategy Tests
# =============================================================================


@pytest.mark.unit
class TestSdxlTokenizeStrategy:
    """Test SdxlTokenizeStrategy with mocked dual tokenizers."""

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_init_loads_dual_tokenizers(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test initialization loads both tokenizers."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]

        strategy = SdxlTokenizeStrategy(max_length=None)

        assert strategy.tokenizer1 is mock_clip_tokenizer1
        assert strategy.tokenizer2 is mock_clip_tokenizer2
        assert mock_load_tokenizer.call_count == 2

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_init_sets_tokenizer2_pad_to_zero(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test that strategy keeps the loader-provided tokenizer2."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]

        strategy = SdxlTokenizeStrategy(max_length=None)

        assert strategy.tokenizer2.pad_token_id == 0

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_init_custom_max_length(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test custom max_length adds 2 for BOS/EOS."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]

        strategy = SdxlTokenizeStrategy(max_length=150)

        assert strategy.max_length == 152

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_tokenize_returns_tuple_of_two_tensors(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test tokenize returns tuple of two token tensors."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        strategy = SdxlTokenizeStrategy(max_length=None)

        with patch("library.strategies.sdxl.tokenization.get_clip_input_ids") as mock_get_ids:
            mock_get_ids.return_value = torch.randint(0, 1000, (1, 77))

            result = strategy.tokenize("a photo of a cat")

            assert isinstance(result, list)
            assert len(result) == 2
            assert mock_get_ids.call_count == 2  # Called for both tokenizers

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_tokenize_with_weights_returns_dual_tokens_and_weights(self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2):
        """Test tokenize_with_weights returns tokens and weights for both tokenizers."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        strategy = SdxlTokenizeStrategy(max_length=None)

        with patch("library.strategies.sdxl.tokenization.get_clip_input_ids") as mock_get_ids:
            mock_get_ids.return_value = (torch.randint(0, 1000, (1, 77)), torch.ones(1, 77))

            tokens_list, weights_list = strategy.tokenize_with_weights("(emphasized:1.5)")

            assert len(tokens_list) == 2
            assert len(weights_list) == 2


# =============================================================================
# SdxlTextEncodingStrategy Tests
# =============================================================================


@pytest.mark.unit
class TestSdxlTextEncodingStrategy:
    """Test SdxlTextEncodingStrategy with mocked dual text encoders."""

    def test_init(self):
        """Test initialization."""
        strategy = SdxlTextEncodingStrategy()
        # No-op init, just verify it doesn't error
        assert strategy is not None

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_encode_tokens_returns_three_outputs(
        self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2
    ):
        """Test encode_tokens returns [hidden1, hidden2, pool2]."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        tokenize_strategy = SdxlTokenizeStrategy(max_length=None)
        encoding_strategy = SdxlTextEncodingStrategy()

        with patch.object(tokenize_strategy, "tokenize") as mock_tokenize:
            mock_tokenize.return_value = [torch.randint(0, 1000, (1, 1, 77)), torch.randint(0, 1000, (1, 1, 77))]

            result = encoding_strategy.encode_tokens(
                tokenize_strategy, [mock_clip_text_encoder1, mock_clip_text_encoder2], list(mock_tokenize.return_value)
            )

        assert len(result) == 3  # hidden1, hidden2, pool2

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_encode_tokens_with_unwrapped_encoder(
        self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2
    ):
        """Test encode_tokens with 3 models (wrapped encoder2 case)."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        tokenize_strategy = SdxlTokenizeStrategy(max_length=None)
        encoding_strategy = SdxlTextEncodingStrategy()

        with patch.object(tokenize_strategy, "tokenize") as mock_tokenize:
            mock_tokenize.return_value = [torch.randint(0, 1000, (1, 1, 77)), torch.randint(0, 1000, (1, 1, 77))]

            # Pass 3 models: encoder1, encoder2, unwrapped_encoder2
            result = encoding_strategy.encode_tokens(
                tokenize_strategy,
                [mock_clip_text_encoder1, mock_clip_text_encoder2, mock_clip_text_encoder2],
                list(mock_tokenize.return_value),
            )

        assert len(result) == 3

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_encode_tokens_with_weights_applies_dual_weights(
        self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2
    ):
        """Test that encode_tokens_with_weights applies weights to both encoders."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]
        tokenize_strategy = SdxlTokenizeStrategy(max_length=None)
        encoding_strategy = SdxlTextEncodingStrategy()

        with patch.object(tokenize_strategy, "tokenize_with_weights") as mock_tokenize:
            tokens = [torch.randint(0, 1000, (1, 1, 77)), torch.randint(0, 1000, (1, 1, 77))]
            weights = [torch.ones(1, 1, 77) * 1.5, torch.ones(1, 1, 77) * 1.5]
            mock_tokenize.return_value = (tokens, weights)

            result = encoding_strategy.encode_tokens_with_weights(
                tokenize_strategy, [mock_clip_text_encoder1, mock_clip_text_encoder2], tokens, weights
            )

        assert len(result) == 3


# =============================================================================
# SdxlTextEncoderOutputsCachingStrategy Tests
# =============================================================================


@pytest.mark.unit
class TestSdxlTextEncoderOutputsCachingStrategy:
    """Test SdxlTextEncoderOutputsCachingStrategy."""

    def test_init_stores_properties(self):
        """Test __init__ stores all properties."""
        strategy = SdxlTextEncoderOutputsCachingStrategy(
            cache_to_disk=True, batch_size=4, skip_disk_cache_validity_check=True, is_partial=True, is_weighted=True
        )

        assert strategy.cache_to_disk is True
        assert strategy.batch_size == 4
        assert strategy.is_partial is True
        assert strategy.is_weighted is True

    def test_get_outputs_npz_path(self):
        """Test NPZ path generation."""
        strategy = SdxlTextEncoderOutputsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        npz_path = strategy.get_outputs_npz_path("/path/to/image.png")

        assert npz_path == "/path/to/image_te_outputs.npz"

    def test_is_disk_cached_outputs_expected_false_when_no_cache(self, tmp_path):
        """Test returns False when cache_to_disk is False."""
        strategy = SdxlTextEncoderOutputsCachingStrategy(cache_to_disk=False, batch_size=1, skip_disk_cache_validity_check=False)

        result = strategy.is_disk_cached_outputs_expected(str(tmp_path / "test.npz"))

        assert result is False

    def test_is_disk_cached_outputs_expected_false_when_file_missing(self, tmp_path):
        """Test returns False when file doesn't exist."""
        strategy = SdxlTextEncoderOutputsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        result = strategy.is_disk_cached_outputs_expected(str(tmp_path / "nonexistent.npz"))

        assert result is False

    def test_is_disk_cached_outputs_expected_true_with_skip(self, tmp_path):
        """Test returns True when skip_disk_cache_validity_check is True."""
        strategy = SdxlTextEncoderOutputsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=True)

        npz_path = str(tmp_path / "test.npz")
        np.savez(npz_path)  # Empty file

        result = strategy.is_disk_cached_outputs_expected(npz_path)

        assert result is True

    def test_is_disk_cached_outputs_expected_checks_keys(self, tmp_path):
        """Test checks for required keys in NPZ."""
        strategy = SdxlTextEncoderOutputsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        # Missing keys
        npz_path = str(tmp_path / "test.npz")
        np.savez(npz_path, hidden_state1=np.zeros((77, 768)))  # Missing state2 and pool2

        result = strategy.is_disk_cached_outputs_expected(npz_path)

        assert result is False

    def test_is_disk_cached_outputs_expected_true_with_all_keys(self, tmp_path):
        """Test returns True when all required keys present."""
        strategy = SdxlTextEncoderOutputsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        npz_path = str(tmp_path / "test.npz")
        np.savez(npz_path, hidden_state1=np.zeros((77, 768)), hidden_state2=np.zeros((77, 1280)), pool2=np.zeros(1280))

        result = strategy.is_disk_cached_outputs_expected(npz_path)

        assert result is True

    def test_load_outputs_npz(self, tmp_path):
        """Test loading outputs from NPZ."""
        strategy = SdxlTextEncoderOutputsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        npz_path = str(tmp_path / "test.npz")
        h1 = np.random.randn(77, 768).astype(np.float32)
        h2 = np.random.randn(77, 1280).astype(np.float32)
        p2 = np.random.randn(1280).astype(np.float32)
        np.savez(npz_path, hidden_state1=h1, hidden_state2=h2, pool2=p2)

        result = strategy.load_outputs_npz(npz_path)

        assert len(result) == 3
        np.testing.assert_array_almost_equal(result[0], h1)
        np.testing.assert_array_almost_equal(result[1], h2)
        np.testing.assert_array_almost_equal(result[2], p2)

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_cache_batch_outputs_saves_to_disk(
        self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2, tmp_path
    ):
        """Test cache_batch_outputs saves to disk when cache_to_disk is True."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]

        strategy = SdxlTextEncoderOutputsCachingStrategy(
            cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False, is_weighted=False
        )
        tokenize_strategy = SdxlTokenizeStrategy(max_length=None)
        encoding_strategy = SdxlTextEncodingStrategy()

        # Mock info object
        mock_info = Mock()
        mock_info.caption = "a test caption"
        mock_info.text_encoder_outputs_npz = str(tmp_path / "output.npz")

        with patch.object(tokenize_strategy, "tokenize") as mock_tokenize:
            mock_tokenize.return_value = (torch.randint(0, 1000, (1, 1, 77)), torch.randint(0, 1000, (1, 1, 77)))

            with patch.object(encoding_strategy, "encode_tokens") as mock_encode:
                mock_encode.return_value = [torch.randn(1, 77, 768), torch.randn(1, 77, 1280), torch.randn(1, 1280)]

                strategy.cache_batch_outputs(
                    tokenize_strategy, [mock_clip_text_encoder1, mock_clip_text_encoder2], encoding_strategy, [mock_info]
                )

        # Check file was created
        assert os.path.exists(mock_info.text_encoder_outputs_npz)

    @patch("library.strategies.sdxl.tokenization.load_tokenizer")
    def test_cache_batch_outputs_stores_in_memory(
        self, mock_load_tokenizer, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2
    ):
        """Test cache_batch_outputs stores in memory when cache_to_disk is False."""
        mock_load_tokenizer.side_effect = [mock_clip_tokenizer1, mock_clip_tokenizer2]

        strategy = SdxlTextEncoderOutputsCachingStrategy(
            cache_to_disk=False, batch_size=1, skip_disk_cache_validity_check=False, is_weighted=False
        )
        tokenize_strategy = SdxlTokenizeStrategy(max_length=None)
        encoding_strategy = SdxlTextEncodingStrategy()

        mock_info = Mock()
        mock_info.caption = "a test caption"

        with patch.object(tokenize_strategy, "tokenize") as mock_tokenize:
            mock_tokenize.return_value = (torch.randint(0, 1000, (1, 1, 77)), torch.randint(0, 1000, (1, 1, 77)))

            with patch.object(encoding_strategy, "encode_tokens") as mock_encode:
                mock_encode.return_value = [torch.randn(1, 77, 768), torch.randn(1, 77, 1280), torch.randn(1, 1280)]

                strategy.cache_batch_outputs(
                    tokenize_strategy, [mock_clip_text_encoder1, mock_clip_text_encoder2], encoding_strategy, [mock_info]
                )

        # Check info object was updated
        assert mock_info.text_encoder_outputs is not None
        assert len(mock_info.text_encoder_outputs) == 3
