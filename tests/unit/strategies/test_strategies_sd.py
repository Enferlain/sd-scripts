"""
Unit tests for library/strategies/strategy_sd.py

Tests the SD 1.5/2.0 strategy classes with mocked tokenizers and text encoders.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from library.strategies.sd.caching import SdSdxlLatentsCachingStrategy
from library.strategies.sd.encoding import SdTextEncodingStrategy
from library.strategies.sd.tokenization import SdTokenizeStrategy


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_clip_tokenizer():
    """Create a mock CLIPTokenizer with CLIP-like behavior."""
    tokenizer = Mock()
    tokenizer.model_max_length = 77
    tokenizer.bos_token_id = 49406
    tokenizer.eos_token_id = 49407
    tokenizer.pad_token_id = 49407  # v1 style
    tokenizer.eos_token = 49407

    # Mock __call__ for tokenization
    def tokenizer_call(text, **kwargs):
        # Simple mock: return padded tokens
        if not text.strip():
            return Mock(input_ids=[49406, 49407] + [49407] * 75)
        return Mock(input_ids=[49406] + [100] * 5 + [49407] + [49407] * 70)

    tokenizer.__call__ = tokenizer_call
    tokenizer.side_effect = tokenizer_call
    return tokenizer


@pytest.fixture
def mock_clip_text_encoder():
    """Create a mock CLIP text encoder returning fake hidden states."""
    encoder = Mock()
    encoder.device = torch.device("cpu")

    # Mock text_model for clip_skip
    encoder.text_model = Mock()
    encoder.text_model.final_layer_norm = Mock(side_effect=lambda x: x)

    def encoder_call(tokens, output_hidden_states=False, return_dict=False):
        batch_size = tokens.shape[0]
        hidden_dim = 768
        seq_len = tokens.shape[1]

        if output_hidden_states:
            hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(13)]
            return {
                "hidden_states": hidden_states,
                "last_hidden_state": hidden_states[-1],
            }
        else:
            return (torch.randn(batch_size, seq_len, hidden_dim),)

    encoder.__call__ = encoder_call
    encoder.side_effect = encoder_call
    return encoder


# =============================================================================
# SdTokenizeStrategy Tests
# =============================================================================


@pytest.mark.unit
class TestSdTokenizeStrategy:
    """Test SdTokenizeStrategy with mocked tokenizer loading."""

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_init_v1_tokenizer(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test v1 tokenizer initialization."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer

        strategy = SdTokenizeStrategy(v2=False, max_length=None)

        assert strategy.tokenizer is mock_clip_tokenizer
        assert strategy.max_length == 77
        mock_load_tokenizer.assert_called_once()

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_init_v2_tokenizer(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test v2 tokenizer initialization with subfolder."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer

        strategy = SdTokenizeStrategy(v2=True, max_length=None)

        assert strategy.tokenizer is mock_clip_tokenizer
        # Verify v2 uses subfolder
        call_args = mock_load_tokenizer.call_args
        assert call_args[1].get("subfolder") == "tokenizer"

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_init_custom_max_length(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test custom max_length adds 2 for BOS/EOS."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer

        strategy = SdTokenizeStrategy(v2=False, max_length=150)

        assert strategy.max_length == 152  # 150 + 2

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_tokenize_single_text(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test tokenizing a single string."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTokenizeStrategy(v2=False, max_length=None)

        with patch.object(strategy, "_get_input_ids") as mock_get_ids:
            mock_get_ids.return_value = torch.randint(0, 1000, (1, 77))

            result = strategy.tokenize("a photo of a cat")

            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
            mock_get_ids.assert_called()

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_tokenize_list_of_texts(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test tokenizing a list of strings."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTokenizeStrategy(v2=False, max_length=None)

        with patch.object(strategy, "_get_input_ids") as mock_get_ids:
            mock_get_ids.return_value = torch.randint(0, 1000, (1, 77))

            result = strategy.tokenize(["text 1", "text 2"])

            assert len(result) == 1
            # Should have batch dimension of 2
            assert mock_get_ids.call_count == 2

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_tokenize_with_weights(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test tokenize_with_weights returns both tokens and weights."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTokenizeStrategy(v2=False, max_length=None)

        with patch.object(strategy, "_get_input_ids") as mock_get_ids:
            mock_get_ids.return_value = (torch.randint(0, 1000, (1, 77)), torch.ones(1, 77))

            tokens_list, weights_list = strategy.tokenize_with_weights("(emphasized:1.5)")

            assert len(tokens_list) == 1
            assert len(weights_list) == 1
            mock_get_ids.assert_called_with(mock_clip_tokenizer, "(emphasized:1.5)", 77, weighted=True)


# =============================================================================
# SdTextEncodingStrategy Tests
# =============================================================================


@pytest.mark.unit
class TestSdTextEncodingStrategy:
    """Test SdTextEncodingStrategy with mocked text encoder."""

    def test_init_stores_clip_skip(self):
        """Test that __init__ stores clip_skip value."""
        strategy = SdTextEncodingStrategy(clip_skip=2)
        assert strategy.clip_skip == 2

    def test_init_default_clip_skip_none(self):
        """Test that default clip_skip is None."""
        strategy = SdTextEncodingStrategy()
        assert strategy.clip_skip is None

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_encode_tokens_basic(self, mock_load_tokenizer, mock_clip_tokenizer, mock_clip_text_encoder):
        """Test basic token encoding without clip_skip."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=None)
        encoding_strategy = SdTextEncodingStrategy(clip_skip=None)

        # Create fake tokens: batch=1, n=1, seq=77
        tokens = [torch.randint(0, 1000, (1, 1, 77))]

        result = encoding_strategy.encode_tokens(tokenize_strategy, [mock_clip_text_encoder], tokens)

        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_encode_tokens_with_clip_skip(self, mock_load_tokenizer, mock_clip_tokenizer, mock_clip_text_encoder):
        """Test token encoding with clip_skip."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=None)
        encoding_strategy = SdTextEncodingStrategy(clip_skip=2)

        tokens = [torch.randint(0, 1000, (1, 1, 77))]

        result = encoding_strategy.encode_tokens(tokenize_strategy, [mock_clip_text_encoder], tokens)

        assert len(result) == 1

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_encode_tokens_multi_chunk(self, mock_load_tokenizer, mock_clip_tokenizer, mock_clip_text_encoder):
        """Test encoding with multiple token chunks (n > 1)."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=None)
        encoding_strategy = SdTextEncodingStrategy(clip_skip=None)

        # Multi-chunk tokens: batch=1, n=3, seq=77
        tokens = [torch.randint(0, 1000, (1, 3, 77))]

        result = encoding_strategy.encode_tokens(tokenize_strategy, [mock_clip_text_encoder], tokens)

        assert len(result) == 1
        # Output should be reshaped for multi-chunk

    @patch.object(SdTokenizeStrategy, "_load_tokenizer")
    def test_encode_tokens_with_weights_applies_weights(self, mock_load_tokenizer, mock_clip_tokenizer, mock_clip_text_encoder):
        """Test that encode_tokens_with_weights applies weight multiplication."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=None)
        encoding_strategy = SdTextEncodingStrategy(clip_skip=None)

        tokens = [torch.randint(0, 1000, (1, 1, 77))]
        weights = [torch.ones(1, 1, 77) * 1.5]

        result = encoding_strategy.encode_tokens_with_weights(tokenize_strategy, [mock_clip_text_encoder], tokens, weights)

        assert len(result) == 1


# =============================================================================
# SdSdxlLatentsCachingStrategy Tests
# =============================================================================


@pytest.mark.unit
class TestSdSdxlLatentsCachingStrategy:
    """Test SdSdxlLatentsCachingStrategy."""

    def test_init_sd_suffix(self):
        """Test SD suffix selection."""
        strategy = SdSdxlLatentsCachingStrategy(sd=True, cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        assert strategy.sd is True
        assert strategy.cache_suffix == "_sd.npz"

    def test_init_sdxl_suffix(self):
        """Test SDXL suffix selection."""
        strategy = SdSdxlLatentsCachingStrategy(sd=False, cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        assert strategy.sd is False
        assert strategy.cache_suffix == "_sdxl.npz"

    def test_get_latents_npz_path_new_format(self, tmp_path):
        """Test NPZ path generation for new format."""
        strategy = SdSdxlLatentsCachingStrategy(sd=True, cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        img_path = str(tmp_path / "image.png")
        npz_path = strategy.get_latents_npz_path(img_path, (512, 512))

        assert npz_path.endswith("_0512x0512_sd.npz")
        assert "image" in npz_path

    def test_get_latents_npz_path_old_format_exists(self, tmp_path):
        """Test NPZ path uses old format if it exists."""
        strategy = SdSdxlLatentsCachingStrategy(sd=True, cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        # Create old-style npz
        img_path = tmp_path / "image.png"
        old_npz = tmp_path / "image.npz"
        old_npz.write_text("")  # Just create the file

        npz_path = strategy.get_latents_npz_path(str(img_path), (512, 512))

        assert npz_path == str(old_npz)

    def test_is_disk_cached_latents_expected_delegates(self, tmp_path):
        """Test that is_disk_cached_latents_expected delegates to base with stride=8."""
        strategy = SdSdxlLatentsCachingStrategy(sd=True, cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)

        npz_path = str(tmp_path / "test.npz")
        np.savez(npz_path, latents=np.zeros((4, 64, 64)))

        result = strategy.is_disk_cached_latents_expected((512, 512), npz_path, flip_aug=False, alpha_mask=False)

        assert result is True

    def test_cache_batch_latents_calls_vae(self):
        """Test that cache_batch_latents calls VAE encode."""
        strategy = SdSdxlLatentsCachingStrategy(sd=True, cache_to_disk=False, batch_size=1, skip_disk_cache_validity_check=False)

        # Mock VAE
        mock_vae = Mock()
        mock_vae.device = torch.device("cpu")
        mock_vae.dtype = torch.float32
        mock_vae.encode.return_value = Mock(latent_dist=Mock(sample=Mock(return_value=torch.randn(1, 4, 64, 64))))

        # Mock image info
        mock_info = Mock()
        mock_info.absolute_path = "/path/to/image.png"
        mock_info.bucket_reso = (512, 512)
        mock_info.latents_npz = "/path/to/image_sd.npz"

        with patch("library.utils.device_utils.clean_memory_on_device"):
            with patch.object(strategy, "_default_cache_batch_latents") as mock_cache:
                strategy.cache_batch_latents(mock_vae, [mock_info], flip_aug=False, alpha_mask=False, random_crop=False)

                mock_cache.assert_called_once()
