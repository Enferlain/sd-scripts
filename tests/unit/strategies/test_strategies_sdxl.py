"""
Unit tests for library/strategies/strategy_sdxl.py

Tests the SDXL strategy classes with mocked dual tokenizers and text encoders.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from library.strategies.sdxl.encoding import SdxlTextEncodingStrategy
from library.strategies.sdxl.tokenization import SdxlTokenizeStrategy
from library.strategies.sdxl.training import SdxlTrainingStrategy


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




@pytest.mark.unit
class TestSdxlTrainingStrategyNewPipeline:
    """Tests for the SDXL training strategy token/TE helpers."""

    def test_tokenize_captions_returns_dual_chunked_tensors(self, mock_clip_tokenizer1, mock_clip_tokenizer2):
        strategy = SdxlTrainingStrategy()

        result = strategy.tokenize_captions([mock_clip_tokenizer1, mock_clip_tokenizer2], ["caption 1", "caption 2"], 75)

        assert len(result) == 2
        assert all(isinstance(tokens, torch.Tensor) for tokens in result)
        assert all(tokens.shape[0] == 2 for tokens in result)
        assert all(tokens.ndim == 3 for tokens in result)

    def test_encode_te_outputs_in_memory_returns_cpu_te_outputs(
        self, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2
    ):
        strategy = SdxlTrainingStrategy()

        with patch("library.strategies.sdxl.training.encode_input_ids_sdxl") as mock_encode:
            mock_encode.return_value = [
                torch.randn(1, 77, 768),
                torch.randn(1, 77, 1280),
                torch.randn(1, 1280),
            ]

            result = strategy.encode_te_outputs_in_memory(
                text_encoders=[mock_clip_text_encoder1, mock_clip_text_encoder2],
                tokenizers=[mock_clip_tokenizer1, mock_clip_tokenizer2],
                caption="a photo of a cat",
                max_token_length=75,
                device=torch.device("cpu"),
            )

        assert set(result) == {"hidden_state1", "hidden_state2", "pool2"}
        assert all(t.device.type == "cpu" for t in result.values())

    def test_get_text_cond_fallback_tokenizes_captions_with_strategy_helper(
        self, mock_clip_tokenizer1, mock_clip_tokenizer2, mock_clip_text_encoder1, mock_clip_text_encoder2
    ):
        strategy = SdxlTrainingStrategy()
        cfg = Mock()
        cfg.training.max_token_length = 75
        cfg.performance.precision.full_fp16 = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")
        accelerator.unwrap_model.return_value = mock_clip_text_encoder2
        batch = {"captions": ["caption 1", "caption 2"]}

        token_tensors = [torch.randint(0, 1000, (2, 1, 77)), torch.randint(0, 1000, (2, 1, 77))]
        encoded_outputs = [torch.randn(2, 77, 768), torch.randn(2, 77, 1280), torch.randn(2, 1280)]

        with patch.object(strategy, "tokenize_captions", return_value=token_tensors) as mock_tokenize, patch(
            "library.strategies.sdxl.training.encode_input_ids_sdxl",
            return_value=encoded_outputs,
        ) as mock_encode:
            result = strategy._get_text_cond(
                cfg=cfg,
                accelerator=accelerator,
                batch=batch,
                tokenizers=[mock_clip_tokenizer1, mock_clip_tokenizer2],
                text_encoders=[mock_clip_text_encoder1, mock_clip_text_encoder2],
                weight_dtype=torch.float32,
            )

        mock_tokenize.assert_called_once_with([mock_clip_tokenizer1, mock_clip_tokenizer2], batch["captions"], 75)
        mock_encode.assert_called_once()
        assert len(result) == 3
