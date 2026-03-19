"""
Unit tests for library/strategies/strategy_sd.py

Tests the SD 1.5/2.0 strategy classes with mocked tokenizers and text encoders.
"""

from contextlib import nullcontext

import pytest
import torch
from unittest.mock import Mock, patch

from library.strategies.sd.caching import SdLatentsPipelineStrategy, SdTextEncoderPipelineStrategy
from library.strategies.sd.encoding import SdTextEncodingStrategy
from library.strategies.sd.tokenization import SdTokenizeStrategy
from library.strategies.sd.training import SdTrainingStrategy


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


@pytest.fixture
def sd_strategy_cfg():
    """Create a minimal SD strategy config mock."""
    cfg = Mock()
    cfg.model.model_type = "sd1"
    cfg.model.tokenizer_cache_dir = None
    cfg.training.max_token_length = 75
    cfg.training.clip_skip = None
    return cfg


# =============================================================================
# SdTokenizeStrategy Tests
# =============================================================================


@pytest.mark.unit
class TestSdTokenizeStrategy:
    """Test SdTokenizeStrategy with mocked tokenizer loading."""

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_init_v1_tokenizer(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test v1 tokenizer initialization."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer

        strategy = SdTokenizeStrategy(v2=False, max_length=None)

        assert strategy.tokenizer is mock_clip_tokenizer
        assert strategy.max_length == 77
        mock_load_tokenizer.assert_called_once()

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_init_v2_tokenizer(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test v2 tokenizer initialization with subfolder."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer

        strategy = SdTokenizeStrategy(v2=True, max_length=None)

        assert strategy.tokenizer is mock_clip_tokenizer
        # Verify v2 uses subfolder
        call_args = mock_load_tokenizer.call_args
        assert call_args[1].get("subfolder") == "tokenizer"

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_init_custom_max_length(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test custom max_length adds 2 for BOS/EOS."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer

        strategy = SdTokenizeStrategy(v2=False, max_length=150)

        assert strategy.max_length == 152  # 150 + 2

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_tokenize_single_text(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test tokenizing a single string."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTokenizeStrategy(v2=False, max_length=None)

        with patch("library.strategies.sd.tokenization.get_clip_input_ids") as mock_get_ids:
            mock_get_ids.return_value = torch.randint(0, 1000, (1, 77))

            result = strategy.tokenize("a photo of a cat")

            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
            mock_get_ids.assert_called()

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_tokenize_list_of_texts(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test tokenizing a list of strings."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTokenizeStrategy(v2=False, max_length=None)

        with patch("library.strategies.sd.tokenization.get_clip_input_ids") as mock_get_ids:
            mock_get_ids.return_value = torch.randint(0, 1000, (1, 77))

            result = strategy.tokenize(["text 1", "text 2"])

            assert len(result) == 1
            # Should have batch dimension of 2
            assert mock_get_ids.call_count == 2

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_tokenize_with_weights(self, mock_load_tokenizer, mock_clip_tokenizer):
        """Test tokenize_with_weights returns both tokens and weights."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTokenizeStrategy(v2=False, max_length=None)

        with patch("library.strategies.sd.tokenization.get_clip_input_ids") as mock_get_ids:
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
        strategy = SdTextEncodingStrategy(Mock(), clip_skip=2)
        assert strategy.clip_skip == 2

    def test_init_default_clip_skip_none(self):
        """Test that default clip_skip is None."""
        strategy = SdTextEncodingStrategy(Mock())
        assert strategy.clip_skip is None

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_encode_tokens_basic(self, mock_load_tokenizer, mock_clip_tokenizer, mock_clip_text_encoder):
        """Test basic token encoding without clip_skip."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=None)
        encoding_strategy = SdTextEncodingStrategy(tokenize_strategy.tokenizer, clip_skip=None)

        # Create fake tokens: batch=1, n=1, seq=77
        tokens = [torch.randint(0, 1000, (1, 1, 77))]

        result = encoding_strategy.encode_tokens([mock_clip_text_encoder], tokens)

        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_encode_tokens_with_clip_skip(self, mock_load_tokenizer, mock_clip_tokenizer, mock_clip_text_encoder):
        """Test token encoding with clip_skip."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=None)
        encoding_strategy = SdTextEncodingStrategy(tokenize_strategy.tokenizer, clip_skip=2)

        tokens = [torch.randint(0, 1000, (1, 1, 77))]

        result = encoding_strategy.encode_tokens([mock_clip_text_encoder], tokens)

        assert len(result) == 1

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_encode_tokens_multi_chunk(self, mock_load_tokenizer, mock_clip_tokenizer, mock_clip_text_encoder):
        """Test encoding with multiple token chunks (n > 1)."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=None)
        encoding_strategy = SdTextEncodingStrategy(tokenize_strategy.tokenizer, clip_skip=None)

        # Multi-chunk tokens: batch=1, n=3, seq=77
        tokens = [torch.randint(0, 1000, (1, 3, 77))]

        result = encoding_strategy.encode_tokens([mock_clip_text_encoder], tokens)

        assert len(result) == 1
        # Output should be reshaped for multi-chunk

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_encode_tokens_with_weights_applies_weights(self, mock_load_tokenizer, mock_clip_tokenizer, mock_clip_text_encoder):
        """Test that encode_tokens_with_weights applies weight multiplication."""
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=None)
        encoding_strategy = SdTextEncodingStrategy(tokenize_strategy.tokenizer, clip_skip=None)

        tokens = [torch.randint(0, 1000, (1, 1, 77))]
        weights = [torch.ones(1, 1, 77) * 1.5]

        result = encoding_strategy.encode_tokens_with_weights([mock_clip_text_encoder], tokens, weights)

        assert len(result) == 1


@pytest.mark.unit
class TestSdTrainingStrategyNewPipeline:
    """Tests for the SD new-pipeline strategy surface."""

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_create_latent_caching_strategy_returns_pipeline_strategy(self, mock_load_tokenizer, mock_clip_tokenizer, sd_strategy_cfg):
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTrainingStrategy(sd_strategy_cfg)
        cfg = Mock()
        cfg.performance.precision.no_half_vae = False
        cfg.data.preprocessing.flip_aug = True

        result = strategy.create_latent_caching_strategy(cfg)

        assert isinstance(result, SdLatentsPipelineStrategy)
        assert result.flip_aug is True
        assert result.dtype == "fp16"

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_create_te_caching_strategy_returns_pipeline_strategy(self, mock_load_tokenizer, mock_clip_tokenizer, sd_strategy_cfg):
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTrainingStrategy(sd_strategy_cfg)
        cfg = Mock()
        cfg.training.clip_skip = 2
        cfg.training.max_token_length = 150

        result = strategy.create_te_caching_strategy(cfg)

        assert isinstance(result, SdTextEncoderPipelineStrategy)
        assert result.clip_skip == 2
        assert result.max_token_length == 150

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_tokenize_captions_returns_clip_named_tensor(self, mock_load_tokenizer, mock_clip_tokenizer, sd_strategy_cfg):
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=75)
        strategy = SdTrainingStrategy(sd_strategy_cfg)

        result = strategy.tokenize_captions([tokenize_strategy.tokenizer], ["caption 1", "caption 2"], 75)

        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape[0] == 2

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_encode_te_outputs_in_memory_returns_hidden_state(
        self, mock_load_tokenizer, mock_clip_tokenizer, mock_clip_text_encoder, sd_strategy_cfg
    ):
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        tokenize_strategy = SdTokenizeStrategy(v2=False, max_length=75)
        strategy = SdTrainingStrategy(sd_strategy_cfg)

        result = strategy.encode_te_outputs_in_memory(
            text_encoders=[mock_clip_text_encoder],
            tokenizers=[tokenize_strategy.tokenizer],
            caption="a photo of a cat",
            max_token_length=75,
            device=torch.device("cpu"),
        )

        assert "hidden_state" in result
        assert isinstance(result["hidden_state"], torch.Tensor)

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_process_batch_accepts_new_text_encoder_outputs_key(self, mock_load_tokenizer, mock_clip_tokenizer, sd_strategy_cfg):
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTrainingStrategy(sd_strategy_cfg)
        batch = {
            "text_encoder_outputs": {"hidden_state": torch.randn(1, 77, 768)},
        }
        cfg = Mock()
        cfg.data.caption.weighted_captions = False
        cfg.performance.precision.full_fp16 = False
        cfg.loss.masked.masked_loss = False
        cfg.loss.loss_multiplier = None
        cfg.loss.edm2.edm2_loss_weighting = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")

        with (
            patch("library.strategies.sd.training.prepare_latents", return_value=torch.randn(1, 4, 64, 64)),
            patch.object(
                strategy,
                "get_noise_pred_and_target",
                return_value=(
                    torch.randn(1, 4, 64, 64),
                    torch.randn(1, 4, 64, 64),
                    torch.tensor([10]),
                    None,
                ),
            ) as mock_noise,
        ):
            strategy.process_batch(
                batch=batch,
                text_encoders=[Mock()],
                unet=Mock(),
                trainable_model=Mock(),
                vae=Mock(),
                noise_scheduler=Mock(),
                vae_dtype=torch.float32,
                weight_dtype=torch.float32,
                accelerator=accelerator,
                cfg=cfg,
                is_train=False,
                train_text_encoder=False,
            )

        text_conds = mock_noise.call_args.args[5]
        assert len(text_conds) == 1
        assert isinstance(text_conds[0], torch.Tensor)

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_process_batch_accepts_new_input_ids_key(self, mock_load_tokenizer, mock_clip_tokenizer, sd_strategy_cfg):
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTrainingStrategy(sd_strategy_cfg)
        batch = {
            "captions": ["a cat"],
            "input_ids": {"clip": torch.randint(0, 100, (1, 1, 77))},
        }
        cfg = Mock()
        cfg.data.caption.weighted_captions = False
        cfg.performance.precision.full_fp16 = False
        cfg.loss.masked.masked_loss = False
        cfg.loss.loss_multiplier = None
        cfg.loss.edm2.edm2_loss_weighting = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")
        accelerator.autocast.return_value = nullcontext()

        with (
            patch("library.strategies.sd.training.prepare_latents", return_value=torch.randn(1, 4, 64, 64)),
            patch.object(
                strategy,
                "get_noise_pred_and_target",
                return_value=(
                    torch.randn(1, 4, 64, 64),
                    torch.randn(1, 4, 64, 64),
                    torch.tensor([10]),
                    None,
                ),
            ),
            patch.object(strategy, "encode_tokens", return_value=[torch.randn(1, 77, 768)]) as mock_encode_tokens,
        ):
            strategy.process_batch(
                batch=batch,
                text_encoders=[Mock()],
                unet=Mock(),
                trainable_model=Mock(),
                vae=Mock(),
                noise_scheduler=Mock(),
                vae_dtype=torch.float32,
                weight_dtype=torch.float32,
                accelerator=accelerator,
                cfg=cfg,
                is_train=False,
                train_text_encoder=False,
            )

        mock_encode_tokens.assert_called_once()
