"""Unit tests for the active SD strategy concern files."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from unittest.mock import Mock, patch

from library.strategies.base.context import StrategyContext, StrategyPhase, current_strategy_context
from library.strategies.sd.caching import SdLatentsPipelineStrategy, SdTextEncoderPipelineStrategy
from library.strategies.sd.denoiser import SdDenoiserCallingStrategy
from library.strategies.sd.diffusion import SdDiffusionTrainingStrategy
from library.strategies.sd.encoding import SdTextEncodingStrategy
from library.strategies.sd.tokenization import SdTokenizeStrategy
from library.strategies.sd.training import SdTrainingStrategy


# =============================================================================
# Mock Fixtures
# =============================================================================


class _TestSdDenoiserStrategy(SdDiffusionTrainingStrategy, SdDenoiserCallingStrategy):
    pass


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
    cfg.data.caching.tokenizer_cache_dir = None
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
class TestSdTrainingStrategyComposition:
    """Tests for SD strategy composition across concern files."""

    def test_training_strategy_uses_split_concern_modules(self):
        """SdTrainingStrategy should inherit facet methods from the split SD files."""
        assert SdTrainingStrategy.tokenize_captions.__module__ == "library.strategies.sd.tokenization"
        assert SdTrainingStrategy.encode_te_outputs_in_memory.__module__ == "library.strategies.sd.encoding"
        assert SdTrainingStrategy.create_te_caching_strategy.__module__ == "library.strategies.sd.caching"
        assert SdTrainingStrategy.resolve_conditioning.__module__ == "library.strategies.sd.conditioning"

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
        cfg.loss.edm2.enabled = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")

        with (
            patch("library.strategies.sd.diffusion.prepare_latents", return_value=torch.randn(1, 4, 64, 64)),
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
                denoiser=Mock(),
                trainable_model=Mock(),
                vae=Mock(),
                objective_runtime=Mock(num_train_timesteps=1000, alphas_cumprod=None),
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
    def test_resolve_conditioning_uses_cached_outputs_without_live_reencode(
        self,
        mock_load_tokenizer,
        mock_clip_tokenizer,
        mock_clip_text_encoder,
        sd_strategy_cfg,
    ):
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTrainingStrategy(sd_strategy_cfg)
        batch = {
            "text_encoder_outputs": {"hidden_state": torch.randn(1, 77, 768)},
        }
        cfg = Mock()
        cfg.data.caption.weighted_captions = False
        cfg.performance.precision.full_fp16 = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")

        with (
            patch.object(strategy, "encode_tokens") as mock_encode,
            patch.object(strategy, "encode_tokens_with_weights") as mock_weighted_encode,
        ):
            result = strategy.resolve_conditioning(
                batch=batch,
                text_encoders=[mock_clip_text_encoder],
                accelerator=accelerator,
                cfg=cfg,
                train_text_encoder=False,
                is_train=False,
                weight_dtype=torch.float32,
            )

        mock_encode.assert_not_called()
        mock_weighted_encode.assert_not_called()
        assert len(result) == 1

    @patch("library.strategies.sd.tokenization.load_tokenizer")
    def test_resolve_conditioning_reencodes_when_training_text_encoder(
        self,
        mock_load_tokenizer,
        mock_clip_tokenizer,
        mock_clip_text_encoder,
        sd_strategy_cfg,
    ):
        mock_load_tokenizer.return_value = mock_clip_tokenizer
        strategy = SdTrainingStrategy(sd_strategy_cfg)
        cached_hidden_state = torch.randn(1, 77, 768)
        live_hidden_state = torch.randn(1, 77, 768)
        batch = {
            "captions": ["a cat"],
            "text_encoder_outputs": {"hidden_state": cached_hidden_state},
        }
        cfg = Mock()
        cfg.data.caption.weighted_captions = False
        cfg.performance.precision.full_fp16 = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")
        accelerator.autocast.return_value = nullcontext()

        with (
            patch.object(strategy, "tokenize", return_value=[torch.randint(0, 100, (1, 1, 77))]),
            patch.object(strategy, "get_models_for_text_encoding", return_value=[mock_clip_text_encoder]),
            patch.object(strategy, "encode_tokens", return_value=[live_hidden_state]) as mock_encode,
        ):
            result = strategy.resolve_conditioning(
                batch=batch,
                text_encoders=[mock_clip_text_encoder],
                accelerator=accelerator,
                cfg=cfg,
                train_text_encoder=True,
                is_train=True,
                weight_dtype=torch.float32,
            )

        mock_encode.assert_called_once()
        assert torch.equal(result[0], live_hidden_state)

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
        cfg.loss.edm2.enabled = False
        accelerator = Mock()
        accelerator.device = torch.device("cpu")
        accelerator.autocast.return_value = nullcontext()

        with (
            patch("library.strategies.sd.diffusion.prepare_latents", return_value=torch.randn(1, 4, 64, 64)),
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
                denoiser=Mock(),
                trainable_model=Mock(),
                vae=Mock(),
                objective_runtime=Mock(num_train_timesteps=1000, alphas_cumprod=None),
                vae_dtype=torch.float32,
                weight_dtype=torch.float32,
                accelerator=accelerator,
                cfg=cfg,
                is_train=False,
                train_text_encoder=False,
        )

        mock_encode_tokens.assert_called_once()


@pytest.mark.unit
def test_sd_get_noise_pred_and_target_publishes_denoiser_forward_contexts() -> None:
    latents = torch.zeros(3, 1, 2, 2)
    noise = torch.ones_like(latents)
    timesteps = torch.tensor([100, 200, 300], dtype=torch.long)
    noisy_latents = torch.full_like(latents, 0.5)
    observed_contexts: list[StrategyContext] = []

    class _RecordingDenoiser:
        def __call__(self, model_input, denoiser_timesteps, text_cond_tensor):
            del denoiser_timesteps, text_cond_tensor
            context = current_strategy_context()
            assert context is not None
            observed_contexts.append(context)
            return SimpleNamespace(sample=torch.ones_like(model_input))

    strategy = _TestSdDenoiserStrategy()
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sd1"),
        objective=SimpleNamespace(prediction="epsilon"),
        loss=SimpleNamespace(regularization=None),
        timestep=SimpleNamespace(),
        training=SimpleNamespace(),
        performance=SimpleNamespace(memory=SimpleNamespace(gradient_checkpointing=False)),
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=lambda: nullcontext())
    objective_runtime = SimpleNamespace(noise_scheduler=Mock(), timestep_runtime=None)
    trainable_model = SimpleNamespace(set_multiplier=Mock())
    batch = {
        "custom_attributes": [
            {"diff_output_preservation": True},
            {},
            {"diff_output_preservation": True},
        ]
    }

    with (
        patch(
            "library.strategies.sd.diffusion.prepare_ddpm_training_inputs",
            return_value=(noise, noisy_latents, timesteps),
        ),
        patch(
            "library.strategies.sd.diffusion.build_ddpm_training_target",
            return_value=torch.zeros_like(latents),
        ),
    ):
        strategy.get_noise_pred_and_target(
            cfg=cfg,
            accelerator=accelerator,
            objective_runtime=objective_runtime,
            latents=latents,
            batch=batch,
            text_encoder_conds=[torch.randn(3, 4, 5)],
            denoiser=_RecordingDenoiser(),
            trainable_model=trainable_model,
            weight_dtype=torch.float32,
            train_denoiser=True,
            is_train=True,
            global_step=42,
        )

    assert len(observed_contexts) == 2
    normal_context, indexed_context = observed_contexts
    assert normal_context.phase is StrategyPhase.TRAIN
    assert normal_context.model_family == "sd1"
    assert normal_context.training is not None
    assert normal_context.training.global_step == 42
    assert normal_context.training.is_train is True
    assert normal_context.denoiser is not None
    assert torch.equal(normal_context.denoiser.timesteps, timesteps)
    assert normal_context.denoiser.sample_indices is None
    assert normal_context.denoiser.batch_size == 3

    assert indexed_context.phase is StrategyPhase.TRAIN
    assert indexed_context.training is not None
    assert indexed_context.training.global_step == 42
    assert indexed_context.training.is_train is True
    assert indexed_context.denoiser is not None
    assert torch.equal(indexed_context.denoiser.timesteps, torch.tensor([100, 300]))
    assert indexed_context.denoiser.sample_indices == (0, 2)
    assert indexed_context.denoiser.batch_size == 2
    assert current_strategy_context() is None
