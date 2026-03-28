"""
Unit tests for base strategy contracts and shared helpers.

Focus on pure functions and components testable with light mocking.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from library.models.sd.tokenizer import load_tokenizer
from library.strategies.shared.clip.tokenization import get_clip_weighted_input_ids
from library.optimizers.optimizer_utils import (
    get_text_encoders_train_flags,
    should_train_denoiser,
    should_train_text_encoder,
)
from library.strategies.base.contracts import (
    CachingStrategy,
    DiffusionTrainingStrategy,
    ModelLoadingStrategy,
    ModelPreparationStrategy,
    TextEncodingStrategy,
    TokenizationStrategy,
    TrainingStrategy,
    ValidationStrategy,
)
from library.training.diffusion import prepare_latents
from library.training.noise_utils import get_noise_scheduler
from library.training.trainer_utils import all_reduce_trainable, restore_rng_state, switch_rng_state


# =============================================================================
# Tokenization helper tests
# =============================================================================


class DummyTokenizationStrategy(TokenizationStrategy):
    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        return []

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return [], []


class DummyTextEncodingStrategy(TextEncodingStrategy):
    def encode_tokens(self, models: list[object], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        return []

    def encode_tokens_with_weights(
        self,
        models: list[object],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return []


@pytest.mark.unit
class TestTokenizationHelpers:
    """Test CLIP-family weighted token helper behavior."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock CLIPTokenizer with necessary attributes."""
        tokenizer = Mock()
        tokenizer.model_max_length = 77
        tokenizer.bos_token_id = 49406
        tokenizer.eos_token_id = 49407
        tokenizer.pad_token_id = 49407  # v1 style

        # Mock __call__ to return input_ids matching CLIP tokenizer behavior
        # When called with a word, return BOS + tokens for word + EOS
        def tokenizer_call(text, **kwargs):
            # Simple tokenization: one token per non-empty word
            if not text.strip():
                return Mock(input_ids=[49406, 49407])  # BOS, EOS only
            words = text.split()
            tokens = [100 + i for i in range(max(1, len(words)))]
            return Mock(input_ids=[49406] + tokens + [49407])

        tokenizer.__call__ = tokenizer_call
        tokenizer.side_effect = tokenizer_call
        return tokenizer

    def test_normal_text_returns_weight_1(self, mock_tokenizer):
        """Test that normal text without brackets has weight 1.0."""
        _input_ids, weights = get_clip_weighted_input_ids(mock_tokenizer, "normal text", max_length=77)

        # Weights for content tokens should be 1.0
        assert weights[0, 0].item() == 1.0  # BOS weight
        # All weights should be 1.0 for unmodified text
        for i in range(weights.shape[1]):
            assert weights[0, i].item() == 1.0

    def test_single_parentheses_increases_weight(self, mock_tokenizer):
        """Test that (word) increases weight by 1.1."""
        _input_ids, weights = get_clip_weighted_input_ids(mock_tokenizer, "(important)", max_length=77)

        # The token for "important" should have weight ~1.1
        # Position 1 is after BOS
        assert weights[0, 1].item() == pytest.approx(1.1, abs=0.001)

    def test_explicit_weight(self, mock_tokenizer):
        """Test that (word:1.5) sets weight to 1.5."""
        _input_ids, weights = get_clip_weighted_input_ids(mock_tokenizer, "(emphasized:1.5)", max_length=77)

        assert weights[0, 1].item() == pytest.approx(1.5, abs=0.001)

    def test_square_brackets_decrease_weight(self, mock_tokenizer):
        """Test that [word] decreases weight by 1/1.1."""
        _input_ids, weights = get_clip_weighted_input_ids(mock_tokenizer, "[weak]", max_length=77)

        assert weights[0, 1].item() == pytest.approx(1 / 1.1, abs=0.001)

    def test_nested_parentheses(self, mock_tokenizer):
        """Test that nested parentheses multiply weights."""
        _input_ids, weights = get_clip_weighted_input_ids(mock_tokenizer, "((double))", max_length=77)

        # 1.1 * 1.1 = 1.21
        assert weights[0, 1].item() == pytest.approx(1.21, abs=0.001)

    def test_zero_weight(self, mock_tokenizer):
        """Test that (word:0) sets weight to 0."""
        _input_ids, weights = get_clip_weighted_input_ids(mock_tokenizer, "(invisible:0)", max_length=77)

        assert weights[0, 1].item() == 0.0

    def test_bos_and_eos_weights_are_1(self, mock_tokenizer):
        """Test that BOS and padding weights are 1.0."""
        _input_ids, weights = get_clip_weighted_input_ids(mock_tokenizer, "(test:2.0)", max_length=77)

        assert weights[0, 0].item() == 1.0  # BOS
        # All padding should be 1.0
        for i in range(3, 77):
            assert weights[0, i].item() == 1.0


# =============================================================================
# Model Tokenizer Loading Tests
# =============================================================================


@pytest.mark.unit
class TestModelTokenizerLoading:
    """Test the shared model-layer tokenizer loader."""

    def test_load_from_hub_when_no_cache(self, tmp_path):
        """Test loading tokenizer from HuggingFace hub."""
        mock_model_class = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.save_pretrained = Mock()

        result = load_tokenizer(mock_model_class, "openai/clip-vit-base", subfolder=None, tokenizer_cache_dir=str(tmp_path))

        assert result is mock_tokenizer
        mock_model_class.from_pretrained.assert_called_once_with("openai/clip-vit-base", subfolder=None)
        # Should save to cache
        mock_tokenizer.save_pretrained.assert_called_once()

    def test_load_from_cache_when_exists(self, tmp_path):
        """Test loading tokenizer from local cache."""
        mock_model_class = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_tokenizer

        # Create the cache directory
        cache_path = tmp_path / "openai_clip-vit-base"
        cache_path.mkdir()

        result = load_tokenizer(mock_model_class, "openai/clip-vit-base", tokenizer_cache_dir=str(tmp_path))

        assert result is mock_tokenizer
        # Should load from cache path
        mock_model_class.from_pretrained.assert_called_once_with(str(cache_path))

    def test_load_without_cache_dir(self):
        """Test loading tokenizer without cache directory."""
        mock_model_class = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_tokenizer

        result = load_tokenizer(mock_model_class, "openai/clip-vit-base", subfolder="tokenizer")

        assert result is mock_tokenizer
        mock_model_class.from_pretrained.assert_called_once_with("openai/clip-vit-base", subfolder="tokenizer")


# =============================================================================
# TrainingStrategy Facets - Phase 2 Boundary Tests
# =============================================================================


class _DummyDiffusionStrategy(DiffusionTrainingStrategy):
    def process_batch(self, *args, **kwargs):
        raise NotImplementedError


class _DummyLoadingStrategy(ModelLoadingStrategy):
    def load_target_model(self, cfg, weight_dtype, accelerator):
        return "dummy", [], None, None


class _DummyValidationStrategy(ValidationStrategy):
    def calculate_val_loss(self, *args, **kwargs):
        return None, None


class _DummyCachingStrategy(CachingStrategy):
    def create_latent_caching_strategy(self, cfg):
        return object()

    def create_te_caching_strategy(self, cfg):
        return object()


@pytest.mark.unit
class TestTrainingStrategyPhase2Facets:
    def test_training_strategy_composes_phase2_facets(self):
        """TrainingStrategy should compose the new capability facets."""
        assert issubclass(TrainingStrategy, ModelPreparationStrategy)
        assert issubclass(TrainingStrategy, DiffusionTrainingStrategy)
        assert issubclass(TrainingStrategy, TokenizationStrategy)
        assert issubclass(TrainingStrategy, TextEncodingStrategy)

    def test_phase2_methods_live_on_facet_classes(self):
        """Moved shared helpers should live on their facet bases, not TrainingStrategy itself."""
        assert "tokenize" in TokenizationStrategy.__dict__
        assert "tokenize_with_weights" in TokenizationStrategy.__dict__
        assert "tokenize_captions" in TokenizationStrategy.__dict__
        assert "encode_tokens" in TextEncodingStrategy.__dict__
        assert "encode_tokens_with_weights" in TextEncodingStrategy.__dict__
        assert "get_models_for_text_encoding" in TextEncodingStrategy.__dict__
        assert "encode_te_outputs_in_memory" in TextEncodingStrategy.__dict__
        assert "create_latent_caching_strategy" in CachingStrategy.__dict__
        assert "load_denoiser_lazily" in ModelLoadingStrategy.__dict__
        assert "cast_text_encoder" in ModelPreparationStrategy.__dict__
        assert "cast_vae" in ModelPreparationStrategy.__dict__
        assert "cast_denoiser" in ModelPreparationStrategy.__dict__
        assert "post_process_trainable" in ModelPreparationStrategy.__dict__
        assert "calculate_val_loss" in ValidationStrategy.__dict__
        assert "is_train_denoiser" not in ModelPreparationStrategy.__dict__
        assert "is_train_text_encoder" not in ModelPreparationStrategy.__dict__
        assert "get_text_encoders_train_flags" not in ModelPreparationStrategy.__dict__

        assert "load_denoiser_lazily" not in TrainingStrategy.__dict__
        assert "tokenize_captions" not in TrainingStrategy.__dict__
        assert "tokenize_captions" not in CachingStrategy.__dict__
        assert "initialize" not in TrainingStrategy.__dict__
        assert "get_models_for_text_encoding" not in TrainingStrategy.__dict__
        assert "get_models_for_text_encoding" not in CachingStrategy.__dict__
        assert "encode_te_outputs_in_memory" not in TrainingStrategy.__dict__
        assert "encode_te_outputs_in_memory" not in CachingStrategy.__dict__
        assert "calculate_val_loss" not in TrainingStrategy.__dict__

    def test_model_loading_default_lazy_denoiser_hook_still_raises(self):
        """Default lazy-load hook remains opt-in for model families that need it."""
        strategy = _DummyLoadingStrategy()
        with pytest.raises(NotImplementedError, match="load_denoiser_lazily"):
            strategy.load_denoiser_lazily(cfg=Mock(), weight_dtype=torch.float16, accelerator=Mock(), text_encoders=[])

    def test_model_prep_methods_require_explicit_strategy_implementation(self):
        """Model-prep hooks no longer silently default in the base facet."""
        with pytest.raises(TypeError):
            ModelPreparationStrategy()

    def test_diffusion_helper_prepare_latents_scales_encoded_latents(self):
        """Shared diffusion helper should still apply VAE latent scaling."""
        caching_config = Mock()
        caching_config.vae_batch_size = None
        log_fn = Mock()
        vae = Mock()
        latent_dist = Mock()
        latent_dist.sample.return_value = torch.ones((1, 4, 8, 8))
        vae.encode.return_value.latent_dist = latent_dist

        batch = {"images": torch.ones((1, 3, 64, 64))}
        latents = prepare_latents(
            batch,
            caching_config,
            torch.device("cpu"),
            vae,
            torch.float32,
            vae_latent_scale=2.0,
            log_fn=log_fn,
        )

        assert torch.equal(latents, torch.full((1, 4, 8, 8), 2.0))

    def test_noise_scheduler_helper_builds_scheduler(self):
        """Noise scheduler construction now lives in training.noise_utils."""
        cfg = Mock()
        cfg.loss.regularization.zero_terminal_snr = False
        scheduler = get_noise_scheduler(cfg, torch.device("cpu"))
        assert scheduler.config.num_train_timesteps == 1000

    def test_training_runtime_helper_all_reduce_runs_on_gradients(self):
        """Gradient all-reduce helper now lives in trainer_utils."""
        accelerator = Mock()
        accelerator.reduce.side_effect = lambda grad, reduction="mean": grad
        module = torch.nn.Linear(2, 2)
        module.weight.grad = torch.ones_like(module.weight)

        all_reduce_trainable(accelerator, module)

        accelerator.reduce.assert_called()

    def test_trainability_helpers_resolve_lr_policy(self):
        """LR-based trainability now lives in the shared optimizer helper layer."""
        learning_rates = Mock()
        learning_rates.denoiser = None
        learning_rates.text_encoders = [1e-5, 0.0]

        assert should_train_denoiser(learning_rates) is True
        assert should_train_text_encoder(learning_rates) is True
        assert get_text_encoders_train_flags(learning_rates, [Mock(), Mock(), Mock()]) == [True, False, False]

    def test_rng_helpers_can_round_trip_rng_state(self):
        """Validation RNG helpers now live in trainer_utils."""
        accelerator = Mock()
        accelerator.device = torch.device("cpu")

        torch.manual_seed(123)
        np.random.seed(123)
        import random

        random.seed(123)
        original_torch = torch.rand(3)
        original_np = np.random.rand(3)
        original_py = random.random()

        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        states = switch_rng_state(999, accelerator)
        _ = torch.rand(3)
        _ = np.random.rand(3)
        _ = random.random()
        restore_rng_state(states, accelerator)

        assert torch.equal(torch.rand(3), original_torch)
        assert np.allclose(np.random.rand(3), original_np)
        assert random.random() == original_py
