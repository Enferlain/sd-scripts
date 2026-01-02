"""
Unit tests for library/strategies/strategy_base.py

Tests the base strategy classes: TokenizeStrategy, TextEncodingStrategy,
TextEncoderOutputsCachingStrategy, and LatentsCachingStrategy.

Focus on pure functions and components testable with light mocking.
"""

import os
import pytest
import numpy as np
import torch
from unittest.mock import Mock

from library.strategies.strategy_base import (
    TokenizeStrategy,
    TextEncodingStrategy,
    TextEncoderOutputsCachingStrategy,
    LatentsCachingStrategy,
)


# =============================================================================
# TokenizeStrategy - Parse Prompt Attention Tests
# =============================================================================

@pytest.mark.unit
class TestTokenizeStrategyParsePromptAttention:
    """Test the parse_prompt_attention inner function via _get_weighted_input_ids."""

    @pytest.fixture
    def strategy(self):
        """Create a TokenizeStrategy instance for testing."""
        return TokenizeStrategy()

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

    def test_normal_text_returns_weight_1(self, strategy, mock_tokenizer):
        """Test that normal text without brackets has weight 1.0."""
        input_ids, weights = strategy._get_weighted_input_ids(
            mock_tokenizer, "normal text", max_length=77
        )
        
        # Weights for content tokens should be 1.0
        assert weights[0, 0].item() == 1.0  # BOS weight
        # All weights should be 1.0 for unmodified text
        for i in range(weights.shape[1]):
            assert weights[0, i].item() == 1.0

    def test_single_parentheses_increases_weight(self, strategy, mock_tokenizer):
        """Test that (word) increases weight by 1.1."""
        input_ids, weights = strategy._get_weighted_input_ids(
            mock_tokenizer, "(important)", max_length=77
        )
        
        # The token for "important" should have weight ~1.1
        # Position 1 is after BOS
        assert weights[0, 1].item() == pytest.approx(1.1, abs=0.001)

    def test_explicit_weight(self, strategy, mock_tokenizer):
        """Test that (word:1.5) sets weight to 1.5."""
        input_ids, weights = strategy._get_weighted_input_ids(
            mock_tokenizer, "(emphasized:1.5)", max_length=77
        )
        
        assert weights[0, 1].item() == pytest.approx(1.5, abs=0.001)

    def test_square_brackets_decrease_weight(self, strategy, mock_tokenizer):
        """Test that [word] decreases weight by 1/1.1."""
        input_ids, weights = strategy._get_weighted_input_ids(
            mock_tokenizer, "[weak]", max_length=77
        )
        
        assert weights[0, 1].item() == pytest.approx(1/1.1, abs=0.001)

    def test_nested_parentheses(self, strategy, mock_tokenizer):
        """Test that nested parentheses multiply weights."""
        input_ids, weights = strategy._get_weighted_input_ids(
            mock_tokenizer, "((double))", max_length=77
        )
        
        # 1.1 * 1.1 = 1.21
        assert weights[0, 1].item() == pytest.approx(1.21, abs=0.001)

    def test_zero_weight(self, strategy, mock_tokenizer):
        """Test that (word:0) sets weight to 0."""
        input_ids, weights = strategy._get_weighted_input_ids(
            mock_tokenizer, "(invisible:0)", max_length=77
        )
        
        assert weights[0, 1].item() == 0.0

    def test_bos_and_eos_weights_are_1(self, strategy, mock_tokenizer):
        """Test that BOS and padding weights are 1.0."""
        input_ids, weights = strategy._get_weighted_input_ids(
            mock_tokenizer, "(test:2.0)", max_length=77
        )
        
        assert weights[0, 0].item() == 1.0  # BOS
        # All padding should be 1.0
        for i in range(3, 77):
            assert weights[0, i].item() == 1.0


# =============================================================================
# TokenizeStrategy - Singleton Pattern Tests
# =============================================================================

@pytest.mark.unit
class TestTokenizeStrategySingleton:
    """Test the singleton pattern for TokenizeStrategy."""

    def setup_method(self):
        """Reset singleton before each test."""
        TokenizeStrategy._strategy = None

    def teardown_method(self):
        """Reset singleton after each test."""
        TokenizeStrategy._strategy = None

    def test_get_strategy_returns_none_initially(self):
        """Test that get_strategy returns None when no strategy is set."""
        assert TokenizeStrategy.get_strategy() is None

    def test_set_strategy_stores_instance(self):
        """Test that set_strategy stores the strategy instance."""
        strategy = TokenizeStrategy()
        TokenizeStrategy.set_strategy(strategy)
        
        assert TokenizeStrategy.get_strategy() is strategy

    def test_set_strategy_twice_raises_error(self):
        """Test that setting strategy twice raises RuntimeError."""
        strategy1 = TokenizeStrategy()
        strategy2 = TokenizeStrategy()
        
        TokenizeStrategy.set_strategy(strategy1)
        
        with pytest.raises(RuntimeError, match="already set"):
            TokenizeStrategy.set_strategy(strategy2)


# =============================================================================
# TokenizeStrategy - Load Tokenizer Tests
# =============================================================================

@pytest.mark.unit
class TestTokenizeStrategyLoadTokenizer:
    """Test the _load_tokenizer method."""

    @pytest.fixture
    def strategy(self):
        return TokenizeStrategy()

    def test_load_from_hub_when_no_cache(self, strategy, tmp_path):
        """Test loading tokenizer from HuggingFace hub."""
        mock_model_class = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.save_pretrained = Mock()

        result = strategy._load_tokenizer(
            mock_model_class,
            "openai/clip-vit-base",
            subfolder=None,
            tokenizer_cache_dir=str(tmp_path)
        )

        assert result is mock_tokenizer
        mock_model_class.from_pretrained.assert_called_once_with(
            "openai/clip-vit-base", subfolder=None
        )
        # Should save to cache
        mock_tokenizer.save_pretrained.assert_called_once()

    def test_load_from_cache_when_exists(self, strategy, tmp_path):
        """Test loading tokenizer from local cache."""
        mock_model_class = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_tokenizer

        # Create the cache directory
        cache_path = tmp_path / "openai_clip-vit-base"
        cache_path.mkdir()

        result = strategy._load_tokenizer(
            mock_model_class,
            "openai/clip-vit-base",
            tokenizer_cache_dir=str(tmp_path)
        )

        assert result is mock_tokenizer
        # Should load from cache path
        mock_model_class.from_pretrained.assert_called_once_with(str(cache_path))

    def test_load_without_cache_dir(self, strategy):
        """Test loading tokenizer without cache directory."""
        mock_model_class = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_tokenizer

        result = strategy._load_tokenizer(
            mock_model_class,
            "openai/clip-vit-base",
            subfolder="tokenizer"
        )

        assert result is mock_tokenizer
        mock_model_class.from_pretrained.assert_called_once_with(
            "openai/clip-vit-base", subfolder="tokenizer"
        )


# =============================================================================
# LatentsCachingStrategy - Path Parsing Tests
# =============================================================================

@pytest.mark.unit
class TestLatentsCachingStrategyPathParsing:
    """Test the path parsing utilities."""

    @pytest.fixture
    def strategy(self):
        """Create a LatentsCachingStrategy instance."""
        return LatentsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False
        )

    def test_parse_size_from_npz_path(self, strategy):
        """Test extracting image size from npz path."""
        npz_path = "/path/to/image_512x768_latents.npz"
        w, h = strategy.get_image_size_from_disk_cache_path("/path/to/image.png", npz_path)
        
        assert w == 512
        assert h == 768

    def test_parse_size_from_complex_path(self, strategy):
        """Test size parsing with complex filename."""
        npz_path = "C:/data/dataset_v2/img_001_1024x1024_sd15.npz"
        w, h = strategy.get_image_size_from_disk_cache_path("C:/data/img.jpg", npz_path)
        
        assert w == 1024
        assert h == 1024

    def test_parse_size_from_asymmetric_resolution(self, strategy):
        """Test size parsing with non-square resolution."""
        npz_path = "/images/photo_768x512_cache.npz"
        w, h = strategy.get_image_size_from_disk_cache_path("/images/photo.png", npz_path)
        
        assert w == 768
        assert h == 512


# =============================================================================
# LatentsCachingStrategy - Save/Load NPZ Tests
# =============================================================================

@pytest.mark.unit
class TestLatentsCachingSaveLoad:
    """Test save and load functionality for latents."""

    @pytest.fixture
    def strategy(self):
        return LatentsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False
        )

    def test_save_basic_latents(self, strategy, tmp_path):
        """Test saving basic latents to disk."""
        npz_path = str(tmp_path / "test_latents.npz")
        latents = torch.randn(4, 64, 64)
        original_size = [512, 512]
        crop_ltrb = [0, 0, 512, 512]

        strategy.save_latents_to_disk(
            npz_path, latents, original_size, crop_ltrb
        )

        # Verify the file exists and has correct keys
        assert os.path.exists(npz_path)
        npz = np.load(npz_path)
        assert "latents" in npz
        assert "original_size" in npz
        assert "crop_ltrb" in npz
        assert npz["latents"].shape == (4, 64, 64)

    def test_save_with_flipped_latents(self, strategy, tmp_path):
        """Test saving latents with flipped version."""
        npz_path = str(tmp_path / "test_latents.npz")
        latents = torch.randn(4, 64, 64)
        flipped = torch.randn(4, 64, 64)
        original_size = [512, 512]
        crop_ltrb = [0, 0, 512, 512]

        strategy.save_latents_to_disk(
            npz_path, latents, original_size, crop_ltrb,
            flipped_latents_tensor=flipped
        )

        npz = np.load(npz_path)
        assert "latents_flipped" in npz
        assert npz["latents_flipped"].shape == (4, 64, 64)

    def test_save_with_alpha_mask(self, strategy, tmp_path):
        """Test saving latents with alpha mask."""
        npz_path = str(tmp_path / "test_latents.npz")
        latents = torch.randn(4, 64, 64)
        alpha_mask = torch.randn(1, 64, 64)
        original_size = [512, 512]
        crop_ltrb = [0, 0, 512, 512]

        strategy.save_latents_to_disk(
            npz_path, latents, original_size, crop_ltrb,
            alpha_mask=alpha_mask
        )

        npz = np.load(npz_path)
        assert "alpha_mask" in npz

    def test_save_with_resolution_suffix(self, strategy, tmp_path):
        """Test saving latents with resolution suffix for multi-resolution."""
        npz_path = str(tmp_path / "test_latents.npz")
        latents = torch.randn(4, 64, 64)
        original_size = [512, 512]
        crop_ltrb = [0, 0, 512, 512]

        strategy.save_latents_to_disk(
            npz_path, latents, original_size, crop_ltrb,
            key_reso_suffix="_64x64"
        )

        npz = np.load(npz_path)
        assert "latents_64x64" in npz
        assert "original_size_64x64" in npz
        assert "crop_ltrb_64x64" in npz

    def test_load_basic_latents(self, strategy, tmp_path):
        """Test loading basic latents from disk."""
        npz_path = str(tmp_path / "test_latents.npz")
        
        # Save first
        latents = torch.randn(4, 64, 64)
        strategy.save_latents_to_disk(
            npz_path, latents, [512, 512], [0, 0, 512, 512]
        )

        # Load
        loaded, original_size, crop_ltrb, flipped, alpha = strategy.load_latents_from_disk(
            npz_path, (512, 512)
        )

        assert loaded is not None
        assert loaded.shape == (4, 64, 64)
        assert original_size == [512, 512]
        assert crop_ltrb == [0, 0, 512, 512]
        assert flipped is None
        assert alpha is None

    def test_load_with_optional_arrays(self, strategy, tmp_path):
        """Test loading latents with flipped and alpha mask."""
        npz_path = str(tmp_path / "test_latents.npz")
        
        # Save with all optional fields
        latents = torch.randn(4, 64, 64)
        flipped = torch.randn(4, 64, 64)
        alpha = torch.randn(1, 64, 64)
        strategy.save_latents_to_disk(
            npz_path, latents, [512, 512], [0, 0, 512, 512],
            flipped_latents_tensor=flipped,
            alpha_mask=alpha
        )

        # Load
        loaded, _, _, loaded_flipped, loaded_alpha = strategy.load_latents_from_disk(
            npz_path, (512, 512)
        )

        assert loaded_flipped is not None
        assert loaded_alpha is not None


# =============================================================================
# LatentsCachingStrategy - Disk Cache Expected Tests
# =============================================================================

@pytest.mark.unit
class TestLatentsCachingDiskExpected:
    """Test _default_is_disk_cached_latents_expected."""

    @pytest.fixture
    def strategy(self):
        return LatentsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=False
        )

    @pytest.fixture
    def strategy_skip_check(self):
        return LatentsCachingStrategy(
            cache_to_disk=True,
            batch_size=1,
            skip_disk_cache_validity_check=True
        )

    @pytest.fixture
    def strategy_no_disk(self):
        return LatentsCachingStrategy(
            cache_to_disk=False,
            batch_size=1,
            skip_disk_cache_validity_check=False
        )

    def test_returns_false_when_cache_to_disk_false(self, strategy_no_disk, tmp_path):
        """Test that False is returned when cache_to_disk is False."""
        npz_path = str(tmp_path / "test.npz")
        
        result = strategy_no_disk._default_is_disk_cached_latents_expected(
            latents_stride=8,
            bucket_reso=(512, 512),
            npz_path=npz_path,
            flip_aug=False,
            apply_alpha_mask=False
        )
        
        assert result is False

    def test_returns_false_when_file_not_exists(self, strategy, tmp_path):
        """Test that False is returned when npz file doesn't exist."""
        npz_path = str(tmp_path / "nonexistent.npz")
        
        result = strategy._default_is_disk_cached_latents_expected(
            latents_stride=8,
            bucket_reso=(512, 512),
            npz_path=npz_path,
            flip_aug=False,
            apply_alpha_mask=False
        )
        
        assert result is False

    def test_returns_true_when_skip_validity_check(self, strategy_skip_check, tmp_path):
        """Test that True is returned when skip_disk_cache_validity_check is True."""
        npz_path = str(tmp_path / "test.npz")
        # Create empty npz
        np.savez(npz_path)
        
        result = strategy_skip_check._default_is_disk_cached_latents_expected(
            latents_stride=8,
            bucket_reso=(512, 512),
            npz_path=npz_path,
            flip_aug=False,
            apply_alpha_mask=False
        )
        
        assert result is True

    def test_returns_false_when_latents_key_missing(self, strategy, tmp_path):
        """Test that False is returned when latents key is missing."""
        npz_path = str(tmp_path / "test.npz")
        np.savez(npz_path, other_data=np.array([1, 2, 3]))
        
        result = strategy._default_is_disk_cached_latents_expected(
            latents_stride=8,
            bucket_reso=(512, 512),
            npz_path=npz_path,
            flip_aug=False,
            apply_alpha_mask=False
        )
        
        assert result is False

    def test_returns_true_when_all_keys_present(self, strategy, tmp_path):
        """Test that True is returned when all required keys are present."""
        npz_path = str(tmp_path / "test.npz")
        np.savez(npz_path, latents=np.zeros((4, 64, 64)))
        
        result = strategy._default_is_disk_cached_latents_expected(
            latents_stride=8,
            bucket_reso=(512, 512),
            npz_path=npz_path,
            flip_aug=False,
            apply_alpha_mask=False
        )
        
        assert result is True

    def test_returns_false_when_flip_aug_key_missing(self, strategy, tmp_path):
        """Test that False is returned when flip_aug is True but key missing."""
        npz_path = str(tmp_path / "test.npz")
        np.savez(npz_path, latents=np.zeros((4, 64, 64)))
        
        result = strategy._default_is_disk_cached_latents_expected(
            latents_stride=8,
            bucket_reso=(512, 512),
            npz_path=npz_path,
            flip_aug=True,
            apply_alpha_mask=False
        )
        
        assert result is False

    def test_returns_false_when_alpha_mask_key_missing(self, strategy, tmp_path):
        """Test that False is returned when apply_alpha_mask is True but key missing."""
        npz_path = str(tmp_path / "test.npz")
        np.savez(npz_path, latents=np.zeros((4, 64, 64)))
        
        result = strategy._default_is_disk_cached_latents_expected(
            latents_stride=8,
            bucket_reso=(512, 512),
            npz_path=npz_path,
            flip_aug=False,
            apply_alpha_mask=True
        )
        
        assert result is False


# =============================================================================
# LatentsCachingStrategy - Singleton Pattern Tests
# =============================================================================

@pytest.mark.unit
class TestLatentsCachingStrategySingleton:
    """Test the singleton pattern for LatentsCachingStrategy."""

    def setup_method(self):
        LatentsCachingStrategy._strategy = None

    def teardown_method(self):
        LatentsCachingStrategy._strategy = None

    def test_get_strategy_returns_none_initially(self):
        assert LatentsCachingStrategy.get_strategy() is None

    def test_set_strategy_stores_instance(self):
        strategy = LatentsCachingStrategy(True, 1, False)
        LatentsCachingStrategy.set_strategy(strategy)
        
        assert LatentsCachingStrategy.get_strategy() is strategy

    def test_set_strategy_twice_raises_error(self):
        strategy1 = LatentsCachingStrategy(True, 1, False)
        strategy2 = LatentsCachingStrategy(True, 1, False)
        
        LatentsCachingStrategy.set_strategy(strategy1)
        
        with pytest.raises(RuntimeError, match="already set"):
            LatentsCachingStrategy.set_strategy(strategy2)


# =============================================================================
# TextEncoderOutputsCachingStrategy Tests
# =============================================================================

@pytest.mark.unit
class TestTextEncoderOutputsCachingStrategy:
    """Test TextEncoderOutputsCachingStrategy properties and singleton."""

    def setup_method(self):
        TextEncoderOutputsCachingStrategy._strategy = None

    def teardown_method(self):
        TextEncoderOutputsCachingStrategy._strategy = None

    def test_init_stores_properties(self):
        """Test that __init__ stores all properties correctly."""
        strategy = TextEncoderOutputsCachingStrategy(
            cache_to_disk=True,
            batch_size=4,
            skip_disk_cache_validity_check=True,
            is_partial=True,
            is_weighted=True
        )
        
        assert strategy.cache_to_disk is True
        assert strategy.batch_size == 4
        assert strategy.skip_disk_cache_validity_check is True
        assert strategy.is_partial is True
        assert strategy.is_weighted is True

    def test_init_default_values(self):
        """Test that __init__ uses correct default values."""
        strategy = TextEncoderOutputsCachingStrategy(
            cache_to_disk=False,
            batch_size=1,
            skip_disk_cache_validity_check=False
        )
        
        assert strategy.is_partial is False
        assert strategy.is_weighted is False

    def test_singleton_pattern(self):
        """Test singleton set/get pattern."""
        strategy = TextEncoderOutputsCachingStrategy(True, 1, False)
        TextEncoderOutputsCachingStrategy.set_strategy(strategy)
        
        assert TextEncoderOutputsCachingStrategy.get_strategy() is strategy

    def test_singleton_double_set_raises(self):
        """Test that setting singleton twice raises error."""
        s1 = TextEncoderOutputsCachingStrategy(True, 1, False)
        s2 = TextEncoderOutputsCachingStrategy(True, 1, False)
        
        TextEncoderOutputsCachingStrategy.set_strategy(s1)
        
        with pytest.raises(RuntimeError, match="already set"):
            TextEncoderOutputsCachingStrategy.set_strategy(s2)


# =============================================================================
# TextEncodingStrategy - Singleton Pattern Tests
# =============================================================================

@pytest.mark.unit
class TestTextEncodingStrategySingleton:
    """Test the singleton pattern for TextEncodingStrategy."""

    def setup_method(self):
        TextEncodingStrategy._strategy = None

    def teardown_method(self):
        TextEncodingStrategy._strategy = None

    def test_get_strategy_returns_none_initially(self):
        assert TextEncodingStrategy.get_strategy() is None

    def test_set_strategy_stores_instance(self):
        strategy = TextEncodingStrategy()
        TextEncodingStrategy.set_strategy(strategy)
        
        assert TextEncodingStrategy.get_strategy() is strategy

    def test_set_strategy_twice_raises_error(self):
        strategy1 = TextEncodingStrategy()
        strategy2 = TextEncodingStrategy()
        
        TextEncodingStrategy.set_strategy(strategy1)
        
        with pytest.raises(RuntimeError, match="already set"):
            TextEncodingStrategy.set_strategy(strategy2)
