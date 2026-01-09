"""
Unit tests for caption processor.

Tests all caption augmentation features:
- Tag shuffle
- Caption dropout
- Tag dropout
- Wildcards
- Token warmup
- Protected tags
- Keep tokens
"""

import random

from library.data.caption_processor import CaptionConfig, process_caption


class TestCaptionProcessorBasic:
    """Basic caption processing tests."""

    def test_no_processing_returns_unchanged(self):
        """With default config, caption should be unchanged."""
        config = CaptionConfig()
        result = process_caption("a cat, sitting, on a mat", config)
        assert result == "a cat, sitting, on a mat"

    def test_prefix_applied(self):
        """Prefix should be prepended."""
        config = CaptionConfig(prefix="masterpiece")
        result = process_caption("a cat", config)
        assert result == "masterpiece a cat"

    def test_suffix_applied(self):
        """Suffix should be appended."""
        config = CaptionConfig(suffix="high quality")
        result = process_caption("a cat", config)
        assert result == "a cat high quality"

    def test_prefix_and_suffix(self):
        """Both prefix and suffix applied."""
        config = CaptionConfig(prefix="best quality", suffix="4k")
        result = process_caption("a cat", config)
        assert result == "best quality a cat 4k"


class TestCaptionDropout:
    """Caption dropout tests."""

    def test_dropout_rate_zero_keeps_caption(self):
        """Zero dropout rate keeps caption."""
        config = CaptionConfig(caption_dropout_rate=0.0)
        rng = random.Random(42)
        result = process_caption("a cat", config, rng)
        assert result == "a cat"

    def test_dropout_rate_one_drops_caption(self):
        """100% dropout rate drops caption."""
        config = CaptionConfig(caption_dropout_rate=1.0)
        rng = random.Random(42)
        result = process_caption("a cat", config, rng)
        assert result == ""

    def test_dropout_is_probabilistic(self):
        """Dropout should vary with RNG seed."""
        config = CaptionConfig(caption_dropout_rate=0.5)
        results = set()
        for seed in range(100):
            rng = random.Random(seed)
            results.add(process_caption("a cat", config, rng))
        # Should have both empty and non-empty results
        assert "" in results
        assert "a cat" in results


class TestTagShuffle:
    """Tag shuffle tests."""

    def test_shuffle_changes_order(self):
        """Shuffle should change tag order."""
        config = CaptionConfig(shuffle_caption=True)
        rng = random.Random(42)
        result = process_caption("a, b, c, d, e", config, rng)
        # Tags should be present but possibly reordered
        result_tags = set(result.split(", "))
        assert result_tags == {"a", "b", "c", "d", "e"}

    def test_shuffle_deterministic_with_seed(self):
        """Same seed should produce same shuffle."""
        config = CaptionConfig(shuffle_caption=True)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        result1 = process_caption("a, b, c, d, e", config, rng1)
        result2 = process_caption("a, b, c, d, e", config, rng2)
        assert result1 == result2

    def test_shuffle_different_seeds_differ(self):
        """Different seeds should (usually) produce different shuffles."""
        config = CaptionConfig(shuffle_caption=True)
        rng1 = random.Random(1)
        rng2 = random.Random(2)
        # Run multiple times to ensure we see different results
        results = set()
        for seed in range(20):
            rng = random.Random(seed)
            results.add(process_caption("a, b, c, d, e, f, g", config, rng))
        assert len(results) > 1  # Should have multiple different orderings


class TestTagDropout:
    """Tag dropout tests."""

    def test_tag_dropout_zero_keeps_all(self):
        """Zero tag dropout keeps all tags."""
        config = CaptionConfig(caption_tag_dropout_rate=0.0)
        rng = random.Random(42)
        result = process_caption("a, b, c", config, rng)
        assert result == "a, b, c"

    def test_tag_dropout_one_drops_all(self):
        """100% tag dropout drops all flex tags."""
        config = CaptionConfig(caption_tag_dropout_rate=1.0)
        rng = random.Random(42)
        result = process_caption("a, b, c", config, rng)
        assert result == ""

    def test_tag_dropout_is_probabilistic(self):
        """Tag dropout should vary per tag."""
        config = CaptionConfig(caption_tag_dropout_rate=0.5)
        results = set()
        for seed in range(50):
            rng = random.Random(seed)
            results.add(process_caption("a, b, c, d, e", config, rng))
        # Should have varying number of tags
        assert len(results) > 1


class TestProtectedTags:
    """Protected tags tests."""

    def test_protected_tags_survive_dropout(self):
        """Protected tags should not be dropped."""
        config = CaptionConfig(
            caption_tag_dropout_rate=1.0,  # Drop everything
            protected_tags={"cat", "dog"},
        )
        rng = random.Random(42)
        result = process_caption("cat, bird, dog, fish", config, rng)
        result_tags = set(result.split(", "))
        assert "cat" in result_tags
        assert "dog" in result_tags
        assert "bird" not in result_tags
        assert "fish" not in result_tags

    def test_protected_tags_case_insensitive(self):
        """Protected tags matching should be case-insensitive."""
        config = CaptionConfig(
            caption_tag_dropout_rate=1.0,
            protected_tags={"CAT"},
        )
        rng = random.Random(42)
        result = process_caption("cat, bird", config, rng)
        assert "cat" in result


class TestWildcards:
    """Wildcard resolution tests."""

    def test_wildcard_picks_one_option(self):
        """Wildcard should resolve to one option."""
        config = CaptionConfig(enable_wildcard=True)
        rng = random.Random(42)
        result = process_caption("a {cat|dog|bird}", config, rng)
        assert result in ["a cat", "a dog", "a bird"]

    def test_wildcard_deterministic(self):
        """Same seed should produce same wildcard choice."""
        config = CaptionConfig(enable_wildcard=True)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        result1 = process_caption("a {cat|dog}", config, rng1)
        result2 = process_caption("a {cat|dog}", config, rng2)
        assert result1 == result2

    def test_wildcard_escaped_braces(self):
        """Escaped braces should be preserved."""
        config = CaptionConfig(enable_wildcard=True)
        rng = random.Random(42)
        result = process_caption("a {{cat}} with {dog|bird}", config, rng)
        assert "{cat}" in result

    def test_multiline_wildcard_picks_line(self):
        """With wildcard enabled, random line is picked."""
        config = CaptionConfig(enable_wildcard=True)
        results = set()
        for seed in range(50):
            rng = random.Random(seed)
            results.add(process_caption("line1\nline2\nline3", config, rng))
        assert results == {"line1", "line2", "line3"}


class TestKeepTokens:
    """Keep tokens (fixed prefix) tests."""

    def test_keep_tokens_preserves_first_n(self):
        """First N tokens should be preserved and not shuffled."""
        config = CaptionConfig(keep_tokens=2, shuffle_caption=True)
        rng = random.Random(42)
        result = process_caption("first, second, a, b, c", config, rng)
        tokens = result.split(", ")
        # First two should always be first and second
        assert tokens[0] == "first"
        assert tokens[1] == "second"

    def test_keep_tokens_separator(self):
        """Separator should split fixed and flex regions."""
        config = CaptionConfig(keep_tokens_separator="|||", shuffle_caption=True)
        rng = random.Random(42)
        result = process_caption("fixed1, fixed2 ||| a, b, c", config, rng)
        tokens = result.split(", ")
        assert tokens[0] == "fixed1"
        assert tokens[1] == "fixed2"

    def test_keep_tokens_separator_with_suffix(self):
        """Second separator marks fixed suffix."""
        config = CaptionConfig(keep_tokens_separator="|||", shuffle_caption=True)
        rng = random.Random(42)
        result = process_caption("prefix ||| a, b, c ||| suffix", config, rng)
        tokens = result.split(", ")
        assert tokens[0] == "prefix"
        assert tokens[-1] == "suffix"


class TestTokenWarmup:
    """Token warmup tests."""

    def test_warmup_limits_tokens_at_start(self):
        """At step 0, should have minimum tokens."""
        config = CaptionConfig(token_warmup_step=100, token_warmup_min=1)
        result = process_caption("a, b, c, d, e", config, current_step=0, max_train_steps=100)
        tokens = result.split(", ")
        assert len(tokens) == 1

    def test_warmup_full_tokens_at_end(self):
        """At/after warmup step, should have all tokens."""
        config = CaptionConfig(token_warmup_step=100, token_warmup_min=1)
        result = process_caption("a, b, c, d, e", config, current_step=100, max_train_steps=100)
        tokens = result.split(", ")
        assert len(tokens) == 5

    def test_warmup_gradual_increase(self):
        """Tokens should increase gradually during warmup."""
        config = CaptionConfig(token_warmup_step=100, token_warmup_min=1)
        counts = []
        for step in [0, 25, 50, 75, 100]:
            result = process_caption("a, b, c, d, e", config, current_step=step, max_train_steps=100)
            counts.append(len(result.split(", ")))
        # Should be increasing
        assert counts == sorted(counts)
        assert counts[0] < counts[-1]


class TestNeedsProcessing:
    """Test needs_processing() helper."""

    def test_default_config_no_processing(self):
        """Default config doesn't need processing."""
        config = CaptionConfig()
        assert not config.needs_processing()

    def test_shuffle_needs_processing(self):
        """Shuffle enabled needs processing."""
        config = CaptionConfig(shuffle_caption=True)
        assert config.needs_processing()

    def test_dropout_needs_processing(self):
        """Dropout enabled needs processing."""
        config = CaptionConfig(caption_dropout_rate=0.1)
        assert config.needs_processing()

    def test_wildcard_needs_processing(self):
        """Wildcard enabled needs processing."""
        config = CaptionConfig(enable_wildcard=True)
        assert config.needs_processing()

    def test_prefix_suffix_no_processing(self):
        """Prefix/suffix alone doesn't need per-batch processing."""
        config = CaptionConfig(prefix="masterpiece", suffix="4k")
        assert not config.needs_processing()
