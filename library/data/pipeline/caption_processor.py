"""
Caption processing utilities for the data pipeline.

Handles all caption augmentations:
- Tag shuffle (randomize tag order)
- Caption dropout (drop entire caption)
- Tag dropout (drop individual tags)
- Wildcard resolution ({cat|dog} → pick one)
- Token warmup (gradually increase tags during training)
- Protected tokens (immune to dropout/shuffle)
- Prefix/suffix application

Ported from legacy BaseDataset.process_caption() with clean interface.
"""

import random
import re
from dataclasses import dataclass, field


@dataclass
class CaptionConfig:
    """Configuration for caption processing.

    Attributes:
        shuffle_caption: Randomize order of caption parts (except fixed tokens).
        caption_dropout_rate: Probability of dropping entire caption to "".
        caption_dropout_every_n_epochs: Drop caption every N epochs (0=disabled).
        caption_tag_dropout_rate: Probability of dropping each individual part.
        enable_wildcard: Enable {option1|option2} wildcard resolution.
        keep_tokens: Number of tokens at start to keep fixed (legacy mode).
        keep_tokens_separator: Separator to mark fixed regions (e.g. "|||").
            Tokens before first separator are fixed prefix.
            Tokens after second separator are fixed suffix.
        caption_separator: Separator for splitting caption into parts (default ", ").
        secondary_separator: Replaced with caption_separator if set.
        prefix: Prepend to caption.
        suffix: Append to caption.
        token_warmup_min: Minimum tokens during warmup.
        token_warmup_step: Training step at which full tokens are used (0=disabled).
            If < 1, treated as fraction of max_train_steps.
        protected_tags: Parts that are immune to dropout (case-insensitive match).
    """

    shuffle_caption: bool = False
    caption_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0
    caption_tag_dropout_rate: float = 0.0
    enable_wildcard: bool = False
    keep_tokens: int = 0
    keep_tokens_separator: str = ""
    caption_separator: str = ", "
    secondary_separator: str = ""
    prefix: str = ""
    suffix: str = ""
    token_warmup_min: int = 0
    token_warmup_step: int = 0
    protected_tags: set[str] = field(default_factory=set)

    def needs_processing(self) -> bool:
        """Check if any processing is needed (determines if TE can be cached)."""
        return (
            self.shuffle_caption
            or self.caption_dropout_rate > 0
            or self.caption_dropout_every_n_epochs > 0
            or self.caption_tag_dropout_rate > 0
            or self.enable_wildcard
            or self.token_warmup_step > 0
        )


def process_caption(
    caption: str,
    config: CaptionConfig,
    rng: random.Random | None = None,
    current_step: int = 0,
    current_epoch: int = 0,
    max_train_steps: int = 0,
) -> str:
    """
    Process a caption with configured augmentations.

    Args:
        caption: Raw caption text.
        config: Processing configuration.
        rng: Random generator for deterministic processing. If None, uses global random.
        current_step: Current training step (for token warmup).
        current_epoch: Current epoch (for epoch-based dropout).
        max_train_steps: Total training steps (for warmup calculation).

    Returns:
        Processed caption string.
    """
    if rng is None:
        rng = random.Random()

    # Apply prefix/suffix
    if config.prefix:
        caption = config.prefix + " " + caption
    if config.suffix:
        caption = caption + " " + config.suffix

    # Check caption dropout
    is_dropout = config.caption_dropout_rate > 0 and rng.random() < config.caption_dropout_rate
    if config.caption_dropout_every_n_epochs > 0:
        is_dropout = is_dropout or (current_epoch % config.caption_dropout_every_n_epochs == 0)

    if is_dropout:
        return ""

    # Process wildcards
    if config.enable_wildcard:
        caption = _process_wildcards(caption, rng)
    else:
        # If not using wildcards, take first line only
        caption = caption.split("\n")[0]

    # Tag-level processing (shuffle, dropout, warmup)
    if config.shuffle_caption or config.token_warmup_step > 0 or config.caption_tag_dropout_rate > 0:
        caption = _process_tags(caption, config, rng, current_step, max_train_steps)

    # Replace secondary separator
    if config.secondary_separator:
        caption = caption.replace(config.secondary_separator, config.caption_separator)

    return caption


def _process_wildcards(caption: str, rng: random.Random) -> str:
    """Resolve wildcards like {option1|option2|option3}."""
    # Handle multiline: pick random line
    if "\n" in caption:
        caption = rng.choice(caption.split("\n"))

    # Escape {{ and }} to avoid matching
    replacer1 = "⦅"
    replacer2 = "⦆"
    while replacer1 in caption or replacer2 in caption:
        replacer1 += "⦅"
        replacer2 += "⦆"

    caption = caption.replace("{{", replacer1).replace("}}", replacer2)

    # Replace {option1|option2} with random choice
    def replace_wildcard(match: re.Match) -> str:
        options = match.group(1).split("|")
        return rng.choice(options)

    caption = re.sub(r"\{([^}]+)\}", replace_wildcard, caption)

    # Unescape
    caption = caption.replace(replacer1, "{").replace(replacer2, "}")

    return caption


def _process_tags(
    caption: str,
    config: CaptionConfig,
    rng: random.Random,
    current_step: int,
    max_train_steps: int,
) -> str:
    """Process individual tags: shuffle, dropout, warmup."""
    sep = config.caption_separator

    # Split into fixed (prefix), flex (shuffleable), and fixed_suffix parts
    fixed_tokens: list[str] = []
    flex_tokens: list[str] = []
    fixed_suffix_tokens: list[str] = []

    if config.keep_tokens_separator and config.keep_tokens_separator in caption:
        # Use separator to determine fixed regions
        parts = caption.split(config.keep_tokens_separator)
        fixed_part = parts[0]
        flex_part = parts[1] if len(parts) > 1 else ""
        fixed_suffix_part = parts[2] if len(parts) > 2 else ""

        fixed_tokens = [t.strip() for t in fixed_part.split(sep) if t.strip()]
        flex_tokens = [t.strip() for t in flex_part.split(sep) if t.strip()]
        fixed_suffix_tokens = [t.strip() for t in fixed_suffix_part.split(sep) if t.strip()]
    else:
        # Use keep_tokens count
        tokens = [t.strip() for t in caption.strip().split(sep) if t.strip()]
        if config.keep_tokens > 0:
            fixed_tokens = tokens[: config.keep_tokens]
            flex_tokens = tokens[config.keep_tokens :]
        else:
            flex_tokens = tokens

    # Token warmup: limit flex tokens based on training progress
    warmup_step = config.token_warmup_step
    if warmup_step > 0:
        # Handle fractional warmup_step (e.g., 0.1 = 10% of training)
        if warmup_step < 1 and max_train_steps > 0:
            warmup_step = int(warmup_step * max_train_steps)

        if current_step < warmup_step and warmup_step > 0:
            progress = current_step / warmup_step
            tokens_len = int(config.token_warmup_min + progress * (len(flex_tokens) - config.token_warmup_min))
            flex_tokens = flex_tokens[:tokens_len]

    # Part dropout (respecting protected tags)
    if config.caption_tag_dropout_rate > 0:
        protected_lower = {t.lower() for t in config.protected_tags}
        flex_tokens = [t for t in flex_tokens if t.lower() in protected_lower or rng.random() >= config.caption_tag_dropout_rate]

    # Shuffle flex tokens
    if config.shuffle_caption:
        rng.shuffle(flex_tokens)

    # Reassemble
    all_tokens = fixed_tokens + flex_tokens + fixed_suffix_tokens
    return sep.join(all_tokens)
