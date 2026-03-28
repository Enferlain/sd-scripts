"""Shared Hugging Face tokenizer loading helpers for CLIP-family model code."""

import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


def load_tokenizer(
    model_class: Any,
    model_id: str,
    subfolder: str | None = None,
    tokenizer_cache_dir: str | None = None,
) -> Any:
    """Load a tokenizer from cache or the upstream model repo."""
    tokenizer = None
    local_tokenizer_path: str | None = None

    if tokenizer_cache_dir:
        local_tokenizer_path = os.path.join(tokenizer_cache_dir, model_id.replace("/", "_"))
        if os.path.exists(local_tokenizer_path):
            logger.info("load tokenizer from cache: %s", local_tokenizer_path)
            tokenizer = model_class.from_pretrained(local_tokenizer_path)

    if tokenizer is None:
        tokenizer = model_class.from_pretrained(model_id, subfolder=subfolder)

    if local_tokenizer_path is not None and not os.path.exists(local_tokenizer_path):
        logger.info("save Tokenizer to cache: %s", local_tokenizer_path)
        tokenizer.save_pretrained(local_tokenizer_path)

    return tokenizer


__all__ = ["load_tokenizer"]
