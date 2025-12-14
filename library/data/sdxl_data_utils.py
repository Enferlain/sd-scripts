import os
import logging
from typing import Any

from transformers import CLIPTokenizer

from library.constants import TOKENIZER1_PATH, TOKENIZER2_PATH
from library.utils.common_utils import setup_logging # todo is it needed?

setup_logging()  # todo is it needed?
logger = logging.getLogger(__name__)


def load_tokenizers(args: Any):
    logger.info("prepare tokenizers")

    original_paths = [TOKENIZER1_PATH, TOKENIZER2_PATH]
    tokeniers = []

    tokenizer_cache_dir = getattr(args, "tokenizer_cache_dir", None)

    for i, original_path in enumerate(original_paths):
        tokenizer: CLIPTokenizer = None
        local_tokenizer_path = None
        if tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(tokenizer_cache_dir, original_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(original_path)

        if tokenizer_cache_dir and local_tokenizer_path and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        if i == 1:
            tokenizer.pad_token_id = 0  # fix pad token id to make same as open clip tokenizer

        tokeniers.append(tokenizer)

    max_token_length = getattr(args, "max_token_length", None)
    if max_token_length is not None:
        logger.info(f"update token length: {max_token_length}")

    return tokeniers
