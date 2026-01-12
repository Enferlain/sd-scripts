import logging
import os
from typing import Optional, Any

import torch
from transformers import CLIPTokenizer

from library.constants import re_attention
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TokenizeStrategy:
    _strategy = None  # strategy instance: actual strategy class

    _re_attention = re_attention

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TokenizeStrategy"]:
        return cls._strategy

    def _load_tokenizer(self, model_class: Any, model_id: str, subfolder: str | None = None, tokenizer_cache_dir: str | None = None) -> Any:
        """
        Load tokenizer from cache or download it.

        Args:
            model_class: Tokenizer class (e.g. CLIPTokenizer)
            model_id: Model ID (e.g. "openai/clip-vit-large-patch14")
            subfolder: Subfolder in the model repo (e.g. "tokenizer")
            tokenizer_cache_dir: Directory to cache the tokenizer

        Returns:
            Tokenizer instance
        """
        tokenizer = None
        local_tokenizer_path: str | None = None
        if tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(tokenizer_cache_dir, model_id.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = model_class.from_pretrained(local_tokenizer_path)  # same for v1 and v2

        if tokenizer is None:
            tokenizer = model_class.from_pretrained(model_id, subfolder=subfolder)

        if local_tokenizer_path is not None and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        return tokenizer

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        """
        Tokenize text.

        Args:
            text: Text or list of text to tokenize

        Returns:
            List of token tensors
        """
        raise NotImplementedError

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Tokenize text with weights.

        Args:
            text: Text or list of text to tokenize

        Returns:
            Tuple of lists of token tensors and weight tensors
            ([tokens1, tokens2, ...], [weights1, weights2, ...])
        """
        raise NotImplementedError

    def _get_weighted_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get weighted input ids.

        Args:
            tokenizer: Tokenizer instance
            text: Text to tokenize
            max_length: Max length of tokens (including starting and ending tokens)

        Returns:
            Tuple of token tensor and weight tensor
        """

        def parse_prompt_attention(text):
            r"""
            Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
            Accepted tokens are:
            (abc) - increases attention to abc by a multiplier of 1.1
            (abc:3.12) - increases attention to abc by a multiplier of 3.12
            [abc] - decreases attention to abc by a multiplier of 1.1
            \( - literal character '('
            \[ - literal character '['
            \) - literal character ')'
            \] - literal character ']'
            \\ - literal character '\'
            anything else - just text
            >>> parse_prompt_attention('normal text')
            [['normal text', 1.0]]
            >>> parse_prompt_attention('an (important) word')
            [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
            >>> parse_prompt_attention('(unbalanced')
            [['unbalanced', 1.1]]
            >>> parse_prompt_attention('\(literal\]')
            [['(literal]', 1.0]]
            >>> parse_prompt_attention('(unnecessary)(parens)')
            [['unnecessaryparens', 1.1]]
            >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
            [['a ', 1.0],
            ['house', 1.5730000000000004],
            [' ', 1.1],
            ['on', 1.0],
            [' a ', 1.1],
            ['hill', 0.55],
            [', sun, ', 1.1],
            ['sky', 1.4641000000000006],
            ['.', 1.1]]
            """

            res: list[list[str | float]] = []
            round_brackets = []
            square_brackets = []

            round_bracket_multiplier = 1.1
            square_bracket_multiplier = 1 / 1.1

            def multiply_range(start_position, multiplier):
                for p in range(start_position, len(res)):
                    res[p][1] *= multiplier

            for m in TokenizeStrategy._re_attention.finditer(text):
                text = m.group(0)
                weight = m.group(1)

                if text.startswith("\\"):
                    res.append([text[1:], 1.0])
                elif text == "(":
                    round_brackets.append(len(res))
                elif text == "[":
                    square_brackets.append(len(res))
                elif weight is not None and len(round_brackets) > 0:
                    multiply_range(round_brackets.pop(), float(weight))
                elif text == ")" and len(round_brackets) > 0:
                    multiply_range(round_brackets.pop(), round_bracket_multiplier)
                elif text == "]" and len(square_brackets) > 0:
                    multiply_range(square_brackets.pop(), square_bracket_multiplier)
                else:
                    res.append([text, 1.0])

            for pos in round_brackets:
                multiply_range(pos, round_bracket_multiplier)

            for pos in square_brackets:
                multiply_range(pos, square_bracket_multiplier)

            if len(res) == 0:
                res = [["", 1.0]]

            # merge runs of identical weights
            i = 0
            while i + 1 < len(res):
                if res[i][1] == res[i + 1][1]:
                    res[i][0] = str(res[i][0]) + str(res[i + 1][0])
                    res.pop(i + 1)
                else:
                    i += 1

            return res

        def get_prompts_with_weights(text: str, max_length: int):
            r"""
            Tokenize a list of prompts and return its tokens with weights of each token. max_length does not include starting and ending token.

            No padding, starting or ending token is included.
            """
            truncated = False

            texts_and_weights = parse_prompt_attention(text)
            tokens = []
            weights = []
            for word, weight in texts_and_weights:
                # tokenize and discard the starting and the ending token
                token = tokenizer(word).input_ids[1:-1]
                tokens += token
                # copy the weight by length of token
                weights += [weight] * len(token)
                # stop if the text is too long (longer than truncation limit)
                if len(tokens) > max_length:
                    truncated = True
                    break
            # truncate
            if len(tokens) > max_length:
                truncated = True
                tokens = tokens[:max_length]
                weights = weights[:max_length]
            if truncated:
                logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
            return tokens, weights

        def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad):
            r"""
            Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
            """
            tokens = [bos] + tokens + [eos] + [pad] * (max_length - 2 - len(tokens))
            weights = [1.0] + weights + [1.0] * (max_length - 1 - len(weights))
            return tokens, weights

        if max_length is None:
            max_length = tokenizer.model_max_length

        tokens, weights = get_prompts_with_weights(text, max_length - 2)
        tokens, weights = pad_tokens_and_weights(
            tokens, weights, max_length, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        return torch.tensor(tokens).unsqueeze(0), torch.tensor(weights).unsqueeze(0)

    def _get_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: int | None = None, weighted: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Get input ids for SD1.5/2.0/SDXL.

        Args:
            tokenizer: Tokenizer instance
            text: Text to tokenize
            max_length: Max length of tokens
            weighted: Whether to return weights

        Returns:
            Input ids tensor (or tuple of input ids and weights if weighted=True)
        """
        if max_length is None:
            max_length = tokenizer.model_max_length - 2

        weights: torch.Tensor | None = None
        if weighted:
            input_ids, weights = self._get_weighted_input_ids(tokenizer, text, max_length)
        else:
            input_ids = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids

        if max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # When 77 or more, it becomes "<BOS> .... <EOS> <EOS> <EOS>" totaling 227 etc., so convert it to "<BOS>...<EOS>" triplet
                # 1111 seems to split by comma, but for now simply
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # When 77 or more, it becomes "<BOS> .... <EOS> <PAD> <PAD>..." totaling 227 etc., so convert it to "<BOS>...<EOS> <PAD> <PAD> ..." triplet
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # If the end is <EOS> <PAD> or <PAD> <PAD>, nothing needs to be done
                    # If the end is x <PAD/EOS>, change the end to <EOS> (if x <EOS>, no change result)
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # If the beginning is <BOS> <PAD> ..., change it to <BOS> <EOS> <PAD> ...
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77

            if weighted and weights is not None:
                weights = weights.squeeze(0)
                new_weights = torch.ones(input_ids.shape)
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    b = i // (tokenizer.model_max_length - 2)
                    new_weights[b, 1 : 1 + tokenizer.model_max_length - 2] = weights[i : i + tokenizer.model_max_length - 2]
                weights = new_weights

        if weighted and weights is not None:
            return input_ids, weights
        return input_ids
