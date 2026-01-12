from typing import Any

import torch

from library.strategies.base.encoding import TextEncodingStrategy
from library.strategies.base.tokenization import TokenizeStrategy
from library.strategies.sd.tokenization import SdTokenizeStrategy


class SdTextEncodingStrategy(TextEncodingStrategy):
    """
    Text encoding strategy for SD1.5 and SD2.0.
    """

    def __init__(self, clip_skip: int | None = None) -> None:
        self.clip_skip = clip_skip

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode tokens.

        Args:
            tokenize_strategy: TokenizeStrategy instance
            models: List of models
            tokens: List of token tensors

        Returns:
            List of encoded tensors
        """
        text_encoder = models[0]
        tokens_tensor = tokens[0]
        assert isinstance(tokenize_strategy, SdTokenizeStrategy)
        sd_tokenize_strategy: SdTokenizeStrategy = tokenize_strategy

        # tokens_tensor: b,n,77
        b_size = tokens_tensor.size()[0]
        max_token_length = tokens_tensor.size()[1] * tokens_tensor.size()[2]
        model_max_length = sd_tokenize_strategy.tokenizer.model_max_length
        tokens_tensor = tokens_tensor.reshape((-1, model_max_length))  # batch_size*3, 77

        tokens_tensor = tokens_tensor.to(text_encoder.device)

        if self.clip_skip is None:
            encoder_hidden_states = text_encoder(tokens_tensor)[0]
        else:
            enc_out = text_encoder(tokens_tensor, output_hidden_states=True, return_dict=True)
            encoder_hidden_states = enc_out["hidden_states"][-self.clip_skip]
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

        # bs*3, 77, 768 or 1024
        encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

        if max_token_length != model_max_length:
            v1 = sd_tokenize_strategy.tokenizer.pad_token_id == sd_tokenize_strategy.tokenizer.eos_token_id
            if not v1:
                # v2: Restore the triplet of <BOS>...<EOS> <PAD> ... to <BOS>...<EOS> <PAD> ...
                states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
                for i in range(1, max_token_length, model_max_length):
                    chunk = encoder_hidden_states[:, i : i + model_max_length - 2]  # From after <BOS> to before the last
                    if i > 0:
                        for j in range(len(chunk)):
                            if tokens_tensor[j, 1] == sd_tokenize_strategy.tokenizer.eos_token:
                                # Empty, i.e., <BOS> <EOS> <PAD> ... pattern
                                chunk[j, 0] = chunk[j, 1]  # Copy the value of the next <PAD>
                    states_list.append(chunk)  # From after <BOS> to before <EOS>
                states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # Either <EOS> or <PAD>
                encoder_hidden_states = torch.cat(states_list, dim=1)
            else:
                # v1: Restore the triplet of <BOS>...<EOS> to <BOS>...<EOS>
                states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
                for i in range(1, max_token_length, model_max_length):
                    states_list.append(encoder_hidden_states[:, i : i + model_max_length - 2])  # From after <BOS> to before <EOS>
                states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
                encoder_hidden_states = torch.cat(states_list, dim=1)

        return [encoder_hidden_states]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        encoder_hidden_states = self.encode_tokens(tokenize_strategy, models, tokens)[0]

        weights_tensor = weights[0].to(encoder_hidden_states.device)

        # apply weights
        if weights_tensor.shape[1] == 1:  # no max_token_length
            # weights: ((b, 1, 77), (b, 1, 77)), hidden_states: (b, 77, 768), (b, 77, 768)
            encoder_hidden_states = encoder_hidden_states * weights_tensor.squeeze(1).unsqueeze(2)
        else:
            # weights: ((b, n, 77), (b, n, 77)), hidden_states: (b, n*75+2, 768), (b, n*75+2, 768)
            for i in range(weights_tensor.shape[1]):
                encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] = encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] * weights_tensor[
                    :, i, 1:-1
                ].unsqueeze(-1)

        return [encoder_hidden_states]
