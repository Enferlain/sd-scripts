import torch
from unittest.mock import Mock

from library.models.sd.text_encoder import get_hidden_states_sd, apply_hidden_state_weights_sd


def _make_sd_tokenizer(*, pad_token_id: int = 49407, eos_token_id: int = 49407):
    tokenizer = Mock()
    tokenizer.model_max_length = 77
    tokenizer.pad_token_id = pad_token_id
    tokenizer.eos_token_id = eos_token_id
    tokenizer.eos_token = eos_token_id
    return tokenizer


def _make_sd_text_encoder(hidden_dim: int = 768):
    encoder = Mock()
    encoder.device = torch.device("cpu")
    encoder.text_model = Mock()
    encoder.text_model.final_layer_norm = Mock(side_effect=lambda x: x)

    def encoder_call(tokens, output_hidden_states=False, return_dict=False):
        batch_size, seq_len = tokens.shape
        if output_hidden_states:
            hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(13)]
            return {
                "hidden_states": hidden_states,
                "last_hidden_state": hidden_states[-1],
            }
        return (torch.randn(batch_size, seq_len, hidden_dim),)

    encoder.__call__ = encoder_call
    encoder.side_effect = encoder_call
    return encoder


class TestGetHiddenStatesSD:
    def test_basic_encoding(self):
        tokenizer = _make_sd_tokenizer()
        text_encoder = _make_sd_text_encoder()
        input_ids = torch.randint(0, 1000, (2, 1, 77))

        hidden_states = get_hidden_states_sd(input_ids, tokenizer, text_encoder)

        assert hidden_states.shape == (2, 77, 768)

    def test_clip_skip_path(self):
        tokenizer = _make_sd_tokenizer()
        text_encoder = _make_sd_text_encoder()
        input_ids = torch.randint(0, 1000, (1, 1, 77))

        hidden_states = get_hidden_states_sd(input_ids, tokenizer, text_encoder, clip_skip=2)

        assert hidden_states.shape == (1, 77, 768)
        text_encoder.text_model.final_layer_norm.assert_called()

    def test_multi_chunk_v1_reconstruction(self):
        tokenizer = _make_sd_tokenizer()
        text_encoder = _make_sd_text_encoder()
        input_ids = torch.randint(0, 1000, (1, 3, 77))

        hidden_states = get_hidden_states_sd(input_ids, tokenizer, text_encoder)

        assert hidden_states.shape == (1, 227, 768)


class TestApplyHiddenStateWeightsSD:
    def test_single_chunk_weights(self):
        hidden_states = torch.ones(1, 77, 4)
        weights = torch.ones(1, 1, 77) * 1.5

        weighted = apply_hidden_state_weights_sd(hidden_states, weights)

        assert weighted.shape == hidden_states.shape
        torch.testing.assert_close(weighted[:, 0, :], torch.full((1, 4), 1.5))

    def test_multi_chunk_weights(self):
        hidden_states = torch.ones(1, 152, 2)
        weights = torch.ones(1, 2, 77) * 2.0

        weighted = apply_hidden_state_weights_sd(hidden_states, weights)

        assert weighted.shape == hidden_states.shape
        torch.testing.assert_close(weighted[:, 1, :], torch.full((1, 2), 2.0))
