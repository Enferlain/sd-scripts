import torch

from transformers import CLIPTokenizer


def get_hidden_states_sd(
    input_ids: torch.Tensor,
    tokenizer: CLIPTokenizer,
    text_encoder: torch.nn.Module,
    clip_skip: int | None = None,
    weight_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Encode SD/SD2 CLIP input IDs into hidden states.

    Args:
        input_ids: Token tensor shaped ``[batch, n_chunks, 77]``.
        tokenizer: CLIP tokenizer used for sequence sizing.
        text_encoder: SD text encoder module.
        clip_skip: Optional CLIP skip value.
        weight_dtype: Optional output dtype conversion.

    Returns:
        Hidden states shaped ``[batch, sequence, hidden_dim]``.
    """
    batch_size = input_ids.size(0)
    max_token_length = input_ids.size(1) * input_ids.size(2)
    model_max_length = tokenizer.model_max_length

    flat_input_ids = input_ids.reshape((-1, model_max_length)).to(text_encoder.device)

    if clip_skip is None:
        encoder_hidden_states = text_encoder(flat_input_ids)[0]
    else:
        enc_out = text_encoder(flat_input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    encoder_hidden_states = encoder_hidden_states.reshape((batch_size, -1, encoder_hidden_states.shape[-1]))

    if max_token_length != model_max_length:
        if tokenizer.pad_token_id != tokenizer.eos_token_id:
            # Restore <BOS>...<EOS> <PAD> ... layout for v2-style tokenizers.
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]
            for i in range(1, max_token_length, model_max_length):
                chunk = encoder_hidden_states[:, i : i + model_max_length - 2]
                if i > 0:
                    for j in range(len(chunk)):
                        if flat_input_ids[j, 1] == tokenizer.eos_token:
                            chunk[j, 0] = chunk[j, 1]
                states_list.append(chunk)
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
            encoder_hidden_states = torch.cat(states_list, dim=1)
        else:
            # Restore <BOS>...<EOS> layout for v1-style tokenizers.
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]
            for i in range(1, max_token_length, model_max_length):
                states_list.append(encoder_hidden_states[:, i : i + model_max_length - 2])
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
            encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def apply_hidden_state_weights_sd(encoder_hidden_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Apply prompt weights to SD hidden states.

    Args:
        encoder_hidden_states: Hidden states from ``get_hidden_states_sd``.
        weights: Token weights shaped ``[batch, n_chunks, 77]``.

    Returns:
        Weighted hidden states tensor.
    """
    weights = weights.to(encoder_hidden_states.device)

    if weights.shape[1] == 1:
        return encoder_hidden_states * weights.squeeze(1).unsqueeze(2)

    for i in range(weights.shape[1]):
        encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] = encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] * weights[
            :, i, 1:-1
        ].unsqueeze(-1)

    return encoder_hidden_states
