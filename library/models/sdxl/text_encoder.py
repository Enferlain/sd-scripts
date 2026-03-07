import torch

from accelerate import Accelerator

from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection


def pool_workaround(
    text_encoder: CLIPTextModelWithProjection | torch.nn.Module, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
) -> torch.Tensor:
    """
    Workaround for CLIP's pooling bug.

    CLIP's pooling function returns the hidden states for the max token id as the pooled output
    instead of the hidden states for the EOS token. If we use Textual Inversion, we need to use
    the hidden states for the EOS token as the pooled output.

    Args:
        text_encoder (CLIPTextModelWithProjection): The text encoder model.
        last_hidden_state (torch.Tensor): The last hidden state from the text encoder.
        input_ids (torch.Tensor): The input token IDs.
        eos_token_id (int): The ID of the EOS token.

    Returns:
        torch.Tensor: The pooled output using the hidden state corresponding to the EOS token.
    """
    # input_ids: b*n,77
    # find index for EOS token

    # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
    # eos_token_index = torch.where(input_ids == eos_token_id)[1]
    # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # Create a mask where the EOS tokens are
    eos_token_mask = (input_ids == eos_token_id).int()

    # Use argmax to find the last index of the EOS token for each element in the batch
    eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
    eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # get hidden states for EOS token
    pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]

    # apply projection: projection may be of different dtype than last_hidden_state
    pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
    pooled_output = pooled_output.to(last_hidden_state.dtype)

    return pooled_output


def get_hidden_states_sdxl(
    max_token_length: int | None,
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel | torch.nn.Module,
    text_encoder2: CLIPTextModelWithProjection | torch.nn.Module,
    weight_dtype: torch.dtype | None = None,
    accelerator: Accelerator | None = None,
    unwrapped_text_encoder2: CLIPTextModelWithProjection | torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get hidden states for SDXL from two text encoders.

    Args:
        max_token_length (int): The maximum token length.
        input_ids1 (torch.Tensor): Input IDs for the first text encoder.
        input_ids2 (torch.Tensor): Input IDs for the second text encoder.
        tokenizer1 (CLIPTokenizer): The first tokenizer.
        tokenizer2 (CLIPTokenizer): The second tokenizer.
        text_encoder1 (CLIPTextModel): The first text encoder.
        text_encoder2 (CLIPTextModelWithProjection): The second text encoder.
        weight_dtype (torch.dtype, optional): The weight data type. Defaults to None.
        accelerator (Accelerator, optional): The accelerator for distributed training. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - hidden_states1 (torch.Tensor): Hidden states from the first text encoder.
            - hidden_states2 (torch.Tensor): Hidden states from the second text encoder.
            - pool2 (torch.Tensor): Pooled output from the second text encoder.
    """
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penultimate layer

    # pool2 = enc_out["text_embeds"]
    pool_encoder2 = unwrapped_text_encoder2
    if pool_encoder2 is None:
        pool_encoder2 = text_encoder2 if accelerator is None else accelerator.unwrap_model(text_encoder2)
    pool2 = pool_workaround(pool_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: restore <BOS>...<EOS> from three consecutive <BOS>...<EOS>
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i : i + tokenizer1.model_max_length - 2])  # From after <BOS> to before <EOS>
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        # v2: restore <BOS>...<EOS> <PAD> ... from three consecutive <BOS>...<EOS> <PAD> ... sequences.
        # Honestly, I am not sure if this implementation is correct.
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2]  # From after <BOS> to before the last one
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # Empty, i.e., the pattern <BOS> <EOS> <PAD> ...
            #             chunk[j, 0] = chunk[j, 1]  # Copy the value of the next <PAD>
            states_list.append(chunk)  # From after <BOS> to before <EOS>
        states_list.append(hidden_states2[:, -1].unsqueeze(1))  # Either <EOS> or <PAD>
        hidden_states2 = torch.cat(states_list, dim=1)

        # pool uses the first one of n
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional peft training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def encode_input_ids_sdxl(
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel | torch.nn.Module,
    text_encoder2: CLIPTextModelWithProjection | torch.nn.Module,
    weight_dtype: torch.dtype | None = None,
    unwrapped_text_encoder2: CLIPTextModelWithProjection | torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode SDXL token tensors into hidden states and pooled output.

    Args:
        input_ids1: First encoder token tensor shaped ``[batch, n_chunks, 77]``.
        input_ids2: Second encoder token tensor shaped ``[batch, n_chunks, 77]``.
        tokenizer1: First tokenizer.
        tokenizer2: Second tokenizer.
        text_encoder1: First text encoder.
        text_encoder2: Second text encoder.
        weight_dtype: Optional output dtype conversion for hidden states.
        unwrapped_text_encoder2: Optional unwrapped TE2 used only for pooling.

    Returns:
        Tuple of ``(hidden_states1, hidden_states2, pool2)``.
    """
    max_token_length = None if input_ids1.size(1) == 1 else input_ids1.size(1) * input_ids1.size(2)

    input_ids1 = input_ids1.to(next(text_encoder1.parameters()).device)
    input_ids2 = input_ids2.to(next(text_encoder2.parameters()).device)

    return get_hidden_states_sdxl(
        max_token_length,
        input_ids1,
        input_ids2,
        tokenizer1,
        tokenizer2,
        text_encoder1,
        text_encoder2,
        weight_dtype=weight_dtype,
        accelerator=None,
        unwrapped_text_encoder2=unwrapped_text_encoder2,
    )


def apply_hidden_state_weights_sdxl(
    hidden_states1: torch.Tensor,
    hidden_states2: torch.Tensor,
    weights1: torch.Tensor,
    weights2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply prompt weights to SDXL hidden states.

    Args:
        hidden_states1: CLIP-L hidden states.
        hidden_states2: CLIP-G hidden states.
        weights1: CLIP-L token weights.
        weights2: CLIP-G token weights.

    Returns:
        Tuple of weighted hidden states.
    """
    weights_tensors = [weights1.to(hidden_states1.device), weights2.to(hidden_states1.device)]

    if weights_tensors[0].shape[1] == 1:
        hidden_states1 = hidden_states1 * weights_tensors[0].squeeze(1).unsqueeze(2)
        hidden_states2 = hidden_states2 * weights_tensors[1].squeeze(1).unsqueeze(2)
        return hidden_states1, hidden_states2

    for weight, hidden_states in zip(weights_tensors, [hidden_states1, hidden_states2]):
        for i in range(weight.shape[1]):
            hidden_states[:, i * 75 + 1 : i * 75 + 76] = hidden_states[:, i * 75 + 1 : i * 75 + 76] * weight[:, i, 1:-1].unsqueeze(-1)

    return hidden_states1, hidden_states2
