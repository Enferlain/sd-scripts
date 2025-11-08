import torch
import argparse
import random
import math
import re
from .utils import setup_logging

setup_logging()
import logging

import pywt
from library import train_util
from diffusers import DDPMScheduler
import ast
import json

from torch import Tensor
from torch import nn
from torch.types import Number
import torch.nn.functional as F
from typing import List, Optional, Union, Protocol, Any, Mapping

logger = logging.getLogger(__name__)


def prepare_scheduler_for_custom_training(noise_scheduler, device, mu=None, b=None):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)
    noise_scheduler.all_timesteps = torch.linspace(0, 999, 1000).to(dtype=torch.long, device=device)

    # If user specified Laplace-based sampling arguments, compute them
    if mu is not None and b is not None:
        # Make sure mu and b are floats:
        mu = float(mu)
        b = float(b)

        logger.info(f"Using Laplace-weighted timesteps with mu={mu}, b={b}")

        log_snr = all_snr.log()  # log of snr
        # laplace_weights formula (paper style)
        laplace_weights = ((log_snr - mu).abs() / (-b)).exp() / (2 * b)
        laplace_weights /= laplace_weights.mean()

        noise_scheduler.laplace_weights = laplace_weights.to(device)
        logger.info("Laplace weights computed and stored in noise_scheduler.laplace_weights")


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    # fix beta: zero terminal SNR
    logger.info(f"fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # logger.info(f"original: {noise_scheduler.betas}")
    # logger.info(f"fixed: {betas}")

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod


def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    loss = loss * snr_weight
    return loss


def scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler):
    scale = get_snr_scale(timesteps, noise_scheduler)
    loss = loss * scale
    return loss


def get_snr_scale(timesteps, noise_scheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    scale = snr_t / (snr_t + 1)
    # # show debug info
    # logger.info(f"timesteps: {timesteps}, snr_t: {snr_t}, scale: {scale}")
    return scale


def add_v_prediction_like_loss(loss, timesteps, noise_scheduler, v_pred_like_loss):
    scale = get_snr_scale(timesteps, noise_scheduler)
    # logger.info(f"add v-prediction like loss: {v_pred_like_loss}, scale: {scale}, loss: {loss}, time: {timesteps}")
    loss = loss + loss / scale * torch.full_like(input=scale, fill_value=v_pred_like_loss)
    return loss


def apply_debiased_estimation(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler: DDPMScheduler,
                              v_prediction=False, image_size=None):
    # Check if we have SNR values available
    if not (hasattr(noise_scheduler, "all_snr") or hasattr(noise_scheduler, "get_snr_for_timestep")):
        return loss

    if hasattr(noise_scheduler, "get_snr_for_timestep") and not callable(noise_scheduler.get_snr_for_timestep):
        return loss

    # Get SNR values with image_size consideration
    if hasattr(noise_scheduler, "get_snr_for_timestep") and callable(noise_scheduler.get_snr_for_timestep):
        snr_t: torch.Tensor = noise_scheduler.get_snr_for_timestep(timesteps, image_size)
    else:
        snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])

    # Cap the SNR to avoid numerical issues
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)

    # Apply weighting based on prediction type
    if v_prediction:
        weight = 1 / (snr_t + 1)
    else:
        weight = 1 / torch.sqrt(snr_t)
    loss = weight * loss
    return loss


# TODO train_utilと分散しているのでどちらかに寄せる

def parse_wavelet_weights(weights_str):
    if weights_str is None:
        return None

    # Try parsing as a dictionary (for formats like "{'ll1':0.1,'lh1':0.01}")
    if weights_str.strip().startswith('{'):
        try:
            return ast.literal_eval(weights_str)
        except (ValueError, SyntaxError) as e1:
            logger.warning(e1)
            try:
                return json.loads(weights_str.replace("'", '"'))
            except json.JSONDecodeError as e2:
                logger.warning(e2)
                pass

    # Parse format like "ll1=0.1,lh1=0.01,hl1=0.01,hh1=0.05"
    result = {}
    for pair in weights_str.split(','):
        if '=' in pair:
            key, value = pair.split('=', 1)
            result[key.strip()] = float(value.strip())

    return result


def add_custom_train_arguments(parser: argparse.ArgumentParser, support_weighted_captions: bool = True):
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper. / 低いタイムステップでの高いlossに対して重みを減らすためのgamma値、低いほど効果が強く、論文では5が推奨",
    )
    parser.add_argument(
        "--scale_v_pred_loss_like_noise_pred",
        action="store_true",
        help="scale v-prediction loss like noise prediction loss / v-prediction lossをnoise prediction lossと同じようにスケーリングする",
    )
    parser.add_argument(
        "--v_pred_like_loss",
        type=float,
        default=None,
        help="add v-prediction like loss multiplied by this value / v-prediction lossをこの値をかけたものをlossに加算する",
    )
    parser.add_argument(
        "--debiased_estimation_loss",
        action="store_true",
        help="debiased estimation loss / debiased estimation loss",
    )
    if support_weighted_captions:
        parser.add_argument(
            "--weighted_captions",
            action="store_true",
            default=False,
            help="Enable weighted captions in the standard style (token:1.3). No commas inside parens, or shuffle/dropout may break the decoder. / 「[token]」、「(token)」「(token:1.3)」のような重み付きキャプションを有効にする。カンマを括弧内に入れるとシャッフルやdropoutで重みづけがおかしくなるので注意",
        )

    parser.add_argument("--wavelet_loss", action="store_true", help="Activate wavelet loss. Default: False")
    parser.add_argument("--wavelet_loss_alpha", type=float, default=0.98, help="Wavelet loss alpha. Default: 0.98")
    parser.add_argument("--wavelet_loss_type",
                        help="Wavelet loss type l1, l2, huber, smooth_l1. Default to --loss_type value.")
    parser.add_argument("--wavelet_loss_delta",
                        help="For loss types that are adjustable via beta/delta/scale/huber_c etc. Defaults to huber_c.")
    parser.add_argument("--wavelet_loss_schedule",
                        choices=["constant", "exponential", "snr"],
                        help="For loss types that are schedulable.")
    parser.add_argument("--wavelet_loss_transform", default="swt",
                        help="Wavelet transform type of DWT, SWT, QWT. Default: swt")
    parser.add_argument("--wavelet_loss_wavelet", default="sym7", help="Wavelet. Default: sym7")
    parser.add_argument("--wavelet_loss_level", type=int, default=2,
                        help="Wavelet loss level 1 (main), 2 (details), or 3. Higher levels are available for DWT for higher resolution training. Default: 2")
    # parser.add_argument("--wavelet_loss_rectified_flow", default=True, help="Use rectified flow to estimate clean latents before wavelet loss")
    parser.add_argument("--wavelet_loss_band_level_weights", type=str, default=None,
                        help="Wavelet loss band level weights, uses band weights if not defined for a given level. Input example: {'ll1': 0.1, 'lh1': 0.01, 'hl1': 0.01, 'hh1': 0.05, 'll2': 0.1, 'lh2': 0.01, 'hl2': 0.01, 'hh2': 0.05} E.x. Default: none.")
    parser.add_argument("--wavelet_loss_band_weights", type=str,
                        default=r"{ 'll': 0.1, 'lh': 0.01, 'hl': 0.01, 'hh': 0.05}", help="Wavelet loss band weights.")
    parser.add_argument("--wavelet_loss_ll_level_threshold", default=None,
                        help="Wavelet loss which level to calculate the loss for the low frequency (ll). -1 means last n level. Default: None")
    parser.add_argument(
        "--wavelet_loss_quaternion_component_weights",
        type=str,
        default=r"{ 'r' : 0.25, 'i' : 0.5, 'j' : 0.5, 'k' : 0.5 }",
        help="Quaternion Wavelet loss component weights.",
    )


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
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

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
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
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(tokenizer, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2): min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
        tokenizer,
        text_encoder,
        text_input: torch.Tensor,
        chunk_length: int,
        clip_skip: int,
        eos: int,
        pad: int,
        no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2): (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                        text_input_chunk[j, 1] = eos

            if clip_skip is None or clip_skip == 1:
                text_embedding = text_encoder(text_input_chunk)[0]
            else:
                enc_out = text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                text_embedding = enc_out["hidden_states"][-clip_skip]
                text_embedding = text_encoder.text_model.final_layer_norm(text_embedding)

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        if clip_skip is None or clip_skip == 1:
            text_embeddings = text_encoder(text_input)[0]
        else:
            enc_out = text_encoder(text_input, output_hidden_states=True, return_dict=True)
            text_embeddings = enc_out["hidden_states"][-clip_skip]
            text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)
    return text_embeddings


def get_weighted_text_embeddings(
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        device,
        max_embeddings_multiples: Optional[int] = 3,
        no_boseos_middle: Optional[bool] = False,
        clip_skip=None,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, prompt, max_length - 2)

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        tokenizer,
        text_encoder,
        prompt_tokens,
        tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)

    # assign weights to the prompts and normalize in the sense of mean
    previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * prompt_weights.unsqueeze(-1)
    current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    return text_embeddings


# https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
def pyramid_noise_like(noise, device, iterations=6, discount=0.4, scaling_factor=None):
    if scaling_factor is None:
        scaling_factor = torch.ones(noise.shape[0], device=device)
    scaling_factor_reshaped = scaling_factor.view(-1, 1, 1, 1)

    b, c, w, h = noise.shape  # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode="bilinear").to(device)
    for i in range(iterations):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        wn, hn = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
        noise += (u(torch.randn(b, c, wn, hn).to(device)) * (discount * scaling_factor_reshaped) ** i)
        if wn == 1 or hn == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


# https://www.crosslabs.org//blog/diffusion-with-offset-noise
def apply_noise_offset(latents, noise, noise_offset, adaptive_noise_scale):
    if noise_offset is None:
        return noise
    if adaptive_noise_scale is not None:
        # latent shape: (batch_size, channels, height, width)
        # abs mean value for each channel
        latent_mean = torch.abs(latents.mean(dim=(2, 3), keepdim=True))

        # multiply adaptive noise scale to the mean value and add it to the noise offset
        noise_offset = noise_offset + adaptive_noise_scale * latent_mean
        noise_offset = torch.clamp(noise_offset, 0.0, None)  # in case of adaptive noise scale is negative

    noise = noise + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
    return noise


def apply_masked_loss(loss, batch):
    if "conditioning_images" in batch:
        # conditioning image is -1 to 1. we need to convert it to 0 to 1
        mask_image = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)  # use R channel
        mask_image = mask_image / 2 + 0.5
        # print(f"conditioning_image: {mask_image.shape}")
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        # alpha mask is 0 to 1
        mask_image = batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)  # add channel dimension
        # print(f"mask_image: {mask_image.shape}, {mask_image.mean()}")
    else:
        return loss

    # resize to the same size as the loss
    mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
    loss = loss * mask_image
    return loss


class LossCallableMSE(Protocol):
    def __call__(
            self,
            input: Tensor,
            target: Tensor,
            size_average: Optional[bool] = None,
            reduce: Optional[bool] = None,
            reduction: str = "mean"
    ) -> Tensor: ...


class LossCallableReduction(Protocol):
    def __call__(
            self,
            input: Tensor,
            target: Tensor,
            reduction: str = "mean"
    ) -> Tensor: ...


LossCallable = LossCallableReduction | LossCallableMSE


class WaveletTransform:
    """Base class for wavelet transforms."""

    def __init__(self, wavelet='db4', device=torch.device("cpu"), dtype=torch.float32):
        """Initialize wavelet filters."""
        assert pywt.Wavelet is not None, "PyWavelets module not available. Please install `pip install PyWavelets`"

        # Create filters from wavelet
        wav = pywt.Wavelet(wavelet)
        self.dec_lo = torch.tensor(wav.dec_lo, device=device, dtype=dtype)
        self.dec_hi = torch.tensor(wav.dec_hi, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def decompose(self, x: Tensor) -> dict[str, list[Tensor]]:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("WaveletTransform subclasses must implement decompose method")


class DiscreteWaveletTransform(WaveletTransform):
    """Discrete Wavelet Transform (DWT) implementation."""

    def decompose(self, x: Tensor, level=1) -> dict[str, list[Tensor]]:
        """
        Perform multi-level DWT decomposition.
        
        Args:
            x: Input tensor [B, C, H, W]
            level: Number of decomposition levels
            
        Returns:
            Dictionary containing decomposition coefficients
        """
        bands: dict[str, list[Tensor]] = {
            'll': [],
            'lh': [],
            'hl': [],
            'hh': [],
        }

        # Start low frequency with input
        ll = x

        for _ in range(level):
            ll, lh, hl, hh = self._dwt_single_level(ll)

            bands['lh'].append(lh)
            bands['hl'].append(hl)
            bands['hh'].append(hh)
            bands['ll'].append(ll)

        return bands

    def _dwt_single_level(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform single-level DWT decomposition."""
        batch, channels, height, width = x.shape
        x_view = x.view(batch * channels, 1, height, width)  # Renamed to avoid conflict if x is on different device

        # Calculate proper padding for the filter size
        filter_size = self.dec_lo.size(0)
        pad_size = filter_size // 2

        # Pad for proper convolution
        try:
            x_pad = F.pad(x_view, (pad_size,) * 4, mode="reflect")
        except RuntimeError:
            # Fallback for very small tensors
            x_pad = F.pad(x_view, (pad_size,) * 4, mode="constant")

        # Apply filter to rows
        # Ensure filters are on the same device as x_pad
        dec_lo_filt_row = self.dec_lo.view(1, 1, -1, 1).to(device=x_pad.device, dtype=x_pad.dtype)
        dec_hi_filt_row = self.dec_hi.view(1, 1, -1, 1).to(device=x_pad.device, dtype=x_pad.dtype)

        lo = F.conv2d(x_pad, dec_lo_filt_row, stride=(2, 1))
        hi = F.conv2d(x_pad, dec_hi_filt_row, stride=(2, 1))

        # Apply filter to columns
        dec_lo_filt_col = self.dec_lo.view(1, 1, 1, -1).to(device=lo.device, dtype=lo.dtype)
        dec_hi_filt_col = self.dec_hi.view(1, 1, 1, -1).to(device=lo.device, dtype=lo.dtype)

        ll = F.conv2d(lo, dec_lo_filt_col, stride=(1, 2))
        lh = F.conv2d(lo, dec_hi_filt_col, stride=(1, 2))
        hl = F.conv2d(hi, dec_lo_filt_col, stride=(1, 2))
        hh = F.conv2d(hi, dec_hi_filt_col, stride=(1, 2))

        # Reshape back to batch format and ensure original device/dtype
        ll = ll.view(batch, channels, ll.shape[2], ll.shape[3]).to(device=x.device, dtype=x.dtype)
        lh = lh.view(batch, channels, lh.shape[2], lh.shape[3]).to(device=x.device, dtype=x.dtype)
        hl = hl.view(batch, channels, hl.shape[2], hl.shape[3]).to(device=x.device, dtype=x.dtype)
        hh = hh.view(batch, channels, hh.shape[2], hh.shape[3]).to(device=x.device, dtype=x.dtype)

        return ll, lh, hl, hh


class StationaryWaveletTransform(WaveletTransform):
    """Stationary Wavelet Transform (SWT) implementation."""

    def __init__(self, wavelet="db4", device=torch.device("cpu"), dtype=torch.float32):
        """Initialize wavelet filters."""
        super().__init__(wavelet, device, dtype)

        # Store original filters
        self.orig_dec_lo = self.dec_lo.clone()
        self.orig_dec_hi = self.dec_hi.clone()

    def decompose(self, x: Tensor, level=1) -> dict[str, list[Tensor]]:
        """Perform multi-level SWT decomposition."""
        bands = {
            "ll": [],
            "lh": [],
            "hl": [],
            "hh": [],
        }

        # Start with input as low frequency
        ll = x

        for j in range(level):
            # Get upsampled filters for current level
            dec_lo, dec_hi = self._get_filters_for_level(j)

            # Decompose current approximation
            ll_new, lh, hl, hh = self._swt_single_level(ll, dec_lo, dec_hi)  # ll is approximation

            # Store results in bands
            bands["ll"].append(ll_new)
            bands["lh"].append(lh)
            bands["hl"].append(hl)
            bands["hh"].append(hh)
            ll = ll_new  # Next level's input is current approximation
        return bands

    def _get_filters_for_level(self, level: int) -> tuple[Tensor, Tensor]:
        """Get upsampled filters for the specified level."""
        if level == 0:
            return self.orig_dec_lo, self.orig_dec_hi

        # Calculate number of zeros to insert
        zeros = 2 ** level - 1

        # Create upsampled filters
        upsampled_dec_lo = torch.zeros(len(self.orig_dec_lo) + (len(self.orig_dec_lo) - 1) * zeros,
                                       device=self.orig_dec_lo.device, dtype=self.orig_dec_lo.dtype)
        upsampled_dec_hi = torch.zeros(len(self.orig_dec_hi) + (len(self.orig_dec_hi) - 1) * zeros,
                                       device=self.orig_dec_hi.device, dtype=self.orig_dec_hi.dtype)

        # Insert original coefficients with zeros in between
        upsampled_dec_lo[:: zeros + 1] = self.orig_dec_lo
        upsampled_dec_hi[:: zeros + 1] = self.orig_dec_hi

        return upsampled_dec_lo, upsampled_dec_hi

    def _swt_single_level(self, x: Tensor, dec_lo: Tensor, dec_hi: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform single-level SWT decomposition with 1D convolutions."""
        batch, channels, height, width = x.shape
        # Ensure filters are on the same device as input x
        dec_lo = dec_lo.to(x.device)
        dec_hi = dec_hi.to(x.device)

        ll = torch.zeros_like(x)  # SWT keeps dimensions
        lh = torch.zeros_like(x)
        hl = torch.zeros_like(x)
        hh = torch.zeros_like(x)

        # Prepare 1D filter kernels
        dec_lo_1d = dec_lo.view(1, 1, -1)
        dec_hi_1d = dec_hi.view(1, 1, -1)
        pad_len = dec_lo.size(0) - 1

        for b in range(batch):
            for c in range(channels):
                # Extract single channel/batch and reshape for 1D convolution
                x_bc = x[b, c]  # Shape: [height, width]

                # Process rows with 1D convolution
                # Reshape to [width, 1, height] for treating each row as a batch
                x_rows = x_bc.transpose(0, 1).unsqueeze(1)  # Shape: [width, 1, height]

                # Pad for circular convolution
                x_rows_padded = F.pad(x_rows, (pad_len, 0), mode="circular")

                # Apply filters to rows
                x_lo_rows = F.conv1d(x_rows_padded, dec_lo_1d)  # [width, 1, height]
                x_hi_rows = F.conv1d(x_rows_padded, dec_hi_1d)  # [width, 1, height]

                # Reshape and transpose back
                x_lo_rows = x_lo_rows.squeeze(1).transpose(0, 1)  # [height, width]
                x_hi_rows = x_hi_rows.squeeze(1).transpose(0, 1)  # [height, width]

                # Process columns with 1D convolution
                # Reshape for column filtering (no transpose needed)
                x_lo_cols = x_lo_rows.unsqueeze(1)  # [height, 1, width]
                x_hi_cols = x_hi_rows.unsqueeze(1)  # [height, 1, width]

                # Pad for circular convolution
                x_lo_cols_padded = F.pad(x_lo_cols, (pad_len, 0), mode="circular")
                x_hi_cols_padded = F.pad(x_hi_cols, (pad_len, 0), mode="circular")

                # Apply filters to columns
                ll[b, c] = F.conv1d(x_lo_cols_padded, dec_lo_1d).squeeze(1)  # [height, width]
                lh[b, c] = F.conv1d(x_lo_cols_padded, dec_hi_1d).squeeze(1)  # [height, width]
                hl[b, c] = F.conv1d(x_hi_cols_padded, dec_lo_1d).squeeze(1)  # [height, width]
                hh[b, c] = F.conv1d(x_hi_cols_padded, dec_hi_1d).squeeze(1)  # [height, width]

        return ll, lh, hl, hh


class QuaternionWaveletTransform(WaveletTransform):
    """
    Quaternion Wavelet Transform implementation.
    Combines real DWT with three Hilbert transforms along x, y, and xy axes.
    """

    def __init__(self, wavelet="db4", device=torch.device("cpu"), dtype=torch.float32):
        """Initialize wavelet filters and Hilbert transforms."""
        super().__init__(wavelet, device, dtype)  # self.dec_lo, self.dec_hi, self.device, self.dtype set here

        # Register Hilbert transform filters. These will be moved to self.device and self.dtype
        # by _create_hilbert_filter using self.device and self.dtype from super().__init__
        self.hilbert_x = self._create_hilbert_filter("x")
        self.hilbert_y = self._create_hilbert_filter("y")
        self.hilbert_xy = self._create_hilbert_filter("xy")

    def _create_hilbert_filter(self, direction: str) -> Tensor:
        """Create a Hilbert transform filter for the specified direction."""
        if direction == "x":
            filt_vals = [
                [-0.0106, -0.0329, -0.0308, 0.0000, 0.0308, 0.0329, 0.0106],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ]
        elif direction == "y":
            filt_vals = [
                [-0.0106, 0.0000], [-0.0329, 0.0000], [-0.0308, 0.0000],
                [0.0000, 0.0000],
                [0.0308, 0.0000], [0.0329, 0.0000], [0.0106, 0.0000],
            ]
        elif direction == "xy":
            filt_vals = [
                [-0.0011, -0.0035, -0.0033, 0.0000, 0.0033, 0.0035, 0.0011],
                [-0.0035, -0.0108, -0.0102, 0.0000, 0.0102, 0.0108, 0.0035],
                [-0.0033, -0.0102, -0.0095, 0.0000, 0.0095, 0.0102, 0.0033],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0033, 0.0102, 0.0095, 0.0000, -0.0095, -0.0102, -0.0033],
                [0.0035, 0.0108, 0.0102, 0.0000, -0.0102, -0.0108, -0.0035],
                [0.0011, 0.0035, 0.0033, 0.0000, -0.0033, -0.0035, -0.0011],
            ]
        else:
            raise ValueError(f"Unknown Hilbert direction: {direction}")

        filt = torch.tensor(filt_vals, device=self.device, dtype=self.dtype)
        return filt.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H_filt, W_filt]

    def _apply_hilbert(self, x: Tensor, direction: str) -> Tensor:
        """Apply Hilbert transform in specified direction with correct padding."""
        batch, channels, height, width = x.shape
        # Reshape for group convolution if C > 1, or process per channel.
        # Original code flattens batch and channels. Let's stick to that for consistency.
        x_flat = x.reshape(batch * channels, 1, height, width)

        if direction == "x":
            h_filter = self.hilbert_x
        elif direction == "y":
            h_filter = self.hilbert_y
        else:  # 'xy'
            h_filter = self.hilbert_xy

        # Ensure filter is on the same device as input
        h_filter = h_filter.to(device=x_flat.device, dtype=x_flat.dtype)

        filter_h, filter_w = h_filter.shape[2:]
        pad_h = (filter_h - 1) // 2
        pad_w = (filter_w - 1) // 2

        pad_h_left, pad_h_right = pad_h, pad_h
        pad_w_left, pad_w_right = pad_w, pad_w

        if filter_h % 2 == 0: pad_h_right += 1
        if filter_w % 2 == 0: pad_w_right += 1

        x_pad = F.pad(x_flat, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")
        x_hilbert = F.conv2d(x_pad, h_filter)

        # Cropping to ensure output matches input H, W
        # This handles cases where padding + convolution might result in slightly larger output
        # than input, especially with asymmetric padding for even kernels.
        out_h, out_w = x_hilbert.shape[2:]
        crop_h_top, crop_w_left = 0, 0

        if out_h > height:
            crop_h_top = (out_h - height) // 2
        if out_w > width:
            crop_w_left = (out_w - width) // 2

        x_hilbert_cropped = x_hilbert[:, :, crop_h_top: crop_h_top + height, crop_w_left: crop_w_left + width]

        return x_hilbert_cropped.reshape(batch, channels, height, width)

    def _dwt_single_level(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform single-level DWT decomposition. (Copied from original for QWT use)"""
        batch, channels, height, width = x.shape
        # Reshape input to be (B*C, 1, H, W) for 2D conv with single input channel filters
        x_view = x.view(batch * channels, 1, height, width)

        filter_size = self.dec_lo.size(0)
        pad_size = filter_size // 2

        try:
            x_pad = F.pad(x_view, (pad_size,) * 4, mode="reflect")
        except RuntimeError:
            x_pad = F.pad(x_view, (pad_size,) * 4, mode="constant")

        # Ensure filters are on the same device and dtype as x_pad
        # These filters (dec_lo, dec_hi) are initialized in WaveletTransform.__init__
        # to self.device and self.dtype. If x is on a different device, filters need to move.
        current_device = x_pad.device
        current_dtype = x_pad.dtype

        dec_lo_r = self.dec_lo.view(1, 1, -1, 1).to(device=current_device, dtype=current_dtype)
        dec_hi_r = self.dec_hi.view(1, 1, -1, 1).to(device=current_device, dtype=current_dtype)
        dec_lo_c = self.dec_lo.view(1, 1, 1, -1).to(device=current_device, dtype=current_dtype)
        dec_hi_c = self.dec_hi.view(1, 1, 1, -1).to(device=current_device, dtype=current_dtype)

        lo = F.conv2d(x_pad, dec_lo_r, stride=(2, 1))
        hi = F.conv2d(x_pad, dec_hi_r, stride=(2, 1))

        ll = F.conv2d(lo, dec_lo_c, stride=(1, 2))
        lh = F.conv2d(lo, dec_hi_c, stride=(1, 2))
        hl = F.conv2d(hi, dec_lo_c, stride=(1, 2))
        hh = F.conv2d(hi, dec_hi_c, stride=(1, 2))

        # Reshape back to (B, C, H', W') and ensure original x's device/dtype
        ll = ll.view(batch, channels, ll.shape[2], ll.shape[3]).to(device=x.device, dtype=x.dtype)
        lh = lh.view(batch, channels, lh.shape[2], lh.shape[3]).to(device=x.device, dtype=x.dtype)
        hl = hl.view(batch, channels, hl.shape[2], hl.shape[3]).to(device=x.device, dtype=x.dtype)
        hh = hh.view(batch, channels, hh.shape[2], hh.shape[3]).to(device=x.device, dtype=x.dtype)
        return ll, lh, hl, hh

    def _decompose_single_component(self, component_signal: Tensor, level: int) -> dict[str, list[Tensor]]:
        """
        Helper function to perform multi-level DWT on a single component signal.
        Args:
            component_signal: Tensor to decompose [B, C, H, W]
            level: Number of decomposition levels
        Returns:
            Dictionary of bands for this component.
        """
        bands: dict[str, list[Tensor]] = {'ll': [], 'lh': [], 'hl': [], 'hh': []}
        current_ll = component_signal

        for _ in range(level):
            ll_next, lh, hl, hh = self._dwt_single_level(current_ll)
            bands['ll'].append(ll_next)
            bands['lh'].append(lh)
            bands['hl'].append(hl)
            bands['hh'].append(hh)
            current_ll = ll_next  # Update for the next decomposition level

        return bands

    def decompose(self, x: Tensor, level: int = 1) -> dict[str, dict[str, list[Tensor]]]:
        """
        Perform multi-level QWT decomposition sequentially for memory efficiency.
        Args:
            x: Input tensor [B, C, H, W]
            level: Number of decomposition levels
        Returns:
            Dictionary containing quaternion wavelet coefficients
            Format: {component: {band: [level1_coeff, level2_coeff, ...]}}
            where component ∈ {r, i, j, k} and band ∈ {ll, lh, hl, hh}
        """
        qwt_coeffs: dict[str, dict[str, list[Tensor]]] = {
            "r": {}, "i": {}, "j": {}, "k": {}
        }

        # Real part (original signal)
        # No need to store x separately as ll_r, just pass to helper
        qwt_coeffs["r"] = self._decompose_single_component(x, level)

        # i-component (x-Hilbert transform of x)
        x_hilbert_x = self._apply_hilbert(x, "x")
        qwt_coeffs["i"] = self._decompose_single_component(x_hilbert_x, level)
        del x_hilbert_x  # Attempt to free memory sooner

        # j-component (y-Hilbert transform of x)
        x_hilbert_y = self._apply_hilbert(x, "y")
        qwt_coeffs["j"] = self._decompose_single_component(x_hilbert_y, level)
        del x_hilbert_y

        # k-component (xy-Hilbert transform of x)
        x_hilbert_xy = self._apply_hilbert(x, "xy")
        qwt_coeffs["k"] = self._decompose_single_component(x_hilbert_xy, level)
        del x_hilbert_xy

        return qwt_coeffs


class WaveletLoss(nn.Module):
    """Wavelet-based loss calculation module."""

    def __init__(
            self,
            wavelet="db4",
            level=3,
            transform_type="dwt",
            loss_fn: LossCallable = F.mse_loss,
            device=torch.device("cpu"),
            band_level_weights: Optional[dict[str, float]] = None,
            band_weights: Optional[dict[str, float]] = None,
            quaternion_component_weights: dict[str, float] | None = None,
            ll_level_threshold: Optional[int] = -1,
            dtype=torch.float32,
    ):
        """
        Args:
            wavelet: Wavelet family (e.g., 'db4', 'sym7')
            level: Decomposition level
            transform_type: Type of wavelet transform ('dwt' or 'swt')
            loss_fn: Loss function to apply to wavelet coefficients
            device: Computation device
            band_level_weights: Optional custom weights for different bands on different levels
            band_weights: Optional custom weights for different bands
            component_weights: Weights for quaternion components
            ll_level_threshold: Level when applying loss for ll. Default -1 or last level.
        """
        super().__init__()
        self.level = level
        self.wavelet = wavelet
        self.transform_type = transform_type
        self.loss_fn = loss_fn
        self.device = device
        self.ll_level_threshold = ll_level_threshold if ll_level_threshold is not None else None  # -1 means last level
        self.dtype = dtype

        # Initialize transform based on type
        if transform_type == "dwt":
            self.transform = DiscreteWaveletTransform(wavelet, device=device, dtype=dtype)
        elif transform_type == "swt":  # swt
            self.transform = StationaryWaveletTransform(wavelet, device=device, dtype=dtype)
        elif transform_type == "qwt":
            self.transform = QuaternionWaveletTransform(wavelet, device=device, dtype=dtype)

            # Register Hilbert filters as buffers            
            # These hilbert filters are already part of self.transform object.
            # Registering them here as buffers is redundant but harmless.
            # If QWT internal hilbert filters change, these won't auto-update unless re-registered.
            self.register_buffer("hilbert_x", self.transform.hilbert_x.clone().detach())
            self.register_buffer("hilbert_y", self.transform.hilbert_y.clone().detach())
            self.register_buffer("hilbert_xy", self.transform.hilbert_xy.clone().detach())

            # Default weights
            self.component_weights = quaternion_component_weights or {
                "r": 1.0,  # Real part (standard wavelet)
                "i": 0.7,  # x-Hilbert (imaginary part)
                "j": 0.7,  # y-Hilbert (imaginary part)
                "k": 0.5,  # xy-Hilbert (imaginary part)
            }
        else:
            raise RuntimeError(f"Invalid transform type {transform_type}")

        # Register wavelet filters (dec_lo, dec_hi) from the transform object
        # These are already on the correct device/dtype from transform's init.
        self.register_buffer("dec_lo", self.transform.dec_lo.clone().detach())
        self.register_buffer("dec_hi", self.transform.dec_hi.clone().detach())

        # Default weights from paper:
        # "Training Generative Image Super-Resolution Models by Wavelet-Domain Losses"
        self.band_level_weights = band_level_weights or {}

        self.band_weights = band_weights or {"ll": 0.1, "lh": 0.01, "hl": 0.01, "hh": 0.05}

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:  # Return type was tuple, now just Tensor loss
        """Calculate wavelet loss between prediction and target."""
        # Ensure inputs are on the module's device and dtype
        pred = pred.to(device=self.device, dtype=self.dtype)
        target = target.to(device=self.device, dtype=self.dtype)

        if isinstance(self.transform, QuaternionWaveletTransform):
            return self.quaternion_forward(pred, target)  # quaternion_forward now returns Tensor

        pred_coeffs = self.transform.decompose(pred, self.level)
        target_coeffs = self.transform.decompose(target, self.level)

        total_loss = torch.tensor(0.0, device=pred.device, dtype=self.dtype)

        # Loop from level 1 to self.level (inclusive)
        for i in range(self.level):  # pred_coeffs lists are 0-indexed by level
            level_num = i + 1  # For weight keys (1-indexed)

            # LL band consideration based on ll_level_threshold
            if self.ll_level_threshold is not None:
                # ll_level_threshold: if positive, it's the max level index (1-based) up to which LL is included.
                #                     if negative, it's count from the last level (e.g., -1 is only the last LL).
                #                     if 0, no LL bands are included.
                actual_ll_threshold_level = self.ll_level_threshold
                if self.ll_level_threshold < 0:
                    actual_ll_threshold_level = self.level + self.ll_level_threshold + 1  # Convert to 1-based index

                if level_num >= actual_ll_threshold_level and actual_ll_threshold_level > 0:  # Include LL if current level is at or past threshold
                    band = "ll"
                    weight_key = f"{band}{level_num}"
                    # Get coefficients for the current level i (0-indexed)
                    pred_c = pred_coeffs[band][i]
                    target_c = target_coeffs[band][i]

                    # Padding is not needed here as DWT levels will have different sizes
                    # The loss_fn should handle tensors of same shape. DWT ensures corresponding bands have same shape.
                    weight = self.band_level_weights.get(weight_key, self.band_weights[band])
                    band_loss = weight * self.loss_fn(pred_c, target_c)
                    total_loss += band_loss.mean()  # Ensure scalar

            # High frequency bands (LH, HL, HH)
            for band in ["lh", "hl", "hh"]:
                weight_key = f"{band}{level_num}"
                if band in pred_coeffs and i < len(pred_coeffs[band]):  # Check if band and level exist
                    pred_c = pred_coeffs[band][i]
                    target_c = target_coeffs[band][i]

                    weight = self.band_level_weights.get(weight_key, self.band_weights[band])
                    band_loss = weight * self.loss_fn(pred_c, target_c)
                    total_loss += band_loss.mean()  # Ensure scalar
        return total_loss

    def quaternion_forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Calculate QWT loss between prediction and target.
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
        Returns:
            Tuple of (total loss, detailed component losses)
        """
        assert isinstance(self.transform, QuaternionWaveletTransform), "Not a quaternion wavelet transform"
        # Apply QWT to both inputs
        pred_qwt = self.transform.decompose(pred, self.level)
        target_qwt = self.transform.decompose(target, self.level)

        # Initialize total loss and component losses
        total_loss = torch.tensor(0.0, device=pred.device, dtype=self.dtype)

        # Calculate loss for each quaternion component, band and level
        for component in ["r", "i", "j", "k"]:
            component_weight = self.component_weights[component]

            for band in ["ll", "lh", "hl", "hh"]:
                band_weight = self.band_weights[band]

                for level_idx in range(self.level):
                    band_level_key = f"{band}{level_idx + 1}"
                    # band_level_weights take priority over band_weight if exists
                    if band_level_key in self.band_level_weights:
                        level_weight = self.band_level_weights[band_level_key]
                    else:
                        level_weight = band_weight

                    # Get coefficients at this level
                    pred_coeff = pred_qwt[component][band][level_idx]
                    target_coeff = target_qwt[component][band][level_idx]

                    # Calculate loss
                    level_loss = self.loss_fn(pred_coeff, target_coeff)
                    level_loss = level_loss.mean()

                    # Apply weights
                    weighted_loss = component_weight * level_weight * level_loss

                    # Add to total loss
                    total_loss += weighted_loss

        return total_loss

    def set_loss_fn(self, loss_fn: LossCallable):
        """
        Set loss function to use. Wavelet loss wants l1 or huber loss.
        """
        self.loss_fn = loss_fn


"""
##########################################
# Perlin Noise
def rand_perlin_2d(device, shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)),
            dim=-1,
        )
        % 1
    )
    angles = 2 * torch.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )
    dot = lambda grad, shift: (
        torch.stack((grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), dim=-1)
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return 1.414 * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(device, shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape, device=device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(device, shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def perlin_noise(noise, device, octaves):
    _, c, w, h = noise.shape
    perlin = lambda: rand_perlin_2d_octaves(device, (w, h), (4, 4), octaves)
    noise_perlin = []
    for _ in range(c):
        noise_perlin.append(perlin())
    noise_perlin = torch.stack(noise_perlin).unsqueeze(0)   # (1, c, w, h)
    noise += noise_perlin # broadcast for each batch
    return noise / noise.std()  # Scaled back to roughly unit variance
"""
