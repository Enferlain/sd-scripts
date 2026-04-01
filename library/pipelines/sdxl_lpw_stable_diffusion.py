# copy from https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py

import inspect
import numpy as np
import PIL.Image
import torch

from typing import Any
from collections.abc import Callable
from tqdm import tqdm
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import SchedulerMixin, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import logging, PIL_INTERPOLATION

from library.constants import SDXL_VAE_LATENT_SCALE
from library.data.prompt_utils import parse_prompt_attention
from library.models.sdxl.conversion import get_size_embeddings
from library.strategies.sdxl.encoding import pool_workaround
from library.models.sdxl import unet as sdxl_original_unet
from library.models.sdxl import control_net as sdxl_original_control_net

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_prompts_with_weights(
    pipe: StableDiffusionPipeline, prompt: list[str], max_length: int
) -> tuple[list[list[int]], list[list[float]]]:
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.

    Args:
        pipe (StableDiffusionPipeline): The pipeline to use for tokenization.
        prompt (List[str]): A list of prompts to tokenize.
        max_length (int): The maximum length of the tokens.

    Returns:
        Tuple[List[List[int]], List[List[float]]]: A tuple containing the tokens and their weights.
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
            token = pipe.tokenizer(word).input_ids[1:-1]
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


def pad_tokens_and_weights(
    tokens: list[list[int]],
    weights: list[list[float]],
    max_length: int,
    bos: int,
    eos: int,
    pad: int,
    no_boseos_middle: bool | None = True,
    chunk_length: int = 77,
) -> tuple[list[list[int]], list[list[float]]]:
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.

    Args:
        tokens (List[List[int]]): The tokens to pad.
        weights (List[List[float]]): The weights to pad.
        max_length (int): The maximum length of the tokens.
        bos (int): The beginning of sequence token id.
        eos (int): The end of sequence token id.
        pad (int): The padding token id.
        no_boseos_middle (bool, optional):
            Whether to prevent adding BOS/EOS tokens in the middle of chunks. Defaults to True.
        chunk_length (int, optional): The length of each chunk. Defaults to 77.

    Returns:
        Tuple[List[List[int]], List[List[float]]]: A tuple containing the padded tokens and weights.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_hidden_states(
    text_encoder: CLIPTextModel, input_ids: torch.Tensor, is_sdxl_text_encoder2: bool, eos_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Get the hidden states and pool from the text encoder.

    Args:
        text_encoder (CLIPTextModel): The text encoder.
        input_ids (torch.Tensor): The input ids.
        is_sdxl_text_encoder2 (bool): Whether the text encoder is the second text encoder of SDXL.
        eos_token_id (int): The end of sequence token id.
        device (torch.device): The device to use.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the hidden states and the pool.
    """
    if not is_sdxl_text_encoder2:
        # text_encoder1: same as SD1/2
        enc_out = text_encoder(input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=True)
        hidden_states = enc_out["hidden_states"][11]
        pool = None
    else:
        # text_encoder2
        enc_out = text_encoder(input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=True)
        hidden_states = enc_out["hidden_states"][-2]  # penuultimate layer
        # pool = enc_out["text_embeds"]
        pool = pool_workaround(text_encoder, enc_out["last_hidden_state"], input_ids, eos_token_id)
    hidden_states = hidden_states.to(device)
    if pool is not None:
        pool = pool.to(device)
    return hidden_states, pool


def get_unweighted_text_embeddings(
    pipe: StableDiffusionPipeline,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    is_sdxl_text_encoder2: bool,
    no_boseos_middle: bool | None = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.

    Args:
        pipe (StableDiffusionPipeline): The pipeline object.
        text_input (torch.Tensor): The input text tensor.
        chunk_length (int): The length of each chunk.
        clip_skip (int): The number of layers to skip in the CLIP model.
        eos (int): The end of sequence token id.
        pad (int): The padding token id.
        is_sdxl_text_encoder2 (bool): Whether the text encoder is the second text encoder of SDXL.
        no_boseos_middle (bool, optional):
            Whether to prevent adding BOS/EOS tokens in the middle of chunks. Defaults to True.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the text embeddings and the pool.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    text_pool = None
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # ordinary characters at the end
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOS only, rest is PAD
                        text_input_chunk[j, 1] = eos

            text_embedding, current_text_pool = get_hidden_states(
                pipe.text_encoder, text_input_chunk, is_sdxl_text_encoder2, eos, pipe.device
            )
            if text_pool is None:
                text_pool = current_text_pool

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
        text_embeddings, text_pool = get_hidden_states(pipe.text_encoder, text_input, is_sdxl_text_encoder2, eos, pipe.device)
    return text_embeddings, text_pool


def get_weighted_text_embeddings(
    pipe: "SdxlStableDiffusionLongPromptWeightingPipeline",
    prompt: str | list[str],
    uncond_prompt: str | list[str] | None = None,
    max_embeddings_multiples: int | None = 3,
    no_boseos_middle: bool | None = False,
    skip_parsing: bool | None = False,
    skip_weighting: bool | None = False,
    clip_skip: int | None = None,
    is_sdxl_text_encoder2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        pipe (`StableDiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
        clip_skip (`int`, *optional*):
            The number of layers to skip in the CLIP model.
        is_sdxl_text_encoder2 (`bool`, *optional*, defaults to `False`):
            Whether the text encoder is the second text encoder of SDXL.
    """
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [token[1:-1] for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = pipe.tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

    # get the embeddings
    text_embeddings, text_pool = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        is_sdxl_text_encoder2,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)

    if uncond_prompt is not None:
        uncond_embeddings, uncond_pool = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            clip_skip,
            eos,
            pad,
            is_sdxl_text_encoder2,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, text_pool, uncond_embeddings, uncond_pool
    return text_embeddings, text_pool, None, None


def preprocess_image(image: PIL.Image.Image) -> torch.Tensor:
    """
    Preprocess the image.

    Args:
        image (PIL.Image.Image): The image to preprocess.

    Returns:
        torch.Tensor: The preprocessed image.
    """
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask: PIL.Image.Image, scale_factor: int = 8) -> torch.Tensor:
    """
    Preprocess the mask.

    Args:
        mask (PIL.Image.Image): The mask to preprocess.
        scale_factor (int, optional): The scale factor. Defaults to 8.

    Returns:
        torch.Tensor: The preprocessed mask.
    """
    mask = mask.convert("L")
    w, h = mask.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=PIL_INTERPOLATION["nearest"])
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask


def prepare_controlnet_image(
    image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor,
    width: int,
    height: int,
    batch_size: int,
    num_images_per_prompt: int,
    device: torch.device,
    dtype: torch.dtype,
    do_classifier_free_guidance: bool = False,
    guess_mode: bool = False,
) -> torch.Tensor:
    """
    Prepare the controlnet image.

    Args:
        image (PIL.Image.Image): The image to prepare.
        width (int): The width of the image.
        height (int): The height of the image.
        batch_size (int): The batch size.
        num_images_per_prompt (int): The number of images per prompt.
        device (torch.device): The device to use.
        dtype (torch.dtype): The data type.
        do_classifier_free_guidance (bool, optional):
            Whether to use classifier free guidance. Defaults to False.
        guess_mode (bool, optional): Whether to use guess mode. Defaults to False.

    Returns:
        torch.Tensor: The prepared controlnet image.
    """
    if not isinstance(image, torch.Tensor):
        if isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            images = []

            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)

            image = images

            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)

    return image


class SdxlStableDiffusionLongPromptWeightingPipeline:
    r"""
    Pipeline for text-to-image generation using Stable Diffusion without tokens length limit, and support parsing
    weighting in prompt.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    # if version.parse(version.parse(diffusers.__version__).base_version) >= version.parse("0.9.0"):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: list[CLIPTextModel],
        tokenizer: list[CLIPTokenizer],
        unet: sdxl_original_unet.SdxlUNet2DConditionModel | sdxl_original_control_net.SdxlControlledUNet,
        scheduler: SchedulerMixin,
        # clip_skip: int,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
        clip_skip: int = 1,
        strategy: Any | None = None,
    ):
        # clip skip is ignored currently
        self.tokenizer = tokenizer[0]
        self.text_encoder = text_encoder[0]
        self.unet = unet
        self.scheduler = scheduler
        self.safety_checker = safety_checker
        self.feature_extractor = feature_extractor
        self.requires_safety_checker = requires_safety_checker
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.progress_bar = lambda x: tqdm(x, leave=False)

        self.clip_skip = clip_skip
        self.tokenizers = tokenizer
        self.text_encoders = text_encoder
        self.strategy = strategy

    #     self.__init__additional__()

    # def __init__additional__(self):
    #     if not hasattr(self, "vae_scale_factor"):
    #         setattr(self, "vae_scale_factor", 2 ** (len(self.vae.config.block_out_channels) - 1))

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
            # self.vae.to(device=self.device)
        if dtype is not None:
            self.dtype = dtype

        # do not move Text Encoders to device, because Text Encoder should be on CPU

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def check_inputs(self, prompt: str | list[str], height: int, width: int, strength: float, callback_steps: int):
        """
        Check the validity of the inputs.

        Args:
            prompt (str or List[str]): The prompt or prompts.
            height (int): The height of the image.
            width (int): The width of the image.
            strength (float): The strength of the transformation.
            callback_steps (int): The number of steps between callbacks.

        Raises:
            ValueError: If any of the inputs are invalid.
        """
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.")

    def get_timesteps(self, num_inference_steps: int, strength: float, device: torch.device, is_text2img: bool) -> tuple[torch.Tensor, int]:
        """
        Get the timesteps for the diffusion process.

        Args:
            num_inference_steps (int): The number of inference steps.
            strength (float): The strength of the transformation.
            device (torch.device): The device to use.
            is_text2img (bool): Whether it is text-to-image generation.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the timesteps and the number of inference steps.
        """
        if is_text2img:
            return self.scheduler.timesteps.to(device), num_inference_steps
        else:
            # get the original timesteps using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(device)
            return timesteps, num_inference_steps - t_start

    def run_safety_checker(self, image: torch.Tensor, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, bool | None]:
        """
        Run the safety checker on the image.

        Args:
            image (torch.Tensor): The image to check.
            device (torch.device): The device to use.
            dtype (torch.dtype): The data type.

        Returns:
            Tuple[torch.Tensor, Optional[bool]]: A tuple containing the image and a boolean indicating if it contains NSFW concepts.
        """
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values.to(dtype))
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        """
        Decode the latents to images.

        Args:
            latents (torch.Tensor): The latents to decode.

        Returns:
            np.ndarray: The decoded images.
        """
        with torch.no_grad():
            latents = 1 / SDXL_VAE_LATENT_SCALE * latents

            image = self.vae.decode(latents.to(self.vae.dtype)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            return image

    def prepare_extra_step_kwargs(self, generator: torch.Generator, eta: float) -> dict[str, Any]:
        """
        Prepare extra arguments for the scheduler step.

        Args:
            generator (torch.Generator): The generator to use.
            eta (float): The eta value for DDIM scheduler.

        Returns:
            Dict[str, Any]: A dictionary containing the extra arguments.
        """
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        image: torch.Tensor | None,
        timestep: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Prepare the latents for the diffusion process.

        Args:
            image (torch.Tensor, optional): The initial image for image-to-image generation.
            timestep (torch.Tensor): The timesteps for adding noise.
            batch_size (int): The batch size.
            height (int): The height of the image.
            width (int): The width of the image.
            dtype (torch.dtype): The data type.
            device (torch.device): The device to use.
            generator (torch.Generator, optional): The generator to use.
            latents (torch.Tensor, optional): The initial latents.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                A tuple containing the latents, the original initial latents (if image provided), and the noise.
        """
        if image is None:
            shape = (
                batch_size,
                self.unet.in_channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )

            if latents is None:
                if device.type == "mps":
                    # randn does not work reproducibly on mps
                    latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
                else:
                    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            else:
                if latents.shape != shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
                latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents, None, None
        else:
            init_latent_dist = self.vae.encode(image).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = SDXL_VAE_LATENT_SCALE * init_latents
            init_latents = torch.cat([init_latents] * batch_size, dim=0)
            init_latents_orig = init_latents
            shape = init_latents.shape

            # add noise to latents using the timesteps
            if device.type == "mps":
                noise = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
            else:
                noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.add_noise(init_latents, noise, timestep)
            return latents, init_latents_orig, noise

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        image: torch.FloatTensor | PIL.Image.Image | None = None,
        mask_image: torch.FloatTensor | PIL.Image.Image | None = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        num_images_per_prompt: int | None = 1,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        latents: torch.FloatTensor | None = None,
        max_embeddings_multiples: int | None = 3,
        output_type: str | None = "pil",
        return_dict: bool = True,
        controlnet: sdxl_original_control_net.SdxlControlNet | None = None,
        controlnet_image=None,
        callback: Callable[[int, int, torch.FloatTensor], None] | None = None,
        is_cancelled_callback: Callable[[], bool] | None = None,
        callback_steps: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            controlnet (`diffusers.ControlNetModel`, *optional*):
                A controlnet model to be used for the inference. If not provided, controlnet will be disabled.
            controlnet_image (`torch.FloatTensor` or `PIL.Image.Image`, *optional*):
                `Image`, or tensor representing an image batch, to be used as the starting point for the controlnet
                inference.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timesteps: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            `None` if cancelled by `is_cancelled_callback`,
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if controlnet is not None and controlnet_image is None:
            raise ValueError("controlnet_image must be provided if controlnet is not None.")

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        if self.strategy is None:
            raise RuntimeError("SDXL sampling pipeline requires an explicit training strategy")

        text_input_ids, text_weights = self.strategy.tokenize_with_weights(prompt)
        hidden_states_1, hidden_states_2, text_pool = self.strategy.encode_tokens_with_weights(
            self.text_encoders, text_input_ids, text_weights
        )
        text_embeddings = torch.cat([hidden_states_1, hidden_states_2], dim=-1)

        if do_classifier_free_guidance:
            input_ids, weights = self.strategy.tokenize_with_weights(negative_prompt or "")
            hidden_states_1, hidden_states_2, uncond_pool = self.strategy.encode_tokens_with_weights(self.text_encoders, input_ids, weights)
            uncond_embeddings = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
        else:
            uncond_embeddings = None
            uncond_pool = None

        unet_dtype = self.unet.dtype
        dtype = unet_dtype
        if hasattr(dtype, "itemsize") and dtype.itemsize == 1:  # fp8
            dtype = torch.float16
            self.unet.to(dtype)

        # 4. Preprocess image and mask
        if isinstance(image, PIL.Image.Image):
            image = preprocess_image(image)
        if image is not None:
            image = image.to(device=self.device, dtype=dtype)
        if isinstance(mask_image, PIL.Image.Image):
            mask_image = preprocess_mask(mask_image, self.vae_scale_factor)
        if mask_image is not None:
            mask = mask_image.to(device=self.device, dtype=dtype)
            mask = torch.cat([mask] * batch_size * num_images_per_prompt)
        else:
            mask = None

        # ControlNet is not working yet in SDXL, but keep the code here for future use
        if controlnet_image is not None:
            controlnet_image = prepare_controlnet_image(
                controlnet_image, width, height, batch_size, 1, self.device, controlnet.dtype, do_classifier_free_guidance, False
            )

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device, image is None)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents, init_latents_orig, noise = self.prepare_latents(
            image,
            latent_timestep,
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # create size embs and concat embeddings for SDXL
        orig_size = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1).to(device, dtype)
        crop_size = torch.zeros_like(orig_size)
        target_size = orig_size
        embs = get_size_embeddings(orig_size, crop_size, target_size, device).to(device, dtype)

        # make conditionings
        text_pool = text_pool.to(device, dtype)
        if do_classifier_free_guidance:
            text_embedding = torch.cat([uncond_embeddings, text_embeddings]).to(device, dtype)

            uncond_pool = uncond_pool.to(device, dtype)
            cond_vector = torch.cat([text_pool, embs], dim=1).to(dtype)
            uncond_vector = torch.cat([uncond_pool, embs], dim=1).to(dtype)
            vector_embedding = torch.cat([uncond_vector, cond_vector])
        else:
            text_embedding = text_embeddings.to(device, dtype)
            vector_embedding = torch.cat([text_pool, embs], dim=1)

        # 8. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # FIXME SD1 ControlNet is not working

            # predict the noise residual
            if controlnet is not None:
                input_resi_add, mid_add = controlnet(latent_model_input, t, text_embedding, vector_embedding, controlnet_image)
                noise_pred = self.unet(latent_model_input, t, text_embedding, vector_embedding, input_resi_add, mid_add)
            else:
                noise_pred = self.unet(latent_model_input, t, text_embedding, vector_embedding)
            noise_pred = noise_pred.to(dtype)  # U-Net changes dtype in LoRA training

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if mask is not None:
                # masking
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

            # call the callback, if provided
            if i % callback_steps == 0:
                if callback is not None:
                    callback(i, t, latents)
                if is_cancelled_callback is not None and is_cancelled_callback():
                    return None

        self.unet.to(unet_dtype)
        return latents

    def latents_to_image(self, latents: torch.Tensor) -> list[PIL.Image.Image]:
        """
        Convert latents to a PIL image.

        Args:
            latents (torch.Tensor): The latents to convert.

        Returns:
            List[PIL.Image.Image]: The converted images.
        """
        # 9. Post-processing
        image = self.decode_latents(latents.to(self.vae.dtype))
        image = self.numpy_to_pil(image)
        return image

    # copy from pil_utils.py
    def numpy_to_pil(self, images: np.ndarray) -> Image.Image:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def text2img(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int | None = 1,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        latents: torch.FloatTensor | None = None,
        max_embeddings_multiples: int | None = 3,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback: Callable[[int, int, torch.FloatTensor], None] | None = None,
        is_cancelled_callback: Callable[[], bool] | None = None,
        callback_steps: int = 1,
    ):
        r"""
        Function for text-to-image generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timesteps: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            is_cancelled_callback=is_cancelled_callback,
            callback_steps=callback_steps,
        )

    def img2img(
        self,
        image: torch.FloatTensor | PIL.Image.Image,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        strength: float = 0.8,
        num_inference_steps: int | None = 50,
        guidance_scale: float | None = 7.5,
        num_images_per_prompt: int | None = 1,
        eta: float | None = 0.0,
        generator: torch.Generator | None = None,
        max_embeddings_multiples: int | None = 3,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback: Callable[[int, int, torch.FloatTensor], None] | None = None,
        is_cancelled_callback: Callable[[], bool] | None = None,
        callback_steps: int = 1,
    ):
        r"""
        Function for image-to-image generation.
        Args:
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timesteps: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            is_cancelled_callback=is_cancelled_callback,
            callback_steps=callback_steps,
        )

    def inpaint(
        self,
        image: torch.FloatTensor | PIL.Image.Image,
        mask_image: torch.FloatTensor | PIL.Image.Image,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        strength: float = 0.8,
        num_inference_steps: int | None = 50,
        guidance_scale: float | None = 7.5,
        num_images_per_prompt: int | None = 1,
        eta: float | None = 0.0,
        generator: torch.Generator | None = None,
        max_embeddings_multiples: int | None = 3,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback: Callable[[int, int, torch.FloatTensor], None] | None = None,
        is_cancelled_callback: Callable[[], bool] | None = None,
        callback_steps: int = 1,
    ):
        r"""
        Function for inpaint.
        Args:
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. This is the image whose masked region will be inpainted.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to inpaint the masked area. Must be between 0 and 1. When `strength`
                is 1, the denoising process will be run on the masked area for the full number of iterations specified
                in `num_inference_steps`. `image` will be used as a reference for the masked area, adding more
                noise to that region the larger the `strength`. If `strength` is 0, no inpainting will occur.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The reference number of denoising steps. More denoising steps usually lead to a higher quality image at
                the expense of slower inference. This parameter will be modulated by `strength`, as explained above.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timesteps: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            is_cancelled_callback=is_cancelled_callback,
            callback_steps=callback_steps,
        )
