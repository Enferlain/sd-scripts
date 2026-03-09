import os
import gc

import contextlib
import json
import toml
import re
import logging
import time
import torch

from PIL import Image
from accelerate import Accelerator
from accelerate.state import PartialState

from diffusers import (
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)

from library.constants import SCHEDULER_TIMESTEPS, SCHEDULER_LINEAR_START, SCHEDULER_LINEAR_END, SCHEDULER_SCHEDULE
from library.utils.device_utils import clean_memory_on_device
from library.pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from library.config.dataclasses.output import SamplingConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.output import SavingConfig
from library.config.dataclasses.loss import LossConfig

logger = logging.getLogger(__name__)


def get_my_scheduler(
    *,
    sample_sampler: str,
    v_parameterization: bool,
):
    """
    Returns a scheduler object based on the provided sampler name and parameterization settings.

    Args:
        sample_sampler (str): The name of the sampler to use (e.g., "ddim", "pndm", "euler_a").
        v_parameterization (bool): Whether to use v-parameterization.

    Returns:
        SchedulerMixin: The initialized scheduler object.
    """
    sched_init_args = {}
    if sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif sample_sampler == "ddpm":  # ddpm is excluded because it causes issues
        scheduler_cls = DDPMScheduler
    elif sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif sample_sampler == "lms" or sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif sample_sampler == "euler" or sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif sample_sampler == "euler_a" or sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sample_sampler == "dpmsolver" or sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sample_sampler
    elif sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif sample_sampler == "dpm_2" or sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif sample_sampler == "dpm_2_a" or sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    if v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDULER_SCHEDULE,
        **sched_init_args,
    )

    # set clip_sample to True
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # logger.info("set clip_sample to True")
        scheduler.config.clip_sample = True

    return scheduler


# NOTE: SD-specific sample_images() moved to sd_sample_generation.py


def line_to_prompt_dict(line: str) -> dict:
    """
    Parses a string line into a prompt dictionary.

    Args:
        line (str): The input string containing the prompt and arguments.
                    Arguments are expected to be in the format "--arg value".

    Returns:
        dict: A dictionary containing the parsed prompt and arguments.
    """
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"l ([\d.]+)", parg, re.IGNORECASE)  # scale (CFG)
            if m:
                prompt_dict["scale"] = float(m.group(1))
                continue

            m = re.match(r"g ([\d.]+)", parg, re.IGNORECASE)  # guidance scale
            if m:  # guidance scale
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"ss (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_sampler"] = m.group(1)
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["controlnet_image"] = m.group(1)
                continue

            m = re.match(r"ctr (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["cfg_trunc_ratio"] = float(m.group(1))
                continue

            m = re.match(r"rcfg (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["renorm_cfg"] = float(m.group(1))
                continue

            m = re.match(r"fs (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["flow_shift"] = m.group(1)
                continue

        except ValueError as ex:
            logger.error(f"Exception in parsing: {parg}")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> list[dict]:
    """
    Loads prompts from a file.

    Supported file formats are .txt, .toml, and .json.

    Args:
        prompt_file (str): The path to the prompt file.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a prompt
                    and its associated settings.
    """
    # read prompts
    if prompt_file.endswith(".txt"):
        with open(prompt_file, encoding="utf-8") as f:
            lines = f.readlines()
        raw_prompts: list[str | dict] = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, encoding="utf-8") as f:
            data = toml.load(f)
        raw_prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, encoding="utf-8") as f:
            raw_prompts = json.load(f)
    else:
        raise ValueError(f"Unsupported prompt file format: {prompt_file}. Supported formats: .txt, .toml, .json")

    # Build result list with proper types
    prompts: list[dict] = []
    for i, p in enumerate(raw_prompts):
        prompt_dict = line_to_prompt_dict(p) if isinstance(p, str) else p
        assert isinstance(prompt_dict, dict)
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)
        prompts.append(prompt_dict)

    return prompts


def sample_images_check(sampling_config: SamplingConfig, epoch: int | None, steps: int) -> bool:
    """
    Checks if sample images should be generated at the current step or epoch.

    Args:
        sampling_config (SamplingConfig): The sampling configuration.
        epoch (int, optional): The current epoch number.
        steps (int): The current step number.

    Returns:
        bool: True if images should be generated, False otherwise.
    """
    if steps == 0:
        return sampling_config.sample_at_first

    # Check if neither sampling mode is configured
    if sampling_config.sample_every_n_steps is None and sampling_config.sample_every_n_epochs is None:
        return False

    # Epoch-based sampling takes precedence when configured
    if sampling_config.sample_every_n_epochs is not None:
        # At end of epoch (epoch is not None), check epoch divisibility
        return epoch is not None and epoch % sampling_config.sample_every_n_epochs == 0

    # Step-based sampling (only when epoch-based is not configured)
    # Skip at epoch boundaries (epoch is not None) to avoid double-sampling
    if epoch is not None:
        return False
    return steps % sampling_config.sample_every_n_steps == 0


def sample_images_common(
    pipe_class,
    accelerator: Accelerator,
    sampling_config: SamplingConfig,
    training_config: TrainingConfig,
    saving_config: SavingConfig,
    loss_config: LossConfig,
    epoch: int | None,
    steps: int,
    device,
    vae,
    tokenizer,
    text_encoder,
    unet_wrapped,
    prompt_replacement: tuple[str, str] | None = None,
    controlnet=None,
    tokenize_strategy=None,
    text_encoding_strategy=None,
):
    """
    Common function for generating sample images during training.

    This function handles the setup of the pipeline, loading of prompts, and distribution
    of work across available devices.

    Args:
        pipe_class: The pipeline class to use (e.g., StableDiffusionLongPromptWeightingPipeline).
        accelerator (Accelerator): The accelerator instance for distributed training.
        sampling_config (SamplingConfig): Configuration for sampling.
        training_config (TrainingConfig): Configuration for training.
        saving_config (SavingConfig): Configuration for saving outputs.
        loss_config (LossConfig): Configuration related to loss (used for v_parameterization).
        epoch (int, optional): The current epoch.
        steps (int): The current step.
        device: The device to run inference on.
        vae: The VAE model.
        tokenizer: The tokenizer.
        text_encoder: The text encoder model(s).
        unet_wrapped: The UNet model (wrapped).
        prompt_replacement (tuple, optional): A tuple (target, replacement) to modify prompts.
        controlnet: ControlNet model (optional).
        tokenize_strategy: Runtime tokenization strategy used by sampling pipelines that
            need model-family-specific prompt handling.
        text_encoding_strategy: Runtime text-encoding strategy used by sampling pipelines
            that need model-family-specific prompt handling.
    """

    if not sample_images_check(sampling_config, epoch, steps):
        return

    logger.info("")
    logger.info(f"generating sample images at step: {steps}")
    if not os.path.isfile(sampling_config.sample_prompts):
        logger.error(f"No prompt file: {sampling_config.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    # Save original devices for all models (they may be on CPU for memory efficiency)
    org_vae_device = vae.device

    # unwrap unet and text_encoder(s), saving their original devices
    unet = accelerator.unwrap_model(unet_wrapped)
    org_unet_device = unet.device

    if isinstance(text_encoder, (list, tuple)):
        text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
        org_te_devices = [te.device for te in text_encoder]
    else:
        text_encoder = accelerator.unwrap_model(text_encoder)
        org_te_devices = [text_encoder.device]

    # Save original VAE dtype for restoration after sampling
    org_vae_dtype = vae.dtype

    # Apply sample_vae_dtype if specified (allows fp16 VAE for sampling even if training uses fp32)
    if sampling_config.sample_vae_dtype is not None:
        sample_dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        sample_vae_dtype = sample_dtype_map.get(sampling_config.sample_vae_dtype)
        if sample_vae_dtype is not None and sample_vae_dtype != org_vae_dtype:
            logger.info(f"Casting VAE from {org_vae_dtype} to {sample_vae_dtype} for sampling")
            vae.to(dtype=sample_vae_dtype)

    # Move VAE to device (text encoders and unet will be moved by pipeline.to())
    vae.to(distributed_state.device)

    # read prompts
    if sampling_config.sample_prompts.endswith(".txt"):
        with open(sampling_config.sample_prompts, encoding="utf-8") as f:
            lines = f.readlines()
        raw_prompts: list[str | dict] = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif sampling_config.sample_prompts.endswith(".toml"):
        with open(sampling_config.sample_prompts, encoding="utf-8") as f:
            data = toml.load(f)
        raw_prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif sampling_config.sample_prompts.endswith(".json"):
        with open(sampling_config.sample_prompts, encoding="utf-8") as f:
            raw_prompts = json.load(f)
    else:
        logger.error(f"Unsupported prompt file format: {sampling_config.sample_prompts}. Supported formats: .txt, .toml, .json")
        return

    default_scheduler = get_my_scheduler(sample_sampler=sampling_config.sample_sampler, v_parameterization=loss_config.v_parameterization)

    pipe_kwargs = {
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "tokenizer": tokenizer,
        "scheduler": default_scheduler,
        "safety_checker": None,
        "feature_extractor": None,
        "requires_safety_checker": False,
        "clip_skip": training_config.clip_skip,
    }
    if pipe_class is SdxlStableDiffusionLongPromptWeightingPipeline:
        pipe_kwargs["tokenize_strategy"] = tokenize_strategy
        pipe_kwargs["text_encoding_strategy"] = text_encoding_strategy

    pipeline = pipe_class(**pipe_kwargs)
    pipeline.to(distributed_state.device)
    save_dir = saving_config.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # Build result list with proper types
    prompts: list[dict] = []
    for i, p in enumerate(raw_prompts):
        prompt_dict = line_to_prompt_dict(p) if isinstance(p, str) else p
        assert isinstance(prompt_dict, dict)
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)
        prompts.append(prompt_dict)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    with contextlib.suppress(Exception):
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        with torch.no_grad():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator,
                    sampling_config,
                    training_config,
                    saving_config,
                    loss_config,
                    pipeline,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    prompt_replacement,
                    controlnet=controlnet,
                )
                # Per-prompt cleanup: free GPU memory between prompts to prevent accumulation
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

    else:
        # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
        # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
        per_process_prompts = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad(), distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
            for prompt_dict in prompt_dict_lists[0]:
                sample_image_inference(
                    accelerator,
                    sampling_config,
                    training_config,
                    saving_config,
                    loss_config,
                    pipeline,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    prompt_replacement,
                    controlnet=controlnet,
                )

    # clear pipeline and cache to reduce vram usage
    del pipeline
    del default_scheduler

    torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    # Restore all models to their original devices and dtypes (critical for training)
    # This matches the pattern used in caching code (SdxlTrainingStrategy.cache_text_encoder_outputs_if_needed)
    vae.to(device=org_vae_device, dtype=org_vae_dtype)
    unet.to(org_unet_device)

    # Restore text encoders - handle both single and list cases
    if isinstance(text_encoder, (list, tuple)):
        for te, org_device in zip(text_encoder, org_te_devices):
            te.to(org_device)
    else:
        text_encoder.to(org_te_devices[0])

    # Aggressive cleanup to prevent VRAM accumulation between sampling runs
    gc.collect()
    clean_memory_on_device(accelerator.device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def sample_image_inference(
    accelerator: Accelerator,
    sampling_config: SamplingConfig,
    training_config: TrainingConfig,
    saving_config: SavingConfig,
    loss_config: LossConfig,
    pipeline: StableDiffusionLongPromptWeightingPipeline | SdxlStableDiffusionLongPromptWeightingPipeline,
    save_dir: str,
    prompt_dict: dict,
    epoch: int | None,
    steps: int,
    prompt_replacement: tuple[str, str] | None,
    controlnet=None,
):
    """
    Performs the actual image inference for a single prompt.

    Args:
        accelerator (Accelerator): The accelerator instance.
        sampling_config (SamplingConfig): Sampling configuration.
        training_config (TrainingConfig): Training configuration.
        saving_config (SavingConfig): Saving configuration.
        loss_config (LossConfig): Loss configuration.
        pipeline: The inference pipeline.
        save_dir (str): Directory to save the generated images.
        prompt_dict (dict): Dictionary containing the prompt and parameters.
        epoch (int, optional): Current epoch.
        steps (int): Current step.
        prompt_replacement (tuple, optional): Tuple for prompt replacement.
        controlnet: ControlNet model (optional).
    """
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 7.5)
    seed = prompt_dict.get("seed")
    controlnet_image = prompt_dict.get("controlnet_image")
    prompt: str = prompt_dict.get("prompt", "")
    sampler_name: str = prompt_dict.get("sample_sampler", sampling_config.sample_sampler)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed()

    scheduler = get_my_scheduler(
        sample_sampler=sampler_name,
        v_parameterization=loss_config.v_parameterization,
    )
    pipeline.scheduler = scheduler

    if controlnet_image is not None:
        controlnet_image = Image.open(controlnet_image).convert("RGB")
        controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)  # PIL.Image.Resampling.LANCZOS

    height = max(64, height - height % 8)  # round to divisible by 8
    width = max(64, width - width % 8)  # round to divisible by 8
    logger.info(f"prompt: {prompt}")
    logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    logger.info(f"sample_sampler: {sampler_name}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    with accelerator.autocast(), torch.no_grad():
        latents = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=sample_steps,
            guidance_scale=scale,
            negative_prompt=negative_prompt,
            controlnet=controlnet,
            controlnet_image=controlnet_image,
        )

    # VAE decode - convert latents to image
    with torch.no_grad():
        image = pipeline.latents_to_image(latents)[0]

    # Free latents immediately after decode
    del latents

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = (
        f"{'' if saving_config.output_name is None else saving_config.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    )
    image.save(os.path.join(save_dir, img_filename))

    # send images to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")

        import wandb

        # not to commit images to avoid inconsistency between training and logging steps
        wandb_tracker.log(
            {f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False
        )  # positive prompt as caption, commit=False avoids step mismatch TODO: Parameter 'step' unfilled

    # Cleanup per-inference tensors to prevent accumulation
    del image, scheduler
