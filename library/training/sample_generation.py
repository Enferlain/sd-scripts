import os
import gc

import contextlib
import json
import toml
import re
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
import torch

from PIL import Image
from accelerate import Accelerator
from accelerate.state import PartialState
from tqdm import tqdm

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
from library.config.dataclasses.objective import ObjectiveConfig
from library.config.dataclasses.output import SamplingConfig
from library.config.dataclasses.loss import LossConfig
from library.config.dataclasses.output import SavingConfig
from library.config.dataclasses.training import TrainingConfig
from library.objectives.ddpm import DDPM_PREDICTION_TYPE_V

logger = logging.getLogger(__name__)


@dataclass
class SamplingRequest:
    """Normalized sampling request shared across repo-owned and backend-owned samplers."""

    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    sample_steps: int
    guidance_scale: float
    seed: int | None
    sample_sampler: str
    prompt_index: int
    flow_shift: float | None = None
    controlnet_image: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class SamplingBackend(Protocol):
    """Backend contract for turning a normalized sampling request into a final image."""

    def generate_image(self, accelerator: Accelerator, request: SamplingRequest) -> Image.Image:
        """Generate one image for the provided request."""


@dataclass
class LocalPipelineSamplingBackend:
    """Adapter for the existing repo-local latent-returning sampling pipelines."""

    _is_sampling_backend = True
    pipeline: Any
    prediction_type: str
    controlnet: Any = None

    def to(self, device: torch.device) -> None:
        """Move the wrapped pipeline to the requested device."""
        self.pipeline.to(device)

    def generate_image(self, accelerator: Accelerator, request: SamplingRequest) -> Image.Image:
        """Run the local pipeline path and decode its latent output to a final image."""
        scheduler = get_my_scheduler(
            sample_sampler=request.sample_sampler,
            prediction_type=self.prediction_type,
        )
        self.pipeline.scheduler = scheduler

        controlnet_image = None
        if request.controlnet_image is not None:
            controlnet_image = Image.open(request.controlnet_image).convert("RGB")
            controlnet_image = controlnet_image.resize((request.width, request.height), Image.LANCZOS)

        with accelerator.autocast(), torch.no_grad():
            result = self.pipeline(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.sample_steps,
                guidance_scale=request.guidance_scale,
                negative_prompt=request.negative_prompt,
                controlnet=self.controlnet,
                controlnet_image=controlnet_image,
            )

        if hasattr(result, "images"):
            image = result.images[0]
        else:
            image = self.pipeline.latents_to_image(result)[0]
            del result

        del scheduler
        return image


def get_my_scheduler(
    *,
    sample_sampler: str,
    prediction_type: str,
):
    """
    Returns a scheduler object based on the provided sampler name and parameterization settings.

    Args:
        sample_sampler (str): The name of the sampler to use (e.g., "ddim", "pndm", "euler_a").
        prediction_type (str): Active DDPM prediction type for the scheduler.

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

    if prediction_type == DDPM_PREDICTION_TYPE_V:
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


def get_sampling_prompt_dicts(sampling_config: SamplingConfig) -> list[dict] | None:
    """
    Resolve the prompt source for sampling.

    Preference order:
    1. Prompt file from ``sample_prompt_file``
    2. Inline single prompt from ``sample_prompt``
    """
    if sampling_config.sample_prompt_file:
        if not os.path.isfile(sampling_config.sample_prompt_file):
            logger.error(f"No prompt file: {sampling_config.sample_prompt_file}")
            return None
        return load_prompts(sampling_config.sample_prompt_file)

    if sampling_config.sample_prompt:
        return [{"prompt": sampling_config.sample_prompt, "enum": 0}]

    logger.error("No sampling prompt source configured. Set output.sampling.sample_prompt or sample_prompt_file.")
    return None


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


def build_sampling_request(
    sampling_config: SamplingConfig,
    prompt_dict: dict[str, Any],
    prompt_replacement: tuple[str, str] | None = None,
) -> SamplingRequest:
    """Build one normalized sampling request from prompt-level overrides and config defaults."""
    assert isinstance(prompt_dict, dict)
    prompt: str = prompt_dict.get("prompt", sampling_config.sample_prompt or "")
    negative_prompt = prompt_dict.get("negative_prompt", sampling_config.sample_negative_prompt)
    sample_steps = prompt_dict.get("sample_steps", sampling_config.sample_steps if sampling_config.sample_steps is not None else 30)
    width = prompt_dict.get("width", sampling_config.sample_width if sampling_config.sample_width is not None else 512)
    height = prompt_dict.get("height", sampling_config.sample_height if sampling_config.sample_height is not None else 512)
    guidance_scale = prompt_dict.get(
        "scale",
        prompt_dict.get("guidance_scale", sampling_config.sample_cfg_scale if sampling_config.sample_cfg_scale is not None else 7.5),
    )
    seed = prompt_dict.get("seed", sampling_config.sample_seed)
    controlnet_image = prompt_dict.get("controlnet_image")
    sample_sampler = prompt_dict.get("sample_sampler", sampling_config.sample_sampler)
    flow_shift = (
        float(prompt_dict["flow_shift"])
        if prompt_dict.get("flow_shift") is not None
        else (float(sampling_config.sample_flow_shift) if sampling_config.sample_flow_shift is not None else None)
    )

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    width = max(64, width - width % 8)
    height = max(64, height - height % 8)

    return SamplingRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        sample_steps=sample_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        sample_sampler=sample_sampler,
        prompt_index=int(prompt_dict["enum"]),
        flow_shift=flow_shift,
        controlnet_image=controlnet_image,
    )


def set_sampling_seed(seed: int | None) -> None:
    """Set per-request RNG state for deterministic or random sample generation."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        return

    torch.seed()
    if torch.cuda.is_available():
        torch.cuda.seed()


def _coerce_sampling_backend(
    backend_or_pipeline: Any,
    objective_config: ObjectiveConfig,
    *,
    controlnet: Any = None,
) -> SamplingBackend:
    """Wrap legacy local pipelines in a backend adapter while allowing custom backends directly."""
    if getattr(backend_or_pipeline, "_is_sampling_backend", False) is True:
        return backend_or_pipeline
    return LocalPipelineSamplingBackend(
        pipeline=backend_or_pipeline,
        prediction_type=objective_config.prediction,
        controlnet=controlnet,
    )


def sample_images_common(
    accelerator: Accelerator,
    sampling_config: SamplingConfig,
    training_config: TrainingConfig,
    saving_config: SavingConfig,
    objective_config: ObjectiveConfig,
    loss_config: LossConfig,
    epoch: int | None,
    steps: int,
    backend_or_pipeline,
    prompt_replacement: tuple[str, str] | None = None,
    controlnet=None,
):
    """
    Common function for generating sample images during training.

    This function handles the setup of the pipeline, loading of prompts, and distribution
    of work across available devices.

    Args:
        accelerator (Accelerator): The accelerator instance for distributed training.
        sampling_config (SamplingConfig): Configuration for sampling.
        training_config (TrainingConfig): Configuration for training.
        saving_config (SavingConfig): Configuration for saving outputs.
        objective_config (ObjectiveConfig): Configuration for objective/runtime behavior.
        loss_config (LossConfig): Configuration related to loss.
        epoch (int, optional): The current epoch.
        steps (int): The current step.
        backend_or_pipeline: Sampling backend or legacy local pipeline object.
        prompt_replacement (tuple, optional): A tuple (target, replacement) to modify prompts.
        controlnet: ControlNet model (optional).
    """
    logger.info("")
    logger.info(f"[sample] generating sample images at step: {steps}")
    prompts = get_sampling_prompt_dicts(sampling_config)
    if prompts is None:
        return

    backend = _coerce_sampling_backend(
        backend_or_pipeline,
        objective_config,
        controlnet=controlnet,
    )
    distributed_state = PartialState()
    if hasattr(backend, "to"):
        backend.to(distributed_state.device)
    save_dir = saving_config.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    with contextlib.suppress(Exception):
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        with torch.no_grad():
            prompt_iter = tqdm(prompts, desc="Sampling prompts", disable=not accelerator.is_local_main_process)
            for prompt_dict in prompt_iter:
                sample_image_inference(
                    accelerator,
                    sampling_config,
                    training_config,
                    saving_config,
                    objective_config,
                    loss_config,
                    backend,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    prompt_replacement,
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
            prompt_iter = tqdm(
                prompt_dict_lists[0],
                desc=f"Sampling prompts (rank {distributed_state.process_index})",
                disable=not accelerator.is_local_main_process,
            )
            for prompt_dict in prompt_iter:
                sample_image_inference(
                    accelerator,
                    sampling_config,
                    training_config,
                    saving_config,
                    objective_config,
                    loss_config,
                    backend,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    prompt_replacement,
                )

    # clear pipeline and cache to reduce vram usage
    del backend

    torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)


def sample_image_inference(
    accelerator: Accelerator,
    sampling_config: SamplingConfig,
    training_config: TrainingConfig,
    saving_config: SavingConfig,
    objective_config: ObjectiveConfig,
    loss_config: LossConfig,
    backend_or_pipeline,
    save_dir: str,
    prompt_dict: dict,
    epoch: int | None,
    steps: int,
    prompt_replacement: tuple[str, str] | None,
):
    """
    Performs the actual image inference for a single prompt.

    Args:
        accelerator (Accelerator): The accelerator instance.
        sampling_config (SamplingConfig): Sampling configuration.
        training_config (TrainingConfig): Training configuration.
        saving_config (SavingConfig): Saving configuration.
        loss_config (LossConfig): Loss configuration.
        backend_or_pipeline: Sampling backend or legacy local pipeline object.
        save_dir (str): Directory to save the generated images.
        prompt_dict (dict): Dictionary containing the prompt and parameters.
        epoch (int, optional): Current epoch.
        steps (int): Current step.
        prompt_replacement (tuple, optional): Tuple for prompt replacement.
    """
    del training_config, loss_config
    request = build_sampling_request(sampling_config, prompt_dict, prompt_replacement)
    backend = _coerce_sampling_backend(backend_or_pipeline, objective_config)
    set_sampling_seed(request.seed)

    logger.info(f"prompt: {request.prompt}")
    logger.info(f"negative_prompt: {request.negative_prompt}")
    logger.info(f"height: {request.height}")
    logger.info(f"width: {request.width}")
    logger.info(f"sample_steps: {request.sample_steps}")
    logger.info(f"scale: {request.guidance_scale}")
    logger.info(f"sample_sampler: {request.sample_sampler}")
    if request.flow_shift is not None:
        logger.info(f"flow_shift: {request.flow_shift}")
    if request.seed is not None:
        logger.info(f"seed: {request.seed}")

    image = backend.generate_image(accelerator, request)

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if request.seed is None else f"_{request.seed}"
    i = request.prompt_index
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
            {f"sample_{i}": wandb.Image(image, caption=request.prompt)}, commit=False
        )  # positive prompt as caption, commit=False avoids step mismatch TODO: Parameter 'step' unfilled

    # Cleanup per-inference tensors to prevent accumulation
    del image
