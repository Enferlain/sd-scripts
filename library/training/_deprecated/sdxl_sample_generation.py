from types import SimpleNamespace

from library.training.sample_generation import (
    get_my_scheduler,
    sample_images_check,
    sample_images_common,
)
from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from library.utils.device_utils import clean_memory_on_device
import gc
import torch


def sample_images(*args, **kwargs):
    """
    Generates sample images using the SDXL pipeline.

    This function wraps sample_images_common with the SdxlStableDiffusionLongPromptWeightingPipeline.

    Args:
        *args: Variable length argument list passed to sample_images_common.
        **kwargs: Arbitrary keyword arguments passed to sample_images_common.
    """
    strategy = kwargs.pop("strategy", None)
    (
        accelerator,
        sampling_config,
        training_config,
        saving_config,
        loss_config,
        epoch,
        steps,
        _device,
        vae,
        tokenizer,
        text_encoder,
        denoiser_wrapped,
        *rest,
    ) = args
    objective_config = kwargs.pop("objective_config", None)
    prediction_type = "v_prediction" if loss_config.v_parameterization else "epsilon"
    if objective_config is None:
        objective_config = SimpleNamespace(target=prediction_type)

    if not sample_images_check(sampling_config, epoch, steps):
        return

    org_vae_device = vae.device
    org_vae_dtype = vae.dtype
    denoiser = accelerator.unwrap_model(denoiser_wrapped)
    org_denoiser_device = denoiser.device
    text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
    org_te_devices = [te.device for te in text_encoder]

    if sampling_config.sample_vae_dtype is not None:
        sample_dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        sample_vae_dtype = sample_dtype_map.get(sampling_config.sample_vae_dtype)
        if sample_vae_dtype is not None and sample_vae_dtype != org_vae_dtype:
            vae.to(dtype=sample_vae_dtype)

    vae.to(accelerator.device)
    try:
        pipeline = SdxlStableDiffusionLongPromptWeightingPipeline(
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            unet=denoiser,
            scheduler=get_my_scheduler(
                sample_sampler=sampling_config.sample_sampler,
                prediction_type=prediction_type,
            ),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            clip_skip=training_config.clip_skip,
            strategy=strategy,
        )
        return sample_images_common(
            accelerator,
            sampling_config,
            training_config,
            saving_config,
            objective_config,
            loss_config,
            epoch,
            steps,
            pipeline,
            *rest,
            **kwargs,
        )
    finally:
        vae.to(device=org_vae_device, dtype=org_vae_dtype)
        denoiser.to(org_denoiser_device)
        for te, org_device in zip(text_encoder, org_te_devices):
            te.to(org_device)
        gc.collect()
        clean_memory_on_device(accelerator.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
