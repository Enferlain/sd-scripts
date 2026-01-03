from library.training.sample_generation import sample_images_common
from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline


def sample_images(*args, **kwargs):
    """
    Generates sample images using the SDXL pipeline.

    This function wraps sample_images_common with the SdxlStableDiffusionLongPromptWeightingPipeline.

    Args:
        *args: Variable length argument list passed to sample_images_common.
        **kwargs: Arbitrary keyword arguments passed to sample_images_common.
    """
    return sample_images_common(SdxlStableDiffusionLongPromptWeightingPipeline, *args, **kwargs)
