from library.training.sample_generation import sample_images_common
from library.pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline


def sample_images(*args, **kwargs):
    """
    Generates sample images using the Stable Diffusion pipeline.

    This function wraps sample_images_common with the StableDiffusionLongPromptWeightingPipeline.

    Args:
        *args: Variable length argument list passed to sample_images_common.
        **kwargs: Arbitrary keyword arguments passed to sample_images_common.
    """
    return sample_images_common(StableDiffusionLongPromptWeightingPipeline, *args, **kwargs)
