from library.training.sample_generation import sample_images_common
from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline

def sample_images(*args, **kwargs):
    return sample_images_common(SdxlStableDiffusionLongPromptWeightingPipeline, *args, **kwargs)
