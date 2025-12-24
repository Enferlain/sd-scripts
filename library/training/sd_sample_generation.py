from library.training.sample_generation import sample_images_common
from library.pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

def sample_images(*args, **kwargs):
    return sample_images_common(StableDiffusionLongPromptWeightingPipeline, *args, **kwargs)
