from typing import Any

from library.constants import SDXL_VAE_LATENT_SCALE
from library.strategies.sdxl.caching import SdxlCachingStrategy
from library.strategies.sdxl.checkpointing import SdxlCheckpointingStrategy
from library.strategies.sdxl.denoiser import SdxlDenoiserCallingStrategy
from library.strategies.sdxl.diffusion import SdxlDiffusionTrainingStrategy
from library.strategies.sdxl.encoding import SdxlTextEncodingStrategy
from library.strategies.sdxl.loading import SdxlModelLoadingStrategy
from library.strategies.sdxl.model_preparation import SdxlModelPreparationStrategy
from library.strategies.sdxl.sampling import SdxlSampleGenerationStrategy
from library.strategies.sdxl.tokenization import SdxlTokenizeStrategy
from library.strategies.sdxl.validation import SdxlValidationStrategy

from library.strategies.base.contracts import TrainingStrategy


class SdxlTrainingStrategy(
    SdxlModelLoadingStrategy,
    SdxlTokenizeStrategy,
    SdxlTextEncodingStrategy,
    SdxlCachingStrategy,
    SdxlSampleGenerationStrategy,
    SdxlCheckpointingStrategy,
    SdxlValidationStrategy,
    SdxlDiffusionTrainingStrategy,
    SdxlDenoiserCallingStrategy,
    SdxlModelPreparationStrategy,
    TrainingStrategy,
):
    """
    SDXL implementation of the training strategy.
    """

    vae_latent_scale: float = SDXL_VAE_LATENT_SCALE

    # Instance state set during model loading
    load_stable_diffusion_format: bool = False
    logit_scale: Any = None
    ckpt_info: Any = None
    max_token_length: int = 0

    def __init__(self, cfg: Any):
        """Construct a fully initialized SDXL training strategy from config."""
        SdxlTokenizeStrategy.__init__(
            self,
            cfg.training.max_token_length,
            cfg.data.caching.tokenizer_cache_dir,
        )
        self.max_token_length = self.max_length
        self.load_stable_diffusion_format = False
        self.logit_scale = None
        self.ckpt_info = None
        self.la_sampler = None
        self.live_plotter_process = None
