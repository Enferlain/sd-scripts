from typing import Any

from library.constants import SD_VAE_LATENT_SCALE
from library.strategies.base.contracts import TrainingStrategy
from library.strategies.sd.caching import SdCachingStrategy
from library.strategies.sd.checkpointing import SdCheckpointingStrategy
from library.strategies.sd.denoiser import SdDenoiserCallingStrategy
from library.strategies.sd.diffusion import SdDiffusionTrainingStrategy
from library.strategies.sd.encoding import SdTextEncodingStrategy
from library.strategies.sd.loading import SdModelLoadingStrategy
from library.strategies.sd.model_preparation import SdModelPreparationStrategy
from library.strategies.sd.sampling import SdSampleGenerationStrategy
from library.strategies.sd.tokenization import SdTokenizeStrategy
from library.strategies.sd.validation import SdValidationStrategy


class SdTrainingStrategy(
    SdModelLoadingStrategy,
    SdTokenizeStrategy,
    SdTextEncodingStrategy,
    SdCachingStrategy,
    SdSampleGenerationStrategy,
    SdCheckpointingStrategy,
    SdValidationStrategy,
    SdDiffusionTrainingStrategy,
    SdDenoiserCallingStrategy,
    SdModelPreparationStrategy,
    TrainingStrategy,
):
    """SD 1.5/2.0 implementation of the training strategy."""

    vae_latent_scale: float = SD_VAE_LATENT_SCALE
    max_token_length: int = 0
    clip_skip: int | None = None

    def __init__(self, cfg: Any):
        """Construct a fully initialized SD training strategy from config."""
        SdTokenizeStrategy.__init__(
            self,
            cfg.model.model_type == "sd2",
            cfg.training.max_token_length,
            cfg.data.caching.tokenizer_cache_dir,
        )
        self.max_token_length = self.max_length
        self.clip_skip = cfg.training.clip_skip
        self.la_sampler = None
        self.live_plotter_process = None
