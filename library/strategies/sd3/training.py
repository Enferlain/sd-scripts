from typing import Any

from library.strategies.base.contracts import TrainingStrategy
from library.strategies.sd3.caching import Sd3CachingStrategy
from library.strategies.sd3.checkpointing import Sd3CheckpointingStrategy
from library.strategies.sd3.denoiser import Sd3DenoiserCallingStrategy
from library.strategies.sd3.diffusion import Sd3DiffusionTrainingStrategy
from library.strategies.sd3.encoding import Sd3TextEncodingStrategy
from library.strategies.sd3.loading import Sd3ModelLoadingStrategy
from library.strategies.sd3.model_preparation import Sd3ModelPreparationStrategy
from library.strategies.sd3.sampling import Sd3SampleGenerationStrategy
from library.strategies.sd3.tokenization import DEFAULT_SD3_T5_MAX_LENGTH, Sd3TokenizeStrategy
from library.strategies.sd3.validation import Sd3ValidationStrategy


class Sd3TrainingStrategy(
    Sd3ModelLoadingStrategy,
    Sd3TokenizeStrategy,
    Sd3TextEncodingStrategy,
    Sd3CachingStrategy,
    Sd3SampleGenerationStrategy,
    Sd3CheckpointingStrategy,
    Sd3ValidationStrategy,
    Sd3DiffusionTrainingStrategy,
    Sd3DenoiserCallingStrategy,
    Sd3ModelPreparationStrategy,
    TrainingStrategy,
):
    """SD3 implementation of the training strategy."""

    vae_latent_scale: float = 1.0
    max_token_length: int = 0

    def __init__(self, cfg: Any):
        """Construct a fully initialized SD3 training strategy from config."""
        t5_max_length = getattr(cfg.model, "t5xxl_max_token_length", None)
        if t5_max_length is None:
            t5_max_length = cfg.training.max_token_length
        if t5_max_length is None:
            t5_max_length = DEFAULT_SD3_T5_MAX_LENGTH

        Sd3TokenizeStrategy.__init__(
            self,
            t5xxl_max_length=t5_max_length,
            tokenizer_cache_dir=cfg.data.caching.tokenizer_cache_dir,
        )
        Sd3TextEncodingStrategy.__init__(
            self,
            tokenizers=self.tokenizers,
            apply_lg_attn_mask=getattr(cfg.model, "apply_lg_attn_mask", False),
            apply_t5_attn_mask=getattr(cfg.model, "apply_t5_attn_mask", False),
            l_dropout_rate=float(getattr(cfg.model, "clip_l_dropout_rate", 0.0)),
            g_dropout_rate=float(getattr(cfg.model, "clip_g_dropout_rate", 0.0)),
            t5_dropout_rate=float(getattr(cfg.model, "t5_dropout_rate", 0.0)),
        )

        self.max_token_length = t5_max_length
        self.apply_lg_attn_mask = bool(getattr(cfg.model, "apply_lg_attn_mask", False))
        self.apply_t5_attn_mask = bool(getattr(cfg.model, "apply_t5_attn_mask", False))
        self.live_plotter_process = None
        self._model_version = "medium"


__all__ = ["Sd3TrainingStrategy"]
