"""SDXL-specific conditioning payloads used by strategy facets."""

from dataclasses import dataclass
from typing import Any

import torch

from library.strategies.base.contracts import ConditioningStrategy, ModelConditioning


@dataclass
class SdxlConditioning(ModelConditioning):
    """
    SDXL micro-conditioning metadata carried alongside cached latents.

    SDXL uses original image size, crop coordinates, and target size as
    auxiliary conditioning inputs during denoiser calls.
    """

    original_size_hw: tuple[int, int]
    """Original image size (height, width) before any processing."""

    crop_top_left: tuple[int, int]
    """Crop offset (top, left) in bucket pixel space."""

    target_size_hw: tuple[int, int]
    """Target/bucket resolution (height, width) the image was resized to."""


class SdxlConditioningStrategy(ConditioningStrategy):
    """Conditioning facet for SDXL training strategies."""

    def _get_cached_text_conds(
        self,
        batch: Any,
        accelerator: Any,
        weight_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Load cached SDXL text-encoder outputs already attached to the batch."""
        te_outputs = batch.get("text_encoder_outputs")
        if te_outputs is None:
            return []
        return [
            te_outputs["hidden_state1"].to(accelerator.device, dtype=weight_dtype),
            te_outputs["hidden_state2"].to(accelerator.device, dtype=weight_dtype),
            te_outputs["pool2"].to(accelerator.device, dtype=weight_dtype),
        ]

    def _encode_live_text_conds(
        self,
        cfg: Any,
        accelerator: Any,
        batch: Any,
        text_encoders: list[Any],
        weight_dtype: torch.dtype,
        train_text_encoder: bool,
        is_train: bool,
    ) -> list[torch.Tensor]:
        """Encode SDXL text conditioning from captions or cached input IDs."""
        te_device = text_encoders[0].device
        models = self.get_models_for_text_encoding(cfg, accelerator, text_encoders)

        with torch.set_grad_enabled(is_train and train_text_encoder):
            if cfg.data.caption.weighted_captions:
                captions = batch.get("captions", [])
                if not captions:
                    raise ValueError("Weighted captions require raw captions in the batch - cannot encode text from cached inputs alone")

                input_ids_list, weights_list = self.tokenize_with_weights(captions)
                input_ids_list = [input_ids.to(te_device) for input_ids in input_ids_list]
                encoded_text_conds = self.encode_tokens_with_weights(models, input_ids_list, weights_list)
            else:
                input_ids = batch.get("input_ids")

                if input_ids is None:
                    captions = batch.get("captions", [])
                    if not captions:
                        raise ValueError("Batch has neither 'input_ids' nor 'captions' - cannot encode text")

                    input_ids_list = self.tokenize(captions)
                    input_ids_list = [input_ids.to(te_device) for input_ids in input_ids_list]
                else:
                    input_ids_list = [
                        input_ids["clip_l"].to(te_device),
                        input_ids["clip_g"].to(te_device),
                    ]

                encoded_text_conds = self.encode_tokens(models, input_ids_list)

            return [cond.to(accelerator.device, dtype=weight_dtype) for cond in encoded_text_conds]

    def _merge_text_conds(
        self,
        cached_text_conds: list[torch.Tensor],
        encoded_text_conds: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Prefer live-encoded outputs when they are recomputed for the current batch."""
        if not cached_text_conds:
            return encoded_text_conds

        merged_text_conds = list(cached_text_conds)
        for i, cond in enumerate(encoded_text_conds):
            if cond is not None:
                merged_text_conds[i] = cond
        return merged_text_conds

    def resolve_conditioning(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        train_text_encoder: bool,
        is_train: bool,
        weight_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resolve SDXL text conditioning from cached outputs, cached tokens, or live captions.

        Returns:
            Tuple of (encoder_hidden_states1, encoder_hidden_states2, pool2).
        """
        cached_text_conds = self._get_cached_text_conds(batch, accelerator, weight_dtype)
        needs_live_encoding = not cached_text_conds or any(cond is None for cond in cached_text_conds) or train_text_encoder

        if not needs_live_encoding:
            return tuple(cached_text_conds)

        encoded_text_conds = self._encode_live_text_conds(
            cfg=cfg,
            accelerator=accelerator,
            batch=batch,
            text_encoders=text_encoders,
            weight_dtype=weight_dtype,
            train_text_encoder=train_text_encoder,
            is_train=is_train,
        )
        return tuple(self._merge_text_conds(cached_text_conds, encoded_text_conds))
