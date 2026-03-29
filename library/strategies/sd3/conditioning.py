from typing import Any

import torch

from library.strategies.base.contracts import ConditioningStrategy
from library.strategies.sd3.encoding import Sd3TextConditioning, Sd3TokenizedText


class Sd3ConditioningStrategy(ConditioningStrategy):
    """Conditioning facet for SD3 flow-matching training strategies."""

    def _get_cached_text_conds(
        self,
        batch: Any,
        accelerator: Any,
        weight_dtype: torch.dtype,
    ) -> Sd3TextConditioning | None:
        """Load cached SD3 text-encoder outputs already attached to the batch."""
        te_outputs = batch.get("text_encoder_outputs")
        if te_outputs is None:
            return None

        conditioning = Sd3TextConditioning.from_te_output_dict(
            te_outputs,
            device=accelerator.device,
            weight_dtype=weight_dtype,
        )
        return self.drop_cached_text_encoder_outputs(conditioning)

    def _encode_live_text_conds(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        weight_dtype: torch.dtype,
        train_text_encoder: bool,
        is_train: bool,
    ) -> Sd3TextConditioning:
        """Encode SD3 text conditioning from token-cache tensors or live captions."""
        models = self.get_models_for_text_encoding(cfg, accelerator, text_encoders)

        if cfg.data.caption.weighted_captions:
            raise NotImplementedError("SD3 weighted captions are not implemented in the current strategy port")

        input_ids_dict = batch.get("input_ids")
        if input_ids_dict is not None:
            tokenized = Sd3TokenizedText.from_input_ids_dict(
                input_ids_dict,
                self.tokenizers,
                device=accelerator.device,
            )
        else:
            captions = batch.get("captions", [])
            if not captions:
                raise ValueError("Batch has neither 'input_ids' nor 'captions' - cannot encode SD3 text")
            tokenized = self.tokenize_to_payload(captions).to(accelerator.device)

        with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
            encoded_text_conds = self.encode_tokenized_text(models, tokenized)

        return encoded_text_conds.move_to(accelerator.device, weight_dtype=weight_dtype)

    def _merge_text_conds(
        self,
        cached_text_conds: Sd3TextConditioning | None,
        encoded_text_conds: Sd3TextConditioning,
    ) -> Sd3TextConditioning:
        """Prefer live-encoded outputs when they are recomputed for the current batch."""
        if cached_text_conds is None:
            return encoded_text_conds
        return cached_text_conds.merged_with(encoded_text_conds)

    def resolve_conditioning(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        train_text_encoder: bool,
        is_train: bool,
        weight_dtype: torch.dtype,
    ) -> Sd3TextConditioning:
        """Resolve SD3 text conditioning from cache or a live encoding pass."""
        cached_text_conds = self._get_cached_text_conds(batch, accelerator, weight_dtype)
        needs_live_encoding = (
            cached_text_conds is None
            or cached_text_conds.lg_out is None
            or cached_text_conds.t5_out is None
            or cached_text_conds.lg_pooled is None
            or train_text_encoder
        )

        if not needs_live_encoding:
            return cached_text_conds

        encoded_text_conds = self._encode_live_text_conds(
            batch=batch,
            text_encoders=text_encoders,
            accelerator=accelerator,
            cfg=cfg,
            weight_dtype=weight_dtype,
            train_text_encoder=train_text_encoder,
            is_train=is_train,
        )
        return self._merge_text_conds(cached_text_conds, encoded_text_conds)
