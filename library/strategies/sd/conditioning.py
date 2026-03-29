from typing import Any

import torch

from library.strategies.base.contracts import ConditioningStrategy


class SdConditioningStrategy(ConditioningStrategy):
    """Conditioning facet for SD 1.5/2.0 training strategies."""

    def _get_cached_text_conds(
        self,
        batch: Any,
        accelerator: Any,
        weight_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Load cached SD text-encoder outputs already attached to the batch."""
        te_outputs = batch.get("text_encoder_outputs")
        if te_outputs is None:
            return []
        return [te_outputs["hidden_state"].to(accelerator.device, dtype=weight_dtype)]

    def _encode_live_text_conds(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        weight_dtype: torch.dtype,
        train_text_encoder: bool,
        is_train: bool,
    ) -> list[torch.Tensor]:
        """Encode SD text conditioning from captions or cached input IDs."""
        models = self.get_models_for_text_encoding(cfg, accelerator, text_encoders)

        with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
            if cfg.data.caption.weighted_captions:
                input_ids_list, weights_list = self.tokenize_with_weights(batch["captions"])
                encoded_text_conds = self.encode_tokens_with_weights(models, input_ids_list, weights_list)
            else:
                input_ids_dict = batch.get("input_ids")
                if input_ids_dict is not None:
                    input_ids_list = [input_ids_dict["clip"].to(accelerator.device)]
                else:
                    input_ids_list = [ids.to(accelerator.device) for ids in self.tokenize(batch["captions"])]
                encoded_text_conds = self.encode_tokens(models, input_ids_list)

            if cfg.performance.precision.full_fp16:
                encoded_text_conds = [cond.to(weight_dtype) for cond in encoded_text_conds]

        return encoded_text_conds

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
    ) -> list[torch.Tensor]:
        """Resolve SD text conditioning from cache or a live encoding pass."""
        cached_text_conds = self._get_cached_text_conds(batch, accelerator, weight_dtype)
        needs_live_encoding = not cached_text_conds or cached_text_conds[0] is None or train_text_encoder

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
