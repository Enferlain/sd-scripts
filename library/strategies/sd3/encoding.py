from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

from library.strategies.base.contracts import TextEncodingStrategy


@dataclass
class Sd3TokenizedText:
    """Named SD3 token payload used between tokenization and encoding."""

    clip_l_input_ids: torch.Tensor
    clip_g_input_ids: torch.Tensor
    t5_input_ids: torch.Tensor
    clip_l_attn_mask: torch.Tensor
    clip_g_attn_mask: torch.Tensor
    t5_attn_mask: torch.Tensor

    @classmethod
    def from_tensor_list(cls, tensors: list[torch.Tensor]) -> Sd3TokenizedText:
        """Build a named token payload from the current contract-order tensor list."""
        clip_l_input_ids, clip_g_input_ids, t5_input_ids, clip_l_attn_mask, clip_g_attn_mask, t5_attn_mask = tensors
        return cls(
            clip_l_input_ids=clip_l_input_ids,
            clip_g_input_ids=clip_g_input_ids,
            t5_input_ids=t5_input_ids,
            clip_l_attn_mask=clip_l_attn_mask,
            clip_g_attn_mask=clip_g_attn_mask,
            t5_attn_mask=t5_attn_mask,
        )

    @classmethod
    def from_input_ids_dict(
        cls,
        input_ids: dict[str, torch.Tensor],
        tokenizers: list[Any],
        *,
        device: Any | None = None,
    ) -> Sd3TokenizedText:
        """Build token ids plus reconstructed masks from cached input-id tensors."""
        tokenizer_l, tokenizer_g, tokenizer_t5 = tokenizers
        clip_l_input_ids = input_ids["clip_l"]
        clip_g_input_ids = input_ids["clip_g"]
        t5_input_ids = input_ids["t5"]

        if device is not None:
            clip_l_input_ids = clip_l_input_ids.to(device)
            clip_g_input_ids = clip_g_input_ids.to(device)
            t5_input_ids = t5_input_ids.to(device)

        return cls(
            clip_l_input_ids=clip_l_input_ids,
            clip_g_input_ids=clip_g_input_ids,
            t5_input_ids=t5_input_ids,
            clip_l_attn_mask=(clip_l_input_ids != tokenizer_l.pad_token_id).long(),
            clip_g_attn_mask=(clip_g_input_ids != tokenizer_g.pad_token_id).long(),
            t5_attn_mask=(t5_input_ids != tokenizer_t5.pad_token_id).long(),
        )

    def to(self, device: Any) -> Sd3TokenizedText:
        """Move token tensors and masks to a device."""
        return Sd3TokenizedText(
            clip_l_input_ids=self.clip_l_input_ids.to(device),
            clip_g_input_ids=self.clip_g_input_ids.to(device),
            t5_input_ids=self.t5_input_ids.to(device),
            clip_l_attn_mask=self.clip_l_attn_mask.to(device),
            clip_g_attn_mask=self.clip_g_attn_mask.to(device),
            t5_attn_mask=self.t5_attn_mask.to(device),
        )

    def token_ids_only(self) -> list[torch.Tensor]:
        """Return only the token-id tensors used by per-epoch token caching."""
        return [self.clip_l_input_ids, self.clip_g_input_ids, self.t5_input_ids]

    def to_tensor_list(self) -> list[torch.Tensor]:
        """Return the current contract-order tensor list."""
        return [
            self.clip_l_input_ids,
            self.clip_g_input_ids,
            self.t5_input_ids,
            self.clip_l_attn_mask,
            self.clip_g_attn_mask,
            self.t5_attn_mask,
        ]


@dataclass
class Sd3TextConditioning:
    """Named SD3 encoded conditioning payload used before denoiser calls."""

    lg_out: torch.Tensor | None
    t5_out: torch.Tensor | None
    lg_pooled: torch.Tensor | None
    clip_l_attn_mask: torch.Tensor | None
    clip_g_attn_mask: torch.Tensor | None
    t5_attn_mask: torch.Tensor | None

    @classmethod
    def from_tensor_list(cls, tensors: list[torch.Tensor | None]) -> Sd3TextConditioning:
        """Build a named conditioning payload from the current contract-order tensor list."""
        lg_out, t5_out, lg_pooled, clip_l_attn_mask, clip_g_attn_mask, t5_attn_mask = tensors
        return cls(
            lg_out=lg_out,
            t5_out=t5_out,
            lg_pooled=lg_pooled,
            clip_l_attn_mask=clip_l_attn_mask,
            clip_g_attn_mask=clip_g_attn_mask,
            t5_attn_mask=t5_attn_mask,
        )

    @classmethod
    def from_te_output_dict(
        cls,
        te_outputs: dict[str, torch.Tensor],
        *,
        device: Any | None = None,
        weight_dtype: torch.dtype | None = None,
    ) -> Sd3TextConditioning:
        """Build conditioning from the dataloader/cache TE-output dict shape."""
        lg_out = te_outputs["lg_out"]
        lg_pooled = te_outputs["lg_pooled"]
        t5_out = te_outputs.get("t5_out")
        clip_l_attn_mask = te_outputs.get("clip_l_attn_mask")
        clip_g_attn_mask = te_outputs.get("clip_g_attn_mask")
        t5_attn_mask = te_outputs.get("t5_attn_mask")

        conditioning = cls(
            lg_out=lg_out,
            t5_out=t5_out,
            lg_pooled=lg_pooled,
            clip_l_attn_mask=clip_l_attn_mask,
            clip_g_attn_mask=clip_g_attn_mask,
            t5_attn_mask=t5_attn_mask,
        )
        if device is None:
            return conditioning
        return conditioning.move_to(device, weight_dtype=weight_dtype)

    def to_tensor_list(self) -> list[torch.Tensor | None]:
        """Return the current contract-order tensor list."""
        return [
            self.lg_out,
            self.t5_out,
            self.lg_pooled,
            self.clip_l_attn_mask,
            self.clip_g_attn_mask,
            self.t5_attn_mask,
        ]

    def move_to(
        self,
        device: Any,
        *,
        weight_dtype: torch.dtype | None = None,
    ) -> Sd3TextConditioning:
        """Move conditioning tensors to a device, casting float features if requested."""

        def _move_feature(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            kwargs = {"device": device}
            if weight_dtype is not None and tensor.dtype.is_floating_point:
                kwargs["dtype"] = weight_dtype
            return tensor.to(**kwargs)

        def _move_mask(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return None if tensor is None else tensor.to(device)

        return Sd3TextConditioning(
            lg_out=_move_feature(self.lg_out),
            t5_out=_move_feature(self.t5_out),
            lg_pooled=_move_feature(self.lg_pooled),
            clip_l_attn_mask=_move_mask(self.clip_l_attn_mask),
            clip_g_attn_mask=_move_mask(self.clip_g_attn_mask),
            t5_attn_mask=_move_mask(self.t5_attn_mask),
        )

    def merged_with(self, override: Sd3TextConditioning) -> Sd3TextConditioning:
        """Prefer non-None values from an override conditioning payload."""
        return Sd3TextConditioning(
            lg_out=override.lg_out if override.lg_out is not None else self.lg_out,
            t5_out=override.t5_out if override.t5_out is not None else self.t5_out,
            lg_pooled=override.lg_pooled if override.lg_pooled is not None else self.lg_pooled,
            clip_l_attn_mask=override.clip_l_attn_mask if override.clip_l_attn_mask is not None else self.clip_l_attn_mask,
            clip_g_attn_mask=override.clip_g_attn_mask if override.clip_g_attn_mask is not None else self.clip_g_attn_mask,
            t5_attn_mask=override.t5_attn_mask if override.t5_attn_mask is not None else self.t5_attn_mask,
        )

    def select(self, index: int) -> Sd3TextConditioning:
        """Select a single sample from a batched conditioning payload."""
        return Sd3TextConditioning(
            lg_out=None if self.lg_out is None else self.lg_out[index],
            t5_out=None if self.t5_out is None else self.t5_out[index],
            lg_pooled=None if self.lg_pooled is None else self.lg_pooled[index],
            clip_l_attn_mask=None if self.clip_l_attn_mask is None else self.clip_l_attn_mask[index],
            clip_g_attn_mask=None if self.clip_g_attn_mask is None else self.clip_g_attn_mask[index],
            t5_attn_mask=None if self.t5_attn_mask is None else self.t5_attn_mask[index],
        )

    def to_te_output_dict(self) -> dict[str, torch.Tensor]:
        """Return the dataloader/cache dict shape for encoded SD3 text outputs."""
        assert self.lg_out is not None, "lg_out must be present to build TE output dict"
        assert self.lg_pooled is not None, "lg_pooled must be present to build TE output dict"
        assert self.clip_l_attn_mask is not None, "clip_l_attn_mask must be present to build TE output dict"
        assert self.clip_g_attn_mask is not None, "clip_g_attn_mask must be present to build TE output dict"
        assert self.t5_attn_mask is not None, "t5_attn_mask must be present to build TE output dict"

        outputs = {
            "lg_out": self.lg_out,
            "lg_pooled": self.lg_pooled,
            "clip_l_attn_mask": self.clip_l_attn_mask,
            "clip_g_attn_mask": self.clip_g_attn_mask,
            "t5_attn_mask": self.t5_attn_mask,
        }
        if self.t5_out is not None:
            outputs["t5_out"] = self.t5_out
        return outputs


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device for a text encoder module."""
    return next(model.parameters()).device


def build_sd3_attention_masks(
    tokenizers: list[Any],
    input_ids: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Reconstruct attention masks from cached token ids."""
    tokenized = Sd3TokenizedText.from_input_ids_dict(
        {"clip_l": input_ids[0], "clip_g": input_ids[1], "t5": input_ids[2]},
        tokenizers,
    )
    return [
        tokenized.clip_l_attn_mask,
        tokenized.clip_g_attn_mask,
        tokenized.t5_attn_mask,
    ]


def encode_sd3_tokens(
    models: list[Any],
    tokens: Sd3TokenizedText | list[torch.Tensor],
    *,
    apply_lg_attn_mask: bool = False,
    apply_t5_attn_mask: bool = False,
    l_dropout_rate: float = 0.0,
    g_dropout_rate: float = 0.0,
    t5_dropout_rate: float = 0.0,
    enable_dropout: bool = True,
) -> Sd3TextConditioning:
    """Encode SD3 token tensors into CLIP/T5 conditioning outputs."""
    if isinstance(tokens, list):
        tokens = Sd3TokenizedText.from_tensor_list(tokens)

    clip_l, clip_g, t5xxl = models
    clip_l = clip_l if clip_l is None else clip_l
    clip_g = clip_g if clip_g is None else clip_g
    t5xxl = t5xxl if t5xxl is None else t5xxl

    l_tokens = tokens.clip_l_input_ids
    g_tokens = tokens.clip_g_input_ids
    t5_tokens = tokens.t5_input_ids
    l_attn_mask = tokens.clip_l_attn_mask
    g_attn_mask = tokens.clip_g_attn_mask
    t5_attn_mask = tokens.t5_attn_mask

    if l_tokens is None or clip_l is None:
        assert g_tokens is None, "g_tokens must be None if l_tokens is None"
        lg_out = None
        lg_pooled = None
        l_attn_mask = None
        g_attn_mask = None
    else:
        assert g_tokens is not None, "g_tokens must not be None if l_tokens is not None"

        batch_size, l_seq_len = l_tokens.shape
        g_seq_len = g_tokens.shape[1]

        non_drop_l_indices = []
        non_drop_g_indices = []
        for i in range(l_tokens.shape[0]):
            drop_l = enable_dropout and (l_dropout_rate > 0.0 and random.random() < l_dropout_rate)
            drop_g = enable_dropout and (g_dropout_rate > 0.0 and random.random() < g_dropout_rate)
            if not drop_l:
                non_drop_l_indices.append(i)
            if not drop_g:
                non_drop_g_indices.append(i)

        if len(non_drop_l_indices) > 0 and len(non_drop_l_indices) < batch_size:
            l_tokens = l_tokens[non_drop_l_indices]
            l_attn_mask = l_attn_mask[non_drop_l_indices]
        if len(non_drop_g_indices) > 0 and len(non_drop_g_indices) < batch_size:
            g_tokens = g_tokens[non_drop_g_indices]
            g_attn_mask = g_attn_mask[non_drop_g_indices]

        if len(non_drop_l_indices) > 0:
            clip_l_device = get_model_device(clip_l)
            nd_l_attn_mask = l_attn_mask.to(clip_l_device)
            prompt_embeds = clip_l(
                l_tokens.to(clip_l_device),
                nd_l_attn_mask if apply_lg_attn_mask else None,
                output_hidden_states=True,
            )
            nd_l_pooled = prompt_embeds[0]
            nd_l_out = prompt_embeds.hidden_states[-2]
        if len(non_drop_g_indices) > 0:
            clip_g_device = get_model_device(clip_g)
            nd_g_attn_mask = g_attn_mask.to(clip_g_device)
            prompt_embeds = clip_g(
                g_tokens.to(clip_g_device),
                nd_g_attn_mask if apply_lg_attn_mask else None,
                output_hidden_states=True,
            )
            nd_g_pooled = prompt_embeds[0]
            nd_g_out = prompt_embeds.hidden_states[-2]

        if len(non_drop_l_indices) == batch_size:
            l_pooled = nd_l_pooled
            l_out = nd_l_out
        else:
            clip_l_device = get_model_device(clip_l)
            l_pooled = torch.zeros((batch_size, 768), device=clip_l_device, dtype=torch.float32)
            l_out = torch.zeros((batch_size, l_seq_len, 768), device=clip_l_device, dtype=torch.float32)
            l_attn_mask = torch.zeros((batch_size, l_seq_len), device=clip_l_device, dtype=l_attn_mask.dtype)
            if len(non_drop_l_indices) > 0:
                l_pooled[non_drop_l_indices] = nd_l_pooled
                l_out[non_drop_l_indices] = nd_l_out
                l_attn_mask[non_drop_l_indices] = nd_l_attn_mask

        if len(non_drop_g_indices) == batch_size:
            g_pooled = nd_g_pooled
            g_out = nd_g_out
        else:
            clip_g_device = get_model_device(clip_g)
            g_pooled = torch.zeros((batch_size, 1280), device=clip_g_device, dtype=torch.float32)
            g_out = torch.zeros((batch_size, g_seq_len, 1280), device=clip_g_device, dtype=torch.float32)
            g_attn_mask = torch.zeros((batch_size, g_seq_len), device=clip_g_device, dtype=g_attn_mask.dtype)
            if len(non_drop_g_indices) > 0:
                g_pooled[non_drop_g_indices] = nd_g_pooled
                g_out[non_drop_g_indices] = nd_g_out
                g_attn_mask[non_drop_g_indices] = nd_g_attn_mask

        lg_pooled = torch.cat((l_pooled, g_pooled), dim=-1)
        lg_out = torch.cat([l_out, g_out], dim=-1)

    if t5xxl is None or t5_tokens is None:
        t5_out = None
        t5_attn_mask = None
    else:
        batch_size, t5_seq_len = t5_tokens.shape
        non_drop_t5_indices = []
        for i in range(t5_tokens.shape[0]):
            drop_t5 = enable_dropout and (t5_dropout_rate > 0.0 and random.random() < t5_dropout_rate)
            if not drop_t5:
                non_drop_t5_indices.append(i)

        if len(non_drop_t5_indices) > 0 and len(non_drop_t5_indices) < batch_size:
            t5_tokens = t5_tokens[non_drop_t5_indices]
            t5_attn_mask = t5_attn_mask[non_drop_t5_indices]

        if len(non_drop_t5_indices) > 0:
            t5_device = get_model_device(t5xxl)
            nd_t5_attn_mask = t5_attn_mask.to(t5_device)
            nd_t5_out, _ = t5xxl(
                t5_tokens.to(t5_device),
                nd_t5_attn_mask if apply_t5_attn_mask else None,
                return_dict=False,
                output_hidden_states=True,
            )

        if len(non_drop_t5_indices) == batch_size:
            t5_out = nd_t5_out
        else:
            t5_device = get_model_device(t5xxl)
            t5_out = torch.zeros((batch_size, t5_seq_len, 4096), device=t5_device, dtype=torch.float32)
            t5_attn_mask = torch.zeros((batch_size, t5_seq_len), device=t5_device, dtype=t5_attn_mask.dtype)
            if len(non_drop_t5_indices) > 0:
                t5_out[non_drop_t5_indices] = nd_t5_out
                t5_attn_mask[non_drop_t5_indices] = nd_t5_attn_mask

    return Sd3TextConditioning(
        lg_out=lg_out,
        t5_out=t5_out,
        lg_pooled=lg_pooled,
        clip_l_attn_mask=l_attn_mask,
        clip_g_attn_mask=g_attn_mask,
        t5_attn_mask=t5_attn_mask,
    )


def drop_cached_sd3_text_encoder_outputs(
    conditioning: Sd3TextConditioning,
    *,
    l_dropout_rate: float = 0.0,
    g_dropout_rate: float = 0.0,
    t5_dropout_rate: float = 0.0,
) -> Sd3TextConditioning:
    """Apply SD3 encoder dropout to cached text encoder outputs."""
    lg_out = conditioning.lg_out
    t5_out = conditioning.t5_out
    lg_pooled = conditioning.lg_pooled
    l_attn_mask = conditioning.clip_l_attn_mask
    g_attn_mask = conditioning.clip_g_attn_mask
    t5_attn_mask = conditioning.t5_attn_mask

    if lg_out is not None and lg_pooled is not None:
        for i in range(lg_out.shape[0]):
            drop_l = l_dropout_rate > 0.0 and random.random() < l_dropout_rate
            if drop_l:
                lg_out[i, :, :768] = torch.zeros_like(lg_out[i, :, :768])
                lg_pooled[i, :768] = torch.zeros_like(lg_pooled[i, :768])
                if l_attn_mask is not None:
                    l_attn_mask[i] = torch.zeros_like(l_attn_mask[i])
            drop_g = g_dropout_rate > 0.0 and random.random() < g_dropout_rate
            if drop_g:
                lg_out[i, :, 768:] = torch.zeros_like(lg_out[i, :, 768:])
                lg_pooled[i, 768:] = torch.zeros_like(lg_pooled[i, 768:])
                if g_attn_mask is not None:
                    g_attn_mask[i] = torch.zeros_like(g_attn_mask[i])

    if t5_out is not None:
        for i in range(t5_out.shape[0]):
            drop_t5 = t5_dropout_rate > 0.0 and random.random() < t5_dropout_rate
            if drop_t5:
                t5_out[i] = torch.zeros_like(t5_out[i])
                if t5_attn_mask is not None:
                    t5_attn_mask[i] = torch.zeros_like(t5_attn_mask[i])

    return Sd3TextConditioning(
        lg_out=lg_out,
        t5_out=t5_out,
        lg_pooled=lg_pooled,
        clip_l_attn_mask=l_attn_mask,
        clip_g_attn_mask=g_attn_mask,
        t5_attn_mask=t5_attn_mask,
    )


def concat_sd3_encodings(
    conditioning: Sd3TextConditioning,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate CLIP and T5 conditioning into SD3 context tensors."""
    assert conditioning.lg_out is not None, "lg_out must be present for SD3 denoiser conditioning"
    assert conditioning.lg_pooled is not None, "lg_pooled must be present for SD3 denoiser conditioning"
    lg_out = conditioning.lg_out
    t5_out = conditioning.t5_out
    lg_pooled = conditioning.lg_pooled

    lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
    if t5_out is None:
        t5_out = torch.zeros((lg_out.shape[0], 77, 4096), device=lg_out.device, dtype=lg_out.dtype)
    return torch.cat([lg_out, t5_out], dim=-2), lg_pooled


class Sd3TextEncodingStrategy(TextEncodingStrategy):
    """Text encoding strategy for SD3."""

    def __init__(
        self,
        tokenizers: list[Any] | None = None,
        *,
        apply_lg_attn_mask: bool | None = None,
        apply_t5_attn_mask: bool | None = None,
        l_dropout_rate: float = 0.0,
        g_dropout_rate: float = 0.0,
        t5_dropout_rate: float = 0.0,
    ) -> None:
        self._tokenizers = tokenizers
        self.apply_lg_attn_mask = apply_lg_attn_mask
        self.apply_t5_attn_mask = apply_t5_attn_mask
        self.l_dropout_rate = l_dropout_rate
        self.g_dropout_rate = g_dropout_rate
        self.t5_dropout_rate = t5_dropout_rate

    def _get_tokenizers(self) -> list[Any]:
        """Resolve tokenizer state from explicit construction or the tokenization facet."""
        tokenizers = getattr(self, "_tokenizers", None)
        if tokenizers is not None:
            return tokenizers

        tokenizers = getattr(self, "tokenizers", None)
        if tokenizers is None:
            raise RuntimeError(
                "SD3 text encoding requires tokenizers. Pass them to "
                "Sd3TextEncodingStrategy(...) or mix in Sd3TokenizeStrategy."
            )
        return tokenizers

    def encode_tokens(self, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode SD3 token tensors into CLIP/T5 outputs."""
        return self.encode_tokenized_text(models, tokens).to_tensor_list()

    def encode_tokenized_text(
        self,
        models: list[Any],
        tokens: Sd3TokenizedText | list[torch.Tensor],
    ) -> Sd3TextConditioning:
        """Encode named SD3 token payloads into named conditioning outputs."""
        apply_lg_attn_mask = False if self.apply_lg_attn_mask is None else self.apply_lg_attn_mask
        apply_t5_attn_mask = False if self.apply_t5_attn_mask is None else self.apply_t5_attn_mask
        return encode_sd3_tokens(
            models,
            tokens,
            apply_lg_attn_mask=apply_lg_attn_mask,
            apply_t5_attn_mask=apply_t5_attn_mask,
            l_dropout_rate=self.l_dropout_rate,
            g_dropout_rate=self.g_dropout_rate,
            t5_dropout_rate=self.t5_dropout_rate,
        )

    def encode_te_outputs_in_memory(
        self,
        text_encoders: list[Any],
        tokenizers: list[Any],
        caption: str,
        max_token_length: int,
        device: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute SD3 text encoder outputs for a single caption."""
        del device  # tokenizers/models carry their own device placement
        from library.strategies.sd3.tokenization import tokenize_sd3_text

        t5_max_length = max_token_length if max_token_length is not None else None
        tokens = tokenize_sd3_text(tokenizers[0], tokenizers[1], tokenizers[2], t5_max_length or 256, [caption])

        with torch.no_grad():
            conditioning = encode_sd3_tokens(
                text_encoders,
                tokens,
                apply_lg_attn_mask=False if self.apply_lg_attn_mask is None else self.apply_lg_attn_mask,
                apply_t5_attn_mask=False if self.apply_t5_attn_mask is None else self.apply_t5_attn_mask,
                enable_dropout=False,
            )

        return {key: value.squeeze(0).cpu() for key, value in conditioning.to_te_output_dict().items()}

    def get_models_for_text_encoding(self, cfg: Any, accelerator: Any, text_encoders: list[Any]) -> list[Any]:
        """Return SD3 text encoders for encoding."""
        del cfg, accelerator
        return text_encoders

    def drop_cached_text_encoder_outputs(self, conditioning: Sd3TextConditioning) -> Sd3TextConditioning:
        """Apply encoder dropout to cached SD3 text outputs."""
        return drop_cached_sd3_text_encoder_outputs(
            conditioning,
            l_dropout_rate=self.l_dropout_rate,
            g_dropout_rate=self.g_dropout_rate,
            t5_dropout_rate=self.t5_dropout_rate,
        )

    def concat_encodings(self, conditioning: Sd3TextConditioning) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate SD3 encoder outputs into context and pooled tensors."""
        return concat_sd3_encodings(conditioning)


__all__ = [
    "Sd3TextConditioning",
    "Sd3TextEncodingStrategy",
    "Sd3TokenizedText",
    "build_sd3_attention_masks",
    "concat_sd3_encodings",
    "drop_cached_sd3_text_encoder_outputs",
    "encode_sd3_tokens",
]
