"""Unit tests for SD3 strategy-local payload helpers."""

from types import SimpleNamespace

import pytest
import torch

from library.strategies.sd3.diffusion import build_sd3_flow_target
from library.strategies.sd3.checkpointing import Sd3CheckpointingStrategy
from library.strategies.sd3.encoding import Sd3TextConditioning, Sd3TokenizedText, concat_sd3_encodings


@pytest.mark.unit
def test_sd3_tokenized_text_reconstructs_attention_masks_from_cached_ids() -> None:
    tokenizers = [
        SimpleNamespace(pad_token_id=49407),
        SimpleNamespace(pad_token_id=0),
        SimpleNamespace(pad_token_id=0),
    ]
    input_ids = {
        "clip_l": torch.tensor([[49406, 10, 49407, 49407]]),
        "clip_g": torch.tensor([[1, 2, 0, 0]]),
        "t5": torch.tensor([[7, 8, 0, 0]]),
    }

    tokenized = Sd3TokenizedText.from_input_ids_dict(input_ids, tokenizers)

    assert torch.equal(tokenized.clip_l_attn_mask, torch.tensor([[1, 1, 0, 0]]))
    assert torch.equal(tokenized.clip_g_attn_mask, torch.tensor([[1, 1, 0, 0]]))
    assert torch.equal(tokenized.t5_attn_mask, torch.tensor([[1, 1, 0, 0]]))
    assert len(tokenized.to_tensor_list()) == 6


@pytest.mark.unit
def test_sd3_text_conditioning_roundtrips_cache_shape_and_merges_named_fields() -> None:
    conditioning = Sd3TextConditioning(
        lg_out=torch.ones(2, 77, 2048),
        t5_out=torch.ones(2, 16, 4096),
        lg_pooled=torch.ones(2, 2048),
        clip_l_attn_mask=torch.ones(2, 77, dtype=torch.long),
        clip_g_attn_mask=torch.ones(2, 77, dtype=torch.long),
        t5_attn_mask=torch.ones(2, 16, dtype=torch.long),
    )

    cache_dict = conditioning.to_te_output_dict()
    roundtrip = Sd3TextConditioning.from_te_output_dict(cache_dict)

    assert torch.equal(roundtrip.lg_out, conditioning.lg_out)
    assert torch.equal(roundtrip.lg_pooled, conditioning.lg_pooled)
    assert torch.equal(roundtrip.t5_attn_mask, conditioning.t5_attn_mask)

    override = Sd3TextConditioning(
        lg_out=None,
        t5_out=torch.zeros(2, 16, 4096),
        lg_pooled=None,
        clip_l_attn_mask=None,
        clip_g_attn_mask=None,
        t5_attn_mask=torch.zeros(2, 16, dtype=torch.long),
    )
    merged = conditioning.merged_with(override)

    assert torch.equal(merged.lg_out, conditioning.lg_out)
    assert torch.equal(merged.t5_out, override.t5_out)
    assert torch.equal(merged.t5_attn_mask, override.t5_attn_mask)


@pytest.mark.unit
def test_concat_sd3_encodings_uses_named_conditioning_payload() -> None:
    conditioning = Sd3TextConditioning(
        lg_out=torch.ones(1, 77, 2048),
        t5_out=None,
        lg_pooled=torch.ones(1, 2048),
        clip_l_attn_mask=torch.ones(1, 77, dtype=torch.long),
        clip_g_attn_mask=torch.ones(1, 77, dtype=torch.long),
        t5_attn_mask=torch.ones(1, 77, dtype=torch.long),
    )

    context, pooled = concat_sd3_encodings(conditioning)

    assert context.shape == (1, 154, 4096)
    assert pooled.shape == (1, 2048)


@pytest.mark.unit
def test_sd3_checkpoint_metadata_keeps_family_specific_attn_mask_fields() -> None:
    strategy = Sd3CheckpointingStrategy()
    metadata: dict[str, str] = {}
    cfg = SimpleNamespace(
        timestep=SimpleNamespace(
            timestep_sampling="cosine_shaped",
            rf_loss_weighting_scheme="cosmap",
            logit_mean=0.1,
            logit_std=1.2,
            cosine_shape_scale=1.5,
        ),
        model=SimpleNamespace(apply_lg_attn_mask=True, apply_t5_attn_mask=False),
    )

    strategy.update_metadata(metadata, cfg)

    assert metadata["ss_apply_lg_attn_mask"] == "True"
    assert metadata["ss_apply_t5_attn_mask"] == "False"
    assert "ss_timestep_sampling" not in metadata
    assert "ss_rf_loss_weighting_scheme" not in metadata


@pytest.mark.unit
def test_sd3_flow_target_matches_paper_velocity_direction() -> None:
    latents = torch.tensor([[[[1.0]]], [[[2.0]]]])
    noise = torch.tensor([[[[4.0]]], [[[7.0]]]])

    target = build_sd3_flow_target(latents, noise)

    assert torch.equal(target, noise - latents)
