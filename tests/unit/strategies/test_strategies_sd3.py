"""Unit tests for SD3 strategy-local payload helpers."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from library.losses.loss_modifiers import NoOpLossModifier
from library.config.dataclasses.timestep import TimestepConfig
from library.objectives.rectified_flow import RectifiedFlowObjectiveRuntime
from library.strategies.base.context import StrategyContext, StrategyPhase, current_strategy_context
from library.strategies.sd3.checkpointing import Sd3CheckpointingStrategy
from library.strategies.sd3.denoiser import Sd3DenoiserCallingStrategy
from library.strategies.sd3.diffusion import Sd3DiffusionTrainingStrategy, build_sd3_flow_target
from library.strategies.sd3.encoding import Sd3TextConditioning, Sd3TokenizedText, concat_sd3_encodings


class _TestSd3DenoiserStrategy(Sd3DiffusionTrainingStrategy, Sd3DenoiserCallingStrategy):
    pass


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


@pytest.mark.unit
def test_sd3_get_noise_pred_and_target_publishes_denoiser_forward_contexts() -> None:
    latents = torch.zeros(3, 1, 2, 2)
    noise = torch.ones_like(latents)
    timesteps = torch.tensor([100, 200, 300], dtype=torch.long)
    noisy_model_input = torch.full_like(latents, 0.5)
    weighting = torch.ones(3, 1, 1, 1)
    observed_contexts: list[StrategyContext] = []

    class _RecordingDenoiser:
        def __call__(self, model_input, denoiser_timesteps, *, context, y) -> torch.Tensor:
            del denoiser_timesteps, context, y
            observed_context = current_strategy_context()
            assert observed_context is not None
            observed_contexts.append(observed_context)
            return torch.ones_like(model_input)

    strategy = _TestSd3DenoiserStrategy()
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sd3"),
        objective=SimpleNamespace(prediction="flow"),
        performance=SimpleNamespace(memory=SimpleNamespace(gradient_checkpointing=False)),
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=lambda: nullcontext())
    objective_runtime = RectifiedFlowObjectiveRuntime(
        name="rectified_flow",
        num_train_timesteps=1000,
        timestep_runtime=None,
        loss_modifier=NoOpLossModifier(),
        timestep_config=TimestepConfig(),
        loss_weighting_scheme="none",
    )
    objective_runtime.build_training_batch_state = Mock(
        return_value=SimpleNamespace(
            noise=noise,
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            sigmas=torch.full((3, 1, 1, 1), 0.5),
            loss_weighting=weighting,
        )
    )

    conditioning = Sd3TextConditioning(
        lg_out=torch.ones(3, 77, 2048),
        t5_out=None,
        lg_pooled=torch.ones(3, 2048),
        clip_l_attn_mask=torch.ones(3, 77, dtype=torch.long),
        clip_g_attn_mask=torch.ones(3, 77, dtype=torch.long),
        t5_attn_mask=torch.ones(3, 77, dtype=torch.long),
    )
    trainable_model = SimpleNamespace(set_multiplier=Mock())
    batch = {
        "custom_attributes": [
            {"diff_output_preservation": True},
            {},
            {"diff_output_preservation": True},
        ]
    }

    strategy.get_noise_pred_and_target(
        cfg=cfg,
        accelerator=accelerator,
        objective_runtime=objective_runtime,
        latents=latents,
        batch=batch,
        text_encoder_conds=conditioning,
        denoiser=_RecordingDenoiser(),
        trainable_model=trainable_model,
        weight_dtype=torch.float32,
        train_denoiser=True,
        is_train=True,
        global_step=42,
    )

    assert len(observed_contexts) == 2
    normal_context, indexed_context = observed_contexts
    assert normal_context.phase is StrategyPhase.TRAIN
    assert normal_context.model_family == "sd3"
    assert normal_context.training is not None
    assert normal_context.training.global_step == 42
    assert normal_context.training.is_train is True
    assert normal_context.denoiser is not None
    assert torch.equal(normal_context.denoiser.timesteps, timesteps)
    assert normal_context.denoiser.sample_indices is None
    assert normal_context.denoiser.batch_size == 3

    assert indexed_context.phase is StrategyPhase.TRAIN
    assert indexed_context.training is not None
    assert indexed_context.training.global_step == 42
    assert indexed_context.training.is_train is True
    assert indexed_context.denoiser is not None
    assert torch.equal(indexed_context.denoiser.timesteps, torch.tensor([100, 300]))
    assert indexed_context.denoiser.sample_indices == (0, 2)
    assert indexed_context.denoiser.batch_size == 2
    assert current_strategy_context() is None
