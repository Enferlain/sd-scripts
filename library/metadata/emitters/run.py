"""Central emitters for training-run metadata assembly."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from library.metadata.dataclasses.model import ModelSpecFacts
from library.metadata.dataclasses.run import RunMetadataFacts
from library.metadata.projections import stringify_metadata_mapping
from library.metadata.providers import MetadataProviderResult
from library.metadata.records import MetadataIdentity, MetadataValue, ModelComponentMetadataRecord, RunMetadataRecord
from library.metadata.versions import METADATA_PAYLOAD_VERSION

if TYPE_CHECKING:
    from library.data import DatasetManifest
    from library.objectives import ObjectiveDefinition


# Reference only for a future alias/compatibility option. This mapping is not
# used in the active metadata path.
LEGACY_SS_METADATA_ALIASES = {
    "session_id": "ss_session_id",
    "training_started_at": "ss_training_started_at",
    "training_finished_at": "ss_training_finished_at",
    "output_name": "ss_output_name",
    "learning_rate": "ss_learning_rate",
    "text_encoder_lr": "ss_text_encoder_lr",
    "unet_lr": "ss_unet_lr",
    "num_train_images": "ss_num_train_images",
    "num_validation_images": "ss_num_validation_images",
    "num_reg_images": "ss_num_reg_images",
    "num_batches_per_epoch": "ss_num_batches_per_epoch",
    "num_epochs": "ss_num_epochs",
    "gradient_checkpointing": "ss_gradient_checkpointing",
    "gradient_accumulation_steps": "ss_gradient_accumulation_steps",
    "max_train_steps": "ss_max_train_steps",
    "lr_warmup_steps": "ss_lr_warmup_steps",
    "lr_scheduler": "ss_lr_scheduler",
    "mixed_precision": "ss_mixed_precision",
    "full_fp16": "ss_full_fp16",
    "is_v2": "ss_v2",
    "base_model_version": "ss_base_model_version",
    "clip_skip": "ss_clip_skip",
    "max_token_length": "ss_max_token_length",
    "cache_latents": "ss_cache_latents",
    "seed": "ss_seed",
    "lowram": "ss_lowram",
    "noise_offset": "ss_noise_offset",
    "multires_noise_iterations": "ss_multires_noise_iterations",
    "multires_noise_discount": "ss_multires_noise_discount",
    "adaptive_noise_scale": "ss_adaptive_noise_scale",
    "zero_terminal_snr": "ss_zero_terminal_snr",
    "sd_scripts_commit_hash": "ss_sd_scripts_commit_hash",
    "optimizer": "ss_optimizer",
    "max_grad_norm": "ss_max_grad_norm",
    "caption_dropout_rate": "ss_caption_dropout_rate",
    "caption_dropout_every_n_epochs": "ss_caption_dropout_every_n_epochs",
    "caption_tag_dropout_rate": "ss_caption_tag_dropout_rate",
    "face_crop_aug_range": "ss_face_crop_aug_range",
    "prior_loss_weight": "ss_prior_loss_weight",
    "min_snr_gamma": "ss_min_snr_gamma",
    "ip_noise_gamma": "ss_ip_noise_gamma",
    "debiased_estimation": "ss_debiased_estimation",
    "noise_offset_random_strength": "ss_noise_offset_random_strength",
    "ip_noise_gamma_random_strength": "ss_ip_noise_gamma_random_strength",
    "loss_type": "ss_loss_type",
    "huber_schedule": "ss_huber_schedule",
    "huber_scale": "ss_huber_scale",
    "huber_c": "ss_huber_c",
    "fp8_base": "ss_fp8_base",
    "fp8_base_unet": "ss_fp8_base_unet",
    "validation_seed": "ss_validation_seed",
    "validation_split": "ss_validation_split",
    "max_validation_steps": "ss_max_validation_steps",
    "validate_every_n_epochs": "ss_validate_every_n_epochs",
    "validate_every_n_steps": "ss_validate_every_n_steps",
    "run_validation_at_start": "ss_run_validation_at_start",
    "run_validation_at_end": "ss_run_validation_at_end",
    "resize_interpolation": "ss_resize_interpolation",
    "adapter_method": "ss_adapter_module",
    "adapter_rank": "ss_adapter_rank",
    "adapter_alpha": "ss_adapter_alpha",
    "adapter_neuron_dropout": "ss_adapter_neuron_dropout",
    "training_comment": "ss_training_comment",
    "scale_weight_norms": "ss_scale_weight_norms",
    "batch_size_per_device": "ss_batch_size_per_device",
    "total_batch_size": "ss_total_batch_size",
    "resolution": "ss_resolution",
    "color_aug": "ss_color_aug",
    "flip_aug": "ss_flip_aug",
    "random_crop": "ss_random_crop",
    "shuffle_caption": "ss_shuffle_caption",
    "enable_bucket": "ss_enable_bucket",
    "bucket_no_upscale": "ss_bucket_no_upscale",
    "min_bucket_reso": "ss_min_bucket_reso",
    "max_bucket_reso": "ss_max_bucket_reso",
    "keep_tokens": "ss_keep_tokens",
    "pretrained_model_name": "ss_sd_model_name",
    "vae_name": "ss_vae_name",
    "epoch": "ss_epoch",
    "steps": "ss_steps",
    "timestep_sampling": "ss_timestep_sampling",
    "rf_loss_weighting_scheme": "ss_rf_loss_weighting_scheme",
    "training_shift": "ss_training_shift",
    "logit_mean": "ss_logit_mean",
    "logit_std": "ss_logit_std",
    "cosine_shape_scale": "ss_cosine_shape_scale",
}

LEGACY_SS_COMPATIBILITY_ONLY_FIELDS = (
    "ss_dataset_dirs",
    "ss_reg_dataset_dirs",
    "ss_tag_frequency",
    "ss_bucket_info",
    "ss_sd_model_hash",
    "ss_new_sd_model_hash",
    "ss_vae_hash",
    "ss_new_vae_hash",
)


@dataclass(frozen=True)
class TrainingMetadataBuildContext:
    """Inputs needed to assemble training-run metadata facts."""

    cfg: Any
    manifest: DatasetManifest
    val_manifest: DatasetManifest | None
    session_id: int
    training_started_at: float
    model_version: str
    num_train_epochs: int
    optimizer_name: str
    optimizer_args: str
    num_batches_per_epoch: int
    total_batch_size: int
    objective: ObjectiveDefinition | None = None


@dataclass(frozen=True)
class TrainingMetadataBundle:
    """Training metadata facts for checkpoint projections."""

    full: RunMetadataFacts


@dataclass(frozen=True)
class TrainingMetadataState:
    """Active training metadata state for checkpoint exports."""

    full: RunMetadataFacts

    @classmethod
    def from_bundle(cls, bundle: TrainingMetadataBundle) -> TrainingMetadataState:
        """Create checkpoint metadata state from the initial training bundle."""
        return cls(full=bundle.full)

    def with_metadata(self, metadata: Mapping[str, MetadataValue]) -> TrainingMetadataState:
        """Return a copy with metadata facts merged into the active run view."""
        full = self.full.with_metadata(stringify_metadata_mapping(metadata))
        return TrainingMetadataState(full=full)

    def with_fact(self, key: str, value: MetadataValue) -> TrainingMetadataState:
        """Return a copy with one updated training-run fact."""
        return self.with_metadata({key: value})

    def build_checkpoint_metadata(
        self,
        *,
        model_metadata: Mapping[str, str],
        no_metadata: bool,
        artifact_identifier: str = "checkpoint",
        step: int | None = None,
        epoch: int | None = None,
    ) -> dict[str, str]:
        """Project this training metadata state for a checkpoint artifact."""
        from library.metadata.emitters.checkpoint import build_checkpoint_metadata

        return build_checkpoint_metadata(
            training_facts=self.full,
            model_facts=ModelSpecFacts.from_modelspec_metadata(model_metadata),
            no_metadata=no_metadata,
            artifact_identifier=artifact_identifier,
            step=step,
            epoch=epoch,
        )


def build_training_run_metadata(
    facts: RunMetadataFacts,
    *,
    provider_id: str = "training.run",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build collected metadata for typed training-run facts."""
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=[_build_training_run_record(facts, producer=provider_id)],
    )


def build_model_spec_metadata(
    facts: ModelSpecFacts,
    *,
    provider_id: str = "model.modelspec",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build collected metadata for typed model-spec-compatible facts."""
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=[_build_model_spec_record(facts, producer=provider_id)],
    )


def build_objective_run_metadata(cfg: Any, objective_name: str) -> dict[str, MetadataValue]:
    """Return repo-owned objective/runtime metadata for the active objective."""
    if objective_name != "rectified_flow":
        return {}

    return {
        "timestep_sampling": cfg.timestep.timestep_sampling,
        "rf_loss_weighting_scheme": cfg.timestep.rf_loss_weighting_scheme,
        "training_shift": cfg.timestep.training_shift,
        "logit_mean": cfg.timestep.logit_mean,
        "logit_std": cfg.timestep.logit_std,
        "cosine_shape_scale": cfg.timestep.cosine_shape_scale,
    }


def build_training_metadata_bundle(context: TrainingMetadataBuildContext) -> TrainingMetadataBundle:
    """Build typed training-run facts for checkpoint metadata export."""
    from library.objectives import build_objective

    metadata = _build_training_run_metadata(context)
    objective = context.objective if context.objective is not None else build_objective(context.cfg)
    metadata.update(build_objective_run_metadata(context.cfg, objective.name))

    string_metadata = stringify_metadata_mapping(metadata)
    run_identifier = str(context.session_id)
    return TrainingMetadataBundle(full=RunMetadataFacts(run_identifier=run_identifier, metadata=string_metadata))


def _build_training_run_record(facts: RunMetadataFacts, *, producer: str) -> RunMetadataRecord:
    return RunMetadataRecord(
        identity=MetadataIdentity(
            entity_type="run",
            identifier=facts.run_identifier,
            schema_version=METADATA_PAYLOAD_VERSION,
        ),
        producer=producer,
        facts=dict(facts.metadata),
        schema_version=METADATA_PAYLOAD_VERSION,
    )


def _build_model_spec_record(facts: ModelSpecFacts, *, producer: str) -> ModelComponentMetadataRecord:
    metadata: dict[str, MetadataValue] = dict(facts.compatibility_metadata)
    if facts.architecture is not None:
        metadata["architecture"] = facts.architecture
    if facts.implementation is not None:
        metadata["implementation"] = facts.implementation
    if facts.prediction_type is not None:
        metadata["prediction_type"] = facts.prediction_type
    return ModelComponentMetadataRecord(
        identity=MetadataIdentity(
            entity_type="model",
            identifier=facts.model_identifier,
            schema_version=METADATA_PAYLOAD_VERSION,
        ),
        producer=producer,
        facts=metadata,
        schema_version=METADATA_PAYLOAD_VERSION,
    )


def _build_training_run_metadata(context: TrainingMetadataBuildContext) -> dict[str, MetadataValue]:
    from library.utils.hash_utils import get_git_revision_hash

    cfg = context.cfg
    manifest = context.manifest

    num_train_images = sum(e.num_repeats for e in manifest.entries.values() if not e.is_reg)
    num_reg_images = sum(e.num_repeats for e in manifest.entries.values() if e.is_reg)
    num_val_images = sum(e.num_repeats for e in context.val_manifest.entries.values()) if context.val_manifest else 0

    metadata: dict[str, MetadataValue] = {
        "session_id": context.session_id,
        "training_started_at": context.training_started_at,
        "output_name": cfg.output.saving.output_name,
        "learning_rate": cfg.optimizer.learning_rates.base,
        "text_encoder_lr": cfg.optimizer.learning_rates.text_encoders,
        "unet_lr": cfg.optimizer.learning_rates.denoiser,
        "num_train_images": num_train_images,
        "num_validation_images": num_val_images,
        "num_reg_images": num_reg_images,
        "num_batches_per_epoch": context.num_batches_per_epoch,
        "num_epochs": context.num_train_epochs,
        "gradient_checkpointing": cfg.performance.memory.gradient_checkpointing,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "max_train_steps": cfg.training.max_train_steps,
        "lr_warmup_steps": cfg.optimizer.scheduler.lr_warmup_steps,
        "lr_scheduler": cfg.optimizer.scheduler.lr_scheduler,
        "mixed_precision": cfg.performance.precision.mixed_precision,
        "full_fp16": bool(cfg.performance.precision.full_fp16),
        "is_v2": bool(cfg.model.model_type == "sd2"),
        "base_model_version": context.model_version,
        "clip_skip": cfg.training.clip_skip,
        "max_token_length": cfg.training.max_token_length,
        "cache_latents": bool(cfg.data.caching.cache_latents),
        "seed": cfg.training.seed,
        "lowram": cfg.performance.memory.lowram,
        "noise_offset": cfg.loss.regularization.noise_offset,
        "multires_noise_iterations": cfg.loss.regularization.multires_noise_iterations,
        "multires_noise_discount": cfg.loss.regularization.multires_noise_discount,
        "adaptive_noise_scale": cfg.loss.regularization.adaptive_noise_scale,
        "zero_terminal_snr": cfg.loss.regularization.zero_terminal_snr,
        "sd_scripts_commit_hash": get_git_revision_hash(),
        "optimizer": context.optimizer_name + (f"({context.optimizer_args})" if len(context.optimizer_args) > 0 else ""),
        "max_grad_norm": cfg.optimizer.max_grad_norm,
        "caption_dropout_rate": cfg.data.caption.caption_dropout_rate,
        "caption_dropout_every_n_epochs": cfg.data.caption.caption_dropout_every_n_epochs,
        "caption_tag_dropout_rate": cfg.data.caption.caption_tag_dropout_rate,
        "face_crop_aug_range": cfg.data.preprocessing.face_crop_aug_range,
        "prior_loss_weight": cfg.loss.prior_loss_weight,
        "min_snr_gamma": cfg.loss.snr.min_snr_gamma,
        "ip_noise_gamma": cfg.loss.regularization.ip_noise_gamma,
        "debiased_estimation": bool(cfg.loss.snr.debiased_estimation_loss),
        "noise_offset_random_strength": cfg.loss.regularization.noise_offset_random_strength,
        "ip_noise_gamma_random_strength": cfg.loss.regularization.ip_noise_gamma_random_strength,
        "loss_type": cfg.loss.loss_type,
        "huber_schedule": cfg.loss.huber.huber_schedule,
        "huber_scale": cfg.loss.huber.huber_scale,
        "huber_c": cfg.loss.huber.huber_c,
        "fp8_base": bool(cfg.performance.precision.fp8_base),
        "fp8_base_unet": bool(cfg.performance.precision.fp8_base_unet),
        "validation_seed": cfg.validation.validation_seed,
        "validation_split": float(cfg.validation.validation_split),
        "max_validation_steps": cfg.validation.max_validation_steps,
        "validate_every_n_epochs": cfg.validation.validate_every_n_epochs,
        "validate_every_n_steps": cfg.validation.validate_every_n_steps,
        "run_validation_at_start": cfg.validation.run_at_start,
        "run_validation_at_end": cfg.validation.run_at_end,
        "resize_interpolation": cfg.data.preprocessing.resize_interpolation,
    }

    _append_adapter_metadata(metadata, cfg)
    _append_dataset_metadata(metadata, cfg, total_batch_size=context.total_batch_size)
    _append_model_source_metadata(metadata, cfg)
    return metadata


def _append_adapter_metadata(metadata: dict[str, MetadataValue], cfg: Any) -> None:
    from library.adapters.methods.peft.config_resolution import (
        get_adapter_peft_config,
        get_method_config,
        resolve_adapter_method_registration,
    )

    peft_config = get_adapter_peft_config(cfg)
    if peft_config is None:
        return

    registration = resolve_adapter_method_registration(peft_config)
    _registration, method_config = get_method_config(peft_config)
    metadata_config = getattr(cfg.output, "metadata", None)
    metadata["adapter_method"] = registration.name
    metadata["adapter_rank"] = getattr(method_config, "rank", None)
    metadata["adapter_alpha"] = getattr(method_config, "alpha", None)
    metadata["adapter_neuron_dropout"] = getattr(method_config, "dropout", None)
    metadata["training_comment"] = getattr(metadata_config, "training_comment", None)
    metadata["scale_weight_norms"] = peft_config.scale_weight_norms


def _append_dataset_metadata(
    metadata: dict[str, MetadataValue],
    cfg: Any,
    total_batch_size: int,
) -> None:
    metadata.update(
        {
            "batch_size_per_device": cfg.training.train_batch_size,
            "total_batch_size": total_batch_size,
            "resolution": cfg.data.preprocessing.resolution,
            "color_aug": bool(cfg.data.preprocessing.color_aug),
            "flip_aug": bool(cfg.data.preprocessing.flip_aug),
            "random_crop": bool(cfg.data.preprocessing.random_crop),
            "shuffle_caption": bool(cfg.data.caption.shuffle_caption),
            "enable_bucket": bool(cfg.data.bucketing.enable_bucket),
            "bucket_no_upscale": bool(cfg.data.bucketing.bucket_no_upscale),
            "min_bucket_reso": cfg.data.bucketing.min_bucket_reso,
            "max_bucket_reso": cfg.data.bucketing.max_bucket_reso,
            "keep_tokens": cfg.data.caption.keep_tokens,
        }
    )


def _append_model_source_metadata(metadata: dict[str, MetadataValue], cfg: Any) -> None:
    if cfg.model.pretrained_model_name_or_path is not None:
        sd_model_name = cfg.model.pretrained_model_name_or_path
        if os.path.exists(sd_model_name):
            sd_model_name = os.path.basename(sd_model_name)
        metadata["pretrained_model_name"] = sd_model_name

    if cfg.model.vae is not None:
        vae_name = cfg.model.vae
        if os.path.exists(vae_name):
            vae_name = os.path.basename(vae_name)
        metadata["vae_name"] = vae_name
