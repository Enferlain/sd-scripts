# Metadata System Inventory

Date: 2026-05-14

This note is phase 1 of the metadata-system work: establish what already exists.
It intentionally avoids designing the future backbone. Later passes should decide
coverage areas, gather older notes/resources, and combine this inventory with the
desired architecture.

## Current Metadata Meanings

The repo currently uses "metadata" for several distinct concepts:

- Artifact metadata: string key/value metadata embedded in checkpoint or adapter
  artifacts, especially safetensors.
- Model-spec metadata: `modelspec.*` keys for SAI Model Spec compatibility.
- Training-run metadata: `ss_*` keys describing the run, dataset, optimizer,
  objective, and source model.
- Dataset/sample metadata: captions, tags, dimensions, repeat/split state, cache
  paths, and bucket assignments used to build manifests and epochs.
- Cache metadata: safetensors metadata used to validate latent, text-encoder, and
  epoch-token caches.
- Runtime orchestration metadata: optimizer/scheduler capability and grouping
  facts that guide training behavior without becoming artifact metadata.
- Observability metadata: logged artifact kind/format metadata and run/report
  context for logging sinks.
- Method-local adapter metadata: adapter-specific persistence facts such as VeRA
  projection reconstruction settings.

These meanings are real and useful, but they do not yet share one vocabulary,
schema boundary, or ownership model.

## Artifact And Model Metadata

Main code:

- `library/utils/model_metadata.py`
- `library/config/dataclasses/output.py::MetadataConfig`
- `configs/_defaults/output/default.yaml`
- `library/constants.py`

Current shape:

- `MetadataConfig` is the user-facing config surface under `output.metadata`.
  It contains human/model-card style fields such as title, author, description,
  license, tags, thumbnail, trigger phrase, preprocessor, and training comment.
- `ModelSpecMetadata` represents SAI Model Spec 1.0.1 and serializes to
  `modelspec.*` keys.
- `get_model_metadata_from_config(...)` builds `modelspec.*` metadata from
  `MetadataConfig` plus explicit family/runtime inputs such as SDXL/v2 flags,
  LoRA/textual-inversion flags, resolution, timestep range, clip skip, prediction
  type, and optional model-family type strings.
- `build_minimum_adapter_metadata(...)` still exists for minimal adapter metadata,
  using the legacy `ss_*` minimum key set.
- `load_metadata_from_safetensors(...)` reads artifact metadata back from
  safetensors.
- `SS_METADATA_MINIMUM_KEYS` in `library/constants.py` is a small legacy list
  used when `output.saving.no_metadata` asks to save only the minimum metadata.

Noted boundaries:

- `MetadataConfig` is model-card/user-authored metadata, not a general runtime
  metadata config.
- `get_model_metadata_from_config(...)` is broad and family-flag driven. Strategy
  checkpointing code currently narrows this by calling it from family-specific
  wrappers.
- Artifact metadata must be `dict[str, str]` for safetensors compatibility.

## Training-Run Metadata

Main code:

- `library/metadata/emitters/run.py`
- `library/metadata/emitters/checkpoint.py`
- `library/training/metadata.py` compatibility wrapper
- `library/training/runners/trainer.py`
- `library/strategies/*/checkpointing.py`

Current shape:

- `build_training_metadata_bundle(...)` builds typed full/minimum
  `RunMetadataFacts` through the central metadata emitter path. Inputs include
  root config, training/validation manifests, session timing, model version,
  optimizer name/args, epoch/batch counts, total batch size, and objective
  definition.
- The produced metadata includes run facts, optimizer/scheduler facts, precision
  flags, dataset counts, bucket/dataset/tag summaries, augmentation settings,
  validation settings, source model and VAE names/hashes, and objective metadata.
- Adapter/PEFT-specific keys are added when an active PEFT config exists:
  `ss_adapter_module`, rank, alpha, dropout, training comment, and scale weight
  norms.
- `build_objective_ss_metadata(...)` adds objective-specific fields, currently
  for RF/SD3-style objective facts.
- `Trainer._initialize_training_metadata()` stores typed full/minimum metadata
  state, then calls `strategies.update_metadata(...)` through the remaining
  strategy compatibility bridge.
- `Trainer._build_checkpoint_metadata()` chooses full-vs-minimum facts based on
  `cfg.output.saving.no_metadata`, then routes run/model/artifact facts through
  central checkpoint emitters and projections.
- `scripts/_deprecated/sd_peft.py` and
  `scripts/_deprecated/sdxl_peft_copy.py` are retired stubs with replacement
  `train.py` preset guidance.

Noted boundaries:

- This path is still root-config heavy, but it lives in trainer-level
  orchestration where broad config access is acceptable by current repo rules.
- The strategy hook is stable according to `docs_design/strategy_contract_pressure_audit.md`.
- `ss_*` keys are effectively compatibility artifact metadata, not an internal
  typed schema.

## Strategy Checkpoint Metadata

Main code:

- `library/strategies/sd/checkpointing.py`
- `library/strategies/sdxl/checkpointing.py`
- `library/strategies/sd3/checkpointing.py`
- `library/strategies/base/contracts.py`

Current shape:

- Strategies implement `update_metadata(metadata, cfg)` and
  `get_model_metadata(cfg)`.
- SD and SDXL wrap `get_model_metadata_from_config(...)` with family-specific
  flags, prediction-type resolution, resolution, timestep range, and clip skip.
- SD3 uses a distinct model metadata shape through the same strategy seam.

Noted boundaries:

- Strategy checkpointing owns family-specific model identity and save-policy
  details today.
- Older design notes suggest a possible future Stable-Diffusion-family metadata
  helper, but deferred it because SD3 and save behavior diverge.

## Safetensors Metadata IO

Main code:

- `library/utils/safetensors_utils.py`
- Direct `safetensors.safe_open(...).metadata()` / `safetensors.torch.save_file(..., metadata=...)`
  usage in data and adapter modules.

Current shape:

- `MemoryEfficientSafeOpen.metadata()` returns `header["__metadata__"]` or `{}`.
- The custom save helper validates/writes safetensors metadata under
  `__metadata__`.
- Many paths still call safetensors directly where they need simple read/write
  behavior.

Noted boundaries:

- Safetensors metadata is a storage format boundary. It should stay string-only
  and should not become the internal shape for richer metadata.

## Dataset And Sample Metadata

Main code:

- `library/data/structures.py`
- `library/data/scanners.py`
- `library/data/manifest.py`
- `docs_design/DATA_PIPELINE_CURRENT.md`
- `docs_design/future_ideas/data_accounting.md`
- `docs_design/future_ideas/async_data.md`
- `docs_design/future_ideas/data_shards.md`

Current shape:

- `CacheEntry` is the current atomic sample/manifest record. It carries sample
  identity, source path, original size, bucket resolution, resized size, caption,
  tags, repeats, regularization flag, split, latent/TE cache paths, optional
  in-memory TE outputs, and cache augmentation flags.
- `Bucket` carries bucket resolution and image IDs.
- `BatchInfo` and epoch manifests carry batch-level metadata for training views.
- `scan_metadata_file(...)` loads FineTuning-style JSON metadata:
  `{image_key: {"caption": "...", "tags": "...", "train_resolution": [w, h]}}`.
  Captions prefer `caption`, then `tags`, and missing dimensions are read from
  the image file.
- Current docs already identify metadata-first structuring as a future direction:
  discover sample identity, dimensions, captions, split/repeat/source state, and
  bucket assignment before payload readiness.

Noted boundaries:

- The current manifest already contains useful structural metadata, but it is not
  a long-lived mutable sample registry.
- Future notes repeatedly point toward a persistent sample registry with readiness
  state as the scalable metadata store for async/large-scale pipelines.
- Dataset metadata and artifact metadata are different domains even though both
  currently use string-ish dictionaries in places.

## Cache Metadata

Main code:

- `library/data/caching_engine.py`
- `library/data/epoch_preparation.py`
- model-family cache strategy implementations
- `library/data/dataloader.py`

Current shape:

- `CacheBackend.is_cache_valid(...)` is the model-specific validation seam for
  cache files. It is expected to validate required tensor keys, shapes, optional
  flipped/alpha data, and stored metadata against the manifest entry.
- `build_vae_cache_signature(...)` produces a stable JSON signature for VAE cache
  invalidation from VAE/source path, size, mtime, and padding mode.
- `tokenize_epoch_manifest(...)` writes epoch token safetensors metadata:
  epoch, seed, sample count, batch count, max token length, encoder names, and
  a `manifest_hash`.
- `TrainingDataset._validate_token_metadata(...)` warns on manifest-hash mismatch
  and errors on sample-count mismatch.

Noted boundaries:

- Cache metadata currently validates payload compatibility; it is not a complete
  provenance record.
- Future sharded-cache and async-preparation notes will likely need stronger
  namespace/hash metadata and readiness metadata.

## Optimizer And Scheduler Runtime Metadata

Main code:

- `library/metadata/dataclasses/optimization.py`
- `library/optimization/types.py`
- `library/optimization/registry.py`
- `library/optimization/optimizer_utils.py`
- `library/optimization/scheduler.py`
- `library/optimization/grouping.py`
- `library/optimization/targets.py`

Current shape:

- `ParameterGroup.metadata` stores planning/debug metadata such as `param_names`,
  intentionally separate from runtime optimizer options.
- `ParameterGroup.to_optimizer_dict(...)` only includes selected safe metadata
  keys when explicitly requested.
- `LogicalParameterGroup` stores trainer-facing group identity independently of
  execution group layout.
- `SchedulerRuntimeMetadata` declares scheduler ownership mode/target:
  external/embedded/none and optimizer/base_optimizer.
- `OptimizerRuntimeMetadata` currently declares whether train/eval toggling is
  supported.
- `library/optimization/types.py` keeps compatibility aliases to the central
  optimizer runtime fact dataclasses.
- `OptimizationTargetRef.metadata` carries target-selection side data; builder
  helpers copy metadata defensively.
- Optimizer/scheduler registries are explicitly capability-metadata oriented.

Noted boundaries:

- This is good precedent for typed runtime metadata that does not leak into
  artifact metadata or optimizer payloads by default.
- Runtime capability metadata should remain typed and owned by the optimization
  runtime; any recorded/exported metadata view should be emitted through the
  central metadata system when that slice exists.

## Adapter Metadata

Main code:

- `library/adapters/shared/state_io.py`
- `library/adapters/runtime/targets.py`
- `library/adapters/shared/trainables.py`
- `library/adapters/shared/reporting.py`
- `library/adapters/methods/peft/*/state_dict.py`
- `library/adapters/methods/peft/vera/state_dict.py`

Current shape:

- Adapter export/save requests accept optional `dict[str, str]` metadata and pass
  it through method runtimes into exported artifacts.
- Adapter checkpoint-state hooks save/load `train_state.json` separately for
  accelerator checkpoint resume.
- `AdapterResolvedTarget.metadata` currently records `component_key` for adapter
  target provenance.
- Repo-owned adapter trainable refs carry provenance used by optimizer grouping,
  diagnostics, and benchmark reports.
- Most PEFT methods pass artifact metadata through unchanged.
- VeRA adds method-local safetensors metadata:
  `sd_scripts_vera.save_projection` and
  `sd_scripts_vera.projection_prng_key`, then uses it on reload to reconstruct
  shared projection behavior without sentinel tensors.

Noted boundaries:

- Adapter artifact metadata is mostly a passthrough from the trainer-level
  checkpoint metadata builder, except for method-local additions like VeRA.
- Method-local metadata keys should stay namespaced and owned by the method.

## Observability And Report Metadata

Main code:

- `library/logging/metrics.py`
- `library/logging/reports.py`
- `library/logging/resource_monitor.py`

Current shape:

- `TrainingObserver` now has lifecycle calls, metrics logging, console logging,
  startup summary logging, artifact registration, and finish semantics.
- `LoggedArtifact` stores `path`, `kind`, and arbitrary artifact metadata.
- Benchmark reports are registered as artifacts with kind/format metadata after
  they are written.
- `RunReportContext` is the report payload boundary for benchmark report
  construction.
- `BasicResourceMonitor` normalizes run/config/git metadata values for JSONL and
  phase/resource events.

Noted boundaries:

- Observability metadata is run/report/logging metadata, not artifact checkpoint
  metadata.
- Recent changes have already moved this toward explicit typed events and context
  objects.

## Checkpoint Resume Metadata

Main code:

- `library/training/checkpointing.py`
- `library/adapters/shared/state_io.py`

Current shape:

- `save_train_state_metadata(...)` writes `train_state.json` with current epoch
  and step next to accelerator checkpoint state.
- `load_train_state_metadata(...)` restores epoch/step into `ResumeState` and
  mutable current epoch/step holders.
- Adapter checkpoint hooks use these helpers while narrowing accelerator state to
  the adapter runtime.

Noted boundaries:

- This is resume-state metadata, not artifact descriptive metadata.
- The filename and JSON shape are currently tiny and specific.

## Existing Tests

Representative coverage:

- `tests/unit/utils/test_utils_sai_model_spec.py` covers `ModelSpecMetadata` and
  model-spec construction.
- `tests/unit/training/test_training_metadata.py` covers objective metadata.
- `tests/unit/training/test_training_checkpointing.py` covers minimum adapter
  metadata.
- `tests/integration/test_checkpoint_io.py` covers safetensors metadata
  round-trip and train-state hooks.
- `tests/unit/data/test_scanners.py` covers metadata JSON scanning.
- `tests/unit/data/test_sdxl_cache_roundtrip.py` covers cache metadata
  preservation and cache invalidation behavior.
- `tests/unit/data/test_epoch_preparation.py` covers epoch-token metadata.
- `tests/unit/training/test_training_optimizer.py` covers optimizer/scheduler
  runtime metadata and parameter-group metadata isolation.
- `tests/unit/logging/test_step_logging.py` covers logged artifact metadata.

## Initial Gaps To Keep In View

These are inventory observations, not proposed solutions:

- There is no central metadata package or typed root vocabulary.
- `dict[str, str]` is used both as a storage requirement and as an internal
  convenience shape; those concerns are not always separated.
- `ss_*`, `modelspec.*`, cache metadata, logging artifact metadata, and runtime
  capability metadata are all separate key spaces.
- Dataset/sample metadata is manifest-bound today and not yet a persistent sample
  registry with readiness state.
- Some metadata builders are broad and flag-driven, especially model-spec
  construction.
- Family strategy hooks are stable but may not be the final metadata ownership
  boundary.
- Older docs already point toward metadata-first dataset structuring and run
  warehouse ideas, but those are not yet combined with the current code surfaces.
