# Metadata System Inventory

Date: 2026-05-14 (inventory update 2026-07-20)

This note began as the pre-backbone inventory. The original concern survey below
remains useful historical context, but the active authority is
`library/metadata/README.md` plus archived/completed OpenSpec requirements.

Since the original inventory, active model metadata has moved to typed,
relational facts:

- family packages declare ordered top-level component topology;
- central builders create run-qualified realization/component identities;
- strategy checkpoint facets resolve family-specific artifact meaning into
  central typed facts;
- the trainer files those facts through `MetadataRuntime` at model-load and
  checkpoint boundaries;
- central projections own `kuro.*`, `modelspec.*`, model-family `ss_*`, and
  final safetensors stringification.

The old strategy dictionary hooks, `get_model_metadata_from_config(...)`, the
broad `ModelSpecMetadata` / `build_metadata*` construction surface, and the
`ModelSpecFacts` compatibility-dictionary round trip are removed from the
production library. Supported textual-inversion artifact saves use typed family
facts and central projection even though their wider runtime remains
transitional. Genuinely non-active scripts and standalone tools do not define
this library boundary.

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

These meanings remain real and useful. The central metadata system now supplies
their shared recording/relationship/projection vocabulary as each concern is
migrated; it does not collapse their distinct domain ownership.

## Artifact And Model Metadata

Main code:

- `library/metadata/dataclasses/model.py`
- `library/metadata/builders/model.py`
- `library/metadata/projections.py`
- `library/config/dataclasses/output.py::MetadataConfig`
- `configs/_defaults/output/default.yaml`
- `library/constants.py`

Current shape after the model-family migration:

- `MetadataConfig` is the user-facing config surface under `output.metadata`.
  It contains human/model-card style fields such as title, author, description,
  license, tags, thumbnail, trigger phrase, preprocessor, and training comment.
- Active training converts `MetadataConfig` presentation fields into
  `ModelArtifactPresentation` and family-resolved `ModelArtifactFacts`.
- `ModelSpecCompatibilityProjection` represents the active SAI ModelSpec 1.0.1
  export boundary and maps canonical artifact facts to `modelspec.*` keys.
- The legacy `ModelSpecMetadata` / `build_metadata*` and minimum-adapter
  dictionary helpers are removed; active compatibility output comes from
  canonical facts and central projections.
- Thumbnail-file conversion is private model-artifact builder input handling.
- Artifact implementation version reuses the Git revision already recorded in
  canonical run metadata rather than querying a second metadata utility.
- `library/utils/safetensors_utils.py::load_metadata_from_safetensors(...)`
  reads external artifact headers at the safetensors IO boundary.
- `SS_METADATA_MINIMUM_KEYS` in `library/constants.py` is a small legacy list
  used when `output.saving.no_metadata` asks to save only the minimum metadata.

Noted boundaries:

- `MetadataConfig` is model-card/user-authored metadata, not a general runtime
  metadata config.
- Family checkpoint facets accept a narrow typed artifact context and return
  canonical facts; central code does not branch on the family name to render
  ModelSpec output.
- Artifact metadata must be `dict[str, str]` for safetensors compatibility.

### Model structure and tensor capability gap

The completed model-family migration intentionally records family declaration,
run-scoped realization, top-level components, and artifact semantics. Its
archived design made arbitrary module state, parameter names, and module-level
topology non-goals for that slice. Those exclusions describe the completed
migration boundary; they do not mean that granular model structure is outside
the long-term metadata system.

The repository already has partial discovery mechanisms:

- loaded components provide qualified top-level ownership and live modules;
- model inspection can enumerate named modules, parameters, and buffers and
  render shape, dtype, trainability, and buffer persistence;
- optimization grouping creates component-qualified named parameter references;
- startup diagnostics aggregate parameter counts and logical bytes per
  component and estimate gradient/optimizer-state memory.

These paths do not yet form one canonical metadata capability. Tensor details
are used transiently or rendered directly, component memory excludes buffers,
and the metadata backend does not expose a typed single-tensor query. Shared or
tied storage, physical residency, quantization, sharding/offload, and temporal
runtime state also need explicit semantics before their sizes can be compared
or aggregated safely.

The possible query and model-catalog continuation is not yet an accepted system
design. Current-state evidence, candidate boundaries, and open questions are
kept in `docs_design/metadata_query_capability.md`. Beads issues
`sd-scripts-8ck`, `sd-scripts-edl`, and `sd-scripts-b25` track source-model
identity, query research, and the dependent model work; qualified component
identity linkage remains `sd-scripts-d79`.

## Training-Run Metadata

Main code:

- `library/metadata/emitters/run.py`
- `library/metadata/emitters/checkpoint.py`
- `library/metadata/builders/run.py`
- `library/metadata/builders/model.py`
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
  validation settings, configured source-model and VAE names, and objective
  metadata. The active builder does not produce source model or VAE hashes.
- Adapter/PEFT-specific keys are added when an active PEFT config exists:
  `ss_adapter_module`, rank, alpha, dropout, training comment, and scale weight
  norms.
- `build_objective_ss_metadata(...)` adds objective-specific fields, currently
  for RF/SD3-style objective facts.
- `Trainer._initialize_training_metadata()` stores canonical typed run metadata
  state without a family dictionary-mutation bridge.
- `Trainer._file_model_realization_metadata()` files the qualified realization,
  ordered components, and optional family contribution once after model load.
- `Trainer._build_checkpoint_metadata()` files one family-resolved artifact fact
  and projects the explicitly scoped accepted snapshot while preserving the
  current `no_metadata` behavior.
- `scripts/_deprecated/sd_peft.py` and
  `scripts/_deprecated/sdxl_peft_copy.py` are retired stubs with replacement
  `train.py` preset guidance.

Noted boundaries:

- This path is still root-config heavy, but it lives in trainer-level
  orchestration where broad config access is acceptable by current repo rules.
- Configured model and VAE names are run metadata, not stable source identities.
  Local paths are reduced to basenames, and no source-to-realization relation or
  verified cross-run equality is currently recorded.
- Family-specific semantics remain stable at the typed checkpoint facet rather
  than at a generic metadata dictionary hook.
- `ss_*` keys are effectively compatibility artifact metadata, not an internal
  typed schema.

## Strategy Checkpoint Metadata

Main code:

- `library/strategies/sd/checkpointing.py`
- `library/strategies/sdxl/checkpointing.py`
- `library/strategies/sd3/checkpointing.py`
- `library/strategies/base/contracts.py`

Current shape:

- All active families implement `resolve_model_artifact_facts(context)`.
- SD3 additionally implements `ModelFamilyMetadataStrategy` for explicit
  versioned attention-mask facts.
- No active strategy renders `modelspec.*` or mutates checkpoint metadata.

Noted boundaries:

- Strategy checkpointing owns family-specific artifact meaning and save-policy
  behavior. Central metadata owns accepted schemas, qualified identities,
  relationships, and projections.
- Future families extend the same component declaration and typed resolver
  contracts instead of adding a central family switch or a local metadata
  mini-framework.

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

- `tests/unit/metadata/test_model_facts.py` covers model-artifact builder input
  normalization, including thumbnail conversion.
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

## Initial Gaps And Current Status

These were inventory observations. Completed items are recorded here so this
historical document does not contradict the implemented system:

- [Resolved] `library/metadata/` is the central typed catalog/runtime/backend,
  with builders, emitters, relationships, projections, and storage.
- [Resolved for active run/model/checkpoint paths] Safetensors
  `dict[str, str]` is now a final projection boundary rather than the internal
  authoring shape.
- [Partly resolved] `ss_*` and `modelspec.*` are compatibility projections;
  cache and remaining adapter-method concerns still await their own typed slices.
- Dataset/sample metadata is manifest-bound today and not yet a persistent sample
  registry with readiness state.
- [Resolved] Broad flag-driven ModelSpec builders are removed from the
  production library. The supported textual-inversion save boundary now uses
  typed family facts and central projection despite inheriting a deprecated
  wider runtime; only genuinely non-active callers remain outside the boundary.
- [Resolved] Family strategy checkpoint facets own semantic resolution while
  central metadata owns schemas, qualified identities, and projections.
- Older docs already point toward metadata-first dataset structuring and run
  warehouse ideas, but those are not yet combined with the current code surfaces.
