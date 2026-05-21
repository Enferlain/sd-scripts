# Legacy Metadata Migration Control

Date: 2026-05-15
Status: migration-control note after first metadata backbone slice; active training builder replacement implemented

Companion notes:

- `docs_design/metadata_system_inventory.md`
- `docs_design/metadata_system_design.md`
- `openspec/changes/metadata-backbone/`

## Purpose

The first metadata backbone slice added `library/metadata/` plus active
checkpoint metadata routing through provider/projection seams. The first real
replacement slice moved active training-run metadata creation into the
backbone-facing path and deleted the old `library/training/training_metadata.py`
helper instead of keeping it as a wrapper.

The target shape is a single central metadata system, not `metadata.py` files
sprinkled through normal domains. Central recorded dataclasses live under
`library/metadata/dataclasses/`, central emitters/providers and projections live
under `library/metadata/`, and normal domain code only owns the source objects
plus the lifecycle call sites that hand those facts to the central system.
Local metadata modules remain exception-only, with adapter methods as the clear
current case and model families only if we explicitly bless them later.

The main rule is:

> Replace one concern at a time, and delete the old active helper for that
> concern once its behavior is represented by typed facts/providers/projections
> and compatibility tests prove the exported shape.

Compatibility is about exported artifact keys (`ss_*`, `modelspec.*`, method
keys), not about keeping old internal helper modules alive.

## Migration Classifications

Use these labels while migrating each surface:

- **Deleted active helper**: the old active helper is removed after the active
  caller and tests use the backbone-facing API.
- **Deprecated-only legacy**: stale imports remain only in deprecated/reference
  scripts and do not keep active helper modules alive.
- **Central recorded schema**: metadata dataclasses recorded by the repo backend
  live under `library/metadata/dataclasses/<concern>.py`, following the config
  dataclass catalog pattern.
- **Central emitter/builder**: normal metadata assembly code lives under
  `library/metadata/emitters/` or another central metadata module. It accepts
  explicit domain-owned inputs and emits central recorded schemas.
- **Domain source ownership**: domain code owns the runtime objects and
  lifecycle call sites that decide when metadata should be emitted. It does not
  grow local metadata helper modules by default.
- **Plugin-like local exception**: local metadata schemas are allowed only for
  named plugin-like areas, currently adapter methods. Model families may become
  an exception only after an explicit design decision.
- **Projection-owned export**: string keys such as `ss_*`, `modelspec.*`, and
  `kuro.*` are emitted only by projections.
- **Storage / IO boundary**: code reads or writes an external format and should
  stay format-focused, not become the internal metadata model.
- **Deferred domain slice**: important metadata surface that should wait for a
  dedicated follow-up such as data/cache, observability, optimizer, or SQLite.

## Current Surface Map

| Surface | Current role | Migration classification | Near-term action |
| --- | --- | --- | --- |
| `library/training/metadata.py::build_training_metadata_bundle` | Gathers config, manifest, optimizer/runtime, objective, and source-model facts for training-run metadata. | Transitional centralization seam. | Move toward `library/metadata/emitters/run.py` or `checkpoint.py`. Recorded dataclasses stay in `library/metadata/dataclasses/`; do not let training remain a metadata schema or helper island. |
| `library/training/metadata.py` provider wrappers | Provider wrappers and checkpoint projection helper for active checkpoint artifacts. | Transitional assembly plus projection orchestration. | Keep together only until central emitters exist. Do not recreate `metadata_providers.py` or use this as a pattern for every domain folder. |
| `library/training/metadata.py::build_objective_ss_metadata` | Adds RF/objective-specific exported compatibility facts. | Transitional compatibility assembly. | Keep in the active builder for now; move stable recorded objective schemas into `library/metadata/dataclasses/` and compatibility mapping into projections/emitters when objective metadata gets its own slice. |
| `library/training/runners/trainer.py::_initialize_training_metadata` | Creates `TrainingMetadataState` from typed full/minimum `RunMetadataFacts` and lets strategy family hooks add current family-specific compatibility facts. | Domain lifecycle call site with one remaining strategy dict bridge. | Keep as the active call site. Replace `update_metadata(metadata, cfg)` with family hooks/providers in a later family metadata slice. |
| `library/training/runners/trainer.py::_build_checkpoint_metadata` | Now routes active checkpoint metadata through the backbone. | Domain lifecycle call site. | Keep as the first active integration point. Avoid adding new direct metadata dict composition here. |
| Strategy `update_metadata(metadata, cfg)` hooks | Family-specific mutation bridge for current compatibility facts; currently SD3 attention-mask keys use this. | Family hook; possible plugin/facet discussion later. | Keep hook only until family metadata hooks/providers exist. If model families are not made plugin-like, stable recorded model-family schemas should live centrally. |
| Strategy `get_model_metadata(cfg)` hooks | Family-specific wrapper around `get_model_metadata_from_config(...)`. | Family hook now; central recorded model facts later. | Keep as the active model-family narrowing seam. Later return or feed central typed model facts instead of raw `modelspec.*` dicts unless model families are explicitly made plugin-like. |
| `library/utils/model_metadata.py::ModelSpecMetadata` | Dataclass for SAI Model Spec export keys. | Projection-owned export, possible compatibility dataclass. | Keep as compatibility implementation for `modelspec.*`. Later move or wrap under metadata projection ownership when call sites are ready. |
| `library/utils/model_metadata.py::get_model_metadata_from_config` | Broad flag-driven builder for `modelspec.*`. | Compatibility wrapper and projection implementation detail. | Keep until SD/SDXL/SD3 family providers supply typed facts. Add explicit tests before changing output behavior. |
| `library/utils/model_metadata.py::build_metadata*` | Legacy ModelSpec construction helpers used by tools/tests. | Compatibility wrapper. | Do not delete until tool and test consumers migrate. Prefer wrapping projection logic rather than broadening the helper. |
| `library/utils/model_metadata.py::build_minimum_adapter_metadata` | Builds legacy minimum adapter metadata from constants. | Compatibility wrapper; projection-owned export. | Keep until `SsCompatibilityProjection` owns all minimum adapter use cases and legacy tests are redirected. |
| `library/utils/model_metadata.py::load_metadata_from_safetensors` | Reads artifact metadata from safetensors files. | Storage / IO boundary. | Keep format-focused. It may later move beside safetensors projection helpers, but should not become an internal metadata source of truth. |
| `library/constants.py::SS_METADATA_*` and `SS_METADATA_MINIMUM_KEYS` | Legacy `ss_*` key constants. | Projection-owned export keys. | `library.metadata.keys` owns these now. Keep compatibility re-exports in `library.constants` until downstream imports settle. |
| `library/utils/safetensors_utils.py::MemoryEfficientSafeOpen.metadata` | Reads raw safetensors `__metadata__`. | Storage / IO boundary. | Leave in IO utility. Metadata backbone should project into this boundary, not replace low-level safetensors readers immediately. |
| Direct `safetensors.save_file(..., metadata=...)` calls | External artifact metadata writes. | Storage / IO boundary. | Route artifact-producing training paths through projections first. Domain-local direct writes can stay until their central emitters/projections exist. |
| Adapter export metadata passthrough | Adapter runtime receives `dict[str, str]` metadata from trainer. | Storage / IO boundary plus plugin-like adapter method facts. | Keep passthrough for compatibility. Add method/provider facts where adapter methods need durable reconstruction or compatibility facts. |
| VeRA `sd_scripts_vera.*` metadata | Method-local reconstruction facts for projection-bank behavior. | Plugin-like local exception plus method-owned projection. | Keep method-local keys and parsing. Later add a VeRA producer/projection that emits the same keys from typed local method facts. |
| Adapter target/trainable/reporting provenance | Runtime/debug metadata for optimizer grouping and diagnostics. | Plugin-like local exception, deferred adapter slice. | Do not force into checkpoint artifact metadata. Later expose typed adapter target/trainable facts for reports, compatibility, and diagnostics. |
| Dataset manifest metadata (`CacheEntry`, `Bucket`, `BatchInfo`) | Current sample/view/cache-path structural state. | Deferred data/cache emitter slice. | Keep manifest-owned source objects for now. Future `sd-scripts-ucs` should define central typed sample/view/readiness/cache recorded schemas and central emitters fed by data call sites. |
| Cache validation metadata and epoch token metadata | Payload compatibility and token-file validation facts. | Deferred data/cache domain slice plus storage / IO boundary. | Keep validation local for now. Later emit typed cache namespace, payload, readiness, and invalidation facts. |
| Optimizer/scheduler runtime metadata | Typed capability and planning facts, not artifact metadata. | Central recorded optimizer runtime schema slice. | Keep runtime ownership local. The runtime fact dataclasses now live in `library/metadata/dataclasses/optimization.py`; later add central emitter/provider output without leaking all of it into checkpoint metadata. |
| Logging/report/resource metadata | Observability events, reports, artifact registration, resource facts. | Deferred observability emitter slice. | Keep logging presentation/sink boundaries. Later add central metadata events/artifact records that logging can consume. |
| Checkpoint resume `train_state.json` | Accelerator checkpoint resume state. | Storage / IO boundary, not artifact descriptive metadata. | Keep separate from artifact metadata. Later optionally describe state artifacts, but do not merge resume-state JSON into safetensors metadata. |

## Immediate Migration Sequence

1. Add central compatibility-key ownership.

   Move `SS_METADATA_*` and `SS_METADATA_MINIMUM_KEYS` toward
   `library/metadata/keys.py`, leaving compatibility re-exports from
   `library/constants.py` until downstream imports settle.

   Status: implemented. `library.metadata.keys` owns the legacy `ss_*` key
   constants and `library.constants` keeps compatibility re-exports.

2. Add typed recorded metadata dataclasses.

   Introduce `library/metadata/dataclasses/` as the central recorded-schema
   catalog, similar to `library/config/dataclasses/`. Adapter method metadata is
   the current explicit plugin-like exception and should not become a generic
   central adapter schema.

   Status: initial recorded metadata dataclasses now exist for run metadata,
   checkpoint artifacts, and model-spec-compatible facts. A generic adapter fact
   dataclass was intentionally not kept because adapter metadata is plugin-like.

3. Convert emitters/provider wrappers to consume typed facts.

   `TrainingRunMetadataProvider`, `ModelSpecMetadataProvider`, and
   `CheckpointArtifactMetadataProvider` should stop treating old dictionaries as
   their authoring shape once typed facts exist. Normal metadata assembly should
   move toward central emitters. Adapter method providers should be added from
   plugin-like adapter method implementations instead of using a generic central
   adapter fact class.

   Status: checkpoint provider wrappers now consume typed run/model/artifact fact
   dataclasses. `build_checkpoint_metadata(...)` accepts typed facts directly.
   Legacy `ss_adapter_*` keys remain compatibility metadata only until
   adapter-owned providers exist.

4. Replace the active training builder.

   Move active training-run metadata construction to
   a backbone-facing API, store active full/minimum facts in
   `TrainingMetadataState`, and delete `library/training/training_metadata.py`
   instead of preserving it as a wrapper. The first landing spot may be
   transitional; the target pattern is central metadata emitters plus local
   trainer call sites, not `metadata.py` per domain.

   Status: implemented. Active tests and integration references now use
   `build_training_metadata_bundle(...)`. The deprecated SD and SDXL PEFT script
   copies are fenced with clear replacement guidance instead of importing removed
   training metadata helpers.

5. Add parity tests before shrinking old helpers in other domains.

   Tests should lock:

   - `ss_*` full metadata output for representative adapter checkpoint saves.
   - `ss_*` minimum/no-metadata output.
   - `modelspec.*` output for SD, SDXL, and SD3 paths.
   - SD3 family-specific attention-mask `ss_*` fields.
   - VeRA `sd_scripts_vera.*` reconstruction metadata.

6. Remove or replace old helpers in the next domain slice.

   Prefer deleting active helpers after their callers move to the
   backbone-facing API. Use a wrapper only when there is a deliberate public
   compatibility contract beyond deprecated/reference scripts.

## What Not To Do Yet

- Do not restore `create_training_metadata(...)` as a compatibility wrapper for
  active code.
- Do not move all model-family implementation details into central
  `library/metadata/`; only centralize normal recorded schemas, emitters, and
  projections unless model families are explicitly treated as facets.
- Do not treat safetensors `dict[str, str]` as the internal metadata model.
- Do not fold data/cache, observability, optimizer, and resume-state metadata
  into checkpoint artifact metadata just because the word "metadata" appears.
- Do not create `metadata.py` in every normal domain package as a convenience
  layer.
- Do not remove exported `ss_*` or `modelspec.*` compatibility keys until
  exported parity is tested.

## Proposed Next Implementation Bead

The next implementation bead should stay narrow and replace one remaining
metadata concern end-to-end:

> Move SD/SDXL/SD3 model-family metadata facts out of raw `modelspec.*` dict
> builders and strategy mutation hooks into central recorded model facts plus
> explicit family hooks/providers, while keeping existing exported
> `modelspec.*` and family `ss_*` metadata identical.

Acceptance should require focused parity tests. Deprecated/reference scripts do
not require compatibility wrappers unless the repo deliberately promotes one as
a supported public facade.
