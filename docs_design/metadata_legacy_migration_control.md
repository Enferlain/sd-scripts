# Legacy Metadata Migration Control

Date: 2026-05-15 (updated 2026-07-19)
Status: migration-control record; active training/model-family dictionary seams replaced, including textual-inversion artifact export

Companion notes:

- `docs_design/metadata_system_inventory.md`
- `docs_design/metadata_system_design.md`
- `openspec/changes/archive/2026-05-22-metadata-backbone/`

## Purpose

The first metadata backbone slice added `library/metadata/` plus active
checkpoint metadata routing through provider/projection seams. The first real
replacement slice moved active training-run metadata creation into the
backbone-facing path and deleted the old `library/training/training_metadata.py`
helper instead of keeping it as a wrapper.

The target shape is a single central metadata system, not `metadata.py` files
sprinkled through normal domains. Central recorded dataclasses live under
`library/metadata/dataclasses/`, central builders, emitters, and projections live
under `library/metadata/`, and normal domain code only owns the source objects
plus the lifecycle call sites that hand those facts to the central system.
Local metadata modules remain exception-only, with adapter methods as the clear
current case. Model-family semantics use existing strategy facets returning
central typed facts rather than separate family metadata modules.

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
- **Central builder**: reusable conversion from explicit domain-owned inputs to
  accepted typed items or retained identity state lives in
  `library/metadata/builders/` and does not file or project those items.
- **Central emitter**: registered typed-item-to-record/event/relationship
  conversion lives in `library/metadata/emitters/`.
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
| `library/metadata/builders/run.py` | Assembles typed training-run bundles and retained checkpoint-facing state from explicit runtime inputs. | Central builder. | Keep source conversion separate from emitter routing and artifact projection. |
| `library/metadata/builders/model.py` | Assembles run-qualified model realization/component state and artifact-resolution contexts. | Central builder. | Reuse its qualified identities from artifact/resource/optimization producers; do not reconstruct them per concern. |
| `library/training/runners/trainer.py::_initialize_training_metadata` | Creates `TrainingMetadataState` from canonical run facts. | Domain lifecycle call site. | Keep free of family compatibility mutation hooks. |
| `library/training/runners/trainer.py::_file_model_realization_metadata` | Files realization, ordered components, and optional versioned family contribution once after loading. | Domain lifecycle call site. | Retain the accepted identity state only after filing succeeds. |
| `library/training/runners/trainer.py::_build_checkpoint_metadata` | Resolves and files typed artifact facts, then projects an explicitly scoped accepted snapshot. | Domain lifecycle and artifact boundary. | Keep family semantics in the facet and export spelling in central projections. |
| Strategy `resolve_model_artifact_facts(context)` | Resolves family-owned architecture/reference/objective/artifact-role meaning into shared typed facts. | Family semantic facet. | Extend this contract for future families; do not restore dictionary-returning metadata hooks. |
| Optional `ModelFamilyMetadataStrategy` | Resolves explicit namespaced/versioned family-local realization facts such as SD3 attention-mask settings. | Family semantic facet feeding central schema. | Use only for genuine family-local facts; central runtime/emitter/projection ownership remains unchanged. |
| Removed strategy `update_metadata(...)` / `get_model_metadata(...)` hooks | Former mutation and pre-rendered ModelSpec bridges. | Deleted active helpers. | Do not restore as wrappers. Deprecated/reference scripts do not keep them alive. |
| Removed `library/utils/model_metadata.py::get_model_metadata_from_config` | Former broad config-plus-family-flags ModelSpec facade. | Deleted active helper. | Supported textual-inversion saves now use typed facts/projections. Genuinely non-active callers do not keep the facade alive. |
| Removed `library/utils/model_metadata.py::ModelSpecMetadata`, `build_metadata*`, and related family switches | Former broad flag-driven ModelSpec construction. | Deleted production helpers. | Non-active callers outside the production library do not retain this surface or govern completion of the model-family migration. |
| Removed `library/utils/model_metadata.py::build_minimum_adapter_metadata` | Former minimum-adapter dictionary helper with no production caller. | Deleted production helper. | Canonical run facts and `SsCompatibilityProjection` own active minimum-output behavior. |
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

## Deferred Capability Ledger

This ledger records metadata capabilities whose old implementation was removed
or found in the wrong layer before their final owner was ready. It is not a task
tracker: Beads remains the source of truth for actionable work. A ledger entry
exists so a useful capability can be deliberately placed later without keeping
its old broad helper alive or relying on conversation history.

Use these dispositions:

- **migrated**: represented and tested at its settled owner;
- **retained boundary**: narrow format/IO behavior remains useful where it is;
- **repair now**: required by a supported active path in this change;
- **deferred with owner**: useful capability has an intended architectural owner
  and a Beads issue when the work is actionable;
- **family onboarding obligation**: concrete behavior belongs to a future family
  implementation, not to a central family-name switch;
- **discarded implementation detail**: old API shape has no independent product,
  provenance, or compatibility meaning.

| Capability recovered from the old model utility | Disposition | Intended owner / evidence | Durable follow-up |
| --- | --- | --- | --- |
| ModelSpec required/optional fields, date formatting, resolution, timestep range, encoder layer, presentation fields, and extension fields | Migrated | `ModelArtifactFacts`, `builders/model.py`, family resolvers, and `ModelSpecCompatibilityProjection` | Covered by model dataclass, resolver, projection, and parity tests. |
| SD/SD2/SDXL/SD3 architecture and reference-implementation resolution | Migrated | Existing family checkpointing facets return canonical artifact facts. | Future corrections remain family-owned and projection-independent. |
| Textual-inversion artifact role, architecture suffix, default title, and ModelSpec export | Migrated | Supported SD/SDXL textual-inversion saves use family resolvers plus central ModelSpec/Kuro projection even though the shared implementation file is deprecated. | OpenSpec task 8.7 in `sd-scripts-ao3`; broader runtime migration is `sd-scripts-58e`. |
| Implementation-version lookup, thumbnail data-URL conversion, and safetensors metadata reads | Retained boundary | Narrow repository/version or format-IO helpers in `library/utils/model_metadata.py`; they are not canonical model semantics. | Reassess placement only if another format/storage abstraction needs the same behavior. |
| Artifact hash generation and `hash_sha256` population | Deferred with owner | Artifact lifecycle/evidence concern; the canonical artifact fact already has a field but no producer should invent a value. | `sd-scripts-yzj`. |
| Source checkpoint metadata ingestion | Deferred with owner | Typed source-artifact provenance distinct from loaded realizations and produced artifacts; external `modelspec.*`/`ss_*` remain evidence, not canonical truth. | `sd-scripts-8ck`. |
| Merge-source title lookup and `merged_from` derivation | Deferred with owner | Durable source-to-output artifact relationships first; filename/title fallback belongs only in presentation projection. The configured `merged_from` string remains supported now. | `sd-scripts-yfp`. |
| Minimum adapter facts and serialized adapter constructor/method arguments such as historical `ss_adapter_args` | Deferred with owner | Adapter method/plugin facts feed the shared `SsCompatibilityProjection`; do not restore `build_minimum_adapter_metadata`. | `sd-scripts-r61`. |
| ModelSpec extension-field collision precedence | Deferred with owner | Central ModelSpec projection policy; current apply-last behavior remains compatibility-tested. | `sd-scripts-1vk`. |
| Optional UNet/VAE dtype claims | Deferred with owner | Artifact/runtime precision provenance can populate the existing artifact fields once save-time semantics and consumers are explicit. | Leave in this ledger until a precision/artifact slice supplies evidence; do not infer from configured intent. |
| Flux, Chroma, Lumina, Hunyuan Image, and other future-family architecture/implementation identifiers | Family onboarding obligation | The corresponding family strategy owns concrete semantic resolution when that family becomes supported; central metadata accepts the shared facts without naming the family. | Create family-specific work when the model implementation is in scope; do not preserve the old central switch. |
| `ModelSpecMetadata`, `BASE_METADATA`, generic `build_metadata*`, and boolean family-selection signatures | Discarded implementation detail | Their useful field semantics are represented elsewhere; the compatibility-shaped container and broad dispatch API are not capabilities. | None. Do not restore as a convenience facade. |

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

   Status: recorded dataclasses now cover run metadata, checkpoint artifacts,
   qualified model realizations/components, versioned family contributions, and
   artifact-facing model facts. The transitional `ModelSpecFacts` dictionary
   wrapper was removed. A generic adapter fact dataclass was intentionally not
   kept because adapter metadata is plugin-like.

3. Convert emitters/provider wrappers to consume typed facts.

   `TrainingRunMetadataProvider`, `ModelSpecMetadataProvider`, and
   `CheckpointArtifactMetadataProvider` should stop treating old dictionaries as
   their authoring shape once typed facts exist. Normal metadata assembly should
   move toward central emitters. Adapter method providers should be added from
   plugin-like adapter method implementations instead of using a generic central
   adapter fact class.

   Status: registered emitters consume typed run/model/artifact fact dataclasses.
   Checkpoint projection consumes an accepted, explicitly scoped snapshot;
   `ModelSpecCompatibilityProjection` no longer recovers meaning from prefixed
   legacy records. Legacy `ss_adapter_*` keys remain compatibility metadata only
   until adapter-owned facts exist.

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

   Status: SD1/SD2, SDXL DDPM/RF, SD3, adapter/full-model, attention-mask,
   extension-collision, and current no-metadata behavior have focused typed
   projection coverage. Frozen historical dictionaries remain comparison
   evidence without executing the deleted broad helper.

6. Remove or replace old helpers in the next domain slice.

   Prefer deleting active helpers after their callers move to the
   backbone-facing API. Use a wrapper only when there is a deliberate public
   compatibility contract beyond deprecated/reference scripts.

   Status: active model-family `get_model_metadata(...)` and
   `update_metadata(...)` hooks and the transitional `ModelSpecFacts` emitter
   round trip are removed. The broad `ModelSpecMetadata` / `build_metadata*`
   surface is removed, while supported textual-inversion saves are being moved
   from its final broad-helper call to typed family facts and central projection.

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

## Current Follow-Up Boundary

The model-family implementation now includes supported textual-inversion
artifact saves; final verification/archive remains open. Remaining metadata
concerns should stay separate and are classified in the capability ledger
above. A supported launcher is active even if it delegates to a deprecated
implementation file; genuinely non-active reference scripts and standalone
tools remain outside this ownership boundary and do not require production
compatibility wrappers unless the repo deliberately promotes one as a supported
public facade.
