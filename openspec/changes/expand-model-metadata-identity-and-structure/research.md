# Model Metadata Identity And Structure Research Ledger

## Purpose

This ledger records the evidence used to turn `docs_design/model_metadata_continuation_research.md` into this OpenSpec. It is not a second design authority: the proposal, design, and capability specs contain the accepted change. The longer pre-OpenSpec note retains external links and exploratory alternatives.

Research was refreshed against the working tree on 2026-07-20. Codebase-memory was used for initial architecture/discovery and current source was read directly where the graph index lagged recent metadata changes.

## Accepted Starting Point

The archived `centralize-model-family-metadata` and `family-declared-loaded-components` changes already established:

- family-owned ordered top-level component declarations;
- run-qualified model realization and component identities;
- central metadata builders, emitters, registry, validation, records, edges, storage, and projections;
- retained realization state for later artifact/component references;
- compatibility output as projections rather than canonical internal facts.

They explicitly excluded source ingestion, hashes, tensor/module inventories, and parameter-level topology. This continuation therefore extends rather than replaces those layers.

## Current Code Findings

### Persistence and lookup

- `library/metadata/runtime.py` defaults `MetadataRuntime` to `InMemoryMetadataBackend` and exposes filing, buffering, snapshots, and validation. It has no catalog registration or typed query operation.
- `library/metadata/storage.py` provides append-only in-memory and SQLite stores. SQLite schema v1 stores records/events/edges and can survive process restart, but it has no uniqueness/assignment rule for model evidence.
- `library/metadata/backends.py` exposes filtered snapshot reads. `record_for()` selects the last inserted matching record and therefore is not a safe conflict-resolution or composition-revision policy.
- `library/metadata/graph.py` provides identity/relationship lookup over a snapshot but is not a portable identity issuer.

Conclusion: durable storage exists, but stable catalog assignment/resolution does not. The persistence concern raised during discussion is real. It does not require an external service or separate canonical model database; it requires one central behavioral authority over persistent accepted records.

### Realization lifecycle

- `library/training/runners/trainer.py` creates a random run/session ID and files one `ModelRealizationState` after initial `load_target_model()`.
- `library/training/phases/model_prep.py` can later call `load_denoiser_lazily()`, replace the loaded-component surface, and prepare trainables/precision.
- The existing realization facts have no composition revision or finalization state, so the filed `present` values can become stale before training.

Conclusion: finalization semantics are required. Append-only successful composition observations fit the existing store and preserve useful initial/deferred evidence better than delayed-only filing or mutation. Failed materialization is an attempt event, not an observed composition revision.

### Loader provenance

- `ModelLoadingStrategy.load_target_model()` returns only model version plus family-declared components.
- SD and SDXL loaders distinguish local checkpoint versus repository sources, conversions, external VAE replacement, and runtime replacements.
- SDXL attempts variant-specific loading and falls back.
- SD3 can combine a unified checkpoint with independent CLIP-L, CLIP-G, T5, VAE, and deferred denoiser sources.
- These successful loader decisions are not preserved in the existing model metadata facts.

Conclusion: configuration cannot reconstruct truthful provenance. Every successful family loader must populate one common source/selection/component-binding/decision/transformation result schema alongside the authoritative component surface.

### Structural discovery

- `library/models/parameter_dump.py` already enumerates named modules, parameters, and buffers and reports shape/dtype/trainability/persistence, but its canonical product is text-oriented and it has no accepted metadata identity.
- `library/optimization/targets.py` creates useful component/module/parameter live references and component-qualified selectors, but those are execution references rather than durable cross-run metadata facts.
- resource summaries/accounting independently calculate parameter counts and bytes and currently can use bare family-local component owner keys.
- `MemoryEfficientSafeOpen` in `library/utils/safetensors_utils.py` can inspect safetensors keys, header metadata, shapes, dtypes, and offsets without loading every tensor value.
- `calculate_hash` in `library/utils/hash_utils.py` streams file bytes in bounded chunks.

Conclusion: the repo already has primitives worth reusing. It lacks central typed structural descriptors, dimension-scoped coverage/source semantics, distinct path-binding/live-object/storage/source-key identities, resolver/query coordination, and reuse of unchanged accepted facts.

## External Technical Constraints Retained From Pre-OpenSpec Research

- PyTorch named traversal may deduplicate shared objects; the traversal policy must be explicit.
- A module state dictionary normally includes parameters and persistent buffers, not non-persistent buffers. Artifact and live inventories cannot claim identical completeness.
- Object/storage pointers are observation-local. Shared storage can be described within an observation but cannot be a portable tensor identity.
- Safetensors headers expose keys/dtypes/shapes/offsets cheaply, but the format does not preserve every live sharing relationship.
- Pickle-based PyTorch checkpoint inspection has a trust/code-execution boundary that metadata lookup must not bypass.
- Remote model names, branches, and tags are mutable; repository evidence should resolve immutable revisions and selected files.

The external sources and links supporting these points remain in `docs_design/model_metadata_continuation_research.md` so this ledger does not duplicate a second citation list.

## Identity Question Findings

No single identifier answers every model identity question:

- an exact file digest is stable under rename and proves byte identity;
- a directory manifest digest can identify selected repository content independent of absolute path;
- a source-selection identity distinguishes selected state/namespaces such as EMA and non-EMA within one representation;
- a normalized state fingerprint can compare content under one declared normalization, but only within that policy's proven semantics;
- a publisher/repo-carried portable identity can preserve lineage across representations and installations;
- a run realization identifies one execution composition and must change per run;
- an artifact identity identifies one output representation.

The practical catalog approach is therefore progressive:

1. classify and preserve a carried portable claim with its trust evidence;
2. match or fully hash exact serialized content to obtain the source-representation identity;
3. identify the selected state/namespace separately;
4. resolve or create a separate provisional catalog/revision assignment rooted in selection evidence;
5. add structural/state fingerprints when the comparison needs them;
6. treat weak similarities as candidates only.

This satisfies the product requirement that the ID stick with a model rather than being newly random for every user while remaining honest about differently serialized or modified representations.

## Assessment Of Raised Suggestions

### “Stable catalog identity needs a persistence owner”

**Underlying point: required.** Cross-run alias, identity, and lineage mapping cannot live only in the default run-scoped in-memory runtime.

**Suggested form: not automatically accepted.** A dedicated external service or separate canonical registry database is unnecessary for the first slice. The design selects a central catalog authority over persistent accepted metadata records, with derived indexes and SQLite as the first backing.

### “Realization finalization needs explicit semantics”

**Underlying point: required.** Deferred materialization makes the current one-shot presence record stale.

**Suggested versioned composition form: accepted after source verification.** It matches append-only storage, retains initial observations for diagnostics, and allows semantic latest/final views. It is more suitable here than mutating the first record.

### “Exact hashes still risk collapsing identity layers”

**Underlying point: required clarification.** The design already distinguished representation equality from model equality, but “derive a deterministic identity” did not name the target identity layer. Exact file/manifest evidence now derives only a source-representation identity. Source selection and provisional catalog/revision assignment are separate records and relationships.

Adversarial cases retained as required fixtures include one checkpoint with EMA/non-EMA selections, checkpoints whose non-model contents differ, safetensors metadata-only changes, checkpoint/repository conversion, precision/quantization changes, conflicting publisher claims over identical bytes, and realization composition with component overrides.

### “Failure is an attempt rather than composition”

**Underlying point: accepted.** Composition revisions now represent only successfully observed model surfaces (`initial`, `intermediate`, `final`). Started/succeeded/failed materialization attempts are separate events and failures do not consume composition revisions.

### “Completeness and structural identity need more dimensions”

**Underlying point: accepted.** One overall completeness enum would overclaim or make useful inventories uniformly partial. Inventories now carry extensible dimension-scoped coverage claims. Structural metadata separately represents revision/realization path bindings, observation-local live objects, storage groups, and serialized source keys with non-bijective mappings.

### “Carried IDs need trust classification”

**Underlying point: accepted without requiring signatures now.** Carried claims distinguish authenticated issuer evidence, trusted local-producer evidence, unsigned/self-asserted claims, and invalid/unsupported claims. Syntax alone cannot merge logical identities.

### “Split the implementation change”

**Underlying operational concern: acknowledged, not adopted automatically.** One umbrella OpenSpec keeps the cross-layer identity invariants together and already has section reviews plus a hard dependency pause. If that pause becomes long-lived or unreconcilable, the remaining structural/fingerprint sections may be extracted without weakening the completed provenance contract.

### “Be careful with transformations”

**Underlying point: required and already aligned with the narrative.** The design turns it into an invariant: transformations needed to explain/reproduce source-to-logical materialization are provenance; transient execution changes are scoped observations; persisted changes are lineage/revision evidence.

The provided examples inform the taxonomy but do not define an exhaustive universal list.

## Beads And Change Dependencies

- `sd-scripts-8ck`: the provenance/catalog portion of this OpenSpec.
- `sd-scripts-edl`: system-wide typed metadata query capability; a hard prerequisite before structural query integration.
- `sd-scripts-d79`: qualified resource-component identity; a hard prerequisite before resource structural-owner integration.
- `sd-scripts-b25`: structural/tensor catalog work covered by the second half of this OpenSpec after the dependency gate.

The OpenSpec intentionally pauses between those halves. Provenance/catalog work must not create a model-only query system just to make later structural tasks appear unblocked.

## Remaining Decisions Deferred To Bounded Implementation Research

The design intentionally leaves only decisions that require inspecting the implementation boundary or testing normalization claims:

- concrete portable issuer namespace/string encoding;
- the project-consistent default persistent catalog location/configuration;
- which known conversions, if any, qualify for canonical normalized-state equality rather than candidate matching;
- transformation-by-transformation classification as alternate representation, descendant revision, or unresolved relation.

Each is assigned to a numbered implementation milestone before dependent production behavior can land. Until proven, the safe behavior is to preserve separate identities and explicit evidence.
