# Model Metadata Identity And Structure Research Ledger

## Purpose

This ledger records the evidence used to turn `docs_design/model_metadata_continuation_research.md` into this OpenSpec. It is not a second design authority: the proposal, design, and capability specs contain the accepted change. The longer pre-OpenSpec note retains external links and exploratory alternatives.

Research was refreshed against the working tree on 2026-07-21. Codebase-memory was used for initial architecture/discovery and current source was read directly where branch semantics required source-level verification.

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
- SD and SDXL loaders distinguish local checkpoint versus Diffusers sources, conversions, external VAE replacement, and runtime replacements.
- SDXL attempts variant-specific Diffusers loading and falls back to the default variant.
- SD3 loads MMDiT from one required unified safetensors checkpoint, can source text encoders and VAE from embedded namespaces or local safetensors sidecars, and can omit text encoders. Its current sidecar precedence differs by component.
- No active family currently overrides `load_denoiser_lazily()`. The trainer supports a deferred surface update, but the base strategy hook raises and active SD3 loading requires MMDiT during initial loading.
- These successful loader decisions are not preserved in the existing model metadata facts.

Conclusion: configuration cannot reconstruct truthful provenance. Every successful family loader must populate one common source/selection/component-binding/decision/transformation result schema alongside the authoritative component surface.

## Active Loader Evidence Inventory

### Scope and evidence mapping

This inventory covers the active `train.py` strategy path and the lower-level model loaders it reaches:

- `library/strategies/{sd,sdxl,sd3}/loading.py`;
- `library/models/{sd,sdxl,sd3}/loader.py`;
- checkpoint and VAE helpers directly invoked by those loaders.

It excludes `tools/`, deprecated launchers, the separately deferred textual-inversion path, save-only conversions, tokenizer bootstrap, and adapter state loading. A helper branch is included when it can change the source, selected state, component binding, omission, fallback history, or source-to-runtime interpretation of a family-declared loaded component.

Each successful branch must eventually map to the common evidence categories below. The names are semantic categories for tasks 4 and 5, not a second schema definition.

| Evidence category | Required content |
| --- | --- |
| Source observation | Supplied reference, resolved reference, source kind, relevant local/remote observation, and known limitations. |
| Source representation | Exact local-file evidence, selected directory/repository manifest evidence, or an explicit statement that exact evidence is not yet available. |
| Source selection | Selected variant, subfolder, checkpoint root, state namespace, component subset, or other extraction policy distinct from the representation. |
| Component binding | Family-declared component key bound to the source selection that actually supplied it, including explicit absent/deferred state. |
| Loading decision or attempt | Preferred choice, attempted fallback, successful choice, omission, override, or failure outcome in order. |
| Materialization transformation | Namespace extraction, key remapping, structural conversion, synthesized state, or replacement needed to explain source-to-runtime interpretation. |
| Runtime observation | Device/dtype placement, execution-only module replacement, offload, or other per-run behavior that does not redefine source identity. |
| Limitation | Missing immutable revision, unreported unused keys, unsafe/unbounded source inspection, unretained load diagnostics, or another claim the loader cannot prove. |

### SD 1.x and SD 2.x

The active path is `SdModelLoadingStrategy.load_target_model()` to `library.models.sd.loader.load_target_model()` and `_load_target_model()`.

| Current branch | Current behavior | Evidence required |
| --- | --- | --- |
| Supplied source is a symlink | Resolves the configured reference with `os.path.realpath()` before classifying it. | Preserve supplied reference, resolved target observation, and symlink resolution; identity derives from the resolved representation, not either path string. |
| Resolved source is a file | Treats it as a local Stable Diffusion checkpoint. | Local-file source observation and exact representation evidence; selection describing checkpoint-root state and SD1/SD2 interpretation. |
| Resolved source is not a file | Passes the reference to `StableDiffusionPipeline.from_pretrained()`. It may be a local Diffusers directory or remote repository reference. | Record the source that actually resolves, selected Diffusers layout/components, and immutable revision/file evidence when available. If the loader cannot distinguish a missing local path from a remote name or expose the resolved revision, retain that limitation. |
| Local checkpoint is safetensors | Loads the safetensors root mapping. | Representation format plus selected root mapping. |
| Local checkpoint is pickle/PyTorch | Loads with `weights_only=False`, then selects `checkpoint["state_dict"]` when present or the root mapping otherwise. | Trust/security limitation, container-root selection, and whether a wrapper was removed. Container metadata is not model-state identity. |
| Older text-encoder namespace is present | Rewrites three `cond_stage_model.transformer.*` prefixes to insert `text_model`. | Versioned key-remapping transformation scoped to CLIP-L and the selected checkpoint state. |
| Checkpoint is interpreted as SD1 versus SD2 | Uses configured `model_type`; SD2 also carries the linear-projection interpretation flag. | Selection/materialization policy and family/model-version claim; do not infer this solely from a filename. |
| Checkpoint components materialize | Extracts/converts LDM UNet, VAE, and CLIP namespaces into repository runtime modules. SD1 uses `convert_ldm_clip_checkpoint_v1()` while SD2 uses the separate `convert_ldm_clip_checkpoint_v2()` policy. | One selection or component-specific selections from the same representation, three ordered component bindings, namespace-extraction transformations, and separate explicit conversion-policy identifiers for the SD1 and SD2 LDM-to-runtime conversions. |
| Diffusers pipeline materializes | Uses pipeline text encoder and VAE directly, then rebuilds the Diffusers UNet as the repository's original UNet class from its state dict. | Diffusers component/subfolder selections and bindings; structural Diffusers-UNet-to-repository-UNet conversion. |
| `model.vae` is configured | Intended to replace the primary VAE with the shared external-VAE loader result. The current SD code invokes `load_vae` on the loaded VAE object rather than the module helper, so this branch is not verified as executable and requires focused coverage during task 5.1. | Separate VAE source/selection, explicit replacement decision, VAE component rebinding, transformations from the external VAE branch, and an implementation limitation until corrected and tested. |
| Non-zero VAE padding mode is configured | Mutates Conv2d padding mode after the VAE has materialized. | Component-scoped runtime realization observation with configured mode; it must not create a new source representation identity. |
| Low-RAM loading is enabled | Chooses the accelerator as load device and moves components to it. | Runtime device-placement observation only. |
| RamTorch is enabled | Replaces eligible UNet, VAE, and CLIP-L modules after source loading. | Component-scoped runtime transformation/observation; if a later artifact persists a semantic change, that save boundary decides lineage. |
| Attention backend options are applied | Replaces/configures UNet attention implementation and VAE xformers behavior. | Runtime realization observation, not source provenance. |

There is no active EMA/non-EMA selection branch in this loader. The future source-selection type must support such distinctions, but current SD evidence must not claim an EMA choice that the loader did not make explicitly.

### SDXL

The active path is `SdxlModelLoadingStrategy.load_target_model()` to `library.models.sdxl.loader.load_target_model()` and `_load_target_model()`.

| Current branch | Current behavior | Evidence required |
| --- | --- | --- |
| Supplied source is a symlink | Uses `os.readlink()` once before classification rather than canonical `realpath()` resolution. | Preserve supplied and observed target references plus the exact representation ultimately opened. Record the resolution limitation rather than assuming canonical path semantics. |
| Resolved source is a file | Treats it as a local SDXL checkpoint. | Local-file source observation, exact representation evidence, selected root mapping, and SDXL interpretation. |
| Resolved source is not a file | Passes it to `StableDiffusionXLPipeline.from_pretrained()`. | Successful local-directory or remote-repository source, selected layout, immutable revision/files when available, and limitations when the API result does not expose them. |
| Local checkpoint is safetensors with mmap disabled | Reads the complete file and deserializes it without mmap. | Same representation/selection identity as other access modes; record access method only as loader/runtime evidence. |
| Local checkpoint is safetensors with mmap enabled | Tries device-qualified `load_file()`, then catches any exception and retries the same file without a device. | Ordered loading attempts and selected access outcome; the retry is not a new source identity. Retain that the broad exception is not currently classified. |
| Local checkpoint is pickle/PyTorch | Loads the container, selects `state_dict` when present or root state otherwise, and retains epoch/global-step claims when available. | Trust limitation, root selection, and namespaced source claims for epoch/global step; those claims are not source identity. |
| Checkpoint components materialize | Extracts the UNet, CLIP-L, CLIP-G, and VAE namespaces; converts CLIP-G and VAE state into runtime layouts. | Ordered component bindings to the checkpoint selection, namespace extractions, and versioned SDXL checkpoint-to-runtime conversions. Preserve `logit_scale` and load diagnostics as source/materialization facts or limitations rather than hiding them on the strategy object. |
| Diffusers load uses fp16 weights | First requests `variant="fp16"` when `weight_dtype` is fp16. | Preferred source-selection attempt with requested variant and outcome. |
| fp16 variant raises `OSError` | Retries the same reference with `variant=None`; the retry currently omits the first call's explicit `torch_dtype`. | Failed preferred attempt, fallback decision/reason, successful default-variant selection, and effective dtype evidence/limitation. Configured intent must not be reported as the selected variant. |
| Non-fp16 Diffusers load | Requests only the default variant and propagates failure. | Default-variant selection and outcome; no fictional fallback event. |
| Diffusers text encoders are not fp32 | Casts CLIP-L and CLIP-G to fp32. | Component-scoped runtime dtype observation. |
| Diffusers UNet materializes | Converts Diffusers UNet keys and rebuilds the repository SDXL UNet. | Structural conversion transformation and denoiser component binding to the selected Diffusers UNet state. |
| `model.vae` is configured | Replaces the primary VAE through the shared external-VAE loader. | Separate VAE source/selection, replacement decision, component rebinding, and nested VAE transformations/fallbacks. |
| Non-zero VAE padding mode is configured | Mutates VAE Conv2d padding behavior. | Component-scoped runtime realization observation. |
| Low-RAM, RamTorch, or attention options apply | Moves components or replaces/configures execution modules after source materialization. | Runtime observations only, scoped to affected components. |

`load_stable_diffusion_format`, `logit_scale`, and `ckpt_info` survive today only as mutable strategy attributes for checkpoint saving. The typed loading result must retain the corresponding source/materialization evidence without treating save-oriented convenience state as the canonical provenance schema.

### Shared external VAE loader used by SD/SDXL

`library.models.sd.vae.load_vae()` has its own source and fallback branches which become part of the VAE replacement evidence:

| Current branch | Current behavior | Evidence required |
| --- | --- | --- |
| Reference is a directory or is not a local file | Treats it as a Diffusers local/remote source and first calls `from_pretrained(..., subfolder=None)`. | Supplied/resolved source observation, preferred root selection attempt, and immutable repository/file evidence when obtainable. A nonexistent local-looking path remains ambiguous without stronger resolution evidence. |
| Root Diffusers VAE load raises `OSError` | Retries `from_pretrained(..., subfolder="vae")`. | Ordered failed root attempt, fallback reason, and successful `vae` subfolder selection. |
| Local file ends in `.bin` | Loads a PyTorch state mapping directly into the runtime AutoencoderKL shape. | Local-file representation, pickle trust limitation, selected root state, and direct-layout interpretation. |
| Other local file is safetensors or pickle | Loads the file, unwraps `state_dict` when present, detects whether keys already have `first_stage_model.` prefix, synthesizes that prefix for VAE-only state when absent, then converts LDM VAE keys. | Representation format, root selection, prefix-detection decision, optional prefix-synthesis transformation, and versioned LDM-VAE-to-runtime conversion. |

The external VAE always becomes a separate component source selection and an explicit replacement of the VAE supplied by the primary source. It does not make the primary model representation disappear and does not by itself establish model-revision equivalence.

The LDM VAE converter contains Diffusers-version-dependent attention-key handling. The materialization transformation therefore needs the converter/schema version and relevant implementation/library version; recording only a generic `converted=true` flag would not reproduce the selected mapping. The same rule applies to the SD1/SD2 and SDXL per-key conversion policies: individual key branches belong to one versioned transformation policy rather than becoming unrelated provenance events. Task 4 must assign explicit repository conversion-policy identifiers where the current code has none; a Git commit or library version may accompany that identifier but must not silently substitute for the declared policy version.

### SD3 and SD3.5-family loader

The active path is `Sd3ModelLoadingStrategy.load_target_model()` to `library.models.sd3.loader.load_target_model()` and `_load_target_model()`.

| Current branch | Current behavior | Evidence required |
| --- | --- | --- |
| Primary source | Always passes `pretrained_model_name_or_path` to the safetensors loader; there is no Diffusers/local-directory fallback. | Required local-file source observation and exact representation evidence, or an explicit unsupported-source failure. Do not describe this as a repository source merely because the configured string resembles one. |
| Safetensors access mode | With mmap disabled, uses the custom per-key reader. Otherwise it tries device-qualified `load_file()` and catches any exception before retrying without the device argument. This applies to the unified checkpoint and sidecars. | Ordered access attempts and effective loader/dtype outcome as loading evidence; both access paths retain the same source representation/selection identity. The broad unclassified exception is a limitation. |
| MMDiT materializes | Always extracts `model.diffusion_model.*`, infers `medium`, `5-medium`, or `5-large` from selected tensor structure, and loads with `strict=False`. | Denoiser binding to the unified-checkpoint selection, namespace extraction, versioned model-type derivation evidence, and retained missing/unexpected-key limitations. |
| CLIP-L sidecar path is configured | Skips embedded CLIP-L selection and loads the sidecar safetensors file. | Sidecar source/selection, explicit override decision, and CLIP-L component binding. |
| CLIP-L path is absent and embedded sentinel key exists | Extracts `text_encoders.clip_l.*` from the unified checkpoint. | Unified source selection, namespace extraction, and CLIP-L binding. |
| Neither CLIP-L source is available | Returns `None`. | Explicit absent component binding and omission reason. |
| CLIP-L source lacks `text_projection.weight` | Synthesizes an identity projection tensor before loading. | Versioned synthesized-state materialization transformation and source limitation; the synthesized key is not evidence that it existed in the representation. |
| Embedded CLIP-G sentinel exists | Uses embedded `text_encoders.clip_g.*` even if a sidecar path was configured. | Unified source selection and a decision showing that embedded state won. Preserve configured sidecar intent separately if relevant. |
| Embedded CLIP-G is absent and sidecar path exists | Loads the sidecar safetensors file. | Sidecar fallback selection and CLIP-G binding. |
| Neither CLIP-G source is available | Returns `None`. | Explicit absent component binding and omission reason. |
| Embedded T5 sentinel exists | Uses embedded `text_encoders.t5xxl.*` even if a sidecar path was configured. | Unified source selection and embedded-wins decision. |
| Embedded T5 is absent and sidecar path exists | Loads the sidecar safetensors file. | Sidecar fallback selection and T5 component binding. |
| Neither T5 source is available | Returns `None`. | Explicit absent component binding and omission reason. |
| External VAE path is configured | Loads the sidecar safetensors state instead of the unified VAE namespace. | Sidecar source/selection, explicit override decision, and VAE component rebinding. |
| VAE path is absent | Extracts required `first_stage_model.*` state from the unified checkpoint. | Unified source selection, namespace extraction, and VAE binding. Missing/empty required state must remain a failure/limitation, not an absent optional component. |
| Text modules materialize with `strict=False` | Missing and unexpected keys are logged but not returned. | Per-component materialization result/limitation; successful loading must not imply complete key correspondence. |
| Prefixed state is destructively popped | Selected namespaces are removed from the in-memory root mapping; leftover/unknown keys are not reported. | Source-key selection policy and an explicit unaccounted-key limitation until complete source-key coverage is implemented. |
| FP8, dtype, low-RAM, block swap, scaled positional embeddings, random crop, or padding options apply | Casts/moves modules or alters execution behavior after selecting state. | Component- or realization-scoped runtime observations, not new source/catalog identities. Loaded-FP8 versus cast-to-FP8 may be distinguished as runtime evidence without claiming different source bytes. |
| RamTorch is enabled in the strategy wrapper | Replaces eligible MMDiT, VAE, and present text-encoder modules. | Component-scoped runtime transformation/observation. |

The sidecar precedence above is intentionally asymmetric and is transcribed from the current conditions rather than inferred from a desired uniform policy. `load_clip_l()` inspects embedded state only when no CLIP-L path was supplied, so its path is an override. `load_clip_g()` and `load_t5xxl()` inspect and select embedded state before consulting whether a path was supplied, so their paths are fallbacks used only when the embedded sentinel is absent. `load_vae()` checks the path first, so its path is an override. Task 5 may deliberately change this behavior, but the typed result must report whichever source actually won rather than normalizing the history after the fact.

The current component-sidecar attributes (`clip_l`, `clip_g`, `t5xxl`, `blocks_to_swap`, `enable_scaled_pos_embed`, and `pos_emb_random_crop_rate`) are read through temporary `getattr()` calls but are not declared by `ModelConfig` or the default model YAML. Task 1.3 must decide their typed configuration ownership or explicitly retire unreachable branches before production provenance types depend on them.

There is currently no SD3 denoiser sidecar, omission, or deferred-materialization branch: MMDiT is required during initial load, the public loader asserts it is present, and no SD3 strategy overrides `load_denoiser_lazily()`. The common future schema can represent deferred denoisers, but active-family migration must not fabricate such evidence. Adding that behavior would be a separate functional change rather than a metadata-only migration.

### Cross-family transformation boundary

The inventory yields this initial classification:

- **Source/materialization provenance:** checkpoint-root selection, component namespace extraction, key remapping, Diffusers/checkpoint structural conversion, synthesized CLIP-L projection, source fallback, component omission, and external component replacement.
- **Runtime realization observation:** device placement, ordinary dtype casts, low-RAM placement, block swapping/offload, attention backend replacement, RamTorch replacement, VAE padding mode, scaled positional-embedding setup, and positional random-crop behavior.
- **Persisted lineage/revision evidence:** none of the load-time branches automatically create a descendant revision. If an altered live realization is later saved, the save boundary classifies the persisted transformation and lineage.

This mapping is deliberately family-neutral. SD, SDXL, SD3, and future loaders populate the same ordered source/selection/binding/decision/transformation/limitation collections; family-specific details belong in operation parameters or a namespaced extension, not in separate provenance schemas.

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
