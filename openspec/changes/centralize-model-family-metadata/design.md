## Context

The metadata backbone already provides typed accepted items, one registry and routing table, validation, records/events/edges, in-memory and SQLite backends, graph indexes, query views, and projections. Observability and resource intelligence use that path in live runtime code. Model metadata only partially uses it.

The current model path has three transitional seams:

1. `CheckpointingStrategy.get_model_metadata(cfg)` returns an already-rendered `dict[str, str]` of SAI ModelSpec keys.
2. `ModelSpecFacts.from_modelspec_metadata(...)` reconstructs a small typed wrapper from that export dictionary while retaining the whole dictionary as `compatibility_metadata`.
3. `CheckpointingStrategy.update_metadata(metadata, cfg)` mutates training metadata for family-specific fields such as the SD3 attention-mask settings.

The checkpoint emitter then stores those pre-rendered keys on a model record and `ModelSpecCompatibilityProjection` merely selects facts whose names already start with `modelspec.`. This makes the projection nominal rather than semantic: external key ownership still lives upstream in strategies and `library/utils/model_metadata.py`.

The completed `family-declared-loaded-components` change establishes a better source boundary. Each family declares ordered `LoadedModelComponentSpec` values with a stable family-local key, public name, roles, and capabilities; runtime binds those declarations to live module objects. Generic code already consumes the loaded-component collection instead of reconstructing the old `text_encoders/vae/denoiser` tuple.

Repository research also exposed an identity problem. Current resource facts can identify a component using a bare key such as `vae` or `denoiser`. Those keys intentionally repeat across SD, SDXL, SD3, and future families. A bare key is useful inside one realization but is not a durable cross-run graph identity.

External constraints remain deliberately narrow:

- SAI ModelSpec 1.0.1 defines `modelspec.*` keys in a safetensors header and distinguishes required, recommended, and optional artifact metadata.
- Safetensors allows a free-form string-to-string `__metadata__` map; arbitrary typed JSON values are not accepted at that boundary.
- Existing SD/SDXL/SD3 checkpoint and adapter outputs have compatibility value and must retain their current projected fields while the internal path changes, except where a historical implementation identifier names the wrong family reference codebase.

The loaded-component OpenSpec is archived into the base capabilities, so this design consumes that contract directly. The completed step-metric change remains a separate concern: a typed runtime metric event is not automatically a durable metadata fact.

## Goals / Non-Goals

**Goals:**

- Make repo-owned typed facts, rather than `modelspec.*` or `ss_*` dictionaries, authoritative for model metadata.
- Distinguish static family declaration, run-specific loaded realization, and artifact-specific metadata.
- Define durable, qualified model and component identities suitable for cross-run and cross-family snapshots.
- Preserve family ownership of model meaning while centralizing accepted schemas, routing, emitters, relationships, validation, and projections.
- Reuse family-declared component order and semantics without persisting live module objects or creating a second declaration registry.
- Preserve SD/SDXL/SD3 ModelSpec and family-specific `ss_*` output parity, including adapter versus full-model artifact differences, while explicitly correcting inaccurate reference-implementation claims.
- Leave a clean relational source for later resource, optimization, adapter, artifact-lineage, and analytics work.

**Non-Goals:**

- Redesign model loading, checkpoint tensor serialization, or the loaded-component contract.
- Persist arbitrary module state, parameters, tensor names, or module-level topology.
- Redesign optimizer grouping, adapter targeting, or resource accounting.
- Absorb adapter-method-local reconstruction metadata such as VeRA fields into a generic model-family schema.
- Define a durable step-metric warehouse or treat tracker payload dictionaries as metadata facts.
- Change the product semantics of `output.saving.no_metadata`; that remains tracked separately.
- Add new ModelSpec fields, change SAI ModelSpec versions, or calculate tensor hashes as part of this migration.
- Preserve internal helper APIs solely for deprecated/reference scripts.

## Decisions

### Decision: Model metadata has declaration, realization, and artifact layers

The model domain will expose three related layers rather than one overloaded metadata dictionary:

- **Family declaration** is static family-owned information: family/model type, supported top-level component declarations, family-local component keys, public labels, roles, capabilities, and declaration order.
- **Model realization** is observed run truth: which model family/version was loaded for a run and which declared components were present. It may reference runtime-derived facts but never stores the live module object.
- **Artifact model facts** describe one output boundary: artifact role such as adapter or full model, architecture/implementation identifiers, presentation fields, resolution, prediction/timestep semantics, and explicitly supported extension fields needed for ModelSpec projection.

This split prevents artifact format details from becoming the model runtime identity. It also allows one loaded realization to produce multiple artifact kinds whose compatibility metadata legitimately differs.

Alternative considered: expand the existing `ModelSpecFacts` wrapper while continuing to use it for runtime and artifact state. Rejected because ModelSpec describes a distributed file, not the complete runtime model topology, and its architecture value changes for adapter versus full-model outputs.

### Decision: Families resolve meaning; the metadata package owns accepted shapes

Shared dataclasses live under `library/metadata/dataclasses/model.py`, and central emitters convert them to records and edges. Family checkpointing/model facets resolve family-specific values such as architecture identifiers, implementation identifiers, model version interpretation, and SD3-only settings from narrow typed inputs.

This follows the model-layer ownership split:

- `library/models/<family>/__init__.py` remains the small declarative home for `LOADED_MODEL_COMPONENT_SPECS` and does not absorb metadata workflow logic;
- `library/models/components.py` continues to own shared loaded-component resolution/filtering helpers and does not become a trainer or checkpoint metadata builder;
- `library/strategies/<family>/checkpointing.py` owns artifact/checkpoint behavior and resolves the family-specific typed facts needed at that workflow boundary;
- `library/metadata/` owns the accepted durable shapes and their routing, validation, graph, and external projections.

Family facets return typed facts. They do not render `modelspec.*`, mutate training dictionaries, call metadata backends, or own a second metadata registry. A separate family-local `metadata.py` mini-framework is not introduced for SD/SDXL/SD3.

This settles the earlier central-versus-family question as:

- central schema and transport ownership;
- family-owned semantic resolution at the existing family facet;
- central projection ownership for external keys.

Adapter methods remain the explicit plugin-like exception and participate through their own later provider integration.

Alternative considered: move all architecture selection logic into a central flag-driven builder. Rejected because it recreates the current broad helper and forces the central package to branch over every family.

Alternative considered: place checkpoint/export fact resolution in `library/models/components.py` because it starts from component declarations. Rejected because the model-layer contract explicitly keeps training/checkpoint workflow behavior in strategies and keeps the shared component module free of orchestration.

### Decision: Future families extend contracts instead of central branches

A future model family joins the common metadata path by:

1. declaring its ordered top-level components through the loaded-component contract;
2. implementing its strategy-owned typed resolver for model/artifact semantics;
3. returning the shared realization, component, and artifact fact types;
4. using the same registry routes, emitters, relationships, and field-driven compatibility projections.

Central emitters and ModelSpec projection do not branch on family names. Architecture and implementation identifiers are data resolved by the family facet. Shared semantics use shared typed fields. A genuinely family-local fact either stays local when no durable consumer exists or uses an explicit namespaced/versioned family contribution; it is not promoted into a universal field or inferred from component names merely to avoid adding a new schema.

If a future family needs an external format other than ModelSpec, that format receives a separate projection over accepted canonical facts. The internal model-realization contract remains unchanged.

Alternative considered: add each future family to a central architecture/implementation switch. Rejected because it would recreate the broad flag-driven builder and make central metadata a registry of model-family behavior.

Alternative considered: allow an unrestricted compatibility dictionary on the shared model fact. Rejected because it would make export keys authoritative again and bypass typed/versioned extension decisions.

### Decision: Loaded-component declarations are the only component inventory

Model-realization facts are projected from the family's `LoadedModelComponentSpec`/`LoadedModelComponent` collection. Metadata preserves:

- declaration order;
- stable family-local `key`;
- `public_name` as display/public provenance;
- roles and capabilities;
- observed presence in the realization.

The live `module` value is never accepted as metadata. Metadata does not infer components from trainer convenience properties, role positions, parameter names, or hardcoded SD-family names. Multiple components sharing a role remain separate.

The initial slice records top-level components only. Module- and parameter-level expansion remains owned by targeting and optimization provenance.

The conversion boundary reads the public loaded-component contract; it does not move or duplicate declarations into metadata dataclasses. Pure loaded-component lookup/filtering remains in `library/models/components.py`, while constructing accepted metadata items occurs at the runtime lifecycle call site or in a narrow central conversion helper that accepts already-resolved declaration values.

Alternative considered: let each metadata consumer reconstruct a component list from trainer slots. Rejected because it would create a second, diffusion-shaped source of truth immediately after removing that contract.

### Decision: Component identities are qualified by model realization

A family-local component key is not globally unique. Durable identities therefore use a qualified structure equivalent to:

```text
run/<run-id>/model/<model-realization-id>
run/<run-id>/model/<model-realization-id>/component/<component-key>
```

The exact serialization can use existing metadata namespace/identifier fields, but it must retain all of these semantic scopes:

- owning run;
- owning model realization;
- family-local component key.

The declared key remains a queryable fact and the public label remains separate. Component records link to the owning realization with `contained_in`; the realization links to the run with a run-scope relationship; produced artifacts link to the realization with an explicit provenance edge when both identities are available.

Missing scope identity causes the edge or filing operation to be omitted/rejected according to item validation. Placeholder run, model, or component identities are not synthesized.

Alternative considered: use bare `component_key` as the record identifier and rely on query views to supply run context. Rejected because snapshots and SQLite storage span runs, making collisions and ambiguous evidence unavoidable.

### Decision: Runtime filing uses the existing accepted-item registry

Low-volume model realization and component facts are filed when model loading has completed and stable run/model identities are available. Accepted item types and emitter routes are added to `library/metadata/registry.py`, which remains the only supported-type/routing catalog.

Artifact-facing facts are filed or assembled at the checkpoint/artifact lifecycle boundary because artifact role and output metadata can differ from the loaded realization. Checkpoint projection consumes accepted facts/snapshots, not a strategy-rendered dictionary.

No model-specific provider object or parallel collector hierarchy is introduced. `MetadataRuntime.file(...)` remains the public runtime operation; central emitters remain internal routing targets.

Alternative considered: keep checkpoint metadata as an isolated in-memory builder path. Rejected as the end-state because it prevents model/component facts from participating in resource, optimization, artifact-lineage, and analytics queries. A bounded in-memory snapshot may still be used inside projection tests or as a migration bridge.

### Decision: Compatibility keys are created only by projections

Canonical records use repo-owned semantic names such as `architecture`, `implementation`, `prediction_type`, `component_key`, and `roles`. They do not store duplicate `modelspec.architecture` or `ss_*` facts as the source of truth.

`ModelSpecCompatibilityProjection` becomes a real mapping from typed artifact/model facts to SAI ModelSpec keys. Family-specific Kohya compatibility fields are mapped by the appropriate central compatibility projection from accepted family facts. `SafetensorsMetadataProjection` remains the final stringification boundary.

Projection behavior must preserve current active outputs for:

- SD1/SD2 adapter artifacts;
- SDXL adapter artifacts;
- SDXL full-model safetensors checkpoints;
- SD3 full-model artifacts;
- DDPM epsilon/v prediction values and rectified-flow omission behavior;
- resolution, timestep range, clip-skip/encoder-layer, title, author, description, license, tags, date, and current extension fields;
- SD3 attention-mask compatibility fields;
- the existing `no_metadata` behavior until its dedicated product decision changes it.

One compatibility difference is intentional. The canonical `implementation` fact identifies the stable family reference codebase described by ModelSpec rather than inheriting the old broad helper's mixed family/serializer heuristics:

- SD1: `https://github.com/CompVis/stable-diffusion`;
- SD2: `https://github.com/Stability-AI/stablediffusion`;
- SDXL: `https://github.com/Stability-AI/generative-models`;
- SD3 and SD3.5: `https://github.com/Stability-AI/sd3.5`.

The SD3.5 repository is an inference-only reference implementation for SD3/3.5, not a complete training stack. It is still the applicable public architecture reference and must be described with that limitation in documentation. Artifact role and serializer do not replace these family reference identities with an unrelated repository. Frozen pre-migration fixtures retain the old values as baseline evidence; typed projection tests assert the corrected values.

The projection owns the external prefix, key spelling, omission rules, and string encoding. Internal consumers use canonical fields.

Alternative considered: retain prefixed keys on records and continue filtering them in projection code. Rejected because it leaves rendering ownership in family/runtime code and gives internal analytics two names for the same fact.

### Decision: Artifact role is explicit

Family fact resolution receives a small explicit artifact context rather than booleans such as `is_sdxl`, `is_v2`, `is_lora`, `is_textual_inversion`, and `is_stable_diffusion_ckpt` spread across a generic builder.

At minimum, the context distinguishes:

- the source model family/version;
- output artifact role/kind;
- serialization format where it affects ModelSpec implementation semantics;
- objective prediction semantics;
- user-authored presentation metadata;
- resolution/timestep/encoder-layer facts already owned by their domains.

The context remains an input to family resolution, not a new persistent god object. Helpers accept the narrowest typed pieces they need.

Alternative considered: preserve the broad boolean signature and merely change its return type. Rejected because the signature itself encodes family detection and artifact behavior in a central compatibility helper.

### Decision: Required facts validate before projection

Item validation checks identity and field-local constraints before accepted facts enter the backend. Projection validation checks the facts required for the selected external claim.

When emitting SAI ModelSpec metadata, the projection stamps the ModelSpec version it implements and must have canonical architecture, implementation, and title facts, plus resolution for current image-model artifacts where applicable. The specification identifier is projection-owned format/version information, not a runtime producer fact. Optional omission behavior remains family/artifact-specific and parity-tested. The safetensors boundary stringifies values only after semantic validation.

Component validation requires non-empty qualified ownership, unique family-local keys within one realization, stable declaration order, and serializable roles/capabilities. It rejects live module objects and duplicate qualified component identities.

Alternative considered: let safetensors saving fail on malformed dictionaries. Rejected because that loses semantic error context and permits incomplete records into durable storage.

### Decision: Compatibility parity is captured before deleting helpers

Focused tests first capture the existing exported dictionaries for representative SD, SDXL, and SD3 configurations and for adapter/full-model distinctions. Those fixtures remain unchanged as historical baseline evidence. The implementation then moves fact production behind typed family resolution and central projections; comparison tests require parity except for the explicitly corrected family reference-implementation identifiers.

Once active callers no longer use `get_model_metadata_from_config`, `CheckpointingStrategy.get_model_metadata`, or `CheckpointingStrategy.update_metadata`, those active seams are removed rather than retained as wrappers. Format-focused safetensors IO may remain, and deprecated/reference scripts do not justify preserving active helper APIs.

Alternative considered: retain old methods as adapters indefinitely. Rejected because the repository's migration policy is to remove replaced internal compatibility layers after external parity is proven.

## Risks / Trade-offs

- [The change can blur model identity, source identity, and output artifact identity] -> Keep declaration, realization, and artifact facts separate and require explicit relationships between them.
- [Qualified identities may diverge from current resource component identifiers] -> Introduce one shared identity constructor and migrate resource/optimization producers to reference it rather than inventing concern-local qualification.
- [Moving ModelSpec logic can subtly change omission/default behavior] -> Capture parity fixtures before implementation and compare complete dictionaries for every active family/artifact path, with the family reference-implementation correction called out as an intentional field-level difference.
- [Family facets can become metadata mini-frameworks] -> Limit them to typed semantic resolution; keep schemas, emitters, validation, registry, records, and projections central.
- [Persisting component capabilities can freeze an immature vocabulary] -> Version realization facts, preserve declared values exactly, and treat vocabulary evolution as declaration schema evolution rather than universal semantics.
- [User extension fields can collide with standard ModelSpec keys] -> Preserve the current observable policy during this migration, add focused collision coverage, and make any policy change a separate explicit compatibility decision.
- [Filing every realization/component adds low-volume records] -> File once at model-load completion; do not buffer, poll, or repeat per step.
- [The implementation may drift from the archived component contract] -> Validate against the base `loaded-model-components` capability and reuse its declarations rather than copying component topology into metadata.

## Migration Plan

1. Use the archived loaded-component capability as the baseline and preserve its family-declared topology contract.
2. Capture complete SD/SDXL/SD3 historical output fixtures, including adapter/full-model differences, RF prediction omission, SD3 attention-mask fields, extension fields, current `no_metadata` behavior, and the old implementation identifiers that the typed path intentionally corrects.
3. Define central declaration/realization/component/artifact fact dataclasses, qualified identity helpers, validation, registry routes, and emitters.
4. Add family-owned typed resolvers to SD, SDXL, and SD3 checkpointing/model facets.
5. File model-realization/component facts at the post-load lifecycle boundary and link them to the run.
6. Rewrite ModelSpec and family `ss_*` projections to map canonical facts into external keys.
7. Switch checkpoint/adapter/full-model export paths to consume accepted facts and projection snapshots.
8. Migrate resource component references to the shared qualified identity where cross-run snapshots require it, without changing resource meaning or accounting semantics.
9. Remove replaced active dictionary hooks/builders, update metadata ownership docs, changelog, and roadmap, and close the migration bead after parity and focused verification pass.

Rollback is repository-internal: before archive/merge, revert the change as one unit. No persisted production schema migration is required for the initial branch work; if development SQLite stores contain pre-change component identities, they may be recreated rather than ambiguously upgraded.

## Open Questions

- Should the model-realization identifier be supplied by the trainer, derived from the run plus model source/version, or generated by a central identity helper? It must be deterministic within a run and collision-safe across stored runs.
- Which existing relationship verb best expresses realization-to-run ownership, or should the shared graph vocabulary add an explicit `realized_in` relationship?
- Does the first slice migrate current resource component identities immediately, or introduce the shared constructor and migrate resource references in a tightly coupled follow-up before cross-run component queries are exposed?
- Which current arbitrary ModelSpec extension fields are relied on by active configs, and what exact collision precedence must the parity fixtures preserve?
