## Context

The archived `centralize-model-family-metadata` change established canonical family declarations, run-qualified realizations, top-level realized components, artifact facts, central builders/emitters, and projections. It deliberately did not identify the source model, record loader decisions, hash model state, or describe structure below top-level components.

Current source confirms four boundaries this continuation must address:

- `ModelLoadingStrategy.load_target_model()` returns only a model-version string and loaded components. SD, SDXL, and SD3 loaders know whether they selected a local checkpoint, a Diffusers repository, a fallback variant, an external VAE/text encoder, or a conversion, but that evidence is discarded before metadata filing.
- `Trainer._file_model_realization_metadata()` files one realization immediately after initial loading. A later preparation phase can materialize a deferred denoiser and replace the loaded-component surface without filing a revised composition.
- `MetadataRuntime` uses an in-memory backend by default. `SQLiteMetadataStore` is durable and append-only, but it does not assign catalog identities or enforce evidence-to-identity mappings. `MetadataSnapshot.record_for()` merely returns the most recently inserted matching record.
- Structural discovery already exists in `library/models/parameter_dump.py`, optimization target expansion, resource summaries, and safetensors header inspection, but those paths produce strings, live references, or concern-local summaries rather than canonical structural descriptors.

The detailed pre-OpenSpec research is retained in `docs_design/model_metadata_continuation_research.md`. The system-wide query research is retained separately in `docs_design/metadata_query_capability.md`. This change consumes the future shared typed query capability; it does not make model metadata its owner.

The design must also respect these product decisions:

- identities and portable evidence are not personal-machine concepts;
- an unseen model receives strong identity evidence rather than only a path-derived local ID;
- in-memory training transformations do not automatically create a new source model;
- saved modifications retain lineage instead of becoming unrelated objects;
- persistent and non-persistent buffers and unknown keys must be representable when an inventory claims the applicable complete scope;
- expensive detail is available on demand or by policy, not always collected.

## Goals / Non-Goals

**Goals:**

- Recognize and catalog source models across runs, machines, renamed files, and carried repository metadata without collapsing distinct identity claims.
- Give portable catalog identity one durable assignment/resolution boundary backed by accepted metadata records.
- Preserve source, component materialization, transformation, realization, output, revision, and lineage provenance.
- Represent evolving realization composition as immutable ordered observations with explicit finalization.
- Provide canonical component identities, revision/realization structural path bindings, observation-scoped object/storage evidence, source-key mappings, and descriptors at selectable granularity.
- Support inexpensive recorded lookups and bounded on-demand resolution through the shared query system.
- Make completeness, source limitations, evidence strength, derivation version, cost, and freshness explicit.
- Reuse accepted structural identities in optimization and resource facts.
- Establish safe, testable groundwork for increasingly format-independent state fingerprints.

**Non-Goals:**

- Prove arbitrary models functionally equivalent.
- Guarantee that differently quantized, converted, pruned, merged, or fine-tuned models share one identity automatically.
- Persist raw tensor contents as metadata or inspect every tensor on every run.
- Treat device placement, allocator residency, ordinary offload, or transient autocast as source-model provenance.
- Build publishing, federation, or a hosted global catalog service in this change.
- Replace the system-wide typed query design with a model-specific query API.
- Redesign checkpoint serialization, optimizer grouping policy, adapter targeting policy, or resource sampling.
- Migrate `tools/` or the textual-inversion runtime.

## Decisions

### Decision: Keep catalog, source, realization, and artifact identity distinct

The system will represent at least these identity layers:

1. **Catalog model/lineage identity**: portable logical lineage identity that can remain attached to a model and its descendants.
2. **Model revision identity**: one immutable selected logical model state within that lineage. Release names/version labels are claims attached to a revision, not the revision identity itself.
3. **Source representation identity**: one concrete serialized container such as a file, selected directory snapshot, or resolved repository representation.
4. **Source selection identity**: the selected state/component namespace extracted from a representation, such as EMA versus non-EMA weights or one component within a multi-object checkpoint.
5. **Run realization identity**: the run-scoped composition materialized for execution, potentially from several source selections.
6. **Produced artifact identity**: one serialized output.
7. **Structural binding and observation identities**: qualified revision or realization paths plus observation-scoped live-object/storage evidence.

An exact file digest proves representation equality, not source-selection, revision, catalog-lineage, or general model equality. One checkpoint representation may contain several selectable states. A canonical state fingerprint may prove equality under one declared normalization policy. A lineage edge states derivation or descent; it does not prove equal behavior. A realization describes execution composition and may combine selections from more than one source representation, catalog model, or revision.

Alternative considered: use one file hash as the model ID. Rejected because serialization order, container metadata, representation format, component layout, and precision can change without expressing the same kind of identity claim.

Alternative considered: use configured names/paths as identity. Rejected because names are aliases, paths are local, and remote references can move unless resolved.

### Decision: Portable identity has carried and deterministic forms

When a source carries a repo-owned portable identity and lineage declaration, the catalog preserves that claim with its issuer/schema/trust evidence. Produced Kuro artifacts embed their assigned portable identity so another installation can submit the same claim rather than necessarily minting a different identity for the same published model.

A carried identity is structurally valid only when its namespace/issuer, identity kind, scheme version, and bounded identifier payload satisfy the registered schema for that issuer. Structural validation does not by itself prove issuer authenticity. Carried claims preserve one of these trust classes: authenticated issuer evidence, trusted local-producer evidence, unsigned/self-asserted portable claim, or invalid/unsupported claim. An unsigned claim can match a previously accepted claim plus consistent exact evidence but cannot independently merge unrelated representations. An invalid or unsupported claim is retained only as unverified namespaced source evidence and cannot select, create, redirect, or merge a catalog identity.

When an unseen external source has no accepted portable identity, the resolver derives a versioned deterministic **source-representation identity** from the strongest exact source evidence available:

- a local file uses a streaming cryptographic digest of the complete file;
- a local directory uses a canonical, versioned manifest over selected relative paths and complete-file digests;
- a remote repository uses its resolved immutable revision plus the selected files and verified content digests once materialized;
- a live-state-only source has no serialized representation identity and may instead provide selected-state fingerprint evidence only when that policy is implemented and its coverage claims satisfy the request.

The representation identifier records its identity scheme/version. Same exact representation evidence produces the same representation identity on another installation. The selected state receives a separate source-selection identity derived from the representation plus the versioned selection/extraction description. When no accepted catalog model/revision exists, the authority may create a provisional catalog/revision assignment rooted in that selection evidence, but the provisional identities, their provisional status, and the assignment relationship remain separate records. Representation equality never becomes revision or catalog equality implicitly.

A later publisher-carried identity, conversion assertion, or stronger fingerprint uses a precisely typed relationship: observation/display alias, identity redirect, `represents`, `derived_from`, or `equivalent_under_policy`. Existing representation identities are never rewritten. Two publishers may make conflicting logical-lineage claims about the same content-addressed representation; the catalog preserves both claims and resolves or flags their conflict independently from the proven byte equality.

Weak evidence such as a display name, filename, mutable repository branch, architecture label, parameter count, or structural similarity can return candidates for review or later resolution but cannot automatically merge catalog entries.

Alternative considered: assign a random local UUID to every unseen input. Rejected because it cannot stick to the model across independent installations unless subsequently embedded, and it makes identical imported sources diverge unnecessarily.

Alternative considered: delay cataloging unknown models until a canonical cross-format fingerprint exists. Rejected because exact representation evidence is already sufficient to create a useful, honest catalog entry.

### Decision: A central catalog authority owns durable registration

The central metadata subsystem will expose a narrow catalog authority responsible for:

- resolving recognition evidence under a versioned policy;
- assigning or accepting portable identifiers;
- registering immutable identity, alias, revision, and lineage facts;
- detecting conflicting strong claims;
- returning the accepted identity and evidence used.

Accepted metadata records and relationships are the canonical durable facts. A persistent lookup index over those records is an implementation aid, not a second source of truth. The existing SQLite store is the first durable backing and must support reopening in a new process and resolving the same strong evidence to the same catalog identity. In-memory storage remains valid for isolated tests and explicitly ephemeral workflows but cannot satisfy cross-run catalog guarantees.

Ephemeral mode requires an explicit constructor/configuration choice and is observable in catalog results. Failure to open or migrate the configured persistent catalog is an error; the active path must not silently fall back to an ephemeral catalog and issue identities with weaker durability.

`MetadataRuntime` remains the run filing API and does not independently invent persistent catalog identities. Model-loading/building code obtains a catalog resolution from the long-lived authority, then files run-scoped realization facts referencing it. The authority belongs in central metadata code; implementation work must inspect neighboring packages before choosing its final module path and must not add a generic root file or domain-local `metadata.py` by reflex.

This is a first-class service boundary, not necessarily a separate process, external service, or giant model-specific database.

Alternative considered: let each run-scoped runtime assign IDs. Rejected because the default runtime is ephemeral and cannot arbitrate evidence across runs.

Alternative considered: make a mutable registry table the canonical truth. Rejected because accepted append-only records and their provenance must remain auditable and portable.

### Decision: Recognition is progressive, versioned, and non-destructive

Recognition proceeds from cheapest trustworthy evidence toward more expensive evidence:

1. validate a carried portable identity claim and classify its issuer/schema/trust evidence;
2. match a previously recorded exact source representation using an immutable repository revision or exact artifact/manifest digest;
3. if the representation is unseen, calculate the required full digest and register the representation identity;
4. resolve the versioned source selection and any separate provisional catalog/revision assignment;
5. calculate a declared structural or canonical state fingerprint only when requested by policy or needed to compare selected states;
6. retain weak matches as candidates without merging.

“Unseen” means that no accepted mapping under the active recognition-policy version matches the available strong evidence. The decision and policy version are recorded so rules can evolve. Mappings are append-only; superseded claims use explicit status/redirect records. Conflicting strong claims fail resolution or produce a conflict record rather than selecting the newest insertion silently.

A strong-evidence conflict exists when the same normalized immutable evidence key under one policy version maps to different source-representation identities, or when one carried logical identity makes claims incompatible with its already accepted exact/selection evidence or lineage kind. Registration is transactional: concurrent registration of the same representation evidence and target is idempotent, while concurrent different targets produce a uniqueness conflict and explicit conflict evidence. Source-selection and composition revision allocation likewise enforce transactional uniqueness for their owner-qualified ordinal/key. Derived indexes are checked against canonical records when opened or rebuilt and can be discarded and rebuilt on inconsistency.

Automatic representation reuse is permitted only for exact deterministic equality under an accepted strong-evidence policy. An authenticated or trusted carried identity consistent with accepted evidence may reuse a logical identity; an unsigned claim alone may not. Display/path aliases, identity redirects, representation-of relationships, and equivalence-under-policy assertions remain different relationship types rather than a generic alias edge. Weak candidates never redirect or establish equivalence. Manual conflict repair is an explicit catalog operation that appends actor, reason, policy, and resolution/supersession evidence; it never edits or deletes the conflicting history. This change does not build a user interface for repair.

Hashing streams bytes and does not require loading the whole model into RAM, but it still reads the complete selected content. The resolver exposes cost before performing optional expensive work and caches accepted immutable results keyed by strong evidence. Cache reuse requires matching algorithm, policy version, source evidence, and coverage scope. A path, size, modification time, inode/file ID, or similar stat tuple is cache-validation evidence rather than cryptographic identity; the resolver must rehash when it lacks accepted immutable evidence proving the current content matches the cached result.

The directory-manifest policy is versioned and deterministic. Its selected logical paths use `/` separators, Unicode normalization, and case-sensitive comparison independent of the host filesystem; duplicates after normalization are rejected. Selected symlink entries are resolved to regular-file content under an explicit policy and retain symlink observation evidence without including machine-local target paths in identity. Special files are rejected. Files are checked for mutation while hashing, and a changed file or selected-file set invalidates the attempt rather than producing a mixed-time manifest.

### Decision: Loading returns a typed result with source evidence

The model-loading contract will return a typed result rather than a tuple. It contains:

- model version and family-declared loaded components;
- a common ordered collection of configured/resolved source observations;
- a common ordered collection of source selections, including immutable revision/variant/subfolder and selected-state/namespace evidence when applicable;
- exact representation artifact or manifest evidence when available;
- a common ordered collection of component-to-selection bindings;
- a common ordered collection of typed loading decisions and transformations covering fallback outcomes, omissions, external replacements, and conversions;
- catalog resolution or enough evidence for the catalog boundary to resolve it;
- explicit evidence limitations.

Family loaders own truthful source resolution because they know which path succeeded. Central builders convert the typed result into accepted catalog/provenance/realization facts. Live module objects remain in the loaded-component surface and are never stored as metadata.

All families produce this same loading-result schema. SD and SDXL populate it with their selected local-checkpoint or Diffusers sources, external VAE bindings, conversions, and fallback decisions. SD3 populates the same collections with its unified checkpoint plus any independently selected CLIP-L, CLIP-G, T5, VAE, or deferred denoiser sources. A future family may declare different component topology and namespaced operation details, but it does not define a family-specific provenance shape or require a central family-name branch.

Alternative considered: reconstruct provenance from config after loading. Rejected because config expresses intent, not the successful source, fallback, conversion, or component composition.

### Decision: Transformations are classified by what a consumer must reproduce

A transformation belongs to **source/materialization provenance** when a consumer needs it to understand how supplied source evidence became the logical realization/component interpretation. This includes namespace extraction, checkpoint-to-repo conversion, component omission, external replacement, source fallback, sharding assembly, and structural conversion.

A change belongs to **runtime realization observation** when it changes how the same realization executes at a phase/time but does not redefine its source: device placement, ordinary `.to()` casts, offload, allocator residency, temporary autocast, and training-only wrappers.

A saved change belongs to **model lineage/revision provenance** when it creates a reusable descendant or alternate representation: training output, merge, pruning, quantization, conversion, component replacement, or other persisted modification. The relationship type and transformation evidence state whether it is a descendant revision, alternate representation, or unresolved relation; it is not automatically a wholly unrelated model.

The boundary is: *would a consumer need this fact to explain or reproduce how the supplied source became this logical component/revision?* If yes, it is provenance. If it only describes transient execution, it remains a scoped runtime observation. “Not source provenance” never means “must not be recorded.”

### Decision: Realization composition is append-only and explicitly finalized

One stable run-scoped realization identity owns an ordered sequence of immutable composition observations. Each observation has:

- a monotonically increasing revision;
- lifecycle state (`initial`, `intermediate`, or `final`);
- observed component identities/presence/component-to-selection bindings in family declaration order;
- the lifecycle boundary and timestamp/order evidence;
- links to the preceding revision and applicable source/materialization facts.

Initial loading files revision 1. A successfully observed deferred materialization or component replacement files another composition revision. Once model preparation has completed, the trainer files a `final` revision even when it is structurally identical to the initial observation; this makes the lifecycle guarantee explicit. Revision allocation is transactionally unique and monotonic for one realization.

Materialization attempts are separate events with `started`, `succeeded`, or `failed` outcomes. An attempt references the effective composition it tried to change; only a successful result that is actually observed can produce the next composition revision. A failed attempt therefore does not become “latest composition,” repeat an unchanged surface as a failed state, or consume a composition revision.

Failure-event filing is best-effort and may occur only when the metadata runtime and realization identity are already valid. Filing must not replace or mask the original loading/preparation exception, and retained error facts must use the existing bounded/sanitized failure policy rather than serializing arbitrary exception state. If safe filing fails, training propagates the original exception with the earlier composition history unchanged.

Views select the highest valid successful observation revision for “latest composition” and require a `final` revision when a consumer asks for finalized composition. History is never mutated. Artifact provenance links to the final composition revision that actually produced it when available. Attempt-event order is independent of composition-revision order.

Alternative considered: delay all realization filing until preparation completes. Rejected because initial and deferred-loading observations are useful for diagnostics and resource intelligence.

Alternative considered: mutate the first realization/component records. Rejected because the metadata backbone and SQLite store are append-oriented and provenance history would be lost.

### Decision: Model structure uses qualified identities and scoped inventories

Structural path-binding identities derive from an accepted catalog revision or realization and the family-declared top-level component identity. Revision-qualified and realization-qualified bindings are separate typed forms: revision bindings describe reusable accepted state, while realization bindings describe what exists in one execution after its selected materialization and runtime replacements. Below that boundary they preserve kind and component-local path:

```text
<owner>/component/<component-key>/module/<path>
<owner>/component/<component-key>/parameter/<path>
<owner>/component/<component-key>/buffer/<path>
<owner>/component/<component-key>/tensor/<source-key>
```

Each qualified path remains a distinct binding identity. A live observation may relate several path bindings to one observation-scoped tensor/parameter object identity, and several live objects may relate to one observation-scoped storage-group identity. Serialized source-key bindings are separate again and may map one-to-many or many-to-one to runtime path bindings after conversion. Object pointers may help deduplicate within an observation but are never persisted as durable identity. Optimization deduplication continues to use live-object semantics rather than assuming path-binding count equals unique trainable-object count.

An inventory declares:

- owner, component, source type, and observation time/revision;
- included kinds and path namespace;
- typed scoped coverage claims and omissions;
- duplicate-removal/traversal policy;
- derivation/schema version;
- cost class and source limitations.

A coverage claim identifies a dimension/scope, status (`complete`, `partial`, `unavailable`, or `unknown`), evidence basis, and omissions. Relevant dimensions include live topology, named state members, serialized source keys, descriptor fields, source-to-runtime mapping, alias/live-object relations, and storage relations. The schema uses a collection of claims rather than one fixed overall completeness enum so future evidence kinds can add dimensions without changing every inventory shape.

A complete live named-state claim accounts for parameters plus persistent and non-persistent buffers under its declared traversal policy. It may simultaneously declare source-key mapping unavailable. A complete selected-artifact-key claim accounts for every selected serialized key but makes no claim about non-persistent live buffers or unselected repository files. Unknown serialized keys are preserved in the artifact/source-key and mapping dimensions rather than being required for live-topology completeness.

Validation rejects any `complete` coverage claim when its dimension, scope, emitted membership, traversal policy, or source limitations are internally inconsistent with the producer contract. If enumeration degrades or omits an applicable member, the corresponding claim becomes `partial`, `unavailable`, or `unknown` with the reason; unrelated dimensions retain their own truthful status.

### Decision: Descriptors are selectable facts, not a binary dump mode

Structural collection supports bounded levels rather than only silent/full output:

- identity/topology: paths, kinds, ownership, parentage;
- descriptor: shape, rank, dtype, layout/stride when knowable, device/source location, trainability, persistence, numel, element size, byte estimates, and storage-group evidence;
- digest/statistical extensions: content digest or bounded derived statistics under explicit policy;
- raw values: outside metadata persistence scope.

Normal runs need not record every descriptor. A caller may request one tensor descriptor, a component inventory, or a declared subset. Resolvers inspect a live model, a safetensors header, a trusted checkpoint, or accepted stored records and return typed facts with source, coverage, quality, and cost. The shared metadata query layer chooses recorded facts when their immutable evidence, coverage claims, and policy remain valid, or invokes an allowed resolver when detail is absent. Resolved durable facts may be filed so unchanged immutable work is not repeated.

This distinction is important:

- **resolve** obtains a fact from a source;
- **file** accepts it into durable metadata;
- **query** selects relevant accepted facts and, when explicitly allowed, coordinates resolution.

Execution-owning code may use recorded immutable entity facts when validity is established. It should use live domain state for mutable execution decisions whose recorded observation may be stale. There is no blanket rule that execution code must always re-inspect live model state.

### Decision: Structural metadata consumes the shared query capability

Model-specific typed query selectors/results may exist, but dispatch, source policy, cost policy, ambiguity, freshness, persistence, and query result semantics belong to the system-wide metadata query capability tracked by `sd-scripts-edl`.

The provenance/catalog milestones can complete without that query API. The structural-query milestones must pause until:

- the shared typed query capability is implemented and accepted;
- qualified resource-component identity work tracked by `sd-scripts-d79` is available.

Direct model helper calls may remain for execution-local enumeration, but they cannot be presented as the completed metadata query capability or become a parallel resolver registry.

### Decision: Keep one architectural change with an enforceable implementation pause

Catalog identity, source selection/provenance, realization composition, structural descriptors, query integration, and fingerprints share identity invariants that are easier to audit in one accepted change. This OpenSpec therefore remains the umbrella implementation contract, with separately reviewed numbered milestones and a hard stop after provenance/catalog completion.

If the shared query/resource dependencies are delayed or their accepted design cannot satisfy the structural requirements, work stops at the dependency gate. The remaining sections may then be extracted into a follow-up OpenSpec without rewriting or weakening the completed catalog/provenance contract. The existence of many tasks alone is not a reason to split the architecture before those boundaries are known.

Alternative considered: split catalog, structure, and fingerprints into independent changes immediately. Deferred because it would duplicate shared identity decisions and make it easier for the later structural change to drift from the catalog/selection semantics. Independent archival boundaries remain available if the dependency pause becomes long-lived.

### Decision: Fingerprints are evidence products with explicit normalization

Fingerprint records identify algorithm, version, source scope, included/excluded categories, key normalization, dtype normalization, tensor byte normalization, component ordering, and scoped coverage claims. At minimum the change supports:

- exact artifact/manifest digests for unseen-source identity;
- structural fingerprints from ordered descriptor inventories;
- an experimental canonical state fingerprint over normalized tensor entries.

The canonical state experiment must test renamed files, reordered serialization, metadata-only changes, known namespace conversions, optional and unknown keys, dtype/precision changes, component replacement, tied/shared state, and one changed tensor. Until a policy has explicit equivalence semantics and passing fixtures, its output is evidence/candidate matching only and cannot auto-merge catalog identities.

The first canonical-state fingerprint in this change remains experimental and candidate-only even after its fixture matrix passes. The matrix determines observed behavior and future policy requirements; promotion to automatic normalized-equality or catalog-merge authority requires a later explicit accepted policy/spec change. Only exact representation evidence and already accepted carried/lineage assertions can trigger automatic identity reuse in this change.

Precision-insensitive or conversion-aware fingerprints are separate policies from exact normalized-state equality. Optional keys are never silently ignored; a policy must classify them as included, explicitly excluded, or semantically mapped.

### Decision: Compatibility output carries identity without becoming canonical

Repo-owned Kuro projections may carry portable catalog/revision/source/lineage identity and selected provenance summaries in model artifacts. Internal consumers continue to use accepted typed facts and relationships. `modelspec.*` and legacy `ss_*` remain compatibility projections and do not become catalog sources of truth merely because old files contain them.

Unknown external metadata is preserved as source evidence when policy permits, with namespace/schema/trust information. It is not automatically promoted to a verified portable identity.

## Risks / Trade-offs

- **[Full hashing adds startup I/O for unseen large sources]** → Stream content, check carried/recorded immutable evidence first, expose cost, perform the full hash once, and reuse versioned accepted results.
- **[Portable IDs can conflict with misleading embedded metadata]** → Record issuer and evidence, validate claims, prefer independently verified strong evidence, and surface conflicts rather than overwriting mappings.
- **[Directory and remote repository identities can drift with selection rules]** → Resolve immutable revisions and version the selected-file manifest policy.
- **[Append-only composition creates multiple records per realization]** → Provide explicit latest/final views keyed by monotonic revision rather than relying on insertion order.
- **[Canonical fingerprints can overclaim equivalence]** → Keep every normalization policy explicit and prohibit automatic merges from normalized fingerprints in this change.
- **[Complete structural inventories can be expensive]** → Use selectable scope/cost policy, header-only safetensors inspection, and cached immutable facts; never make broad coverage the default merely because it exists.
- **[Live and artifact inventories disagree]** → Preserve source kind, lifecycle phase, dimension-specific coverage, and mapping evidence instead of forcing one canonical observation.
- **[Broad cross-cutting scope can obscure handoffs]** → Implement and review numbered OpenSpec sections independently, with a hard dependency handoff between provenance/catalog work and structural/query work.

## Migration Plan

1. Freeze current model-realization behavior and loader-source fixtures before changing contracts.
2. Add catalog/provenance/composition types and persistent registration behind additive central APIs.
3. Migrate SD, SDXL, and SD3 loading to typed results and file initial/final composition observations.
4. Enable portable identity and exact unseen-source digest registration, then carry selected identity/provenance through Kuro projections.
5. Verify restart, rename, fallback, override, deferred-loading, conflict, and lineage behavior; retain the old run-realization projection during compatibility migration.
6. Pause at the documented handoff until the shared query and qualified resource-component dependencies are accepted.
7. Add qualified structural inventories/descriptors/resolvers, then integrate optimization and resource consumers.
8. Introduce fingerprint policies incrementally; keep experimental results non-merging until their test matrix proves the declared semantics.
9. Remove only superseded active seams after parity and migration-ledger checks. `tools/` and textual inversion remain untouched.

The active library must not depend on `tools/` for catalog, provenance, or structure behavior. If the audit finds such a dependency, the active caller is migrated to a library-owned boundary or the milestone stops; `tools/` itself is not edited to complete this change. The dedicated textual-inversion runtime is explicitly exempt from the new typed loading contract during this change and remains a deferred legacy path rather than evidence that the active migration is incomplete.

Rollback is additive through the provenance milestones: callers can temporarily ignore new catalog/provenance records while retaining the existing realization/artifact path. Once loader return contracts and persistent schema migrations land, rollback requires restoring the preceding contract/schema version while leaving append-only new records readable or explicitly unsupported; migrations must not destructively rewrite catalog history.

## Open Questions

- Which portable issuer namespace and concrete string encoding should Kuro-created lineage/revision identifiers use? The implementation milestone must choose and version it before artifacts are emitted.
- Which persistent catalog location/configuration becomes the product default across output directories? It must be cross-run and must not rely on a per-output ephemeral database; implementation research will select the existing project-consistent configuration boundary.
- Which initial known conversion mappings are safe enough for canonical-state comparison rather than candidate-only matching? The fingerprint fixture milestone decides policy-by-policy; absence of proof means no automatic merge.
- Whether a persisted conversion is an alternate representation of one revision or a new descendant revision depends on whether state semantics changed. The initial transformation taxonomy must encode both and reject an unclassified automatic lineage claim.
