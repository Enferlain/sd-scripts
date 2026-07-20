## 1. Freeze The Continuation Baseline

- [ ] 1.1 Add focused fixtures for current SD, SDXL, and SD3 loading results, initial realized-component filing, deferred denoiser behavior, and artifact provenance before changing contracts.
- [ ] 1.2 Inventory every active loader source/fallback/override/conversion branch and map each branch to the source/materialization evidence required by `model-source-provenance`.
- [ ] 1.3 Inspect central metadata package neighbors and configuration ownership, then amend the design with the chosen catalog module placement, portable issuer encoding, and cross-output persistent catalog location before adding production files.
- [ ] 1.4 Verify the frozen baseline, run `review-mcp`, address actionable findings, rerun affected checks, and hand off section 1 before starting section 2.

## 2. Catalog Identity Records And Persistence Authority

- [ ] 2.1 Add accepted typed facts and qualified identity constructors for catalog lineages, immutable selected-state revisions, source representations, source selections, release/name claims, recognition evidence, conflicts, and distinct display-alias/redirect/represents/derivation/equivalence relationships.
- [ ] 2.2 Register central emitters and validation for the catalog facts, including identity-scheme/policy versions, evidence strength, authenticated/trusted-local/unsigned/invalid carried-claim trust classes, and prohibited placeholder identities.
- [ ] 2.3 Add the narrow central catalog authority in the placement selected by task 1.3, with transactional/idempotent exact-representation registration, atomic conflicting-target detection, owner-qualified source-selection allocation, append-only repair operations, and storage-independent behavior over accepted records.
- [ ] 2.4 Extend the SQLite schema/index layer additively for efficient strong-evidence lookup, conflict detection, restart-safe catalog resolution, and index consistency validation/rebuild without making index tables canonical.
- [ ] 2.5 Wire the selected durable catalog configuration so supported active runs resolve the same catalog across output directories; require explicit observable opt-in for ephemeral mode and fail instead of silently falling back when persistent initialization fails.
- [ ] 2.6 Add restart, deterministic representation assignment, separate provisional catalog/revision assignment, multi-selection, index-rebuild, conflict, typed-relationship repair, carried-trust, and ephemeral-mode tests.
- [ ] 2.7 Run section 2 tests, lint/type checks for touched files, `review-mcp`, and hand off the catalog authority before section 3.

## 3. Exact Source Identity And Recognition Policy

- [ ] 3.1 Add versioned source-evidence types for local files, canonical local-directory manifests, immutable remote revisions, selected remote files, source selections, and carried portable claims, with issuer/schema/kind/payload/trust classification and invalid-claim degradation to unverified evidence.
- [ ] 3.2 Implement streaming complete-file hashing for unseen files by reusing the repo hash primitive and record algorithm, size, source observation, and identity-policy version.
- [ ] 3.3 Implement deterministic versioned selected-file manifests with normalized `/` paths, Unicode/case policy, duplicate rejection, selected symlink treatment, special-file rejection, stable selected membership, complete selected-file digests, and mutation-during-hash detection.
- [ ] 3.4 Implement progressive recognition in exact-representation then source-selection/logical-assignment order, with weak candidate results and explicit conflict outcomes that never select by insertion order.
- [ ] 3.5 Implement digest-cache validation that treats path/stat attributes only as cache evidence and rehashes whenever accepted immutable evidence cannot prove the current content matches.
- [ ] 3.6 Add fixtures for byte clones, renamed/moved files, changed bytes with reused stat-like evidence, EMA/non-EMA selection, reordered/partial manifests, normalization collisions, symlinks/special files, mutation during hashing, missing selected files, different file selections at one remote revision, metadata-only file changes, mutable remote names resolving to immutable revisions, and recognition-policy upgrades.
- [ ] 3.7 Run section 3 quality gates, `review-mcp`, and hand off exact representation/selection recognition before section 4.

## 4. Typed Model Loading And Provenance Contract

- [ ] 4.1 Replace the evidence-free loading tuple with one shared typed loading result containing model version, family-declared components, ordered source observations, source selections, component-to-selection bindings, loading decisions, limitations, and transformations classified as source/materialization, runtime observation, or persisted lineage/revision evidence.
- [ ] 4.2 Define typed deferred-materialization updates that preserve prior loading evidence while replacing the authoritative loaded-component surface.
- [ ] 4.3 Add central builders for source representations, selections, component-to-selection bindings, materialization attempts/results, and transformation facts without retaining live modules or accessing metadata storage.
- [ ] 4.4 Add central emitters, registry routes, relationships, and validation for the new provenance facts.
- [ ] 4.5 Update trainer setup to retain typed loading/catalog state while preserving its family-declared component convenience views.
- [ ] 4.6 Add contract tests proving SD/SDXL/SD3 and future families populate the same collection-based schema without family-specific provenance fields or central family-name branches.
- [ ] 4.7 Run section 4 quality gates, `review-mcp`, and hand off the shared loading/provenance contract before family migrations.

## 5. Active Family Source Capture

- [ ] 5.1 Migrate SD loading to report local-checkpoint versus Diffusers source, successful revision/layout, checkpoint conversion, external VAE replacement, and RamTorch/runtime-only transformation classification.
- [ ] 5.2 Migrate SDXL loading to report local-checkpoint versus Diffusers source, variant attempt/fallback, immutable revision/layout, conversions, external VAE replacement, and runtime-only transformations.
- [ ] 5.3 Migrate SD3 loading to populate the same source/selection/binding collections with its unified checkpoint and any independent CLIP-L, CLIP-G, T5, VAE, and denoiser sidecar/deferred sources and omissions.
- [ ] 5.4 Add family fixture coverage for successful sources, fallbacks, missing/deferred components, overrides, conversions, and evidence limitations without requiring GPU-heavy training runs.
- [ ] 5.5 Verify existing model-loading/component behavior and compatibility output remain unchanged apart from the new typed evidence.
- [ ] 5.6 Run section 5 quality gates, `review-mcp`, and hand off active-family provenance before section 6.

## 6. Revisioned Realization Composition

- [ ] 6.1 Add accepted successful realization-composition observation facts with stable realization identity, transactionally unique monotonic revision, `initial`/`intermediate`/`final` state, ordered component-to-selection bindings, prior-revision link, and lifecycle evidence.
- [ ] 6.2 Add separate materialization-attempt events with started/succeeded/failed outcomes and links to the effective/prior and resulting composition when available.
- [ ] 6.3 Add validation and views for latest successful and finalized composition that use revision/state semantics rather than attempt or backend iteration order.
- [ ] 6.4 Change initial post-load filing to create composition revision 1 and retain its accepted identity state.
- [ ] 6.5 Change deferred-loading/model-preparation boundaries to file attempt events, append a composition revision only after successful observation, and append an explicit final revision; sanitize failure events and never mask the original exception.
- [ ] 6.6 Link produced model artifacts to the applicable final composition revision while preserving the stable realization identity.
- [ ] 6.7 Add tests for unchanged initial-to-final composition, successful deferred presence changes, failed attempt without revision consumption, retry after failure, concurrent revision allocation, ambiguous/invalid revisions, and artifact linkage.
- [ ] 6.8 Run section 6 quality gates, `review-mcp`, and hand off realization finalization before section 7.

## 7. Portable Lineage And Provenance Milestone Completion

- [ ] 7.1 Add model revision/lineage production at save boundaries for trained descendants, alternate representations, component replacements, and explicitly unresolved transformation classifications.
- [ ] 7.2 Extend Kuro projections to carry portable catalog/revision/source-representation/source-selection/lineage references, trust classification, and bounded provenance summaries from accepted facts.
- [ ] 7.3 Preserve unsigned/invalid external identifiers according to their trust class and verify ModelSpec/`ss_*` compatibility projections remain non-canonical.
- [ ] 7.4 Add cross-run and simulated cross-install fixtures proving carried claims retain trust, deterministic unseen representation IDs match, source selections and provisional logical assignments remain distinct, conflicting publisher claims survive, and lineage is preserved.
- [ ] 7.5 Update the metadata/model READMEs only with implemented behavior, update the migration control/inventory, CHANGELOG, ROADMAP, and relevant Beads state.
- [ ] 7.6 Run the complete provenance/catalog test slice and affected unit suite, lint/type checks, strict OpenSpec validation, `review-mcp`, and hand off milestone group A.

## 8. Shared Query And Qualified Identity Dependency Gate

- [ ] 8.1 Confirm `sd-scripts-edl` has delivered the accepted system-wide typed metadata query contract for selector/result typing, source/cost policy, ambiguity, freshness, resolver dispatch, and optional filing.
- [ ] 8.2 Confirm `sd-scripts-d79` has delivered accepted qualified resource-component identity links compatible with catalog/realization component identities.
- [ ] 8.3 Reconcile this change's structural tasks and specs with the implemented dependency APIs, amending OpenSpec artifacts where names or boundaries differ without weakening the requirements; if reconciliation would weaken them, stop the milestone and revise the dependency/change design through an explicit reviewed handoff.
- [ ] 8.4 Run strict OpenSpec validation, `review-mcp` on the reconciled boundary, and hand off the dependency gate; do not start section 9 until all section 8 tasks pass.

## 9. Structural Identity And Inventory Schema

- [ ] 9.1 Add separate revision- and realization-qualified path-binding identities plus observation-scoped live-object/storage-group and serialized source-key identities for modules, parameters, and persistent/non-persistent buffers.
- [ ] 9.2 Add explicit many-to-one/one-to-many relationships among path bindings, live objects, storage groups, and source keys without persisting object pointers.
- [ ] 9.3 Add inventory facts for owner/source/scope, included kinds, traversal/duplicate policy, cost class, derivation version, lifecycle observation, and extensible dimension-scoped coverage claims with status/basis/omissions.
- [ ] 9.4 Add typed tensor descriptors for shape/rank/dtype, layout/stride/device/source location when knowable, trainability/persistence, numel, element size, byte estimates, and storage evidence.
- [ ] 9.5 Add central builders, emitters, registry routes, relationships, validation, and views for structural facts without retaining live tensors/modules.
- [ ] 9.6 Add validation tests proving live topology/state-member coverage can be complete while source-key mapping is unavailable, selected-artifact key coverage is independent of repository/live-buffer coverage, and inconsistent complete claims are rejected per dimension.
- [ ] 9.7 Run section 9 quality gates, `review-mcp`, and hand off the structural schema before resolver work.

## 10. Live And Artifact Structural Resolvers

- [ ] 10.1 Refactor reusable traversal semantics from `library/models/parameter_dump.py` into typed live inventory/descriptor resolution while retaining the text renderer as a projection/consumer.
- [ ] 10.2 Implement safetensors header resolution for selected keys, shapes, dtypes, offsets, and metadata without loading tensor values.
- [ ] 10.3 Define and enforce the trust policy for non-safetensors checkpoint inspection, including safe refusal when bounded trusted inspection is unavailable.
- [ ] 10.4 Preserve live versus artifact source, lifecycle, duplicate/path-binding versus live-object traversal, persistence, source-key mapping, storage-sharing, and dimension-scoped coverage in resolver outputs.
- [ ] 10.5 Add bounded single-tensor, selected-subtree, component, and complete-inventory tests across live modules and safetensors fixtures.
- [ ] 10.6 Run section 10 quality gates, `review-mcp`, and hand off resolvers before query integration.

## 11. Shared Typed Query Integration

- [ ] 11.1 Add model structural selectors/results to the implemented shared query surface without adding a model-only dispatcher, cache, or ambiguity policy.
- [ ] 11.2 Register live and artifact resolvers with explicit availability, trust, cost, scope, and persistence policy.
- [ ] 11.3 Prefer valid accepted immutable descriptors when owner evidence, requested coverage claims, and policy version satisfy a query; invoke resolution only when allowed and needed.
- [ ] 11.4 Support optional filing/reuse of resolved durable descriptors and invalidate reuse when source evidence, scope, coverage, algorithm, or policy version changes.
- [ ] 11.5 Add tests for one-tensor lookup, recorded reuse without reinspection, bounded resolver fallback, ambiguity, unavailable/trust-denied sources, cost rejection, and stale-policy resolution.
- [ ] 11.6 Run section 11 quality gates, `review-mcp`, and hand off model query integration before downstream consumers.

## 12. Optimization And Resource Consumer Integration

- [ ] 12.1 Relate optimization component/module/parameter target refs to accepted revision/realization path-binding identities when available while preserving live refs, selectors, grouping policy, and live-object deduplication.
- [ ] 12.2 Migrate structural resource facts from bare component owners to accepted model/component, structural path-binding, and observation-local object/storage identities using the completed dependency boundary.
- [ ] 12.3 Align component/parameter/buffer size summaries with accepted descriptors and preserve measured/structural/estimated/derived semantics.
- [ ] 12.4 Keep early/legacy paths at their current behavior with explicit unresolved ownership when accepted identities are unavailable; degrade resource linkage rather than failing execution or inventing qualified IDs.
- [ ] 12.5 Add integration tests across model inventory, target refs, structural resource facts, accounting evidence, and recorded descriptor reuse.
- [ ] 12.6 Run section 12 quality gates, `review-mcp`, and hand off consumer integration before fingerprint work.

## 13. Structural And State Fingerprint Evidence

- [ ] 13.1 Add versioned exact-artifact, structural-descriptor, and canonical-state fingerprint fact types with explicit algorithm, normalization, ordering, dimension-scoped coverage, and omissions.
- [ ] 13.2 Implement structural fingerprints over deterministic ordered descriptor inventories.
- [ ] 13.3 Implement the first experimental canonical-state fingerprint policy over normalized tensor entries as candidate-only evidence without granting automatic normalized-equality or catalog-merge authority.
- [ ] 13.4 Build fixtures for renamed files, reordered serialization, metadata-only changes, known namespace conversions, optional/unknown keys, precision changes, component replacement, tied/shared state, and one changed tensor.
- [ ] 13.5 Classify observed fixture behavior for exact representation, normalized state, and candidate similarity, but enforce the first canonical-state policy as candidate-only; require a later accepted policy/spec change before any normalized fingerprint can auto-merge identities.
- [ ] 13.6 Run section 13 quality gates, `review-mcp`, and hand off fingerprint evidence before final audit.

## 14. Final Capability Audit And Documentation

- [ ] 14.1 Audit active library callers for parallel source IDs, evidence-free loader tuples, mutable realization composition, bare structural owners, model-only query seams, and dependencies on `tools/`; migrate active callers to library-owned boundaries or stop, while leaving `tools/` unchanged and textual inversion explicitly exempt/deferred.
- [ ] 14.2 Add end-to-end CPU-safe tests covering unseen catalog registration, process restart, typed family loading, initial/final composition, artifact lineage, targeted descriptor query, recorded reuse, and structural resource linkage.
- [ ] 14.3 Run affected unit suites and representative non-GPU integration/config tests, `ruff`, `ty`, `git diff --check`, and strict OpenSpec validation; document any environment-only gates honestly.
- [ ] 14.4 Update implemented-state READMEs, CHANGELOG, ROADMAP, migration/inventory docs, and all linked Beads issues; keep speculative follow-up in `docs_design` or new Beads rather than module READMEs.
- [ ] 14.5 Request final `review-mcp`, wait for its result, address actionable findings, rerun affected gates, and hand off the completed change for archive.
