## 1. Establish the baseline and parity contract

- [x] 1.1 Archive the completed `family-declared-loaded-components` change and revalidate this change against the merged base specification
- [x] 1.2 Inventory every active SD, SDXL, and SD3 model metadata call site, including adapter, full-model, checkpoint, and direct family save paths
- [x] 1.3 Add complete output-parity fixtures for representative SD1, SD2 epsilon/v, SDXL DDPM/RF, and SD3 metadata dictionaries
- [x] 1.4 Add parity coverage for adapter versus full-model artifact roles, SD3 attention-mask fields, user extension fields/collisions, and current `no_metadata` behavior

## 2. Define typed model and identity contracts

- [x] 2.1 Add central schema-only dataclasses for model-family declaration references, run-scoped model realizations, realized loaded components, and artifact-facing model facts
- [x] 2.2 Define a small typed artifact-resolution context that replaces the broad boolean-driven ModelSpec builder inputs
- [x] 2.3 Add shared constructors for qualified model-realization and component identities using run, realization, and family-local component scopes
- [x] 2.4 Add item validation for required identities, unique component keys/order, serializable roles/capabilities, required artifact facts, and rejection of live module objects
- [x] 2.5 Add focused schema and identity tests covering repeated component keys across runs/families and multiple same-role components

## 3. Add central routing, records, and relationships

- [x] 3.1 Register all new accepted model item types and emitter routes in the single central metadata registry
- [x] 3.2 Implement central emitters for model-realization, loaded-component, family-specific, and artifact-facing facts
- [x] 3.3 Emit explicit run-to-realization, realization-to-component, and artifact-to-realization relationships without placeholder identities
- [x] 3.4 Add backend/SQLite round-trip and graph-index tests for qualified model/component records and relationships

## 4. Replace family dictionary production

- [x] 4.1 Define the typed checkpoint/model facet contract for family-owned semantic resolution while keeping component declarations in `library/models/<family>/__init__.py`, shared component helpers in `library/models/components.py`, and checkpoint workflow behavior in `library/strategies/<family>/checkpointing.py`
- [x] 4.2 Implement SD typed fact resolution for SD1/SD2 architecture, implementation, objective, resolution, timestep, encoder-layer, and artifact-role semantics
- [x] 4.3 Implement SDXL typed fact resolution for adapter/full-model and DDPM/RF omission behavior
- [x] 4.4 Implement SD3 typed fact resolution for model version, ModelSpec omission behavior, and attention-mask compatibility facts
- [x] 4.5 Add family-level tests proving resolvers return canonical typed facts and never pre-rendered compatibility keys
- [x] 4.6 Add a synthetic future-family contract test proving shared model/component/artifact facts require no family-name branch in central metadata code

## 5. File model realizations from runtime state

- [x] 5.1 Build realized component facts in a central model metadata builder from the authoritative family-declared loaded-component collection while preserving order and excluding module objects
- [x] 5.2 File model-realization, component, and optional family-contribution facts once at the trainer's post-load lifecycle boundary through the shared metadata runtime
- [x] 5.3 Ensure artifact registration and later resource/optimization references can reuse the same qualified model/component identities
- [x] 5.4 Add metadata-builder/trainer tests for filing order, absent components, repeated roles, and unavailable identity handling without loading real models
- [x] 5.5 Establish central run/model builder modules so reusable domain-input-to-fact assembly is distinct from typed-item-to-record emitters

## 6. Make compatibility projections semantic

- [x] 6.1 Rewrite `ModelSpecCompatibilityProjection` to stamp its supported SAI ModelSpec version and map canonical accepted facts to ModelSpec keys and omission rules
- [x] 6.2 Extend repo-owned `kuro.*` output across the new model entity types with explicit artifact scope, deterministic identity-qualified multi-record output, and no last-write-wins collisions; map only the accepted model-family facts in this change to compatibility-only Kohya `ss_*` output while retaining `SsCompatibilityProjection` as the eventual shared boundary for other concern-owned legacy fields
- [x] 6.3 Keep safetensors stringification as the final boundary and add required-fact projection validation
- [x] 6.4 Add projection tests covering complete parity fixtures, invalid required claims, optional omission, extension fields, and deterministic output

## 7. Migrate checkpoint and artifact paths

- [x] 7.1 Change trainer checkpoint metadata assembly to consume typed artifact/model facts and accepted snapshots instead of strategy dictionaries
- [x] 7.2 Migrate SD/SDXL adapter checkpoint paths and SDXL/SD3 full-model save paths to the same central projection ownership
- [x] 7.3 Preserve current `no_metadata` output behavior without broadening this change into the separate product-policy decision
- [x] 7.4 Link produced checkpoint artifacts to accepted model realizations when stable identities are available

## 8. Remove transitional seams and document the boundary

- [ ] 8.1 Remove active `CheckpointingStrategy.get_model_metadata()` and `update_metadata()` dictionary hooks after all active families move
- [ ] 8.2 Remove or narrow active uses of `get_model_metadata_from_config` and the broad flag-driven ModelSpec builder; do not retain wrappers solely for deprecated scripts
- [ ] 8.3 Update `library/metadata/README.md`, metadata design/migration notes, `ROADMAP.md`, and `CHANGELOG.md` with the settled family/central ownership and qualified identity boundary
- [ ] 8.4 Update `library/models/README.md` if implementation reveals any refinement to the component-code versus model-behavior ownership boundary
- [ ] 8.5 Record any deferred adapter-method, source-metadata ingestion, hash-generation, or extension-policy work as separate beads rather than expanding this slice

## 9. Verification and completion

- [ ] 9.1 Run focused metadata, strategy, trainer, checkpoint-projection, storage, and graph tests
- [ ] 9.2 Run relevant static checks for changed metadata/model/strategy modules
- [ ] 9.3 Compare complete pre/post migration metadata dictionaries for every parity fixture and document intentional differences or omissions as separate decisions, including the corrected family reference-implementation identifiers
- [ ] 9.4 Validate the OpenSpec, review the final diff against the metadata ownership rules, close the model-family migration bead, and archive the change
