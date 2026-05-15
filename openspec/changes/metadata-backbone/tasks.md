## 1. Backbone Package

- [x] 1.1 Create `library/metadata/` with package exports and module boundaries for records, providers, validation, backends, projections, and storage.
- [x] 1.2 Define typed dataclasses for metadata identities, run records, model/component records, artifact records, events, and relationships.
- [x] 1.3 Define provider contracts that let domains emit records, events, relationships, and required-fact declarations.
- [x] 1.4 Define validation errors and fail-fast validators for missing required provider or projection facts.
- [x] 1.5 Define backend and storage interfaces plus an in-memory implementation suitable for tests and first integration.

## 2. Projection Layer

- [x] 2.1 Implement projection interfaces that map typed metadata records into exported artifact dictionaries.
- [x] 2.2 Implement an initial versioned `kuro.*` projection for backbone-owned artifact metadata.
- [x] 2.3 Implement an `ss_*` compatibility projection that preserves currently exported training metadata keys and values.
- [x] 2.4 Implement a `modelspec.*` compatibility projection for SD and SDXL artifact metadata.
- [x] 2.5 Add projection-level validation so required export facts fail before metadata is written to an artifact.

## 3. Provider Integration

- [x] 3.1 Bridge existing training-run metadata construction into a training metadata provider or provider wrapper.
- [x] 3.2 Bridge existing model-family metadata helpers into model/component providers without moving family-specific facts into the backend.
- [x] 3.3 Add a minimal adapter/method provider surface for active adapter checkpoint metadata.
- [x] 3.4 Add assembly code that composes the active checkpoint metadata providers for a checkpoint save.

## 4. Checkpoint Path Migration

- [x] 4.1 Route active checkpoint metadata creation through the metadata backend and projection layer.
- [x] 4.2 Preserve existing metadata suppression or minimal-metadata behavior for checkpoint saves.
- [x] 4.3 Keep existing public helper functions as wrappers or compatibility seams where needed during migration.
- [x] 4.4 Verify exported checkpoint metadata still includes compatible `ss_*` and `modelspec.*` keys where they existed before.

## 5. Tests And Verification

- [x] 5.1 Add unit tests for metadata record creation, relationship composition, provider emission, and in-memory backend behavior.
- [x] 5.2 Add validation tests for missing required facts and omitted optional facts.
- [x] 5.3 Add projection tests for `kuro.*`, `ss_*`, and `modelspec.*` outputs.
- [x] 5.4 Add checkpoint integration coverage for the active metadata path and metadata suppression behavior.
- [x] 5.5 Run targeted pytest coverage for the new metadata package and active checkpoint metadata path.
- [x] 5.6 Run relevant lint/type checks for touched modules.

## 6. Documentation And Handoff

- [x] 6.1 Update developer-facing metadata documentation with package responsibilities, provider ownership, and projection boundaries.
- [x] 6.2 Update `CHANGELOG.md` and `ROADMAP.md` with the implemented metadata backbone status.
- [x] 6.3 Record follow-up beads for deferred SQLite durable storage, broader data/cache metadata integration, and future analytics/export formats.
