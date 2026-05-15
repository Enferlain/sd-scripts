## ADDED Requirements

### Requirement: Typed Metadata Backbone

The system SHALL provide a typed metadata backbone under `library/metadata/`
for metadata identities, records, events, relationships, validation, backend
interfaces, provider contracts, and projection interfaces.

#### Scenario: Define shared metadata records

- WHEN metadata for a run, model component, adapter artifact, or checkpoint
  artifact is collected
- THEN the metadata MUST be represented by typed records before it is exported as
  a raw dictionary

#### Scenario: Preserve relationships between records

- WHEN a checkpoint artifact is produced from a training run, model family, and
  adapter method
- THEN the backbone MUST be able to represent the relationship between those
  metadata records

### Requirement: Domain-Owned Metadata Providers

The system SHALL allow domains, model families, training modes, and adapter
families to provide metadata through provider surfaces without moving their
domain-specific facts into the central backend.

#### Scenario: Compose provider metadata

- WHEN training, model-family, and adapter providers emit metadata for the same
  artifact
- THEN the central backend MUST compose those records without requiring the
  backend to know family-specific implementation details

#### Scenario: Declare required provider facts

- WHEN a provider declares a metadata fact as required for an export
- THEN validation MUST fail if that fact is missing before the export is emitted

### Requirement: Repo-Owned Kuro Projection

The system SHALL provide a projection seam for repo-owned exported metadata keys
under the `kuro.*` namespace.

#### Scenario: Emit repo-owned artifact metadata

- WHEN checkpoint metadata is projected for an artifact
- THEN the exported metadata MUST include versioned `kuro.*` keys for the
  backbone-owned facts included in that projection

#### Scenario: Keep internal schema separate from export keys

- WHEN typed metadata records are changed internally
- THEN projection code MUST remain the boundary that maps internal records to
  exported `kuro.*` keys

### Requirement: Compatibility Metadata Projections

The system SHALL preserve compatibility projections for existing `ss_*` and
`modelspec.*` metadata keys used by current checkpoint artifacts.

#### Scenario: Preserve ss compatibility keys

- WHEN active checkpoint metadata is emitted through the backbone
- THEN existing `ss_*` keys that are currently produced for that checkpoint type
  MUST still be emitted with compatible values

#### Scenario: Preserve modelspec compatibility keys

- WHEN SD or SDXL-style checkpoint metadata currently requires `modelspec.*`
  keys
- THEN the backbone projection MUST continue to emit the compatible
  `modelspec.*` metadata for those artifacts

### Requirement: Checkpoint Metadata Integration

The system SHALL route the active checkpoint artifact metadata path through the
metadata backbone.

#### Scenario: Save checkpoint with metadata enabled

- WHEN an active training flow saves a checkpoint with metadata enabled
- THEN checkpoint export metadata MUST be produced by the metadata backbone and
  projection layer

#### Scenario: Respect metadata suppression

- WHEN an active training flow disables or minimizes checkpoint metadata through
  existing configuration
- THEN the backbone integration MUST preserve the current suppression or minimal
  metadata behavior

### Requirement: Fail-Fast Metadata Validation

The system SHALL fail fast when required metadata facts are missing at provider
or export boundaries.

#### Scenario: Missing required artifact fact

- WHEN an artifact projection requires a fact that no provider supplied
- THEN the export MUST fail with an actionable validation error naming the
  missing fact or provider requirement

#### Scenario: Optional fact omitted

- WHEN an optional metadata fact is not supplied by any provider
- THEN validation MUST allow export to continue without fabricating a value

### Requirement: Storage Backend Seams

The system SHALL provide storage backend interfaces and an in-memory
implementation that are shaped for future durable SQLite storage.

#### Scenario: Use in-memory backend in tests

- WHEN tests or first-slice checkpoint integration collect metadata
- THEN they MUST be able to use an in-memory backend without requiring SQLite or
  any new external dependency

#### Scenario: Keep durable storage replaceable

- WHEN a future SQLite backend is added
- THEN it MUST be able to satisfy the same storage interface used by the
  in-memory backend

### Requirement: Future Domain Extensibility

The system SHALL allow future metadata domains to be added without redesigning
the core backbone.

#### Scenario: Add future data metadata provider

- WHEN future data, cache, accounting, shard, or streamed-input metadata needs to
  be added
- THEN it MUST be possible to add provider records and projections without
  changing existing checkpoint compatibility projections

#### Scenario: Avoid first-slice future feature implementation

- WHEN this change is implemented
- THEN async data ingestion, data-shard accounting, streamed training ingest,
  and exposure accounting MUST remain out of scope unless they are represented
  only by generic extension seams
