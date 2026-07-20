## ADDED Requirements

### Requirement: Every family uses one uniform loading-result schema
Model loading SHALL preserve the source that actually succeeded through one shared typed loading-result schema containing ordered source observations, source selections, component-to-selection bindings, loading decisions, transformations, and evidence limitations.

#### Scenario: Local checkpoint is loaded
- **WHEN** a family loader successfully loads a local checkpoint
- **THEN** the loading result MUST identify the local-file source and resolved artifact evidence
- **AND** it MUST preserve the selected state/namespace and any conversion needed to interpret its components through the common collections

#### Scenario: Repository fallback succeeds
- **WHEN** a preferred remote variant or revision fails and a fallback succeeds
- **THEN** the loading result MUST preserve both the selected successful source and the fallback decision
- **AND** configured intent MUST NOT be reported as the successful source

#### Scenario: Component uses an external override
- **WHEN** a VAE, text encoder, denoiser, or future component is loaded from a source different from the primary source
- **THEN** that component MUST have its own component-to-selection binding and replacement/materialization evidence
- **AND** the realization MUST NOT imply that every component came from the primary source

#### Scenario: Source evidence is unavailable
- **WHEN** a loader cannot determine a source fact reliably
- **THEN** it MUST record the limitation or omit the unsupported claim according to validation
- **AND** it MUST NOT fabricate a revision, digest, component binding, or source kind

#### Scenario: Future family has different topology
- **WHEN** a future family declares components or source layout unlike SD, SDXL, or SD3
- **THEN** it MUST populate the same source, selection, binding, decision, transformation, and limitation collections
- **AND** it MUST NOT require family-specific provenance fields or a central family-name branch

### Requirement: Materialization provenance is selection-aware, component-aware, and ordered
The metadata system SHALL represent how selected state within source representations became family-declared realized components in declaration order.

#### Scenario: Unified artifact supplies several components
- **WHEN** one source artifact supplies multiple declared components
- **THEN** each realized component MUST retain a binding to its source selection and extraction/materialization evidence
- **AND** components MUST remain distinct accepted identities

#### Scenario: One checkpoint supplies EMA and non-EMA state
- **WHEN** separate loads select EMA and non-EMA state from the same exact checkpoint representation
- **THEN** both loading results MUST reference the same source-representation identity
- **AND** their source-selection and component-binding evidence MUST remain distinct

#### Scenario: Declared component is omitted
- **WHEN** a family-declared component is intentionally absent or deferred
- **THEN** composition evidence MUST preserve its declared identity and presence/materialization state
- **AND** the component MUST NOT disappear silently from a claimed complete top-level composition

#### Scenario: Structural conversion changes interpretation
- **WHEN** loading converts a source namespace or representation into the repository's runtime component interpretation
- **THEN** the conversion type/version and affected source/component scopes MUST be recorded
- **AND** the source representation identity MUST remain distinguishable from the runtime component identity

### Requirement: Provenance and runtime observations use an explicit boundary
The system SHALL classify a fact as source/materialization provenance when it is required to explain or reproduce how supplied source became a logical model revision or component, while transient execution state SHALL remain run-scoped observation.

#### Scenario: Checkpoint namespace is extracted
- **WHEN** component interpretation requires extracting or renaming a checkpoint namespace
- **THEN** that operation MUST be representable as source/materialization provenance

#### Scenario: Model moves to a device
- **WHEN** an unchanged realization moves between CPU and accelerator memory
- **THEN** that movement MUST NOT create a new source or catalog model identity
- **AND** it MAY be recorded as a phase-scoped runtime observation

#### Scenario: Runtime modification is persisted
- **WHEN** a runtime modification is saved as a reusable model artifact
- **THEN** artifact/revision lineage MUST record the persisted transformation
- **AND** the original source identity MUST remain intact

### Requirement: Realization composition observations are immutable and revisioned
Each run-scoped model realization SHALL own an ordered append-only sequence of composition observations with explicit lifecycle state.

#### Scenario: Initial loading completes
- **WHEN** the first loaded-component surface becomes available
- **THEN** the system MUST file composition revision 1 with `initial` state
- **AND** it MUST preserve component presence and component-to-selection bindings in family declaration order

#### Scenario: Deferred component materializes
- **WHEN** a later lifecycle boundary loads a previously absent component
- **THEN** the system MUST file a new monotonically increasing composition revision
- **AND** it MUST relate that observation to the preceding revision and new materialization evidence
- **AND** revision allocation MUST be transactionally unique for the owning realization

#### Scenario: Model preparation completes
- **WHEN** model preparation establishes the composition used for training
- **THEN** the system MUST file an explicit `final` composition revision even if no component changed
- **AND** consumers requesting finalized composition MUST select that revision rather than infer finality from timing

### Requirement: Materialization attempts are events rather than composition state
The metadata system SHALL record materialization attempts separately from successful realization-composition observations.

#### Scenario: Materialization attempt starts
- **WHEN** a deferred load or replacement begins from an accepted composition
- **THEN** a materialization-attempt event MAY reference the effective composition and intended component/source-selection change
- **AND** it MUST NOT allocate a new composition revision merely because the attempt started

#### Scenario: Materialization attempt succeeds
- **WHEN** the attempted component surface is successfully materialized and observed
- **THEN** the success event MUST reference the resulting composition observation
- **AND** only that successful observation may allocate the next composition revision

#### Scenario: Materialization attempt fails
- **WHEN** a component materialization fails after an initial observation
- **THEN** the system MAY file a bounded and sanitized failed-attempt event only when runtime and realization identity are already valid
- **AND** the failed attempt MUST reference the composition it attempted to change without becoming a composition revision
- **AND** metadata filing failure MUST NOT replace or mask the original materialization exception

### Requirement: Composition views use semantic order rather than insertion accidents
Metadata views SHALL resolve latest and finalized realization composition by validated realization identity, monotonic revision, and lifecycle state.

#### Scenario: Snapshot contains several composition revisions
- **WHEN** a consumer requests latest composition
- **THEN** the view MUST select the highest valid successful observation revision for that realization
- **AND** it MUST NOT rely on backend iteration order alone

#### Scenario: Final composition is required but missing
- **WHEN** a consumer requests finalized composition and no valid `final` revision exists
- **THEN** the query/view MUST return an explicit unavailable/incomplete result or fail according to its contract
- **AND** it MUST NOT treat `initial` as final implicitly

#### Scenario: Artifact is produced from a realization
- **WHEN** a model artifact is saved with finalized realization context
- **THEN** its provenance MUST reference the applicable final composition revision
- **AND** it MUST retain the stable owning realization identity

### Requirement: Provenance facts remain central and family-extensible
Central metadata SHALL own accepted provenance schemas, validation, records, relationships, and projections while family loaders own truthful source-resolution semantics.

#### Scenario: Future family introduces a new source layout
- **WHEN** a future family maps a new source layout to declared components
- **THEN** it MUST be able to provide typed source/materialization evidence without adding a family-name branch to central emitters
- **AND** genuinely family-local evidence MUST use an explicit namespaced/versioned extension

#### Scenario: Domain code files provenance
- **WHEN** a loader/trainer lifecycle boundary has accepted provenance facts
- **THEN** it MUST file them through central metadata APIs
- **AND** it MUST NOT call storage directly or create a domain-local metadata mini-system
