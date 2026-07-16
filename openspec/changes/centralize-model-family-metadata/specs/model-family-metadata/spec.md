## ADDED Requirements

### Requirement: Typed model facts are the authoritative internal representation
The metadata system SHALL represent model-family, run-specific model-realization, loaded-component, and artifact-facing model facts as accepted typed items rather than as pre-rendered compatibility dictionaries.

#### Scenario: Family supplies model facts
- **WHEN** an active model family resolves metadata for a loaded model or output artifact
- **THEN** it MUST return accepted typed facts with canonical field names
- **AND** it MUST NOT return a `modelspec.*` dictionary as the authoritative internal value

#### Scenario: Internal consumer reads model facts
- **WHEN** a metadata view, analytics consumer, or relationship builder reads model information
- **THEN** it MUST consume canonical accepted facts
- **AND** it MUST NOT parse `modelspec.*`, `ss_*`, or safetensors string values to reconstruct internal meaning

### Requirement: Family declaration, model realization, and artifact facts remain distinct
The metadata system SHALL distinguish static family-declared topology, the model realization observed in one run, and metadata describing a produced artifact.

#### Scenario: Run loads a model family
- **WHEN** model loading completes for a training run
- **THEN** the system MUST record a run-scoped model realization derived from the selected family/version
- **AND** that realization MUST reference family-declared topology without becoming an artifact record

#### Scenario: One realization produces different artifact roles
- **WHEN** one loaded realization produces adapter and full-model artifacts whose external metadata differs
- **THEN** each artifact MUST receive artifact-specific model facts
- **AND** the underlying realization identity and component topology MUST remain unchanged

### Requirement: Family-owned semantics feed central metadata ownership
Model families SHALL own resolution of family-specific model meaning while the central metadata package owns accepted schemas, validation, routing, records, relationships, and projections.

#### Scenario: Family resolves an architecture identifier
- **WHEN** SD, SDXL, SD3, or a future family maps its model version and artifact role to architecture or implementation semantics
- **THEN** that mapping MUST be resolved by a family-owned facet returning central typed facts
- **AND** the family facet MUST NOT render external metadata keys or write to a metadata backend

#### Scenario: Family has no extra run facts
- **WHEN** a family has no additional family-specific metadata for a boundary
- **THEN** it MUST return an empty typed contribution or omit that contribution
- **AND** it MUST NOT retain a no-op dictionary-mutation hook solely for interface compatibility

#### Scenario: Future family joins the shared metadata path
- **WHEN** a future model family declares loaded components and resolves the shared typed model/artifact facts
- **THEN** the central metadata runtime, emitters, relationships, and field-driven compatibility projections MUST accept those facts without a family-name branch
- **AND** family-specific semantics MUST remain in the family resolver or an explicit namespaced/versioned contribution rather than an unrestricted compatibility dictionary

#### Scenario: Future family needs another export format
- **WHEN** a future model family requires an external metadata format not represented by SAI ModelSpec
- **THEN** that format MUST be implemented as a separate projection over accepted canonical facts
- **AND** it MUST NOT replace or fork the shared model-realization contract

### Requirement: Model component facts derive from family declarations
The metadata system SHALL derive top-level component facts from the authoritative family-declared loaded-component surface.

#### Scenario: Recording loaded component topology
- **WHEN** a model realization is filed after model loading
- **THEN** component facts MUST preserve declaration order, stable component key, public name, roles, capabilities, and observed presence
- **AND** they MUST NOT persist the live module object

#### Scenario: Multiple components share a role
- **WHEN** a family declares multiple components with the same role
- **THEN** each component MUST remain a distinct metadata identity in family-declared order
- **AND** metadata MUST NOT collapse them into one role slot

#### Scenario: Generic metadata consumes components
- **WHEN** generic metadata code needs model component structure
- **THEN** it MUST consume the family declaration or loaded-component surface
- **AND** it MUST NOT reconstruct `text_encoders`, `vae`, and `denoiser` as a universal component inventory

### Requirement: Model and component identities are durably qualified
The metadata graph SHALL qualify model-realization and component identities by their owning runtime scopes so family-local component keys remain unambiguous across runs and model families.

#### Scenario: Repeated component key appears in different realizations
- **WHEN** two runs or model families each declare a component key such as `vae` or `denoiser`
- **THEN** the resulting component records MUST have different durable identities
- **AND** each record MUST preserve its family-local component key as a queryable fact

#### Scenario: Component is related to its realization
- **WHEN** a component record is emitted for a model realization
- **THEN** it MUST have an explicit relationship to the qualified owning realization
- **AND** the realization MUST be related to its run when the run identity is available

#### Scenario: Required ownership identity is unavailable
- **WHEN** a producer cannot supply the identity required to qualify a realization or component
- **THEN** it MUST omit or reject the affected filing according to item validation
- **AND** it MUST NOT synthesize placeholder run, model, or component identities

### Requirement: Model realization facts use the shared metadata runtime path
The active runtime SHALL file low-volume model-realization and component facts through the existing metadata registry, runtime, emitter, and backend flow.

#### Scenario: Model loading completes
- **WHEN** the trainer has a stable run identity, model realization, and loaded-component surface
- **THEN** it MUST file the corresponding accepted typed items through `MetadataRuntime.file(...)` or its synchronous batch equivalent
- **AND** domain code MUST NOT call metadata storage or backend APIs directly

#### Scenario: New model fact type is accepted
- **WHEN** a new shared model metadata item is introduced
- **THEN** its accepted type and emitter route MUST be registered in the central metadata registry
- **AND** runtime or validation code MUST NOT add a parallel supported-type list or routing chain

### Requirement: Artifact relationships reuse accepted model identities
Produced model artifacts SHALL relate to the accepted model realization and component identities they describe or derive from when those identities are available.

#### Scenario: Checkpoint is produced from a loaded realization
- **WHEN** a checkpoint or adapter artifact is registered with a known model-realization identity
- **THEN** the metadata graph MUST record explicit provenance between the artifact and realization
- **AND** downstream artifact-lineage consumers MUST be able to resolve the accepted realization record

#### Scenario: Artifact identity is unavailable
- **WHEN** an export boundary cannot provide a stable artifact or realization identity
- **THEN** the corresponding provenance edge MUST be omitted
- **AND** the projection MUST NOT invent an identity only to complete the graph

### Requirement: Compatibility metadata is projected from canonical facts
The metadata system SHALL create `modelspec.*`, family-specific `ss_*`, and safetensors string metadata only at projection boundaries.

#### Scenario: Projecting SAI ModelSpec metadata
- **WHEN** an artifact requests ModelSpec-compatible metadata
- **THEN** `ModelSpecCompatibilityProjection` MUST map canonical artifact/model facts to `modelspec.*` keys
- **AND** accepted records MUST NOT require pre-prefixed duplicate facts as the source of truth

#### Scenario: Projecting family-specific compatibility facts
- **WHEN** accepted family facts have defined Kohya-compatible output keys
- **THEN** a central compatibility projection MUST render the corresponding `ss_*` keys and omission behavior
- **AND** the family strategy MUST NOT mutate the projected dictionary directly

#### Scenario: Rendering safetensors metadata
- **WHEN** accepted typed values are exported to a safetensors header
- **THEN** the final projection MUST produce a string-to-string mapping
- **AND** stringification MUST occur after semantic validation and compatibility mapping

### Requirement: ModelSpec claims validate required artifact facts
The metadata system SHALL validate the facts required by the selected ModelSpec artifact claim before producing compatibility metadata.

#### Scenario: Projection stamps its supported ModelSpec version
- **WHEN** an artifact requests SAI ModelSpec compatibility metadata
- **THEN** the projection MUST emit the specification identifier/version it implements
- **AND** runtime family producers MUST NOT be required to supply that projection-format constant as an observed model fact

#### Scenario: Required ModelSpec fact is missing
- **WHEN** an artifact projection claims SAI ModelSpec compatibility without required architecture, implementation, title, or applicable image resolution facts
- **THEN** projection MUST fail with a metadata validation error identifying the missing fact

#### Scenario: Optional family fact is inapplicable
- **WHEN** prediction type, timestep range, encoder layer, or another optional fact does not apply to the selected family/objective/artifact
- **THEN** projection MUST omit the corresponding external key
- **AND** it MUST NOT emit an inferred placeholder value

### Requirement: Active family exports preserve external parity
The migration SHALL preserve the currently supported SD, SDXL, and SD3 external metadata behavior while replacing the internal dictionary path.

#### Scenario: SD or SDXL adapter metadata is projected
- **WHEN** an active SD or SDXL adapter artifact is exported through the typed path
- **THEN** its complete `modelspec.*` output MUST match the pre-migration output for the same effective inputs

#### Scenario: Full-model metadata is projected
- **WHEN** an SDXL or SD3 full-model safetensors artifact is exported through the typed path
- **THEN** its artifact-role-specific ModelSpec output MUST match the pre-migration output for the same effective inputs

#### Scenario: Objective controls prediction metadata
- **WHEN** DDPM epsilon/v or rectified-flow objective semantics determine ModelSpec prediction behavior
- **THEN** the typed path MUST preserve the current value or omission behavior for the selected family

#### Scenario: SD3 family fields are projected
- **WHEN** SD3 attention-mask facts are supplied by the family facet
- **THEN** checkpoint compatibility output MUST preserve the current family-specific `ss_*` fields and values

#### Scenario: No-metadata policy is requested
- **WHEN** `output.saving.no_metadata` is enabled during this migration
- **THEN** the typed projection path MUST preserve the current observable export behavior
- **AND** this change MUST NOT settle the separately tracked long-term no-metadata product policy

### Requirement: Replaced active dictionary seams are removed
The completed migration SHALL remove active model metadata APIs whose only purpose is to build or mutate compatibility dictionaries.

#### Scenario: All active families use typed facts
- **WHEN** SD, SDXL, and SD3 active checkpoint paths pass parity coverage through central projections
- **THEN** active `get_model_metadata()` and `update_metadata()` dictionary contracts MUST be removed or replaced by typed contracts
- **AND** active callers MUST NOT retain wrapper round-trips through `modelspec.*`

#### Scenario: Legacy helper has only deprecated callers
- **WHEN** a broad flag-driven ModelSpec helper is no longer used by active code
- **THEN** deprecated/reference callers MUST NOT force preservation of that helper as an active public facade
