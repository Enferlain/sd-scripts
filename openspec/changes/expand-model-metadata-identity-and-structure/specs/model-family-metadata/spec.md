## MODIFIED Requirements

### Requirement: Family declaration, model realization, and artifact facts remain distinct
The metadata system SHALL distinguish static family-declared topology, portable catalog lineage/revision facts, serialized source representations, selected state within those representations, the model realization observed in one run, successful revisioned composition observations for that realization, materialization-attempt events, and metadata describing a produced artifact.

#### Scenario: Run loads a model family
- **WHEN** initial model loading completes for a training run
- **THEN** the system MUST record a run-scoped model realization derived from the selected family/version
- **AND** that realization MUST reference available accepted catalog/revision/source-representation/source-selection identities without becoming any of those identities or an artifact record
- **AND** its initial component composition MUST be recorded as composition revision 1

#### Scenario: Realization composition changes after initial filing
- **WHEN** deferred loading, component replacement, or another materialization boundary changes the realization composition
- **THEN** the system MUST append a new ordered composition observation under the stable realization identity
- **AND** it MUST NOT mutate the earlier accepted observation

#### Scenario: Realization materialization fails
- **WHEN** a deferred load or replacement fails without producing a newly observed component surface
- **THEN** the failure MAY be recorded as a materialization-attempt event
- **AND** it MUST NOT become a realization-composition revision

#### Scenario: Realization becomes ready for training
- **WHEN** model preparation finishes and establishes the composition used for training
- **THEN** the system MUST append an explicit final composition observation
- **AND** artifact and structural consumers requiring final state MUST be able to reference that final observation

#### Scenario: One realization produces different artifact roles
- **WHEN** one loaded realization produces adapter and full-model artifacts whose external metadata differs
- **THEN** each artifact MUST receive artifact-specific model facts and applicable catalog revision/lineage facts
- **AND** the underlying realization identity and revisioned composition history MUST remain unchanged

### Requirement: Model realization facts use the shared metadata runtime path
The active runtime SHALL file low-volume model-realization, successful revisioned composition, materialization-attempt, component, source-provenance, and optional family-contribution facts through the existing metadata registry, runtime, emitter, and backend flow after applicable catalog/source-selection identity has been resolved by the durable catalog authority.

#### Scenario: Initial model loading completes
- **WHEN** the trainer has a stable run identity, model realization, loaded-component surface, and typed loading evidence
- **THEN** it MUST resolve available catalog/source identity through the central catalog authority
- **AND** it MUST file the corresponding accepted typed items and optional family contribution through `MetadataRuntime.file(...)` or its synchronous batch equivalent
- **AND** domain code MUST NOT call metadata storage directly

#### Scenario: Model preparation finalizes composition
- **WHEN** deferred materialization and trainable preparation establish final top-level composition
- **THEN** the trainer MUST file the final composition observation through the same accepted-item runtime path
- **AND** it MUST retain the accepted realization/composition identities for later artifact and structural references

#### Scenario: New model fact type is accepted
- **WHEN** a new shared model catalog, provenance, composition, or structure metadata item is introduced
- **THEN** its accepted type and emitter route MUST be registered in the central metadata registry
- **AND** runtime or validation code MUST NOT add a parallel supported-type list or routing chain
