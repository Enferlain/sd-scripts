## MODIFIED Requirements

### Requirement: Family declaration, model realization, and artifact facts remain distinct
The metadata system SHALL distinguish static family-declared topology, portable
catalog lineage/revision facts, serialized source representations, selected
state within those representations, run-scoped model realization, successful
revisioned composition observations, materialization/publication attempts, and
produced-artifact facts. Accepted composition history SHALL derive from
authority-published state, not merely loader success or candidate observation.
Catalog, realization, composition, and runtime participant identities SHALL
remain distinct and explicitly associated where required.

#### Scenario: Run loads a model family
- **WHEN** initial loading candidates are accepted and published for a training run
- **THEN** the system MUST record a run-scoped model realization derived from the selected family/version
- **AND** that realization MUST preserve family-declared topology and reference available accepted catalog/revision/source-representation/source-selection identities without becoming any of those identities or an artifact record
- **AND** its initial accepted component composition MUST be recorded as composition revision 1

#### Scenario: Realization composition changes after initial filing
- **WHEN** authority publication accepts deferred loading, replacement, or another transition changing the realization composition
- **THEN** the system MUST append a new ordered composition observation under the stable realization identity
- **AND** it MUST preserve the originating accepted state/revision correspondence without mutating earlier accepted observations

#### Scenario: Realization materialization fails
- **WHEN** a deferred load or replacement fails without accepted publication
- **THEN** the failure MAY be recorded as a materialization-attempt event
- **AND** it MUST NOT become an accepted realization-composition revision

#### Scenario: Loaded candidate is rejected before publication
- **WHEN** a loader successfully produces a candidate but authority publication rejects it as stale or incompatible
- **THEN** metadata MAY retain its loading evidence and attempt outcome
- **AND** it MUST NOT record the candidate as an accepted composition transition or current participant binding

#### Scenario: Realization becomes ready for training
- **WHEN** accepted preparation publication establishes the composition used at a named training readiness checkpoint
- **THEN** the system MUST append an explicit finalized composition observation for that checkpoint
- **AND** artifact and structural consumers requiring that finalized context MUST be able to reference the corresponding observation
- **AND** finalization MUST NOT prohibit later accepted composition changes or make an older checkpoint's observation current for a newer checkpoint

#### Scenario: One realization produces different artifact roles
- **WHEN** one accepted realization produces adapter and full-model artifacts whose external metadata differs
- **THEN** each artifact MUST receive artifact-specific model facts and applicable catalog revision/lineage facts tied to its actual captured composition
- **AND** producing those artifacts MUST NOT by itself change the underlying realization identity or rewrite its composition history

### Requirement: Model realization facts use the shared metadata runtime path
The active runtime SHALL file low-volume model-realization, successful
revisioned composition, materialization/publication-attempt, component,
source-provenance, and optional family-contribution facts through the existing
metadata registry, runtime, emitter, and backend flow. Applicable
catalog/source-selection identity SHALL be resolved by the durable catalog
authority before dependent filing under the accepted metadata policy. This
filing requirement SHALL NOT make catalog resolution a prerequisite for an
otherwise valid live consumer projection, and neither resolution nor filing
SHALL establish current participant bindings. Required durable coverage and
optional observation delivery SHALL remain distinct from origin-owned outcomes.

#### Scenario: Model loading completes
- **WHEN** loading candidates have become accepted state and run coordination has stable logical run/model identity and a scoped component view with typed loading evidence
- **THEN** it MUST resolve applicable catalog/source identity through the central catalog authority under the accepted policy
- **AND** it MUST file the corresponding accepted typed items and optional family contribution through `MetadataRuntime.file(...)` or its synchronous batch equivalent
- **AND** domain code MUST NOT call metadata storage or backend APIs directly
- **AND** filing MUST NOT require a competing Trainer-owned loaded-component collection

#### Scenario: Model preparation finalizes composition
- **WHEN** accepted preparation publication establishes top-level composition for a named readiness checkpoint
- **THEN** run coordination MUST file that checkpoint's finalized composition observation through the same accepted-item runtime path
- **AND** the responsible runtime owner MUST retain accepted realization/composition associations for later artifact and structural references without making them canonical live bindings

#### Scenario: New model fact type is accepted
- **WHEN** a new shared model catalog, provenance, composition, or structure metadata item is introduced
- **THEN** its accepted type and emitter route MUST be registered in the central metadata registry
- **AND** runtime or validation code MUST NOT add a parallel supported-type list or routing chain

#### Scenario: Required composition filing fails after publication
- **WHEN** authority publication reaches an accepted transition but required durable composition filing fails
- **THEN** the accepted runtime transition MUST remain reached and required history coverage MUST be reported incomplete under its delivery/recovery policy
- **AND** retrying the filing MUST NOT publish the transition again or create a second semantic composition change
