## MODIFIED Requirements

### Requirement: Model loading returns the loaded-component surface
The model-loading boundary SHALL produce one shared typed loading result
containing the model version, family-declared top-level component candidates,
ordered source observations, source selections, component-to-selection
bindings, loading decisions, transformations, and known limitations rather
than a fixed `text_encoders/vae/denoiser` tuple, evidence-free component tuple,
or family-specific provenance fields. The loaded-component surface SHALL remain the authoritative top-level
description of what the loader produced, while the accepted run authority
alone SHALL validate and publish its candidates as current participant
bindings. Loading evidence SHALL NOT become a competing canonical live-binding
collection.

#### Scenario: Loading a training model
- **WHEN** a model loader produces a training-capable model family's components
- **THEN** it MUST preserve declared component keys, order, public labels, generic semantics, candidate loaded values, and the successful source/materialization evidence known at that boundary
- **AND** those candidates MUST enter runtime use through accepted participant materialization

#### Scenario: Families have different source layouts
- **WHEN** SD, SDXL, SD3, or a future family loads a different number or arrangement of sources and components
- **THEN** each family MUST populate the same typed loading-result collections
- **AND** central consumers MUST NOT require family-specific source fields or family-name branches

#### Scenario: Trainer stores the primary loaded-component state
- **WHEN** Trainer setup coordinates loading for an accepted arrangement
- **THEN** current participant bindings MUST be owned by the accepted run authority
- **AND** Trainer MUST NOT retain a competing authoritative loaded-component collection or fixed family slots
- **AND** the accepted arrangement or its scoped loading/provenance projection MUST retain the typed evidence required for catalog resolution and realization-composition filing

#### Scenario: Deferred loading updates components
- **WHEN** a deferred materialization boundary changes component presence or component-to-selection binding
- **THEN** it MUST return a typed loading result or update that preserves prior loading evidence and reports the new candidate and materialization evidence
- **AND** the candidate MUST become current only through an accepted participant transition rather than direct replacement of a Trainer-owned component collection

### Requirement: Top-level components remain the highest generic control surface
Family-declared top-level components SHALL remain the highest generic
model-loading and model-substructure surface. Accepted run participants SHALL
also support independently managed state outside one source model without
forcing that state into family component declarations. Lower-level targeting
SHALL preserve both its model-component provenance and its authority-qualified
participant context.

#### Scenario: Expanding to lower-level targets
- **WHEN** optimization or adapter behavior needs module-level or parameter-level targets within a loaded model component
- **THEN** expansion MUST preserve the declared top-level component identity and provenance
- **AND** the loaded-component contract MUST remain top-level rather than becoming a live module registry

#### Scenario: Avoiding slot-based control surfaces
- **WHEN** a future family exposes components outside the current diffusion tuple
- **THEN** generic runtime code MUST consume accepted participant and component projections without adding universal family slot fields

#### Scenario: Additional run participant is not one loaded-model component
- **WHEN** a run declares an independently managed adapter, teacher, reward model, or other participant alongside a loaded model
- **THEN** the accepted arrangement MUST represent it without redefining the source model family's component inventory
