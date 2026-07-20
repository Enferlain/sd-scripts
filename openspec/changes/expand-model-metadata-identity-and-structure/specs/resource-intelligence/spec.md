## ADDED Requirements

### Requirement: Structural model resources use accepted qualified owners
Resource intelligence SHALL relate model-structural quantities to accepted qualified model/component identities, revision- or realization-scoped structural path bindings, or observation-scoped live-object/storage-group identities when those identities are available.

#### Scenario: Component structural bytes are recorded
- **WHEN** resource intelligence records parameter or buffer bytes for a cataloged model component
- **THEN** the structural resource fact MUST reference the accepted qualified component identity
- **AND** it MUST preserve the inventory/descriptor evidence from which the quantity was derived

#### Scenario: Tensor structural bytes are recorded
- **WHEN** resource intelligence records a size for an accepted tensor descriptor
- **THEN** the resource fact MUST reference the applicable qualified path-binding identity and observation-local object/storage evidence when relevant
- **AND** it MUST preserve whether the quantity is structural, estimated, measured, or derived

#### Scenario: Qualified owner is unavailable
- **WHEN** a legacy or early runtime observation has only a family-local component key
- **THEN** resource filing MUST omit the unavailable qualified structural relationship or use an explicitly transitional unresolved owner fact
- **AND** it MUST NOT claim that the bare key is globally unique

### Requirement: Resource queries do not trigger unbounded structural inspection
Resource consumers SHALL use the shared metadata query cost/source policy when requesting model structural descriptors.

#### Scenario: Accepted descriptor already exists
- **WHEN** resource accounting requests immutable component/tensor size evidence already accepted for the exact revision and scope
- **THEN** it MUST reuse that evidence without re-enumerating unchanged model state

#### Scenario: Expensive inventory is absent
- **WHEN** resource analysis would require a full live or artifact inventory that is not present
- **THEN** it MUST request that resolution under an explicit allowed cost/scope policy or remain incomplete
- **AND** ordinary monitoring MUST NOT trigger unbounded inspection implicitly
