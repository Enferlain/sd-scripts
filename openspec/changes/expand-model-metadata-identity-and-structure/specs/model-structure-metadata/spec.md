## ADDED Requirements

### Requirement: Model structure separates path bindings, live objects, storage, and source keys
The metadata system SHALL represent revision-qualified and realization-qualified component/module/parameter/buffer path bindings as separate typed identities while keeping observation-local live-object, storage-group, and serialized source-key identities distinct.

#### Scenario: Same local path exists in two components
- **WHEN** two components each contain a tensor path such as `weight`
- **THEN** their structural path-binding identities MUST remain distinct through owner and component qualification
- **AND** the component-local path MUST remain a queryable fact

#### Scenario: Revision and realization expose the same path
- **WHEN** one accepted revision path is materialized in a run realization
- **THEN** the revision-qualified reusable binding and realization-qualified runtime binding MUST remain distinct typed identities
- **AND** an explicit relationship MAY connect them when materialization evidence supports it

#### Scenario: Live object is replaced between runs
- **WHEN** the same catalog revision is materialized as different Python objects in later runs
- **THEN** durable structural identity MUST NOT depend on object identity or pointer values
- **AND** observation-local object/storage evidence MUST remain separately scoped

#### Scenario: Several paths name one live object
- **WHEN** one live parameter or tensor object is reachable through multiple qualified paths
- **THEN** every path MUST remain a distinct binding identity
- **AND** the live observation MAY relate those bindings to one observation-scoped object identity

#### Scenario: Shared storage is observed
- **WHEN** multiple live tensor names share storage and the source supports detecting it
- **THEN** the observation MAY assign an observation-scoped storage-group identity
- **AND** shared storage MUST NOT cause the tensor names to collapse into one structural identity

#### Scenario: Source/runtime mapping is non-bijective
- **WHEN** conversion maps one serialized source key to several runtime bindings or several source keys to one runtime binding
- **THEN** the metadata graph MUST preserve the one-to-many or many-to-one mapping explicitly
- **AND** it MUST NOT collapse source-key identity, path-binding identity, live-object identity, or storage identity

### Requirement: Structural inventories declare dimension-scoped coverage
Every structural inventory SHALL declare its owner, source type, included kinds, path/traversal policy, derivation version, cost class, and a typed collection of coverage claims whose dimension/scope, status, evidence basis, and omissions can vary independently.

#### Scenario: Live topology and named state are complete
- **WHEN** a resolver claims complete live-topology and named-state-member coverage for the selected scope
- **THEN** it MUST account for modules, parameters, persistent buffers, and non-persistent buffers under its declared traversal policy
- **AND** it MUST NOT silently omit an applicable category

#### Scenario: Live source-key mapping is unavailable
- **WHEN** live topology/state coverage is complete but the live object does not expose its originating serialized-key mapping
- **THEN** topology/state-member coverage MUST remain complete
- **AND** source-key-mapping coverage MUST be separately marked `unavailable` or `unknown`

#### Scenario: Selected artifact-key coverage is complete
- **WHEN** a resolver claims complete serialized-key coverage for selected safetensors files
- **THEN** it MUST account for every key in those selected files
- **AND** repository-wide key coverage and non-persistent live-buffer coverage MUST remain separate unavailable/unknown claims unless independently observed

#### Scenario: One coverage dimension degrades
- **WHEN** a resolver cannot prove complete coverage for one declared dimension
- **THEN** that claim MUST be marked `partial`, `unavailable`, or `unknown`
- **AND** it MUST preserve the reason and omissions without downgrading unrelated truthful coverage claims

#### Scenario: Complete coverage claim is internally inconsistent
- **WHEN** a coverage claim marked `complete` omits a member required by its dimension/scope contract or contradicts membership, traversal, or limitation facts
- **THEN** metadata validation MUST reject that claim/inventory
- **AND** the producer MUST reclassify the affected dimension with an omission reason

### Requirement: Tensor descriptors are typed and selectable
The model structure capability SHALL provide typed descriptors at selectable scope and granularity rather than requiring an all-or-nothing text dump.

#### Scenario: Caller requests one tensor shape
- **WHEN** a caller requests the descriptor for one qualified tensor
- **THEN** the system MUST be able to return its shape, rank, dtype, ownership, descriptor source, and quality when available
- **AND** it MUST NOT require rendering or retaining a full parameter dump

#### Scenario: Caller requests component descriptor inventory
- **WHEN** a caller requests descriptor-level facts for one component
- **THEN** the resolver MUST be able to include paths, kinds, shapes, dtypes, trainability/persistence where applicable, element counts, and byte estimates
- **AND** it MUST preserve the applicable coverage claims and derivation policy

#### Scenario: Raw values are not requested
- **WHEN** metadata resolves structural or digest facts
- **THEN** it MUST NOT persist raw tensor contents as metadata
- **AND** content access MUST remain bounded to the explicitly selected resolver operation

### Requirement: Live and artifact resolvers preserve source limitations
Model structural resolvers SHALL support live modules and safe artifact metadata sources without pretending their observations are equivalent.

#### Scenario: Safetensors header is inspected
- **WHEN** a safetensors source is queried for keys, shapes, dtypes, or offsets
- **THEN** the resolver MUST be able to obtain those descriptors without loading every tensor value
- **AND** it MUST preserve artifact/header source evidence and limitations

#### Scenario: Live module is inspected
- **WHEN** a live model component is queried
- **THEN** the resolver MUST preserve the lifecycle phase, traversal/duplicate policy, parameter/buffer persistence, and runtime dtype/device observations
- **AND** it MUST distinguish runtime observations from immutable source facts

#### Scenario: Pickle-based checkpoint requires code execution
- **WHEN** structural inspection of a non-safetensors checkpoint cannot be performed through a trusted bounded path
- **THEN** the resolver MUST enforce the shared trust/security policy or decline the query
- **AND** it MUST NOT silently execute an untrusted artifact merely to obtain metadata

### Requirement: Model structural queries use the shared metadata query capability
Model descriptor lookup and optional resolution SHALL integrate with the system-wide typed metadata query contract for source policy, cost, ambiguity, freshness, coverage, and persistence.

#### Scenario: Valid recorded immutable descriptor exists
- **WHEN** a query finds an accepted descriptor whose owner evidence, policy version, and coverage claims satisfy the request
- **THEN** it MUST return that recorded fact without re-inspecting unchanged tensor content

#### Scenario: Requested descriptor is absent
- **WHEN** no accepted descriptor satisfies the query and policy permits an available resolver
- **THEN** the shared query system MAY invoke the model resolver with the declared cost/source scope
- **AND** a durable resolved result MAY be filed for later reuse

#### Scenario: Query dependency is unavailable
- **WHEN** the system-wide typed metadata query capability has not been implemented
- **THEN** the structural metadata milestones MUST remain incomplete
- **AND** model code MUST NOT introduce a parallel general-purpose query dispatcher

### Requirement: Structural fingerprints are versioned evidence
The model structure capability SHALL represent exact, structural, and normalized-state fingerprints as distinct versioned evidence products with explicit coverage and normalization.

#### Scenario: Exact artifact digest is calculated
- **WHEN** complete serialized bytes are hashed
- **THEN** the resulting evidence MUST identify the digest algorithm and exact representation scope
- **AND** it MUST NOT claim format-independent state equality

#### Scenario: Structural fingerprint is calculated
- **WHEN** an ordered descriptor inventory is fingerprinted
- **THEN** the evidence MUST identify included kinds, ordering, path normalization, descriptor fields, and scoped coverage claims
- **AND** it MUST remain distinguishable from a content fingerprint

#### Scenario: Canonical state fingerprint is experimental
- **WHEN** a normalized tensor-state fingerprint policy has not passed its declared equivalence fixtures
- **THEN** its result MAY support candidate matching
- **AND** it MUST NOT automatically merge catalog identities

#### Scenario: First canonical state fingerprint passes its fixtures
- **WHEN** the experimental canonical-state policy in this change passes its declared fixture matrix
- **THEN** it MUST remain candidate-only evidence for catalog recognition
- **AND** automatic normalized-equality or merge authority MUST require a later explicit accepted policy/spec change

#### Scenario: Optional or unknown keys occur
- **WHEN** a fingerprint source includes optional, unknown, or conversion-specific keys
- **THEN** the policy MUST record whether each category is included, mapped, or explicitly excluded
- **AND** it MUST NOT silently ignore the keys while claiming complete state coverage

### Requirement: Existing structural consumers converge on accepted identities
Parameter inspection, optimization targeting, resource accounting, and future analytics SHALL reuse accepted structural identity/descriptor semantics rather than maintaining incompatible identity schemes.

#### Scenario: Parameter dump is rendered
- **WHEN** a human-readable parameter dump is requested after structural metadata is available
- **THEN** it MUST be rendered from or aligned with accepted inventory/descriptor facts
- **AND** the text format MUST NOT become the canonical schema

#### Scenario: Resource fact describes tensor size
- **WHEN** resource intelligence records a structural byte quantity for a known model tensor or component
- **THEN** it MUST reference the accepted qualified structural owner identity
- **AND** the quantity MUST preserve whether it is measured, structural, estimated, or derived
