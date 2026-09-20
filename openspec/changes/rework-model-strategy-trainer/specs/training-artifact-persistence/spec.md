## Purpose

Define trained artifacts as semantic products of coherent accepted state while
keeping publication, physical resources, identity, lineage, and restoration
distinct.

## ADDED Requirements

### Requirement: Artifact persistence has declaration, request, plan, and result stages
Persistence SHALL distinguish the products an accepted arrangement supports,
the lifecycle-specific request made by Trainer, the resolved semantic plan,
and the actual result returned after writing or publication.

#### Scenario: Trainer requests a declared product
- **WHEN** a persistence trigger requests an artifact supported by the accepted arrangement
- **THEN** the capability MUST resolve an artifact plan before physical writing begins

#### Scenario: Unsupported product is requested
- **WHEN** a request names a product or representation not declared by the accepted arrangement
- **THEN** persistence MUST reject it before selecting live objects or creating output resources

### Requirement: Artifact plans select semantic state
An artifact plan SHALL select participant and relationship coverage,
dependencies, semantic transformations, expected members, representation,
packaging expectations, and consistency requirements from a coherent accepted
state projection. It SHALL NOT select state by incidental Python object or
wrapper identity.

#### Scenario: Adapter product spans shared and local state
- **WHEN** an adapter product contains shared banks and target-local state
- **THEN** the plan MUST select the adapter participant and relevant relationships according to product semantics
- **AND** it MUST NOT infer ownership from the current host wrapper

#### Scenario: Full-model product uses several access views
- **WHEN** a family artifact requires state from several participants and unwrapped views
- **THEN** the plan MUST preserve their semantic coverage and accepted consistency boundary

### Requirement: Product, member, and physical resource are separate
An artifact product SHALL contain semantic members, and members SHALL be backed
by one or more physical resources. These levels SHALL NOT be assumed to have
one-to-one cardinality. Packaging labels SHALL NOT imply semantic completeness.

#### Scenario: One member is sharded
- **WHEN** one semantic member is written across several files or blobs
- **THEN** the result MUST identify all backing resources without inventing additional semantic members

#### Scenario: One resource contains several participants
- **WHEN** a checkpoint file encodes state from several participants
- **THEN** the result MUST preserve the separate semantic member coverage despite one physical file

### Requirement: Artifact dimensions remain orthogonal
Semantic coverage, dependency relationships, semantic transformations,
external representation, physical packaging, and consistency SHALL be
represented independently. A product MAY contain embedded state, delta state,
and external references together.

#### Scenario: Adapter delta embeds an auxiliary component
- **WHEN** a product contains adapter delta state, embeds one auxiliary member, and references an external base
- **THEN** its plan and result MUST express all three dependency forms without one misleading completeness flag

### Requirement: Persistence uses one accepted boundary
Every product member SHALL correspond to one accepted arrangement and one
Trainer-established persistence boundary while satisfying each contributor's
declared freshness and consistency relationship. A universal revision counter
SHALL NOT be required when explicit contributor relationships establish
coherence.

#### Scenario: Frozen dependency and updated adapter are persisted together
- **WHEN** a frozen base reference and freshly advanced adapter form one coherent product
- **THEN** the plan MAY include both if their accepted dependency and freshness semantics hold at the boundary

### Requirement: Semantic transformations are declared before serialization
Transformations that change coverage, dependencies, lineage, or realization
meaning SHALL be declared by the product capability. Domain serializers SHALL
perform resolved mechanical conversion and writing but SHALL NOT infer a
semantic merge, delta, pruning, or quantization from incidental runtime state.

#### Scenario: Merged adapter product is requested
- **WHEN** persistence produces a host realization with adapter state folded into it
- **THEN** the product plan MUST declare the merge/fold transformation and resulting lineage before serializer mechanics run

### Requirement: Artifact results report actual output
The artifact result SHALL report actual members and resources, formats, sizes,
checksums, references, omissions, partial status, and failures. Post-write facts
SHALL NOT be guessed from the plan.

#### Scenario: Optional sidecar is not emitted
- **WHEN** a planned optional member is omitted or fails
- **THEN** the result MUST record its actual status and the resources that were successfully produced

### Requirement: Artifacts have identity distinct from participants
An artifact and any immutable model revision it represents SHALL have identity
distinct from live run participants. The result SHALL preserve lineage to the
accepted state it captured without claiming that a later run contains the same
participant incarnation.

#### Scenario: Artifact starts a new run
- **WHEN** another run loads a prior product
- **THEN** the new run MUST create new participant identities
- **AND** the artifact relationship MUST preserve provenance to the source state

### Requirement: Runtime restoration is not trained-artifact persistence
Runtime snapshots SHALL restore execution, optimization, progress, backend,
identity/revision, and accepted capability continuation state. They MAY share
timing, consistency, storage, and writing infrastructure with trained artifacts
but SHALL NOT be one request type differentiated only by a product kind.

#### Scenario: Trainer saves both product and resume state
- **WHEN** one lifecycle boundary emits a trained artifact and a runtime snapshot
- **THEN** each MUST use its own declared semantics and result
- **AND** successful artifact publication MUST NOT imply that exact runtime restoration is possible
