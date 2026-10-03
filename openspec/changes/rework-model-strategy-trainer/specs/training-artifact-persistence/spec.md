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
The declaration and plan SHALL identify the selected state variant and
required versus optional constituents. Selection SHALL NOT be inferred from
trainability or require every component used during training to be exported.

#### Scenario: Adapter product spans shared and local state
- **WHEN** an adapter product contains shared banks and target-local state
- **THEN** the plan MUST select the adapter participant and relevant relationships according to product semantics
- **AND** it MUST NOT infer ownership from the current host wrapper

#### Scenario: Full-model product uses several access views
- **WHEN** a family artifact requires state from several participants and unwrapped views
- **THEN** the plan MUST preserve their semantic coverage and accepted consistency boundary

#### Scenario: Student or EMA autoencoder is exported
- **WHEN** a declared product selects a student or an EMA autoencoder while teacher, discriminator, and optimizer state are training-only
- **THEN** the plan MUST select that state variant and its required constituents without including training-only state by default
- **AND** the result MUST identify the state actually captured rather than substitute another available variant

### Requirement: Products preserve the information required for their declared use
A product SHALL embed or explicitly reference the state and information
required for its declared use, including relevant non-parameter state,
construction, representation, installation, and implementation dependencies.
Implementation dependencies SHALL describe requirements relevant to
reconstruction/use, not automatically the training Python class/object
identity, wrapper, or exact compiled artifact.
Those requirements SHALL come from accepted product behavior rather than a
universal model layout or an assumption that a state dictionary is sufficient.
Capability-owned contributions SHALL remain domain-owned and need not become
separate participants. Consumer dependencies, training/preparation dependencies,
and historical provenance SHALL remain distinguishable. External references
SHALL identify the required state and compatibility conditions; a mutable
load path alone SHALL NOT establish them.

#### Scenario: Autoencoder requires more than weights
- **WHEN** the declared representation requires a codebook or normalization statistics and packing/inverse-transform information
- **THEN** the product MUST preserve or explicitly reference those requirements alongside the selected model state
- **AND** missing required information MUST prevent a claim of completeness for that use

#### Scenario: Control product requires installation information
- **WHEN** a control or adapter product requires target mappings and architecture configuration to reconstruct its installation
- **THEN** the plan and result MUST preserve that information and its compatible host dependencies
- **AND** they MUST NOT infer installation meaning solely from tensor key prefixes

#### Scenario: Consumer uses another compatible implementation
- **WHEN** a supported consumer satisfies the declared architecture, configuration, operation, and representation requirements through another compatible implementation
- **THEN** product usability MUST NOT require the training Python object identity, wrapper, or compiled artifact unless an explicit reconstruction/use requirement makes that necessary

#### Scenario: Preparation dependency is not an inference dependency
- **WHEN** a teacher or tokenizer used during training is unnecessary for the product's declared inference use
- **THEN** it MAY be retained as provenance without being declared a required inference dependency or embedded member

#### Scenario: Base and adapter both changed during training
- **WHEN** an adapter-only product requires the updated base state for its declared use
- **THEN** its external dependency MUST identify compatible updated base state, which MAY be supplied by a separately declared product
- **AND** the product MUST NOT claim to contain the base updates or silently identify the original loaded base as equivalent
- **AND** a product that also embeds base updates MUST declare that expanded coverage rather than retain an adapter-only claim

### Requirement: Product, member, and physical resource are separate
An artifact product SHALL contain semantic members, and members SHALL be backed
by one or more physical resources. These levels SHALL NOT be assumed to have
one-to-one cardinality. Packaging labels SHALL NOT imply semantic completeness.

#### Scenario: One member is sharded
- **WHEN** one semantic member is written across several files or blobs
- **THEN** the result MUST identify all backing resources without inventing additional semantic members
- **AND** missing required shards MUST prevent a claim that the member is complete

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

### Requirement: Artifact products are independent from training treatment
The products an accepted arrangement supports and their semantic coverage
SHALL come from explicit product declarations. Persistence SHALL NOT infer or
restrict the product set solely from whether the run trains existing
participant parameters, PEFT state, or a contract-compatible combination.

#### Scenario: Direct parameter training declares a full-model product
- **WHEN** an accepted arrangement trains selected parameters of existing participants and declares a compatible full-model product
- **THEN** persistence MUST resolve that product from its declared semantic coverage and dependencies
- **AND** it MUST NOT require a separate fine-tune mode classification

#### Scenario: PEFT training declares its products explicitly
- **WHEN** an accepted arrangement trains PEFT state alone or together with existing participant parameters
- **THEN** adapter, merged full-model, or other product support MUST follow the arrangement's accepted product declarations
- **AND** persistence MUST NOT infer one mandatory product category solely from the selected training subjects

### Requirement: Persistence uses one accepted boundary
Every product member SHALL correspond to one accepted arrangement and one
Trainer-established persistence boundary while satisfying each contributor's
declared freshness and consistency relationship. A universal revision counter
SHALL NOT be required when explicit contributor relationships establish
coherence.

#### Scenario: Frozen dependency and updated adapter are persisted together
- **WHEN** a frozen base reference and freshly advanced adapter form one coherent product
- **THEN** the plan MAY include both if their accepted dependency and freshness semantics hold at the boundary

### Requirement: Capture preserves source meaning across later run changes
Persistence SHALL establish a stable product-specific capture under the
accepted source dependencies and consistency obligations before permitting
conflicting live changes. Source access SHALL remain protected until required
reads actually complete, including outstanding backend work. A borrowed live
object or tensor reference SHALL NOT by itself prove stability. The captured
state, topology, relationships, dependencies, state variant, and lineage SHALL
remain authoritative for the product through later serialization/publication.
A later live transition SHALL NOT relabel that capture or invalidate historical
publication solely because the run changed, unless the accepted request
requires a stronger condition such as latest-state publication.

#### Scenario: Topology changes while an older capture is published
- **WHEN** a stable dense-model capture is still being serialized or uploaded after the live arrangement transitions to a MoE topology
- **THEN** the result MUST describe the captured dense topology and its construction meaning
- **AND** publication MAY finish under the accepted historical-product policy without claiming to describe the current MoE state

#### Scenario: Live tensors are not protected
- **WHEN** export borrows tensors that can be updated before its required reads complete
- **THEN** readiness MUST remain unsatisfied unless the accepted capture mechanism prevents an incoherent product

### Requirement: Semantic transformations are declared before serialization
Transformations that change coverage, dependencies, lineage, or realization
meaning SHALL be declared by the product capability. Domain serializers SHALL
perform resolved mechanical conversion and writing but SHALL NOT infer a
semantic merge, delta, pruning, or quantization from incidental runtime state.
Product compatibility SHALL be evaluated under the same accepted obligations
at the earliest authoritative evidence point. Training compatibility SHALL NOT
be treated as evidence that every export transformation is compatible.

#### Scenario: Merged adapter product is requested
- **WHEN** persistence produces a host realization with adapter state folded into it
- **THEN** the product plan MUST declare the merge/fold transformation and resulting lineage before serializer mechanics run

#### Scenario: Training works but low-precision merge is unsupported
- **WHEN** separate adapter execution is compatible with the loaded low-precision base but the requested merge violates accepted product requirements
- **THEN** the merge MUST be rejected at the earliest authoritative evidence point
- **AND** rejection MUST NOT imply that a separately declared compatible adapter product is unsupported

### Requirement: Export does not grant live-state mutation authority
Product transformations and serializers SHALL operate within their accepted
authority. Export SHALL NOT silently change canonical participant state,
bindings, relationships, or runtime modes. Any temporary live-state effects
SHALL be explicitly permitted and coordinated against conflicting use, with
restoration before affected use resumes. Canonical run changes SHALL use the
existing authority-governed transition path. Cleanup failure SHALL gate unsafe
use without erasing known product output or implying rollback.

#### Scenario: Serializer changes live dtype or configuration
- **WHEN** a selected export implementation would change a live module's dtype or configuration
- **THEN** it MUST use a protected independent representation or an explicitly accepted temporary-effect protocol
- **AND** it MUST NOT leave the running arrangement silently changed

#### Scenario: Export completes but temporary-state restoration fails
- **WHEN** artifact writing reaches a known result and cleanup cannot restore the required live execution state
- **THEN** the artifact outcome MUST remain reportable and the affected live use MUST be stopped or gated
- **AND** a written artifact MUST NOT be interpreted as proof that training can safely resume

### Requirement: Artifact results report actual output
The artifact result SHALL report actual members and resources, formats, sizes,
checksums, references, omissions, partial status, and failures. Post-write facts
SHALL NOT be guessed from the plan.
Writing, local resource completion, requested remote publication, and
observation SHALL have distinguishable outcomes where applicable. Complete
product status SHALL require every constituent and dependency required by the
accepted completion policy. Required external dependencies SHALL satisfy
that policy through its specified embedded state, available/resolvable resource,
or stable compatible external reference. A policy MAY accept a valid external
reference without current reachability. Product completion SHALL NOT by itself
claim self-containment, dependency availability for current use, or continued
availability of externally referenced resources.
Pending or uncertain work SHALL NOT be reported
as successful completion. Known partial output SHALL remain reportable;
an exception or absent return value SHALL NOT imply that no resources exist.
Reported checksums SHALL identify their algorithm and measured scope rather
than equate a model/tensor hash with a whole-resource checksum.

#### Scenario: Optional sidecar is not emitted
- **WHEN** a planned optional member is omitted or fails
- **THEN** the result MUST record its actual status and the resources that were successfully produced

#### Scenario: Reference-only base dependency is currently unreachable
- **WHEN** an adapter product has all other required output complete and a stable compatible external base reference satisfying its accepted reference-only completion policy, but the base is currently unreachable
- **THEN** the product MAY be reported complete without embedding or currently fetching that base
- **AND** the result MUST preserve the external dependency without claiming self-containment or availability for current use

#### Scenario: Completion policy requires an available dependency
- **WHEN** an accepted product completion policy requires a resolvable dependency resource and that requirement is not established
- **THEN** a reference alone MUST NOT satisfy completion
- **AND** the result MUST preserve known local output separately from the unmet dependency requirement

#### Scenario: Later required write fails
- **WHEN** some resources are written and a later required member or manifest fails
- **THEN** the result MUST report known resources and the incomplete product without assuming rollback or atomic multi-resource writing

#### Scenario: Local writing succeeds but remote publication is pending or fails
- **WHEN** local resources are complete but requested remote publication has not succeeded
- **THEN** the result MUST preserve local completion separately from the pending, uncertain, or failed remote outcome
- **AND** it MUST NOT claim the requested publication is complete

#### Scenario: Reporting fails after publication
- **WHEN** publication succeeds but a metadata or observation sink fails
- **THEN** the actual product result MUST remain authoritative without claiming successful reporting

### Requirement: Retention respects resource membership and outstanding work
Persistence infrastructure SHALL use actual member/resource associations for
retention and cleanup, respecting shared resources and outstanding accepted
reads or publication work. It SHALL NOT delete or overwrite resources still
required by that work. This SHALL NOT imply permanent retention or guaranteed
availability of every external dependency.

#### Scenario: Retention runs during asynchronous upload
- **WHEN** an accepted upload is still reading files belonging to a product selected for retention cleanup
- **THEN** cleanup MUST preserve required resource access until that work finishes or its accepted cancellation protocol safely releases access

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
identity/revision, accepted operation-owned state, and accepted capability
continuation state. They MAY share timing, consistency, storage, and writing
infrastructure with trained artifacts but SHALL NOT be one request type
differentiated only by a product kind.

#### Scenario: Trainer saves both product and resume state
- **WHEN** one lifecycle boundary emits a trained artifact and a runtime snapshot
- **THEN** each MUST use its own declared semantics and result
- **AND** successful artifact publication MUST NOT imply that exact runtime restoration is possible

#### Scenario: Snapshot succeeds but a product member fails
- **WHEN** a coherent runtime snapshot completes and an independently requested trained product lacks a required member
- **THEN** the snapshot outcome MUST remain independent and MUST NOT upgrade the incomplete product to success
