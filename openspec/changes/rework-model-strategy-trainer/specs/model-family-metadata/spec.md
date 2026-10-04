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

## ADDED Requirements

### Requirement: Runtime participant identity is projected from the accepted run authority
Metadata SHALL record durable projections of accepted participant,
relationship, transition, artifact, and lineage facts when required for
history or provenance. Metadata SHALL NOT establish live participant identity,
infer it from Python objects, or become the runtime binding authority.

#### Scenario: Accepted participant is recorded
- **WHEN** an accepted run participant needs durable metadata identity
- **THEN** metadata MUST consume the authority's logical run identity, authored address, incarnation discriminator, and applicable revisions
- **AND** it MUST preserve those facts separately from catalog, source, model-revision, realization, and artifact identities

#### Scenario: New run loads an artifact
- **WHEN** metadata records that a participant in run B derives from an artifact produced by run A
- **THEN** it MUST record lineage between distinct identities
- **AND** it MUST NOT reuse run A's participant identity merely because addresses or sources match

#### Scenario: Exact runtime restoration is recorded
- **WHEN** a snapshot restores the same logical run authority and participant references in a new execution session
- **THEN** metadata MUST preserve the logical run and participant incarnations and distinguish the new execution-session observation identity from the restored logical identity

### Requirement: Metadata keeps runtime and durable model identity meanings distinct
Metadata SHALL preserve available explicit associations between logical run,
participant, relationship, optimization-unit, execution-session/attempt,
catalog/source/model-revision, realization/composition, and artifact identities
when its consumers require them. Authored addresses, labels, roles, paths,
source equality, and Python object identity SHALL NOT substitute for an
authority-established runtime incarnation. Metadata SHALL validate and store
durable projections without retaining live execution handles as canonical
bindings or deciding current runtime readiness.

#### Scenario: Retired address is used by a new participant
- **WHEN** a retired participant's authored address is used for a new authority-established incarnation
- **THEN** metadata MUST preserve both identities and their applicable history
- **AND** matching addresses or sources MUST NOT overwrite the retired participant's identity

#### Scenario: One participant has several prepared views
- **WHEN** preparation exposes wrappers, backend replicas, or named routes for one accepted participant
- **THEN** metadata MUST retain the supplied participant-to-view/route associations where required
- **AND** it MUST NOT invent new participant incarnations from those physical objects

#### Scenario: Related student and teacher share an origin
- **WHEN** an accepted student and independently evolving teacher derive from the same source
- **THEN** metadata MUST retain their distinct participant identities and explicit lineage or operational relationships
- **AND** a shared role or source MUST NOT collapse them into one model participant

### Requirement: Metadata history preserves observed revisions and accepted outcomes
Required history SHALL retain the accepted source identity, relevant revision
or state-provenance scope, and outcome to which each observation belongs.
Successful accepted transitions SHALL remain distinct from failed, rejected,
stale, or unpublished attempts. Historical observations SHALL NOT be rewritten
as current state solely by ingestion order or wall-clock ordering. Consumers
SHALL use supplied revision/dependency correspondence or report ambiguity when
current meaning cannot be established. Ordinary mutable-state progression
SHALL NOT automatically be treated as a topology or binding revision.

#### Scenario: Compatible replacement preserves identity
- **WHEN** authority publication accepts a compatible binding replacement for an existing participant
- **THEN** metadata MUST preserve its participant incarnation while recording the supplied changed state/revision associations
- **AND** earlier accepted observations MUST retain their original meaning

#### Scenario: Older observation is ingested last
- **WHEN** an observation of revision 2 arrives after an accepted observation of revision 3 in the same ordered source scope
- **THEN** a consumer claiming latest accepted state MUST NOT select revision 2 solely because it was ingested last
- **AND** history MUST retain each observation's actual revision association

#### Scenario: Prepared replacement result is stale
- **WHEN** preparation or materialization completes but its result is rejected as stale and never published
- **THEN** any metadata filing MUST describe an attempt outcome rather than a successful current binding or composition transition

#### Scenario: Ordinary optimizer update changes captured state provenance
- **WHEN** an artifact captures weights after ordinary optimization without a topology or binding transition
- **THEN** metadata MUST retain the capture's supplied state provenance
- **AND** it MUST NOT manufacture a topology revision merely to distinguish the produced state

### Requirement: Artifact and restoration observations report their own achieved guarantees
Metadata SHALL distinguish descriptive product/export facts from actual
publication outcomes and preserve required product/member/resource scope,
captured state/provenance, dependencies, and local/remote/partial/uncertain
outcomes supplied by persistence coordination. Runtime snapshot capture,
publication, restoration readiness, and trained-product outcomes SHALL remain
distinct. A metadata-store snapshot SHALL NOT by itself establish an exact
same-run recovery guarantee or become the live runtime authority.

#### Scenario: Export facts exist before a failed write
- **WHEN** validated header/export facts are prepared and serialization subsequently fails
- **THEN** metadata MUST NOT treat those descriptive facts as proof of product publication or completion
- **AND** known partial resources and uncertain outcomes MUST remain distinguishable from completed resources

#### Scenario: Captured participant changes before reporting
- **WHEN** a product captured participant state at one accepted revision and that participant changes before metadata is filed
- **THEN** the product's provenance MUST retain the captured scope supplied by its result
- **AND** filing MUST NOT replace it with the later live participant state

#### Scenario: Local artifact exists while upload is pending
- **WHEN** persistence reports known local success and pending or failed remote publication
- **THEN** metadata MUST preserve both outcomes under the accepted completion policy rather than claim all destinations succeeded

#### Scenario: Metadata store is saved without required recovery state
- **WHEN** collected metadata records are saved but required optimization, input, or operation continuation state is absent
- **THEN** metadata MUST NOT report a complete runtime-recovery snapshot merely because its own storage succeeded

### Requirement: Metadata delivery cannot change origin-owned truth
Metadata validation, ingestion, export, or retry SHALL NOT establish live
runtime identity, retry training work, or roll back a reached origin-owned
transition or result. Required history/provenance/product/restoration facts
SHALL follow their accepted coverage and durability requirements rather than
being silently treated as optional droppable telemetry. Delivery failure SHALL
remain distinguishable from origin failure, and retry SHALL preserve fact
identity without creating another semantic transition. Where required coverage
is unavailable, consumers SHALL report incompleteness rather than infer absent
work or synthesize missing identity and outcome evidence.

#### Scenario: Metadata filing fails after a binding was accepted
- **WHEN** an accepted binding transition reaches its outcome but its durable observation cannot be filed
- **THEN** metadata failure MUST NOT erase or reverse that accepted transition
- **AND** required history coverage MUST be reported unavailable until an accepted delivery/recovery policy establishes it

#### Scenario: Required artifact provenance cannot be durably published
- **WHEN** the accepted product policy requires durable provenance and that filing fails after a resource write
- **THEN** the product MUST NOT claim complete required coverage
- **AND** result reporting MUST preserve its known resource-write outcome as reached rather than rolled back or never attempted

#### Scenario: One accepted transition is delivered again
- **WHEN** a durable filing is retried for the same accepted transition
- **THEN** consumers MUST NOT interpret that delivery as a second binding transition or a new incarnation
