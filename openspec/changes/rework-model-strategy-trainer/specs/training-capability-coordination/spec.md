## Purpose

Define how the training pipeline selects, validates, schedules, and observes
named capabilities without turning them into optional method discovery.

## ADDED Requirements

### Requirement: Capabilities have named request and result semantics
Every Trainer-recognized capability SHALL define its selection, request,
readiness, result, lifecycle, state effects, external effects, and failure
semantics. Support SHALL be explicit rather than inferred from method presence,
`hasattr()`, family names, or raising defaults.

#### Scenario: Strategy provides validation
- **WHEN** an authored strategy selects a validation capability
- **THEN** the strategy acceptance check MUST record its accepted request/result and readiness contract
- **AND** the pipeline MUST NOT discover support by calling a method and waiting for `NotImplementedError`

#### Scenario: Repository gains another implementation
- **WHEN** the repository makes a new capability implementation available
- **THEN** an existing strategy MUST NOT provide or select it unless that strategy's authored definition does so explicitly

### Requirement: Capability selection is validated before use
An authored strategy SHALL explicitly provide every capability it exposes or
requires. Execution configuration MAY request an exposed capability within its
bounds. Missing or incompatible requested capabilities SHALL fail before their
runtime trigger.

#### Scenario: Full-model persistence is unavailable
- **WHEN** configuration requests a full-model product that the authored strategy does not provide
- **THEN** strategy fulfillment MUST reject the request when the configuration and incompatibility are already known
- **AND** if the contract deliberately permits that bounded request to be supplied later, request validation MUST reject it before the persistence capability becomes ready or any checkpoint resources are created

#### Scenario: Provided capability is not requested
- **WHEN** an accepted arrangement provides a sampling capability but the run's accepted configuration or policy does not request sampling
- **THEN** the pipeline MUST NOT schedule sampling merely because the capability is available

### Requirement: Capability coordination does not imply central implementation
Trainer or delegated pipeline orchestration SHALL own capability trigger
timing, request coordination, and result routing. Model-, objective-,
representation-, adapter-, or serializer-specific behavior MAY remain in
selected domain implementations behind the accepted capability exchange.

#### Scenario: Sampling is triggered
- **WHEN** the pipeline reaches an accepted sampling trigger
- **THEN** pipeline orchestration MUST issue the typed sampling request and coordinate its destination
- **AND** accepted generation behavior MUST perform model-specific inference without adding family branches to generic Trainer code

### Requirement: Capability readiness derives from accepted state
A capability SHALL execute only when its declared participant bindings,
relationships, routes, views, capability state, and consistency dependencies
are current. Readiness checkpoints SHALL describe accepted operation needs
rather than current phase-function names.

#### Scenario: Sampling route is stale
- **WHEN** a selected sampling capability depends on a route invalidated by replacement
- **THEN** the sampling request MUST remain unavailable until that route is refreshed or rebound

### Requirement: Caching separates semantics from storage orchestration
Representation and conditioning implementations SHALL own semantic cache
encoding, decoding, schema, and dependency meaning. Data/cache infrastructure
SHALL own traversal and storage coordination **for cache production**. A
selected live input provider MAY traverse available sources or cache records
under its accepted input policy; training input coordination SHALL still check
readiness and admission before handoff. Neither traversal owner SHALL silently
take over the other's continuation state. Cache results SHALL identify their
accepted dependencies and SHALL become stale when those guarantees fail.

#### Scenario: Autoencoder replacement invalidates latent cache
- **WHEN** accepted autoencoder state changes and a latent cache depends on that realization
- **THEN** the cache MUST no longer be treated as current
- **AND** unrelated conditioning cache MAY remain current only through an explicit valid dependency set

#### Scenario: Cache-backed input is produced asynchronously
- **WHEN** cache production writes a value while a live input provider selects work independently
- **THEN** cache coordination MUST publish readiness only after its accepted storage and dependency checks succeed
- **AND** input coordination MUST separately check whether that ready value is admissible for the selected work and current consumer
- **AND** neither a completed cache write nor a handed-off input may imply that a training action or optimizer unit advanced

### Requirement: Cache requests and results preserve the selected computation
A caching request SHALL identify the selected implementation and representation
boundary, requested work or production scope, relevant input/transformation
meaning, consumer schema, producer dependencies, reuse/consistency obligations,
storage support, and applicable lifecycle and failure policies. Cache
coordination SHALL derive requests and judge results under the existing
run-specific obligations; it SHALL NOT invent another acceptance policy.
Results SHALL remain associated with their request/attempt and distinguish
completed values and actual provenance, storage/readiness outcomes, incomplete
coverage, and failures or uncertainty. No universal image, latent, tokenizer,
tensor-axis, or file-per-sample fields SHALL be required by this exchange.

#### Scenario: One sample has several caption variants
- **WHEN** requested work selects a caption variant for a sample that already has another cached variant
- **THEN** cache lookup and the result MUST retain the requested variant and all relevant conditioning dependencies independently from sample identity
- **AND** a miss MUST follow an accepted wait, production, recompute, fallback, or failure policy rather than silently use the stored different caption

#### Scenario: Cached encoding does not finish the conditioning pipeline
- **WHEN** frozen upstream encoding is cached and a downstream conditioning adapter remains trainable
- **THEN** the cached value MUST supply the complete representation/schema needed by the selected live adapter computation
- **AND** the run MUST preserve that downstream computation and its required derivative path rather than cache or detach it implicitly

#### Scenario: Cache input transformation changes the training meaning
- **WHEN** reuse would freeze a requested stochastic transformation or replace exact caption encoding with an approximate representation edit
- **THEN** fulfillment or admission MUST reject that unaccepted change at its earliest authoritative evidence point
- **AND** such behavior MAY be used only when explicitly selected and accepted with its own semantics

### Requirement: Cache freshness follows actual computation dependencies
Cache evidence SHALL identify the relevant inputs, representation/schema, and
producer states actually used, with sufficient dependency and consistency
meaning to judge reuse under the current obligations. Evidence MAY use a
protected state/snapshot or structured provenance over portions of work where
accepted production permits it; provenance alone SHALL NOT authorize mixed
states. Unchanged participant identity, binding revision, a file path, or a
caption hash SHALL NOT substitute for missing producer-state evidence.
Relevant changes SHALL withdraw affected reuse guarantees before dependent
use. Unrelated changes MAY preserve reuse only with valid precise evidence.
Older-state values MAY remain stored or serve an explicitly accepted version
or lag policy, but SHALL NOT be treated as current under a failed guarantee.
Published availability SHALL NOT be a perpetual current-use guarantee. A query
claiming current usability SHALL evaluate current dependency evidence even
when no binding revision changed. Implementations MAY withdraw guarantees
proactively or re-evaluate at query/admission, but SHALL NOT report a failed
guarantee as current. Persistent reuse across runs SHALL be judged against the
current run's obligations and producer/representation correspondence, not
inherit prior-run readiness or participant identity.

#### Scenario: Encoder weights change without participant replacement
- **WHEN** an encoder's weights change while its participant and binding identity remain the same
- **THEN** values depending on those weights MUST be judged against their actual production-state provenance and the current reuse policy
- **AND** unchanged identity or prepared-route presence MUST NOT keep those values current by itself

#### Scenario: Only a downstream adapter changes
- **WHEN** a cached frozen encoder value has precise upstream dependencies that remain satisfied while a live downstream adapter advances
- **THEN** that cached upstream value MAY remain usable
- **AND** the adapter output MUST still be recomputed under the accepted live computation

#### Scenario: Another run reuses persistent cached values
- **WHEN** a new run finds values stored by an earlier run
- **THEN** reuse MUST establish that their actual input, producer-state, and representation evidence satisfy the new run's current obligations
- **AND** prior readiness or a matching authored participant address MUST NOT prove current compatibility or transfer the earlier participant's runtime identity

#### Scenario: Producer completes after its source state changes
- **WHEN** in-flight cache work finishes after a relevant producer dependency or request obligation changes
- **THEN** publication MUST check current obligations using actual production provenance rather than relabel the result with the launch or completion state
- **AND** consumer admission MUST reject or follow an explicitly accepted version/lag policy before use
- **AND** detached cached output MUST NOT silently replace a required live derivative path

### Requirement: Cache publication and consumer admission are distinct
Cache coordination SHALL publish readiness only for a complete usable value or
bundle whose payload, storage access/index, schema, and dependency evidence
satisfy its accepted consistency boundary. That boundary SHALL be explicit
without requiring one dataset-wide transaction. Published availability SHALL
NOT establish admission for another request or consumer. Input coordination
SHALL check exact work association, representation, freshness/lag, and gradient
requirements before handoff, and required source/storage guarantees SHALL
remain protected through dependent use. Results SHALL describe actual
publication and partial coverage rather than infer readiness from file
existence, queue position, or a successful encoding call.

#### Scenario: Payload write succeeds but publication fails
- **WHEN** payload writing succeeds but its required index, locator, or readiness publication fails
- **THEN** the result MUST report incomplete publication and known external effects without advertising the affected value as ready
- **AND** the storage implementation's accepted recovery MAY verify and adopt, reissue, or remove orphaned work without claiming a crash-safe transaction that it does not provide

#### Scenario: One record fails in a larger production job
- **WHEN** one record fails while other independently complete records satisfy their publication obligations
- **THEN** the result MUST identify actual ready coverage and the failed or uncertain work
- **AND** accepted partial-coverage policy MAY retain the complete records without claiming that the whole scope completed

#### Scenario: Ready value does not match selected work
- **WHEN** a published value is structurally readable but belongs to another caption, transformation, source state, or representation than the consumer accepts
- **THEN** admission MUST reject it or follow an explicitly accepted recovery policy before computation
- **AND** stored validity MUST NOT override the consumer's accepted requirements

### Requirement: Cache storage lifecycle preserves outstanding use
Logical input/work identity and cache dependency meaning SHALL remain distinct
from physical storage access. Selected backends MAY use memory, individual
files, shards, or other supported storage without changing accepted consumer
meaning. Sharing, eviction, compaction, or replacement SHALL preserve required
outstanding reads and readiness guarantees through completion, supported
isolation, or rejection before dependent use. Storage capacity and miss
behavior SHALL follow accepted policies, not silently change input selection.

#### Scenario: Several samples share one cached representation
- **WHEN** several requested inputs have identical complete cache dependencies and supported reuse meaning
- **THEN** they MAY reference one stored value while retaining distinct logical work/handoff identities
- **AND** a cache filename or shard position MUST NOT become their semantic identity

#### Scenario: Compaction or eviction overlaps a consumer read
- **WHEN** storage coordination would relocate or remove a value whose accepted read is outstanding
- **THEN** it MUST preserve that read's required access until completion or use supported isolation
- **AND** if it cannot establish safe access, dependent use MUST remain unavailable rather than rely on an obsolete locator

### Requirement: Independent cache production has bounded lifecycle and owned continuation
Where selected and supported, cache production MAY progress independently of
training consumption without a complete finite manifest or shared step/epoch
clock. Run coordination SHALL manage accepted startup, shutdown, readiness,
cancellation, and failure relationships; cache infrastructure SHALL retain its
production traversal, private workers/queue, and continuation state. It SHALL
expose accepted bounds on work, resource use, and backpressure and respect
producer-state access and distributed data ownership. The input provider SHALL
retain selection, packing, delivery, and exposure continuation separately.
These obligations SHALL NOT require every private worker task to be a graph
node or every supported backend to implement every asynchronous/storage mode.

#### Scenario: Training consumes a ready subset during production
- **WHEN** accepted production continues while an input provider selects currently ready work
- **THEN** production MUST respect accepted capacity/backpressure and report per-work readiness/failure without a universal dataset-completion barrier
- **AND** the provider MUST own the accepted ready-subset selection/exposure policy rather than have cache arrival order silently establish it

#### Scenario: A producer fails or is cancelled with work in flight
- **WHEN** independent cache production fails, times out, or is cancelled
- **THEN** coordination MUST preserve completed publication, known effects, pending or uncertain work, and any cleanup failure according to the accepted lifecycle policy
- **AND** dependent work MUST wait, recover, use an explicitly accepted fallback, or stop before invalid input is delivered
- **AND** resource release or producer shutdown alone MUST NOT certify uncertain source or storage state as usable

#### Scenario: Cache work contributes to exact restoration
- **WHEN** exact continuation depends on independently progressing cache production and input consumption
- **THEN** cache coordination MUST contribute the production position, pending/publication state, dependency evidence, and stored-resource guarantees needed by the accepted recovery policy
- **AND** the input provider MUST separately contribute its selection/packing and unfinished handoff state
- **AND** reconstructible cache work MAY be reissued where the accepted continuation claim permits it
- **AND** neither a saved cache index nor a paused producer alone may claim a coherent exact run snapshot

### Requirement: Validation separates evaluation semantics from traversal
The pipeline SHALL own validation scheduling, data traversal, ordinary
train/eval transitions, aggregation, and result routing. The selected
validation capability SHALL own accepted model/objective evaluation semantics
and return typed results.

#### Scenario: SDXL validation runs
- **WHEN** Trainer coordinates validation for an accepted SDXL arrangement
- **THEN** the capability MUST reuse accepted prepared evaluation behavior
- **AND** it MUST NOT require an active strategy object to repeat the training flow

### Requirement: Sampling separates generation semantics from orchestration
The pipeline SHALL own sampling triggers, requests, destinations, and
observation. The selected sampling capability SHALL own accepted conditioning,
prediction, representation, and objective-compatible generation behavior.

#### Scenario: Objective-specific sampler is needed
- **WHEN** a selected objective requires compatible sampling behavior
- **THEN** the strategy acceptance check MUST validate that pairing
- **AND** generic trigger orchestration MUST not branch on model family to choose it

### Requirement: Persistence and restoration remain distinct capabilities
Trained-artifact persistence and exact runtime restoration SHALL use distinct
accepted requests, results, and identity meanings even if they share a trigger
or storage service. Trainer or delegated pipeline infrastructure SHALL
coordinate restoration of the coherent run. Accepted operations, capabilities,
optimization, and backend integrations SHALL contribute only the continuation
state they own.

#### Scenario: Artifact save succeeds but runtime state is incomplete
- **WHEN** a semantic model product is written without the authority and optimization state required for exact continuation
- **THEN** the product result MUST NOT claim resumable-runtime support

#### Scenario: Product and snapshot have different outcomes
- **WHEN** one lifecycle boundary successfully publishes a trained product but a required runtime-state contributor fails
- **THEN** the trained-product result MUST remain successful according to its declared product semantics
- **AND** the runtime-snapshot result MUST report failure or incompleteness and MUST NOT claim exact restoration

#### Scenario: Contributor participates in exact restoration
- **WHEN** an accepted operation or capability owns state required to resume the same run
- **THEN** it MUST expose that state through the restoration exchange
- **AND** it MUST NOT become the coordinator or identity authority for overall run restoration

### Requirement: Capability results feed observation without transferring ownership
Metadata, logging, metrics, reports, and resource systems MAY consume accepted
capability requests, results, transition facts, and failures. Observation SHALL
NOT mutate capability or binding state or roll back an accepted result.

#### Scenario: Result logging fails
- **WHEN** a capability completed and its observation sink fails afterward
- **THEN** the capability's accepted state/result MUST remain authoritative
