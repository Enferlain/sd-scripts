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
under its accepted input policy; consuming coordination SHALL still check
readiness and admission before use, with training input coordination responsible
for training-input handoff. Neither traversal owner SHALL silently
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
A caching request SHALL identify the selected accepted computation/behavior and
representation boundary, requested work or production scope, relevant
input/transformation meaning, consumer schema, producer dependencies,
reuse/consistency obligations, storage support, and applicable lifecycle and
failure policies. Cache coordination SHALL derive requests and judge results
under the existing
run-specific obligations; it SHALL NOT invent another acceptance policy.
Results SHALL remain associated with their request/attempt and distinguish
completed values and actual provenance, storage/readiness outcomes, incomplete
coverage, and failures or uncertainty. No universal image, latent, tokenizer,
tensor-axis, or file-per-sample fields SHALL be required by this exchange.
Accepted behavior identity/meaning SHALL NOT be equated automatically with
physical callable, compiler-artifact, or backend identity. A different
realization MAY preserve reuse only with evidence that relevant computation,
numerical/stochastic, and dependency obligations remain satisfied; realization
differences SHALL remain dependencies wherever they affect those guarantees.

#### Scenario: Equivalent realization changes without changing cache meaning
- **WHEN** the same accepted encoder behavior uses a different eager/compiled or backend realization
- **THEN** reuse MAY remain compatible only if evidence establishes that its relevant accepted computation and numerical/stochastic dependencies remain satisfied
- **AND** compatibility MUST NOT be inferred from either physical callable identity or a matching semantic label alone

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
Cache coordination SHALL publish readiness only for a complete usable accepted
publication unit, such as a value, bundle, or independently consumable chunk,
whose payload, storage access/index, schema, and dependency evidence
satisfy its accepted consistency boundary. That boundary SHALL be explicit
without requiring one dataset-wide transaction. Published availability SHALL
NOT establish admission for another request or consumer. The consuming
coordination surface SHALL check exact work association, representation,
freshness/lag, and gradient requirements against that consumer's existing
accepted obligations before use. Training input coordination SHALL do so for
training inputs; relevant capability coordination SHALL do so for its own
consumption, directly or through shared input coordination, without creating
another acceptance policy or a universal training-input gateway. Required
source/storage guarantees SHALL remain protected through dependent use.
Results SHALL describe actual publication and partial coverage rather than
infer readiness from file existence, queue position, or a successful encoding
call.

#### Scenario: Consumers share storage but have different requirements
- **WHEN** training and validation or sampling consume a shared cached representation under different accepted requirements
- **THEN** the relevant consuming coordinators MUST judge admission under their respective existing obligations
- **AND** admission for one consumer MUST NOT establish admission for another, such as when detached cached output is allowed for inference but cannot replace a required training derivative path

#### Scenario: Only some independently usable chunks are published
- **WHEN** a selected representation permits independently consumable chunks and only some chunks meet their complete publication obligations
- **THEN** those chunks MAY become ready and be admitted to consumers whose accepted scope requires only them
- **AND** the entire representation or a required full bundle MUST NOT be advertised as ready merely because those chunks are ready

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
and return typed results. It SHALL define measurement contributions and
reduction/weighting meaning; pipeline aggregation SHALL execute that meaning
rather than invent a metric from returned batch values. Outer evaluation
traversal SHALL NOT transfer a selected input provider's private selection,
packing, or continuation ownership to the coordinator.

#### Scenario: SDXL validation runs
- **WHEN** Trainer coordinates validation for an accepted SDXL arrangement
- **THEN** the capability MUST reuse accepted prepared evaluation behavior
- **AND** it MUST NOT require an active strategy object to repeat the training flow

#### Scenario: Evaluation uses a bounded input stream
- **WHEN** accepted evaluation has no finite dataset length and uses a bounded coverage/stopping rule
- **THEN** pipeline coordination MUST honor that rule through the selected provider
- **AND** evaluation MUST use its own continuation or an explicitly accepted sharing policy rather than silently advance the training input cursor
- **AND** cache-backed evaluation inputs MUST satisfy current consumer admission obligations

### Requirement: Validation results preserve measurement and effect meaning
Validation requests SHALL identify accepted evaluation, input scope and policy,
source consistency, runtime/derivative needs, measurement rules, and declared
effects. Results SHALL preserve request/attempt and evaluated-input
associations, actual source-state provenance, relevant coverage and
weighting/reduction evidence, completion/failure meaning, and declared-effect
outcomes. Incomplete work SHALL NOT claim the requested completed measurement;
accepted partial measurements SHALL retain their scope. Empty or undefined
reductions SHALL follow the accepted policy rather than fabricate a score.
Evaluation MAY supply accepted adaptive feedback or state effects and require
derivatives; it SHALL NOT receive implicit optimization advancement or
participant-transition authority. Algorithmic state and reactions SHALL retain
their declared owners and continuation needs, not become generic Trainer
evaluation logic.

#### Scenario: Unequal batches or distributed contributions are reduced
- **WHEN** evaluated inputs have unequal relevant weight, masks, or distributed coverage
- **THEN** pipeline aggregation MUST apply the selected measurement's reduction and coverage rules
- **AND** it MUST NOT silently substitute an unweighted mean of batch means or claim missing contributions completed

#### Scenario: Validation changes an accepted adaptive policy
- **WHEN** selected evaluation measurements feed an accepted later noise-range or other adaptive decision
- **THEN** accepted runtime behavior MUST preserve that feedback and its declared state owner after authoring ends
- **AND** any derivative work and effects MUST stay within the granted profile
- **AND** measurement completion, adaptive-state effects, and a failed reaction MUST remain distinct wherever dependent work relies on them

### Requirement: Sampling separates generation semantics from orchestration
The pipeline SHALL own sampling triggers, requests, destinations, and
observation. The selected sampling capability SHALL own accepted conditioning,
prediction, representation, and objective-compatible generation behavior.
Pipeline coordination SHALL own outer request traversal and output handling
through selected publication implementations. Sampling requests SHALL identify
the selected generator, supported domain parameters, work/output associations,
relevant random-input policy, source/view consistency, destination, declared
effects, and cancellation/partial-failure policy. Universal sampling inputs or
outputs SHALL NOT require image dimensions, prompt strings, PIL images, or PNG.

#### Scenario: Objective-specific sampler is needed
- **WHEN** a selected objective requires compatible sampling behavior
- **THEN** the strategy acceptance check MUST validate that pairing
- **AND** generic trigger orchestration MUST NOT branch on model family to choose it

### Requirement: Sampling results distinguish generation publication and reporting
Sampling results SHALL preserve request/attempt and output associations,
actual input/source-state provenance, produced outputs, and actual published
resources where requested. Generation, writing/publication, observation,
cancellation, failure, and declared effects SHALL remain separately reportable
where accepted consumers depend on them. Partial failure SHALL preserve known
successful, failed, uncertain, and unattempted outcomes without claiming a
job-wide transaction. A planned destination SHALL NOT prove a written output.
Observation failure SHALL NOT erase accepted output or authorize regeneration;
retry/reissue SHALL follow the accepted policy and actual outcome evidence.
Sampling publication SHALL NOT by itself claim trained-artifact persistence
or exact runtime restoration.

#### Scenario: Output is saved before reporting fails
- **WHEN** generation and output publication succeed but a tracker or other sink fails
- **THEN** the accepted output/publication result MUST remain authoritative with its actual resource association
- **AND** the reporting failure MUST remain separate and MUST NOT silently rerun generation

#### Scenario: Only part of a multi-request job publishes
- **WHEN** one requested output publishes and another generates but fails to save
- **THEN** the result MUST distinguish the published output from the generated-but-unpublished output and any unattempted work
- **AND** successful resources MUST NOT be erased from the result or presented as a wholly completed request set

#### Scenario: A selected generator produces non-image or related outputs
- **WHEN** accepted generation returns text, audio/video, or several associated outputs
- **THEN** selected schemas and publication implementations MUST retain their domain meaning and associations
- **AND** generic coordination MUST NOT require an image-shaped payload or add model-family branches

### Requirement: Evaluation and generation use protected current state
Validation and sampling SHALL execute only with current accepted input and
source/view evidence and protected access satisfying their consistency,
numerical, and derivative needs through actual completion. A trigger coordinate
SHALL NOT stand for actual source-state provenance. That provenance MAY be a
protected state, snapshot, or structured evidence over states/segments where
the selected consistency policy permits it; attribution alone SHALL NOT make
mixed-state work valid. Conflicting owner work SHALL wait, reject before use,
or use supported isolation/synchronization under the existing accepted access
relationships. Neither capability SHALL bypass readiness by unwrapping,
moving, or recasting live state. Capability due/completion/failure relationships
SHALL NOT be forced into one optimizer clock or a universal sample-then-validate
sequence.

#### Scenario: Sampling conflicts with temporary parameter perturbation
- **WHEN** a due sampling request would read state protected by an in-progress two-pass optimization region
- **THEN** it MUST wait, reject before use, or use an accepted isolated view without observing the temporary perturbation as ordinary source state
- **AND** stale inference views or uncertain handback MUST NOT authorize dependent use

#### Scenario: Evaluation spans a source-state change
- **WHEN** an evaluation requires a consistent source but training could change that source during traversal
- **THEN** coordination MUST protect that consistency or reject the unsupported overlap
- **AND** results MUST identify actual source provenance rather than label all work with the trigger's requested state
- **AND** a multi-state measurement MAY execute only under an explicitly accepted consistency and attribution policy

### Requirement: Capability temporary projections have scoped cleanup
Pipeline coordination SHALL establish and restore ordinary runtime projections
and relevant placement, dtype, and RNG state under their accepted owners and
numerical policy. Cleanup SHALL cover partial setup, execution failure,
cancellation, and actual backend completion and preserve the prior relevant
state rather than assume every module/optimizer returns to training mode.
Unsupported RNG or mutable-state overlap SHALL NOT claim isolation or exact
replay. Uncertain restoration SHALL keep dependent use unavailable until
accepted recovery establishes validity; releasing exclusion alone SHALL NOT
prove readiness. Primary and cleanup failures, known results/effects, and
uncertainty SHALL remain distinct. Missing or invalid results SHALL NOT prove
that no state effect occurred. Capability and selected algorithm owners SHALL
contribute their continuation state without claiming that local cleanup alone
establishes a coherent exact run snapshot.

#### Scenario: Temporary setup fails after changing only some projections
- **WHEN** validation or sampling changes a mode, placement, dtype, or RNG scope and then setup fails
- **THEN** cleanup MUST restore the changed prior state or mark dependent use unsafe
- **AND** later execution MUST NOT be admitted merely because cleanup released a scope

#### Scenario: Cleanup fails after work produced known results
- **WHEN** selected evaluation/generation or publication reaches a known outcome and restoration later fails
- **THEN** reporting MUST retain that outcome and separately report the cleanup failure and uncertain state
- **AND** unsafe dependent work MUST remain stopped until accepted recovery

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
