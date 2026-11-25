## Purpose

Define the shared composition language for stateful, heterogeneous data and
model work: region, flow, resource and realization contracts; invocation,
pending work and settlement; access modes and version admission; ownership of
items and leases; and the reference semantics that realizations
conform to. Training is one domain that composes these contracts; a data-only
composition is valid without any training meaning.

## ADDED Requirements

### Requirement: Maintained region contracts govern nested composition
Maintained region definitions SHALL describe their invocation, capture,
flow, resource, authority and lifecycle contracts. Parent conformance SHALL
compose that knowledge over actual IR rather than case names, diagnostic
histories or saved projections. Runtime invocation/request identities SHALL
remain distinct from static operation origins.

#### Scenario: A parent consumes a maintained child region contract
- **WHEN** verification or preparation composes a maintained child region
- **THEN** its registered definition MUST expose the required entry/yield types, capture mappings, invocation sources, admission/closure rules, child protocol, effects, resource/authority requirements, interruption and completion/settlement relationships
- **AND** the applicable obligations MUST be derived from actual operations, regions and references

#### Scenario: A handler body is invoked more than once
- **WHEN** an accepted handler template receives distinct requests
- **THEN** its changing arguments, locals and outstanding obligations MUST be associated with their distinct invocations
- **AND** one static SSA definition or a reused item key MUST NOT merge those invocation identities

#### Scenario: A data-only composition uses the common region contracts
- **WHEN** maintained producer, consumer and control activities require no model or optimization work
- **THEN** their hierarchy, admission, pending work, effects and settlement MUST remain representable through the same shared region contract meanings
- **AND** no dummy model, optimizer or training clock may supply otherwise missing lifecycle semantics

### Requirement: Execution contracts govern SSA availability
Every region SHALL define when operand values become available. An ordered
body SHALL admit only its entry arguments, permitted explicit captures and
results of earlier local operations. Other graph or recurrence execution
SHALL use an explicit applicable rule rather than inherit ordered assumptions.

#### Scenario: An ordered use or capture precedes its definition
- **WHEN** an ordered operation uses or captures a result that is not yet available
- **THEN** acceptance and preparation MUST reject the use before dependent execution
- **AND** the diagnostic MUST identify the operand/use and definition
- **AND** equivalent invalid wiring inside a derived word or its normalized expansion MUST also reject

#### Scenario: Repetition passes an initialized carry
- **WHEN** recurrence enters through a declared initialized loop argument and yields an available next value
- **THEN** ordered-value verification MUST preserve that recurrence
- **AND** a sibling region's entry or a later local result MUST NOT substitute for the declared argument

### Requirement: One implementation contract bounds resource effects
Installed implementation contracts SHALL state required resource reads/updates
through their typed argument positions and declare possible additional unknown
effects. Preparation and transformation legality
SHALL consume that declaration, separately from invocation authority, actual
reported effects/outcomes and physical settlement. Requirements SHALL follow
actual argument positions and resource identities, not an enclosing scope alone.

#### Scenario: An update position receives a read lease
- **WHEN** an actual argument cannot satisfy its implementation's update requirement
- **THEN** preparation MUST reject the mismatch before dependent readiness or effects
- **AND** the bounded corruption control MUST reject before resource factories or physical work enter
- **AND** runtime active-authority/current-version checks MUST remain in force for executed calls

#### Scenario: A captured read is used inside an unrelated writer
- **WHEN** a call's update position refers to a captured read of one resource while another resource has an enclosing writer
- **THEN** preparation MUST check the actual operand's issuer, resource identity/type and granted mode
- **AND** the unrelated writer MUST NOT supply the missing authority

#### Scenario: Unknown effects accompany otherwise valid selected work
- **WHEN** a valid implementation declares additional unknown effects
- **THEN** ordinary execution MAY proceed under its declared authority and completion rules
- **AND** transformations MUST conservatively honor the unknown effects
- **AND** neither unknown nor undeclared effects may authorize an otherwise undeclared resource update

### Requirement: Filtering declares output cardinality and claim settlement
A filtering invocation SHALL declare zero-or-one output and exactly-once input
disposition after required use. Retained output SHALL preserve actual input and
version provenance. Dropped admitted work SHALL remain distinct from input
admission rejection. Reservations, ready outputs and admitted claims SHALL
participate in their applicable outstanding-work and backpressure contracts.

#### Scenario: Filtering drops an admitted input while demand remains
- **WHEN** a filter successfully produces no downstream item for an admitted input
- **THEN** the input MUST receive its declared drop disposition exactly once and unused output capacity MUST be released
- **AND** no downstream item admission or fulfilled request may be invented
- **AND** pending retained-output demand MUST remain an enclosing obligation

#### Scenario: Filtering retains a value across an independent producer lifetime
- **WHEN** filtering forwards one admitted input into the retained-output flow
- **THEN** its actual source association and observed version dependencies MUST remain inspectable
- **AND** input and output claims or reservations MUST settle under their respective owners without duplication
- **AND** consumer-group completion MAY request producer stop while physical production still needs draining

#### Scenario: Stop or cancellation interrupts physical filtering
- **WHEN** filtering has entered physical work before a stop, failure or cancellation boundary
- **THEN** required input protection, disposition and cleanup MUST await that entered work's settlement
- **AND** completed retained products and later independent failures MUST remain separate outcomes

### Requirement: Local rewriting preserves observable contracts
Local transformation legality SHALL follow selected effects, actual SSA
wiring, authority and completion contracts. A legal rewrite SHALL preserve
values, input association, stage outcomes, validation and required settlement.
Purity alone SHALL NOT authorize removing observable coordination boundaries.

#### Scenario: Adjacent pure stages have compatible wiring and boundaries
- **WHEN** selected contracts and actual def/use relationships establish a legal local composition
- **THEN** a rewrite MAY remap their results and realize a combined local fragment
- **AND** stage entry, physical completion, validation, failure and source association MUST remain correct
- **AND** the transformed representation and execution distinction MUST be inspectable

#### Scenario: A valid opaque or authority boundary interrupts a local chain
- **WHEN** otherwise compatible stages are separated by unknown effects or an incompatible authority/completion boundary
- **THEN** the unfused valid strategy MUST remain executable
- **AND** unsafe fusion and a cold-loaded written form forging that fusion MUST reject before dependent effects

### Requirement: Requested work participates in enclosing settlement
Accepted requests SHALL become enclosing obligations before handler admission.
Contracts SHALL define request sources, closure, live bounds and dispositions.
Parent settlement SHALL require the declared source closure and settlement or
accepted transfer/disposition of pending/admitted requests, entered uses and
cleanup. A handler template alone SHALL NOT imply an admitted invocation.

#### Scenario: A requested evaluation waits for input after its driver ends
- **WHEN** a completed driver has created an accepted evaluation request whose input is not yet admitted
- **THEN** the enclosing consumer group MUST remain unsettled until that request and its required uses settle or receive an accepted disposition or transfer
- **AND** absence of a currently executing consumer MUST NOT imply group completion

#### Scenario: Completion creates a request while the parent checks settlement
- **WHEN** a successful child boundary promises a request under the accepted contract
- **THEN** the parent MUST NOT observe that completion for settlement before the request obligation or its accepted disposition is owned
- **AND** waiting for request capacity MUST retain the emitting invocation's outstanding obligation

#### Scenario: Orderly stop follows admission of work that promises a request
- **WHEN** the selected orderly-stop contract permits an already admitted iteration to finish and that iteration promises a completion-generated request
- **THEN** stop MUST preserve that request's registration and permitted completion
- **AND** the iteration's admission MUST NOT be confused with readiness or acquisition of all its individual inputs

#### Scenario: Run cancellation prevents a pending handler from executing
- **WHEN** the accepted cancellation policy closes further handler admission
- **THEN** every accepted pending request MUST retain its explicit unfulfilled/cancelled disposition rather than appear executed
- **AND** entered physical uses and required cleanup MUST still settle with separately known failures and outcomes

### Requirement: Completion-triggered producer stop covers consumer lifetimes
Completion-triggered producer stop SHALL cover dependent consumer/request
lifetimes or an accepted disposal/transfer rule. Consumer-group settlement MAY
request stop without awaiting producer termination; run settlement SHALL
account for both. Natural exhaustion, failure and cancellation SHALL retain
their own dispositions. Known whole-lifetime cycles SHALL be checked separately
from input waits and initialized feedback.

#### Scenario: Training completion stops a producer still needed by evaluation
- **WHEN** an authored completion-triggered stop uses training settlement while a permitted requested evaluator outside that lifetime closure can still need producer input
- **THEN** conformance MUST reject the missing consumer/request coverage unless an explicit accepted disposal or transfer rule satisfies it
- **AND** an operation name or a temporarily empty queue MUST NOT establish the missing coverage

#### Scenario: Consumer settlement requests stop during physical production
- **WHEN** the consumer group settles while an admitted producer call remains physically active
- **THEN** the group MAY complete and request producer stop while the run waits for producer drain and product disposition
- **AND** a completed evaluation result and a later producer failure MUST retain their separate outcomes

#### Scenario: Feedback forms a cycle without a whole-lifetime wait cycle
- **WHEN** a producer and consumer exchange feedback under accepted initialization, bounded-live-work and control-progress rules
- **THEN** the data/control cycle MUST NOT be rejected solely as a whole-lifetime completion cycle
- **AND** queue capacity alone MUST NOT be treated as proof of progress or termination

### Requirement: Prepared realizations preserve semantic correspondence
Normalized documents, prepared fragments and cold-loaded written forms SHALL
correspond to the verified accepted structure, captures, implementations, lifetimes
and request contracts they realize. Rejection SHALL distinguish invalid
accepted meaning from valid meaning that the selected target cannot realize,
and SHALL report each at its earliest authoritative boundary.

#### Scenario: A saved prepared claim omits an enclosing request obligation
- **WHEN** normalization or cold loading presents a prepared claim inconsistent with the verified source, capture, implementation, lifetime or request contracts
- **THEN** that claim MUST NOT publish dependent readiness, authorize execution or permit effects depending on it
- **AND** stored hashes and observation summaries MUST NOT stand in for semantic correspondence

#### Scenario: A target cannot realize a semantically supported contract
- **WHEN** target matching establishes that a supported region protocol or bound cannot be realized by the selected target
- **THEN** the system MUST identify and reject that realization limitation at the earliest authoritative boundary
- **AND** it MUST NOT omit required work, alter the semantic contract or claim execution readiness

#### Scenario: Invalid meaning and target limitation are reported differently
- **WHEN** one composition violates a semantic contract and another valid composition exceeds a target's supported protocols or bounds
- **THEN** the first rejection MUST identify the violated semantic contract
- **AND** the second MUST be reported as a realization limitation of that target without changing the accepted meaning or the language-wide rules

### Requirement: Selected code is identified by evidence the system computes
In one composition a name SHALL select exactly one installed implementation,
setup, materializer or deriver. The system SHALL identify every piece of code
a run selects by evidence it computes, never by a version kept by hand: a
hash of its declaration, its exact outputs on fixtures it ships with the
environment evidence that can change them, or, without fixtures, a hash of
the source of the module that defines it.

#### Scenario: Two installed pieces claim one name
- **WHEN** a composition installs two pieces of code under one name
- **THEN** the composition MUST be refused when it is put together

#### Scenario: Exact fixture outputs differ
- **WHEN** a piece's exact fixture outputs differ byte for byte from a record
- **THEN** the core MUST ask the fixture's installed comparison policy for a verdict and record it beside both exact records, without reading the numbers itself
- **AND** a fixture whose policy is not installed MUST count the difference as changed code, naming the missing policy

### Requirement: An implementation's state is written down or owned
Every piece of state bound into a selected implementation SHALL be
represented in its evidence, or acceptance SHALL refuse the implementation,
naming the declared way to go ahead: configuration as written settings, and
state that changes a later result behind an owner's lease. An implementation
MAY keep scratch of its own, state that never changes a result and is never
saved or restored.

#### Scenario: An implementation holds a value that can change
- **WHEN** a selected implementation captures or binds a value that can change while staying the same object, such as a list, a dict or a model object, or is a callable object or a method bound to an object
- **THEN** acceptance, and a host's check before it runs a run, MUST refuse it, naming the value and the declared way to go ahead
- **AND** an implementation whose captured or bound values cannot change and read the same in every program, such as numbers, strings or tuples of them, MUST be accepted with those values written into its evidence

#### Scenario: A selected implementation is compiled code
- **WHEN** a selected implementation is compiled code installed directly, whose source the system cannot read
- **THEN** acceptance, and a host's check before it runs a run, MUST refuse it until evidence of the binary exists
- **AND** the refusal MUST name the declared way to go ahead: a Python implementation that calls the compiled code

### Requirement: A run's code records are complete and readable
A run SHALL record the repository revision it started from and any
uncommitted changes. A code record SHALL keep its parts and print in full,
its hash being only a short name for that printout, and a mismatch SHALL be
reported part by part.

#### Scenario: A run starts from edited code
- **WHEN** a run starts with uncommitted changes in the repository
- **THEN** its records MUST name the revision and the changed files with their hashes
- **AND** nothing made from the edited code MAY be taken for something made from that revision

#### Scenario: Installed code differs from a record
- **WHEN** the evidence of installed code differs from a record
- **THEN** the report MUST name each part that differs, such as the declaration, which fixture's outputs and by how much, or which file

### Requirement: Item claims and leases follow an ownership discipline
Admitted item claims and leases SHALL be checked as owned values,
not ordinary copyable results. Along every execution path an item claim SHALL be
consumed, transferred, explicitly forked or given a disposition exactly once.
A lease SHALL NOT outlive its scope except through an accepted
owner transfer. These checks SHALL derive from types and region contracts, not
per-composition rules.

#### Scenario: A lease escapes through a result or nested capture
- **WHEN** an authored yield or capture would outlive the protected scope without a retained dependency or accepted owner transfer keeping every required use protected
- **THEN** conformance MUST reject that escape before the dependent use becomes executable
- **AND** physical settlement of the producing call alone MUST NOT prove that a retained derivative or consumer use has ended

#### Scenario: An item claim is duplicated without a fork
- **WHEN** an authored body passes one admitted item claim to two consumers without an explicit fork operation
- **THEN** conformance MUST reject the duplicated claim
- **AND** sharing the payload value MUST remain permitted where its contract allows it

#### Scenario: Mutually exclusive branches each dispose the same claim
- **WHEN** an accepted choice routes one item claim into alternatives of which exactly one executes per invocation
- **THEN** each alternative MAY consume the claim
- **AND** the check MUST rely on the choice's region contract rather than on SSA use counts

### Requirement: Retained products declare dependencies and release
A resource-backed product SHALL distinguish its payload from its retained
storage/state dependencies. Its contract SHALL identify dependency arguments,
permitted borrowing/mutation, escape and release rules. Producing-call
completion SHALL NOT release obligations needed by dependent physical uses.
Actual storage SHALL remain outside semantic IR.

#### Scenario: A scoped retained product is borrowed by later physical work
- **WHEN** a selected call produces a scoped retained product from declared leases or earlier retained products
- **THEN** consumers MUST retain the actual dependencies and observed versions through physical settlement
- **AND** read-only borrowing MUST NOT authorize mutation or an undeclared owner transfer
- **AND** explicit release MUST follow every dependent use, and a retained product derived from it MUST keep those dependencies and observed versions until its own release

#### Scenario: A result that depends on a lease outlives the access body
- **WHEN** a selected call's result is declared to depend on leased state and is used after that access body ends, within the same activity
- **THEN** the leased state MUST stay protected until that result's last use, with its release explicit in the normalized form
- **AND** the result MUST NOT be yielded into a flow or captured by another activity

#### Scenario: A scoped retained product escapes or is reused prematurely
- **WHEN** a retained product leaves its activity, is used after release, or conflicts with a known update to its retained owner
- **THEN** verification or preparation MUST reject the invalid claim before dependent execution
- **AND** cold correspondence MUST check the declared retention against installed implementation contracts

#### Scenario: Cancellation interrupts an entered retained consumer
- **WHEN** cancellation closes admission while a consumer still physically uses a retained product
- **THEN** cleanup MUST keep its dependencies protected until that consumer settles
- **AND** neither diagnostic erasure nor producing-call completion may authorize early release

### Requirement: The core knows only what its own promises need
The core SHALL know a concept only where it needs it to decide when work may
run, which state it may touch at which version, how long results and leases
stay protected, when work is complete, how work is admitted to devices and
ranks, and which outcomes the schedule can change. A domain SHALL use the
program's structure through reachability over declared relations, annotations
written as data, and setups, whose meaning the core never interprets.

#### Scenario: A domain asks which calls lie on a path
- **WHEN** a domain asks, for one declared relation, which calls lie on a path from sources to sinks it names
- **THEN** the path MUST follow declared edges and every value the core's own constructs pass on, through captures, yields, repeat carries, every alternative of a choice, streams and requests
- **AND** the path MUST stop only at an input whose implementation declares no edge for that relation, and what a path between activities means MUST be left to the asking domain

#### Scenario: A saved prepared form carries annotations and setups
- **WHEN** a saved prepared form is loaded
- **THEN** its annotations and setups MUST be derived again and the form refused if they differ
- **AND** calls MUST fuse only where their annotations and setups are the same

### Requirement: Setups establish what calls need and undo it
A call setup SHALL change only its worker's own context, such as grad mode or
autocast, and undo it when the calls return. A held setup SHALL change only
state reached through the handles of the resources it names, and its hold
SHALL bind while a lease its calls use is in use by a running call or kept by
a retained result. No setup SHALL change a version, take a lease, wait on
another activity or issue collectives.

#### Scenario: Calls in one access need different states of a resource
- **WHEN** calls in one access body need different states of a resource that a held setup changes
- **THEN** a later call MUST wait until no running call uses the hold's leases and no retained result keeps them
- **AND** preparation MUST refuse a call that would wait on a hold its own activity keeps binding

#### Scenario: A held setup changes a resource's storage
- **WHEN** a held setup changes the storage of a resource, as an in-place swap does
- **THEN** the resource's updates, its uses through other routes and the captures of their snapshots MUST wait while the change is in place and until it is undone
- **AND** the undo MUST run however the calls end, at the latest when the last lease the hold covers is released

### Requirement: Lifecycle semantics have an executable reference
The shared invocation protocols SHALL have a deterministic reference semantics
that executes normalized IR and derives admission, completion, interruption,
settlement and dispositions from child states. Verifier rules and prepared
runtimes SHALL be checked against it. Neither the verifier nor a runtime SHALL
be the only definition of lifecycle correctness.

#### Scenario: A runtime disagrees with the reference
- **WHEN** a prepared runtime reports owner-visible lifecycle facts for a supported strategy that differ from the reference under the same explicitly supplied external inputs, completion and scheduling choices
- **THEN** conformance MUST report a defect in the runtime or the reference
- **AND** the runtime's observed behavior MUST NOT be accepted as correct by default

#### Scenario: A runtime omits an admission or request
- **WHEN** the reference is given external choices for a supported strategy
- **THEN** it MUST derive legal admissions and request obligations from normalized IR and its protocol states
- **AND** it MUST NOT adopt the runtime's admitted-work trace as its expected result
- **AND** comparison MUST permit private bookkeeping differences that preserve the declared observable meaning

#### Scenario: A strategy admits several legal interleavings
- **WHEN** a supported strategy can interleave differently at its suspension points
- **THEN** conformance MUST compare the runtime against the reference across a systematic or randomized set of those interleavings
- **AND** a single passing schedule MUST NOT certify the protocol

#### Scenario: The reference evaluates a model composition
- **WHEN** the reference executes a composition containing model or training work
- **THEN** selected computations MAY be replaced by declared stand-ins that honor their contracts and supply fixture values
- **AND** establishing lifecycle outcomes MUST NOT require Torch, a model or an optimizer

### Requirement: Resource uses select a defined access mode
Every resource use SHALL select an access realization from a closed,
versioned set: immutable published snapshot, protected live read, owner update
or scoped grant. Its freshness/version-selection requirement SHALL be stated
separately. The contract SHALL define the evidence for the combination. A
target unable to establish that evidence SHALL reject it rather than
substitute another realization or freshness guarantee.

#### Scenario: A live shared read runs on a capable target
- **WHEN** a use selects protected current read of state its owner may update concurrently
- **THEN** the target MUST establish exclusion and physical-use completion over the use extent, including device-ordering evidence and, when sharded, collective-ordering evidence
- **AND** the use's product MUST record the actual state version it observed
- **AND** incompatible owner updates MUST NOT race the protected use even though producer and update activities may progress independently between uses

#### Scenario: A target cannot realize live sharing
- **WHEN** the selected target cannot establish protected current read evidence
- **THEN** preparation MUST reject that use as a realization limitation
- **AND** it MUST NOT silently substitute a snapshot or an unprotected read

#### Scenario: Current-at-start freshness accompanies protected live access
- **WHEN** an evaluation selects protected live access with current-at-start freshness
- **THEN** the owner MUST resolve freshness and acquire protection atomically when the use starts
- **AND** protection MUST retain the selected version through the required use extent

#### Scenario: A snapshot read decouples from owner updates
- **WHEN** a use selects a published snapshot
- **THEN** its protection obligation MUST cover that snapshot's lifetime rather than the owner's live state
- **AND** the owner's updates MUST NOT be required to wait for that use
- **AND** current-at-start snapshot selection MUST establish coherent isolation/publication before the snapshot is used

#### Scenario: Requested work selects the version at its request
- **WHEN** a use inside requested work selects the version of owner state that was current when its request obligation was first owned
- **THEN** that version MUST remain available until the request's required uses complete or a detached realization holds it, by making conflicting updates wait or by isolating a snapshot of it
- **AND** handler admission MUST NOT be the point at which that version is selected
- **AND** a target unable to establish either MUST reject the selection rather than let the work observe a later version
- **AND** a request that ends cancelled or unfulfilled MUST release that version
- **AND** the use's product MUST still record the actual state version it observed

### Requirement: Version selection, device admission and placement are independent
Which version a use observes, whether work is admitted onto a device beside
other work, and where work runs SHALL be decided separately; none SHALL be
inferred from another. A strategy SHALL state only whether work of different
activities may overlap on a device; the selected target SHALL decide whether
permitted overlap is realized. Work SHALL NOT wait while holding a device for
something that may need that device to progress.

#### Scenario: Permitted overlap does not fit the device
- **WHEN** a strategy permits two activities' work to overlap on a device that cannot hold both
- **THEN** the runtime MUST run that work in turn without changing which versions it observes

#### Scenario: Meaning requires work to be live together
- **WHEN** a strategy's meaning requires work to be live together on a device that cannot hold it
- **THEN** preparation MUST reject it as a realization limitation

#### Scenario: A reserved version and a device admission would wait on each other
- **WHEN** the work that would release a reserved version needs a device held by work that waits on that version
- **THEN** preparation MUST reject a combination whose wait cycle it can prove
- **AND** at run time waiting work MUST obtain its state conditions and admitted inputs before the device, or give the device up while it waits

#### Scenario: A frozen participant leaves the device between uses
- **WHEN** a strategy permits a participant to leave the device between uses and the target moves it out
- **THEN** moving it out and back MUST NOT change its version or publish anything, since its values stay identical
- **AND** a permission the target cannot realize, such as for storage that a unit's optimizer state refers to, MUST be rejected as a target limit

### Requirement: Flow admission declares its version policy
A flow whose products depend on versioned owner state SHALL declare an
admission policy over the actual production version: exact current, bounded
lag, or any version with recorded provenance. Admission SHALL evaluate the
version actually observed during production, not a version assumed when
the work began. The policy SHALL name owner/incarnation, version lineage, lag metric and
the comparison boundary.

#### Scenario: Admission and execution entry observe different current state
- **WHEN** a product passes exact-current admission and its owner advances before computation entry
- **THEN** the admission claim MUST remain scoped to its declared comparison boundary
- **AND** a requirement for current state at execution entry MUST use a separately established atomic freshness/protection check

#### Scenario: A product depends on several versioned owners
- **WHEN** admission compares a product with more than one version dependency
- **THEN** it MUST retain and evaluate the required owner/incarnation and lineage-specific version map under the declared metric
- **AND** incomparable versions MUST receive an explicit policy disposition rather than be compared through a global training step

#### Scenario: A bounded-lag flow receives an older product
- **WHEN** a product's actual production version is older than the consumer's current version but within the declared lag bound
- **THEN** admission MUST accept it and retain its actual version as provenance
- **AND** a product outside the bound MUST be rejected with an explicit disposition rather than reported as consumed

#### Scenario: An exact-current flow receives work begun before an update
- **WHEN** a producer began work before an owner update but observed the updated version during production
- **THEN** admission MUST evaluate the observed version
- **AND** a product that observed only the earlier version MUST be rejected under the exact-current policy

### Requirement: Compositions report their reproducibility class
Access modes, policies and flow admission rules SHALL declare whether their
observable outcomes are schedule-independent given the same inputs, seeds and
configuration. Verification SHALL derive a composition's reproducibility class
from those declarations, and SHALL identify the operations that make it
schedule-dependent.

#### Scenario: Snapshot access with deterministic version assignment
- **WHEN** every versioned use reads a snapshot chosen by a deterministic rule over accepted progress
- **THEN** verification MUST report the composition as schedule-independent with respect to version selection

#### Scenario: A live shared read makes outcomes schedule-dependent
- **WHEN** a composition contains a protected current read whose observed version depends on scheduling
- **THEN** verification MUST report the composition as schedule-dependent
- **AND** the report MUST name that use and its admission policy

#### Scenario: Accepted ordering fixes a protected read's selected version
- **WHEN** the contracts order a protected live read after the last relevant update and prevent another writer before its use ends
- **THEN** verification MAY establish schedule independence with respect to that version selection
- **AND** the overall class MUST still account for input, admission, RNG and other declared decisions

#### Scenario: Deterministic snapshot selection coexists with admission races
- **WHEN** snapshot versions are chosen deterministically but accepted input or admission choices can vary with scheduling
- **THEN** verification MUST retain the resulting possible schedule dependence of the composition
- **AND** snapshot selection alone MUST NOT certify reproducibility of the whole composition

#### Scenario: A declaration is missing
- **WHEN** an operation's reproducibility declaration is absent
- **THEN** verification MUST report the composition's class as unknown rather than reproducible

### Requirement: Numerical choices are fixed for the run
The core SHALL define no numerics: a call's numbers belong to what realizes
it, such as its implementation, its route's realization, kernels and the
device. Choices that can change a run's numbers, such as an implementation, a
route's realization or a device setting such as TF32, SHALL be fixed for the
run: the strategy states those it relies on, and those it leaves open SHALL
be chosen at launch and recorded with the run.

#### Scenario: A run resumes after leaving a device setting open
- **WHEN** a run whose strategy left a device setting open is resumed
- **THEN** the resume MUST apply the setting chosen and recorded at the run's launch
- **AND** it MUST NOT choose the setting again

#### Scenario: Devices and implementations report facts
- **WHEN** devices and implementations report facts such as the device model and library versions
- **THEN** the run MUST record them as reported, and a saved run that lacks a fact first recorded later MUST NOT be refused for lacking it
- **AND** only the facts a reproducibility policy names MAY be compared

### Requirement: Regions declare quiescence boundaries
Each maintained region contract SHALL declare where a consistent cut of its
invocations, pending obligations and owned state may be taken, or declare that
none is supported. Exact restoration and governed transitions SHALL use only
declared boundaries. Without one, dependent restoration or transition claims
SHALL be reported unavailable rather than approximated.

#### Scenario: A cut is requested while a producer remains open
- **WHEN** an exact-restoration snapshot is requested for a parallel composition whose producer is still producing
- **THEN** the cut MUST be taken only at the boundaries the composed contracts declare
- **AND** it MUST record pending requests, unpublished and ready products and their dispositions as the contracts require

#### Scenario: A region declares no quiescence boundary
- **WHEN** an exact-restoration or governed-transition claim depends on a region that declares no supported boundary
- **THEN** that claim MUST be reported unavailable
- **AND** a cut taken at an arbitrary suspension point MUST NOT be presented as exact

### Requirement: Replicated decisions remain rank-consistent
Under a replicated execution target, each policy choice, request creation,
stop decision and admission outcome that affects collective work SHALL be
declared either deterministic from replicated state or decided once and
distributed. A composition whose decisions cannot be shown rank-consistent
SHALL be rejected for that target.

#### Scenario: A policy uses rank-local randomness on a replicated target
- **WHEN** a policy selects collective work from rank-local state without a supported replicated agreement or distribution protocol
- **THEN** preparation MUST reject the composition for that target
- **AND** the rejection MUST identify the undeclared decision

#### Scenario: A decision is made once and distributed
- **WHEN** a stop, request or policy decision is declared as decided on one rank and distributed before dependent collective work, including a decision using that rank's local randomness
- **THEN** the target MUST realize that distribution before any rank acts on the decision

#### Scenario: A single-process target runs the same composition
- **WHEN** the selected target executes without replication
- **THEN** rank-consistency declarations MUST NOT restrict otherwise valid decisions

### Requirement: A run keeps the GIL off unless it names what turns it on
A run SHALL keep the GIL off unless a selection it names needs a module that
turns it on. Every piece a run loads SHALL declare any such module it needs,
the run SHALL record the selections it names, and modules known to turn the
GIL on SHALL stay unavailable unless a named selection needs them. The GIL
SHALL never be a correctness mechanism: correctness rests on the ownership
rules.

#### Scenario: A selection needs a module that turns the GIL on
- **WHEN** a strategy selects a piece that needs a module that turns the GIL on, and the run does not name that selection
- **THEN** fulfillment MUST refuse it, naming the module and the declared way to go ahead, naming the selection
- **AND** once the run names it, the run MUST load it and report that the GIL is on and which selection turned it on

#### Scenario: The GIL turns on outside a named load
- **WHEN** the GIL turns on while the run loads or runs anything other than a named selection
- **THEN** the run MUST fail, naming the module that turned it on where that can be known
- **AND** the run MUST check that the GIL is off when it starts, after each load, and periodically while it runs
