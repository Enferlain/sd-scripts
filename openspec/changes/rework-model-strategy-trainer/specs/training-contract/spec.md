## Purpose

Define the training contract system that guides explicit strategy authoring,
fulfills one complete authored definition against a pre-existing contract, and
supplies the authoritative run-specific obligations used throughout governed
realization and later transitions.

## ADDED Requirements

### Requirement: The active contract guides strategy authoring
The system SHALL make the active core contract, recognized capability
contracts, feature contracts, contract version, and execution/ownership
profiles available before a strategy is authored for that contract.

#### Scenario: Author uses maintained contract vocabulary
- **WHEN** a repository author defines a maintained strategy
- **THEN** the author MUST be able to select and wire behavior described by the active contract
- **AND** the author MUST NOT have to infer the Trainer boundary from current family-specific calls

### Requirement: Offered ownership profiles define policies and scoped grants
Before authoring, the active contract SHALL expose supported standard
advancement policies and alternative execution grants, including phase owners,
state-access and effect bounds, contribution and synchronization obligations,
handback, failure, and continuation meanings. The authored strategy SHALL select
from those offered meanings; neither a wrapper's internal behavior nor a
runtime callback SHALL create permission. Region-local grants SHALL leave
unrelated work's accepted owners intact.

#### Scenario: Standard policy coordinates independently or jointly due units
- **WHEN** a strategy selects a supported standard policy for differently due units or shared computation offered to several disjoint units
- **THEN** fulfillment MUST retain its addressed sources, gradient routes, windows, ordering, and outcome obligations
- **AND** it MUST NOT require an imperative transfer merely because the cadence differs from one optimizer step per action

#### Scenario: A selected wrapper performs a retained phase internally
- **WHEN** a wrapper advances an optimizer inside a region whose offered grant transfers backward and temporary editing but not final advancement
- **THEN** fulfillment MUST reject that ownership mismatch unless the strategy selects a different supported profile with the required duties
- **AND** library availability MUST NOT silently expand the grant

#### Scenario: Ordinary profile is asked to guarantee atomic joint advancement
- **WHEN** authored behavior requires all-or-nothing multi-unit advancement that the selected profile does not provide
- **THEN** fulfillment MUST reject the unsupported guarantee rather than infer rollback or replay from one action boundary

### Requirement: Authors assemble strategies explicitly
Strategy authors SHALL explicitly select and wire training intent, participants,
relationships, objective behavior, features, capabilities, and bounded choices.
Strategy fulfillment SHALL NOT silently select a missing implementation, add a
dependency, or infer the intended arrangement.

#### Scenario: Required behavior is missing
- **WHEN** an authored definition omits a provider required by the features, capabilities, or bounded choices it selected
- **THEN** strategy fulfillment MUST reject the definition with the missing obligation
- **AND** it MUST NOT choose a provider on the author's behalf

#### Scenario: Configuration requests a provided capability
- **WHEN** execution configuration requests use or settings of a capability already exposed by the authored strategy
- **THEN** the request MAY parameterize that accepted capability within its declared bounds
- **AND** it MUST NOT select a different internal implementation or assemble a new strategy feature

### Requirement: Authored behavior declares what is needed to judge it
Authored behavior MAY include stateful algorithms, schedules, reactions to
observations, bounded decisions from live inputs, declared effects, and
permitted authority-governed transitions. Authors SHALL declare the inputs,
owned state, effect vocabulary and bounds, authority, and decision bounds needed
to judge that behavior.

#### Scenario: Author selects adaptive behavior
- **WHEN** an authored algorithm changes later behavior from accepted runtime observations
- **THEN** the strategy MUST identify the algorithm, its observation inputs, its owned state, and its possible effect vocabulary and bounds
- **AND** it MUST NOT be rejected merely because its outputs cannot be fixed during authoring

#### Scenario: Author selects an input policy
- **WHEN** a strategy selects input sources, transformations, mixture or packing behavior, or bounded stage changes
- **THEN** fulfillment MUST preserve the selected provider, consumer representation, state owner, dependency and admission meanings
- **AND** it MUST NOT infer a universal image batch, epoch schedule, or hidden input fallback

### Requirement: Declarations preserve cross-owner relationships
Where another owner, preparation, coordination, observation, or restoration
depends on them, authored declarations SHALL also preserve work and handoff
relationships, gradient demands and routing, required numerical-state lifetimes,
phase ownership, and completion/recovery meanings. Private implementation
details SHALL NOT require universal graph nodes or public tensor operators
merely because they exist.

#### Scenario: Author selects several uses of one participant
- **WHEN** an authored action invokes one participant through different roles, prepared views, or gradient paths
- **THEN** the strategy MUST describe those uses and their dependencies without declaring duplicate participants merely to express the calls

#### Scenario: Author requests differentiated work across owners
- **WHEN** selected computation exposes outputs or derivatives to another accepted owner
- **THEN** fulfillment MUST retain the derivative demands, applicable seeds and routing, gradient cuts, destinations, lifetime requirements, and responsible owners
- **AND** concrete backend evidence that is not yet available MUST remain an explicit readiness obligation rather than an author assertion of support

### Requirement: Strategy fulfillment produces acceptance or rejection
The training contract system SHALL evaluate one complete authored strategy using
the active contract version and execution/ownership profile. It SHALL return
useful acceptance or rejection information and SHALL produce an accepted run
arrangement only on success. Fulfillment requirements do not prescribe whether
conformance rules are represented as contract data, shared validators,
conformance implementations, or a combination.

#### Scenario: Complete definition is accepted
- **WHEN** every selected concern and bounded choice is mutually compatible under the active contract
- **THEN** fulfillment MUST produce one semantically accepted run arrangement with its authoritative obligations and one authority seed
- **AND** it MUST distinguish that semantic acceptance from fulfilled executable readiness

#### Scenario: Complete definition is rejected
- **WHEN** selected behavior is missing, incompatible, has no definition in the active contract, or uses undeclared extension authority
- **THEN** fulfillment MUST return detailed author-facing rejection information
- **AND** no Trainer-acceptable arrangement or authority seed may be produced

#### Scenario: A choice is untested or conflicts with a declared meaning
- **WHEN** a selected combination would run but is untested, lacks provenance, or conflicts with a component's declared meaning
- **THEN** fulfillment MUST accept it and report it
- **AND** it MUST NOT reject a choice for those reasons alone

### Requirement: Fulfillment specializes the contract into one run-specific obligation set
Successful fulfillment SHALL specialize the general contract meanings into one
authoritative, run-specific obligation set and produce the seed of the authority
that governs those obligations.

#### Scenario: Runtime evidence is intentionally deferred
- **WHEN** the contract permits a declared participant or capability to become ready later
- **THEN** fulfillment MUST establish every meaning and compatibility decision knowable from the authored definition
- **AND** the accepted arrangement MUST retain the exact unsatisfied runtime obligation and evidence requirement
- **AND** dependent use MUST remain unavailable until accepted runtime evidence fulfills it

#### Scenario: A decision is deliberately made during execution
- **WHEN** authored behavior chooses among alternatives from changing inputs or accepted owned state
- **THEN** fulfillment MUST evaluate the decision policy, the bounds of its alternatives, their authority, and their possible effects against the active contract and profile
- **AND** when accepted, the arrangement MUST preserve that policy for execution rather than replacing it with one authoring-time outcome

#### Scenario: The strategy declares a mid-run transition
- **WHEN** an authored strategy declares a transition that changes the arrangement
- **THEN** fulfillment MUST verify the arrangement after that transition and derive its obligations, each with its readiness checkpoint
- **AND** those obligations MUST come into force only when the transition takes effect

### Requirement: Obligations are evaluated at the earliest authoritative evidence point
Every contract obligation SHALL be evaluated at the earliest lifecycle point
where the evidence needed to judge it is authoritative. A later realization or
transition stage SHALL evaluate new evidence against the current run-specific
obligation set and SHALL NOT reinterpret the authored strategy or independently
derive compatibility rules.

#### Scenario: Incompatibility is already knowable during fulfillment
- **WHEN** the authored selections, active contract, and known implementation conformance establish an incompatibility
- **THEN** fulfillment MUST reject it before materialization or preparation
- **AND** it MUST NOT defer the same judgment to a later producer

#### Scenario: Evidence requires a concrete realization
- **WHEN** an obligation depends on an artifact, live object, resolved relationship, optimization realization, or backend result that does not yet exist
- **THEN** the accepted arrangement MUST identify that obligation as unfulfilled at the applicable readiness checkpoint
- **AND** the later producer MUST supply evidence to the same contract-governed authority rather than define another validity rule

### Requirement: Readiness checkpoints are derived during strategy fulfillment
Strategy fulfillment SHALL derive and record the readiness checkpoints that
apply to the accepted arrangement and SHALL associate every obligation with the
first checkpoint at which it must be fulfilled. A use SHALL remain unavailable
while any obligation required by its checkpoint is unfulfilled.

#### Scenario: Core execution requires a participant later
- **WHEN** the contract permits a participant to remain declared and unbound after fulfillment but requires it for core execution
- **THEN** the accepted arrangement MUST record that materialization and any required preparation as obligations of the core-execution readiness checkpoint

### Requirement: Late choices are evaluated against current obligations
A bounded configuration choice known during fulfillment SHALL be evaluated then;
a request deliberately made later SHALL be evaluated before its capability use
begins against the run's current obligations.

#### Scenario: Capability request is made after fulfillment
- **WHEN** an accepted capability permits a bounded request to be supplied later
- **THEN** the request MUST be validated before that capability becomes ready
- **AND** later timing MUST NOT permit a choice that strategy fulfillment would have rejected under the same obligations

### Requirement: Normal Trainer construction cannot bypass strategy fulfillment
The normal Trainer construction path SHALL accept only an established run
arrangement, not an unchecked authored strategy or a separately assembled
strategy/mode/objective set.

#### Scenario: Caller presents an authored strategy directly
- **WHEN** a caller attempts to start Trainer runtime with a definition that has not passed strategy fulfillment
- **THEN** construction MUST fail before model materialization or optimization work begins

### Requirement: Contract surfaces remain consumer-defined
The contract system SHALL distinguish the universal meanings required by the
Trainer engine, named operations coordinated by the training pipeline, and
reusable feature contracts consumed during strategy authoring. Selection of a
feature or capability SHALL make its applicable obligations mandatory without
making it a universal core field.

#### Scenario: Capability is selected by one strategy
- **WHEN** a strategy selects sampling, caching, persistence, or another recognized capability
- **THEN** strategy fulfillment MUST apply that capability's complete applicable requirements
- **AND** strategies that do not select it MUST NOT be forced to provide a placeholder implementation

#### Scenario: Feature is used internally
- **WHEN** an authored strategy uses a representation or conditioning feature to implement accepted behavior
- **THEN** the feature MUST satisfy its author-facing contract
- **AND** the Trainer MUST NOT discover or coordinate the feature merely because it exists

### Requirement: Known behavior uses shared conformance knowledge
Acceptance of maintained library behavior SHALL use shared conformance
knowledge consistent with the active contract. Ordinary strategy authors SHALL
provide the declarations and evidence required by that knowledge but SHALL NOT
have to redefine ordinary validity rules for each strategy. The exact
representation and code ownership of checking logic remain to be derived.

#### Scenario: Maintained implementation is selected
- **WHEN** an authored strategy selects a known maintained implementation
- **THEN** strategy fulfillment MUST use the applicable shared conformance knowledge
- **AND** it MUST NOT accept a bare author assertion or role/type inference as proof

#### Scenario: A maintained region supplies shared semantic contracts
- **WHEN** supported compositions depend on a maintained region's boundary
- **THEN** its registered definition MUST expose the region, flow, resource and realization contracts that `workflow-composition` requires for those compositions
- **AND** conformance MUST derive obligations from the actual accepted structure and those definitions rather than author assertions, saved summaries or a whole-case selector
- **AND** this MUST NOT require every leaf computation to become a separate owner or expose its private mathematics

#### Scenario: Several strategies use the same supported coordination meaning
- **WHEN** ordinary authored arrangements use an offered gradient, handoff, advancement, or ownership policy
- **THEN** conformance MUST apply shared rules for that meaning
- **AND** authors MUST NOT recreate its ordinary validators or accept a weaker meaning through strategy-local checking

#### Scenario: Internal lowering changes while accepted meaning stays fixed
- **WHEN** preparation fuses, partitions, or otherwise realizes selected work differently
- **THEN** shared conformance MUST judge required owner-visible interactions, numerical and gradient policy, effects, outcomes, and recovery relationships
- **AND** equal final values or nominal type compatibility alone MUST NOT certify equivalence

#### Scenario: A supported composition is placed inside another supported scope
- **WHEN** maintained activities are composed through a different permitted nesting or request relationship
- **THEN** shared conformance MUST evaluate the actual region, flow, capture and lifetime contracts without selecting rules by fixture identity or exact operation count
- **AND** any target-specific restriction MUST remain an explicit realization limitation rather than a silently weaker meaning

#### Scenario: A callable is offered without the required region protocol
- **WHEN** an implementation would introduce input admission, pending work or authority beyond the maintained operation's declared boundary
- **THEN** fulfillment MUST reject that missing contract or require a supported semantic extension
- **AND** registering the callable MUST NOT itself establish the missing lifecycle or authority meaning

### Requirement: Custom behavior uses an explicit conformance or extension path
A custom implementation that preserves the active Trainer boundary SHALL file
explicit conformance to that boundary. Behavior that changes Trainer
responsibility or execution authority SHALL target an explicit supported
contract extension or execution/ownership profile.

#### Scenario: Custom operation preserves standard ownership
- **WHEN** a custom structured operation supplies model or objective computation while leaving standard Trainer mechanics intact
- **THEN** strategy fulfillment MUST judge it against the standard execution contract

#### Scenario: Experiment requests different authority
- **WHEN** an experiment requests control of a mechanic retained by the standard profile
- **THEN** strategy fulfillment MUST require a declared supported extension/profile describing the changed owners and exchanges
- **AND** it MUST reject nominal conformance to the unchanged standard contract

#### Scenario: Two-pass extension omits a required grant or recovery claim
- **WHEN** a custom region requests two backward passes and temporary parameter editing but lacks the intermediate gradient authority, scoped state access, accumulation compatibility, or recovery meaning required by that supported profile
- **THEN** fulfillment MUST reject the definition rather than treat a coarse backward/edit declaration as executable
- **AND** it MUST NOT grant optimizer advancement merely because a selected implementation uses an optimizer wrapper internally

#### Scenario: Backend evidence for an accepted extension is deferred
- **WHEN** a supported extension's authored ownership split is valid but safe parameter access, synchronization, or clean handback depends on the current backend realization
- **THEN** fulfillment MUST retain those exact unfulfilled obligations and the applicable readiness checkpoint
- **AND** governed realization MUST reject an incompatible backend before the imperative region executes

### Requirement: One accepted contract governs one run authority
One contract version and execution/ownership profile SHALL govern an accepted
run authority for its lifetime. The authority SHALL identify the current
arrangement and run-specific obligation revision. Permitted arrangement
amendments SHALL be ones the accepted arrangement declares, with obligations
derived at fulfillment under that same contract. A different contract version or Trainer extension SHALL require new
strategy fulfillment unless an explicit migration contract is later defined.

#### Scenario: Arrangement amendment retains the contract
- **WHEN** an accepted run brings declared participants into use or retires them through a permitted amendment
- **THEN** all affected obligations MUST be evaluated under the authority's existing contract and profile

#### Scenario: A change the arrangement does not declare is requested
- **WHEN** a running authority receives a change that its accepted arrangement does not declare
- **THEN** the authority MUST reject it without deriving new obligations
- **AND** making that change MUST require a new run with its own strategy fulfillment

#### Scenario: Caller requests another contract version
- **WHEN** live runtime behavior would require another core contract version or ownership profile
- **THEN** the existing authority MUST NOT silently adopt it
- **AND** new strategy fulfillment MUST be required
