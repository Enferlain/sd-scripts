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

### Requirement: Capability selection is validated before use
An authored strategy SHALL explicitly provide every capability it exposes or
requires. Execution configuration MAY request an exposed capability within its
bounds. Missing or incompatible requested capabilities SHALL fail before their
runtime trigger.

#### Scenario: Full-model persistence is unavailable
- **WHEN** configuration requests a full-model product that the accepted strategy does not provide
- **THEN** the strategy acceptance check or later request validation MUST reject the run before checkpoint time

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
SHALL own traversal and storage coordination. Cache results SHALL identify
their accepted dependencies and SHALL become stale when those guarantees fail.

#### Scenario: Autoencoder replacement invalidates latent cache
- **WHEN** accepted autoencoder state changes and a latent cache depends on that realization
- **THEN** the cache MUST no longer be treated as current
- **AND** unrelated conditioning cache MAY remain current only through an explicit valid dependency set

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
or storage service.

#### Scenario: Artifact save succeeds but runtime state is incomplete
- **WHEN** a semantic model product is written without the authority and optimization state required for exact continuation
- **THEN** the product result MUST NOT claim resumable-runtime support

### Requirement: Capability results feed observation without transferring ownership
Metadata, logging, metrics, reports, and resource systems MAY consume accepted
capability requests, results, transition facts, and failures. Observation SHALL
NOT mutate capability or binding state or roll back an accepted result.

#### Scenario: Result logging fails
- **WHEN** a capability completed and its observation sink fails afterward
- **THEN** the capability's accepted state/result MUST remain authoritative
