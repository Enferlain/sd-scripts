## MODIFIED Requirements

### Requirement: Training observability uses repo-owned structured signals
The training system SHALL expose repo-owned structured observability signals
for human console summaries, tracker metrics, and report-generation paths.
Signals SHALL derive from accepted arrangement, transition, optimization,
capability, execution, and artifact facts rather than requiring each sink to
reconstruct views independently from raw runtime objects.

#### Scenario: Startup diagnostics are consumed by more than one sink
- **WHEN** startup diagnostics are emitted for a training run
- **THEN** the system MUST produce structured summary data before sink-specific formatting
- **AND** console output and report-generation paths MUST be able to consume that shared structured summary data

#### Scenario: Training layers do not define sink-local formats
- **WHEN** Trainer, accepted behavior, capability, or pipeline code emits observability facts
- **THEN** those producers MUST NOT be forced to own final sink-specific formatting rules for every consumer

### Requirement: Training and observability ownership stay separate
Training execution SHALL own timing and accepted fact/result production.
Observability SHALL own formatting, routing, buffering, and persistence of
those observations. Observability SHALL consume accepted arrangement,
transition, optimization, capability, execution, and artifact facts rather
than introspecting an active strategy, `TrainingMode`, or unrestricted Trainer
state.

#### Scenario: Training phase emits a fact
- **WHEN** Trainer or accepted behavior produces a lifecycle event, result, transition, or metric
- **THEN** that producer MAY decide when and which accepted fact is emitted
- **AND** observability-owned code MUST decide how it is formatted, routed, buffered, or persisted

#### Scenario: Observability does not become orchestration owner
- **WHEN** observability records participant, route, optimization, or capability state
- **THEN** it MUST consume an accepted projection or result
- **AND** it MUST NOT mutate or reconstruct canonical current state

### Requirement: Backbone observability derives from declared loaded components
Training observability SHALL derive top-level model/backbone structure from
family-declared loaded components within an accepted, scoped participant view.
It SHALL NOT use fixed Trainer slots, `TrainingMode`-owned filtered state, or an
active strategy object as the canonical source.

#### Scenario: Building fine-tune startup diagnostics
- **WHEN** startup diagnostics are emitted for base-model training
- **THEN** observability MUST consume the accepted component and optimization projections for that run
- **AND** it MUST NOT reconstruct structure from hardcoded text-encoder, VAE, denoiser, or primary-trainable fields

#### Scenario: Building resource startup breakdowns
- **WHEN** component memory estimates are emitted
- **THEN** resource observation MUST use an accepted component view and its declared ordering/labels
- **AND** it MUST NOT assume the only backbone buckets are current diffusion roles

## ADDED Requirements

### Requirement: Whole-run observations preserve owner scopes and independent progress
Structured observations and reports SHALL preserve logical run and
execution-session scope and the owner, activity, request, input, attempt,
participant/relationship, unit, revision, and state-provenance associations
required to interpret the supplied fact. Independent activity coordinates
SHALL remain distinguishable; one global step, input, or display label SHALL
NOT substitute for all cross-owner identity. Known, skipped, failed, uncertain,
and unattempted outcomes SHALL remain distinguishable wherever the accepted
consumer depends on them. Reporting SHALL NOT strengthen completion, atomicity,
replay, or numerical parameter-change guarantees beyond the origin result.

#### Scenario: Producer completes independently of training
- **WHEN** an independently progressing producer makes a result ready before its consuming action is due
- **THEN** observation MUST preserve the producer's own progress and actual dependency provenance
- **AND** it MUST NOT report input consumption or optimization contribution merely because production completed

#### Scenario: Two units have different outcomes from one action
- **WHEN** one optimization unit returns, another has an uncertain outcome, and later mechanics remain unattempted
- **THEN** reporting MUST preserve their separate known, uncertain, and unattempted outcomes and their common attempt associations
- **AND** a single action-success flag MUST NOT erase those distinctions or prove numerical parameter change

#### Scenario: Several inputs feed one accepted action
- **WHEN** an action consumes several admitted inputs from different source scopes
- **THEN** required observations MUST retain each input-to-attempt association
- **AND** one input identifier MUST NOT be used as a substitute for the action's identity

#### Scenario: Same logical run resumes in another execution session
- **WHEN** exact same-run restoration establishes a new execution session
- **THEN** observations MUST distinguish the new session and attempts while retaining the restored logical run and runtime identities
- **AND** late observations from the earlier session MUST NOT masquerade as new current execution

### Requirement: Diagnostics and resource accounting use accepted subject and ownership evidence
Startup, live, and final summaries SHALL consume scoped component,
optimization, capability, progress, and measurement facts rather than derive
canonical topology or ownership from mode/strategy class names, fixed Trainer
slots, or unrestricted runtime objects. Declared ordering, public labels,
adapter provenance, and aliases SHALL remain available for presentation without
becoming identity selectors. Resource observations SHALL preserve their basis,
measurement/estimate distinction, physical scope, owner evidence, and relevant
validity. Physical sharing or deduplication SHALL remain observation-local and
SHALL NOT merge logical participants or semantic optimization units. Unsupported
owner attribution SHALL remain an accounting gap rather than an inferred claim.

#### Scenario: Compound arrangement has several models and a producer
- **WHEN** a summary describes a run with student/teacher participants, multiple optimization units, and an independent input activity
- **THEN** it MUST render their supplied scopes and relationships without rebuilding the arrangement from one family table or one optimizer object
- **AND** family-specific labels and ordering MUST remain usable within their declared views

#### Scenario: Component address is reused after retirement
- **WHEN** resource facts for a retired participant and its successor share a component label or authored address
- **THEN** accounting MUST preserve their supplied owner identity and validity scopes rather than merge them by that label

#### Scenario: Shared physical state appears in several views
- **WHEN** an authorized measurement producer supplies physical alias/shared-storage evidence for several accepted views
- **THEN** resource consumers MAY deduplicate within that observation's supported scope
- **AND** they MUST retain the distinct logical participant/unit associations and MUST NOT generalize that deduplication into runtime identity

#### Scenario: Only a phase label supports a resource measurement
- **WHEN** measurement evidence identifies timing or a phase but does not establish a resource owner
- **THEN** observation MAY retain the measurement
- **AND** accounting MUST NOT invent participant or operation ownership from that timing alone

### Requirement: Observation degradation is separate from execution and required feedback
Observation sinks SHALL report delivery/degradation according to their accepted
policy without rolling back reached execution or publication outcomes. Optional
telemetry MAY use bounded drop/buffering policies, but required history,
provenance, product metadata, restoration state, and algorithmic feedback SHALL
NOT be silently treated as disposable metrics. Required feedback SHALL remain
an accepted owner-to-owner runtime exchange independent of whether a copy is
logged. Retrying observation delivery SHALL NOT repeat computation, optimizer
advancement, a transition, or algorithmic feedback. Latency-critical execution
SHALL NOT require synchronous optional telemetry storage or repeated static
topology discovery solely for sink delivery.

#### Scenario: Tracker fails after a successful product publication
- **WHEN** persistence has established a successful product outcome and an optional tracker fails to receive it
- **THEN** that product outcome MUST remain authoritative and observation degradation MUST be reported separately

#### Scenario: Optional telemetry drops adaptive-loss metrics
- **WHEN** the optional observation buffer drops metrics also copied from a required adaptive-feedback exchange
- **THEN** the accepted adaptive owner MUST still receive the required feedback through its runtime exchange
- **AND** flushing or retrying the metric copy MUST NOT update that owner again

#### Scenario: Missing observation follows an uncertain backend action
- **WHEN** a backend action may have reached effects but its optional completion observation is missing
- **THEN** reporting MUST preserve the available uncertainty or gap
- **AND** missing telemetry MUST NOT prove that the action never ran or authorize replay

#### Scenario: Low-cost prepared work has optional high-frequency metrics
- **WHEN** optional telemetry is produced repeatedly in a latency-critical prepared activity
- **THEN** sink delivery MUST use its bounded policy without requiring synchronous optional storage there
- **AND** the metrics path MUST NOT rediscover the full accepted topology on every invocation
