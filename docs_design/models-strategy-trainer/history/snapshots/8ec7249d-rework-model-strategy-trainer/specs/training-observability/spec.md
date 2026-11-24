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
