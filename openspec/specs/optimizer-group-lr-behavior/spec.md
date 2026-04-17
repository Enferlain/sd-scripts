# optimizer-group-lr-behavior Specification

## Purpose
TBD - created by archiving change improve-group-lr-optimizer-behavior. Update Purpose after archive.
## Requirements
### Requirement: Grouped optimizer learning rates are authoritative
When the optimization layer materializes explicit execution groups with per-group learning rates, the system SHALL treat those group learning rates as the authoritative runtime optimizer LRs for that grouped path.

#### Scenario: Group-only fine-tune config builds from explicit group learning rates
- **WHEN** a fine-tune configuration sets `optimizer.learning_rates.base` to `null` and provides explicit named groups that each resolve to execution groups with `lr`
- **THEN** the optimizer construction path SHALL use the grouped learning-rate configuration without failing due to a missing fallback base LR

### Requirement: Baseline learning-rate semantics distinguish inherit, frozen, and train
The system SHALL interpret baseline learning-rate fields consistently across trainability selection, grouping, and validation.

#### Scenario: Null component LR inherits from a defined base fallback
- **WHEN** `optimizer.learning_rates.base` is positive and a component baseline LR field is `null`
- **THEN** that component baseline path SHALL inherit the positive base LR

#### Scenario: Null component LR does not create a baseline training path when no fallback exists
- **WHEN** `optimizer.learning_rates.base` is `null` and a component baseline LR field is `null`
- **THEN** that component baseline path SHALL remain unselected for baseline training unless an explicit named group selects parameters from it

#### Scenario: Zero baseline LR freezes the component baseline path
- **WHEN** a component baseline LR field is `0`
- **THEN** that component baseline path SHALL be treated as frozen rather than trainable

### Requirement: Unsupported grouped LR configurations fail with repo-owned errors
The system SHALL reject unsupported grouped learning-rate configurations with clear repo-owned errors instead of leaking backend-specific constructor type/value failures.

#### Scenario: Repo rejects grouped LR configuration before backend type errors leak through
- **WHEN** the optimizer factory cannot satisfy a configured grouped learning-rate plan for the selected optimizer backend
- **THEN** the repo SHALL raise an error that describes the unsupported grouped learning-rate configuration at the repo boundary

