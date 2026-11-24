## Purpose

Define preparation attempts that produce coherent usable execution state without losing participant identity or publishing stale results.

## ADDED Requirements

### Requirement: Preparation derives from coherent accepted state
Preparation SHALL be one complete attempt-scoped job derived from a coherent accepted snapshot. Its membership SHALL include every participant required for the requested preparation, independently of trainability. An attempt SHALL be allowed to use multiple backend calls. Backend composites SHALL NOT replace participant identity.

#### Scenario: Frozen member needs preparation
- **WHEN** a frozen participant is required by accepted execution and needs backend preparation
- **THEN** it is included despite not belonging to an optimization unit

### Requirement: Preparation publishes one coherent result
All fallible preparation and validation that can precede publication SHALL do so. Authority routes and associated Trainer backend/optimization state SHALL become current together only after freshness and completeness checks succeed. Publication SHALL expose no intermediate mixed state.

#### Scenario: Stale candidate
- **WHEN** a relevant binding, relationship or owned preparation state changes before a candidate is published
- **THEN** that stale candidate is rejected
- **AND** it cannot restore guarantees withdrawn by a newer transition

### Requirement: Preparation failure respects mutation behavior
Replacement-only failure SHALL discard unpublished candidates without changing previously current state. Before destructive in-place mutation, the system SHALL withdraw affected usability guarantees. Failure after such mutation SHALL leave affected state unavailable until explicit recovery succeeds.

#### Scenario: In-place preparation fails
- **WHEN** backend preparation mutates current state and then fails
- **THEN** old execution routes are not advertised as usable merely because no replacement was published
- **AND** the failure identifies the affected state requiring recovery
