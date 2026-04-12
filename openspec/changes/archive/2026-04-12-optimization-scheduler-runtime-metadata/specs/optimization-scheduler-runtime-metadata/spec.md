## ADDED Requirements

### Requirement: Optimization plan exposes scheduler/runtime metadata
The optimization layer SHALL represent scheduler ownership and scheduler target selection as explicit optimization-plan metadata.

#### Scenario: Optimizer phase stores scheduler runtime metadata
- **WHEN** the optimizer phase prepares the optimizer and scheduler for a plan-aware training path
- **THEN** the stored optimization plan SHALL include scheduler/runtime metadata describing scheduler mode and scheduler target
- **AND** that metadata SHALL be available before scheduler construction completes

### Requirement: Scheduler entrypoint consumes plan scheduler/runtime metadata
The shared scheduler construction entrypoint SHALL use plan-provided scheduler/runtime metadata when available.

#### Scenario: Plan metadata selects scheduler target
- **WHEN** scheduler construction receives an optimization plan whose scheduler/runtime metadata targets the base optimizer
- **THEN** the scheduler SHALL attach to the optimizer runtime indicated by the plan metadata
- **AND** it SHALL NOT need to reconstruct that decision heuristically for that call path

### Requirement: Scheduler metadata migration preserves compatibility
The first scheduler/runtime metadata slice SHALL preserve current scheduler behavior for callers that do not provide an optimization plan.

#### Scenario: Legacy callers still use fallback resolution
- **WHEN** scheduler construction is invoked without plan metadata
- **THEN** the existing capability- and wrapper-based resolution path SHALL remain available
- **AND** scheduler behavior SHALL remain compatible with current wrapper and schedule-free integration expectations
