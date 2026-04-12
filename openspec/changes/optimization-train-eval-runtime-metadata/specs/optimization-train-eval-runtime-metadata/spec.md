## ADDED Requirements

### Requirement: Optimization plan exposes optimizer train/eval runtime metadata
The optimization layer SHALL represent optimizer-owned train/eval runtime behavior as explicit optimization-plan metadata for plan-aware paths.

#### Scenario: Optimizer phase stores train/eval runtime metadata
- **WHEN** the optimizer phase prepares a plan-aware optimizer runtime
- **THEN** the stored optimization plan SHALL indicate whether the optimizer runtime participates in train/eval mode switching
- **AND** that metadata SHALL be available to training orchestration before runtime mode transitions occur

### Requirement: Training orchestration uses shared optimizer runtime mode helpers
Training orchestration SHALL switch optimizer runtime mode through shared helper functions instead of directly depending on raw callback-pair handling at each call site.

#### Scenario: Shared helper switches optimizer runtime mode
- **WHEN** a plan-aware training path enters or exits an eval-mode section
- **THEN** orchestration SHALL use a shared helper to switch optimizer runtime state
- **AND** the helper SHALL safely no-op when optimizer runtime metadata indicates no train/eval switching is required

### Requirement: Train/eval runtime metadata migration preserves compatibility
The first optimizer-runtime metadata slice SHALL preserve existing behavior for compatibility paths that still rely on callback pairs.

#### Scenario: Legacy callback behavior remains available
- **WHEN** an optimizer path still provides callback pairs during the migration
- **THEN** the new train/eval runtime metadata SHALL not break that path
- **AND** schedule-free optimizer behavior SHALL remain compatible with current expectations
