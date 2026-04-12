## ADDED Requirements

### Requirement: Optimization plan exposes explicit execution groups
The optimization layer SHALL represent execution optimizer groups as explicit optimization-plan-owned objects instead of only as an unnamed compatibility payload.

#### Scenario: Base-path plan carries execution groups explicitly
- **WHEN** the active fine-tune path builds an optimization plan
- **THEN** the plan SHALL expose explicit execution groups alongside logical groups
- **AND** the execution-group model SHALL remain materializable into the legacy optimizer dict payload required for optimizer construction

### Requirement: Base-path grouping uses execution-oriented plan terminology
The shared base-path grouping and fine-tune optimizer preparation flow SHALL build optimization plans through explicit execution-group naming.

#### Scenario: Shared grouping helper returns execution groups
- **WHEN** the shared fine-tune grouping helper assembles optimizer inputs
- **THEN** it SHALL return execution-group-oriented data rather than a generically named parameter-group payload
- **AND** the fine-tune mode SHALL consume that execution-group-oriented data when building the optimization plan

### Requirement: Execution-group migration preserves compatibility
The first execution-group slice SHALL preserve compatibility for existing optimizer-construction and test paths while moving the architecture toward explicit execution-group modeling.

#### Scenario: Legacy materialization still works
- **WHEN** existing optimizer-construction code materializes execution groups for the shared factory path
- **THEN** the materialized optimizer dict payload SHALL remain behaviorally compatible with the current optimizer factory expectations
- **AND** compatibility accessors MAY remain available during the migration
