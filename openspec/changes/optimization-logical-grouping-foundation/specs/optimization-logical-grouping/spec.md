## ADDED Requirements

### Requirement: Optimization plan exposes stable logical groups
The optimization layer SHALL expose a trainer-facing optimization plan that represents logical parameter groups separately from backend execution groups.

#### Scenario: Fine-tune mode builds a base-path plan
- **WHEN** the active fine-tune path prepares optimizer inputs for denoiser and text encoders
- **THEN** it SHALL produce an optimization plan containing stable logical groups for the trainable components
- **AND** the logical groups SHALL preserve deterministic ordering for trainer diagnostics and LR reporting

### Requirement: Trainer-facing diagnostics use logical-group metadata
Training diagnostics SHALL derive optimizer-group naming and parameter summaries from logical-group metadata instead of reconstructing meaning from runtime optimizer `param_groups`.

#### Scenario: Startup diagnostics report logical groups
- **WHEN** the trainer emits startup diagnostics after optimizer creation
- **THEN** it SHALL use logical-group metadata from the optimization plan to label groups
- **AND** it SHALL NOT require runtime optimizer groups to preserve trainer-facing group identity

### Requirement: LR reporting remains stable across runtime group rewrites
The optimization layer SHALL preserve stable LR reporting metadata even when wrappers or other runtime integrations rewrite execution optimizer groups internally.

#### Scenario: Scheduler logging reads plan ordering
- **WHEN** step logging records learning rates for the active optimizer
- **THEN** the reported LR names SHALL come from the optimization plan's logical-group ordering
- **AND** the reporting path SHALL NOT depend on runtime optimizer group labels being reconstructed heuristically

### Requirement: Base-path grouping remains independent from current adapter compatibility
The first logical-grouping foundation SHALL target the active base fine-tune path without requiring the current adapter/PEFT optimizer seam to define the architecture.

#### Scenario: Adapter seam is not required for base-path planning
- **WHEN** the optimization plan foundation is introduced for the base fine-tune path
- **THEN** the implementation SHALL be valid without redesigning the current adapter optimizer contract
- **AND** future adapter work MAY target the optimization-plan seam later without blocking this change
