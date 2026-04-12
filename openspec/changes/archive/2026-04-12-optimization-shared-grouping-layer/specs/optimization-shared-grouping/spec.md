## ADDED Requirements

### Requirement: Shared grouping module builds base-path logical groups
The optimization layer SHALL provide a shared grouping module that builds base-path logical optimization groups without requiring `FineTuneMode` to assemble those groups inline.

#### Scenario: Fine-tune grouping flows through shared optimization code
- **WHEN** the active fine-tune path prepares optimizer inputs for the denoiser and trainable text encoders
- **THEN** the logical-group and execution-group assembly SHALL be produced through shared optimization-layer grouping helpers
- **AND** `FineTuneMode` SHALL remain responsible for trainable selection rather than duplicating group-construction policy

### Requirement: Shared grouping preserves current logical-group behavior
The shared grouping module SHALL preserve the deterministic ordering, labels, and learning-rate assignment used by the current base fine-tune path.

#### Scenario: Denoiser and text encoders keep stable ordering
- **WHEN** the shared grouping helper builds groups for a fine-tune run with a trainable denoiser and selected text encoders
- **THEN** the resulting logical groups SHALL appear in the same trainer-facing order as the current path
- **AND** the resulting group labels and declared learning rates SHALL match the configured denoiser and text-encoder behavior

### Requirement: Shared grouping remains independent from adapter compatibility
The first shared grouping slice SHALL target the active base fine-tune path without requiring the current adapter optimization seam to conform to the new grouping helper.

#### Scenario: Base-path grouping does not depend on adapter integration
- **WHEN** the shared grouping module is introduced for the fine-tune path
- **THEN** the implementation SHALL be valid without redesigning the current adapter-owned optimizer preparation path
- **AND** future adapter work MAY adopt the shared grouping seam later without blocking this change
