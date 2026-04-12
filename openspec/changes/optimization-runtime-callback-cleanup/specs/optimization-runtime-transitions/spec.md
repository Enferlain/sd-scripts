## ADDED Requirements

### Requirement: Plan-aware optimizer builds SHALL not require legacy runtime callbacks
The optimization layer SHALL allow plan-aware mode implementations to return optimizer build metadata without providing separate optimizer train/eval callback functions.

#### Scenario: Fine-tune build returns a plan-aware result
- **WHEN** the fine-tune mode constructs an optimizer build result for the base path
- **THEN** the result includes the optimizer identity, optimizer instance, and optimization plan metadata
- **AND** the result does not require separate optimizer train/eval callback fields to describe runtime transitions

### Requirement: Shared orchestration SHALL treat callback pairs as compatibility-only state
The optimizer phase SHALL keep supporting the legacy tuple contract for compatibility-oriented callers, but shared orchestration SHALL derive plan-aware runtime transitions from optimization-plan metadata instead of mode-provided callback pairs.

#### Scenario: Optimizer phase normalizes a legacy tuple build result
- **WHEN** a mode returns the legacy optimizer tuple including train/eval callbacks
- **THEN** the optimizer phase preserves that compatibility state for the caller
- **AND** the plan-aware runtime contract remains optional for that path

#### Scenario: Orchestration runs a plan-aware training or eval transition
- **WHEN** shared orchestration needs to switch optimizer runtime state for a plan-aware path
- **THEN** it uses the plan-owned runtime metadata and shared optimizer runtime helper
- **AND** it does not depend on trainer callback-pair attributes being populated
