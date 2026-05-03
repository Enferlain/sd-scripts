# repo-owned-lora-method Specification

## Purpose
TBD - created by archiving change refresh-repo-owned-lora. Update Purpose after archive.
## Requirements
### Requirement: Repo-owned LoRA runtime realizes from resolved adapter targets
The LoRA PEFT method SHALL build and load LoRA runtime state from
optimization-owned resolved adapter targets rather than from method-owned model
traversal or discovery logic.

#### Scenario: Building a LoRA runtime for training
- **WHEN** `AdapterMode` prepares a PEFT run whose selected method is `lora`
- **THEN** the repo-owned LoRA runtime MUST construct LoRA modules only from
  the resolved adapter targets in the build request
- **AND** it MUST NOT become the owner of model traversal, regex target
  discovery, or component-selection policy

#### Scenario: Loading LoRA weights against model context
- **WHEN** LoRA weights are loaded for continuation, merge, or inference-style
  setup
- **THEN** the repo-owned LoRA runtime MUST reconstruct itself against the
  resolved adapter targets in the from-weights request
- **AND** it MUST participate through `LoadedAdapterRuntime`

### Requirement: Repo-owned LoRA method owns LoRA module behavior
The absorbed LoRA method SHALL own its module math, validation, merge/export
behavior, and state-dict reconstruction in repo-owned code.

#### Scenario: Realizing a supported LoRA target module
- **WHEN** the repo-owned LoRA runtime realizes a supported Linear or Conv
  target
- **THEN** the repo-owned LoRA module implementation MUST own the LoRA
  parameter layout, forward behavior, and merged-weight reconstruction for that
  target

#### Scenario: Saving and reloading LoRA method state
- **WHEN** the repo-owned LoRA runtime saves or reloads LoRA method state
- **THEN** it MUST use repo-owned LoRA state-dict helpers and method-local
  checkpoint keys
- **AND** it MUST NOT require the legacy built-in LoRA adapter class as the
  steady-state runtime dependency

### Requirement: Repo-owned LoRA config stays method-local and policy-light
The forward LoRA PEFT config surface SHALL keep LoRA settings method-local and
MUST NOT re-own optimization policy through LoRA-specific compatibility knobs.

#### Scenario: Translating LoRA config into runtime settings
- **WHEN** the active PEFT method branch is `adapter.peft.lora`
- **THEN** the LoRA method config translator MUST build normalized runtime
  settings only from method-local LoRA settings such as rank, alpha, dropout,
  Conv rank, Conv alpha, and method-local validation

#### Scenario: Excluding optimizer-policy concerns from LoRA method config
- **WHEN** a setting primarily expresses target-selection, grouping, or
  optimizer policy rather than LoRA method math
- **THEN** the LoRA method refresh MUST NOT treat that setting as part of the
  forward repo-owned LoRA method surface

### Requirement: Repo-owned LoRA trainable refs preserve target provenance
The repo-owned LoRA runtime SHALL expose trainable parameter refs that preserve
the source resolved-target provenance for supported LoRA modules.

#### Scenario: Describing LoRA trainable parameters for optimization grouping
- **WHEN** optimization asks the LoRA runtime for trainable parameter refs
- **THEN** each returned ref MUST identify the LoRA algorithm, component,
  component key, and target path
- **AND** it MUST preserve the source resolved-target provenance when the LoRA
  runtime can identify the concrete source target

#### Scenario: Keeping grouping independent from LoRA internals
- **WHEN** optimization groups trainable LoRA parameters
- **THEN** it MUST consume repo-owned trainable refs and provenance
- **AND** it MUST NOT depend on legacy LoRA naming quirks or built-in adapter
  wrapper internals as the grouping interface

