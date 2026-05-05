## ADDED Requirements

### Requirement: Repo-owned VeRA runtime realizes from resolved adapter targets
The VeRA PEFT method SHALL build and load VeRA runtime state from
optimization-owned resolved adapter targets rather than from method-owned model
traversal or architecture-specific discovery logic.

#### Scenario: Building a VeRA runtime for training
- **WHEN** `AdapterMode` prepares a PEFT run whose selected method is `vera`
- **THEN** the repo-owned VeRA runtime MUST construct VeRA state only from the
  resolved adapter targets in the build request
- **AND** it MUST NOT become the owner of target discovery, regex traversal, or
  component-selection policy

#### Scenario: Loading VeRA weights against model context
- **WHEN** VeRA weights are loaded for continuation, merge, or inference-style
  setup
- **THEN** the repo-owned VeRA runtime MUST reconstruct itself against the
  resolved adapter targets in the from-weights request
- **AND** it MUST participate through `LoadedAdapterRuntime`

### Requirement: Repo-owned VeRA runtime owns shared projection state
The repo-owned VeRA runtime SHALL own shared projection tensors used by all
adapted VeRA targets within one runtime instance.

#### Scenario: Realizing shared projection tensors for a VeRA run
- **WHEN** a VeRA runtime is created for a set of resolved adapter targets
- **THEN** it MUST create and own shared VeRA projection state for that runtime
- **AND** per-target VeRA modules MUST reference the shared projection state
  rather than duplicating full projection tensors per target

#### Scenario: Serving per-target VeRA modules from shared state
- **WHEN** the VeRA runtime realizes multiple supported targets with different
  input or output dimensions
- **THEN** each target-bound VeRA module MUST derive its effective projection
  view from the runtime-owned shared state
- **AND** the runtime MUST preserve correct target-local behavior for each
  adapted module

#### Scenario: Realizing VeRA against linear-like wrapper targets
- **WHEN** a resolved adapter target is a supported linear-like wrapper such as
  Transformers `Conv1D`
- **THEN** the repo-owned VeRA runtime MUST treat it as a valid VeRA target
- **AND** it MUST preserve the target module's stored weight layout when
  reconstructing or merging the effective VeRA delta

### Requirement: Repo-owned VeRA method owns method-local config and artifact policy
The forward VeRA PEFT config surface SHALL stay method-local and the VeRA
method SHALL own persistence behavior for both shared projections and per-target
trainable parameters.

#### Scenario: Translating VeRA config into runtime settings
- **WHEN** the active PEFT method branch is `adapter.peft.vera`
- **THEN** the VeRA config translator MUST build normalized runtime settings
  only from method-local VeRA inputs such as rank, dropout, initialization, and
  projection persistence policy

#### Scenario: Saving and reloading VeRA method state
- **WHEN** the repo-owned VeRA runtime saves or reloads VeRA method state
- **THEN** it MUST use repo-owned VeRA state-dict helpers and method-local
  checkpoint keys for both shared runtime state and per-target trainable state
- **AND** it MUST NOT require the HF PEFT VeRA tuner as a steady-state runtime
  dependency

#### Scenario: Reconstructing shared projections when projections are omitted
- **WHEN** the repo-owned VeRA runtime exports an artifact with
  `save_projection=False`
- **THEN** it MUST omit the shared `vera_A` / `vera_B` tensors from the
  exported state dict instead of introducing empty sentinel keys
- **AND** it MUST persist enough method-local metadata to reconstruct the
  shared projection bank deterministically on reload when the artifact format
  supports metadata

### Requirement: Repo-owned VeRA trainable refs preserve target provenance
The repo-owned VeRA runtime SHALL expose trainable parameter refs that preserve
source resolved-target provenance for all trainable per-target VeRA parameters.

#### Scenario: Describing VeRA trainable parameters for optimization grouping
- **WHEN** optimization asks the VeRA runtime for trainable parameter refs
- **THEN** each returned ref MUST identify the VeRA algorithm, component,
  component key, and target path
- **AND** it MUST preserve the source resolved-target provenance when the VeRA
  runtime can identify the concrete source target

#### Scenario: Keeping grouping independent from VeRA internal sharing
- **WHEN** optimization groups trainable VeRA parameters
- **THEN** it MUST consume repo-owned trainable refs and provenance
- **AND** it MUST NOT depend on internal details of VeRA shared projection
  ownership to identify which target a trainable parameter belongs to
