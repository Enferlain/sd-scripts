## MODIFIED Requirements

### Requirement: Repo-owned VeRA runtime realizes from resolved adapter targets
The VeRA PEFT method SHALL build and load VeRA runtime state from
governed PEFT target-resolution results rather than from method-owned model
traversal or architecture-specific discovery logic. The strategy SHALL author
semantic target intent, and the method SHALL then act through method-specific
adapter realization and
participant/relationship transitions rather than through `AdapterMode`
ownership.

#### Scenario: Building a VeRA runtime for training
- **WHEN** the pipeline materializes an accepted adapter participant whose selected method is `vera`
- **THEN** the repo-owned VeRA runtime MUST construct VeRA state only from the resolved adapter targets in the realization request
- **AND** it MUST NOT become the owner of target discovery, regex traversal, component-selection policy, or training orchestration

#### Scenario: Loading VeRA weights against model context
- **WHEN** VeRA weights are loaded for continuation, merge, or inference-style setup
- **THEN** the repo-owned VeRA runtime MUST reconstruct itself against the resolved adapter targets in the from-weights request
- **AND** it MUST participate through `LoadedAdapterRuntime` or its accepted successor
