## MODIFIED Requirements

### Requirement: Repo-owned LoRA runtime realizes from resolved adapter targets
The LoRA PEFT method SHALL build and load LoRA runtime state from
optimization-owned resolved adapter targets rather than from method-owned model
traversal or discovery logic. Realization SHALL occur through the accepted
adapter capability and participant/relationship transitions rather than
through `AdapterMode` ownership.

#### Scenario: Building a LoRA runtime for training
- **WHEN** the pipeline materializes an accepted adapter participant whose selected method is `lora`
- **THEN** the repo-owned LoRA runtime MUST construct LoRA modules only from the resolved adapter targets in the realization request
- **AND** it MUST NOT become the owner of model traversal, regex target discovery, component-selection policy, or training orchestration

#### Scenario: Loading LoRA weights against model context
- **WHEN** LoRA weights are loaded for continuation, merge, or inference-style setup
- **THEN** the repo-owned LoRA runtime MUST reconstruct itself against the resolved adapter targets in the from-weights request
- **AND** it MUST participate through `LoadedAdapterRuntime` or its accepted successor
