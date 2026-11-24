## MODIFIED Requirements

### Requirement: Shared optimization target refs
The repository SHALL define shared, policy-neutral target references for
component, module, and parameter structure. A semantic target reference SHALL
preserve its authority-qualified participant/component/substructure meaning
independently from any current live-object projection. Live modules and
parameters SHALL be available only through revision-pinned consumer views. The
existing capability and code namespace SHALL NOT establish semantic policy
ownership.

#### Scenario: Representing a module target
- **WHEN** an accepted consumer resolves a module target from current state
- **THEN** the target MUST identify its participant, declared model component, component-local path, public selector, module type, and source revisions
- **AND** any live module handle MUST be scoped to the resolved projection rather than treated as target identity

#### Scenario: Representing a parameter target
- **WHEN** an accepted consumer resolves a parameter target
- **THEN** the target MUST identify its participant, declared model component, parameter path, public selector, owner provenance, and source revisions
- **AND** current `nn.Parameter` identity MUST NOT become its durable semantic identity

#### Scenario: Representing a component target
- **WHEN** an accepted consumer represents a top-level model component as a target
- **THEN** the target MUST preserve its accepted participant context plus the declared public component label and internal component key

### Requirement: Target refs are policy-neutral
Shared target refs SHALL describe selected structure, provenance, and source
dependencies without owning strategy authoring, participant lifecycle,
semantic adapter target intent, concrete target-resolution policy, trainability,
grouping, or adapter behavior.

#### Scenario: Fine-tune mode consumes parameter targets
- **WHEN** accepted training-subject intent resolves to parameter targets
- **THEN** Trainer optimization MAY use their current projections to realize trainability
- **AND** grouping and lifecycle policy MUST remain outside the target-ref value

#### Scenario: Adapter mode consumes module targets
- **WHEN** governed PEFT realization resolves authored adapter target intent to module targets
- **THEN** the selected method MAY use the scoped live modules for adapter realization
- **AND** neither the target refs nor their live handles may become an independent targeting policy or authoritative participant binding

### Requirement: Existing grouping behavior remains stable
Introducing authority-qualified target refs SHALL preserve current
component-qualified selector and learning-rate behavior unless another explicit
delta changes it.

#### Scenario: Fine-tune group matching remains parameter-selector based
- **WHEN** learning-rate groups use `match` patterns
- **THEN** those patterns MUST continue matching component-qualified parameter selectors

#### Scenario: Adapter grouping remains component-LR based in this change
- **WHEN** adapter trainable refs are grouped before a later explicit adapter-grouping change
- **THEN** grouping MUST continue to use the accepted component learning-rate policy

### Requirement: Top-level component target refs derive from declared loaded components
Shared target refs for model-owned structure SHALL derive top-level model
component identity from family declarations and participant context from the
accepted run authority. They SHALL NOT derive either meaning from fixed
`text_encoders/vae/denoiser` slots.

#### Scenario: Building component target refs
- **WHEN** an accepted consumer builds a component target from an accepted model participant
- **THEN** the ref MUST preserve the declared component key and public label plus the authority-qualified participant identity and source revisions

#### Scenario: Multiple components share one semantic role
- **WHEN** a family declares multiple components with a shared role
- **THEN** target refs MUST preserve each as distinct and MUST NOT collapse them into a universal role bucket

### Requirement: Module and parameter refs expand from declared components
Module and parameter refs SHALL expand from a coherent current projection of an
accepted participant's declared top-level components. They SHALL preserve
component-qualified selector compatibility, provenance, and source revisions.

#### Scenario: Building module or parameter refs beneath a declared component
- **WHEN** lower-level targets are resolved beneath an accepted top-level component
- **THEN** they MUST inherit declared component identity and accepted participant context
- **AND** they MUST become stale when their source projection's dependencies no longer hold
