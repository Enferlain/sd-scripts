## MODIFIED Requirements

### Requirement: Shared optimization target refs
The repository SHALL define shared, policy-neutral target references for
component, module, and parameter structure. A semantic target reference SHALL
preserve its authority-qualified participant and selected substructure meaning
independently from any current live-object projection. Live modules and
parameters SHALL be available only through revision-pinned consumer views. The
existing capability and code namespace SHALL NOT establish semantic policy
ownership.

Targets with genuine model-component provenance SHALL also preserve declared
component identity and component-local addressing. Accepted participant-owned
substructure without that provenance SHALL remain representable through the
same shared target meanings without inventing a family component. Genuine host
target associations SHALL remain provenance, not a substitute for the semantic
participant that owns the selected state.

#### Scenario: Representing a module target
- **WHEN** an accepted consumer resolves a module target from current state
- **THEN** the target MUST identify its participant, selected module substructure, public selector, module type, source revisions, and dependency evidence
- **AND** where the module has declared model-component provenance, it MUST preserve that component identity, component-local path, and component-qualified selector
- **AND** any live module handle MUST be scoped to the resolved projection rather than treated as target identity

#### Scenario: Representing a parameter target
- **WHEN** an accepted consumer resolves a parameter target
- **THEN** the target MUST identify its participant, selected parameter substructure, public selector, owner provenance, source revisions, and dependency evidence
- **AND** where the parameter has declared model-component provenance, it MUST preserve that component identity, component-local parameter path, and component-qualified selector
- **AND** current `nn.Parameter` identity MUST NOT become its durable semantic identity

#### Scenario: Representing a component target
- **WHEN** an accepted consumer represents a top-level model component as a target
- **THEN** the target MUST preserve its accepted participant context plus the declared public component label and internal component key

#### Scenario: Resolving a standalone participant's learned parameter
- **WHEN** accepted training-subject intent selects a learned parameter owned by an independently managed participant with no family-declared component provenance
- **THEN** Trainer optimization MUST be able to resolve that subject through the shared participant-qualified parameter target meaning and a current revision-pinned live projection
- **AND** the target MUST preserve the selected substructure, public selector, owner provenance, and source dependencies without creating or borrowing a family component identity
- **AND** missing catalog or model-structural associations MUST NOT prevent otherwise valid target construction

#### Scenario: Rejecting invented component ancestry
- **WHEN** a candidate target claims a declared component association that its accepted participant and actual provenance do not establish
- **THEN** the resolving consumer MUST reject that association rather than accept an invented component or substitute another participant's ownership
- **AND** genuine adapter host-target provenance MUST remain distinct from the participant ownership of the adapter's learned state

#### Scenario: Rejecting stale standalone parameter access
- **WHEN** a standalone parameter target's live projection no longer satisfies its participant or source dependencies
- **THEN** the consuming operation MUST reject that stale live access and resolve a current permitted projection before dependent use
- **AND** a surviving parameter object or matching public selector MUST NOT establish current authority or revive a retired participant

### Requirement: Target refs preserve selector compatibility
Targets with model-component provenance SHALL preserve the existing
component-qualified selector surface exposed by model inspection and
fine-grained grouping. Targets without that provenance SHALL expose their
accepted participant/substructure selector meaning without impersonating a
declared public model component. The concrete selector representation for that
case SHALL preserve accepted subject selection without changing existing
component-backed selector behavior.

#### Scenario: Building a parameter target selector
- **WHEN** a parameter target ref is built for a parameter inside a public
  component such as `unet`, `clip_l`, or `mmdit`
- **THEN** its selector MUST use the form `component.local_parameter_path`

#### Scenario: Building a module target selector
- **WHEN** a module target ref is built for a module inside a public component
  such as `unet`, `clip_l`, or `mmdit`
- **THEN** its selector MUST use the form `component.local_module_path`

#### Scenario: Selecting standalone participant substructure
- **WHEN** a consumer selects module or parameter substructure without declared model-component provenance
- **THEN** the selector MUST retain its accepted participant-qualified subject meaning
- **AND** it MUST NOT require a synthetic public component prefix to make that subject selectable

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
- **THEN** those patterns MUST continue matching existing component-qualified parameter selectors for component-backed targets

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
Module and parameter refs for component-backed structure SHALL expand from a
coherent current projection of an accepted participant's declared top-level
components and preserve component-qualified selector compatibility, provenance,
and source revisions. Refs for accepted participant-owned substructure outside
that surface SHALL instead preserve the owning participant and selected
substructure under the same authority and freshness requirements. Neither
absence of family components nor an optional catalog association SHALL permit
raw live-object identity to replace semantic target identity.

#### Scenario: Building module or parameter refs beneath a declared component
- **WHEN** lower-level targets are resolved beneath an accepted top-level component
- **THEN** they MUST inherit declared component identity and accepted participant context
- **AND** they MUST become stale when their source projection's dependencies no longer hold
