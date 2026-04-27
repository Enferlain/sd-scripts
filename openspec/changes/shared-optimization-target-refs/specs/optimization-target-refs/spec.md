## ADDED Requirements

### Requirement: Shared optimization target refs
The optimization layer SHALL define a shared target-reference model that can
represent component, module, and parameter targets.

#### Scenario: Representing a module target
- **WHEN** optimization resolves a module target
- **THEN** the target ref MUST identify its kind as `module`
- **AND** it MUST include the public component label, internal component key,
  component-local path, component-qualified selector, live module object, and
  module type

#### Scenario: Representing a parameter target
- **WHEN** optimization resolves a parameter target
- **THEN** the target ref MUST identify its kind as `parameter`
- **AND** it MUST include the public component label, internal component key,
  component-local parameter path, component-qualified selector, live parameter
  object, and owner-module provenance when available

#### Scenario: Representing a component target
- **WHEN** optimization represents a top-level component as a target
- **THEN** the target ref MUST identify its kind as `component`
- **AND** it MUST preserve both public component label and internal component key

### Requirement: Target refs preserve selector compatibility
Optimization target refs SHALL use the same component-qualified selector
surface that the model inspection and fine-grained grouping paths expose.

#### Scenario: Building a parameter target selector
- **WHEN** a parameter target ref is built for a parameter inside a public
  component such as `unet`, `clip_l`, or `mmdit`
- **THEN** its selector MUST use the form `component.local_parameter_path`

#### Scenario: Building a module target selector
- **WHEN** a module target ref is built for a module inside a public component
  such as `unet`, `clip_l`, or `mmdit`
- **THEN** its selector MUST use the form `component.local_module_path`

### Requirement: Target refs are policy-neutral
Shared target refs SHALL describe selected structure and provenance without
owning training-mode lifecycle or optimizer-group policy.

#### Scenario: Fine-tune mode consumes parameter targets
- **WHEN** fine-tune mode resolves trainable base-model parameters
- **THEN** the mode MAY use parameter target refs to set `requires_grad`
- **AND** optimization MUST still own the grouping policy for those parameters

#### Scenario: Adapter mode consumes module targets
- **WHEN** adapter mode prepares an adapter run
- **THEN** the mode MAY pass module target refs into adapter runtime
  construction
- **AND** optimization MUST still own target-selection and grouping policy

### Requirement: Existing grouping behavior remains stable
Introducing shared target refs SHALL NOT change the current user-facing
fine-tune or adapter grouping behavior.

#### Scenario: Fine-tune group matching remains parameter-selector based
- **WHEN** fine-tune learning-rate groups are configured with `match` patterns
- **THEN** those patterns MUST continue matching component-qualified parameter
  selectors

#### Scenario: Adapter grouping remains component-LR based in this change
- **WHEN** adapter trainable refs are grouped for optimizer construction
- **THEN** grouping MUST continue to use component learning-rate policy unless a
  later change defines adapter-specific group selector behavior
