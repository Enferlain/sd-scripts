## ADDED Requirements

### Requirement: Inspect supported model runtimes
The tool SHALL load a supported model through the repo's existing model support path and inspect the resulting real runtime objects rather than relying on a separate inspection-description layer or training pipeline selection state.

#### Scenario: Inspecting a supported model
- **WHEN** a user runs the tool for a supported model family with the required checkpoint inputs
- **THEN** the tool loads the model's real runtime components and bases all inspection output on those loaded objects

#### Scenario: Unsupported model family
- **WHEN** a user requests inspection for a model family the repo does not currently support
- **THEN** the tool fails with a clear unsupported-model error instead of inventing a partial or placeholder dump

### Requirement: Default output is parameter-oriented and component-ordered
The tool SHALL default to emitting named parameters grouped under existing top-level component names and ordered deterministically so users can reason about what is available for manual training parameter-group configuration.

#### Scenario: Parameter dump for a supported model
- **WHEN** a user runs the tool without selecting an alternate inspection view
- **THEN** the output lists named parameters under the model's top-level components using the existing component-name metadata

#### Scenario: Parameter dump reflects actual runtime state
- **WHEN** the tool emits the default parameter view
- **THEN** it reports parameters that exist on the loaded runtime objects rather than inferring training-enabled parameters from optimizer groups, mode selection, or config-derived trainability

### Requirement: Module inspection shows module types under the same component ordering
The tool SHALL provide a module inspection view that lists named modules and their types under the same top-level component ordering used for parameter inspection.

#### Scenario: Module view requested
- **WHEN** a user requests the module inspection view
- **THEN** the tool emits named modules grouped under top-level components and includes each module's runtime type

### Requirement: Optional state inspection includes buffers
The tool SHALL provide an optional state inspection view that includes both named parameters and named buffers for the loaded runtime objects.

#### Scenario: State view requested
- **WHEN** a user requests the state inspection view
- **THEN** the tool emits parameters and buffers grouped under top-level components

#### Scenario: Buffer metadata
- **WHEN** buffers are included in the state view
- **THEN** the output reflects the runtime buffer properties present on the loaded objects, including persistence and current `requires_grad` state

### Requirement: The tool must not require new dump-specific library APIs
The inspection workflow SHALL not depend on per-model dump-specific loader/package APIs or richer component-description metadata beyond the already-existing top-level component naming metadata.

#### Scenario: Tool implementation boundary
- **WHEN** the tool is implemented for supported model families
- **THEN** any inspection-specific orchestration remains in tool code and does not require new dump-specific public APIs to be added to the model loaders or packages
