## ADDED Requirements

### Requirement: Component-qualified selectors identify target refs
The component-qualified selector namespace SHALL be used by shared optimization
target refs for both module and parameter targets.

#### Scenario: Parameter target selector matches inspection output
- **WHEN** a parameter target ref is built for a model parameter
- **THEN** its selector MUST match the component-qualified parameter name that
  the inspection tool exposes for that parameter

#### Scenario: Module target selector uses the same component prefix
- **WHEN** a module target ref is built for a model module
- **THEN** its selector MUST use the same public component prefix used by
  parameter selectors for that component
- **AND** the remaining selector path MUST be the module's component-local path

### Requirement: Internal component keys stay separate from public selectors
Shared target refs SHALL preserve internal component keys separately from public
component-qualified selector strings.

#### Scenario: Public selector differs from internal component key
- **WHEN** a model family exposes a public component label such as `unet`,
  `clip_l`, `clip_g`, `mmdit`, or `t5xxl`
- **THEN** target refs MUST preserve that label in the public selector
- **AND** target refs MUST also preserve the internal component key used by
  training code
