## ADDED Requirements

### Requirement: Top-level component target refs derive from declared loaded components
Shared optimization target refs SHALL derive their top-level component identity from the family-declared loaded-component contract rather than from the fixed `text_encoders/vae/denoiser` tuple.

#### Scenario: Building component target refs
- **WHEN** the optimization layer builds a component target ref for a loaded top-level model component
- **THEN** that ref MUST preserve the declared component key and public component label from the loaded-component contract
- **AND** it MUST NOT require synthetic SD-shaped slot names as the source of component identity

#### Scenario: Multiple components share one semantic role
- **WHEN** a family declares multiple loaded components that share a semantic role
- **THEN** optimization target refs MUST preserve each component as a distinct top-level targetable component
- **AND** they MUST NOT collapse those components into one universal role bucket

### Requirement: Module and parameter refs expand from declared components
Module and parameter target refs SHALL continue to expand from top-level component identity while sourcing that identity from declared loaded components.

#### Scenario: Building module or parameter refs beneath a declared component
- **WHEN** module-level or parameter-level target refs are built
- **THEN** they MUST inherit their top-level component identity from the corresponding declared loaded component
- **AND** they MUST continue to expose component-qualified selector and provenance information built from that declared component identity
