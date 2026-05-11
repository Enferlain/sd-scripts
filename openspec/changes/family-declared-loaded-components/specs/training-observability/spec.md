## ADDED Requirements

### Requirement: Backbone observability derives from declared loaded components
Training observability SHALL derive top-level backbone component structure from the family-declared loaded-component contract rather than from the fixed trainer slot trio.

#### Scenario: Building fine-tune startup diagnostics
- **WHEN** startup diagnostics are emitted for a fine-tune-style run
- **THEN** the observability path MUST consume components supplied through the loaded-component contract or a mode-owned filtered view of that contract
- **AND** it MUST NOT reconstruct top-level component structure from hardcoded `text_encoders/vae/denoiser` slots

#### Scenario: Building resource startup breakdowns
- **WHEN** startup component-memory estimates are emitted
- **THEN** the resource-monitor path MUST consume the declared component view supplied by the runtime or mode
- **AND** it MUST NOT assume the only possible backbone buckets are text encoders, a VAE, and one denoiser

### Requirement: Observability preserves family-declared top-level ordering
Human-facing startup summaries and top-level resource breakdowns SHALL preserve family-declared component ordering unless a consumer defines a different ordering rule explicitly.

#### Scenario: Rendering component diagnostics
- **WHEN** startup component diagnostics are rendered for a model family
- **THEN** the default component order MUST follow the family-declared loaded-component order
- **AND** observability code MUST NOT impose an SD-shaped component order as a generic default

#### Scenario: Rendering future non-SD-shaped families
- **WHEN** a future family declares top-level components with a different structure than current SD families
- **THEN** observability output MUST still render those components in the declared order
- **AND** it MUST NOT hide or rename them through compatibility slot assumptions
