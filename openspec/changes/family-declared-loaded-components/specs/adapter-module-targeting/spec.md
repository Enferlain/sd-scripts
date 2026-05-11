## ADDED Requirements

### Requirement: Adapter component scope derives from declared loaded components
Optimization-owned adapter targeting SHALL derive candidate top-level component scope from the family-declared loaded-component contract rather than from the fixed `text_encoders/vae/denoiser` loader tuple.

#### Scenario: Resolving targetable top-level components
- **WHEN** an adapter training run is prepared
- **THEN** optimization MUST evaluate the family-declared loaded components and their targeting-relevant semantics to determine which top-level components are in scope
- **AND** it MUST NOT assume the only possible top-level component buckets are text encoders, a VAE, and one denoiser

#### Scenario: Future family declares non-standard targetable components
- **WHEN** a model family declares targetable top-level components beyond the current diffusion backbone examples
- **THEN** adapter targeting MUST be able to represent and expand those components without adding new universal loader slots

### Requirement: Adapter target provenance preserves declared component identity
Resolved adapter targets SHALL preserve the declared component identity provided by the loaded-component contract.

#### Scenario: Building component-root adapter targets
- **WHEN** the system builds component-root or module-resolved adapter targets
- **THEN** each target MUST preserve the declared component key and public label from the loaded-component contract
- **AND** downstream adapter runtime code MUST receive that declared identity through repo-owned provenance rather than through reconstructed slot names
