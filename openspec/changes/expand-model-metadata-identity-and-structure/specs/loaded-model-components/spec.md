## MODIFIED Requirements

### Requirement: Model loading returns the loaded-component surface
The active model-loading strategy contract SHALL return one shared typed loading-result shape containing the family-declared loaded-component surface plus ordered source observations, source selections, component-to-selection bindings, decisions, transformations, and limitations known by the successful loader instead of the fixed `text_encoders/vae/denoiser` tuple, an evidence-free component tuple, or family-specific provenance fields.

#### Scenario: Loading a training model
- **WHEN** `ModelLoadingStrategy.load_target_model()` loads a training-capable model family
- **THEN** it MUST return the model version together with the family-declared loaded-component surface in a typed loading result
- **AND** the loaded-component surface MUST remain the authoritative top-level component representation for downstream runtime code
- **AND** the result MUST preserve the successful source selection and component-materialization evidence known at that boundary through the shared collections

#### Scenario: Families have different source layouts
- **WHEN** SD, SDXL, SD3, or a future family loads a different number or arrangement of sources/components
- **THEN** each family MUST populate the same typed loading-result collections
- **AND** central consumers MUST NOT require family-specific source fields or family-name branches

#### Scenario: Trainer stores the primary loaded-component state
- **WHEN** trainer setup receives loaded model state from the strategy layer
- **THEN** the trainer MUST treat the loaded-component surface as its primary top-level model representation
- **AND** generic runtime code MUST NOT depend on the old tuple fields as the long-term contract
- **AND** the trainer MUST retain the typed loading/provenance state required for catalog resolution and realization composition filing

#### Scenario: Deferred loading updates components
- **WHEN** `load_denoiser_lazily()` or another family materialization boundary changes component presence or component-to-selection binding
- **THEN** it MUST return an updated typed loading result or typed update preserving the authoritative component surface and new materialization evidence
- **AND** it MUST NOT discard the prior loading evidence required to explain the final composition
