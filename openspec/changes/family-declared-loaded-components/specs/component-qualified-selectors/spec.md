## MODIFIED Requirements

### Requirement: Parameter dumps use component-qualified selector names
The inspection tool SHALL expose parameter-oriented selector names in the form `component.local_name`, where `component` is the public selector prefix derived from the family-declared loaded-component contract and `local_name` is the real runtime parameter path within that component.

#### Scenario: Dumping SDXL parameter selectors
- **WHEN** the inspection tool dumps parameter selectors for an SDXL runtime
- **THEN** the output SHALL use names such as `unet.input_blocks.0.0.weight` and `clip_l.text_model.embeddings.token_embedding.weight`

#### Scenario: Dumping SD3 parameter selectors
- **WHEN** the inspection tool dumps parameter selectors for an SD3 runtime
- **THEN** the output SHALL use names such as `mmdit.<path>`, `clip_l.<path>`, `clip_g.<path>`, `t5xxl.<path>`, and `vae.<path>` rather than training-internal placeholders

### Requirement: Internal training placeholders are not the public selector surface
Training-internal placeholders such as `denoiser`, `text_encoder1`, and `text_encoder2` SHALL NOT remain the canonical public selector source when the family-declared loaded-component contract defines the component selector prefixes.

#### Scenario: User writes a selector from dump output
- **WHEN** a user copies a selector prefix directly from the parameter dump output into a learning-rate group pattern
- **THEN** that selector surface SHALL match the same runtime parameters without translation into internal training placeholder names

#### Scenario: Future family exposes different top-level components
- **WHEN** a future family declares public selector prefixes that do not map to the old diffusion tuple
- **THEN** the public selector surface MUST still use those declared component prefixes
- **AND** it MUST NOT translate them through SD-shaped compatibility names
