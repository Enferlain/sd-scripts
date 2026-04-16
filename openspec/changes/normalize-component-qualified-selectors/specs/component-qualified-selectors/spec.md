## ADDED Requirements

### Requirement: Parameter dumps use component-qualified selector names
The inspection tool SHALL expose parameter-oriented selector names in the form `component.local_name`, where `component` is the model-facing component label derived from existing model package metadata and `local_name` is the real runtime parameter path within that component.

#### Scenario: Dumping SDXL parameter selectors
- **WHEN** the inspection tool dumps parameter selectors for an SDXL runtime
- **THEN** the output SHALL use names such as `unet.input_blocks.0.0.weight` and `clip_l.text_model.embeddings.token_embedding.weight`

#### Scenario: Dumping SD3 parameter selectors
- **WHEN** the inspection tool dumps parameter selectors for an SD3 runtime
- **THEN** the output SHALL use names such as `mmdit.<path>`, `clip_l.<path>`, `clip_g.<path>`, `t5xxl.<path>`, and `vae.<path>` rather than training-internal placeholders

### Requirement: Fine-grained optimizer matching uses the same selector namespace
Fine-grained optimizer group matching SHALL evaluate `groups` and `groups_file` patterns against the same component-qualified selector names exposed by the inspection dump.

#### Scenario: Regex targets a specific component
- **WHEN** a learning-rate group pattern matches `^unet\\.` or `^mmdit\\.`
- **THEN** only parameters in that matching component SHALL be selected

#### Scenario: Regex targets a specific text encoder
- **WHEN** a learning-rate group pattern matches `^clip_l\\.` or `^t5xxl\\.`
- **THEN** only parameters in that matching component SHALL be selected

### Requirement: Internal training placeholders are not the public selector surface
Training-internal labels such as `denoiser`, `text_encoder1`, and `text_encoder2` SHALL NOT remain the canonical public selector namespace for parameter dumps or fine-grained optimizer matching.

#### Scenario: User writes a selector from dump output
- **WHEN** a user copies a selector prefix directly from the parameter dump output into a learning-rate group pattern
- **THEN** that selector surface SHALL match the same runtime parameters without translation into internal training placeholder names

### Requirement: Obsolete block learning-rate config is removed
The config schema and defaults SHALL remove `optimizer.learning_rates.blocks` once component-qualified selector matching becomes the supported fine-grained targeting surface.

#### Scenario: Reading default optimizer config
- **WHEN** a user inspects the default optimizer config surface
- **THEN** `optimizer.learning_rates.blocks` SHALL no longer appear as an available learning-rate field
