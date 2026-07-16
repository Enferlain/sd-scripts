# component-qualified-selectors Specification

## Purpose
TBD - created by archiving change normalize-component-qualified-selectors. Update Purpose after archive.
## Requirements
### Requirement: Parameter dumps use component-qualified selector names
The inspection tool SHALL expose parameter-oriented selector names in the form `component.local_name`, where `component` is the public selector prefix derived from the family-declared loaded-component contract and `local_name` is the real runtime parameter path within that component.

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
Training-internal placeholders such as `denoiser`, `text_encoder1`, and `text_encoder2` SHALL NOT remain the canonical public selector source when the family-declared loaded-component contract defines the component selector prefixes.

#### Scenario: User writes a selector from dump output
- **WHEN** a user copies a selector prefix directly from the parameter dump output into a learning-rate group pattern
- **THEN** that selector surface SHALL match the same runtime parameters without translation into internal training placeholder names

#### Scenario: Future family exposes different top-level components
- **WHEN** a future family declares public selector prefixes that do not map to the old diffusion tuple
- **THEN** the public selector surface MUST still use those declared component prefixes
- **AND** it MUST NOT translate them through SD-shaped compatibility names

### Requirement: Obsolete block learning-rate config is removed
The config schema and defaults SHALL remove `optimizer.learning_rates.blocks` once component-qualified selector matching becomes the supported fine-grained targeting surface.

#### Scenario: Reading default optimizer config
- **WHEN** a user inspects the default optimizer config surface
- **THEN** `optimizer.learning_rates.blocks` SHALL no longer appear as an available learning-rate field

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

