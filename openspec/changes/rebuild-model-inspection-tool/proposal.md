## Why

The current parameter dump work has repeatedly drifted into the wrong shape: it mixes tool concerns with library concerns, leaks training/config pipeline concepts into a model inspection task, and keeps accreting ad hoc glue instead of becoming a clear tool. We need a stable contract for this work now because the repo already supports loading real models, and the inspection tool should simply ride that reality rather than inventing a second integration surface.

## What Changes

- Replace the current incremental `dump_named_parameters.py` direction with a proper model inspection tool whose identity is: load a supported model through the repo's existing loading support, inspect the real runtime objects, and emit deterministic YAML views.
- Make parameter inspection the primary use case: the default output SHALL list named parameters, ordered under existing top-level component names, so users can identify what is available for manual training parameter-group configuration.
- Support module inspection as a first-class view so users can see module types and hierarchy under the same component ordering.
- Keep state/buffer inspection as a secondary bonus view instead of the primary contract.
- Remove dump-specific library-facing seams that were added only to support the tool; the tool SHALL not require new per-model inspection APIs beyond the already-existing component name metadata.
- Document repo-flow issues exposed by this work, especially places where loading support is brittle or too awkward for tooling to consume cleanly.

## Capabilities

### New Capabilities
- `model-inspection-tool`: Inspect supported model runtimes and emit ordered YAML views of parameters, modules, and optional state grouped under existing component names.

### Modified Capabilities

## Impact

- Affected code: `tools/model_management/dump_named_parameters.py`, `library/models/parameter_dump.py`, and any temporary dump-specific edits that were added to model loading code.
- Affected systems: model loading/tooling integration for supported families, YAML inspection output shape, and the workflow used to manually reason about parameter-group training configuration.
- Repo-flow impact: this change should surface any weak seams in the existing loading/tooling story and record them explicitly instead of masking them with new dump-specific abstraction layers.
