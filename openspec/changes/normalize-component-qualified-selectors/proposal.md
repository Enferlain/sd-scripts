## Why

The current fine-grained optimizer group matcher exposes internal training placeholders such as `denoiser` and `text_encoder1`, while the inspection dumps expose component-grouped local names without a canonical fully qualified selector namespace. That mismatch makes regex selection ambiguous, leaks internal runtime bookkeeping into user-facing config, and diverges from the model-facing component names already declared in each model package `__init__.py`.

## What Changes

- Normalize the public selector namespace for parameter dumps and fine-grained optimizer matching to `component.local_name`, using existing `NAMED_PARAMETER_COMPONENT_NAMES` metadata as the source of component labels.
- Update the inspection dump output so parameter-oriented YAML emits component-qualified selector names instead of bare local names.
- Update fine-grained optimizer group matching to operate on the same component-qualified selector namespace shown in the dumps.
- Remove the obsolete `optimizer.learning_rates.blocks` surface from config schema, defaults, and related runtime handling.
- Treat the normalized component-qualified selector namespace as the shared external surface for future non-parameter module selectors.

## Capabilities

### New Capabilities
- `component-qualified-selectors`: Defines the canonical external selector namespace used by inspection dumps and fine-grained optimizer matching.

### Modified Capabilities
- None.

## Impact

- Affected code: `tools/model_management/dump_named_parameters.py`, optimizer grouping/matching code, optimizer dataclasses/default configs, and related tests.
- Affected user-facing config: `optimizer.learning_rates.groups`, `optimizer.learning_rates.groups_file`, and parameter dump YAML output.
- Follow-on benefit: future module-selector work can reuse the same selector namespace instead of introducing another matching surface.
