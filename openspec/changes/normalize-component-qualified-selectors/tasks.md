## 1. Normalize the selector namespace

- [x] 1.1 Replace training-internal public matcher names (`denoiser`, `text_encoder1`, `text_encoder2`) with model-facing component labels derived from existing `NAMED_PARAMETER_COMPONENT_NAMES`.
- [x] 1.2 Add or update shared selector-name helpers so dumps and optimizer matching build the same `component.local_name` namespace.

## 2. Align dumps and matching

- [x] 2.1 Update the parameter dump output to emit component-qualified selector names consistently across supported model families.
- [x] 2.2 Update fine-grained optimizer group matching to evaluate patterns against the same component-qualified selector names shown in the dump output.
- [x] 2.3 Add focused tests covering SDXL and SD3-style component names in both dump output and optimizer group matching.

## 3. Remove obsolete surface and capture follow-up

- [x] 3.1 Remove `optimizer.learning_rates.blocks` from config dataclasses, defaults, and any stale runtime/tests.
- [x] 3.2 Update changelog and any related planning notes to document the canonical component-qualified selector namespace and the removal of `blocks`.
