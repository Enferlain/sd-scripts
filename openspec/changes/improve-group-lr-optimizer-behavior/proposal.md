## Why

Fine-grained optimizer groups can already carry explicit per-group learning rates, but the optimizer factory still treats `optimizer.learning_rates.base` as the constructor-level LR source. That breaks valid group-driven configs such as `base: null` plus explicit named groups, causes backend-specific crashes like bitsandbytes rejecting `lr=None`, and leaves the user-facing meaning of `null` versus `0` unclear right where trainability is being controlled.

## What Changes

- Make optimizer construction respect explicit group learning rates as the primary runtime source when grouped params already define `lr`, instead of blindly forwarding `learning_rates.base` into the optimizer constructor path.
- Define and document the user-facing LR semantics consistently:
  - `null` means inherit from fallback, or “no fallback” when `base` itself is `null`
  - `0` means frozen for the baseline component path rather than “disabled”
  - positive values mean train with that LR
- Tighten validation and factory behavior so unsupported LR combinations fail clearly rather than surfacing backend-specific type/value errors.
- Add focused regression coverage for grouped optimizer construction, `base: null` group-only configs, and the frozen-versus-inherited LR behavior.

## Capabilities

### New Capabilities
- `optimizer-group-lr-behavior`: Defines the contract for how grouped learning rates, constructor-level optimizer initialization, and user-facing `null`/`0` LR semantics interact.

### Modified Capabilities
- None.

## Impact

- Affected code: `library/optimization/optimizer_factory.py`, optimizer grouping/trainability helpers, config validation, optimizer dataclasses/default config help text, and related tests.
- Affected user-facing config: `optimizer.learning_rates.base`, `optimizer.learning_rates.denoiser`, `optimizer.learning_rates.text_encoders`, `optimizer.learning_rates.groups`, and `optimizer.learning_rates.groups_file`.
- Affected behavior: group-only fine-tune configs should either build cleanly from explicit group LRs or fail with repo-owned validation/factory errors instead of backend constructor crashes.
