## Why

The active training/runtime contract still assumes every model family can be loaded and reasoned about as `text_encoders + vae + denoiser`. That assumption now leaks through trainer setup, startup diagnostics, resource accounting, adapter targeting, optimizer grouping, selector generation, and model-inspection tooling, making future non-SD-shaped model families harder to add without reopening the same architectural boundary in multiple places.

The repo is also about to revisit metadata as a broader topic, which makes this the right time to define the real top-level component contract instead of reinforcing the current SD-shaped tuple under new names. We need a family-declared loaded-component surface that generic code can consume through roles/capabilities and stable component metadata rather than through fixed slot names.

## What Changes

- Introduce a new repo-owned loaded-component contract rooted in family-declared top-level model components instead of the fixed `text_encoders/vae/denoiser` tuple.
- **BREAKING** Replace the current `ModelLoadingStrategy.load_target_model()` tuple contract with a family-declared loaded-component surface that becomes the primary trainer/runtime representation.
- Define how generic training/runtime code consumes component structure through family declarations and component roles/capabilities instead of hardcoded SD-shaped slot assumptions.
- Move trainer, startup diagnostics, resource accounting, selector generation, optimization grouping, adapter targeting, and model-inspection tooling onto the loaded-component surface.
- Define a no-compatibility transition: the repo should not preserve the old trio as the long-term contract solely because current families already fit it.

## Capabilities

### New Capabilities
- `loaded-model-components`: Family-declared top-level loaded-component contract for trainer/runtime code, metadata, targeting, and tooling.

### Modified Capabilities
- `training-observability`: Startup diagnostics and resource-oriented observability must derive model component structure from the loaded-component contract rather than from the fixed SD-shaped trainer slots.
- `component-qualified-selectors`: Public selector generation must derive component prefixes from declared loaded components rather than from the old tuple contract and its placeholder assumptions.
- `adapter-module-targeting`: Adapter target resolution must consume family-declared loaded components instead of expanding only the hardcoded `text_encoders/vae/denoiser` buckets.
- `optimization-target-refs`: Shared target refs must remain component-based while sourcing top-level component structure from family-declared loaded components.

## Impact

- Affected code includes the model-loading strategy contract, trainer setup/state, diagnostics/resource-monitor inputs, adapter runtime context/targeting, optimization grouping/target refs, selector generation, and the model-inspection dump tool.
- This change is intentionally architecture-level and breaking for internal repo seams; it is meant to remove a narrowing assumption before more model families and metadata work depend on it.
- Future model families with different top-level component structure, inherent adapter-like components, multiple conditioning towers, or non-diffusion layouts should be able to participate without redefining the same boundary in each subsystem.
