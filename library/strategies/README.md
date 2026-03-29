"""Strategy layer notes and conventions."""

# Strategies

The strategy layer owns model-family behavior that the trainer depends on:
loading, tokenization, text encoding, caching, diffusion behavior, denoiser
calling, checkpointing, sampling, and other training-facing runtime policy.

## Contract philosophy

The current contracts in [base/contracts.py](/mnt/d/Projects/sd-scripts/library/strategies/base/contracts.py)
are not final. They reflect what SD and SDXL needed when the first version of
the strategy system was introduced, not the final shape of the architecture.

Contracts should be treated as capability contracts:

- A contract should exist only for behavior that is genuinely shared across
  model families.
- A model family does not need to implement every possible kind of behavior.
- Not every method on every contract should be assumed to apply to every model.
- A behavior should become part of a shared contract only when multiple models
  need the trainer or shared runtime to depend on it explicitly.

In practice, this means we should avoid forcing a new model into an existing
contract just because the contract already exists. When a model exposes new
behavior, the right question is whether that behavior is a stable cross-model
responsibility, an optional hook, or something that should remain model-specific
for now.

## Current shape

The strategy layer now has four distinct roles:

- `base/contracts.py`: required strategy facets that shared trainer/runtime code
  depends on directly
- `base/features.py`: optional strategy features that only some model families
  implement
- `shared/`: cross-family strategy-owned behavior that is genuinely reused
- `<family>/`: family-specific strategy facets and assembly

This means "strategy code" is broader than the required `TrainingStrategy`
surface. A behavior can still belong to `library/strategies/` even if it is an
optional feature or a family-local implementation detail.

## Required vs optional

Use the following rule:

- put a behavior in `base/contracts.py` when shared trainer/runtime code needs
  to depend on it for active model families
- put a behavior in `base/features.py` when it is a real strategy capability,
  but not every active model family is expected to support it

Current example:

- `WeightedPromptStrategy` lives in `base/features.py` because weighted prompt
  tokenization / encoding is real strategy behavior, but it is not universal
  across all active families

## Shared vs family-owned strategy code

Do not create a shared module just because multiple families use the same
library or component type. Shared strategy code should exist only when the same
training-facing behavior is reused together and changes for the same reasons.

Current examples:

- shared CLIP-family prompt tokenization behavior lives under
  `library/strategies/shared/clip/`
- tokenizer bootstrap/loading stays outside that behavior layer because it is
  component/bootstrap code, not strategy behavior
- family-specific assembly stays under `library/strategies/<family>/`

## Conditioning

Conditioning is now treated as a real strategy concern.

- `ConditioningStrategy.resolve_conditioning(...)` is the shared high-level seam
- each family owns its own conditioning implementation under
  `library/strategies/<family>/conditioning.py`
- family payload shapes stay local unless a stronger convention emerges later

The intent is to standardize the responsibility, not to force all families into
the same helper steps or concrete conditioning object shape.
