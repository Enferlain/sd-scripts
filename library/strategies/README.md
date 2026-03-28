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
