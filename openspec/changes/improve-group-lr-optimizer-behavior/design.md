## Context

The optimization layer now supports fine-grained named parameter groups with explicit per-group learning rates, including file-backed groups under `optimizer.learning_rates.groups_file`. In the current flow, grouping logic already materializes execution groups with per-group `lr`, but optimizer construction still treats `optimizer.learning_rates.base` as the constructor-level LR source. That creates a mismatch between the grouped runtime plan and the factory boundary, especially for configs that intentionally set `base: null` to mean “no fallback baseline.”

This mismatch has two visible consequences:

- valid group-driven configs can fail during optimizer initialization with backend-specific errors such as bitsandbytes rejecting `lr=None`
- the user-facing semantics of `null` and `0` are harder to reason about because baseline selection, freezing, and optimizer construction are not fully aligned

The change is cross-cutting enough to warrant an explicit design because it touches config semantics, trainability selection, optimizer factory behavior, and error-reporting expectations.

## Goals / Non-Goals

**Goals:**

- Make grouped optimizer configs behave consistently when execution groups already define explicit learning rates.
- Keep the meaning of `null`, `0`, and positive LR values stable across grouping, trainability, validation, and optimizer construction.
- Prefer repo-owned behavior and repo-owned error messages over backend-specific crashes.
- Preserve the distinction between “inherit,” “frozen,” and “train” without requiring a full config redesign in this change.

**Non-Goals:**

- Introduce a brand-new mode-based training config surface.
- Redesign PEFT-specific optimizer grouping beyond the existing fail-fast boundary.
- Guarantee support for every third-party optimizer backend in configurations it fundamentally does not support.
- Change the higher-level model/runtime meaning of optional modules versus always-present modules.

## Decisions

### Explicit group LRs are authoritative for grouped optimizer execution

When optimizer execution groups already define `lr`, those values are the runtime source of truth. The factory layer must not treat `learning_rates.base` as the authoritative LR for that grouped path.

Rationale:

- grouped configs already encode the relevant training intent
- re-deriving that intent from `base` is lossy
- the current behavior is what creates the `base: null` mismatch

Alternatives considered:

- Continue treating `base` as the constructor default even for fully explicit groups. Rejected because it preserves the current bug and couples grouping semantics to a fallback value the user intentionally disabled.
- Synthesize a constructor LR from the first group or another “safe” group. Rejected because it is a workaround that invents configuration not explicitly chosen by the user.

### `null` means inherit or “no fallback”; `0` means frozen baseline path

The change will standardize LR semantics across the optimization layer:

- `base: <positive>` provides the fallback baseline LR
- `base: null` means there is no fallback baseline LR
- component LR `null` means “inherit from base if one exists”
- component LR `0` means the baseline component path is frozen, not absent from runtime
- component LR `> 0` means the baseline component path trains at that LR

Rationale:

- users need a stable mental model that matches observed behavior
- “frozen” is more accurate than “disabled” for denoisers and text encoders that still participate in runtime
- this preserves compatibility with the current LR-driven config shape while clarifying intent

Alternatives considered:

- Treat `0` as literal zero-step optimization while still including the component in optimizer groups. Rejected because it is less user-friendly and creates a confusing half-trained state.
- Overload `null` to also mean frozen. Rejected because it would collapse inheritance and freezing into one ambiguous sentinel.

### Unsupported constructor-LR requirements must fail clearly, not leak backend errors

If a backend initialization path still cannot be satisfied from explicit grouped params alone, the repo should raise a clear, repo-owned error describing the unsupported configuration instead of allowing the backend to surface a low-level type/value exception.

Rationale:

- users configure the repo, not bitsandbytes internals
- unsupported combinations should fail at the repo boundary with actionable guidance
- this keeps future backend-specific handling explicit rather than accidental

Alternatives considered:

- Allow backend constructor errors to surface directly. Rejected because the current behavior already demonstrates how confusing those failures are.

## Risks / Trade-offs

- [Cross-layer semantic drift] → Update grouping helpers, config help text, validation, and tests together so the same `null`/`0` meanings are enforced everywhere.
- [Backend-specific constructor assumptions] → Centralize constructor-LR handling in the optimizer factory and add backend-focused regression coverage for grouped configs.
- [User confusion during transition] → Use “frozen” consistently in docs/help text and reserve “disabled” for truly absent optional features.

## Migration Plan

1. Update the optimizer factory boundary so grouped execution plans no longer blindly forward `learning_rates.base` as constructor LR.
2. Align grouping/trainability helpers and config help text on the standardized `null` versus `0` meanings.
3. Add regression tests for grouped fine-tune configs, especially `base: null` plus explicit named groups and backend-specific optimizer coverage.
4. Roll back by restoring the previous factory behavior only if a regression shows that grouped params are not sufficient for a supported backend path.

## Open Questions

- Do any registered optimizers besides bitsandbytes still reject grouped initialization unless a constructor-level LR is present, even when every param group defines its own `lr`?
- Should future follow-up work model backend constructor requirements explicitly in registry metadata, or is clean omission of constructor LR sufficient once the current bug is fixed?
