## Why

Optimizer train/eval transitions are still handled as a loose pair of callbacks derived from schedule-free detection. That works, but it leaves another runtime concern outside the optimization plan even though the plan is becoming the orchestration-facing source of truth for optimizer behavior.

## What Changes

- Add explicit optimizer runtime train/eval metadata to the optimization plan.
- Populate that metadata centrally for plan-aware optimizer paths.
- Add shared helper functions for switching optimizer runtime state between train and eval.
- Update orchestration points to use the shared runtime-mode helper instead of directly calling stored callback pairs.
- Keep existing callback fields available for compatibility while the new runtime-metadata path becomes the primary orchestration route.

## Capabilities

### New Capabilities
- `optimization-train-eval-runtime-metadata`: Represent optimizer-owned train/eval runtime behavior as explicit optimization-plan metadata with shared orchestration helpers.

### Modified Capabilities

## Impact

- Affected code: `library/optimization/types.py`, `library/optimization/optimizer_utils.py`, training orchestration helpers, training loop/finalization call sites, and focused tests.
- Affected systems: schedule-free optimizer runtime mode switching and plan-aware training orchestration.
- Dependencies: no new runtime dependency; preserves existing schedule-free optimizer behavior while making the orchestration seam explicit.
