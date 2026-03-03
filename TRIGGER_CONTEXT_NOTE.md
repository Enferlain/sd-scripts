 # Trigger Context Pattern Note

Date: 2026-03-01

## Context

This repo already favors explicit typed boundaries and testable units (`DEVELOPMENT_GUIDE.md`). Validation scheduling was centralized into `ValidationScheduler.should_run(ctx)` with a typed `ValidationStepContext` and matrix tests.

At the same time, trigger policy still exists in multiple places:

- step-time decisions in `library/training/phases/training_loop.py`
- epoch-end decisions in `library/training/phases/training_loop.py`
- sampling cadence/precedence logic in `library/training/sample_generation.py`

Roadmap signals more trigger policy work:

- sampling conflict validation
- logging interval gating (`log_every_n_steps`)
- further validation-system work

## Decision

`*Context` + scheduler is valid as a **local policy pattern** for trigger/scheduling decisions.

It is **not** a new top-level architecture axis (unlike `Trainer`/`Mode`/`Strategy`).

### Definition of `*Context` in this repo

A `*Context` for scheduling means:

- immutable runtime facts (prefer scalar fields)
- no side effects
- no service/model/dataloader references
- no embedded business behavior

Examples of allowed fields: `global_step`, `current_epoch`, `is_last_step_in_epoch`, `is_training_end`, `has_validation_data`.

### Anti-goal

Do not create context objects that "smuggle behavior" (methods that perform actions, mutate state, or call services). Keep behavior in scheduler/policy functions.

### Adoption rule

Use a typed context object only when all are true:

1. Decision logic is policy-heavy and easy to get wrong with inline conditionals.
2. Inputs are many and reused across 2+ call sites.
3. A dedicated test matrix adds clear value.

Do not use it when:

1. There is only one simple call site.
2. A plain function with explicit scalar args is clearer.

## Side-by-side style guidance

Both styles are acceptable:

- `ValidationScheduler.should_run(ctx: ValidationStepContext) -> bool`
- `should_run_validation(global_step: int, current_epoch: int, ...) -> bool`

Prefer the context form when input count/reuse is high and tests benefit from matrix-style cases. Prefer explicit args for simple single-site rules.

## Consequences

If upcoming roadmap items are implemented, likely maintainable direction is:

- one shared step trigger context (runtime facts only)
- small schedulers for validation/sampling/checkpoint/logging decisions
- execution remains in training loop; policy is centralized and unit-tested

If that consolidation does not happen, keep context usage minimal and avoid one-off context types.

## Testing expectation

Scheduler/policy modules should use table- or matrix-driven unit tests near the policy code, so cadence/precedence changes are easy to review.
