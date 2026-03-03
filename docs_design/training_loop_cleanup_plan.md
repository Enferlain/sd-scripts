# Training Loop Cleanup Plan (2026-03-03)

## Objective

Define a clear, low-risk path to improve `library/training/phases/training_loop.py` while preserving current behavior and aligning with:

1. Trigger policy cleanup direction (`TRIGGER_CONTEXT_NOTE.md`)
2. Upcoming resource monitor integration (`docs_design/resource_monitor_plan.md`)

## Current State Summary

The training loop is functional and tested, but it currently combines too many responsibilities in one function:

1. Core optimization step math
2. Trigger policy (validation/sampling/checkpoint timing)
3. Side-effect orchestration (sampling, validation, save/remove checkpoint)
4. Tracker logging and progress updates
5. Live plotter streaming and timestep plot logic
6. Legacy env-var resource tracker branches

This increases cognitive load and makes policy changes riskier than necessary.

## Primary Issues To Address

## 1. Overloaded `run_training_loop()`

`run_training_loop()` currently does both policy decisions and action execution, with step and epoch-end branches in one function.

Impact:

1. Harder to reason about correctness when adding/changing triggers
2. Harder to test isolated behavior
3. Increased chance of regressions when touching unrelated concerns

## 2. Trigger policy is only partially centralized

Validation now uses a typed scheduler/context pattern, but sampling and checkpoint triggers are still inline/duplicated.

Impact:

1. Inconsistent policy style
2. Trigger behavior spread across multiple branches
3. More difficult future integration with resource monitor/logging orchestration

## 3. Epoch semantics inconsistency in checkpoint metadata path

Step-based checkpoint saves pass zero-based epoch loop variable; epoch-end saves pass one-based epoch state. This can produce inconsistent epoch metadata semantics.

Impact:

1. Metadata interpretation ambiguity
2. Potential downstream confusion in tools/scripts consuming `ss_epoch`

## 4. Legacy resource tracking env-var branches remain in phase code

`BENCHMARK_RESOURCES` branches still exist in `caching.py` and `training_loop.py`, while resource monitor design assumes config-driven central monitor hooks.

Impact:

1. Split resource observability mechanisms
2. Extra branch complexity in hot-path orchestration code

## Target Architecture Direction

## A. Keep loop orchestration, extract policy and side-effect units

Refactor into small helpers inside `training_loop.py` (or neighboring phase helpers) with clear boundaries:

1. `process_optimization_step(...)`
2. `compute_step_actions(...)`
3. `run_step_side_effects(...)`
4. `emit_step_logging(...)`
5. `handle_epoch_end(...)`

This keeps readability without introducing a new top-level framework.

## B. Unify trigger style across validation/sampling/checkpoint

Adopt a consistent trigger policy pattern:

1. Runtime facts (`is_*`, `has_*`) in context/state
2. `should_*` or scheduler decisions for action timing
3. Execution logic separated from decision logic

## C. Align epoch semantics

Use one explicit epoch convention across all checkpoint save paths and metadata updates, documented in code comments and tests.

## D. Move resource tracking to monitor hooks only

Remove direct env-var resource tracker branches and rely on `ResourceMonitor` integration points defined in `resource_monitor_plan.md`.

## Sequencing Recommendation

Order matters to reduce risk and rework:

1. **Phase 0: Correctness-first micro-fixes**
   - Resolve epoch semantic inconsistency for checkpoint metadata
   - Add/adjust targeted tests for epoch metadata expectations

2. **Phase 1: Trigger policy cleanup**
   - Apply `TRIGGER_CONTEXT_NOTE.md` direction
   - Consolidate trigger decision logic for sampling/checkpoint alongside validation style
   - Keep behavior unchanged unless explicitly intended and tested

3. **Phase 2: Training loop structural split**
   - Extract helper functions for step/epoch orchestration
   - Reduce branch density in main loop body

4. **Phase 3: Resource monitor integration**
   - Remove env-var resource tracker branches
   - Add monitor hooks (`phase_start/end`, `step_end`) at defined boundaries
   - Validate overhead and rank behavior per resource monitor plan

### Progress Status (2026-03-03)

1. ✅ **Phase 0 complete**: step-checkpoint epoch metadata now uses 1-based `current_epoch_state.value` and is covered by integration assertions.
2. ✅ **Phase 1 complete**: trigger policy extracted into `library/training/phases/triggers.py` with typed contexts/action objects and dedicated unit tests.
3. ✅ **Phase 2 complete**: `run_training_loop()` now delegates large side-effect/logging/epoch-finalization branches to focused helpers in `training_loop.py`, reducing branch density while keeping behavior stable.
4. ✅ **Phase 3 complete (baseline hooks)**: env-var resource tracker branches were removed from training/caching phases and replaced with Trainer-owned, config-driven monitor hooks (`phase_start/end`, `step_end`). Sampled/deep collector fidelity and JSONL streaming are tracked in `resource_monitor_plan.md`.

Reason for this order:

1. Correctness issues should be fixed before structural work
2. Trigger consolidation gives a stable policy layer before resource monitor hooks depend on step/phase boundaries
3. Resource monitor then lands on a cleaner loop with clearer boundaries

## Non-Goals

1. No changes to model math, optimizer behavior, or convergence logic
2. No rewrite of `Trainer`/`Mode`/`Strategy` architecture
3. No dashboard redesign in this cleanup

## Validation Strategy

For each phase, run relevant tests and keep behavior parity checks:

1. Unit tests for trigger policy and metadata semantics
2. Existing training loop integration tests for checkpoint and step progression behavior
3. Short benchmark/smoke runs for log and loop stability

## Exit Criteria

This cleanup is considered complete when:

1. Trigger decisions are centralized/consistent (not split across ad-hoc inline branches)
2. Epoch metadata semantics are consistent across step and epoch save paths
3. `run_training_loop()` is significantly easier to read and modify
4. Env-var resource tracker branches are removed from training/caching phases
5. Resource monitor hooks integrate without altering core training behavior
