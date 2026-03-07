# Training Orchestration Follow-Up

## Purpose

Capture small runner/phase follow-up work noticed during post-cleanup review.

This is **not** a correctness emergency and **not** a performance plan.
The current training flow is in a good state. These items are about keeping
the generic training layer readable and resilient as later work lands.

This note is also intentionally **separate from strategy cleanup**.
Strategy inspection/refactoring is a future task; this document is only about
the shared orchestration layer (`Trainer` + phases).

## Current Assessment

The architecture direction is correct:

- scripts are thin entry points
- root config dataclasses define valid mode surfaces
- `Trainer` orchestrates phases
- `TrainingMode` owns mode-specific divergence
- `TrainingStrategy` owns model-family behavior
- phase modules are largely mode-agnostic

The recent cleanup also closed the main correctness gaps that were found:

- epoch-end actions no longer fire for partial epochs truncated by `max_train_steps`
- resource monitor phase/session cleanup now survives interruption paths

## Why Keep This Note

The remaining observations are mild, but they are the places where future
changes could become harder to reason about:

1. `training_loop.py` still carries dense orchestration logic in one place.
2. Some lifecycle rules are correct because of sequencing discipline, not
   because the structure makes them impossible to misuse.
3. Future work in modes/logging/validation/resume can put pressure on this
   layer even if the architecture remains correct.

## Follow-Up Items

### 1. Make epoch completion an explicit concept

Current code correctly avoids epoch-end side effects for partial epochs, but
the logic is still encoded procedurally inside the loop.

Desired cleanup:

- introduce an explicit `epoch_completed` / `epoch_interrupted` outcome
- or extract a small helper that returns structured epoch execution state
- keep epoch-end save/sample policy keyed off that explicit result

Benefit:

- resume / early-stop / step-cap edits become easier to reason about
- future readers do not need to reconstruct why batch-consumption checks exist

### 2. Consider scoped lifecycle helpers for shared phases

Resource monitor cleanup is now robust, but phase/session ownership is still
manually open-coded.

Desired cleanup:

- add a tiny helper/context manager for shared phase lifecycle
- use it where the generic layer owns `phase_start` / `phase_end`
- keep ownership in runner/phases, not modes/strategies

Benefit:

- lowers the chance of reintroducing failure-path leaks
- makes future shared phases easier to add consistently

### 3. Keep trainer orchestration generic as future work lands

`Trainer` currently owns:

- startup diagnostics
- tracker initialization
- timestep sampler/live plotter setup
- step-0 sampling/validation
- final save/session closing

That is acceptable for now. The follow-up is simply to keep these as generic
orchestration concerns and avoid letting mode-specific exceptions accumulate
there later.

Desired outcome:

- preserves the intended `config -> strategy/mode -> runner/phases` direction
- prevents special-case logic from creeping back into the shared layer

### 4. Sequence this after strategy cleanup, not before

Some of the remaining density in runner/phases may become easier to reshape
after the strategy layer is reviewed.

This means the work is still desirable, but the right ordering is to avoid
redesigning orchestration around assumptions that may change during strategy
cleanup.

Benefit:

- avoids churn
- keeps architecture work ordered by dependency pressure

## Non-Goals

- no new runners
- no performance tuning
- no strategy API redesign here
- no mode-layer rework unless an actual boundary violation appears

## Acceptance Criteria

When this follow-up is done, the result should:

1. preserve current runtime behavior
2. preserve mode/strategy separation
3. not introduce additional abstraction layers without concrete payoff
4. make interruption and epoch-boundary behavior easier to read in source
