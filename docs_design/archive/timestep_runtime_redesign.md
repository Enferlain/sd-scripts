# Timestep Runtime Redesign

## Goal

Redesign timestep sampling as a first-class subsystem under `library/timesteps/`
instead of leaving policy split across:

- `Trainer`
- strategy instance state
- `library/training/diffusion.py`
- logging / plotting helpers

This is **not** a cleanup-only task.
The goal is to improve the shape of the system so timestep behavior is easier
to reason about, easier to extend, and less dependent on cross-module
conventions.

## Why This Needs Redesign

The active path currently has no single owner for timestep behavior.

### Current ownership is split

Today:

- [`library/timesteps/timestep_utils.py`](/mnt/d/Projects/sd-scripts/library/timesteps/timestep_utils.py)
  parses dynamic schedules and constructs adaptive samplers
- [`library/training/runners/trainer.py`](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py)
  stores dynamic timestep runtime state
- [`library/training/diffusion.py`](/mnt/d/Projects/sd-scripts/library/training/diffusion.py)
  decides which timestep mode actually runs
- SD / SDXL diffusion strategies update adaptive sampler state
- logging and plotting infer sampler identity from both config and sampler type

This means creation, runtime state, sampling, observation, and presentation are
owned by different layers.

### `shift` exposes the architecture break clearly

The best example is `shift`:

- config declares `shift` as a timestep mode
- `init_timestep_sampler(...)` declines to own it and leaves a comment saying it
  is handled elsewhere
- the real implementation lives inside
  [`get_noise_noisy_latents_and_timesteps(...)`](/mnt/d/Projects/sd-scripts/library/training/diffusion.py)

That is the opposite of a coherent timestep subsystem.

### Config mutation hides real runtime mode

[`init_timestep_sampler(...)`](/mnt/d/Projects/sd-scripts/library/timesteps/timestep_utils.py)
currently mutates `cfg.timestep.timestep_sampling` for several modes.

That creates semantic drift:

- config says what the user requested
- runtime mutates it into what the code actually wants
- downstream logging/plotting have to infer the truth from mixed signals

### Adaptive sampler state is strategy-owned by accident

`strategies.la_sampler` exists on SD / SDXL strategy instances, but the
trainer creates it.

That is not a model-family concern. It is timestep runtime state currently
stored on the nearest available object.

## Design Goals

1. `library/timesteps/` owns timestep-sampling policy.
2. `Trainer` owns the timestep runtime instance.
3. strategies stop storing `la_sampler`.
4. `library/training/diffusion.py` stops deciding timestep mode.
5. runtime state should be explicit and queryable without mutating config.
6. `shift`, `uniform`, adaptive modes, and scheduler-weighted modes should all
   be explicit code paths, not fallback behavior.
7. logging / plotting should read runtime description from the timestep
   subsystem instead of reverse-engineering it from config strings.

## Non-Goals

- no redesign of SD / SDXL denoiser calling
- no rework of the loss system itself
- no redesign of validation fixed-timestep evaluation in this first pass
- no requirement to preserve the current internal module split if a better one
  under `library/timesteps/` is clearer

## Recommended Ownership Decision

### Keep

- `Trainer` owns orchestration
- strategies own model-family diffusion behavior
- `library/timesteps/` owns timestep policy and runtime

### Change

- remove timestep runtime ownership from strategy instance state
- remove timestep mode branching from `library/training/diffusion.py`
- stop mutating `cfg.timestep.*` to express effective runtime mode

## Target Shape

### 1. A trainer-owned timestep runtime

Add a runtime object under `library/timesteps/`.

Recommended name:

- `TimestepRuntime`

This object should own:

- requested mode vs effective mode
- current min/max timestep range
- parsed dynamic schedule state
- adaptive sampler instance, if any
- timestep sampling for the current batch
- optional observation/update from per-sample loss
- a runtime description payload for logging/plotting

### 2. Explicit timestep mode semantics

The runtime should have one explicit dispatch point for active timestep modes:

- `uniform`
- `shift`
- `log_snr_uniform`
- `adaptive_log_snr`

Unsupported modes should fail explicitly.

They should **not** fall through a generic `!= "uniform"` branch.

### 3. Diffusion helper consumes explicit timesteps

`library/training/diffusion.py` should no longer decide timestep mode.

Instead, shared diffusion code should consume explicit sampled timesteps.

That means the generic helper layer should be responsible for:

- generating base noise
- adding noise to latents
- applying multires / IP noise adjustments

but **not** for deciding whether the run is using `shift`, adaptive sampling,
or another timestep mode.

### 4. Trainer observes adaptive sampler updates

Adaptive sampler updates should move out of SD / SDXL strategies.

Current update sites:

- [`library/strategies/sd/diffusion.py`](/mnt/d/Projects/sd-scripts/library/strategies/sd/diffusion.py)
- [`library/strategies/sdxl/diffusion.py`](/mnt/d/Projects/sd-scripts/library/strategies/sdxl/diffusion.py)

Recommended direction:

- strategies return `BatchLossOutput`
- trainer calls `timestep_runtime.observe(...)` after batch loss is available

This keeps adaptive timestep learning as timestep runtime behavior, not
model-family behavior.

### 5. Logging reads runtime description, not config mutation

Add one small description payload owned by the timestep runtime.

Recommended contents:

- requested mode
- effective mode
- current min/max range
- whether dynamic schedule is active
- adaptive sampler kind, if any
- any sampler-specific metadata the live plotter wants to show

Then:

- `step_logging.py` should use runtime description + runtime object
- `training_plots.py` should use runtime description instead of `cfg` +
  `isinstance(...)` heuristics

## Proposed Module Plan

The exact filenames can change, but the responsibilities should look like this.

### `library/timesteps/runtime.py`

Own:

- `TimestepRuntime`
- runtime range state
- dynamic schedule advancement
- adaptive sampler observation hooks
- runtime description payload

### `library/timesteps/sampling.py`

Own pure timestep selection behavior for explicit modes:

- uniform
- shift
- scheduler-weighted timestep draws
- any other non-adaptive sampling branch

This is where the current `shift` implementation should move.

### `library/timesteps/factory.py`

Own:

- constructing adaptive sampler instances
- translating config into a runtime-ready mode/spec

This replaces the current mixed responsibilities in `timestep_utils.py`.

### `library/timesteps/timestep_utils.py`

During migration:

- keep as a thin compatibility facade if needed

End state:

- either reduced to small wrappers
- or deleted if the new module layout is clearer without it

## Trainer / Strategy Boundary

Recommended boundary:

- trainer owns `self.timestep_runtime`
- trainer advances timestep runtime as training progresses
- trainer samples timesteps for training batches, or passes the runtime into a
  shared helper that does so on its behalf
- trainer applies adaptive observation updates after loss is computed
- strategies receive explicit timesteps or explicit timestep context
- strategies do not hold `la_sampler`

The important part is not the exact parameter list.
The important part is that timestep runtime ownership stops being implicit
strategy instance state.

## Migration Direction

### Stage 1: Add guardrail tests first

Before refactor work:

- add focused unit tests for `init_timestep_sampler(...)`
- add focused unit tests for `parse_dynamic_timestep_schedule(...)`
- add focused unit tests for `shift` timestep behavior
- add focused unit tests for training-loop schedule advancement
- add focused unit tests for adaptive observation/logging behavior

Current coverage is too thin at the timestep-subsystem layer.

### Stage 2: Introduce runtime without changing behavior

- add `TimestepRuntime`
- move dynamic schedule parsing/state into it
- move sampler construction behind runtime/factory code
- keep existing diffusion/strategy behavior working through compatibility calls

### Stage 3: Move timestep mode dispatch out of diffusion helper

- extract explicit `shift` / uniform / weighted-timestep logic into
  `library/timesteps/`
- make diffusion helpers consume explicit timesteps
- remove the `timestep_sampling != "uniform"` fallback branch

### Stage 4: Remove strategy-owned sampler state

- remove `strategies.la_sampler`
- move adaptive observation updates to trainer-owned timestep runtime
- update logging and plotting to read timestep runtime metadata directly

### Stage 5: Remove config mutation and compatibility shims

- stop mutating `cfg.timestep.timestep_sampling`
- stop making logging infer effective mode via `isinstance(...)`
- shrink or delete compatibility wrappers in `timestep_utils.py`

## Immediate Design Rules

If we start this work, these rules should guide each change:

1. Timestep mode semantics must live under `library/timesteps/`.
2. `shift` must become an explicit timestep-owned path, not a diffusion
   fallback.
3. Config should represent user intent, not be mutated to runtime state.
4. Adaptive timestep updates are timestep-runtime behavior, not strategy
   behavior.
5. Logging/plotting should consume a runtime-owned description rather than
   reverse-engineering mode from config mutation.

## Acceptance Criteria

This redesign is in a good state when:

1. there is one clear owner for timestep runtime behavior
2. `shift` is implemented under `library/timesteps/`
3. `library/training/diffusion.py` no longer branches across timestep modes
4. strategies no longer store `la_sampler`
5. runtime mode is explicit without mutating config
6. timestep subsystem behavior has direct unit coverage beyond import/smoke
