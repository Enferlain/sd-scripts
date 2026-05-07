# Timestep Layer Audit

This note is the first stage-specific follow-up to
[training_math_stage_map.md](/mnt/d/Projects/sd-scripts/docs_design/archive/training_math_stage_map.md).

The purpose of this note is to isolate the timestep layer and answer:

- what active settings and code paths currently participate in training-time
  time selection?
- which ones are actually the same kind of user-facing choice?
- which ones are different stages that only happen to interact?

This is not yet a code-move plan.
It is the working map we should use before reshaping config or ownership.

## Scope

This note is only about the timestep layer.

That includes:

- training-time time selection
- timestep range scheduling
- training-time transforms from sampled densities to timestep indices
- the boundary with inference-time flow shift

This note does **not** try to settle:

- loss weighting
- prediction targets
- full path/state construction ownership

Those are separate stages even when they interact with timesteps.

## Active surfaces

### Shared timestep runtime

The shared trainer-owned timestep runtime lives in:

- [runtime.py](/mnt/d/Projects/sd-scripts/library/timesteps/runtime.py)

Its active user-facing selector is:

- `timestep.timestep_sampling`

Current supported values:

- `uniform`
- `shift`
- `log_snr_uniform`
- `adaptive_log_snr`

It also owns:

- `min_timestep`
- `max_timestep`
- `dynamic_timestep_schedule`
- `adaptive_log_snr.*`
- `training_shift` for the sampled training-time distribution

### RF training-time density sampling

The active RF training-time sampling path lives in:

- [rectified_flow.py](/mnt/d/Projects/sd-scripts/library/objectives/rectified_flow.py)

It currently uses:

- `timestep.timestep_sampling`
- `timestep.logit_mean`
- `timestep.logit_std`
- `timestep.cosine_shape_scale`
- `timestep.training_shift`
- `timestep.min_timestep`
- `timestep.max_timestep`

This path does **not** currently use the shared trainer-owned
`TimestepRuntime` as its primary sampling owner.

### Inference-side flow shift

The active RF/discrete-flow inference-side shift lives in:

- [flow.py](/mnt/d/Projects/sd-scripts/library/pipelines/flow.py)
- [sd3/sampling.py](/mnt/d/Projects/sd-scripts/library/strategies/sd3/sampling.py)

It currently uses:

- `output.sampling.sample_flow_shift`

This is sample-time runtime behavior, not training-time timestep selection.

## What the timestep layer actually contains

The current "timestep layer" is not one concept.

It currently contains four different sub-concerns.

### 1. Timestep range scheduling

This answers:

- what timestep range is active right now?

Settings:

- `min_timestep`
- `max_timestep`
- `dynamic_timestep_schedule`

This is a coherent generic concern.

It belongs to shared timestep runtime state.

### 2. Training-time time sampling

This answers:

- how do we choose which training times to emphasize?

Current active values and knobs:

- shared runtime:
  - `uniform`
  - `shift`
  - `log_snr_uniform`
  - `adaptive_log_snr`
- RF path:
  - `uniform`
  - `logit_normal`
  - `cosine_shaped`
  - `logit_mean`
  - `logit_std`
  - `cosine_shape_scale`

This is one conceptual category.

The repo now uses one shared selector surface:

- `timestep_sampling`

but that one surface still spans more than one implementation family.

So the remaining question is no longer "should there be two selectors?",
but rather:

- how much of `timestep_sampling` is truly one coherent shared concept?

### 3. Density-to-index shaping

This answers:

- once we have a normalized time or density sample, how do we warp it before
  converting it into a timestep index?

The current active knob here is:

- `training_shift`

This is currently used by:

- shared `shift` sampling in [runtime.py](/mnt/d/Projects/sd-scripts/library/timesteps/runtime.py)
- RF training-time density-to-timestep shaping in
  [rectified_flow.py](/mnt/d/Projects/sd-scripts/library/objectives/rectified_flow.py)

This suggests that the repo currently has one similarly shaped transform being
applied in at least two training-time places.

That may turn out to be:

- one real shared concept

or:

- two similar formulas that only happen to share a parameter right now

This is still unresolved.

### 4. Inference-time flow shift

This answers:

- what shift should the discrete-flow sampling runtime use at sample time?

Current knob:

- `output.sampling.sample_flow_shift`

This is not part of training-time timestep selection.

It belongs at the inference runtime boundary, even if it is numerically related
to training-time shift transforms.

## Current problems

### 1. One user-facing selector now spans multiple implementation families

The repo now uses one shared selector:

- `timestep.timestep_sampling`

That is an improvement over keeping a second RF-only selector, but the surface
still spans:

- shared trainer-owned runtime modes
- RF-local density-sampling modes

So the next design question is whether those should remain one config surface
with model-family-aware interpretation, or converge toward a more unified
runtime implementation.

### 2. `training_shift` still needs a clearer conceptual boundary

`training_shift` currently participates in:

- shared `shift` training-time sampling
- RF training-time density-to-index shaping

and is numerically echoed by:

- `output.sampling.sample_flow_shift` for inference

This makes it difficult to tell whether "shift" is:

- one real cross-method concept
- one training-time concept plus one inference-time analogue
- or several similarly shaped transforms currently using related words

### 3. `cosine_shaped` is clearer than `mode`, but still just one local option

The rename from `mode` to `cosine_shaped` is an improvement because it points
at the current cosine-based formula family.

But it is still only one local density-sampling option inside the broader
time-sampling stage, not a deep architectural category by itself.

### 4. Validation and presentation still reflect the shared-runtime-only view

Current validation in
[config_validation.py](/mnt/d/Projects/sd-scripts/library/config/config_validation.py)
validates:

- `timestep.timestep_sampling`
- `adaptive_log_snr.*`

but does not currently validate the RF-side time-sampling selector and its
parameters as part of one coherent time-sampling story.

Current presentation in
[training_plots.py](/mnt/d/Projects/sd-scripts/library/logging/training_plots.py)
also describes:

- the shared runtime sampler mode

but not the RF-side density sampling choices as part of the same stage.

That mismatch is another sign the user-facing timestep story is not settled.

### 5. RF training-time sampling still bypasses the shared timestep runtime

The active RF path in
[rectified_flow.py](/mnt/d/Projects/sd-scripts/library/objectives/rectified_flow.py)
constructs timesteps directly.

That makes the repo awkwardly split between:

- a shared timestep subsystem
- an RF-local training-time timestep builder

This does not necessarily mean RF must be forced through the existing runtime
as-is, but it does mean the current timestep story is structurally incomplete.

## Best current interpretation

The safest current interpretation is:

- timestep range scheduling is a clean shared runtime concern
- training-time time sampling is one conceptual category, even though the repo
  currently exposes it through more than one surface
- density-to-index shaping is probably part of that same training-time
  selection stage, but its exact shared-vs-method-specific meaning is not yet
  settled
- inference-time flow shift is a separate sample-time concern

In other words:

- the timestep layer is real
- but it is broader than `library/timesteps/runtime.py`
- and narrower than "everything RF touches"

## Provisional target

The timestep layer should eventually give a contributor one clear answer to:

- how are training times sampled?

without making them mentally merge:

- prediction target
- loss weighting
- path/state construction
- inference runtime selection

That suggests the long-term target should look roughly like this:

### Shared inside the timestep layer

- timestep range scheduling
- training-time time sampling
- any genuinely shared density-to-index shaping

### Outside the timestep layer

- RF loss weighting
- DDPM SNR loss weighting
- prediction target choices
- inference-time sampler / integrator selection
- sample-time flow shift defaults

## Open questions for the next pass

These are the questions that matter most before the next code cleanup.

### 1. How much of `timestep_sampling` should become truly shared runtime behavior?

This is now the main question.

The repo already has one user-facing selector again.

What remains unsettled is whether values like:

- `logit_normal`
- current `mode`

should stay RF-local implementations behind that selector, or whether the
shared timestep runtime should eventually own more of that surface directly.

### 2. Is `training_shift` one real training-time concept?

If yes:

- it may belong in a shared time-sampling layer

If no:

- the current shared name is misleading and should be split

### 3. How much of the RF training-time timestep builder should join the shared timestep runtime?

This is an ownership question, not just a naming question.

The repo does not need to force every method through identical code, but it
should avoid having two independent user-facing timestep-selection stories if
they are really the same stage.

## Short version

The current timestep layer mixes:

- range scheduling
- training-time time sampling
- density-to-index shaping
- inference-time flow shift

The main duplication is no longer two separate selectors.

The remaining tension is that one selector, `timestep_sampling`, currently
covers both shared-runtime modes and RF-local sampling modes.

The next timestep-layer cleanup should start by deciding how unified that
surface should become internally.
