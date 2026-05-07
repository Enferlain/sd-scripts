# Objective / Timestep Taxonomy Audit

Status: archived as addressed after the explicit objective path/prediction split,
objective-owned runtime refactor, and RF runtime ownership cleanup.

This note is a follow-up to the first `library/objectives/` extraction.

The goal here is not to rename or move code immediately. The goal is to map
the active concepts in the current training/runtime layer so future refactors
can use words that match the real variation axes.

## Why this note exists

The current active path still mixes several different kinds of concerns across:

- `library/objectives/`
- `library/timesteps/`
- `library/losses/`
- `library/training/sample_generation.py`
- model-family diffusion / sampling strategy files

Some of the ambiguity comes from familiar community shorthand:

- "objective"
- "prediction type"
- "parameterization"
- "scheduler"
- "timestep sampling"

These are often used loosely in conversation, but in the code they do not all
vary together.

The local paper copies under [papers/](/mnt/d/Projects/sd-scripts/papers) also
support being careful here:

- [Flow Straight and Fast...](/mnt/d/Projects/sd-scripts/papers/Flow%20Straight%20and%20Fast%20Learning%20to%20Generate%20and%20Transfer%20Data%20with%20Rectified%20Flow-2209.03003.md)
  presents rectified flow as an ODE transport / training framework.
- [Flow Matching for Generative Modeling](/mnt/d/Projects/sd-scripts/papers/Flow%20Matching%20for%20Generative%20Modeling-2210.02747.md)
  presents flow matching as a broader training paradigm over probability
  paths, not just a local output-target toggle.
- [Rectified Diffusion...](/mnt/d/Projects/sd-scripts/papers/Rectified%20Diffusion%20Straightness%20Is%20Not%20Your%20Need%20in%20Rectified%20Flow-2410.07303v2.md)
  is especially explicit that "flow-matching", "`v`-prediction", and
  "rectification" can be separated conceptually even when some practical
  implementations bundle them together.

## Current high-level axes

The active path appears to have at least these distinct axes:

### 1. Formulation family

This is the broadest runtime family:

- DDPM-style diffusion
- rectified flow / flow matching

This axis changes more than one local knob. It affects:

- how noisy inputs are constructed
- what the model output means
- how timestep / sigma values are interpreted
- which loss-weighting math makes sense
- what inference sampler family makes sense

This broader reading matches the local rectified-flow and flow-matching paper
copies better than treating RF as only "another prediction type".

### 2. Prediction target within a formulation

Inside the DDPM-style diffusion path, the active code already distinguishes:

- noise prediction (`eps`)
- velocity prediction (`v`)

Potential future extensions would likely include:

- `x0`

This axis answers:

- what target the model is trained to predict

In the active code, `cfg.loss.v_parameterization` currently drives:

- training target selection in `library/strategies/sd/diffusion.py`
- training target selection in `library/strategies/sdxl/diffusion.py`
- sampling scheduler `prediction_type` in `library/training/sample_generation.py`
- metadata/model-identity helpers in SD / SDXL checkpointing paths

SD3 is a useful counterexample here:

- the active repo still keeps `cfg.loss.v_parameterization` as a shared config
  field
- but SD3/rectified-flow checkpoints should not inherit the DDPM
  `epsilon`/`v` metadata axis just because the helper accepts the same boolean
  input
- this is one of the strongest reasons to treat `v_parameterization` as a
  prediction-target concern rather than as a generic loss toggle

This narrower reading also matches the local paper copies better:

- the active repo uses `v_parameterization` to change what the denoiser target
  means inside the diffusion-style path
- the local rectified-diffusion paper copy explicitly discusses
  `v`-prediction as a separable component rather than as the whole
  rectified-flow idea

### 3. Timestep runtime scheme

This is the shared trainer/runtime-facing timestep sampling state owned by
`library/timesteps/runtime.py`.

Current active values:

- `uniform`
- `shift`
- `log_snr_uniform`
- `adaptive_log_snr`

This axis answers:

- how training timestep indices are sampled over time
- whether adaptive timestep state exists
- whether dynamic min/max timestep schedule state is active

This is already a legitimate top-level runtime concern.

### 4. DDPM-specific loss post-processing

The active DDPM-style path has several SNR/scheduler-dependent loss
transformations in `library/losses/loss_weighting.py`.

Current active settings:

- `loss.snr.min_snr_gamma`
- `loss.snr.scale_v_pred_loss_like_noise_pred`
- `loss.snr.v_pred_like_loss`
- `loss.snr.debiased_estimation_loss`

These are not generic tensor loss primitives. They depend on:

- DDPM scheduler state (`all_snr`)
- timesteps
- whether `v_parameterization` is active

This makes them feel closer to DDPM formulation runtime math than to generic
`library/losses/` ownership.

### 5. Inference sampler / scheduler selection

The active SD / SDXL sample-generation path has a separate inference-only axis:

- `output.sampling.sample_sampler`

This chooses schedulers such as:

- `ddim`
- `euler`
- `euler_a`
- `dpmsolver++`

This is separate from training timestep runtime selection. It is also separate
from the broader formulation family, even though some combinations are only
valid for some formulations.

## Current concept map

| Setting / concept | Current home | What it actually controls | Likely category | Notes |
| --- | --- | --- | --- | --- |
| `loss.v_parameterization` | `library/config/dataclasses/loss.py` | DDPM training target and DDPM sampling scheduler `prediction_type` | prediction target inside diffusion | Feels broader than a generic loss toggle; SD3/Flux-style metadata should not inherit it as if it were universal. |
| `loss.regularization.zero_terminal_snr` | `library/config/dataclasses/loss.py` | DDPM scheduler construction | DDPM scheduler/runtime option | Not really "regularization" in the same sense as noise offset or multires noise; it shapes the DDPM scheduler used to build noisy states. |
| `loss.snr.min_snr_gamma` | `library/config/dataclasses/loss.py` | DDPM SNR-based post-loss weighting | DDPM loss post-processing | Coupled to scheduler `all_snr`. |
| `loss.snr.debiased_estimation_loss` | `library/config/dataclasses/loss.py` | DDPM debiased post-loss weighting | DDPM loss post-processing | Also coupled to scheduler `all_snr`. |
| `loss.snr.scale_v_pred_loss_like_noise_pred` | `library/config/dataclasses/loss.py` | v-pred-specific DDPM post-loss rescaling | DDPM loss post-processing | Only valid with `v_parameterization`. |
| `loss.snr.v_pred_like_loss` | `library/config/dataclasses/loss.py` | extra DDPM/v-pred-style post-loss term | DDPM loss post-processing | Explicitly conflicts with `v_parameterization` in validation. |
| `timestep.timestep_sampling` | `library/config/dataclasses/timestep.py` | shared timestep runtime mode | timestep runtime scheme | Already has a coherent runtime meaning. |
| `timestep.min_timestep` / `max_timestep` | `library/config/dataclasses/timestep.py` | active timestep range | timestep runtime range state | Shared runtime concern. |
| `timestep.dynamic_timestep_schedule` | `library/config/dataclasses/timestep.py` | step-driven timestep range updates | timestep runtime range state | Shared runtime concern. |
| `timestep.adaptive_log_snr.*` | `library/config/dataclasses/timestep.py` | adaptive log-SNR sampler config | timestep runtime sampler config | Shared runtime concern. |
| `timestep.training_shift` | `library/config/dataclasses/timestep.py` | shift transform for time-sampling density values before timestep indexing | training-time shaping parameter | Shared by the active training-time shift paths. |
| `timestep.timestep_sampling` with RF-local values | `library/config/dataclasses/timestep.py` | RF timestep-density selection | time sampling | RF now reuses the shared training-time selector for `uniform`, `logit_normal`, and `cosine_shaped`. |
| `timestep.rf_loss_weighting_scheme` | `library/config/dataclasses/timestep.py` | RF loss-weighting selection | RF loss-weighting config | Split out from the earlier overloaded `weighting_scheme` field. |
| `timestep.logit_mean` / `logit_std` | `library/config/dataclasses/timestep.py` | parameters for RF `logit_normal` density sampling | RF timestep-density config | Not shared timestep runtime in general. |
| `timestep.cosine_shape_scale` | `library/config/dataclasses/timestep.py` | parameter for RF `cosine_shaped` density sampling | RF timestep-density config | Clearer than the old `mode_scale` name, but still just one local density-sampling formula. |
| `output.sampling.sample_sampler` | `library/config/dataclasses/output.py` | DDPM-style inference scheduler choice | inference sampler selection | Not a training timestep concept. |
| `output.sampling.sample_flow_shift` | `library/config/dataclasses/output.py` | RF sampling default shift | RF inference runtime config | Sampling-only concept. |

## Strongest taxonomy mismatches today

### 1. RF density sampling and RF loss weighting have now been split, but the surrounding taxonomy still matters

The active RF path no longer overloads one field for both concerns:

- `timestep_sampling` is used for RF timestep-density sampling
- `rf_loss_weighting_scheme` is used for post-loss weighting from sigmas

That split removes the strongest immediate ambiguity, but the repo still needs
to decide where these two RF-specific concepts belong relative to the shared
`timesteps/` runtime surface.

### 2. `training_shift` currently crosses training-time concerns

It is used by:

- shared timestep runtime `shift` mode in `library/timesteps/runtime.py`
- RF density -> timestep shaping in `library/objectives/rectified_flow.py`
- RF sampling-side shift in `library/pipelines/flow.py` / SD3 sampling

That suggests the field may be carrying multiple "shift" meanings that happen
to share a similar transform shape, but are not obviously one stable concept.

### 3. `v_parameterization` and `zero_terminal_snr` live under `loss`, but each is broader than that label

- `v_parameterization` affects training target semantics, sampling scheduler
  configuration, and metadata/model identity
- `zero_terminal_snr` affects DDPM scheduler shaping for noisy-state construction rather than only loss

This does not mean they must move immediately, only that the current parent
label does not describe them particularly well.

### 4. `library/losses/loss_weighting.py` is not really generic loss math

It depends on:

- `DDPMScheduler`
- scheduler-derived `all_snr`
- `v_parameterization`

That makes it feel like DDPM-specific post-loss runtime math rather than a
truly formulation-agnostic `losses/` module.

## Current practical reading of the active path

### DDPM-style diffusion path

The active DDPM path currently consists of:

- formulation owner: `library/objectives/ddpm.py`
- shared timestep runtime: `library/timesteps/runtime.py`
- training target branch: `cfg.loss.v_parameterization`
- scheduler/state-construction option: `cfg.loss.regularization.zero_terminal_snr`
- DDPM post-loss weighting: `library/losses/loss_weighting.py`
- DDPM sampling scheduler selection: `library/training/sample_generation.py::get_my_scheduler`

### Rectified flow path

The active RF path currently consists of:

- formulation owner: `library/objectives/rectified_flow.py`
- RF density sampling knobs: `timestep_sampling`, `logit_mean`, `logit_std`, `cosine_shape_scale`
- RF loss-weighting knob: `rf_loss_weighting_scheme`
- RF shift transform knobs: `training_shift`
- RF sampling-side shift default: `output.sampling.sample_flow_shift`
- RF sampling runtime math: `library/pipelines/flow.py`

Unlike DDPM-style diffusion, RF currently does not use the shared timestep
runtime as its core training-time timestep owner.

## Questions the repo still needs to answer later

These are the next conceptual questions, not immediate code-change requests.

### 1. Is RF "prediction type" or "formulation family" in repo architecture?

Community conversation often treats RF as a sibling to `eps` and `v`.
The active code suggests RF is broader than that because it affects:

- input construction
- sigma/timestep interpretation
- loss weighting
- inference runtime math

The local paper copies do not force one canonical ontology, but they do support
being more careful than "RF is exactly the same category as `v`-prediction":

- rectified flow / flow matching are presented as broader training/path
  frameworks
- `v`-prediction is presented as a distinct design choice that may be used
  within those frameworks

The repo should decide whether it wants to model RF:

- as a sibling to `eps` / `v`
- or as a broader formulation family that may itself have prediction choices

### 2. Which "shift" concepts are truly the same?

Today the word "shift" appears in:

- shared timestep runtime `shift`
- RF training-time density -> timestep shaping
- RF sampling-side shift

These may share a transform family without being one stable conceptual setting.

### 3. Should DDPM post-loss weighting stay under `losses/`?

If the repo wants `library/losses/` to mean generic tensor loss primitives,
then the DDPM SNR/v-pred helpers may eventually want a more formulation-owned
home.

### 4. Does the repo want one explicit config surface for formulation family?

The code now has `library/objectives/`, but config still mostly infers the
active formulation from `model.model_type`.

That is acceptable for now, but it means:

- formulation is not yet a first-class config axis
- some terminology questions are still constrained by model-family inference

## What this note recommends right now

### Do now

- keep `library/objectives/` as the formulation/runtime home
- keep `library/timesteps/` as the shared timestep runtime home
- avoid another code move until the terminology / category questions are more settled

### Do next, before another larger refactor

- pressure-test the category split for:
  - formulation family
  - prediction target
  - timestep runtime scheme
  - DDPM post-loss weighting
  - inference sampler selection
- pressure-test whether the new RF-specific split should stay under `timestep`
  or move under a more explicitly RF-owned config surface later
- decide whether `training_shift` is one real concept or several similarly shaped transforms

## Short version

The current layer is not one thing.

The active code is mixing at least:

- formulation family
- prediction target inside a formulation
- shared timestep runtime scheme
- DDPM-specific post-loss weighting
- inference sampler selection

The strongest current smell used to be the overloaded `weighting_scheme`
field. That immediate problem is now gone, but the broader category boundaries
are still worth settling before another larger refactor.

The next cleanup should probably start from these categories, not from a
renaming pass done by feel.
