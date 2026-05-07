# Training Math Stage Map

Status: archived as addressed after the explicit objective path/prediction split,
objective-owned runtime refactor, and RF runtime ownership cleanup.

This note is meant to keep the next cleanup grounded in what the active
settings actually do during training, rather than in the historical packaging
they arrived with.

It is a follow-up to:

- [timestep_runtime_redesign.md](/mnt/d/Projects/sd-scripts/docs_design/timestep_runtime_redesign.md)
- [objective_timestep_taxonomy_audit.md](/mnt/d/Projects/sd-scripts/docs_design/archive/objective_timestep_taxonomy_audit.md)

Those notes answer:

- who owns timestep runtime today
- which active concepts are currently mixed together

This note answers a different question:

- what are the actual mathematical / runtime stages of training and sampling?

The goal is to organize future work around those stages first, and only then
decide config homes, file layout, and naming.

## Working principle

Do not trust the current split between:

- older SD / SDXL-era settings
- SD3 / RF-era settings

as a conceptual truth.

A lot of the current distinction comes from how features arrived in the repo:

- DDPM-style diffusion settings were integrated gradually into the shared
  trainer/runtime path
- RF settings arrived bundled with the first SD3 port from more self-contained
  script code

That history is useful evidence, but it should not define the architecture by
itself.

## The stages we should organize around

The active path appears to separate most cleanly into five stages.

### 1. Path / state construction

This stage answers:

- how do we build the intermediate model input from clean data, noise, and
  time?

Examples:

- DDPM-style diffusion path:
  - noisy latent construction via scheduler / `add_noise(...)`
- flow-matching / RF-style path:
  - linear interpolation between clean latent and noise
- scheduler-shaping options such as zero-terminal-SNR

This is a broad runtime choice. It changes:

- what `x_t` or model input means
- how time is interpreted
- what target conversions are natural
- what inference runtime is likely to match

This is the strongest current reason not to treat RF as merely "another loss
toggle".

### 2. Time sampling

This stage answers:

- which training times are emphasized?
- how are training time indices or densities sampled?

Examples currently present in the repo:

- `uniform`
- `shift`
- `log_snr_uniform`
- `adaptive_log_snr`
- `logit_normal`
- `cosine_shaped`

These are all the same kind of choice:

- a distribution or heuristic over training time

They may be implemented in different code paths today, but they should be
evaluated as one class of concern.

This means the repo should be suspicious of exposing multiple separate
user-facing knobs that all boil down to "how are training times sampled?"

### 3. Prediction target

This stage answers:

- what does the network output represent at the chosen state / time?

Examples:

- `eps`
- `v`
- possible future `x0`
- any genuinely distinct flow/vector target if the active math requires it

This stage is narrower than path construction.

Changing the path / state construction does not necessarily force one target.
The local paper copies under [papers/](/mnt/d/Projects/sd-scripts/papers)
support being careful here:

- flow matching / rectification can be discussed separately from
  `v`-prediction
- `v`-prediction is not the whole RF idea

The active repo now has one concrete sign of that split:

- SD / SDXL still use `v_parameterization` to control DDPM prediction-target
  behavior and DDPM sample-time scheduler `prediction_type`
- SD3/rectified-flow checkpoints should not serialize that DDPM
  `prediction_type` field as if it were the same axis

### 4. Loss weighting / post-processing

This stage answers:

- after a per-sample loss is computed, how is it reweighted across time or
  sigma?

Examples:

- none / uniform
- Min-SNR
- debiased estimation
- `scale_v_pred_loss_like_noise_pred`
- `v_pred_like_loss`
- RF-side `sigma_sqrt`
- RF-side `cosmap`

These are all the same kind of concern:

- they do not choose the path
- they do not choose the prediction target
- they modify the loss after the main prediction/target comparison exists

This stage is currently split between:

- DDPM-specific helpers in `library/losses/loss_weighting.py`
- RF-specific helpers in `library/objectives/rectified_flow.py`

### 5. Inference integrator / sampler

This stage answers:

- how do we numerically integrate or step the model at sample time?

Examples:

- DDIM
- Euler
- DPM++ families
- RF / discrete-flow sampling runtime math

This is distinct from training-time time sampling.

It is also distinct from path construction, even if some inference methods only
make sense for some path families.

## Current settings mapped to the stages

| Setting / concept | Current home | Stage | What it changes |
| --- | --- | --- | --- |
| `loss.regularization.zero_terminal_snr` | `library/config/dataclasses/loss.py` | path / state construction | reshapes the DDPM scheduler used for noisy-state construction |
| `timestep.timestep_sampling` | `library/config/dataclasses/timestep.py` | time sampling | shared trainer-owned timestep runtime mode |
| `timestep.training_shift` | `library/config/dataclasses/timestep.py` | time sampling | shift applied to the sampled training-time distribution before timestep indexing |
| `timestep.timestep_sampling` with RF-local values | `library/config/dataclasses/timestep.py` | time sampling | RF currently reuses the shared training-time sampler selector for `uniform`, `logit_normal`, and `cosine_shaped` |
| `timestep.logit_mean` / `logit_std` | `library/config/dataclasses/timestep.py` | time sampling | parameters for RF `logit_normal` density sampling |
| `timestep.cosine_shape_scale` | `library/config/dataclasses/timestep.py` | time sampling | parameter for the RF `cosine_shaped` density sampler |
| `loss.v_parameterization` | `library/config/dataclasses/loss.py` | prediction target | changes the DDPM denoiser training target and DDPM inference scheduler `prediction_type` |
| `loss.snr.min_snr_gamma` | `library/config/dataclasses/loss.py` | loss weighting | DDPM post-loss weighting |
| `loss.snr.debiased_estimation_loss` | `library/config/dataclasses/loss.py` | loss weighting | DDPM post-loss weighting |
| `loss.snr.scale_v_pred_loss_like_noise_pred` | `library/config/dataclasses/loss.py` | loss weighting | DDPM / v-pred-specific post-loss rescaling |
| `loss.snr.v_pred_like_loss` | `library/config/dataclasses/loss.py` | loss weighting | DDPM / v-pred-specific extra post-loss term |
| `timestep.rf_loss_weighting_scheme` | `library/config/dataclasses/timestep.py` | loss weighting | RF post-loss weighting in the current code |
| `output.sampling.sample_sampler` | `library/config/dataclasses/output.py` | inference integrator / sampler | DDPM-style sample-time scheduler choice |
| `output.sampling.sample_flow_shift` | `library/config/dataclasses/output.py` | inference integrator / sampler | RF/discrete-flow sample-time shift default |

## Key interpretation

The repo should stop reasoning primarily in terms of:

- "old DDPM settings"
- "new RF settings"

and instead reason in terms of:

- path / state construction
- time sampling
- prediction target
- loss weighting
- inference integrator

That gives a more stable basis for deciding:

- which settings are genuinely separate
- which settings are the same kind of thing with different names or origins
- which config surfaces are redundant

## Immediate implications

### 1. RF density sampling and shared timestep sampling are too close to be treated casually as separate user-facing ideas

The repo now uses one shared user-facing selector:

- `timestep.timestep_sampling`

But that still spans multiple implementation families:

- shared trainer-owned runtime values like `uniform`, `shift`,
  `log_snr_uniform`, and `adaptive_log_snr`
- RF-local training-time density values like `logit_normal` and `cosine_shaped`

So the duplication problem is reduced, but the repo still needs to decide how
much of that surface should eventually share one timestep implementation story.

### 2. RF loss weighting is a much cleaner separate concept

`timestep.rf_loss_weighting_scheme` is a more defensible separate setting,
because it answers a different question:

- not "which times do we train on?"
- but "how do we weight the loss once time / sigma is known?"

### 3. `v_parameterization` is not really a generic loss setting

It belongs to the prediction-target stage, even if its current config home is
under `loss`.

### 4. `zero_terminal_snr` is not really the same kind of thing as noise offset

It belongs to path / state construction through DDPM scheduler shaping, even if
its current config home is under `loss.regularization`.

### 5. `cosine_shaped` is clearer than `mode`, but it is still just one local time-sampling formula

The rename from `mode` to `cosine_shaped` makes the intent clearer because the
current formula is driven by a cosine-based warp.

But it is still just one local time-sampling option inside the broader
time-sampling stage, not evidence of a separate category.

## What this means for future cleanup

Future work should follow this order:

1. classify each setting by stage
2. collapse duplicated user-facing choices within the same stage
3. move code ownership to match those stages
4. only then settle naming and final config homes

That means the next cleanup should not start from:

- "where should RF settings live?"

It should start from:

- "which active settings are path choices?"
- "which active settings are time-sampling choices?"
- "which active settings are prediction choices?"
- "which active settings are loss-weighting choices?"
- "which active settings are inference-integrator choices?"

## Recommended target ownership

This is still provisional, but it is the cleanest current direction:

- `library/objectives/`
  owns path / state construction and prediction-target behavior
- `library/timesteps/`
  owns shared time-sampling runtime behavior
- `library/losses/`
  should tend toward generic tensor loss primitives only
- formulation-specific loss weighting may eventually want to live nearer its
  objective owner if it is not truly generic
- inference sampler / pipeline runtime remains distinct from training-time
  time sampling

## Short version

The active layer should be read as a pipeline of stages, not as a pile of
settings grouped by historical origin.

The most important organizing categories are:

- path / state construction
- time sampling
- prediction target
- loss weighting
- inference integrator

The next cleanup should reduce duplicate user-facing settings inside those
stages, not keep multiplying config roots just because features arrived from
different script eras.
