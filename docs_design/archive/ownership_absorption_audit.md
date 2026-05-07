# Ownership Absorption Audit

This note broadens the old "training vs scripts" follow-up into the more
useful question that the first SD3 / RF port exposed:

- what should stay in `library/strategies/<family>/` or `library/models/<family>/`
- what should be absorbed by the rest of the repo over time
- which ownership axis that absorption should happen along

The important distinction is that the repo now has multiple independent axes:

1. **model family**
   - SD
   - SDXL
   - SD3
   - future families
2. **training objective / runtime formulation**
   - DDPM-style diffusion
   - RF / flow-matching
   - future formulations
3. **training mode**
   - PEFT
   - fine-tune
   - future modes
4. **shared training runtime / orchestration**
   - trainer
   - phases
   - validation/sampling/checkpoint cadence
   - common metadata/logging
5. **launcher / config composition**
   - `train.py`
   - factories
   - Hydra/dataclass schema

The first SD3 port put some formulation-level behavior under the SD3 strategy
simply because that was the first active consumer. This note tries to separate
"reasonable first-port placement" from "likely long-term owner."

## Main conclusion

The repo should not treat "currently lives under SD3" as evidence that a
behavior belongs to the SD3 strategy long-term.

The first RF-style port mainly exposed three kinds of code:

1. **genuinely model-family-specific code**
   - should stay in `strategies/<family>` or `models/<family>`
2. **formulation/runtime code currently parked in SD3**
   - should eventually move to an objective/runtime layer once the repo is
     ready to own that axis explicitly
3. **already-shared runtime code that should absorb more policy over time**
   - trainer/phases/timestep/logging/config should take this on, not a
     model-family shared strategy layer

So the question is not just "share or not share." The question is:

- **which axis owns this behavior?**

## Current axis map

### Model-family-owned today

- component loading and checkpoint conversion
- tokenizer bundle construction
- encoder payload shapes
- conditioning payload shapes
- denoiser call shape
- family-specific checkpoint serialization details
- family-specific sample pipeline construction

### Mode-owned today

- what becomes trainable
- optimizer param-group construction for the mode
- accelerator wrapping for the mode
- mode-specific save behavior
- per-step/per-epoch mode callbacks

### Shared runtime-owned today

- trainer lifecycle
- caching orchestration
- validation scheduling
- sampling cadence and prompt-file loading
- base metadata construction
- timestep runtime for the active DDPM-style path
- loss modifier runtime

### Launcher/config-owned today

- root config-driven launch surface
- `mode -> TrainingMode`
- `model.model_type -> TrainingStrategy`

## Ownership matrix

| Concern | Current home | Likely long-term owner | Status | Why |
| --- | --- | --- | --- | --- |
| SD / SDXL / SD3 component loading | `strategies/<family>/loading.py` + `models/<family>/loader.py` | `model family` | `stay` | Component bundle shape is family-specific. |
| Checkpoint conversion and format-specific save helpers | `models/<family>/conversion.py` | `model family` | `stay` | This is model/checkpoint format ownership, not runtime ownership. |
| Tokenizer bootstrap and family tokenizer bundle assembly | `models/sd/tokenizer.py`, `strategies/<family>/tokenization.py` | `model family` | `stay mostly` | Family encoder bundle shape still matters, even if small helpers are shared. |
| Text-conditioning payload objects (`Sd3TextConditioning`, `SdxlConditioning`, etc.) | `strategies/<family>/encoding.py`, `conditioning.py` | `model family` | `stay` | Payload shape is family-facing API to denoiser/diffusion internals. |
| Denoiser call bridge | `strategies/<family>/denoiser.py` | `model family` | `stay` | Call signatures and conditioning assembly still differ per family. |
| RF / flow timestep density helpers | `strategies/sd3/diffusion.py` | `objective/runtime` | `move later` | This is formulation logic, not SD3 logic in principle. |
| RF / flow post-loss weighting helpers | `strategies/sd3/diffusion.py` | `objective/runtime` | `move later` | Same as above: objective axis, not model-family axis. |
| RF noisy-input construction | `strategies/sd3/diffusion.py` | `objective/runtime` | `move later` | This is the RF analogue of DDPM noisy-latent runtime. |
| RF discrete-flow sampler math | `strategies/sd3/sampling.py` | `objective/runtime` or shared sampling runtime | `move later` | Reusable once another RF consumer exists; not a model-family abstraction. |
| Default noise scheduler creation | `library/training/noise_utils.py` | `objective/runtime` | `needs expansion` | Current helper is explicitly DDPM-shaped; objective selection should own this. |
| Timestep runtime construction | `library/timesteps/runtime.py` + trainer init | `shared runtime`, informed by `objective/runtime` | `needs expansion` | Good home already exists, but it currently centers DDPM-style assumptions. |
| Sampling prompt parsing / file loading / fan-out | `library/training/sample_generation.py` | `shared runtime` | `stay` | Already correctly shared across families. |
| Sampling cadence / trigger policy | `library/training/sample_generation.py`, `phases/validation.py`, `phases/triggers.py` | `shared runtime` | `stay` | Scheduling is orchestration, not model ownership. |
| Base checkpoint metadata assembly | `library/training/training_metadata.py` | `shared runtime` | `stay` | Central metadata construction is the right default. |
| Objective-specific metadata fields (currently SD3 weighting knobs) | `strategies/sd3/checkpointing.py` | `objective/runtime` or shared metadata builder | `move later` | These are formulation knobs, not model identity. |
| Model-spec metadata wrapper | `strategies/*/checkpointing.py::get_model_metadata()` | split: `model family` + maybe small helper families | `defer` | Still reasonable on strategy side for now. |
| What becomes trainable from LR policy | `training/modes/*`, `optimizer_utils.py` | `mode` + shared optimizer policy | `stay` | This is mode/runtime policy, not model-family policy. |
| Per-encoder semantic train flags like "train CLIP vs T5" | partially `strategies/sd3/model_preparation.py` | mixed: `mode/shared policy` plus family mapping | `defer` | Generic LR->flag logic is shared, but semantic role mapping still depends on family layout. |
| Cache incompatibility guard for T5 training + TE caching | `strategies/sd3/model_preparation.py` | mixed: family + runtime policy | `defer` | The incompatibility is formulation/runtime-facing, but still keyed to SD3's current encoder set. |
| RF config knobs currently stored under `cfg.model` via `getattr(...)` | SD3 strategy files | `objective/runtime config` | `move later` | These are not model identity/config in principle. |
| CLIP/T5 attention-mask application knobs | SD3 strategy/config attrs | likely split: family text-encoding behavior, maybe separate encoder/runtime config later | `defer` | Could be family text-encoding policy rather than pure objective policy. |
| Encoder dropout knobs (`clip_l_dropout_rate`, etc.) | SD3 strategy/config attrs | likely family text-conditioning/runtime policy | `defer` | Real training/runtime knobs, but not obviously a shared objective abstraction yet. |

## What should stay in strategies/models

These look healthy where they are:

### 1. Family component ownership

- `library/models/<family>/loader.py`
- `library/models/<family>/conversion.py`
- `library/models/<family>/mmdit.py` / `unet.py` / `vae.py`

Reason:

- they encode component topology and checkpoint format knowledge
- they are reusable outside the trainer
- the reuse axis is model family

### 2. Family conditioning and payload shapes

Examples:

- `Sd3TokenizedText`
- `Sd3TextConditioning`
- `SdxlConditioning`

Reason:

- these are local contracts between tokenization, encoding, conditioning,
  diffusion, and denoiser facets inside a family
- they represent real family-local shapes, not just temporary duplication

### 3. Family denoiser-call shape and sample pipeline construction

Reason:

- even if the surrounding runtime orchestration becomes more shared, the final
  denoiser bridge and pipeline construction are still family-owned

## What the rest of the repo should absorb later

### 1. Objective/runtime formulation logic

This is the biggest absorption target.

The clearest examples are the RF helpers currently living under SD3:

- `resolve_sd3_weighting_scheme(...)`
- `compute_density_for_timestep_sampling(...)`
- `compute_loss_weighting_for_sd3(...)`
- `get_noisy_model_input_and_timesteps(...)`
- `ModelSamplingDiscreteFlow`
- `get_all_sigmas(...)`
- `max_denoise(...)`

These live in SD3 now because SD3 is the first active RF consumer, not because
RF is model-locked to SD3.

Likely future homes:

- `library/timesteps/`
- `library/training/flow/`
- possibly `library/losses/` for loss-weighting pieces
- possibly shared sampling runtime helpers for inference-side RF math

### 2. Objective-driven scheduler creation

`library/training/noise_utils.py::get_noise_scheduler(...)` is currently a
DDPM-shaped default that the trainer always creates.

That works only because:

- SD / SDXL use it directly
- SD3 mostly ignores it in the core objective path

Long-term, the trainer should probably not say:

- "all runs get a DDPM scheduler"

It should say something more like:

- "build the runtime object needed for the active objective/formulation"

That does **not** mean strategies should own generic scheduler creation.
It means the shared runtime needs a cleaner formulation-aware seam.

### 3. Objective-level metadata and config

The config search shows that several RF knobs are still hanging off
`cfg.model` and accessed through `getattr(...)`, for example:

- `weighting_scheme`
- `logit_mean`
- `logit_std`
- `mode_scale`
- `sample_flow_shift`

These do not describe model identity. They describe training or sampling
formulation behavior.

So when this axis gets cleaned up, both of these should move together:

- code ownership
- config ownership

Likely direction:

- a typed objective/runtime config surface
- shared metadata builder absorbing objective metadata fields from that config

## What should probably stay out of strategies/shared

This is the main "don’t generalize the wrong way" warning.

The wrong move would be:

- seeing RF logic in `strategies/sd3/*`
- creating `library/strategies/shared/flow/`
- keeping formulation-level logic under the model-family strategy layer

That would still mix the wrong axes together.

If the repo absorbs these pieces correctly, the path should look more like:

- `trainer` / shared runtime asks for the active objective runtime
- the objective runtime owns scheduler/timestep/noisy-input/loss-weighting
  policy
- the strategy still owns family conditioning, denoiser call shape, and
  family-specific sample pipeline behavior

## Practical near-term calls

### Keep as-is for now

- SD3 RF helpers may stay in `strategies/sd3/*` until a second RF consumer
  appears or the repo is ready for a real objective/runtime layer.
- Family payload objects and denoiser bridges should stay local.
- Model conversion/save helpers should stay under `models/<family>/`.

### Good cleanup soon

- continue small family-sharing cleanups like CLIP helper extraction
- avoid pulling formulation logic into `strategies/shared/`
- prefer building explicit shared runtime seams where the trainer already has a
  home for them (`training/`, `timesteps/`, `losses/`, config schema)

## Concrete source -> target map

This is the practical "best place for now" mapping for the RF pieces currently
living under SD3.

### Training-side RF code

#### `library/strategies/sd3/diffusion.py`

Keep here for now:

- `encode_sd3_images_to_latents(...)`
- `shift_scale_sd3_latents(...)`
- the SD3-specific `DiffusionTrainingStrategy` wiring that bridges family
  conditioning into the shared trainer

Likely to move out later:

- `resolve_sd3_weighting_scheme(...)`
- `compute_density_for_timestep_sampling(...)`
- `compute_loss_weighting_for_sd3(...)`
- `get_noisy_model_input_and_timesteps(...)`

Best place for now:

- `library/timesteps/runtime.py`
  - timestep-density selection
  - shift-aware timestep/sigma shaping
- `library/training/noise_utils.py` or a small neighbor under
  `library/training/`
  - noisy-input construction for the active objective
- possibly `library/losses/`
  - if loss-weighting grows into a shared objective-level policy

Reason:

- these helpers operate on `latents`, `noise`, `timesteps`, `sigmas`, and
  weighting policy
- that is formulation/runtime behavior, not SD3 family identity

### Sampling-side RF code

#### `library/strategies/sd3/sampling.py`

Keep here for now:

- prompt encoding and SD3 conditioning assembly
- SD3-specific sample orchestration around the denoiser and VAE

Likely to move out later:

- `ModelSamplingDiscreteFlow`
- `get_all_sigmas(...)`
- `max_denoise(...)`
- the reusable parts of the Euler flow loop in `_do_sample(...)`

Best place for now:

- `library/pipelines/`

Closest existing analogue:

- `library/pipelines/gradual_latent.py`

Reason:

- these pieces are inference/sample-generation runtime math
- they are closer to sampling pipeline behavior than to model-family strategy
  ownership

### Metadata and config ownership

#### `library/strategies/sd3/checkpointing.py`

Keep here for now:

- SD3 full-model save orchestration
- SD3 model-spec metadata wiring

Likely to move out later:

- `ss_weighting_scheme`
- `ss_logit_mean`
- `ss_logit_std`
- `ss_mode_scale`

Best place for now:

- `library/training/training_metadata.py` or a nearby shared metadata helper

Reason:

- these fields describe objective/runtime behavior
- they do not describe SD3 model identity

#### Temporary config reads from `cfg.model`

Current RF-specific fields read from SD3 strategy code:

- `weighting_scheme`
- `logit_mean`
- `logit_std`
- `mode_scale`
- `sample_flow_shift`

Best place for now:

- `library/config/dataclasses/timestep.py`
  - `weighting_scheme`
  - `logit_mean`
  - `logit_std`
  - `mode_scale`
- `library/config/dataclasses/output.py`
  - `sample_flow_shift` under `output.sampling`

Reason:

- the first four are training/runtime timestep policy knobs
- `sample_flow_shift` is an inference/sampling default
- none of them are model-identity fields

## Suggested execution order

This is the smallest useful order if we want to improve ownership without
forcing a full objective-runtime redesign in one pass.

### Pass 1: config ownership

Move RF config fields to the schema areas that already match their meaning:

- training/runtime knobs -> `timestep`
- inference default -> `output.sampling`

Why first:

- lowest risk
- no algorithm change required
- makes the ownership problem visible in the config surface immediately

### Pass 2: metadata ownership

Stop recording objective knobs from SD3 strategy-owned config reads and source
them from shared config/metadata plumbing instead.

Why second:

- it naturally follows config cleanup
- still low-risk compared to runtime extraction

### Pass 3: training-side helper extraction

Move RF training helpers out of `strategies/sd3/diffusion.py` into the shared
runtime areas that already exist.

Why third:

- by this point the config surface already reflects the right ownership axis
- helper extraction can happen without also redesigning config access

### Pass 4: sampling-side helper extraction

Move reusable RF sampling math out of `strategies/sd3/sampling.py` if and when
it becomes worth centralizing.

Why fourth:

- less urgent than training-side ownership pressure
- the repo already has a plausible sampling-side home in `library/pipelines/`
- this can wait until there is either a second consumer or a clearer shared
  sampling runtime shape

### Strongest future refactor target

If the repo wants to support more than one objective cleanly, the strongest
next abstraction target is:

- an explicit **objective/runtime** layer

not:

- a larger model-family strategy layer

## Short version

- Some code is in SD3 today only because SD3 was the first RF port.
- That does **not** mean it belongs to SD3 long-term.
- The repo should eventually absorb formulation-level code along an
  **objective/runtime** axis, not a model-family shared-strategy axis.
- Strategies/models should keep owning what is truly family-shaped:
  component loading, payloads, conditioning shape, denoiser call, and
  format-specific checkpoint behavior.
