# Cross-Model Extraction Audit

This pass follows the contract-pressure audit and asks a narrower question:

- what code should actually be extracted across active model families
- what code only looks shared because the current ports are adjacent
- what code should move to a different layer instead of becoming a new
  strategy-base/shared seam

Active model families compared here:

- `sd`
- `sdxl`
- `sd3`

This is not a final architecture decision document. It is a working audit to
guide the next cleanup and porting passes.

## Main conclusion

The first SD3 port does **not** currently justify a large new shared strategy
layer.

Most of the real cross-model reuse is still clustered in two places:

1. CLIP-family behavior shared by `sd` and `sdxl`
2. generic trainer/runtime helpers that already live outside model strategies

SD3 mostly adds new behavior relative to the currently active SD / SDXL path:

- CLIP + T5 tokenization/encoding
- rectified-flow / flow-matching timestep density and post-loss weighting
- discrete-flow sampling
- SD3-specific checkpoint save format constraints
- SD3-specific cached/live conditioning rules

That means the current pressure is mostly about:

- extracting a **small number of CLIP-family helpers** cleanly
- avoiding premature "shared" SD3 abstractions
- identifying RF / flow-formulation helpers that may eventually want a
  non-strategy home once the repo has a second consumer for them

## Decision categories

- `extract now`: safe shared extraction candidate with clear same-reason
  change pressure
- `defer`: multiple families look similar, but the code still changes for
  different reasons or the abstraction is not mature enough
- `move elsewhere later`: probably real reusable behavior, but its natural home
  is not a model-family strategy seam
- `keep family-local`: intentionally model-specific behavior

## Matrix

| Concern | Current home | SD | SDXL | SD3 | Decision | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP prompt-attention parsing / long-prompt chunking / weighted token ids | `library/strategies/shared/clip/tokenization.py` | yes | yes | CLIP bootstrap only | `already extracted well` | Good example of real shared strategy behavior. |
| Hugging Face tokenizer bootstrap / cache loading | `library/models/sd/tokenizer.py` | yes | yes | yes | `already extracted well` | Correctly treated as component/bootstrap logic, not strategy logic. |
| Prompt file loading and prompt fan-out | `library/training/sample_generation.py` | yes | yes | yes | `already extracted well` | This is runtime/orchestration behavior, not model-family behavior. |
| Sample cadence checks | `library/training/sample_generation.py` | yes | yes | yes | `already extracted well` | Shared runtime helper with no family policy. |
| Generic latent preparation | `library/training/diffusion.py` | yes | yes | yes | `already extracted well` | Family-specific encoding/scaling enters only through explicit callbacks. |
| DDPM-style noise/timestep runtime | `library/training/diffusion.py` + `library/timesteps/` | yes | yes | no | `already extracted well` | Correctly shared for SD / SDXL without forcing SD3 flow-matching into the same path. |
| Base training metadata assembly | `library/training/training_metadata.py` | yes | yes | yes | `already extracted well` | Strategy metadata hooks are now the right narrow extension point. |
| CLIP text-encoder grad-ckpt workaround | `strategies/sd*/model_preparation.py` | same in SD | same in SDXL | partial analogue | `extract now (CLIP-only helper)` | SD and SDXL are identical; SD3 uses CLIP for indexes 0/1 but T5 diverges. |
| CLIP embedding FP8 workaround | `strategies/sd*/model_preparation.py` | same in SD | same in SDXL | partial analogue | `extract now (CLIP-only helper)` | Same shape as grad-ckpt workaround: share the CLIP path, keep T5 override local. |
| Basic cast decisions (`cast_text_encoder`, `cast_vae`, `cast_denoiser`) | `strategies/*/model_preparation.py` | identical | identical | identical today | `defer` | These are currently trivial and explicit on purpose; extracting them buys little while the model-prep seam is still settling. |
| SD / SDXL model metadata wrapper around `get_model_metadata_from_config(...)` | `strategies/sd/checkpointing.py`, `strategies/sdxl/checkpointing.py` | thin wrapper | thin wrapper | different metadata + save policy | `defer` | Probably extractable into a Stable-Diffusion-family helper later, but not urgent. |
| SD / SDXL LPW sampling orchestration | `strategies/sd/sampling.py`, `strategies/sdxl/sampling.py` | very similar | very similar | no | `defer` | Shared structure is real, but pipeline construction/runtime state still differs enough that a helper may add indirection before it removes duplication. |
| RamTorch application across components | `strategies/*/loading.py` | similar | similar | similar but broader encoder set | `defer` | There is likely a helper hiding here, but the family component bundle shape still differs enough that a generic extraction would probably branch immediately. |
| `replace_unet_modules(...)` / VAE xformers setup | `strategies/sd/loading.py`, `strategies/sdxl/loading.py` | yes | yes | no | `defer` | This is a Stable-Diffusion-family concern, but SD3 currently provides no second consumer. |
| RF / flow-matching timestep density helpers (currently exercised only by SD3 in-repo) | `strategies/sd3/diffusion.py` | no | no | yes | `move elsewhere later` | If another RF / flow family lands, these likely want a `timesteps/` or `training/flow/` home, not `strategies/shared/`. |
| RF / flow post-loss weighting helpers (currently exercised only by SD3 in-repo) | `strategies/sd3/diffusion.py` | no | no | yes | `move elsewhere later` | Same pressure as above; likely reusable by flow-style training formulations, not by model families in general. |
| RF discrete-flow sampler (`ModelSamplingDiscreteFlow`, sigma helpers; currently exercised only by SD3 in-repo) | `strategies/sd3/sampling.py` | no | no | yes | `move elsewhere later` | Feels more like reusable inference/runtime math than family strategy glue, but only after a second consumer exists. |
| Generic sampling prompt parsing syntax | `library/training/sample_generation.py` | yes | yes | yes | `already extracted well` | `flow_shift` support in the parser is already in the correct shared runtime layer. |
| Conditioning resolution high-level seam | `strategies/*/conditioning.py` | yes | yes | yes | `already extracted well` | The contract is shared; helper-level steps should stay family-local for now. |
| Conditioning helper internals (`_get_cached_*`, `_encode_live_*`, `_merge_*`) | `strategies/*/conditioning.py` | yes | yes | yes | `keep family-local` | Same responsibility, different payload shapes and merge rules. |
| SD3 named token / conditioning payload objects | `strategies/sd3/encoding.py` | no | no | yes | `keep family-local` | Good local normalization; not evidence for a universal payload abstraction yet. |
| SDXL pooled-text + micro-conditioning path | `strategies/sdxl/encoding.py`, `strategies/sdxl/conditioning.py` | no | yes | no | `keep family-local` | Distinct enough to justify staying local. |
| SD clip-skip hidden-state handling | `strategies/sd/encoding.py` | yes | no | no | `keep family-local` | Real SD-family behavior, not a cross-family seam. |

## What looks safest to extract now

### 1. CLIP-family text-encoder prep helpers

The cleanest immediate extraction candidate is a tiny shared helper layer for
the CLIP text-encoder preparation behavior currently duplicated in SD and SDXL:

- `prepare_text_encoder_grad_ckpt_workaround(...)`
- `prepare_text_encoder_fp8(...)`

Why this one is safe:

- SD and SDXL implementations are identical
- SD3 partly shares the CLIP side, but also has a clear divergent T5 branch
- the abstraction can stay narrow and capability-shaped

Suggested direction:

- add a small helper module under `library/strategies/shared/clip/`
- keep SD3 free to call the CLIP helper for encoder indexes `0` and `1`
- keep SD3 T5-specific behavior local

This is a better target than extracting the whole `model_preparation.py`
facet, because the rest of the facet still contains family policy:

- SDXL post-trainable freezing
- SD3 T5 cache/train guard
- future denoiser-family-specific prep hooks

### 2. Possibly a small Stable-Diffusion-family metadata helper later

`sd` and `sdxl` both wrap `get_model_metadata_from_config(...)` with very
similar argument shapes.

This does look like real duplication, but it is lower priority than the CLIP
prep helpers because:

- the duplication is already tiny
- the save-path behavior diverges immediately after the metadata helper
- SD3 already needs a distinct metadata shape and save policy

Suggested stance:

- do not extract this first
- revisit only if another Stable-Diffusion-family checkpointing strategy lands
  with the same wrapper shape

## What should not be extracted yet

### 1. SD / SDXL sampling strategies

`library/strategies/sd/sampling.py` and
`library/strategies/sdxl/sampling.py` are structurally close:

- unwrap models
- temporarily move components
- build family pipeline
- call `sample_images_common(...)`
- restore devices/dtypes

But the differences are still meaningful:

- different pipelines
- different text-encoder bundle shapes
- SDXL needs strategy-aware pipeline construction

This means a shared helper could easily turn into:

- generic shell code
- plus an awkward callback bundle
- plus model-family branches

That is exactly the kind of extraction this pass should avoid.

### 2. Loading helpers across families

All three loading strategies share some visible patterns:

- call family loader
- optionally apply RamTorch
- record family runtime state

But the component bundles already differ:

- SD: one text encoder, UNet, VAE
- SDXL: two text encoders, checkpoint metadata extras
- SD3: three possible text encoders, MMDiT, optional missing encoders

A shared extraction here is more likely to hide the family bundle shape than to
clarify it.

### 3. Conditioning helper-level decomposition

The new `resolve_conditioning(...)` seam looks correct, but helper-level
sharing still does not.

The families differ in:

- payload object shape
- cache/live merge rules
- whether partial cached outputs are acceptable
- whether masks and pooled outputs are mandatory

So the right shared move already happened at the high-level contract seam.
The internal helper steps should stay local until a stronger convention
appears.

## What probably wants a different long-term home

### 1. RF / flow-matching runtime math

These helpers are currently local to `library/strategies/sd3/diffusion.py`:

- `compute_density_for_timestep_sampling(...)`
- `compute_loss_weighting_for_sd3(...)`
- `get_noisy_model_input_and_timesteps(...)`

That is reasonable for the first RF-model port in this repo.

But if another RF / flow-based family arrives, these should probably not
become `strategies/shared/flow/` just because two model families use them.
Their more natural home would likely be one of:

- `library/timesteps/`
- `library/training/flow/`
- `library/losses/` for the loss-weighting portion

The reuse axis here is training formulation/runtime math, not model family.

### 2. RF discrete-flow sampling math

These helpers are currently local to `library/strategies/sd3/sampling.py`:

- `ModelSamplingDiscreteFlow`
- `get_all_sigmas(...)`
- `max_denoise(...)`

That is acceptable while SD3 is the only active RF-style consumer in this repo.

If another flow model arrives, they will likely want to move to a runtime math
home that can be shared by:

- family-specific sampling strategies
- possibly validation/inference helpers

Again, the reuse axis is "flow-based sampling runtime", not "model-family
strategy".

## Family-local behavior that looks healthy

Some code is new and model-specific, but that is not a smell.

Healthy examples:

- `Sd3TokenizedText` and `Sd3TextConditioning` in
  `library/strategies/sd3/encoding.py`
- SD3 cached-text dropout and CLIP/T5 concat helpers
- SDXL pooled-output and micro-conditioning handling
- SD clip-skip hidden-state handling

These are good examples of local normalization that improve the port without
forcing a new global abstraction.

## Recommended next actions

### Low-risk cleanup now

1. Extract the duplicated CLIP text-encoder prep helpers into a small shared
   module under `library/strategies/shared/clip/`.
2. Let SD and SDXL call that helper directly.
3. Let SD3 reuse only the CLIP portion for encoder indexes `0` and `1`, while
   keeping T5 behavior local.

### Explicitly defer

1. Do not extract SD / SDXL sampling strategy shells yet.
2. Do not extract family loading helpers yet.
3. Do not turn SD3 flow-matching helpers into a strategy-shared module yet.

### Use as input to the next pass

This extraction audit should feed directly into the next
`library/training/` vs script/library ownership audit:

- anything marked `move elsewhere later` is a candidate to evaluate there
- anything marked `extract now` is a safe small cleanup before the next model
  family raises the same issue again

## Short version

- The strongest real extraction candidate today is a **small CLIP-family model
  preparation helper**.
- The strongest false-generalization risk today is **trying to share RF
  flow-matching or sampling helpers too early through the model-strategy
  layer**, before the repo has a second consumer and a better runtime home for
  them.
- The right next ownership question is not "what else can move into
  `strategies/shared/`?" but "which new reusable runtime pieces belong outside
  model strategy entirely if another flow family arrives?"
