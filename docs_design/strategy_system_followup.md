# Strategy System Follow-Up

## Purpose

Capture the concrete strategy-layer cleanup that is still worth doing after
the recent Trainer/TrainingMode refactor.

The high-level direction is still correct:

- scripts choose config + strategy + mode
- `Trainer` orchestrates shared flow
- `TrainingMode` owns training-mode divergence
- `TrainingStrategy` owns model-family behavior

This note exists to make the remaining gaps precise enough to implement
without rediscovering the boundary issues during the work.

## Investigated Assessment

After checking the current code, the original concern was directionally right,
but a few items need sharper wording.

What is true:

1. active shared phases still leak SDXL-shaped token / TE assumptions
2. the `TrainingStrategy` surface still mixes active, transitional, and legacy
   responsibilities
3. some shared strategy defaults are CLIP-specific, not model-agnostic

What needs correction:

1. not every "old" strategy hook is legacy-only; some are still active in the
   current runtime
2. one cited utility issue is real but only affects deprecated paths
3. the lower data layer already supports variable encoder names better than the
   original note implied

## Current Reality

One important framing detail: the new `Trainer` + phase pipeline is currently
used by active SDXL entrypoints (`scripts/sdxl_peft.py`,
`scripts/sdxl_finetune.py`). The SD strategy is still transitional relative to
that path.

That means the cleanup goal is not "pretend SD is fully unified already."
The goal is to make the active SDXL shared path honest and strategy-driven,
while keeping SD explicitly transitional until its data-path migration is done.

Also: deprecated / legacy code in this repo should be treated as reference
material to replace, not compatibility surface to preserve. The cleanup target
is migration and deletion, not long-term coexistence.

## Confirmed Gaps

### 1. Shared phases still hardcode SDXL token / TE shape

Confirmed active call sites:

- `library/training/phases/training_loop.py:445`
  - epoch tokenization passes `encoder_names=["clip_l", "clip_g"]`
- `library/training/phases/caching.py:134`
  - disk TE caching packs `model=(*trainer.text_encoders, *trainer.tokenizers)`

Why this matters:

- the shared phase layer still knows SDXL-specific encoder names and cache-pack
  ordering
- future model families would still require edits in generic orchestration code

Important nuance:

- the lower data layer is already partly ready for variable shapes
  - `tokenize_epoch_manifest(...)` can infer encoder names from output count
  - `BatchInfo.input_ids` is already a `dict[str, ...]`
- the real leak is the phase-to-strategy handoff, not the manifest schema

### 2. The strategy contract still mixes active, transitional, and legacy surface

This is still a real problem, but the original note was too coarse.

Current status is closer to:

- **Active in current runtime**
  - `get_latents_caching_strategy()` is still used in `Trainer.setup()` to set
    the active `LatentsCachingStrategy`
  - `get_models_for_text_encoding()` is still used by current SD / SDXL batch
    and validation code
- **Active only for the new shared SDXL path**
  - `create_latent_caching_strategy()`
  - `create_te_caching_strategy()`
  - `tokenize_captions()`
  - `encode_te_outputs_in_memory()`
- **Legacy / deprecated-path surface**
  - `get_text_encoder_outputs_caching_strategy()`
  - `cache_text_encoder_outputs_if_needed()`

The current ABC does not make those categories obvious, and
`SdTrainingStrategy` still advertises some new-pipeline methods via
`NotImplementedError` stubs.

That mismatch is real and should be cleaned up.

### 3. Some base strategy defaults are CLIP-specific

Confirmed:

- `library/strategies/base/training.py:619`
  - `prepare_text_encoder_grad_ckpt_workaround()` reaches into
    `text_encoder.text_model.embeddings`
- `library/strategies/base/training.py:629`
  - `prepare_text_encoder_fp8()` does the same

Why this matters:

- these hooks are called from shared phases
- the default implementation already assumes a CLIP-family encoder shape
- that makes the base strategy less trustworthy as a model-agnostic boundary

### 4. Active code still relies on global strategy singletons

This was missing from the original note and is worth capturing.

Confirmed active wiring:

- `Trainer.setup()` still calls:
  - `TokenizeStrategy.set_strategy(...)`
  - `TextEncodingStrategy.set_strategy(...)`
  - `LatentsCachingStrategy.set_strategy(...)`
- the active SDXL sampling pipeline reads global strategy state:
  - `library/pipelines/sdxl_lpw_stable_diffusion.py:978`

Why this matters:

- strategy behavior is not fully passed through explicit object boundaries
- the active runtime still depends on process-global mutable strategy state
- this makes tests, reuse, and future multi-family support harder to reason
  about

This is not a correctness emergency, but it is a real boundary leak.

## Corrections To The Original Note

### 1. `get_latents_caching_strategy()` is not just legacy baggage

It is still part of the active runtime because the current data-loading /
cache-loading path still uses `LatentsCachingStrategy`.

That means it should be treated as an active transitional hook, not a dead
legacy method.

### 2. `get_models_for_text_encoding()` is also still active

This hook is still used by current training and validation code inside the
concrete SD / SDXL strategies, especially for wrapped vs unwrapped TE handling.

It should not be grouped with deprecated-only surface.

### 3. The LR naming example is lower priority than first stated

`library/training/trainer_utils.py:259` still hardcodes:

- `unet`
- `text_encoder1`
- `text_encoder2`

But that helper is only used by deprecated scripts and its unit tests. The
current `Trainer` flow already logs LR data through `generate_step_logs(...)`
plus `trainer.lr_descriptions`.

So the concern is real, but it is not an active shared-runner problem.

## Recommended Cleanup Direction

### 1. Make the active strategy contract explicit

Do not treat the current `TrainingStrategy` ABC as one flat surface.
Document and encode three categories:

- active shared-runner contract
- active transitional hooks
- legacy compatibility hooks

This can be done either with small sub-protocols/mixins or with explicit
sectioning plus comments and tests. The important part is that the runtime
contract becomes honest.

### 2. Move token / TE shape ownership fully into strategies

Shared phases should not decide:

- token cache encoder names
- TE-cache model packing order
- how many encoders exist

Those should come from strategy hooks.

### 3. Remove CLIP assumptions from the generic base class

The shared base should not contain default behavior that assumes
`text_encoder.text_model.embeddings`.

Move those implementations into SD / SDXL concrete strategies, or a dedicated
CLIP-family helper under model-family-owned code if duplication becomes
annoying. They should not stay in the generic base.

One follow-up worth auditing during implementation: some text-encoder behavior
already lives under model-family code (`library/models/sdxl/text_encoder.py`),
while other behavior still lives in strategy/encoding modules. If CLIP-specific
logic is extracted further, it should move toward model-family ownership rather
than back into shared base abstractions.

### 4. Remove active singleton strategy coupling where practical

The target should be:

- `Trainer` passes strategy objects explicitly where the active runtime needs
  them
- sampling pipelines stop discovering strategy behavior through global state

Legacy scripts can keep singleton wiring until they are removed, but the active
shared path should stop depending on it.

This work should include a small pipeline-role audit rather than assuming
`library/pipelines` is only a passive implementation detail. Current evidence
shows:

- active runtime usage for sampling through `sample_images_common(...)`
- a separate `gradual_latent.py` utility module that appears standalone /
  inference-oriented rather than part of the active training runner

That means the cleanup should verify whether the folder should remain a
sampling-focused home, be split by responsibility, or simply be treated as a
sampling dependency boundary.

### 5. Keep SD explicitly transitional until its pipeline migration is done

Do not force fake completeness into `SdTrainingStrategy`.

Either:

- keep SD out of the new shared-path contract until migrated, or
- split the contract so SD only implements the pieces it truly supports today

What should be avoided is a primary contract that implies runtime support which
is actually implemented as `NotImplementedError`.

## Actionable Implementation Plan

### Phase 1: Strategy-owned token / TE cache handoff

Goal: remove the two active SDXL-shaped leaks from shared phases without
changing architecture.

Changes:

- add a strategy hook for epoch-token cache naming
  - example shape: `get_token_cache_encoder_names() -> list[str]`
- add a strategy hook for TE-cache model packing
  - example shape:
    `build_te_cache_model_bundle(cfg, accelerator, text_encoders, tokenizers) -> Any`
- update:
  - `library/training/phases/training_loop.py`
  - `library/training/phases/caching.py`
- implement in:
  - `library/strategies/sdxl/training.py`
  - `library/strategies/sd/training.py` if needed for future compatibility, or
    leave clearly transitional if no active caller exists yet

Tests:

- unit test that epoch tokenization uses strategy-provided encoder names
- unit test that TE disk caching uses strategy-provided model bundle
- regression test that current SDXL caching/token behavior is unchanged

### Phase 2: Split active vs legacy strategy surface

Goal: make the contract honest without redesigning Trainer or TrainingMode.

Changes:

- classify methods in `library/strategies/base/training.py` into:
  - active shared-runner hooks
  - active transitional hooks
  - deprecated / legacy hooks
- decide one of these minimal implementations:
  - small sub-protocols / mixins, or
  - one ABC with clearly separated sections and comments plus tests
- demote deprecated hooks so they are not presented as part of the primary
  active contract
- treat deprecated hooks as migration targets to replace and delete, not as API
  surface to keep supporting indefinitely

Recommended minimum:

- keep `get_latents_caching_strategy()` as active transitional
- keep `get_models_for_text_encoding()` as active
- move `get_text_encoder_outputs_caching_strategy()` and
  `cache_text_encoder_outputs_if_needed()` out of the primary active contract

Tests:

- unit tests that active strategy mocks only need the active contract for new
  runner tests
- explicit tests that deprecated-path hooks are not required by current
  `Trainer` execution

### Phase 3: Remove CLIP-specific defaults from the base strategy

Goal: make the base class generic again.

Changes:

- move:
  - `prepare_text_encoder_grad_ckpt_workaround()`
  - `prepare_text_encoder_fp8()`
  out of generic base defaults
- implement them in concrete SD / SDXL strategies, or a CLIP-family helper in
  model-family-owned code
- leave the base contract abstract or explicitly no-op only where that is
  genuinely generic
- audit whether additional TE behavior currently stranded in strategy modules
  should move into `library/models/...` helpers for model-family ownership

Files:

- `library/strategies/base/training.py`
- `library/strategies/sd/training.py`
- `library/strategies/sdxl/training.py`
- callers already live in:
  - `library/training/phases/optimizer.py`
  - `library/training/phases/model_prep.py`

Tests:

- unit tests for SD and SDXL hook behavior
- negative test that a non-CLIP test double does not depend on hidden
  `.text_model.embeddings` structure in the base class

### Phase 4: Remove active singleton strategy dependence

Goal: make active runtime dependencies explicit.

Changes:

- audit `library/pipelines` usage and define its intended scope before
  refactoring the sampling boundary
- trace which active paths still require:
  - `TokenizeStrategy.set_strategy(...)`
  - `TextEncodingStrategy.set_strategy(...)`
  - `LatentsCachingStrategy.set_strategy(...)`
- stop relying on global lookup in the active SDXL sampling path
  - likely update `sample_images_common()` / pipeline setup to pass the
    tokenize and encoding strategy explicitly
- keep singleton wiring only where deprecated scripts still need it

Files likely involved:

- `library/training/runners/trainer.py`
- `library/training/sample_generation.py`
- `library/pipelines/sdxl_lpw_stable_diffusion.py`
- `library/pipelines/lpw_stable_diffusion.py`
- `library/pipelines/gradual_latent.py` for scope clarification only
- possibly tokenization / encoding base modules if helper accessors are changed

Tests:

- sampling test proving SDXL prompt encoding works without pre-populated global
  singleton state
- regression test that training setup can still run sample generation

### Phase 5: SD migration follow-through

Goal: finish the contract cleanup only after deciding how SD joins the new
shared pipeline.

Changes:

- either migrate SD onto the same token / TE caching path
- or keep SD on a clearly separate transitional path and do not require the
  new-pipeline hooks in the primary strategy contract yet

This phase should be sequenced with the broader SD data-pipeline migration in
the roadmap. It should not be faked early by leaving
`NotImplementedError` methods on the main contract.

## Recommended Order

1. Phase 1: remove the two active shared-phase leaks
2. Phase 2: make the contract categories explicit
3. Phase 3: remove CLIP-specific base defaults
4. Phase 4: remove active singleton dependence
5. Phase 5: finish SD alignment when SD pipeline migration is actually ready

This ordering keeps the cleanup practical:

- first fix the active boundary leak
- then make the contract truthful
- then remove hidden generic-base assumptions
- then clean up the remaining active global state
- only after that finalize SD alignment

## Non-Goals

- no new runners
- no mode-layer redesign
- no rework of the high-level `Trainer` + `TrainingMode` architecture
- no premature full unification of SD before its data path is ready
- no attempt to preserve deprecated reference code as a stable parallel system
- no performance tuning plan

## Acceptance Criteria

When this follow-up is done, the result should:

1. remove SDXL-specific token / TE shape knowledge from shared phases
2. make the active strategy contract distinguishable from legacy compatibility
   surface
3. remove CLIP-family assumptions from generic base defaults
4. stop the active runtime from depending on hidden global strategy state
5. keep SD explicitly transitional until its real migration is complete
6. preserve current SDXL runtime behavior throughout the cleanup
