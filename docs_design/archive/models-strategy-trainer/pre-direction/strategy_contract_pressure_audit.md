# Strategy Contract Pressure Audit

This pass compares the current strategy contracts against the three active
model families:

- `sd`
- `sdxl`
- `sd3`

The goal is not to finalize contracts yet. The goal is to identify which seams
look stable, which are too strict, and which are probably sitting at the wrong
layer.

## Status note

This document started as a pressure scan, not a final design note. Some items
below are now partially or fully resolved in the codebase, so this pass marks
what is already settled versus what is still open.

This audit follows the contract philosophy in
`library/strategies/README.md`:

- A contract should exist only for behavior that is genuinely shared.
- Not every model family needs every possible behavior.
- Some hooks should stay optional.

## Main finding

The current `TrainingStrategy` surface mixes together two different things:

1. trainer-facing contracts that shared runtime code depends on directly
2. family-internal facet glue used only inside strategy implementations

That distinction matters.

Examples of trainer-facing contracts:

- `load_target_model`
- `create_latent_caching_strategy`
- `sample_images`
- `update_metadata`
- `process_batch`
- `call_denoiser`

Examples of family-internal facet glue currently exposed as base contracts:

- `tokenize`
- `tokenize_with_weights`
- `encode_tokens`
- `encode_tokens_with_weights`
- `get_models_for_text_encoding`

Those methods are real responsibilities, but they are not all trainer
contracts. Some of them are only used by diffusion/sampling facets and
pipelines inside a model family.

That does not mean strategies should become loose. The current combined
`TrainingStrategy` interface still makes sense as a single surface containing:

- required capabilities that every active model family needs
- optional capabilities that any model family may use, but not all must use

## Matrix

| Facet / method | Shared consumer today | SD | SDXL | SD3 | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `ModelLoadingStrategy.load_target_model` | trainer setup | yes | yes | yes | `stable` | Core trainer dependency. |
| `ModelLoadingStrategy.load_denoiser_lazily` | model prep phase when denoiser is deferred | optional | optional | optional | `stable optional hook` | Already correctly modeled as optional. |
| `TokenizationStrategy.tokenizers` | trainer setup, caching, sampling | yes | yes | yes | `stable` | Shared runtime expects tokenizers to exist after strategy init. |
| `TokenizationStrategy.tokenize` | diffusion + sampling facets | yes | yes | yes | `needs reclassification` | Shared, but not trainer-facing. Better treated as facet capability than core trainer contract. |
| `TokenizationStrategy.tokenize_with_weights` | SD/SDXL diffusion + LPW pipeline | yes | yes | no | `resolved as optional capability` | This now lives on `WeightedPromptStrategy` rather than the required tokenization facet. |
| `TokenizationStrategy.tokenize_captions` | per-epoch token cache path | yes | yes | yes | `stable feature contract` | Shared phase code uses this when token caching is enabled. |
| `TextEncodingStrategy.encode_tokens` | diffusion + sampling facets | yes | yes | yes | `needs reclassification` | Shared across families, but not directly trainer-facing. |
| `TextEncodingStrategy.encode_tokens_with_weights` | SD/SDXL diffusion + LPW pipeline | yes | yes | no | `resolved as optional capability` | This now lives on `WeightedPromptStrategy` rather than the required text-encoding facet. |
| `TextEncodingStrategy.encode_te_outputs_in_memory` | in-memory TE caching phase | yes | yes | yes | `stable feature contract` | Shared caching phase depends on this directly. |
| `TextEncodingStrategy.get_models_for_text_encoding` | diffusion facets only | passthrough | unwraps TE2 | passthrough | `stable optional hook` | Even when it is a passthrough, it gives the family a clean seam to adjust live encoding model bundles. |
| `CachingStrategy.create_latent_caching_strategy` | caching phase | yes | yes | yes | `stable` | Core shared runtime seam. |
| `CachingStrategy.create_te_caching_strategy` | disk TE caching phase | yes | yes | yes | `stable feature contract` | Shared runtime depends on it when TE disk caching is enabled. |
| `CachingStrategy.get_token_cache_encoder_names` | per-epoch token cache path | yes | yes | yes | `stable feature contract` | Shared loop should not hardcode encoder names. |
| `CachingStrategy.build_te_cache_model_bundle` | disk TE caching phase | yes | yes | yes | `stable feature contract` | Important because SDXL and SD3 pack model bundles differently. |
| `SampleGenerationStrategy.sample_images` | training loop + orchestration helpers | yes | yes | yes | `stable` | Core trainer dependency. |
| `CheckpointingStrategy.update_metadata` | trainer metadata build | yes | yes | yes | `stable` | Shared metadata assembly depends on it. |
| `CheckpointingStrategy.get_model_metadata` | trainer + PEFT mode | yes | yes | yes | `stable` | Shared save paths depend on it. |
| `CheckpointingStrategy.save_model_checkpoint` | finetune mode full-model save path | no override | yes | yes | `stable optional hook` | Already correctly modeled as optional. |
| `ValidationStrategy.calculate_val_loss` | validation scheduler path | yes | yes | yes | `stable` | Shared trainer dependency. |
| `DiffusionTrainingStrategy.process_batch` | main training loop | yes | yes | yes | `stable` | Core trainer dependency. |
| `DenoiserCallingStrategy.call_denoiser` | diffusion facets | yes | yes | yes | `stable` | Clean contract despite very different call signatures behind the seam. |
| `ModelPreparationStrategy.cast_text_encoder` | model prep phase | yes | yes | yes | `stable` | Keep explicit at this stage even when values currently match. |
| `ModelPreparationStrategy.cast_vae` | model prep phase | yes | yes | yes | `stable` | Keep explicit at this stage even when values currently match. |
| `ModelPreparationStrategy.cast_denoiser` | model prep phase | yes | yes | yes | `stable` | Keep explicit at this stage even when values currently match. |
| `ModelPreparationStrategy.prepare_text_encoder_grad_ckpt_workaround` | optimizer setup | yes | yes | yes | `stable optional hook` | All active families need it, but semantics differ per encoder family. |
| `ModelPreparationStrategy.prepare_text_encoder_fp8` | model prep phase when FP8 is enabled | yes | yes | partial | `stable optional hook` | Still useful, but SD3 shows support can be partial by encoder type. |
| `ModelPreparationStrategy.post_process_trainable` | PEFT + finetune mode | no-op | yes | yes | `stable optional hook` | Keep explicit for now; no-op implementations are acceptable where a family does not need extra handling. |

## Highest-pressure areas

### 1. Prompt weighting is a shared optional capability, not a trainer-critical universal one

Status: `done`

`sd` and `sdxl` support:

- `tokenize_with_weights`
- `encode_tokens_with_weights`

The current `sd3` port does not, but that alone should not be read as "prompt
weighting is not a valid contract". A better framing is:

- prompt weighting is mainly an inference-oriented capability
- it applies naturally to CLIP or other non-LLM text encoders
- not every training path depends on it

This was the clearest current example of an optional capability being
mistakable for a mandatory one, and it is now resolved in code:

- weighted prompt support moved off the required tokenization/text-encoding
  facets
- `WeightedPromptStrategy` now owns `tokenize_with_weights()` and
  `encode_tokens_with_weights()`
- SD / SDXL opt into that capability explicitly
- SD3 no longer pretends to support weighted prompts via placeholder methods

### 2. `get_models_for_text_encoding()` is acceptable as a strategy seam

Status: `still open, but lower pressure`

Even though only `sdxl` currently transforms the model list in a meaningful
way, the seam itself is still reasonable:

- some families will just pass models through unchanged
- some families will need wrapped or unwrapped variants
- keeping that decision in strategy avoids leaking family-specific handling
  into diffusion or trainer code

So this does not currently look like a problem. It is a valid optional
capability seam even when most families use the passthrough case.

### 3. Tokenization and text encoding contracts are partly in the wrong layer

Status: `partially addressed`

The trainer directly depends on only a subset of tokenization/text-encoding
behavior:

- `tokenizers`
- `tokenize_captions`
- `encode_te_outputs_in_memory`

The rest is mostly consumed by model-family diffusion or sampling code.

That means the current base interfaces are doing double duty:

- exposing trainer/runtime capabilities
- and expressing internal collaboration between concern files

That is workable for now, but we should not mistake those internal seams for
final trainer contracts.

Since this audit started, one part of the tokenization split has been clarified
in code:

- tokenizer/bootstrap loading now lives in `library/models/sd/tokenizer.py`
- shared CLIP-family tokenization behavior now lives in
  `library/strategies/shared/clip/tokenization.py`

What is still open is the contract question itself: `tokenize`,
`encode_tokens`, and `get_models_for_text_encoding` are still strategy-owned,
but they still look more like strategy-internal collaboration seams than
trainer-facing runtime contracts.

### 4. Model-preparation hooks should stay explicit for now

Status: `still valid`

Even where the current families line up, there is no real benefit in adding
defaults yet. Keeping these hooks explicit makes model-family intent clearer
while the strategy surface is still settling:

- `cast_text_encoder`
- `cast_vae`
- `cast_denoiser`
- `post_process_trainable`

### 5. Per-text-encoder train control is mostly a shared concern, with strategy-owned shape

Status: `still valid`

`get_text_encoders_train_flags()` lives in `library/optimizers/optimizer_utils.py`
and operates only on:

- configured learning rates
- the number of text encoders

That is still a reasonable shared location. The strategy-owned part is mainly:

- how many text encoders the family has
- how they are handled once train flags are resolved

`sd3` already introduces a more semantic split:

- CLIP encoders
- T5 encoder

The current helper is still usable because it only resolves index-aligned train
flags. If a future family needs more than that, the expansion point should be
around family-owned handling semantics rather than moving all train-flag logic
into strategies by default.

## Things that should stay model-specific for now

The current SD3 port adds several helpers that should not be promoted into
base contracts yet:

- cached SD3 dropout behavior for TE outputs
- SD3 CLIP/T5 concatenation helpers
- SD3 flow-matching timestep density helpers
- SD3 direct sampling loop helpers

Those are real responsibilities, but there is not enough evidence yet that they
are stable cross-model contracts.

## New pressure visible after the first SD3 pass

### 6. SD3 text-conditioning payload shape is still awkward

Status: `partially addressed`

The first SD3 pass now makes one remaining design pressure more obvious than it
was when this audit started:

- SD3 text-conditioning still travels through several positional list shapes
  rather than one typed payload
- cached token-id paths reconstruct attention masks later instead of carrying a
  clearer conditioning object through the live/cache boundary
- the current seam works, but it is easier to misuse than the surrounding SD /
  SDXL strategy code

This does not mean SD3 needs a new universal base contract yet. It means SD3
likely wants a clearer local conditioning bundle before any broader
cross-family conditioning abstraction is attempted.

This is now partially addressed in code:

- SD3 has local named payloads for token ids/masks and encoded text
  conditioning
- SD3 strategy internals no longer need to pass positional six-item lists
  across tokenization / encoding / diffusion / denoiser / sampling seams
- the cache/dataloader dict boundary remains unchanged, so this is still a
  local SD3 cleanup rather than a new base conditioning contract

### 7. Conditioning is now a real strategy concern with an initial shared facet

Status: `partially addressed in code`

This pressure scan originally treated conditioning as an open architectural
question. After comparing SD / SDXL / SD3 with Flux / Anima reference code,
the higher-level responsibility now looks stable enough to justify a shared
strategy seam:

- each family resolves denoiser-facing conditioning for a batch
- each family may combine cached outputs, cached token ids, and live encoding
- each family still keeps its concrete payload shape local

This is now reflected in code:

- `ConditioningStrategy.resolve_conditioning(...)` exists on the required
  strategy contract
- `sd/conditioning.py`, `sdxl/conditioning.py`, and `sd3/conditioning.py`
  now own the family cache/live/merge conditioning flow
- diffusion and validation use that seam instead of private `_get_text_conds()`
  helpers

What is still open is everything below that top-level responsibility:

- whether any shared sub-conventions should emerge under
  `resolve_conditioning(...)`
- whether conditioning payload types should eventually converge more than they
  do today
- how future model families like Flux / Anima fit the same seam in practice

## Recommended direction after this audit

### Keep as stable trainer/runtime contracts

- model loading
- caching backend creation
- sample generation
- checkpoint metadata
- validation entrypoint
- `process_batch`
- `call_denoiser`

### Keep optional where they already make sense

- full-model checkpoint save path
- prompt-weight support

### Reclassify rather than generalize immediately

- `tokenize`
- `encode_tokens`
- `get_models_for_text_encoding`

These are important seams, but they are not yet clearly trainer contracts.

## Concrete follow-up questions for the next pass

1. Is the current `get_models_for_text_encoding()` seam sufficient as-is once
   more families are present?
2. How should the single `TrainingStrategy` interface make required versus
   optional capabilities more obvious without loosening the model strategies?
3. Which optional capabilities besides prompt weighting are already visible
   across the first three model families?
4. At what point would per-text-encoder train-flag resolution need something
   richer than index-aligned shared logic?
5. What, if anything, should become a shared sub-convention under the new
   `ConditioningStrategy.resolve_conditioning(...)` seam as more model
   families are ported?
