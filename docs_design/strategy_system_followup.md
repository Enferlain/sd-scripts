# Strategy System Follow-Up

> [!NOTE]
> This document spans multiple refactor stages. References to
> `library/strategies/base/training.py` and `CacheHandler` should be read as
> today’s `library/strategies/base/contracts.py` and `CacheBackend` unless the
> surrounding text is explicitly discussing the older naming state.

## Purpose

Record the current strategy-layer direction after the Trainer/TrainingMode
refactor and the recent tokenization/encoding cleanup.

This note is intentionally short. It is meant to preserve the current
decisions and next steps, not the full history of every design discussion.

## Current Direction

The architecture we are aiming for is:

- scripts choose config + strategy + mode
- `Trainer` owns shared orchestration
- `TrainingMode` owns mode-specific divergence
- `TrainingStrategy` owns model-family behavior

The important strategy-layer rule is:

- `library/strategies/base/contracts.py` is the source of truth for the
  model-family training contract

That means:

- downstream model families should read `base/contracts.py` first
- separate concern files are still fine, but contract definitions should not
  be mixed with helper code, singleton plumbing, or legacy support unless
  there is a strong reason

## Design Rules

### 1. One runtime strategy object per model family

Keep:

- `SdTrainingStrategy`
- `SdxlTrainingStrategy`
- future equivalents

Do not move toward a runtime where `Trainer` juggles separate tokenization,
encoding, caching, checkpointing, or sampling strategy objects.

Concern-specific implementation files are fine. Concern-specific runtime
objects owned directly by `Trainer` are not the goal.

If the rule is “the runner talks to TrainingStrategy,” then doing tokenization/text-encoding explicitly while caching, validation, sampling, and batch processing go through TrainingStrategy is inconsistent. That usually means we are preserving transition structure longer than necessary.

### 2. Expand for implementation, collapse for runtime

The intended shape is:

- contract is organized by concern in `base/training.py`
- model-family implementations can be split across
  `strategies/<model>/tokenization.py`, `encoding.py`, `caching.py`, etc.
- `Trainer` still only talks to one model-family strategy object

This keeps model-family code readable without complicating the runtime model.

### 3. Contract code belongs in `base/training.py`

The training-facing contract should live in `base/training.py`.

If a concern contract is ever split further, it should move into a
contract-only file. It should not be mixed into a file that also carries:

- reusable helpers
- singleton accessors
- current SD / SDXL implementation details
- legacy-only behavior

### 4. `library/models/` owns component-coupled behavior

Use `library/models/` for logic that is really about model components, even if
the file is helper-oriented rather than defining the raw component class.

Examples:

- `library/models/sd/text_encoder.py`
- `library/models/sdxl/text_encoder.py`
- `library/models/sd/tokenizer.py`

This matches the repo's existing component-centric convention:

- `unet.py`
- `vae.py`
- `text_encoder.py`

The same principle should apply to tokenizer-component behavior.

### 5. Shared by accident is not generic

If SD and SDXL both happen to use the same CLIP-family behavior today, that
does not automatically make it good `base/` code for future models.

Keep `base/` for:

- real contract definitions
- minimal genuinely generic helpers

Move current-family behavior toward model-owned code when it is tightly coupled
to the tokenizer, text encoder, VAE, or UNet.

## What Is Already Done

### Shared-phase leaks removed

The shared phase layer no longer hardcodes SDXL token/TE shape assumptions.

Done:

- epoch token caching uses strategy-provided encoder names
- TE disk caching uses a strategy-provided model bundle
- active caching goes through `CachingEngine` + `CacheHandler`, with the
  training-side caching contract in `base/training.py` supplying the
  model-family handlers

### `base/training.py` is the canonical contract home

The recent cleanup moved runtime tokenization/text-encoding contracts into
`base/training.py`.

Current shape:

- `TokenizationStrategy` lives in `base/training.py`
- `TextEncodingStrategy` lives in `base/training.py`
- `TrainingStrategy` owns provider/wiring methods like:
  - `get_tokenize_strategy()`
  - `get_tokenizers()`
  - `tokenize_captions()`
  - `get_text_encoding_strategy()`
  - `get_models_for_text_encoding()`
  - `encode_te_outputs_in_memory()`

The old `base/tokenization.py` and `base/encoding.py` compatibility modules
have been removed.

The active caching contract now also lives here. In practice that means:

- `CachingStrategy` in `base/training.py` is the model-family training-facing
  contract
- `create_latent_caching_strategy()` and `create_te_caching_strategy()` supply
  `CacheHandler` instances for `CachingEngine`
- `get_token_cache_encoder_names()` and `build_te_cache_model_bundle()` keep
  cache-shape and model-packing details out of shared phase code

### Tokenization behavior moved out of `base/`

Current CLIP-family token/chunk/loading behavior has moved into:

- `library/models/sd/tokenizer.py`

SDXL reuses that shared tokenizer-component helper directly, matching the same
"one canonical implementation unless behavior really differs" pattern already
used for VAE code.

The remaining SDXL-specific tokenization concerns still live in SDXL strategy
code, but the token/chunk construction path now routes through the shared
CLIP-family helper instead of keeping a second implementation in
`sdxl/training.py`.

### Generic training mechanics already moved out

Clearly generic shared mechanics have already been extracted from
`base/training.py` into shared utility homes, including:

- scheduler creation
- loss post-processing assembly
- validation RNG save/restore
- all-reduce helpers

That part of the cleanup is not the current problem anymore.

## What Still Matters

### 1. Active vs transitional vs legacy surface is still mixed ✅ Resolved

`base/caching.py` has been deleted. Legacy caching classes
(`LatentsCachingStrategy`, `TextEncoderOutputsCachingStrategy`,
`SdSdxlLatentsCachingStrategy`, `SdxlTextEncoderOutputsCachingStrategy`) and
their singleton `set_strategy`/`get_strategy` methods have been removed.
Legacy factory methods (`get_latents_caching_strategy`,
`get_text_encoder_outputs_caching_strategy`) have been removed from SD/SDXL
strategies.

The active contract in `base/training.py` no longer mixes active hooks with
legacy caching or singleton plumbing. Deprecated `dataset.py` calls are
no-ops.

### 2. `base/caching.py` is not future-facing contract code ✅ Resolved

`base/caching.py` has been deleted entirely. The active caching contract
(`CachingStrategy` facet with `create_latent_caching_strategy()` and
`create_te_caching_strategy()`) lives in `base/training.py` where it belongs.

### 3. Active singleton strategy dependence is now legacy-only ✅ Resolved

Singleton `set_strategy`/`get_strategy` class methods have been removed from
`TokenizationStrategy` and `TextEncodingStrategy`. The `_strategy` class
variable and related tests are gone.

What remains is limited to deprecated code paths:

- `set_current_strategies()` in deprecated `dataset.py` is a no-op
- deprecated scripts no longer import deleted modules

These will disappear when deprecated scripts are deleted.

### 4. Some generic-base defaults are still CLIP-specific ✅ Resolved

`prepare_text_encoder_grad_ckpt_workaround` and `prepare_text_encoder_fp8`
have been moved out of the generic `ModelPreparationStrategy` base. The base
now raises `NotImplementedError`; SD and SDXL strategies provide CLIP-specific
overrides.

No remaining methods in `base/training.py` access CLIP-specific attribute
paths.

### 5. Shared helpers are still shaped around current model families

Some shared helpers, especially sample generation, still read as SD/SDXL-first
rather than fully model-agnostic infrastructure.

That is acceptable for now. It should only be generalized when:

- a real third model family needs the same path
- the strategy boundary becomes clearer by doing so

The repo should avoid speculative abstraction here.

## Practical Ownership Guide

Use this when deciding where new or existing code should live.

### `library/strategies/base/training.py`

Put here:

- what a model-family training strategy must provide
- concern-organized training-facing ABCs
- minimal shared defaults that are truly generic

Do not put here:

- model-component-specific helper logic
- convenience helpers that are only shared by current models
- old-pipeline support code unless it is clearly marked transitional

### `library/strategies/<model>/...`

Put here:

- model-family strategy implementation
- thin adapters/wiring around tokenizer, text-encoder, caching, sampling, etc.
- concern-specific implementation files for readability

This layer answers:

- how does this model family fulfill the training contract?

### `library/models/<model>/...`

Put here:

- model-component behavior
- tensor transforms tied to a specific component family
- tokenizer/text-encoder/VAE/UNet helper logic

This layer answers:

- how does this model component behave?

### Shared generic utility modules

Put here:

- shared orchestration mechanics
- generic training utilities
- loss helpers
- generic runtime utilities

This layer answers:

- what is model-agnostic training infrastructure?

## Next Steps

Items 1–4 are resolved. The sub-strategy architecture is gone, and active
strategies are now born ready. Remaining:

### 1. Revisit shared helpers like sample generation

Only when there is real pressure from another model family or a clearer
generic boundary.

### 2. Clean up `base/training.py` organization and surface polish

The main remaining active work is in the contract home itself:

- improve sectioning/readability in `base/training.py`
- keep sorting active-path logic between contract, concrete strategy wiring, and
  model-layer helpers
- avoid reopening architecture questions that are already settled

### 3. Cosmetic follow-up: concrete strategy dataclass cleanup

`SdTrainingStrategy` and `SdxlTrainingStrategy` currently still carry the
`@dataclass` decorator even though they provide an explicit `__init__(cfg)`.
That is not harmful, but it is slightly misleading because the dataclass no
longer provides the constructor shape. This is worth cleaning up later, but it
is not an architectural issue.

The key principle for future work is:

- simplify the contract
- move component behavior to `library/models/`
- keep one runtime strategy object per model family

## Non-Goals

- no runtime sub-strategy architecture
- no new runner architecture
- no mode-layer redesign
- no preserving deprecated paths as first-class parallel systems
- no spending active refactor effort on `_deprecated` modules or `copy` scripts;
  those are deletion/reference material unless explicitly requested
