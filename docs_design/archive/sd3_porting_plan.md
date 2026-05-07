# SD3 Porting Plan

## Goal

Port Stable Diffusion 3 into the current architecture without reintroducing the old script-centric layout.

The target runtime shape is:

- `library/models/sd3`: model definitions, checkpoint conversion, and component loading helpers
- `library/strategies/sd3`: trainer-facing SD3 behavior implemented against `library/strategies/base/contracts.py`

This document only covers the first task:

- what belongs in `models`
- what belongs in `strategies`
- how upstream SD3 source files map into the new layout

The next task, after the initial split is in place, is a separate pass:

- audit shared assumptions and conventions that SD3 may violate

Porting priority for this work:

- preserve needed SD3 functionality first
- break functionality down into repo conventions second
- generalize only after the SD3 behavior is visible and working

It is acceptable to use temporary SD3-specific seams during the port if that prevents functionality loss.
That includes a temporary SD3 config surface when needed.
The target is still shared config where it makes sense, but not at the cost of missing behavior during the first pass.

## Source Material

Local upstream reference files:

- `upstream_reference/sd3/library/sd3_models.py`
- `upstream_reference/sd3/library/sd3_utils.py`
- `upstream_reference/sd3/library/strategy_sd3.py`
- `upstream_reference/sd3/library/sd3_train_utils.py`
- `upstream_reference/sd3/sd3_train.py`
- `upstream_reference/sd3/sd3_train_network.py`
- `upstream_reference/sd3/sd3_minimal_inference.py`

These are reference-only staging files. Production code should not import from them.

## Split Rule

Use the current strategy contracts as the boundary.

Code goes to `library/models/sd3` when it:

- defines PyTorch modules
- defines model parameter/config structures
- converts or reconstructs checkpoint/state dict layouts
- loads model components from checkpoints or component paths
- can be reused outside the trainer without batch/caching/training knowledge

Code goes to `library/strategies/sd3` when it:

- implements one of the strategy facets from `contracts.py`
- decides how SD3 tokenization or encoding behaves during training
- owns cache formats or cache backend behavior
- computes diffusion training inputs, timesteps, sigmas, or loss weighting
- calls the denoiser with SD3-specific conditioning
- generates sample images
- decides save/checkpoint policy or metadata
- validates SD3-specific config/runtime assumptions

## Models Layout

Planned files under `library/models/sd3`:

- `__init__.py`
- `mmdit.py`
- `vae.py`
- `loader.py`
- `conversion.py`

### `mmdit.py`

Primary source:

- `upstream_reference/sd3/library/sd3_models.py`

Owns:

- `SD3Params`
- positional embedding helpers
- shared attention helpers used by MMDiT
- MMDiT block/module definitions
- `create_sd3_mmdit`

### `vae.py`

Primary source:

- `upstream_reference/sd3/library/sd3_models.py`

Owns:

- SD3 VAE building blocks
- `SDVAE`
- VAE encode/decode helpers that are part of the model implementation

### `loader.py`

Primary source:

- `upstream_reference/sd3/library/sd3_utils.py`

Owns:

- `analyze_state_dict_state`
- `load_mmdit`
- `load_clip_l`
- `load_clip_g`
- `load_t5xxl`
- `load_vae`

Notes:

- `loader.py` should build components and return them
- trainer decisions about when to load, wrap, cast, or offload stay in strategy code

### `conversion.py`

Primary sources:

- `upstream_reference/sd3/library/sd3_utils.py`
- `upstream_reference/sd3/library/sd3_train_utils.py`
- save/load behavior from `upstream_reference/sd3/sd3_train.py`
- save/load behavior from `upstream_reference/sd3/sd3_train_network.py`

Owns:

- full-checkpoint prefix splitting and reassembly
- SD3-specific state dict translation rules
- format-specific save helpers used by strategy checkpointing

Notes:

- this is the SD3 equivalent of the heavier checkpoint translation logic already living in `library/models/sdxl/conversion.py`
- checkpoint policy still belongs in strategy code even when conversion helpers live here

## Strategies Layout

Planned files under `library/strategies/sd3`:

- `__init__.py`
- `training.py`
- `loading.py`
- `tokenization.py`
- `encoding.py`
- `caching.py`
- `diffusion.py`
- `denoiser.py`
- `sampling.py`
- `checkpointing.py`
- `validation.py`
- `model_preparation.py`

### `training.py`

Owns:

- `Sd3TrainingStrategy`
- SD3 composition of all strategy facets
- SD3 family-level instance state

### `loading.py`

Primary sources:

- `upstream_reference/sd3/sd3_train.py`
- `upstream_reference/sd3/sd3_train_network.py`

Owns:

- `ModelLoadingStrategy` implementation for SD3
- trainer-facing orchestration around `library/models/sd3/loader.py`
- model version naming returned to the trainer
- any lazy-load integration if SD3 needs it

### `tokenization.py`

Primary source:

- `upstream_reference/sd3/library/strategy_sd3.py`

Owns:

- CLIP-L tokenizer setup
- CLIP-G tokenizer setup
- T5 tokenizer setup
- SD3 text token packing and returned token tensor ordering
- any token cache naming required by the shared pipeline

### `encoding.py`

Primary source:

- `upstream_reference/sd3/library/strategy_sd3.py`

Owns:

- SD3 text encoder execution
- CLIP-L / CLIP-G / T5 output combination
- attention mask handling
- encoder dropout behavior
- in-memory TE output computation
- encoded output layout consumed by diffusion and caching

Notes:

- SD3 does not currently need a separate `conditioning.py`
- unlike SDXL, the extra SD3 payload is still text-side output structure rather than separate image micro-conditioning metadata

### `caching.py`

Primary source:

- `upstream_reference/sd3/library/strategy_sd3.py`

Owns:

- latent cache backend
- text-encoder output cache backend
- SD3 cache file schema
- TE cache model bundle assembly
- token cache encoder names

### `diffusion.py`

Primary sources:

- `upstream_reference/sd3/library/sd3_train_utils.py`
- `upstream_reference/sd3/sd3_train.py`
- `upstream_reference/sd3/sd3_train_network.py`

Owns:

- `DiffusionTrainingStrategy` implementation
- `ModelSamplingDiscreteFlow`
- sigma helpers
- timestep sampling logic
- SD3 loss weighting logic
- noisy model input preparation
- assembly of SD3 text conditioning for the denoiser call
- batch processing for train and validation

Notes:

- although `ModelSamplingDiscreteFlow` sits in `sd3_utils.py` upstream, it is training/inference behavior, not a reusable model definition, so it belongs in strategy code here

### `denoiser.py`

Primary sources:

- `upstream_reference/sd3/sd3_train.py`
- `upstream_reference/sd3/sd3_train_network.py`

Owns:

- `DenoiserCallingStrategy` implementation for MMDiT
- mapping from strategy-produced text conditioning into `context` and `y`
- any mask-related handling needed at denoiser call time

### `sampling.py`

Primary sources:

- `upstream_reference/sd3/library/sd3_train_utils.py`
- `upstream_reference/sd3/sd3_minimal_inference.py`

Owns:

- sample prompt encoding flow
- SD3 inference denoising loop used for sample images
- VAE decode path used by trainer sampling

### `checkpointing.py`

Primary sources:

- `upstream_reference/sd3/library/sd3_train_utils.py`
- `upstream_reference/sd3/sd3_train.py`
- `upstream_reference/sd3/sd3_train_network.py`

Owns:

- `CheckpointingStrategy` implementation
- SD3 metadata fields
- strategy-facing save hooks
- calls into `library/models/sd3/conversion.py` for actual format-specific serialization

### `validation.py`

Owns:

- SD3-specific validation-loss behavior
- `ValidationStrategy.calculate_val_loss` implementation
- any SD3-specific eval-time batch handling needed to reuse the training path safely

Notes:

- this is for training-time validation behavior, not config validation
- do not move generic multi-encoder assumptions here if they should become shared rules
- this file should only contain logic that is truly SD3-specific

### `model_preparation.py`

Owns:

- text encoder grad-checkpoint setup for CLIP-L, CLIP-G, and T5
- dtype / casting decisions required by SD3 components
- any FP8 embedding workarounds specific to SD3 text encoders

## Upstream-to-Target Map

- `sd3_models.py` -> `models/sd3/mmdit.py`, `models/sd3/vae.py`
- `sd3_utils.py` -> `models/sd3/loader.py`, `models/sd3/conversion.py`
- `strategy_sd3.py` -> `strategies/sd3/tokenization.py`, `strategies/sd3/encoding.py`, `strategies/sd3/caching.py`
- `sd3_train_utils.py` -> `strategies/sd3/diffusion.py`, `strategies/sd3/sampling.py`, `strategies/sd3/checkpointing.py`
- `sd3_train.py` -> `strategies/sd3/loading.py`, `strategies/sd3/diffusion.py`, `strategies/sd3/denoiser.py`, `strategies/sd3/checkpointing.py`, `strategies/sd3/model_preparation.py`
- `sd3_train_network.py` -> `strategies/sd3/loading.py`, `strategies/sd3/diffusion.py`, `strategies/sd3/denoiser.py`, `strategies/sd3/checkpointing.py`
- `sd3_minimal_inference.py` -> `strategies/sd3/sampling.py`

## Things Not To Port Verbatim

These should be translated into the new architecture, not copied into runtime code as-is:

- argparse wiring
- script entrypoint control flow
- script-level save cadence decisions
- duplicated helper code that belongs in a shared layer instead

In particular:

- training argument registration should become config/dataclass work
- trainer lifecycle should stay in trainer/mode code
- model-family behavior should stay behind strategy contracts

Temporary exception during porting:

- if a piece of SD3 functionality cannot be represented cleanly in the current shared config yet, it is acceptable to stage it in an SD3-specific config path first
- model-specific config should be treated as temporary scaffolding unless it proves to be genuinely model-family-specific after the follow-up audit

## First Porting Order

1. `library/models/sd3/mmdit.py`
2. `library/models/sd3/vae.py`
3. `library/models/sd3/loader.py`
4. `library/models/sd3/conversion.py`
5. `library/strategies/sd3/tokenization.py`
6. `library/strategies/sd3/encoding.py`
7. `library/strategies/sd3/caching.py`
8. `library/strategies/sd3/loading.py`
9. `library/strategies/sd3/diffusion.py`
10. `library/strategies/sd3/denoiser.py`
11. `library/strategies/sd3/checkpointing.py`
12. `library/strategies/sd3/sampling.py`
13. `library/strategies/sd3/model_preparation.py`
14. `library/strategies/sd3/validation.py`
15. `library/strategies/sd3/training.py`

## Deferred Follow-Up

After the initial SD3 split exists, do a second pass for shared assumptions and conventions.

Expected audit areas:

- places assuming one or two text encoders
- cache naming assumptions
- text encoder metadata assumptions
- denoiser call signature assumptions
- checkpoint metadata assumptions
- config validation or help text that still only names SD/SDXL families

This second pass should happen after the SD3 code is visible in the repo, not before.
