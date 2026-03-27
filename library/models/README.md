# Model Layer Notes

This note records the current design direction for `library/models/` so we do
not have to rediscover the same boundaries during every model-family port.

## Core Distinction

There are two different kinds of code that often get mixed together:

- **Model component code**
  Code about what a component is and how it works in general.
  Examples:
  - module/class definitions
  - forward-pass internals
  - checkpoint conversion
  - component loading/saving
  - low-level component quirks that are not specific to one training workflow

- **Model behavior code**
  Code about how this repo uses a model family.
  Examples:
  - tokenization rules
  - prompt weighting
  - text-encoding assembly
  - chunk stitching
  - conditioning construction
  - caching behavior/format
  - model-family denoiser call shaping

Rule of thumb:

- `library/models/` owns **model component code**
- `library/strategies/` owns **model behavior code**

This distinction is about **component semantics vs family workflow semantics**,
not about "training vs inference". Inference pipelines may still use
model-family behavior helpers if that is the right ownership.

## Current Findings

- `vae.py` is acceptable as model-layer code even if it contains several VAE
  concerns together, because they still belong to one coherent component domain.
- `tokenizer.py` is a stronger sign of misplaced ownership when it mixes generic
  tokenizer loading, prompt parsing, and model-family tokenization behavior.
- For the current SD / SDXL CLIP usage, repo-owned logic around tokenization,
  prompt weighting, chunk stitching, pooling, and text-encoder output shaping
  should be treated as **model behavior code** unless investigation shows that a
  helper is truly component-level.
- The current SD / SDXL cleanup now treats CLIP-family tokenization and
  text-encoding helpers as strategy-owned behavior. Where ownership between
  families is still awkward, prefer local duplication over inventing a new
  shared/base bucket without a clear design reason.
- `text_encoder.py` and `tokenizer.py` therefore need to be judged by contents,
  not by name alone. The key question is whether a helper is a low-level
  component interaction detail or a model-family encoding/tokenization behavior
  detail.

## Reuse Rules

- Do not reimplement identical code just because a new model family needs an
  existing component.
- Default ownership lives in the first model-family folder that introduces a
  component implementation.
- Reuse alone does not make a component "generic". Do not promote components to
  top-level neutral homes just because another model family might use them.
- If a model family genuinely reuses an existing component implementation, keep
  one implementation and reuse it from the owning component home instead of
  cloning it into the new family folder.
- Name reused components according to what they are, but do not force premature
  extraction from the original owning model family.
- If ownership is unclear, investigate actual semantics first and record the
  conclusion before moving files.

## Practical Guidance For New Model Families

- Start by identifying which files are true component code and which are
  behavior/orchestration code.
- Port component code into `library/models/<family>/` only when that component
  is actually owned by the new family.
- Put tokenization, encoding, conditioning, caching, and similar workflow logic
  in `library/strategies/<family>/` unless investigation shows that a helper is
  truly component-level.
- Do not move code into `models/` just to make strategy files thinner. Thinner
  files are not a sufficient ownership argument by themselves.
- When a current placement feels wrong, write down the reasoning here before
  repeating the same argument in a future port.
