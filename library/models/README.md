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

## Loaded-Component Contract

`library/models/` now also owns the repo-level contract for **top-level loaded
model components**.

- Shared contract helpers live in `library/models/components.py`.
- `library/models/__init__.py` should stay a thin public re-export surface, not
  grow back into the implementation home.
- Each active model family declares `LOADED_MODEL_COMPONENT_SPECS` in its
  package `__init__.py`, in **family-owned order**.
- The family declaration is where we say what the top-level components are; the
  shared helpers are where generic code resolves and queries them.

At the moment that contract is centered on:

- `LoadedModelComponentSpec`
  Family-declared top-level component metadata (`key`, `public_name`, roles,
  capabilities).
- `LoadedModelComponent`
  The live loaded top-level component bound to one declaration.
- `build_loaded_components(...)`
  The helper that binds loaded modules to the family declarations in declared
  order.

### Ownership Rules For This Contract

- Family package `__init__.py` files may own **top-level component
  declarations** such as `LOADED_MODEL_COMPONENT_SPECS`.
- Shared resolution, filtering, and projection helpers belong in
  `library/models/components.py`, not in a family `__init__.py` and not in an
  unrelated utility or tool file.
- The loaded-component collection remains model-owned source truth. Reusable
  conversion of that explicit runtime view into accepted model-realization
  facts belongs to the central `library/metadata/builders/model.py` layer;
  model code does not grow a generic local metadata assembly module for it.
- Do not put trainer/runtime workflow logic into `library/models/components.py`.
  It owns the component contract surface, not training orchestration.
- Do not bury shared component metadata under a user-conditional tool like
  `parameter_dump.py`.

### Behavior Rules For Generic Callers

- Generic code should consume component **roles/capabilities**, not hardcoded
  family-native names such as `clip_l` or `mmdit`.
- Family-native names are fine for declarations and human-facing labels, but
  they are not the generic behavior contract.
- Family-declared order is the default top-level order for diagnostics,
  reporting, and tooling unless a consumer has an explicit reason to override
  it.
- Top-level components are the highest generic control surface here. Lower
  module/parameter targeting should expand from them rather than replace them.

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
- Decide the family's top-level loaded components early and declare them in
  `library/models/<family>/__init__.py` through `LOADED_MODEL_COMPONENT_SPECS`.
- Keep those declarations small and declarative. The family `__init__.py`
  should publish the component surface, not absorb shared resolution logic.
- Port component code into `library/models/<family>/` only when that component
  is actually owned by the new family.
- Put tokenization, encoding, conditioning, caching, and similar workflow logic
  in `library/strategies/<family>/` unless investigation shows that a helper is
  truly component-level.
- Do not move code into `models/` just to make strategy files thinner. Thinner
  files are not a sufficient ownership argument by themselves.
- If generic runtime code needs to reason about the family, prefer adding or
  refining component roles/capabilities over adding new family-name branches.
- When a current placement feels wrong, write down the reasoning here before
  repeating the same argument in a future port.

## Useful External Examples

- [`Tencent-Hunyuan/HunyuanImage-2.1`](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1/blob/0c44d1eda754edd42e69bde984251988fee7a4dc/hyimage/models/text_encoder/__init__.py) has examples of model-side component
  ownership that are useful to keep in mind when judging future text encoder /
  VAE code:
  - `hyimage/models/text_encoder/__init__.py`
  - `hyimage/models/vae/__init__.py`
- These are good reference points for the case where the repo is not just
  calling an external encoder/model, but is actually defining a reusable
  component wrapper/module around it.
- That is different from the current SD / SDXL CLIP helpers in this repo,
  which are mostly about tokenization / encoding behavior on top of
  `transformers` models and therefore fit the strategy layer better.
