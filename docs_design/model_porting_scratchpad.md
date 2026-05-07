# Model Porting Scratchpad

This is a working scratchpad for porting and adapting new model families into
the current repo architecture.

It is intentionally not a finalized guide. The goal is to preserve useful
observations between sessions until there is enough evidence to turn them into
a real skill, checklist, or design note.

## Current stance

- Prefer one more manual model-family port before freezing a formal
  porting skill.
- Record repeated patterns and friction points here as they appear.
- Do not treat notes here as hard rules unless they are later promoted into a
  more formal design document.

## Current model-porting protocol

- `library/models/<family>/` owns component code:
  - module definitions
  - checkpoint conversion
  - component loading/bootstrap
  - low-level runtime/component helpers
- `library/strategies/<family>/` owns model-specific training-facing behavior:
  - tokenization / encoding behavior
  - caching behavior
  - diffusion / denoiser orchestration
  - validation
  - sample generation
  - checkpoint metadata and family-specific full-model save behavior
- `library/strategies/shared/` is for cross-family training-facing behavior
  that has proven to be genuinely shared.
- `library/strategies/base/contracts.py` is for required strategy-facing
  contracts.
- `library/strategies/base/features.py` is for optional strategy features.

## Things recently clarified

### Tokenizer split

- Tokenizer/component bootstrap can live in a model-family home when it is a
  component concern reused by later families.
- Shared CLIP-family behavior belongs under `library/strategies/shared/`.
- Current example:
  - `library/models/sd/tokenizer.py`
  - `library/strategies/shared/clip/tokenization.py`

### Optional feature split

- Weighted prompt support is strategy code, but not a required universal
  strategy contract.
- Current example:
  - `library/strategies/base/features.py::WeightedPromptStrategy`

## What to watch for during future ports

### 1. Ported code that keeps upstream payload shapes

This is currently visible in the first SD3 pass:

- local strategy code still uses positional tensor lists in places where the
  repo prefers named payloads or clearer boundary objects
- this is adaptation debt, not necessarily a sign that the repo architecture
  is wrong

Question to ask:

- is this a global abstraction problem, or just a local port normalization
  problem?

### 2. Shared concern vs fake generality

Do not share code only because multiple families use the same library or
 upstream component type.

Instead ask:

- does the code change for the same reasons?
- do the consumers reuse the whole thing together?
- does extracting it remove duplication without introducing family branches?

### 3. Mode concern vs strategy concern

A seam may be real without being universal:

- strategy concern: family-specific behavior needed by the training system
- mode concern: lifecycle and save logic owned by a training mode

Example to remember:

- weighted prompts are an optional strategy feature
- PEFT adapter saving is mode-owned
- fine-tune full-model saving is strategy-owned but mode-scoped

## Good follow-up candidates after the next model port

- turn repeated porting observations into a real porting skill/checklist
- document recurring "ported shape -> repo shape" normalization steps
- re-evaluate whether any current family-owned helpers want a more explicit
  shared home
