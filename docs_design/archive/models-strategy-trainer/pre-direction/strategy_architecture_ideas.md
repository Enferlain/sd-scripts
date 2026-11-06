# ad hoc notes for future improvements and shapes for the strategy encountered during work on the layer

## Strategy context / denoiser seam

- The denoiser seam feels stronger than it first appears. It is not just "the
  place TLora needs timesteps"; it is one of the few places where many
  strategy-owned runtime facts naturally converge without feeling artificial.
  That makes it a good anchor for future cleanup.
- Denoiser forward currently looks like a natural "execution seam" inside the
  strategy layer. Diffusion code wants to assemble the forward, while denoiser
  code wants to execute it. That split could become more explicit later.
- The explicit-signature version of `call_denoiser(...)` feels more honest than
  a hidden request/helper layer in the current architecture. If a future
  dataclass or request object appears again, it should happen because it makes
  the code read better, makes the boundary feel more natural, and fits the
  surrounding architecture, not just because long signatures look unpleasant
  on their own.
- Family-owned context publication in `sd/denoiser.py`, `sdxl/denoiser.py`,
  and `sd3/denoiser.py` is repetitive, but the repetition is not bad yet. It
  may be better to let another model land first, then extract only what is
  provably identical.

## Architecture boundaries

- `contracts.py` being definition-oriented turned out to be a useful
  constraint. It suggests the strategy layer may benefit from stronger
  file-role boundaries in general: contract definitions, family execution, and
  shared infrastructure each staying visually distinct.
- One useful future rule might be: shared base files should only contain
  concepts the whole layer has already agreed are real. Experiments and local
  shapes should stay closer to the family code until they have actually proven
  themselves.
- More broadly, future strategy work may benefit more from making existing
  seams feel intentional than from adding new abstraction layers. The current
  layer is already much better than it used to be, and a lot of future polish
  may come from tightening what exists.

## Testing / ergonomics

- The recent test adjustments were revealing: tests that override
  `call_denoiser(...)` too aggressively stop matching the architecture once the
  denoiser method owns real behavior. Over time, the strategy layer may want
  more black-box seam tests and fewer tests that replace the seam completely.
- If a strategy seam is important enough to publish context or own execution
  semantics, test helpers should probably observe it through fake downstream
  components rather than by monkeypatching the strategy method away.

## Producer / receiver framing

- The adapter -> PEFT -> TLora work made the producer/receiver boundary feel
  more real. A good guiding principle may be: higher layers stay explicit about
  their own orchestration, while lower layers get stable read access to
  strategy-owned runtime facts when those facts are genuinely in scope.
- `StrategyContext` feels most justified when it prevents new parameter
  threading into lower library layers. It feels less justified when it is used
  as a shortcut between higher-level orchestration components.

## Broader future possibilities

- The strategy layer may eventually want a clearer vocabulary around
  "assembly", "execution", and "publication" rather than only "contracts" and
  "concerns". Those ideas already exist in the code implicitly, even if the
  layer does not name them yet.
- A future redesign could still keep the broader strategy idea while changing
  the implementation shape significantly. The current architecture should not
  be treated as the maximum form of the concept, only the current best step.
