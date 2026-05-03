# Repo-Owned PEFT Method Notes

This is a working scratchpad for the desired shape/requirements for PEFT
method implementations in this repo. Update it along the way so the method
layer can settle into a better system instead of a pile of one-off adapters.

## Current Shape

- PEFT-family config interpretation belongs in the PEFT layer, not in the
  adapter root. That includes active method branch selection, legacy PEFT
  normalization, and translation into the normalized runtime spec consumed by
  the training/adapter upper layers.
- Method config ownership should live beside the method runtime through an
  `AdapterMethodConfigBinding`, not in a parallel central map.
- Method packages currently own four local concerns: config translation,
  per-target module behavior, runtime orchestration over resolved targets, and
  state-dict load/save helpers.
- Runtimes should consume optimization-owned resolved targets and attach
  repo-owned trainable parameter refs. Method internals should not invent a
  second target vocabulary.
- Method modules should expose export/load helpers that operate on method-local
  checkpoint keys. The runtime owns file format concerns and per-target
  missing-key reporting.

## Layering Notes

- The adapter upper layer should not need to know PEFT branch names or PEFT
  config dataclass layout details.
- The PEFT family layer can resolve "which method is active?" because that is
  family semantics, not generic adapter semantics.
- Individual method packages should keep owning method-local validation,
  runtime settings translation, module construction, and state-dict behavior.
- If a helper starts needing knowledge of more than one method, first ask
  whether it belongs at the PEFT family layer before adding another adapter
  root abstraction.

## LoHa / LoKr / LoCon / OFT / BOFT / ABBA / TLora Lessons

- LoHa, LoKr, and LoCon now share enough runtime/state-dict ceremony that a
  small PEFT-family helper for module naming, supported-target filtering,
  trainable-ref attachment, save/load loops, and loaded-runtime merge wiring is
  now a real follow-up candidate rather than a premature abstraction.
- Validation belongs in method config translation first, with direct module
  construction repeating critical invariants. This caught/standardized missing
  rank, unsupported plain dropout, and bypass/DoRA conflicts before runtime
  surprises.
- LoKr shows that method modules need room for algorithm-specific layout
  decisions (`full_matrix`, `decompose_both`, `factor`, Tucker only on one
  Kronecker factor). Avoid forcing all methods into a LoRA-shaped rank/down/up
  abstraction too early.
- The current module classes duplicate target-module introspection and DoRA
  merge math. That may become a shared mixin/helper later, but only after we
  know whether future methods need exactly the same behavior. The general aim
  is deciding after all lycoris methods are in + one unrelated hf peft one.
- DyLoRA is a useful counterexample for method shape: some PEFT methods are
  better modeled as a training policy layered on top of a familiar weight
  parameterization than as a brand-new factorization family. Keep the public
  config surface focused on the extra policy knobs that actually matter.
- For new repo-owned PEFT methods, especially in this early adapter-system
  phase, there is currently little reason to let hypothetical backward
  compatibility constrain the core training/runtime design. Pick the best
  native representation first; if older layout import/export support is needed
  later, prefer explicit compatibility helpers outside the main training path.
- ABBA is a good reminder that "clever" factorization caches need explicit
  invalidation or they quietly become stale training bugs. If a method keeps
  an optimized derived view of trainable weights, either rebuild it per use or
  make the invalidation story explicit and test it directly.
- LoRA is the reminder that "legacy but still active" deserves the same
  architectural cleanup as new method intake. Keeping the most common method on
  a compatibility wrapper quietly leaks old target-discovery, naming, and
  persistence assumptions into the rest of the adapter system.
- Scalar export should stay numerically boring. Folding an unconstrained
  trainable scalar into one exported factor is safer than relying on a
  symmetric `sqrt(scalar)` bake that can go invalid once training drives the
  scalar negative.
- Method-local runtime state should stay explicit and scoped. TLora is a good
  reminder that if a method needs per-forward state such as timestep-aware
  masks, absorbing the method alone does not automatically justify reaching
  into strategies or shared runtime layers. The current repo-owned TLora slice
  intentionally stops at the method package and leaves the vendor-described
  timestep-mask plumbing as documented follow-up work instead of widening the
  architecture boundary implicitly.

## Current Coverage

From vendor/lycoris so far added including checking Hugging Face PEFT for
improvements:

### lycoris

- loha: vendor lycoris/hf peft ✅
- locon: vendor lycoris ✅
- lokr: vendor lycoris/hf peft ✅
- oft: vendor lycoris/hf peft ✅
- boft: vendor lycoris/hf peft ✅
- dylora: vendor lycoris/hf peft ✅
- glora: vendor lycoris/hf peft ✅
- ia3: vendor lycoris ✅
- abba: vendor lycoris/hf peft ✅
- tlora: vendor lycoris ✅
- norms (not really a method, beads plan) ❌

### hf/peft

- shira ❌
- adalora ❌
- adamss ❌
- beft ❌
- c3a ❌
- cartridge ❌
- cpt ❌
- delora ❌
- fourierft ❌
- gralora ❌
- hra ❌
- lily ❌
- miss ❌
- osf ❌
- peanut ❌
- poly ❌
- psoft ❌
- randlora ❌
- road ❌
- vblora ❌
- vera ❌
- waveft ❌
- xlora ❌
- tinylora ❌

### in repo rework

- lora: repo-owned refresh/hf peft ✅
