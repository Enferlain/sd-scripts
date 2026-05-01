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

## LoHa / LoKr / LoCon / OFT Lessons

- LoHa, LoKr, and LoCon now share enough runtime/state-dict ceremony that a
  small PEFT-family helper for module naming, supported-target filtering,
  trainable-ref attachment, save/load loops, and loaded-runtime merge wiring is
  now a real follow-up candidate rather than a premature abstraction.
- Validation belongs in method config translation first, with direct module
  construction repeating critical invariants. This caught/standardized missing
  rank, unsupported plain dropout, and bypass/DoRA conflicts before runtime
  surprises.
- Plain `dropout` remains intentionally unsupported for LoHa/LoKr because the
  absorbed LyCORIS behavior did not apply it consistently in rebuilt-weight
  mode. Keep `rank_dropout` and `module_dropout` as the supported knobs unless
  a future method-specific design gives plain dropout clear semantics.
- LoCon is the counterexample that makes the dropout rule method-local rather
  than family-global: its absorbed behavior still uses plain `dropout`, but
  only in bypass-mode output application and the DoRA input path. That should
  stay documented as an explicit method semantic instead of being normalized
  away into a fake family-wide rule.
- Scalar-mode export folds `scalar` into the first exported factor and resets
  scalar to identity on load. That convention should be documented as an
  export-format rule if it survives more methods.
- LoKr shows that method modules need room for algorithm-specific layout
  decisions (`full_matrix`, `decompose_both`, `factor`, Tucker only on one
  Kronecker factor). Avoid forcing all methods into a LoRA-shaped rank/down/up
  abstraction too early.
- LoKr full-matrix mode is not just a layout toggle: it uses unit scaling by
  overriding alpha to rank, matching the absorbed LyCORIS intent that fully
  materialized Kronecker factors should not get an extra rank scale.
- The current module classes duplicate target-module introspection and DoRA
  merge math. That may become a shared mixin/helper later, but only after we
  know whether future methods need exactly the same behavior.
- OFT is the clearest example so far that not every absorbed method wants a
  fake LoRA-shaped `rank + alpha` surface. The repo-owned OFT path uses
  method-local `factor` and `constraint` names because those are the real
  algorithm knobs in the LyCORIS-derived implementation.
- Repo-owned OFT intentionally absorbs the LyCORIS Diag-OFT direction, not the
  older built-in `oft_deprecated` path. Treat `oft_deprecated` as dead code
  waiting for deletion, not as a compatibility contract to preserve.
- The absorbed OFT bypass path needed real fixes before it was safe to own:
  the vendor/legacy logic had a broken diff branch and a rescale shape bug.
  Keep OFT's bypass/dropout semantics documented as method-local behavior
  instead of treating them as a PEFT-family-wide abstraction too early.
- OFT plain `dropout` should stay a block-transform concern. Reapplying a
  second output-shaped mask in bypass mode just recreates the broken vendor
  broadcast path and makes the semantics harder to reason about.

## Notes

From vendor/lycoris so far added including checking huggingface peft for improvements:

### lycoris:

- loha: vendor lycoris/hf peft ✅
- locon: vendor lycoris ✅
- lokr: vendor lycoris/hf peft✅
- oft: vendor lycoris/hf peft ✅
- boft: vendor lycoris/hf peft ❌
- dylora: vendor lycoris/hf peft ❌
- glora: vendor lycoris/hf peft ❌
- ia3: vendor lycoris/hf peft ❌
- abba: vendor lycoris/hf peft ❌
- tlora: vendor lycoris/hf peft ❌
- norms (not really a method) ❌

### hf/peft:

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

- lora: legacy version/hf peft ❌ (for last)
