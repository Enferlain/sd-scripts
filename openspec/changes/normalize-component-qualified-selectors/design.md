## Context

The repository now has a working model inspection tool that can load real model runtimes and dump parameters, modules, state, and a summary view. That work exposed a mismatch in the public selector surface: optimizer grouping matches against training-internal names such as `denoiser.*` and `text_encoder1.*`, while the inspection tool groups output under model-facing component names such as `unet`, `mmdit`, `clip_l`, `clip_g`, and `t5xxl`.

Those internal names are implementation details of the current fine-tune path, not durable model-facing identifiers. They do not match the model package metadata already declared via `NAMED_PARAMETER_COMPONENT_NAMES`, they do not compose well with regex matching, and they create a drift between “what the user sees in the dump” and “what the config matcher accepts”.

At the same time, `optimizer.learning_rates.blocks` is now obsolete. The direction for finer-grained targeting is named selectors over real loaded model parameters, with future work likely extending the same selector surface to module-oriented selection rather than reviving block-weight syntax.

## Goals / Non-Goals

**Goals:**
- Establish one canonical external selector namespace for parameter dumps and fine-grained optimizer matching.
- Base that namespace on existing model package component labels rather than training-internal placeholders.
- Make the inspection dump output and optimizer matcher speak the same selector language.
- Remove the obsolete `optimizer.learning_rates.blocks` config surface.
- Leave room for future module-selector work to reuse the same namespace.

**Non-Goals:**
- Introduce new dump-specific or selector-specific model APIs in `library/models/*`.
- Redesign the training-mode/runtime ownership of denoiser and text encoder bookkeeping.
- Implement the future non-parameter module selector feature in this change.
- Replace component-grouped output with checkpoint-format prefixes such as SGM serialization keys.

## Decisions

### Use `component.local_name` as the canonical external selector namespace

All user-facing selector strings SHALL be normalized to `component.local_name`, where `component` comes from the existing `NAMED_PARAMETER_COMPONENT_NAMES` metadata and `local_name` is the real runtime parameter path from the loaded component module.

Why:
- It matches how users reason about models: pick a component, then a path within it.
- It makes regex matching unambiguous across the whole model.
- It aligns the optimizer matcher with the inspection dump output.

Alternative considered:
- Keep bare local names in dumps and add a parallel `qualified_name` helper. Rejected because it preserves two competing selector surfaces and keeps ambiguity in the primary output.

### Keep training-internal labels internal

The fine-tune/runtime code may continue to use internal placeholders such as `denoiser` or text encoder lists for control flow, but those names SHALL not remain the public matcher namespace.

Why:
- They are runtime implementation details, not model-facing names.
- They do not generalize cleanly across families like SDXL and SD3.

Alternative considered:
- Alias `denoiser` to `unet`/`mmdit` in config only. Rejected because it preserves a split-brain namespace and does not solve text encoder naming cleanly.

### Remove `optimizer.learning_rates.blocks`

The obsolete block-level learning rate field SHALL be removed from config schema/defaults and any related handling.

Why:
- It no longer reflects the grouping direction of the repo.
- It competes with the more explicit named-selector approach.

Alternative considered:
- Keep it deprecated but present. Rejected because it keeps dead surface area in schema and examples and muddies the migration path.

### Reuse the same namespace for future module selectors

This change will not implement module selectors, but the design SHALL reserve `component.local_name` as the shared public selector namespace for both parameter and future module-oriented targeting.

Why:
- It prevents another naming migration later.
- It keeps dumps, parameter matching, and future module matching aligned.

## Risks / Trade-offs

- [Risk] Existing optimizer group configs using `denoiser.*` or `text_encoder1.*` will stop matching as written. → Mitigation: document the new namespace clearly, update tests/examples, and treat this as an intentional config-surface cleanup.
- [Risk] Some generic optimization code will need model-family-aware component labels at the boundary. → Mitigation: derive labels from existing `NAMED_PARAMETER_COMPONENT_NAMES` metadata rather than adding new model APIs.
- [Risk] Users may confuse component-qualified selector names with checkpoint serialization keys. → Mitigation: document that these are runtime selector names, not SGM/Diffusers save-format keys.
- [Risk] Removing `blocks` could surprise users with stale local configs. → Mitigation: update defaults/changelog and fail clearly if old configs still provide it through typed schema validation or compatibility handling.

## Migration Plan

1. Change the inspection dump output to emit component-qualified parameter names.
2. Change optimizer group matching to consume the same component-qualified names.
3. Remove `optimizer.learning_rates.blocks` from config dataclasses/defaults and any dependent tests.
4. Update tests, changelog, and planning notes to reflect the new canonical selector namespace.
5. Leave future module-selector work to build on the same namespace instead of adding another one.

Rollback is straightforward because the change is localized to selector generation, config matching, and schema surface. Reverting those files restores the previous behavior.

## Open Questions

- Whether the dump should emit only qualified names, or preserve component grouping while making each key component-qualified as well.
- Whether any temporary compatibility shim for legacy `denoiser.*` configs is worth the complexity, or whether a clean break is better.
