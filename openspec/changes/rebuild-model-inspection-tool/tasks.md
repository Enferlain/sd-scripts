## 1. Reset the failed direction

- [x] 1.1 Remove dump-specific loader/package APIs and any other inspection-only library seams that were introduced by the previous attempts.
- [x] 1.2 Decide what remains in `library/models/parameter_dump.py` versus what should become fully tool-local, and document that boundary while rewriting the tool.

## 2. Rebuild the inspection tool

- [x] 2.1 Rewrite `tools/model_management/dump_named_parameters.py` so it has a clear identity as a model inspection tool with explicit views for parameters, modules, and optional state.
- [x] 2.2 Make the default output emit deterministically ordered named parameters under existing top-level component names without reusing optimizer/trainability logic.
- [x] 2.3 Add the module inspection view so named modules and runtime types are emitted under the same component ordering.
- [x] 2.4 Keep state/buffer dumping as an optional secondary view and ensure buffer metadata comes from the real loaded runtime state.

## 3. Validate against real model support

- [x] 3.1 Verify the tool uses the repo's existing model support path for supported families instead of new dump-specific model APIs.
- [x] 3.2 Add focused tests for output ordering, component grouping, and the supported inspection views.
- [ ] 3.3 Run representative real-tool checks, including the SD3 asset that previously exposed loader fragility, and record any remaining loader/runtime issues separately from the tool contract.

## 4. Capture repo-flow findings

- [x] 4.1 Document any subpar loading/tooling seams exposed during this rewrite so they are visible as repo-flow follow-up work rather than being buried inside the inspection tool.
- [x] 4.2 Update changelog and related planning notes to reflect the rebuilt tool direction and any explicitly identified follow-up issues.
