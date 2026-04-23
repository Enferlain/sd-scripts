## Why

The adapter-system rework settled the ownership model for adapter training, but
the active target-resolution path still only resolves component-root targets.
That is too coarse for the richer model-defined targeting the repo already
wants to support, and absorbed LyCORIS-style methods like `loha` are the first
consumer that need optimization to hand them concrete module targets while
grouping remains parameter-native. This slice proves richer resolved targets
through a module-resolved method first without claiming all absorbed methods
are limited to module-only binding.

## What Changes

- Extend the repo's optimization-owned target-resolution path to concrete module
  targets so downstream consumers can consume already-resolved modules instead
  of rediscovering targets.
- Extend the repo-owned adapter target payload so adapter runtimes receive
  module-level provenance (`component`, `component_key`, `path`, `module`) from
  optimization.
- Introduce a first repo-native absorbed LyCORIS method runtime for `loha`
  under `library/adapters/methods/peft/loha/`.
- Treat absorbed methods as a path toward full repo-owned implementations,
  not just repo-owned wrappers around vendor algorithm classes.
- Keep optimizer grouping parameter-based by continuing to consume
  repo-owned trainable parameter refs with provenance rather than module-owned
  grouping logic.
- Keep `PeftMode` as the training-side orchestrator that carries resolved
  targets into adapter runtime construction and passes returned trainable refs
  back into optimization-owned grouping.
- Route the first `loha` build/from-weights/merge flow through the existing
  repo-owned adapter runtime and loaded-runtime seams instead of using the
  vendor LyCORIS wrapper as the repo contract.
- Leave mixed-method target assignment, overlap handling, and conflict
  resolution for a later pass after the first absorbed method lands.

## Capabilities

### New Capabilities
- `adapter-module-targeting`: Optimization-owned module targeting and repo-owned adapter runtime expectations for module-resolved adapter methods such as `loha`.

### Modified Capabilities
- None.

## Impact

- Affected code: `library/optimization/grouping.py`, `library/adapters/runtime/`,
  `library/adapters/registry.py`, `library/adapters/methods/`, and
  `library/training/modes/peft_mode.py`
- Affected tests: adapter runtime registry coverage, PEFT mode coverage, and
  adapter optimizer grouping coverage
- Affected future direction: absorbed LyCORIS methods can be brought into the
  repo-owned adapter system without reintroducing vendor-owned target discovery
  or adapter-owned grouping policy, while still moving toward fully repo-owned
  method implementations rather than leaving vendor algorithm classes as the
  steady-state runtime dependency
