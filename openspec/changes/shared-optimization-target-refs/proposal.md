## Why

Fine-tune grouping and adapter targeting were improved in separate phases, so
the active code now has parameter-oriented selection for fine-tuning and
adapter-local module targets for adapter training. Now that the adapter runtime
path is repo-owned enough to participate in the same architecture, optimization
needs a shared target vocabulary that can represent components, modules, and
parameters without forcing every mode into the same lifecycle.

## What Changes

- Add a shared optimization-owned target-reference model for targetable
  component, module, and parameter items.
- Preserve current fine-tune and adapter training behavior while carrying richer
  target provenance through both paths.
- Promote module type and owner-module provenance into the shared target model
  so future module-type selectors and LyCORIS-style adapter methods do not need
  to inspect mode-specific or adapter-specific internals.
- Make adapter module targeting consume or wrap shared optimization target refs
  instead of defining adapter-only target identity as the long-term contract.
- Keep mode-specific sequencing explicit: fine-tune exposes existing base
  parameters, while adapter mode builds adapter trainables after module targets
  are resolved.
- Do not introduce a broad new selector DSL in this change; this is the
  target-provenance foundation that future selector/grouping work can build on.

## Capabilities

### New Capabilities
- `optimization-target-refs`: Defines shared optimization-owned target refs for
  component, module, and parameter targets, including selector names, module
  type provenance, and object references needed by training modes.

### Modified Capabilities
- `adapter-module-targeting`: Adapter module targets must be represented through
  shared optimization target refs or carry equivalent shared target provenance
  instead of remaining an adapter-only target vocabulary.
- `component-qualified-selectors`: Component-qualified selector names must remain
  the public path surface for parameter targets and become the selector basis
  for shared module and parameter target refs.

## Impact

- Affected code: `library/optimization/grouping.py`, new
  `library/optimization/targets.py`, `library/adapters/runtime/targets.py`,
  `library/adapters/shared/trainables.py`, PEFT adapter runtimes, and focused
  grouping/adapter tests.
- Affected docs/specs: adapter system docs and OpenSpec requirements for target
  provenance and component-qualified selectors.
- No dependency changes are expected.
