## Why

The optimization-plan foundation now exists, but the active fine-tune path still assembles logical groups directly inside `FineTuneMode`. That leaves grouping policy embedded in a mode implementation instead of giving the optimization layer one shared home for trainer-facing grouping behavior.

## What Changes

- Add a shared optimization grouping module for the active base fine-tune path.
- Move denoiser/text-encoder logical-group assembly out of `FineTuneMode` and into shared optimization-layer code.
- Keep mode code responsible for trainable selection and high-level ownership, but not for building logical groups inline.
- Preserve current optimizer construction and scheduler behavior while making grouping policy reusable and easier to extend later.
- Keep adapters/PEFT, wrapper runtime redesign, and fused/multi-optimizer execution out of scope for this slice.

## Capabilities

### New Capabilities
- `optimization-shared-grouping`: Build stable base-path logical optimization groups through a shared optimization-layer grouping module instead of mode-local assembly.

### Modified Capabilities

## Impact

- Affected code: `library/optimization/types.py`, new shared grouping helpers under `library/optimization/`, `library/training/modes/finetune_mode.py`, and focused unit tests.
- Affected systems: fine-tune optimizer input preparation, logical-group construction, and future optimization-layer reuse points.
- Dependencies: no new runtime dependency; continues the existing optimization-plan architecture without changing adapter behavior.
