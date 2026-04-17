## Why

The current adapter path is still shaped too much by legacy PEFT/Kohya-era assumptions, which makes it harder to finish the adapter layer and unblock the next wave of repo work. The repo needs a clearer adapter architecture where optimization owns model-side targeting policy, the adapter system realizes adapter types against those resolved targets, and `PeftMode` continues to own the training-side orchestration.

## What Changes

- Define a repo-owned adapter-system architecture for adapter training instead of continuing to grow the current compatibility-shaped surface.
- Establish the ownership split between strategy, optimization, adapter system, and `PeftMode`.
- Rework the adapter-system direction so optimization continues to own what original-model parts are affected and how returned adapter state is grouped for training.
- Define the adapter system as the layer that realizes adapter types in the context of specific resolved targets rather than as the owner of targeting or optimizer policy.
- Move the design toward explicit adapter-type-specific configuration instead of assuming one broad shared adapter config surface.
- Clarify that adapter persistence belongs to the adapter-training path and remains training-side owned by `PeftMode`, even if adapter-specific runtime behavior participates in save/load.
- Explicitly design for adapter breadth beyond current LyCORIS and built-in diffusion examples so the architecture does not treat today's adapter shapes as universal.

## Capabilities

### New Capabilities
- `adapter-system`: Defines the repo-owned architecture, ownership boundaries, and training/runtime expectations for adapter-based training.

### Modified Capabilities
- None.

## Impact

- Affected docs/design: `docs_design/adapter_system_overview.md`, `docs_design/adapter_layer_direction.md`
- Affected config direction: `library/config/dataclasses/peft.py` and future adapter-type config surfaces
- Affected training/runtime direction: `library/training/modes/peft_mode.py`, `library/adapters/`, and the optimization-to-adapter handoff
- Affected future implementation planning: adapter persistence, adapter runtime concepts, and the long-term replacement of compatibility-shaped adapter interfaces
