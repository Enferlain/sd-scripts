## Why

Metadata is currently spread across checkpoint dictionaries, model-spec helpers,
safetensors storage boundaries, adapter method state, optimizer/runtime facts,
data/cache validation, and logging artifacts without one typed backbone or
projection boundary. This makes it hard to preserve compatibility while adding
repo-owned `kuro.*` metadata, future durable storage, and domain-owned metadata
surfaces for models, adapters, data, cache, optimization, and observability.

## What Changes

- Add a dataclass-first metadata backbone under `library/metadata/` for typed
  records, events, identity, relationships, validation, provider contracts,
  storage seams, and projections.
- Add a provider/metadata-adapter layer so domains keep ownership of their
  metadata facts while the central backend composes and validates cross-domain
  records.
- Add projection seams for repo-owned `kuro.*` artifact keys plus compatibility
  projections for existing `ss_*` and `modelspec.*` metadata.
- Route the active checkpoint metadata path through the metadata backbone while
  preserving current exported compatibility keys.
- Add an in-memory/test backend and storage interfaces shaped for a future
  SQLite durable store without requiring SQLite to be fully implemented in the
  first slice.
- Add fail-fast validation for required producer-owned metadata facts at
  provider and export boundaries.
- Keep this change focused on the backbone and active checkpoint metadata path;
  do not implement async data, sharded caches, streaming ingest, exposure
  accounting, or a full run warehouse in this change.

## Capabilities

### New Capabilities

- `metadata-backbone`: Typed repo-owned metadata collection, validation,
  provider registration, projection, and first checkpoint-artifact integration.

### Modified Capabilities

None.

## Impact

- `library/metadata/`: new central metadata contracts, dataclasses, backend,
  validation, projections, and storage seams.
- `library/training/training_metadata.py`: migrate or wrap existing training-run
  metadata construction behind typed metadata providers/projections.
- `library/utils/model_metadata.py`: preserve current behavior while moving
  model-spec output toward projection ownership.
- `library/training/runners/trainer.py` and checkpoint save paths: request
  metadata projections from the backbone for active checkpoint artifacts.
- `library/strategies/*/checkpointing.py`: provide or bridge model-family
  metadata facts through provider surfaces while preserving current strategy
  behavior.
- Tests: add focused coverage for required fact validation, `kuro.*` projection
  basics, `ss_*` compatibility, `modelspec.*` compatibility for SD/SDXL, and the
  migrated active checkpoint metadata path.
- No new external dependencies are expected for the first slice.
