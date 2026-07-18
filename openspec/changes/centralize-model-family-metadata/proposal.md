## Why

The active model-metadata path is backwards: SD, SDXL, and SD3 first build flat `modelspec.*` export dictionaries, `ModelSpecFacts` then wraps those dictionaries, and strategy hooks still mutate family-specific checkpoint metadata directly. The completed family-declared loaded-component work now provides the missing authoritative model/component boundary, so model metadata can become typed, relational, and reusable without preserving the old diffusion tuple or treating safetensors keys as the internal model.

## What Changes

- Introduce central typed facts for model-family identity, run-specific model realization, declared loaded components, and artifact-facing model metadata.
- Derive persisted component topology from family-owned loaded-component declarations while preserving declared order, stable keys, public labels, roles, and capabilities without persisting live module objects.
- Qualify model and component identities by their owning realization/run so repeated family-local keys cannot collide across runs or model families.
- Replace raw strategy `get_model_metadata()` dictionaries and `update_metadata()` mutation hooks with typed family-owned fact resolution feeding central emitters.
- Route low-volume model-realization facts through the existing metadata registry/runtime/backend path and link them to runs, components, and produced artifacts.
- Make repo-owned `kuro.*`, `modelspec.*`, compatibility-only `ss_*`, and string-only safetensors metadata projection outputs from accepted facts while preserving SD/SDXL/SD3 external compatibility except for explicitly corrected reference-implementation identifiers. `kuro.*` is the preferred repository-native export, not the internal source of truth; this change migrates only the model-family-owned subset of the wider legacy `ss_*` surface.
- Correct canonical implementation claims to the family reference codebases: CompVis Stable Diffusion for SD1, Stability AI `stablediffusion` for SD2, Stability AI `generative-models` for SDXL, and Stability AI `sd3.5` for SD3/3.5.
- Remove the active dependency on broad flag-driven ModelSpec builders once focused parity coverage proves the replacement path.

## Capabilities

### New Capabilities

- `model-family-metadata`: Typed model-family, model-realization, loaded-component, and artifact-projection metadata with qualified identities and compatibility-preserving exports.

### Modified Capabilities

None. This change consumes the completed `family-declared-loaded-components` contract but does not redefine that capability.

## Impact

- Affected code includes central run/model builders under `library/metadata/builders/`, `library/metadata/dataclasses/model.py`, emitter/registry/validation/graph/projection code, the trainer metadata lifecycle, SD/SDXL/SD3 checkpointing facets, checkpoint artifact projection, and the active portions of `library/utils/model_metadata.py`.
- The new capability becomes a shared provenance source for later resource, optimization, adapter, artifact-lineage, analytics, and run-warehouse work.
- Historical parity fixtures remain evidence of the pre-migration output, including the previously inconsistent `modelspec.implementation` values; the typed path intentionally corrects those values rather than preserving inaccurate provenance.
- No new dependency is required. SAI ModelSpec and safetensors remain external compatibility formats rather than internal storage models.
- The archived `family-declared-loaded-components` requirements are the baseline component contract consumed by this change.
