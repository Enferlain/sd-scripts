## Active metadata path inventory

This inventory records the pre-migration behavior that the typed model-family
metadata path must either preserve or change deliberately. It distinguishes
supported compatibility from incidental bugs so parity tests do not silently
turn every transitional behavior into a permanent contract.

### Central orchestration

- `Trainer._initialize_training_metadata()` builds canonical run facts and then
  lets `CheckpointingStrategy.update_metadata()` mutate the exported dictionary.
  SD and SDXL are no-ops; SD3 adds the two attention-mask values.
- `Trainer._build_checkpoint_metadata()` asks the active family for an already
  rendered ModelSpec dictionary on every main or sidecar save.
- `TrainingMetadataState.build_checkpoint_metadata()` wraps that dictionary in
  `ModelSpecFacts`, and the checkpoint emitter stores both prefixed compatibility
  keys and a few duplicate canonical facts.
- `ModelSpecCompatibilityProjection` currently filters pre-rendered
  `modelspec.*` facts instead of mapping semantic facts.

### Family producers and full-model writers

- SD produces adapter-role SD1/SD2 ModelSpec dictionaries. The modern SD
  checkpointing facet has no full-model writer.
- SDXL produces adapter-role dictionaries for the trainer path. Its stable
  full-model writer builds a second full-model dictionary, merges it over the
  trainer metadata, and writes metadata only for safetensors. The second build
  currently defaults resolution to `512x512` and omits timestep and encoder
  fields, while the earlier `kuro.model.*` facts still describe the adapter.
  The `.ckpt` and diffusers writers do not persist the supplied header metadata.
- SD3 produces full-model ModelSpec semantics and writes metadata to the unified
  MMDiT/VAE safetensors artifact. Text-encoder sidecars remain metadata-free.
  An SD3 adapter run therefore currently inherits full-model ModelSpec semantics.
- The broad builder in `library/utils/model_metadata.py` owns architecture,
  implementation, default title/date, resolution, prediction/timestep/encoder
  omission, thumbnail I/O, implementation-version lookup, prefixes, and user
  extension collision precedence.

### Adapter and sidecar consumers

- Adapter save paths pass the projected dictionary through unchanged to their
  safetensors writer. Torch output has no safetensors header metadata.
- VeRA remains a deliberate method-owned exception: it adds reconstruction
  metadata to the upstream dictionary. This change must not absorb those fields
  into generic model-family facts.
- EDM2 sidecars receive the same family/run projection and add their own hash
  fields. Step/epoch sidecars currently use the default artifact identity and
  omit coordinates; the final sidecar supplies its name, step, and epoch.

### Observable compatibility baseline

- Normal checkpoint projection contains `modelspec.*`, `kuro.schema_version`,
  and the applicable `kuro.run.*`, `kuro.model.*`, and `kuro.artifact.*` keys.
- `no_metadata=True` still retains ModelSpec and Kuro model identity/facts while
  omitting run and artifact records.
- SD3 attention masks currently appear as `kuro.run.apply_*_attn_mask`.
  Historical `ss_apply_*_attn_mask` keys are documented but have no active
  producer; restoring them is an explicit compatibility repair in this change.
- User extension fields currently override standard ModelSpec fields on key
  collision, including the title, architecture, and ModelSpec version.

### Deferred or separate paths

- The active SDXL textual-inversion script inherits a deprecated parent that
  still calls the broad ModelSpec helper. It needs an explicit migrate-or-retire
  decision before that helper is removed.
- Stale model-management tools reference removed metadata APIs and should be
  handled as separate tool-cleanup work rather than preserving active facades.
- Adapter-local reconstruction metadata, source-metadata ingestion, tensor hash
  generation, and the long-term `no_metadata` policy remain outside this slice.

### Runtime filing seam

- `LoggingTrainingObserver` owns the one shared `MetadataRuntime`; model facts
  must use that instance rather than creating a trainer- or strategy-local path.
- File realization/component facts after `prepare_models()` resolves any lazy
  denoiser and before startup resource reporting. Guard filing to the main
  process and use `str(session_id)` as the current run identity.
- Use a deterministic target-model realization key and qualified component
  identities derived directly from `trainer.loaded_components`. Resource
  summary fallback labels are not authoritative family-local component keys.
- The current 32-bit process-local session ID is adequate only for this slice's
  within-run qualification; a resume-stable distributed run identity remains
  follow-up work before shared long-lived storage is treated as collision-safe.
