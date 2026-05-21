# Project Roadmap & Future Ideas

## Architecture Overview

The active training stack follows a `train.py` + `Trainer` + `TrainingMode` + `TrainingStrategy` pattern:

```
Root launcher:                       Library modules:
┌─────────────────┐                  ┌─────────────────────────────┐
│    train.py     │ ────────────────►│ library/strategies/         │
│ config-driven   │                  │   base/contracts.py         │
│ mode + family   │                  │   shared/...                │
└────────┬────────┘                  │   sd/training.py            │
         │ builds                    │   sdxl/training.py          │
         │                           └─────────────┬───────────────┘
         ▼                                         │ uses
┌─────────────────┐                  ┌─────────────┴───────────────┐
│ training mode   │                  │ library/training/           │
│ peft / finetune │                  │   runners/trainer.py        │
└─────────────────┘                  │   modes/*.py                │
                                     │   phases/*.py               │
Dedicated entrypoints such as        │   checkpointing.py          │
`scripts/sdxl_textual_inversion.py`  │   sample_generation.py      │
remain outside the root launcher     └─────────────────────────────┘
until their runtime path is migrated.
```

**Representative module separation:**

- `library/strategies/base/` - required contracts and optional feature seams
- `library/strategies/shared/` - genuinely shared strategy-owned behavior
- `library/strategies/sd/`, `library/strategies/sdxl/` - representative family strategy facets
- `library/training/runners/` - Trainer orchestration
- `library/training/modes/` - TrainingMode plugins (AdapterMode, FineTuneMode)
- `library/training/phases/` - Phase functions (setup, caching, model_prep, optimizer, training_loop)
- `library/training/` - Model-agnostic utilities (checkpointing.py, sample_generation.py)

**Current naming direction:** shared trainer/base-strategy seams now prefer `denoiser` terminology, while concrete SD / SDXL implementation details may continue to use `unet` where the component is genuinely architecture-specific.

**Remaining unet -> denoiser follow-up:**

- Concrete SD / SDXL strategy internals still use `unet` in places like `call_unet(...)` and some save/sample helpers; this is acceptable for now, but should be revisited if/when a non-UNet denoiser family starts using the same concrete strategy shape.
- Compatibility-sensitive surfaces still retain `unet` naming for now:
  - config/schema fields like `fp8_base_unet`
  - adapter/internal optimizer APIs that still accept `unet_lr`
  - metadata keys like `ss_unet_lr`
- Deprecated and `copy` script/reference files should not drive naming decisions for the active contract.

---

## Testing Suite

Current testing follow-up is about expanding coverage around the remaining riskier surfaces, not standing up basic pytest infrastructure.

| Category                   | Items                                         | Priority    |
| -------------------------- | --------------------------------------------- | ----------- |
| **Adapter Classes**        | `LoRAAdapter.apply_to()`, `create_adapter()`  | Medium      |
| **Sample Generation**      | `sample_images_common`, inference pipeline    | Medium      |

---

## Pending Training Features

Features intentionally excluded from the Phase 2B `FineTuneMode` migration. Currently fail-fast with `NotImplementedError` in `FineTuneMode.build_optimizer_params()` to prevent silent behavior differences.

- [x] **Pattern-based optimizer groups** — Regex/glob-based param grouping for fine-grained LR control. Same mode-agnostic approach as block LR.
- [ ] **Fused optimizer groups** — Multi-optimizer support with `fused_backward_pass` (per-parameter backward hooks). Complex multi-optimizer logic from legacy `sdxl_finetune.py`.
- [x] **Adapter module/param breakdown** — Adapter-mode startup diagnostics and benchmark-report memory estimates now derive per-component module/parameter counts from repo-owned adapter trainable refs instead of collapsing to one generic `adapter` bucket. Adapter-owned reporting now lives under `library/adapters/shared/reporting.py`, so the shared observability path can render original component-module counts plus adapter-module counts while still defaulting to public model-family labels (`unet`, `clip_l`, etc.) for human-facing output.

> [!IMPORTANT]
> Keep fused/block/pattern paths explicitly fail-fast (as planned), so they don't silently behave differently. Only remove the guards when proper implementations are added.

---

## Configuration Refactoring

### Active TODOs

- [ ] Validation layers with proper boundaries instead of some central some local
- [ ] IMPORTANT: Research and brainstorm on the best possible form for the strategy layer. Combines with `Strategy system follow-up` from code quality todos

---

## Code Quality TODOs

- [ ] **Strategy system follow-up** — The large base-strategy cleanup is mostly done. Remaining work is narrower:
  - SD / SDXL concrete strategy cleanup against the current base contract
  - naming/organization review for `library/strategies/base/contracts.py`
  - later concrete/compatibility `unet` -> `denoiser` cleanup where it still makes sense
  - further `base/`-to-`models/` ownership cleanup where model-specific behavior still sits too high
  - `_deprecated` and `copy` reference files should not receive normal refactor work
  - See `docs_design/strategy_system_followup.md`, `docs_design/strategy_base_decision.md`, and `docs_design/strategy_remaining_facet_audit.md`
- [ ] **`training_plots.py`** - Functions access multiple sub-configs (`cfg.output.saving`, `cfg.output.logging`, `cfg.timestep`) - acceptable for orchestration functions but could be cleaner
- [ ] **Delete legacy training wrappers** (after legacy script deprecation) - Once `*_finetune.py` and `*_textual_inversion.py` scripts are migrated to new data pipeline, delete:
  - `library/training/_deprecated/sd_sample_generation.py`
  - `library/training/_deprecated/sdxl_sample_generation.py`
  - `library/training/_deprecated/sd_checkpointing.py`
  - `library/training/_deprecated/sdxl_checkpointing.py`
  - Strategies now call `sample_images_common()` directly; checkpointing logic can be inlined into strategies when legacy scripts are removed.
- [ ] Investigate naming conventions and possible drifts in the objective class and runtime layers
- [ ] Dep version health check, lots of old versions pinned

### Near-Term Follow-up

- [ ] **SD / SDXL strategy cleanup against the current contract** — The large structural split is done; the remaining work is narrower cleanup and architecture follow-up now that the base contract has settled.
  - SD tokenization / text-encoding / caching ownership now lives in the self-titled facet files, and the remaining SD contract-owned concerns have been split into `sd/loading.py`, `sd/model_preparation.py`, `sd/checkpointing.py`, `sd/sampling.py`, `sd/denoiser.py`, `sd/diffusion.py`, and `sd/validation.py`.
  - `sd/training.py` is now reduced to strategy assembly and init/wiring, mirroring the SDXL composition pattern.
  - SDXL tokenization / text-encoding / caching ownership now also lives in the self-titled facet files, and the remaining SDXL contract-owned concerns have been split into `sdxl/loading.py`, `sdxl/model_preparation.py`, `sdxl/checkpointing.py`, `sdxl/sampling.py`, `sdxl/denoiser.py`, `sdxl/diffusion.py`, and `sdxl/validation.py`.
  - `sdxl/training.py` is now reduced to strategy assembly and init/wiring.
  - SDXL training-time text conditioning now routes through the strategy tokenization / encoding seam instead of calling model helpers directly from `sdxl/diffusion.py`, which keeps the diffusion facet aligned with the settled strategy contract.
  - The new conditioning facet now gives the concern a real home, but the return-shape convention is still intentionally loose and family-local while more model families are ported.
  - Prompt weighting / weighted captions still likely want to become a shared concern rather than a model-by-model accumulation of special cases, especially once cache-policy expectations are made explicit.
- [ ] **Inspection/load seam follow-up** — The rebuilt model inspection tool can now ride `build_training_strategy(...).load_target_model(...)`, and dumps plus fine-grained optimizer matching now share one canonical public selector namespace (`component.local_name`). The remaining follow-up is that the tool still has to fabricate a tiny runtime config because there is no thinner inspection-neutral loading seam yet. Treat that as repo-flow follow-up work rather than teaching the tool or loaders new dump-specific APIs.
- [ ] **EDM2 presence follow-up** — The runtime/config seam is cleaner now and the repo has an initial SDXL adapter preset/example under the current PEFT family, but the feature still needs real docs and clearer guidance on when to use it.
- [ ] **Conditioning architecture follow-up** — Pressure-test the new `ConditioningStrategy` seam against more model families and decide whether any sub-conventions under `resolve_conditioning(...)` are mature enough to standardize.
- [ ] **Prompt weighting / weighted captions review** — Decide whether weighted captions should become an active shared concern and where prompt-weight parsing/application should live.
- [ ] **Regularization-image UX / docs note** — The current DreamBooth-style `reg_data_dir` / `is_reg` path is mechanically correct, but it only helps when those images are genuine class/prior images with matching generic captions or `class_tokens`, not just arbitrary extra images. Make sure future docs/examples call that out explicitly.
- [ ] **Dashboard / logging system rework** — The first-pass training observability refactor now lives under `library/logging/console.py`, `metrics.py`, `summaries.py`, `reports.py`, and `resource_monitor.py`, but the future UI/dashboard side still needs its own broader product/system design. Keep that follow-up focused on new sinks/backends and UI behavior rather than reopening the console/report/resource ownership boundaries that now exist.
- [ ] **Repo layout review** — Re-check whether `library/` folder organization is comfortable or if there's room to improve the layout.
- [ ] **LyCORIS vendor / integration pass** — Treat LyCORIS as a vendor/integration ownership question rather than an external dependency question, since adapter breakdown follow-up depends on tighter ownership and easier modification.
- [ ] **Adapter-system follow-up** — The active adapter rework now has optimization-owned target/grouping ownership, method-local PEFT config under `adapter.peft.<method>` branch presence plus explicit continuation intent, repo-owned LoHa, LoCon, LoKr, OFT, BOFT, DyLoRA, GLoRA, and IA3 method implementations, an explicit persistence split where `AdapterMode` orchestrates checkpoint/export flows while adapter runtime objects participate through repo-owned persistence helpers, and a runtime-layer loaded-runtime/merge-request seam for the built-in from-weights and base-weight-merge flow. The remaining follow-up is broader adapter breadth, generic artifact-initialization / pre-merge config ownership, and LyCORIS/vendor integration work rather than reopening compatibility-era optimizer, persistence, or merge boundaries.
  - The shared target-ref foundation now also lives under `library/optimization/targets.py`, with fine-tune parameter refs and adapter module targets both carrying the same component-qualified selector and provenance model. Future module-type selectors or adapter-specific grouping work should extend that shared target vocabulary rather than reintroducing an adapter-only target surface.
- [ ] **torchao pulled into the repo so it can be modified when wanted** — upstream is restrictive for offloading
- [ ] **Future conditioning/data-flow experiments** — Later exploration area for better caption mutation, TE caching, on-the-fly CPU encoding, queues, async handoff, and related conditioning/data-flow improvements once the current building blocks are settled.

---

## Next Phase After Stabilization

Once the current stabilization / cleanup list above is tied off, the roadmap should shift from architecture settling to capability expansion.

- [ ] **Gradual new model implementations** — Add new model families incrementally on top of the current trainer / strategy foundation instead of trying to land a large multi-model rewrite all at once.
- [ ] **New training approaches** — Open the next wave of work around new training methods once the active SD / SDXL path is stable enough to serve as the reference implementation.
- [ ] **RamTorch vendor / integration pass** — Decide how much RamTorch should be owned and integrated directly in-repo as future model/training work expands.
- [ ] **Alternative VAE training support** — Support training existing model families against different VAEs without forcing that logic to stay script-local or ad hoc.
- [ ] **Additional capability-expansion items** — Keep this section open for the next wave of model/runtime work once the current cleanup phase is no longer the primary constraint.

---

## Future Ideas

- [ ] **Support for feather** - https://github.com/SuriyaaMM/feather
  - Feather is a high-performance emulation library that brings FP8 (E5M2 & E4M3) precision arithmetic to older GPU architectures (Ampere, Turing, Volta) that lack native hardware support. Currently only considered for inference
  - Note: the current `fp8_base` / `fp8_base_unet` config flags have a fairly narrow active effect. In the current training path they mostly drive shared model-prep dtype casting for the denoiser / text encoders (with TE embedding workarounds), plus validation and metadata. They should not be treated as a broad quantization backend or a settled precision architecture.

- [ ] **Investigate 2022-2023 backend code**
  - After cecking sd_original_unet.py we found that it referenced bugs and had workaround for said bugs from 2022-2024. The model backend might be outdated or harming performance/code quality at large. A wider audit of the backend against diffusers or original code might be necessary down the line.

- [ ] Model download/load to memory for training from huggingface

- [ ] Selective activation checkpointing

- [ ] Run warehouse for experiment metadata, telemetry, artifacts, and outcomes
  - supports resource modeling
  - supports config/result analytics
  - supports reproducibility and regression tracking
  - supports future recommendation and forecasting

### Future Improvements

- [ ] Config-hash cache namespace - Auto-segregate caches by config hash (`resolution`, `bucket_steps`, `model_version`) to prevent cross-config issues. See `AUDIT/AUDIT_PHASE_6.md`.
- [ ] **Large-scale dataset manifest optimization** - Current JSON manifest grows ~2KB/entry (100k images = ~200MB JSON). Options:
  - Binary format (msgpack/pickle) for faster I/O
  - Incremental manifest updates instead of full rewrite
  - Lazy loading of manifest entries
  - Sharded manifests by bucket
  - Skip manifest creation if unchanged from previous run
  - Maybe fp8 for te output storage, needs tests
- [ ] **Smarter resource tracking/management** - This helps with training and also with inference, for example falling back to tiled vae when it would hit resource contraints and such. See `docs_design/resource_monitor_plan.md`
- [ ] Old toml to new config translator
- [ ] Constants rework
- [x] Metadata system backbone first slice — `library/metadata/` now provides typed records, emitter/provider seams, validation, backend, storage, and projection seams; active checkpoint metadata routes through the backbone while preserving existing `ss_*` / `modelspec.*` export behavior.
- [x] Metadata legacy migration control — legacy metadata surfaces are classified in `docs_design/metadata_legacy_migration_control.md`, `ss_*` key ownership has moved into `library.metadata.keys` with compatibility re-exports, and the first shared run/model/artifact fact dataclasses now feed checkpoint metadata wrappers.
- [x] Active training metadata builder replacement — the old `library/training/training_metadata.py` and extra `library/training/metadata_providers.py` helper have been removed instead of retained as wrappers; the remaining training metadata seam is transitional while the target shape is central metadata emitters plus local trainer call sites.
- [x] Deprecated PEFT training script fence — `scripts/_deprecated/sd_peft.py` and `scripts/_deprecated/sdxl_peft_copy.py` now fail immediately with replacement `train.py` preset guidance instead of referencing removed training metadata helpers.
- [x] Metadata ownership language clarification — metadata docs and OpenSpec now state the central-system target explicitly: central recorded dataclasses, central emitters/builders, central projections/backend/storage, normal domains as lifecycle call sites, and local metadata modules only for explicit plugin/family exceptions.
- [ ] Metadata system follow-ups — finish-first slices are exported parity, optimization metadata, and analytics/export formats; observability/report metadata emitters are now in place and the logging observer now has the first live direct `MetadataRuntime.file(item)` path. Last-pass slices are model-family metadata facts (`sd-scripts-ao3`), data/cache emitters (`sd-scripts-ucs`), and adapter method metadata providers (`sd-scripts-r61`).
  - [x] Central metadata emitter parity cleanup — the shared metadata dataclasses are now schema-only, record/event conversion lives in central emitter builders, and the active checkpoint plus observability metadata paths consume `MetadataProviderResult` from emitters instead of one-off provider wrapper classes.
  - [x] Durable SQLite storage is now implemented under `library.metadata.storage.SQLiteMetadataStore` with versioned schema setup and focused tests.
- [ ] Investigate the following comment 
  > Disable cuDNN SDPA backend — broken on some H100 clusters with certain cuDNN versions.
  > Falls back to Flash Attention or math backend.
  > #torch.backends.cuda.enable_cudnn_sdp(False)

---

## Training Mode Extensibility

The `Trainer` + `TrainingMode` split is the active extensibility pattern for training-mode differences.

### What Actually Differs (PEFT vs Full Fine-Tuning)

| Concern                     | PEFT (LoRA/Adapter)                          | Full Fine-Tuning                     |
| --------------------------- | -------------------------------------------- | ------------------------------------ |
| **Trainable params**        | `adapter.parameters()` + optionally TE       | `denoiser.parameters()` + optionally TE |
| **Model prep**              | Create adapter, `apply_to()`, freeze base    | `requires_grad_(True)` on base model |
| **Checkpoint save**         | Adapter state dict                           | Full model state dict                |
| **`accelerator.prepare()`** | Wraps adapter                                | Wraps denoiser directly              |
| **Caching**                 | Identical                                    | Identical                            |
| **Training loop**           | Identical (strategy handles `process_batch`) | Identical                            |

### Phase Reusability

- `caching.py` — **100% shareable**
- `training_loop.py` — **~95% shareable** (checkpoint calls go through `trainer.save_checkpoint()`)
- `optimizer.py` — **~80% shareable** (param groups and `accelerator.prepare()` wrapping differ)
- `model_prep.py` — **Current-adapter-family-specific** (today that means PEFT-family adapter creation, not a generic future-adapter seam yet)

### Current Focus

- `Trainer` + `TrainingMode` is settled as the active pattern.
- `AdapterMode` and `FineTuneMode` are both in place.
- Shared phases already route divergent behavior through trainer/mode hooks.
- Remaining work is follow-up cleanup: optimizer-group features, diagnostics, and removal of legacy wrappers once unmigrated paths are gone.

---

## Testability Improvements

---

## CI/CD Setup

- [ ] Pre-commit hooks for running tests

### Current State

- GitHub Actions pytest workflow is in place.
- Coverage reporting/tracking is already set up.

---

## Data Pipeline Rework

**Status:** 🔄 In Progress

See `DATA_PIPELINE_PLAN.md` for design, `DATA_PIPELINE_CURRENT.md` for implementation checklist.

### Current State

- The active data pipeline already has manifests, cache entries, bucket metadata, `CachingEngine`, `TrainingDataset`, and epoch preparation in place.
- The remaining roadmap here is about scaling and future cache-store architecture rather than getting the base pipeline working.

---

## Large Scale Caching Architecture

**Status:** 📋 Design Phase (for 1M+ image datasets)

The current per-image caching approach works well for <100k images but faces challenges at scale:

- **NTFS/filesystem overhead**: Millisecond-level per-file overhead at high file counts
- **Too many file opens**: Training I/O becomes bottleneck

### Current Architecture (Per-Image)

```
cache/
├── img_001_sdxl_latents.safetensors  # Latent + conditioning metadata
├── img_001_sdxl_te.safetensors       # TE outputs (optional, separate file)
└── ...
```

**Pros:** Simple, incremental updates, easy debugging
**Cons:** Doesn't scale beyond ~100k files on Windows/NTFS

### Proposed Future Architecture (Sharded)

```
cache/
├── latents/
│   └── {config_hash}/              # Bucket settings hash (invalidation namespace)
│       ├── bucket_1024x1024/
│       │   ├── shard_0000.safetensors  # ~10k images per shard
│       │   └── shard_0001.safetensors
│       └── bucket_768x1024/
│           └── shard_0000.safetensors
│
├── te_outputs/
│   └── {te_config_hash}/           # Tokenizer + encoder settings hash
│       ├── shard_0000.safetensors  # Keyed by caption hash (dedup!)
│       └── shard_0001.safetensors
│
└── # Tokens: on-the-fly (default) or epoch-level file (optional)
```

### Design Principles

1. **Config-Hash Namespacing**
   - Changing bucket settings writes to a _new_ namespace (new hash directory)
   - No in-place rewriting; old cache remains until explicitly deleted
   - Same pattern as how per-image caches invalidate when settings change

2. **Caption-Hash Deduplication for TE**
   - TE outputs keyed by caption hash, not image ID
   - Same caption → same encoding (dedup across images sharing captions)
   - Hash must include: tokenizer settings, max_length, clip_skip, encoder version

3. **Independent Invalidation**
   - Latent config hash: `bucket_steps`, `base_resolution`, `no_upscale`, etc.
   - TE config hash: `tokenizer_version`, `max_token_length`, `clip_skip`, etc.
   - Change buckets → only rebuild latent shards (TE remains valid)
   - Change captions → only rebuild TE shards (latents remain valid)

4. **Tokens: On-the-Fly Default**
   - Tokenization is ~0.5ms/sequence (negligible vs GPU time)
   - Epoch-level token file available as opt-in for specific workflows
   - Avoids coupling token cache to caption augmentation scheme

### Implementation Plan

1. **Abstract Cache Backend**

   ```python
   class CacheStore(ABC):
       def save(self, key: str, data: dict[str, Tensor]) -> None: ...
       def load(self, key: str) -> dict[str, Tensor]: ...
       def exists(self, key: str) -> bool: ...

   class PerImageCacheStore(CacheStore):  # Current implementation
       ...

   class ShardedCacheStore(CacheStore):   # Future implementation
       ...
   ```

2. **Training-facing API unchanged**
   - Same `CacheData` contract
   - Same `CacheBackend` interface
   - Backend switch via config (`cache_backend: "per_image" | "sharded"`)

3. **Threshold-based recommendation**
   - Default: per-image for <100k images
   - Recommend sharded for 100k+ or Windows/NTFS users
   - Automatic detection possible (count files, check filesystem)

### Tasks

- [ ] Design config-hash computation for latent namespace
- [ ] Design caption-hash computation for TE dedup (include all relevant settings)
- [ ] Implement `ShardedCacheStore` with bucket-based sharding
- [ ] Implement shard-level `get_slice` for efficient batch loading
- [ ] Add `cache_backend` config option
- [ ] Migration utility: per-image → sharded conversion
