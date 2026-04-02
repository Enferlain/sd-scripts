# Project Roadmap & Future Ideas

## Architecture Overview

The codebase follows a Trainer + TrainingMode pattern where scripts are thin entry points, the Trainer orchestrates training phases, TrainingMode plugins handle mode-specific logic (PEFT vs fine-tune), and TrainingStrategy classes handle model-specific operations.

```
Scripts (thin entry points):          Library Modules:
┌─────────────────┐                   ┌─────────────────────────────┐
│   sd_peft.py    │                   │ library/strategies/         │
│   (~55 lines)   │ ───imports───────►│   base/contracts.py         │
└─────────────────┘                   │   sd/training.py            │
┌─────────────────┐                   │   sdxl/training.py          │
│  sdxl_peft.py   │ ───imports───────►└─────────────┬───────────────┘
│  (~55 lines)    │                                 │ uses
└─────────────────┘                   ┌─────────────┴───────────────┐
┌─────────────────┐                   │ library/training/           │
│ sdxl_finetune.py│ ───imports───────►│   runners/trainer.py        │
│  (~55 lines)    │                   │   modes/peft_mode.py        │
└─────────────────┘                   │   modes/finetune_mode.py    │
                                      │   phases/*.py               │
                                      │   checkpointing.py          │
                                      │   sample_generation.py      │
                                      └─────────────────────────────┘
```

**Module Separation (Post-Refactor):**

- `library/strategies/base/` - ABCs and shared logic (`TrainingStrategy`, `TokenizationStrategy`, etc.)
- `library/strategies/sd/` - SD1.5/2 implementations (SdTrainingStrategy, SdTokenizeStrategy)
- `library/strategies/sdxl/` - SDXL implementations (SdxlTrainingStrategy, SdxlTokenizeStrategy)
- `library/training/runners/` - Trainer orchestration
- `library/training/modes/` - TrainingMode plugins (PeftMode, FineTuneMode)
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

### Current Status

**Infrastructure:** ✅ Complete (pytest, fixtures, coverage)

| Category                | Status         |
| ----------------------- | -------------- |
| **Unit Tests (Pure)**   | ✅ Complete    |
| **Unit Tests (Mocked)** | ✅ In Progress |
| **Integration Tests**   | 🔜 Future      |

**Completed:** 1017+ unit tests across configuration, training, data, strategies, networks, losses, pipelines, training modes.

### Integration Tests (Remaining)

| Category                   | Items                                         | Priority    |
| -------------------------- | --------------------------------------------- | ----------- |
| ~~**Checkpoint I/O**~~     | ~~`load_models_from_*`, `save_*_checkpoint`~~ | ~~High~~ ✅ |
| **Adapter Classes**        | `LoRAAdapter.apply_to()`, `create_adapter()`  | Medium      |
| **Sample Generation**      | `sample_images_common`, inference pipeline    | Medium      |
| ~~**Full Training Loop**~~ | ~~Config → Trainer → Step~~                   | ~~High~~ ✅ |

---

## Deferred Training Features

Features intentionally excluded from the Phase 2B `FineTuneMode` migration. Currently fail-fast with `NotImplementedError` in `FineTuneMode.build_optimizer_params()` to prevent silent behavior differences.

- [ ] **Block-level learning rates** — Per-UNet-block LR grouping (legacy `get_block_params_to_optimize()`). Should be implemented as a mode-agnostic optimizer-group feature, not mode-specific.
- [ ] **Pattern-based optimizer groups** — Regex/glob-based param grouping for fine-grained LR control. Same mode-agnostic approach as block LR.
- [ ] **Fused optimizer groups** — Multi-optimizer support with `fused_backward_pass` (per-parameter backward hooks). Complex multi-optimizer logic from legacy `sdxl_finetune.py`.
- [ ] **PEFT module/param breakdown** — Per-component (unet, TE) module and parameter counts in training diagnostics for PEFT mode. Requires an optional adapter protocol method (`get_diagnostics_components()`) that each adapter type implements to report its own per-component allocation. `PeftMode` already has the `hasattr` hook ready — just needs adapter-side implementations. Deferred because adapter internals vary (LoRA, LyCORIS, OFT) and LyCORIS is still external.

> [!IMPORTANT]
> Keep fused/block/pattern paths explicitly fail-fast (as planned), so they don't silently behave differently. Only remove the guards when proper implementations are added.

---

## Configuration Refactoring

### Active TODOs

- [ ] Work on validation in general to figure out a system for catching invalid configs, might need to be post testing
- [ ] Check what "full bf16" means in our repo

### Recently Completed / Settled

- SDXL is now the second active RF consumer:
  - `model.model_type=sdxl` now validates with `objective.path='rectified_flow'` and `objective.prediction='flow'`
  - the SDXL diffusion strategy now branches cleanly between DDPM and RF without splitting into a second giant strategy tree
  - SDXL RF training now uses the direct velocity target (`noise - latents`) while reusing the shared RF runtime for timestep / sigma / weighting assembly
  - SDXL RF sample generation now reuses the shared discrete-flow sampler path instead of pretending DDPM schedulers are universal
- Sampling orchestration now has a first backend seam:
  - `library/training/sample_generation.py` now owns a normalized sampling request plus backend routing instead of assuming every family uses a local latent-returning pipeline with repo-owned scheduler setup
  - SD / SDXL DDPM currently flow through the new seam via a local-pipeline adapter, while SD3 and the SDXL RF path now use model-family backend objects
  - this does not settle the final Diffusers-vs-local pipeline story yet, but it gives the repo one place to host both without forcing every backend into the old `pipeline(...) -> latents -> latents_to_image(...)` contract
- The CLIP-family tokenization split is now explicit enough to stop re-litigating during SD3 work:
  - shared CLIP prompt-tokenization behavior now lives in `library/strategies/shared/clip/tokenization.py`
  - shared Hugging Face tokenizer bootstrap now lives in `library/models/sd/tokenizer.py`
  - SD / SDXL now layer their family-specific tokenization assembly on those shared seams, while SD3 only reuses the bootstrap side until more of its CLIP+T5 behavior proves to be genuinely shared
- The CLIP-family model-preparation split is now a bit cleaner too:
  - shared CLIP text-encoder gradient-checkpointing and FP8 embedding-restore helpers now live in `library/strategies/shared/clip/model_preparation.py`
  - SD / SDXL reuse that shared CLIP helper directly
  - SD3 now also reuses the CLIP branch there for encoder indexes `0` and `1`, while keeping T5-specific model-preparation behavior local
- The first RF config-ownership cleanup is now in place:
  - RF loss-weighting knobs and RF-local time-sampling parameters now have typed config homes under `timestep` instead of only being read as ad hoc `model` fields
  - the SD3 sampling-side `sample_flow_shift` default now has a typed home under `output.sampling`
  - the active SD3 path now reads those typed config homes directly instead of carrying compatibility resolver shims
  - the training-time RF sampling surface is a bit cleaner now too: `logit_normal` is the canonical logit-family sampler, `training_shift` remains as the post-sampling warp, and the redundant `shift`/`logit_scale` pair is gone from the active config surface
- The first loss-weighting ownership cleanup is now in place too:
  - DDPM-only post-loss weighting helpers now live under `library/objectives/ddpm.py` instead of sharing a module with generic masking behavior
  - generic mask application now has its own `library/losses/masking.py` helper
  - SD / SDXL use the DDPM objective-owned post-processing seam directly, while SD3 now imports only the generic masking path
- The first Huber-threshold cleanup is in place too:
  - `get_huber_threshold_if_needed(...)` now lives in `library/losses/huber.py` instead of sharing a file with the raw loss primitives
  - `library/losses/loss.py` now reads more like the actual generic loss-function home, while the threshold helper keeps the timestep/scheduler-aware pre-loss behavior separate
- `zero_terminal_snr` is now described more honestly too:
  - the active config help, validation warning, and design notes now treat it as DDPM scheduler shaping for noisy-state construction rather than as ordinary loss regularization
- The first prediction-target cleanup is in place too:
  - the active schema now uses explicit `objective.path` and `objective.prediction` fields instead of one overloaded `objective.target` field
  - the active config no longer uses `auto` resolution for the objective layer; path and prediction are now declared explicitly in the current schema/defaults
  - `v_parameterization` is now just a legacy compatibility mirror for `objective.prediction == "v_prediction"`
  - the supported combination matrix is explicit too: DDPM allows `epsilon` / `v_prediction`, while rectified flow currently requires the RF-native `flow` prediction label instead of pretending DDPM-style targets are wired there
  - active checkpoint strategies now choose model-spec `prediction_type` explicitly at the caller boundary, so SD3 no longer inherits the DDPM field just because the shared config still carries the legacy mirror
  - DDPM now owns one shared prediction-type mapping for training-target construction and sample-time scheduler setup, instead of repeating the same boolean branch in SD / SDXL diffusion and sampling paths
- Objective-level RF metadata has started moving out of SD3 strategy ownership too:
  - shared training metadata now adds the current RF metadata fields for the active RF consumer
  - `library/strategies/sd3/checkpointing.py` now keeps only SD3-specific attention-mask metadata instead of also owning RF timestep-weighting fields
- The first explicit objective/runtime ownership seam is now in place:
  - `library/objectives/` now owns the active objective definitions plus a small runtime bundle/factory surface
  - trainer runtime initialization now resolves one objective owner instead of wiring scheduler/timestep/loss-modifier pieces independently
  - DDPM scheduler construction now lives under `library/objectives/ddpm.py`, and the RF training helpers now live under `library/objectives/rectified_flow.py`
  - `library/training/noise_utils.py` now keeps only reusable noise-regularization helpers instead of also owning scheduler setup
  - the trainer/strategy seam is now cleaner too: `Trainer` stores one `objective_runtime` bundle, diffusion/validation contracts take that bundle directly, and the active path no longer keeps a DDPM-shaped `trainer.noise_scheduler` compatibility split after objective selection
  - `ObjectiveRuntime` now carries only shared runtime metadata plus objective-owned timestep/loss-modifier state, while DDPM and RF each expose their own typed runtime subclasses for path-specific data
  - the follow-up polish is in too: DDPM-only `alphas_cumprod` metadata moved back down into `DDPMObjectiveRuntime`, and the runtime batch-feedback hook now uses `update_from_batch(...)` instead of the vaguer `observe(...)`
  - RF runtime assembly is now honest: `RectifiedFlowObjective.build_runtime()` no longer inherits DDPM scheduler setup, and DDPM-only EDM2 weighting now fails fast if someone tries to enable it on the RF path
  - RF ownership now reaches the batch-state builder too: `RectifiedFlowObjectiveRuntime` stores the active RF timestep/weighting config and now assembles RF `timesteps`, `sigmas`, noisy model input, and loss weighting for SD3 instead of leaving that construction inside the SD3 diffusion strategy
  - the active SD3 target contract is now cleaner too: generic RF batch-state assembly no longer picks a universal RF target, and the SD3 strategy now owns its paper-style direct velocity target (`noise - latents`) locally instead of supervising a projected clean latent through the shared RF runtime
- The first RF sampling-helper extraction is now in place too:
  - discrete-flow sigma/timestep sampling helpers now live in `library/pipelines/flow.py`
  - `library/strategies/sd3/sampling.py` now keeps SD3 sampling orchestration while importing the reusable discrete-flow math from that pipeline-side module
  - this is a practical pipeline-side home for now, not a claim that the final inference/runtime structure is settled
- Weighted prompt support is now treated as an explicit optional capability rather than a fake universal requirement:
  - the required tokenization/text-encoding facets keep the non-weighted training/runtime path
  - `WeightedPromptStrategy` owns the paired weighted tokenization / weighted encoding seam used by SD / SDXL
  - SD3 no longer pretends to implement weighted prompting until that capability is actually present
- Conditioning is now treated as a first-class strategy concern rather than staying implicit diffusion glue:
  - `ConditioningStrategy.resolve_conditioning(...)` now exists on the required base contract
  - SD / SDXL / SD3 each have a real family `conditioning.py` facet owning cached/live/merge conditioning resolution
  - diffusion and validation now depend on that seam instead of private `_get_text_conds()` helpers
  - family payload shapes still stay local, so the facet starts with one high-level responsibility rather than overcommitting helper-level conventions
- SD3 local text-conditioning shape is now clearer without forcing a new cross-family abstraction:
  - added named SD3-local payloads for token ids/masks and encoded text conditioning
  - SD3 tokenization / encoding / diffusion / denoiser / sampling / caching internals now use those named payloads instead of positional six-item tensor lists
  - the dataloader/cache dict boundary stays unchanged, so this remains a local SD3 normalization step rather than a broader conditioning rewrite
- The active SD / SDXL CLIP helper cleanup is now settled enough to stop blocking SD3 prep:
  - family text-encoding behavior now lives in `library/strategies/*/encoding.py`
  - CLIP-family tokenization helpers no longer pretend to be SD model-layer code
  - the shared CLIP-family tokenization behavior now lives under `library/strategies/shared/clip/`, while tokenizer bootstrap/loading lives under `library/models/sd/tokenizer.py`
  - prompt-attention parsing is back to a single canonical implementation in `library/data/prompt_utils.py`
- The active launcher surface for PEFT/fine-tune is now config-driven: a canonical root `train.py` builds `mode` and `model.model_type` through small factories instead of requiring one near-duplicate active script per mode/model combination. Textual inversion still sits outside that launcher until its runtime path is migrated.
- The temporary launcher compatibility wrappers have been removed again; the active entry surface is just the root `train.py`, which now owns the small amount of launcher orchestration directly.
- Benchmark runs now use that same root launcher path too, so benchmark smoke coverage exercises the active config-driven entry surface instead of separate script-specific launch wiring.
- The root launcher now relies on an explicit `--config-name`, while the internal full-schema fallback lives at `configs/_defaults/default.yaml` without choosing a mode or model family.
- Sampling cadence conflicts now fail fast instead of silently preferring epoch cadence.
- Active config composition now cleanly matches the shared dataclass schema.
- Sampling config now supports inline/default generation parameters plus `sample_prompt_file`.
- Validation config now also fails fast for invalid `validation_split`, non-positive `max_validation_steps`, malformed or empty `validation_timesteps`, conflicting `val_data_dir` + `validation_split`, and scheduled validation with no validation data source instead of leaving those errors to strategy/runtime behavior.
- EDM2 importance-weighting conflict handling now runs through the active nested config path, and the dormant `laplace_timestep_sampling` toggle now fails fast instead of silently acting unsupported.
- The active training path now treats EDM2 as one bundled runtime sidecar instead of several loose trainer fields, which makes the loop/checkpoint/logging wiring easier to follow.
- The active training path now also has a trainer-owned `TimestepRuntime` seam in `library/timesteps/`: trainer owns timestep schedule state and adaptive sampler lifecycle, SD / SDXL strategies no longer update `la_sampler`, and runtime/logging no longer depend on config mutation to know which sampler is active.
- The active timestep sampler surface is now intentionally small: `uniform`, `shift`, `log_snr_uniform`, and `adaptive_log_snr`. The older one-off adaptive samplers were removed from the active path, and config validation now rejects their names instead of leaving them half-supported.
- `loss.edm2` now has one structured feature-owned config surface (`enabled`, `optimizer.*`, `importance.*`, `visualization.*`) instead of a long flat list of `edm2_loss_weighting_*` fields.
- The shared diffusion strategy contract now returns base loss state through the shared `BatchLossOutput`, and the trainer builds either a `NoOpLossModifier` or a concrete EDM2 modifier through one generic post-loss seam instead of threading `edm2_model` through SD / SDXL diffusion code.
- Active checkpoint/logging/plotting paths now also use generic loss-modifier hooks, and the old `trainer.edm2` / `edm2_runtime.py` compatibility layer is gone from the active path.
- EDM2 internals now live under the `library/losses/edm2/` package with separate factory / plotting / validation helpers, and modifier metrics now follow a stricter namespaced contract.
- EDM2 now has a first active SDXL PEFT preset/example surface (`presets/sdxl_peft_edm2` and the runnable `examples/edm2_sdxl_peft`) instead of being discoverable only from defaults and source.

---

## Code Quality TODOs

- [ ] **Strategy system follow-up** — The large base-strategy cleanup is mostly done. Remaining work is narrower:
  - SD / SDXL concrete strategy cleanup against the current base contract
  - naming/organization review for `library/strategies/base/contracts.py`
  - later concrete/compatibility `unet` -> `denoiser` cleanup where it still makes sense
  - further `base/`-to-`models/` ownership cleanup where model-specific behavior still sits too high
  - `_deprecated` and `copy` reference files should not receive normal refactor work
  - See `docs_design/strategy_system_followup.md`, `docs_design/strategy_base_decision.md`, and `docs_design/strategy_remaining_facet_audit.md`
- [x] **Training orchestration hardening follow-up** — Shared training orchestration now has explicit epoch outcomes, scoped shared lifecycle helpers, shared eval-side execution for startup and step-triggered actions, and clearer trainer startup/finalization sequencing. See `docs_design/training_orchestration_followup.md` and `docs_design/training_orchestration_refactor_plan.md`.
- [x] Timestep sampling reimplementation — the active path and compatibility wrappers now go through a trainer-owned `TimestepRuntime`, the sampler surface is narrowed to the intended set, and the sampler-backed runtime interface no longer leaks shift-specific knobs through every sampler API.
- [ ] **`training_plots.py`** - Functions access multiple sub-configs (`cfg.output.saving`, `cfg.output.logging`, `cfg.timestep`) - acceptable for orchestration functions but could be cleaner
- [ ] **Delete legacy training wrappers** (after legacy script deprecation) - Once `*_finetune.py` and `*_textual_inversion.py` scripts are migrated to new data pipeline, delete:
  - `library/training/sd_sample_generation.py`
  - `library/training/sdxl_sample_generation.py`
  - `library/training/sd_checkpointing.py`
  - `library/training/sdxl_checkpointing.py`
  - Strategies now call `sample_images_common()` directly; checkpointing logic can be inlined into strategies when legacy scripts are removed.
- [ ] Investigate naming conventions and possible drifts in the objective class and runtime layers

### Near-Term Follow-up

- [ ] **SD / SDXL strategy cleanup against the current contract** — The large structural split is done; the remaining work is narrower cleanup and architecture follow-up now that the base contract has settled.
  - SD tokenization / text-encoding / caching ownership now lives in the self-titled facet files, and the remaining SD contract-owned concerns have been split into `sd/loading.py`, `sd/model_preparation.py`, `sd/checkpointing.py`, `sd/sampling.py`, `sd/denoiser.py`, `sd/diffusion.py`, and `sd/validation.py`.
  - `sd/training.py` is now reduced to strategy assembly and init/wiring, mirroring the SDXL composition pattern.
  - SDXL tokenization / text-encoding / caching ownership now also lives in the self-titled facet files, and the remaining SDXL contract-owned concerns have been split into `sdxl/loading.py`, `sdxl/model_preparation.py`, `sdxl/checkpointing.py`, `sdxl/sampling.py`, `sdxl/denoiser.py`, `sdxl/diffusion.py`, and `sdxl/validation.py`.
  - `sdxl/training.py` is now reduced to strategy assembly and init/wiring.
  - SDXL training-time text conditioning now routes through the strategy tokenization / encoding seam instead of calling model helpers directly from `sdxl/diffusion.py`, which keeps the diffusion facet aligned with the settled strategy contract.
  - The new conditioning facet now gives the concern a real home, but the return-shape convention is still intentionally loose and family-local while more model families are ported.
  - Prompt weighting / weighted captions still likely want to become a shared concern rather than a model-by-model accumulation of special cases, especially once cache-policy expectations are made explicit.
- [x] **Timestep runtime redesign** — The trainer-owned timestep runtime landed and the active sampler/runtime cleanup is complete for the current SD / SDXL path.
  - `library/timesteps/` now owns active runtime state and adaptive sampler lifecycle for the main trainer path.
  - The active sampler surface is intentionally small: `uniform`, `shift`, `log_snr_uniform`, and `adaptive_log_snr`.
  - The sampler-backed runtime interface now only passes shared sampling inputs; shift-only knobs like `logit_scale` and `training_shift` remain owned by the explicit `shift` path instead of leaking through every sampler API.
  - Benchmark-backed smoke configs exist under `configs/tests/` for both `adaptive_log_snr` and `log_snr_uniform`, and both completed end-to-end manual smoke runs on the SDXL PEFT test setup.
- [ ] **EDM2 presence follow-up** — The runtime/config seam is cleaner now and the repo has an initial SDXL PEFT preset/example, but the feature still needs real docs and clearer guidance on when to use it.
- [ ] **Conditioning architecture follow-up** — Pressure-test the new `ConditioningStrategy` seam against more model families and decide whether any sub-conventions under `resolve_conditioning(...)` are mature enough to standardize.
- [ ] **Prompt weighting / weighted captions review** — Decide whether weighted captions should become an active shared concern and where prompt-weight parsing/application should live.
- [ ] **Dashboard / logging system rework** — Fold the live plotter into a broader dashboard/logging system instead of treating it as a side system.
- [ ] **Repo layout review** — Re-check whether `library/` / `scripts/` placement, and potentially the entry-script layout, still fit the current architecture.
- [ ] **LyCORIS vendor / integration pass** — Treat LyCORIS as a vendor/integration ownership question rather than an external dependency question, since adapter breakdown follow-up depends on tighter ownership and easier modification.
- [ ] **Custom optimizer vendor / integration pass** — Treat customized optimizers as a vendor/integration ownership question rather than an external dependency question, so optimizer behavior can be evolved in-repo as the training stack settles.
- [ ] **Single launch-script review** — Decide whether the current entry surface should grow a unified launcher once the config/layout story is stable enough to support it cleanly.
- [ ] **Future conditioning/data-flow experiments** — Later exploration area for better caption mutation, TE caching, on-the-fly CPU encoding, queues, async handoff, and related conditioning/data-flow improvements once the current building blocks are settled.

---

## Next Phase After Stabilization

Once the current stabilization / cleanup list above is tied off, the roadmap should shift from architecture settling to capability expansion.

- [ ] **Gradual new model implementations** — Add new model families incrementally on top of the current trainer / strategy foundation instead of trying to land a large multi-model rewrite all at once.
- [ ] **New training approaches** — Open the next wave of work around new training methods once the active SD / SDXL path is stable enough to serve as the reference implementation.
- [ ] **Rectified flow / RF support** — Evaluate and implement RF-style training/runtime support when the timestep / conditioning seams are mature enough to carry another training formulation cleanly.
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

- [ ] Model download/load for training from huggingface

- [ ] Selective activation checkpointing

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
- [ ] Metadata system
- [ ] Kahan summation / stochastic rounding / optimal transport check in reference repos

---

## Training Mode Extensibility

### Problem

All phase functions in `library/training/phases/` are typed as `trainer: Trainer`. The `TrainingMode` protocol and `PeftMode` implementation enable pluggable training modes without changing phase code.

### What Actually Differs (PEFT vs Full Fine-Tuning)

| Concern                     | PEFT (LoRA/Adapter)                          | Full Fine-Tuning                     |
| --------------------------- | -------------------------------------------- | ------------------------------------ |
| **Trainable params**        | `adapter.parameters()` + optionally TE       | `unet.parameters()` + optionally TE  |
| **Model prep**              | Create adapter, `apply_to()`, freeze base    | `requires_grad_(True)` on base model |
| **Checkpoint save**         | Adapter state dict                           | Full model state dict                |
| **`accelerator.prepare()`** | Wraps adapter                                | Wraps UNet directly                  |
| **Caching**                 | Identical                                    | Identical                            |
| **Training loop**           | Identical (strategy handles `process_batch`) | Identical                            |

### Phase Reusability

- `caching.py` — **100% shareable**
- `training_loop.py` — **~95% shareable** (checkpoint calls go through `trainer.save_checkpoint()`)
- `optimizer.py` — **~80% shareable** (param groups and `accelerator.prepare()` wrapping differ)
- `model_prep.py` — **PEFT-specific** (adapter creation is inherently a PEFT concept)

### Design Options (Open)

**Option A: One Trainer + `TrainingMode` plugin** (Lightning/Transformers style)

- Rename `PeftTrainer` → `Trainer`, add `trainer.mode: TrainingMode`
- `TrainingMode` protocol has ~4 methods: `prepare_models`, `trainable_params`, `prepare_with_accelerator`, `save_checkpoint`
- Phases call `trainer.mode.*` for divergent operations
- Two axes: `strategies` = model family (SDXL/SD/Flux), `mode` = training mode (PEFT/fine-tune)
- Pro: One class to understand, config-driven. Con: Extra indirection through `mode.*`

**Option B: Multiple trainer classes + `TrainerProtocol`**

- Keep `PeftTrainer`, add `FineTuneTrainer`, both satisfy a shared protocol
- Phases type against the protocol
- Pro: Each trainer is self-contained. Con: Large protocol surface, potential duplication

> [!IMPORTANT]
> Key constraint: this repo prioritizes readability and ease of modification over abstraction. Whatever pattern is chosen must feel intuitive when adding new features. Decision deferred until full fine-tuning support is actively being built.

### Current State

- `Trainer` + `TrainingMode` is the active extensibility pattern.
- `PeftMode` and `FineTuneMode` are both in place.
- Shared phases already route divergent behavior through trainer/mode hooks.
- The remaining work here is no longer the extraction itself; it is follow-up cleanup like optimizer-group features, orchestration hardening, and removing legacy wrappers.

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
