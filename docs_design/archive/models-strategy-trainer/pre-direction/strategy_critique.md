# Critique from agent after review of strategy layer

I did this code-only. I did not read the strategy README or repo docs. One note: the IDE tab `library/strategies/base/operations.py` does not exist on disk in this checkout, so I could not inspect it.

**High-Level Take**

The strategy layer is doing useful work, but it is no longer just a strategy layer. It is a model-family plugin system disguised as a single inherited “strategy” object. The current shape works because the repo has strong conventions and only a few families, but the abstraction is carrying too many axes at once: model loading, tokenizer ownership, encoding, cache backend creation, denoiser invocation, objective/loss math, validation, sample generation, precision policy, checkpoint metadata, and full-model serialization.

The blunt version: `TrainingStrategy` is a god interface split across files.

**What Is Good**

The family split is basically right. `sd`, `sdxl`, and `sd3` each compose loading/tokenization/conditioning/diffusion/etc. from separate files, then aggregate them in the family class, e.g. [sd/training.py](/mnt/d/Projects/sd-scripts/library/strategies/sd/training.py:19), [sdxl/training.py](/mnt/d/Projects/sd-scripts/library/strategies/sdxl/training.py:20), and [sd3/training.py](/mnt/d/Projects/sd-scripts/library/strategies/sd3/training.py:17). That keeps concrete family logic discoverable.

The `StrategyContext` is a clever escape hatch for adapter runtimes that need denoiser-call facts, especially TLora timestep masks. The publication point in denoiser calls, e.g. [sd/denoiser.py](/mnt/d/Projects/sd-scripts/library/strategies/sd/denoiser.py:39), connects cleanly to [tlora/runtime.py](/mnt/d/Projects/sd-scripts/library/adapters/methods/peft/tlora/runtime.py:104).

SD3’s typed payloads are the best direction in the layer. `Sd3TokenizedText` and `Sd3TextConditioning` in [sd3/encoding.py](/mnt/d/Projects/sd-scripts/library/strategies/sd3/encoding.py:12) are much healthier than anonymous ordered tensor lists.

**Main Critique**

`TrainingStrategy` composes eleven base classes in [contracts.py](/mnt/d/Projects/sd-scripts/library/strategies/base/contracts.py:554). That means every model family is treated as supporting the same huge lifecycle surface, whether each capability is truly required, optional, mode-specific, or only needed by one training path.

The contracts are nominally abstract, but semantically loose. Almost every boundary is `Any`, long positional argument lists, `list[Any]`, or tuple/list ordering. See `process_batch` in [contracts.py](/mnt/d/Projects/sd-scripts/library/strategies/base/contracts.py:388), `call_denoiser` in [contracts.py](/mnt/d/Projects/sd-scripts/library/strategies/base/contracts.py:418), and `calculate_val_loss` in [contracts.py](/mnt/d/Projects/sd-scripts/library/strategies/base/contracts.py:351). These signatures document intent but do not protect the architecture.

The trainer leaks deeply into strategy concerns, and strategy concerns leak back into trainer phases. The trainer pulls tokenizers directly from `strategies.tokenizers` [trainer.py](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:238), calls model loading [trainer.py](/mnt/d/Projects/sd-scripts/library/training/runners/trainer.py:331), delegates batch math [training_loop.py](/mnt/d/Projects/sd-scripts/library/training/phases/training_loop.py:477), and later passes the whole trainer back into checkpoint serialization [contracts.py](/mnt/d/Projects/sd-scripts/library/strategies/base/contracts.py:310). That is a circular dependency in practice.

Diffusion strategy currently owns too much objective/loss policy. SD, SDXL, and SD3 all repeat the same broad batch pipeline: prepare latents, resolve conditioning, sample objective inputs, call denoiser, compute loss, apply masks/weights/multipliers. Compare [sd/diffusion.py](/mnt/d/Projects/sd-scripts/library/strategies/sd/diffusion.py:109), [sdxl/diffusion.py](/mnt/d/Projects/sd-scripts/library/strategies/sdxl/diffusion.py:143), and [sd3/diffusion.py](/mnt/d/Projects/sd-scripts/library/strategies/sd3/diffusion.py:122). The family differences are real, but the control flow duplication suggests the wrong owner.

Caching is probably the largest misplaced mass. Strategy-facing cache factories are reasonable, but full cache backend implementations in files like [sdxl/caching.py](/mnt/d/Projects/sd-scripts/library/strategies/sdxl/caching.py:31) and [sd3/caching.py](/mnt/d/Projects/sd-scripts/library/strategies/sd3/caching.py:22) make `library/strategies` partly a data pipeline package. The line counts bear that out: `sdxl/caching.py` is 647 lines, `sd/caching.py` 477, `sd3/caching.py` 401.

Optional capabilities are awkward. `WeightedPromptStrategy` is correctly excluded from the base `TrainingStrategy`, but SD and SDXL inherit it directly while consumers still call `tokenize_with_weights` on concrete strategy objects. That works, but capability detection is implicit and not represented in the main factory or runtime model.

The static registry in [factory.py](/mnt/d/Projects/sd-scripts/library/strategies/factory.py:13) is fine for now, but it hardcodes model-family discovery and makes strategy creation config-coupled. It will get worse with SD3 variants, Flux-like families, video, ControlNet branches, or external plugins.

**Ideal Shape**

I would reshape this into a typed `ModelFamilyPlugin`, not a single inherited `TrainingStrategy`.

At the top level:

```python
@dataclass
class ModelFamilyPlugin:
    family: ModelFamilyId
    tokenizer: TokenizationFacet
    text_encoder: TextEncodingFacet
    conditioning: ConditioningFacet
    denoiser: DenoiserFacet
    latents: LatentFacet
    checkpointing: CheckpointingFacet
    sampling: SamplingFacet | None = None
    weighted_prompts: WeightedPromptFacet | None = None
```

Then the trainer talks to a stable run context, not a long list of arguments. For example: `BatchRequest`, `DenoiserRequest`, `LoadedModelBundle`, `TextConditioning`, `CheckpointSaveRequest`. That would remove a lot of positional coupling and make tests simpler.

Objective ownership should move upward or sideways into the objective layer. The family plugin should answer: “how do I encode latents?”, “how do I build denoiser kwargs?”, “what conditioning object do I need?” The objective layer should own DDPM/RF timestep sampling, target construction, weighting, and common loss post-processing.

I would move cache backend implementations closer to `library/data`, with family-specific codecs registered by the model-family plugin. Strategy should not be the home for image preprocessing, safetensors cache metadata parsing, and cache validation logic.

I would also formalize `StrategyContext` as either an explicit `DenoiserExecutionContext` passed through adapter hooks, or keep the contextvar but name it as a public runtime signal. Right now it is powerful but hidden: adapters depend on it, while most of the strategy contracts do not acknowledge that this ambient channel exists.

**Bottom Line**

This layer is a successful transitional architecture. It carved model-family behavior out of the trainer and deprecated scripts, which is valuable. But it has reached the point where the abstraction should stop growing by inheritance.

The ideal next state is not “clean up the existing ABCs.” It is to demote `TrainingStrategy` from the central object into a compatibility façade, introduce typed facet objects behind it, move objective/loss control flow out of family diffusion classes, and push cache backends toward the data layer. That would make future model families less expensive and make the current SD/SDXL/SD3 differences sharper instead of spread across huge method signatures.