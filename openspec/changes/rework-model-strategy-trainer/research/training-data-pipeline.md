**This is a non-main topic that will need deeper research when the topic becomes current. It is also one of the topics that specifically set up the goals for the current rework**

# Training Data Pipeline

This one may actually have **more architectural impact than it looks like**.

The central statement I'd start with is:

> **The training data pipeline is part of training state, not merely an iterator that happens to feed the Trainer.**

## 1. There are several representation boundaries before the model ever sees a sample

For your current domain, something like:

```text
source file
   ↓
decoded image/video/audio/text
   ↓
augmentation / crop / frame selection
   ↓
tokenization / VAE / text encoder
   ↓
cached representation
   ↓
packing / bucketing / collation
   ↓
training batch
```

Every arrow potentially changes:

```text
representation identity
cacheability
randomness
cost
validity
```

We've already found concrete cases.

Anima caches **Qwen outputs + T5 IDs** but deliberately leaves its trainable LLM adapter after the cache boundary.

FLUX.2 has VAE output followed by model-specific normalization/packing.

So:

> **"Cached latent" or "cached text embedding" is not enough information. A cache must identify exactly which representation boundary it contains.**

## 2. Cache validity depends on producers, not consumers

This is worth making explicit in the note.

Suppose:

```text
image
 ↓
crop
 ↓
VAE
 ↓
normalization
 ↓
packing
 ↓
DiT
```

If you cache here:

```text
VAE ─→ [ CACHE ] ─→ normalization
```

then changing:

```text
DiT weights
```

doesn't invalidate it.

Changing:

```text
VAE weights
```

does.

Changing a crop performed *before* the VAE does too.

Changing something downstream does not.

The correct dependency model is basically:

```text
cache artifact
    depends on
all deterministic/stochastic producers upstream of that boundary
```

This is much stronger than hard-coding:

```text
if train_vae:
    disable_latent_cache
```

because some future representation pipelines may have several trainable stages.

### And stochastic preprocessing matters

If random crop occurs before caching:

```text
random crop → VAE → cache
```

then the cache **freezes one realization of that augmentation**.

If crop occurs after caching, it may remain dynamic—if that representation permits it.

So:

> **Moving a cache boundary can alter the effective training distribution even when values are mathematically valid.**

That's a nasty but important training consideration.

## 3. Batch construction can change model semantics

Packing is a good example.

Transformers now explicitly supports padding-free packed training:

```text
sample A
sample B
sample C

        ↓ pack

[A tokens][B tokens][C tokens]
```

but the model must receive sample boundaries so attention does not leak between logically unrelated examples. This is especially nontrivial for linear/recurrent attention architectures such as Qwen3-Next/Qwen3.5, where simple `position_ids` tricks do not establish the required boundaries. ([Hugging Face][11])

So:

> **Collation is not necessarily shape-only preparation. It can construct execution semantics such as attention masks, position identities, loss masks and sequence boundaries.**

That suggests dataset item vs training example vs model sequence should stay distinct.

```text
dataset record
    ≠
logical training example
    ≠
packed model sequence
    ≠
microbatch
```

## 4. Bucketing is also training behavior

For image/video training:

```text
same source distribution
    ↓
resolution buckets
aspect buckets
frame-count buckets
audio-duration buckets
```

affect which samples coexist in batches.

That's normally sold as efficiency, but it can affect:

```text
gradient composition
effective batch-size variance
padding
resolution distribution
augmentation options
per-step compute
```

So I'd treat bucket assignment as a **sampling/batching policy**, not merely a dataloader optimization.

For multimodal data the appropriate budget may not even be “number of examples”:

```text
8 short images
≈?
1 long 4K video
≈?
30k-token text sequence
```

Modern LLM training often reasons in **tokens per batch** rather than sample count; multimodal trainers may need analogous latent/token/frame budgets.

## 5. Distributed data state has topology of its own

Within one model-parallel replica, TP/PP/CP ranks usually need corresponding views of the **same logical training examples**, while different DP replicas consume different examples.

So:

```text
DP rank
    determines sample partition

TP / PP / CP rank
    generally does NOT mean independent samples
```

This is a subtle reason the data system shouldn't just see:

```text
world_size
rank
```

without knowing the relevant data-parallel group.

Megatron's data loader does exactly this sort of distributed-aware construction. Once initialized, data-parallel ranks consume disjoint portions of a deterministic permutation. ([NVIDIA Docs][12])

### Resume makes this more important

Megatron Energon explicitly checkpoints **dataloader stream position** together with training state. Its checkpoint includes enough loader/dataset/RNG state to resume the same stream rather than starting approximately around the same place. ([NVIDIA Docs][13])

So:

> **`global_step` is not sufficient to reconstruct data position.**

Especially once you have:

```text
streaming
shuffle buffers
multiple workers
mixtures
packing
dynamic batches
augmentations
```

A resumable run may need a bona fide **data-state artifact**.

## 6. Streaming and random-access datasets are fundamentally different execution models

Megatron's classic GPT dataset can build deterministic lookup indices:

```text
document index
sample index
shuffle index
```

and cache those on disk. ([NVIDIA Docs][14])

Energon, on the other hand, supports streaming WebDataset-style multimodal sources and retains stream position/state. ([NVIDIA Docs][13])

That gives two distinct models:

```text
random access:
sample[k] is reconstructible

streaming:
next() depends on evolving stream state
```

A generic trainer shouldn't require all datasets to support one model.

## 7. Dataset caches themselves can be distributed preparation artifacts

Megatron's large-scale loader is another useful example.

For huge jobs, building dataset indices at launch can make every rank wait while one rank creates them. Their recommended flow is to **pre-build the dataset cache**, store shared index files, then have workers memory-map them during training. ([NVIDIA Docs][12])

That is very relevant to the preparation work we've been discussing:

```text
accepted dataset requirements
           ↓
prepare index/cache artifact
           ↓
publish coherent prepared data state
           ↓
training ranks consume it
```

It's basically the data-side equivalent of model preparation.

And note that the cache itself depends on things such as sequence length and expected sample counts/world setup. ([NVIDIA Docs][12])

So:

> **Not all caches are sample feature caches. Some are structural indices required to interpret or schedule the dataset.**

## 8. Mixtures are executable sampling policies, not just percentages

Megatron's `BlendedDataset` is a nice concrete example.

Given datasets and weights, it actually builds:

```text
dataset_index[k]
sample_index[k]
```

so each logical training index deterministically resolves to a particular source dataset and sample. It attempts to keep the realized sampling distribution close to the configured weights rather than independently rolling a random dataset every call. ([NVIDIA Docs][14])

So:

```text
mixture config
    ↓
sampling policy
    ↓
actual sample stream
```

The configured weights alone are not the full execution state.

This matters a lot once the weights change during training.

## 9. Curriculum should be considered a state machine over the whole job

This is the biggest curriculum insight we've gotten from the LLM research.

A curriculum isn't necessarily:

```text
epoch 0-10: easy data
epoch 11-20: hard data
```

Qwen3, for example, changes its corpus emphasis between general training, reasoning-oriented data, and long-context data, while also changing sequence length and learning-rate behavior. ([UsionMedia][15])

Qwen3-VL is even stronger evidence: its stages change **data mixture, sequence length and which components are trainable** together. ([arXiv][16])

DeepSeek-V4's long-context training changes sequence length through:

```text
4K → 16K → 64K → 1M
```

while also changing attention behavior during the progression. ([Awesome AI Papers][17])

So I would define curriculum broadly as:

```text
stage
 ├─ data sources / mixture
 ├─ sample-selection policy
 ├─ representation / resolution
 ├─ sequence length
 ├─ batch/token budget
 ├─ loss/objective weighting
 ├─ component trainability
 └─ possibly model execution topology
```

Therefore:

> **Curriculum stage is training state, not merely dataset configuration.**

And a checkpoint around a stage boundary needs to know which stage's coherent set of policies applies.

## 10. Epochs are not a universally meaningful scheduling axis

This is worth putting in because image-training code often assumes them.

Large-scale language training commonly schedules by:

```text
steps
tokens consumed
```

rather than “epochs”.

A blended/streaming corpus may not even have a useful finite epoch boundary.

For video/image datasets with repeated weighted subsets, the meaning gets fuzzy too.

So trainer lifecycle events should probably support something conceptually broader than:

```text
on_epoch_start()
```

as the fundamental training-time coordinate.

Possible coordinates include:

```text
optimizer steps
samples
tokens
frames
latent elements
wall-clock / compute budget
curriculum stage
```

Not that all should be first-class APIs—but **epoch cannot be assumed to be the universal clock**.

---

[11]: https://huggingface.co/docs/transformers/main/padding_free?utm_source=chatgpt.com "Padding-free training · Hugging Face"
[12]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/data-loading.html?utm_source=chatgpt.com "Data Loading at Scale — Megatron Core"
[13]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/megatron_energon.html?utm_source=chatgpt.com "Megatron Energon — Megatron Core"
[14]: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/datasets.html?utm_source=chatgpt.com "datasets package — Megatron Core"
[15]: https://www.usionmedia.com/docs/qwen3_technical_report.html/attachment/qwen3_technical_report?utm_source=chatgpt.com "2025-05-14
Qwen3 Technical Report
Qwen Team
https:"
[16]: https://arxiv.org/abs/2511.21631?utm_source=chatgpt.com "Qwen3-VL Technical Report"
[17]: https://awesome.papernotes.org/en/era5_genai_explosion/2026_deepseek_v4/?utm_source=chatgpt.com "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence - Awesome AI Papers"
