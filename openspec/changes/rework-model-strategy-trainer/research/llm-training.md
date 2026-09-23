# Research topic: LLM training

**Claim labels:** **Observed** = directly established by source. **Inferred** = follows from implementation/report. **Unknown** = examined sources do not establish it.

The seven useful cases are:

| Recipe | What it stresses |
| --- | --- |
| Qwen3.8-Flash-Next pretraining | heterogeneous optimizer ownership, host-resident trainable parameters, MTP, architecture transition |
| DeepSeek-V4-Flash pretraining | MoE routing state, auxiliary losses, context curriculum, dense→sparse attention transition |
| Qwen3-VL pretraining | multimodal alignment, freeze→unfreeze stages, radically changing sequence lengths/data mixtures |
| MiMo-V2.6-Flash mixed RL | rollouts/environments/graders as training participants; asynchronous agent RL |
| MDLM OpenWebText pretraining | masked-token diffusion, stochastic corruption/noise-time sampling, non-causal denoising |
| LLaDA2.0 AR→dLLM conversion | conversion from an AR checkpoint plus block-size/full-sequence diffusion curriculum |
| BD3-LM OpenWebText training | block-autoregressive + within-block diffusion, per-block noise state, adaptive training schedule state |

GLM-5.3 is worth noting too, but there isn't currently enough published detail on the **specific 5.3 post-training recipe** to fill this template honestly. More on that at the end.

---

# Recipe 1 — Qwen3.8-Flash-Next pretraining

**Implementation/report examined:** Qwen3.8-Flash-Next, August 2026 technical report, *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability*.

Primary sources: [technical report on arXiv](https://arxiv.org/abs/2608.30320?utm_source=chatgpt.com) · [official model repository](https://github.com/QwenLM/Qwen3.8-Flash-Next?utm_source=chatgpt.com)

## 1. Components

**Observed:** the language model is a sparse MoE with **125B backbone parameters / 6B active per token**, plus **51B parameters of n-gram embedding tables** and an MTP module. Its blocks use a 3:1 mixture of Gated DeltaNet and global/sparse-attention layers, with a four-branch Gated Residual stream. ([arXiv][1])

Important independently identifiable state includes:

```text
token embedding
n-gram embedding tables
Gated DeltaNet layers
attention / QSA layers
MoE experts
MoE router
Gated Residual projections
LM output head
MTP module
```

**Observed:** the 51B n-gram tables are intentionally capable of living in **host memory** and being asynchronously prefetched rather than residing with the main accelerator weights. ([arXiv][1])

So “parameters belonging to the model” does not imply “parameters permanently resident on the accelerator.”

## 2. Data and one training step

**Observed:** the primary training representation is still a causal token sequence. The model predicts ordinary next tokens while its MoE router selects experts and its n-gram layer retrieves embeddings derived deterministically from local token context.

The released model also includes MTP trained to predict multiple subsequent tokens. The public sources establish that MTP participates in training, though the exact production loss composition/weighting is not fully specified in the model card. ([Hugging Face][2])

Conceptually:

```text
tokens
  │
  ├─ token lookup
  ├─ n-gram lookup
  │
  ↓
hybrid GDN / attention / MoE backbone
  │
  ├─ ordinary next-token prediction
  └─ MTP future-token predictions
```

**Observed:** the n-gram lookup adds capacity without requiring a corresponding dense compute path through those 51B parameters. ([arXiv][1])

## 3. Optimization and runtime

This is where it gets much less generic.

**Observed:** Qwen applies **Muon to two-dimensional weights functioning as linear maps**, while AdamW remains responsible for the token/n-gram embeddings, output head, MoE router and low-rank Gated Residual projections. ([arXiv][1])

So parameter ownership resembles:

```text
Muon
 ├─ attention matrices
 ├─ GDN matrices
 ├─ MoE expert matrices
 └─ other suitable 2D linear maps

AdamW
 ├─ embeddings
 ├─ n-gram tables
 ├─ output head
 ├─ router
 └─ low-rank GR parameters
```

**Observed:** fused matrices must actually be **split before Muon orthogonalization**, because orthogonalizing the fused object would mix unrelated subspaces. The Muon implementation also repartitions distributed gradient buffers according to estimated orthogonalization cost rather than parameter count and captures the fragmented optimizer step with CUDA graphs. ([arXiv][1])

That is substantially more specific than “give Trainer an optimizer.”

## 4. Structural changes during a run

**Observed.**

The model starts with global/full-attention layers, but during **continued pretraining those layers are replaced by Qwen Sparse Attention (QSA)**. ([arXiv][1])

So there is a real transition:

```text
earlier training
GDN + full attention
        ↓
continued pretraining
GDN + QSA
```

This isn't merely changing an LR or sequence length. A trainable model subsystem changes implementation/behavior while the larger model identity persists.

Qwen also reports that the new architecture + Muon shifts the optimal batch size and learning rate enough that they discard conventional batch-size warmup and begin directly at the target batch size. ([arXiv][1])

## 5. Outputs and restoration

**Observed:** the deployable model includes not only the conventional backbone but the n-gram embedding state and MTP subsystem. The n-gram tables remain semantically part of the model despite being suitable for host-memory placement. ([Hugging Face][2])

**Unknown:** the report does not publish a complete production checkpoint/resume schema covering optimizer shards, dataloader progress, RNG, etc.

## 6. Design pressure

> **Can different parameter groups inside one model require different optimization algorithms based on their semantic/mathematical role rather than merely different learning rates?**

And:

> **Can trainable model state have different placement/lifetime requirements—including host-resident trainable tables—without equating model ownership with device residency?**

And the nasty one:

> **Can a training recipe replace one internal execution mechanism with another during continued training while retaining the surrounding model and optimizer state?**

---

# Recipe 2 — DeepSeek-V4-Flash pretraining

**Implementation/report examined:** DeepSeek-V4-Flash pretraining from *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, April 2026.

Primary source: [DeepSeek-V4 technical report](https://arxiv.org/html/2606.19348?utm_source=chatgpt.com).

## 1. Components

**Observed:** V4-Flash is a **284B-total / 13B-active MoE**. It combines DeepSeekMoE, hybrid Compressed Sparse Attention / Heavily Compressed Attention, Manifold-Constrained Hyper-Connections, and MTP. ([arXiv][3])

Its MoE subsystem itself contains several different kinds of state:

```text
shared expert
routed experts
router scoring
load-balancing correction bias
early hash-routed MoE layers
```

The first several MoE layers use deterministic **hash routing based on token IDs**, whereas later ones use learned routing. ([arXiv][4])

## 2. Data and one training step

At heart, this is still causal LM training.

But the per-step objective is not just one CE scalar.

**Observed:** ordinary language-model training coexists with:

```text
next-token objective
+ MTP auxiliary objective
+ small sequence-wise MoE balance loss
```

The MTP loss weight is **0.3** through most of pretraining and drops to **0.1** when LR decay begins. ([arXiv][4])

The training sequence length itself follows a curriculum:

```text
4K
 ↓
16K
 ↓
64K
 ↓
1M
```

([arXiv][4])

## 3. Optimization and runtime

**Observed:** as with Qwen3.8-Next, this isn't one homogeneous optimizer rule.

DeepSeek uses Muon for most parameters but AdamW for:

```text
embedding module
prediction head
all RMSNorm weights
```

V4-Flash trains on 32T tokens, reaching a batch size of **75.5M tokens**; AdamW and Muon have separate update semantics despite operating in the same training run. ([arXiv][4])

The report also describes specialized distributed infrastructure for Muon, expert parallelism, contextual parallelism and fine-grained activation checkpointing. ([arXiv][4])

## 4. Structural changes during a run

This recipe is an excellent counterexample to a static training graph.

**Observed:** DeepSeek trains with **dense attention for the first 1T tokens**. Sparse attention is introduced only later, at the 64K-context stage. Before normal sparse training begins, the CSA lightning indexer itself gets a short warmup stage. ([arXiv][4])

So:

```text
initial training
dense attention

        ↓

long-context stage
warm up sparse-attention indexer

        ↓

remaining training
CSA/HCA sparse configuration
```

**Observed:** DeepSeek also introduces a *conditional* stability mechanism called **Anticipatory Routing**. During a detected loss spike, routing decisions for the current step are computed from historical parameters rather than the current network; the system later reverts to ordinary synchronous routing. ([arXiv][4])

That's unusually interesting for Trainer architecture because the same MoE router can change **update/execution semantics conditionally in reaction to training state**.

## 5. Outputs and restoration

**Observed:** pretraining yields the V4 base model containing MoE/router state, hybrid-attention/indexer parameters, MTP and mHC state.

A later post-training phase performs FP4 quantization-aware training of MoE expert weights and the sparse-attention indexer QK path; that is a separate training stage, not something we should silently treat as part of this pretraining recipe. ([arXiv][4])

**Unknown:** the paper does not expose the exact production resume/checkpoint object graph.

## 6. Design pressure

> **Can a recipe gradually change context geometry and then activate/train new attention behavior only after reaching a particular curriculum stage?**

And:

> **Can execution policy react dynamically to optimization health—for example activating a delayed-routing mechanism after loss-spike detection—without changing the conceptual model family?**

This is stronger than a simple “stage enum.” Some training behavior is **event-triggered**.

---

# Recipe 3 — Qwen3-VL multimodal pretraining

**Implementation/report examined:** Qwen3-VL technical report, using the published four-stage pretraining curriculum.

Primary sources: [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631?utm_source=chatgpt.com) · [official repository](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com).

## 1. Components

**Observed:** Qwen3-VL has three major model components:

```text
SigLIP-2 vision encoder
        ↓
vision-language merger
        ↓
Qwen3 LLM
```

DeepStack additionally injects projected intermediate ViT features directly into multiple early LLM layers rather than supplying vision only once at the input boundary. ([arXiv][5])

So even “vision encoder → LLM” undersells the relationship: several vision-derived representations feed several language-model layers.

## 2. Data and one training step

**Observed:** sequences can contain text, images and video in an interleaved context. The system uses visual tokens plus text tokens under one causal sequence representation, while video timing is represented partly by explicit textual timestamp tokens. ([arXiv][5])

The four published pretraining stages are:

```text
S0: 67B tokens, 8K
    train merger only

S1: ~1T tokens, 8K
    train all components

S2: ~1T tokens, 32K
    train all components

S3: 100B tokens, 256K
    train all components
```

([NeuroAI Research][6])

The data distribution changes too: early alignment emphasizes image-caption/OCR/visual knowledge data; subsequent phases blend multimodal and text-only data, then progressively emphasize long-context, video and agent-oriented data. ([NeuroAI Research][6])

## 3. Optimization and runtime

**Observed:** S0 optimizes **only the merger**. Both the pretrained vision encoder and language model are frozen.

At S1:

```text
vision encoder  → train
merger          → train
LLM             → train
```

and all remain trainable through the long-context phases. ([NeuroAI Research][6])

So "frozen module" here means lifecycle state, not permanent component ownership.

## 4. Structural changes during a run

Very much present.

The curriculum explicitly changes:

```text
trainable component set
sequence-length contract
dataset mixture
task distribution
```

The most obvious transition is:

```text
S0
vision encoder frozen
LLM frozen
merger trainable

       ↓ stage transition

S1+
vision encoder trainable
LLM trainable
merger trainable
```

and then the accepted sequence length grows by **32×**, from 8K to 256K, across later stages. ([NeuroAI Research][6])

## 5. Outputs and restoration

**Observed:** unlike a vision adapter trained against permanently frozen components, the final Qwen3-VL base artifact contains jointly trained vision encoder, mergers and language backbone.

The post-training pipeline then proceeds through SFT, distillation and RL, but that's downstream of the specific pretraining recipe considered here. ([ArxivLens][7])

**Unknown:** exact optimizer/resume state across stage boundaries is not published at production-code granularity.

## 6. Design pressure

> **Can a training job change which components are trainable at stage boundaries without changing their ownership or replacing the model?**

And:

> **Can data mixture, maximum sequence length and trainable-parameter set evolve together as one coherent stage transition?**

This one is a strong warning against hard-coding “text encoder frozen / backbone trainable” assumptions from diffusion training.

---

# Recipe 4 — MiMo-V2.6-Flash mixed agentic RL

This one is useful because it's almost a completely different meaning of “training step.”

**Implementation/report examined:** Xiaomi MiMo-V2.6-Flash-RL, September 2026.

Primary sources: [official model card and technical-report links](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL?utm_source=chatgpt.com) · [Xiaomi release and training description](https://mimo.mi.com/docs/en-US/news/latest/v2-6?utm_source=chatgpt.com).

## 1. Components

**Observed:** the trainable policy is a 309B-total / 15B-active multimodal MoE supporting text, image, video and audio.

But the training system includes substantially more than the policy:

```text
policy model
rollout inference workers
agent harnesses
interactive environments
task verifiers
groupwise agentic grader
GRPO optimization workers
```

([Hugging Face][8])

The environment and grader aren't “model layers,” but they're indispensable causal participants in producing the optimization target.

## 2. Data and one training step

A “batch” no longer means a tensor of tokenized documents.

**Observed:** one global update starts from **1,568 prompts**, with **16 rollouts each**, producing roughly 25K trajectories and billions of generated tokens per optimization step. The rollouts come from a mixture of coding, general-agent, visual and cybersecurity environments under multiple harnesses. ([Hugging Face][8])

Conceptually:

```text
prompt/task
   ↓
environment + harness
   ↓
student rollout
   ↓
actions / observations / tools / tokens
   ↓
verifiers + groupwise grader
   ↓
relative advantages
   ↓
GRPO policy update
```

**Observed:** Groupwise Reward Synthesis uses contrasting rollouts to construct task-specific rubrics, while Groupwise Advantage Redistribution further ranks trajectories that already pass binary checks. ([Hugging Face][8])

So the target is **computed from relationships among multiple trajectories**, rather than attached independently to one example.

## 3. Optimization and runtime

**Observed:** rollout generation and optimization are fully asynchronous. Training supports trajectories reaching up to the model's 1M-token context. ([Hugging Face][8])

**Observed:** Xiaomi reports freezing the MoE router as RL scale increases to avoid expert-load drift. ([MiMo][9])

That means the policy is neither completely frozen nor completely trainable:

```text
policy backbone/expert state → optimized
router                      → frozen
```

**Unknown from the primary high-level sources used here:** the full per-parameter optimizer configuration. The technical report contains more implementation detail, but the core Trainer pressure does not depend on guessing those values.

## 4. Structural changes during a run

Inside the mixed RL run, **task type is not a stage transition**. Coding, visual, cyber and general-agent tasks coexist in the same training process.

This is important:

```text
NOT:

train coding
→ switch Trainer
→ train visual
→ switch Trainer
→ train cyber

BUT:

one policy update stream
← trajectories from all of them
```

([Hugging Face][8])

After the mixed RL phase, MiMo uses **MOPD2**, a separate multi-prefix/multi-teacher on-policy distillation phase. That's a real later recipe transition rather than just another reward term. ([Hugging Face][8])

## 5. Outputs and restoration

**Observed:** the result of the RL phase is an updated policy checkpoint; the environments, verifiers and grading infrastructure are training dependencies, not inference dependencies.

**Inferred:** exact resumption requires substantially more than weights and optimizer state if the goal is continuation of an asynchronous agent run: rollout/task scheduler state, data-mixture progress and potentially outstanding trajectories matter.

**Unknown:** the public model card alone does not define the exact authoritative checkpoint schema for every distributed subsystem.

## 6. Design pressure

> **Can a training example be an interaction trajectory produced dynamically by external execution environments rather than static dataset material?**

> **Can reward/loss construction depend on an entire group of sibling trajectories rather than one sample independently?**

And:

> **Can components that are not neural models—harnesses, tools, sandboxes, verifiers and graders—be first-class dependencies of a training recipe without being forced into “model component” slots?**

That last one is probably the main RL architecture lesson.

---


# Recipe 5 — MDLM OpenWebText masked-diffusion pretraining

**Implementation examined:** `kuleshov-group/mdlm`, commit `c112c526`, concrete OpenWebText recipe `scripts/train_owt_mdlm.sh`, with the SUBS continuous-time masked-diffusion implementation in `diffusion.py`.

Primary sources: [MDLM paper][12] · [repository/README][13] · [`diffusion.py`][14] · [`train_owt_mdlm.sh`][15].

## 1. Components

**Observed:** the concrete recipe contains a denoising Transformer backbone, a discrete absorbing-state corruption process whose absorbing symbol is the tokenizer's mask token, a noise schedule, and an EMA copy of the trainable state. ([GitHub][13])

The repository deliberately supports several generation factorizations behind the same outer training framework:

```text
autoregressive
masked diffusion / SUBS
D3PM
SEDD
```

For the exact MDLM recipe:

```text
backbone = DiT-style Transformer
parameterization = SUBS
diffusion = absorbing_state
T = 0  # continuous time
time_conditioning = false
sequence length = 1024
```

([GitHub][13])

**Observed:** the log-linear noise schedule used by the default recipe has no learned parameters of its own, although the generic optimizer/EMA plumbing treats the backbone and noise module as participants together. ([GitHub][14])

## 2. Data and one training step

**Observed:** a batch begins as ordinary clean token IDs:

```text
x0: [B,L]
```

A continuous diffusion time/noise level is sampled. `q_xt()` independently chooses token positions according to the resulting masking probability and replaces those positions with the mask token:

```text
clean token sequence x0
        ↓ sample t
        ↓
random absorbing-mask corruption
        ↓
partially masked xt
        ↓
bidirectional denoising Transformer
        ↓
distribution over original tokens
```

([GitHub][14])

The SUBS parameterization hard-codes two useful properties into the model output:

- the model cannot predict `[MASK]` as the clean token;
- already-unmasked tokens are treated as fixed/copy states rather than something the denoiser should rewrite.

**Observed:** in continuous time, the training objective gathers the model log probability of each original token and weights it using the sampled noise schedule. The paper derives this as a simplified/Rao-Blackwellized masked-diffusion objective: effectively a mixture of masked-language-model losses rather than left-to-right next-token CE. ([arXiv][12])

The exact OWT script uses 1024-token sequences and a global batch-size default of 512 through the repository config. ([GitHub][15])

## 3. Optimization and runtime

**Observed:** the implementation uses AdamW over the backbone and noise module, a step-based LR scheduler, BF16 training, gradient clipping, distributed training, and EMA with default decay `0.9999`. ([GitHub][14])

For the default log-linear schedule, **inferred:** optimizer ownership includes the noise module generically even though that concrete module contributes no trainable parameter tensors.

**Observed:** unlike ordinary causal LM training, the corruption operation itself is part of every training step. The same clean sequence can therefore produce a different model input on repeated visits.

## 4. Structural changes during a run

**Not present** in this concrete OWT recipe.

The sampled diffusion time and mask realization change every step, but component ownership, objective family, optimizer structure, and attention semantics remain fixed.

## 5. Outputs and restoration

**Observed:** the learned artifact is the denoising language model; generation begins from mask tokens and iteratively executes reverse denoising rather than appending one next token at a time. ([GitHub][14])

The checkpoint hooks explicitly persist EMA state, while Lightning supplies the ordinary model/optimizer/training-loop checkpoint state. The implementation also recovers completed epoch/batch progress on resume so the dataloader can be fast-forwarded. ([GitHub][14])

## 6. Design pressure

> **Can token corruption/noising be a recipe-owned stochastic transformation performed after tokenization rather than a fixed property of the dataset?**

And:

> **Can the language-model execution primitive be bidirectional masked denoising with same-position targets, rather than assuming causal attention and one-token-shifted labels?**

This is the cleanest counterexample to defining generic LLM training as `input[:-1] → target[1:]`.

---

# Recipe 6 — LLaDA2.0 AR-to-diffusion conversion

**Training recipe examined:** the published LLaDA2.0 AR→dLLM conversion/pretraining recipe from *LLaDA2.0: Scaling Up Diffusion Language Models to 100B*, cross-checked against the released `inclusionAI/dFactory` training implementation at commit `a385b14f`.

Primary sources: [LLaDA2.0 paper][16] · [dFactory masked-diffusion objective][17] · [`train_llada2_bd.py`][18] · [released block-diffusion SFT config][19].

The public `dFactory` code exposes the masking/SFT and block-diffusion mechanics directly; it does **not** expose the exact production launcher/config for every phase of the original 100B pretraining run, so optimizer claims about that production run are kept separate from the released SFT recipe.

## 1. Components

**Observed:** LLaDA2.0 does not begin from a randomly initialized diffusion LM. The published recipe **converts a pretrained autoregressive model into a discrete diffusion model**, preserving inherited model knowledge. The released family contains a 16B `mini` and 100B-total MoE `flash`. ([arXiv][16])

Conceptually:

```text
pretrained AR checkpoint
        ↓ initialize / convert
LLaDA2 diffusion model
        ↓
diffusion adaptation
```

The AR initializer is therefore an **upstream source of model state**, not a second live teacher required for every diffusion-training step.

**Observed:** the released dFactory model/training path includes the LLaDA2 MoE backbone, mask-token corruption, an attention policy capable of full diffusion or block diffusion, and the ordinary distributed training/checkpoint infrastructure. ([GitHub][18])

## 2. Data and one training step

**Observed:** the documented masked-diffusion objective samples `t ∈ [0,1]`, replaces eligible tokens with `[MASK]` with probability determined by `t`, and trains a non-causal mask predictor to reconstruct only the masked positions, weighted by `1/t`. ([GitHub][17])

For SFT, the prompt is explicitly protected from corruption:

```text
prompt X            → always intact
response Y          → randomly masked
                          ↓
                    denoising CE
                    only at masked response positions
```

([GitHub][17])

The concrete dFactory SFT transform makes this literal: labels before `prompt_length` are `-100`; only response positions are maskable; after corruption, labels are retained only where the noisy input actually equals the mask token. ([GitHub][18])

**Observed:** in `block_diffusion_mode`, the training entrypoint concatenates the noisy and clean token sequences and installs a custom block attention mask. With `same_token_labels=true`, the concrete released config computes CE at the **same token locations**, not with a next-token shift. ([GitHub][18])

## 3. Optimization and runtime

**Observed for the released dFactory SFT recipe:** the concrete `llada2_mini_bd_sft.yaml` uses AdamW, cosine LR scheduling, gradient clipping, BF16/mixed precision, FSDP2 full sharding, gradient checkpointing, and CPU/offload support. Its example uses block size 32. ([GitHub][19])

**Unknown for the original large-scale LLaDA2.0 conversion/pretraining run:** the public paper/repository sources examined here do not establish every optimizer group and distributed-state detail at production-code granularity.

This separation matters: the diffusion objective and curriculum are well documented, but we should not silently project the public SFT optimizer configuration backward onto the 100B pretraining run.

## 4. Structural changes during a run

**Observed:** this is the most interesting part of the LLaDA2.0 recipe.

The paper describes a three-phase **block-level WSD** diffusion-training schedule:

```text
warm-up
progressively increase block size
        ↓
stable phase
full-sequence diffusion
        ↓
decay phase
return to compact block diffusion
```

([arXiv][16])

Thus the generation/training factorization itself changes over the course of the run. The model moves between more local block diffusion and full-sequence diffusion while retaining inherited weights.

This is not just context-length curriculum: **block size changes which tokens are jointly denoised and what attention/factorization the objective represents.**

## 5. Outputs and restoration

**Observed:** the final deployable artifact is a diffusion language model; it does not require the original AR checkpoint as a separate inference component. ([arXiv][16])

**Observed for dFactory checkpoints:** the released trainer saves model and optimizer state plus explicit scheduler state, dataloader state, environment-meter state, global step, and Torch RNG state. ([GitHub][18])

**Unknown:** the public sources do not establish whether the original production LLaDA2.0 pretraining checkpoint encoded every phase/curriculum controller in exactly the same structure as dFactory.

## 6. Design pressure

> **Can one model's weights be inherited from a different generation factorization—AR → diffusion—without treating the source model as a runtime teacher?**

And:

> **Can block size / generation factorization itself be curriculum state, changing across training phases while the underlying model identity persists?**

This also means “architecture” and “training objective” are not synonymous. Much of the same model state can survive a fairly radical change in how token dependencies are trained.

---

# Recipe 7 — BD3-LM OpenWebText block-diffusion training

**Implementation examined:** `kuleshov-group/bd3lms`, commit `1c3e8f43`, OpenWebText recipe `scripts/train/train_owt_bd3lm.sh`.

Primary sources: [Block Diffusion paper][20] · [repository/README][21] · [`diffusion.py`][22] · [`configs/config.yaml`][23].

The released OWT recipe uses a 1024-token context, recommends block sizes 4/8/16, and fine-tunes from an MDLM checkpoint that had already been trained for 850K gradient updates. Released BD3-LMs were then trained for 1M steps. ([GitHub][21])

## 1. Components

**Observed:** BD3-LM combines two different factorization ideas inside one language model:

```text
between blocks:
    autoregressive dependency

inside current block:
    discrete diffusion / parallel denoising
```

([arXiv][20])

The implementation contains a denoising Transformer backbone, mask-token absorbing process, noise schedule, block-aware attention mask, EMA state, and block/noise-schedule training state. ([GitHub][22])

No separate AR model runs beside the diffusion model during the concrete fine-tuning recipe. The MDLM checkpoint is initialization.

## 2. Data and one training step

**Observed:** the sequence is divided into blocks, and `_sample_t()` samples **one noise time per block**, not necessarily one time for the entire sequence. The sampled block value is repeated across tokens belonging to that block. ([GitHub][22])

Thus a single training example can look conceptually like:

```text
block 0     block 1     block 2     block 3
 t=.12       t=.81       t=.35       t=.60
   ↓           ↓           ↓           ↓
different mask/noise levels within one sequence
```

The corruption process then masks tokens according to those block-level probabilities.

**Observed:** in the efficient cross-attention training path, the implementation concatenates noisy `x_t` and clean `x0` representations and relies on a specialized block attention topology to expose only the permitted context. The resulting loss still gathers probability of the original token at each position with diffusion-schedule weighting. ([GitHub][22])

So “sequence” is not associated with one global causal mask or one global diffusion time.

## 3. Optimization and runtime

**Observed:** the default implementation uses one AdamW optimizer over backbone + noise-module parameters, BF16, gradient clipping, gradient accumulation as required to reach the global batch size, and EMA (`0.9999`). ([GitHub][23])

The concrete OWT training script uses:

```text
global batch = 512
context = 1024
block size = 16 in the supplied script
training.resample = true
```

and initializes from the released full-sequence MDLM checkpoint unless explicitly set to train from scratch. ([GitHub][21])

## 4. Structural changes during a run

The model component graph remains fixed, but **Observed:** the recipe can adapt its training-noise sampling interval based on measured gradient/loss variance.

During validation, `_clipped_schedule_search()` compares candidate noise intervals and updates `sampling_eps_min` / `sampling_eps_max` to the interval with lower measured variance unless the schedule is configured as fixed. Those values then affect subsequent corruption/time sampling. ([GitHub][22])

So there is persistent training state that is:

```text
not model weights
not optimizer moments
not LR scheduler
but still changes future training examples/objectives
```

This state is also explicitly checkpointed. ([GitHub][22])

## 5. Outputs and restoration

**Observed:** the deployable artifact is one BD3 language model; block-autoregressive generation enables arbitrary-length output while diffusion operates within each generated block. KV caching is supported for this factorization. ([GitHub][21])

**Observed:** checkpoints add EMA state and the adaptive `sampling_eps_min/max` values to ordinary Lightning training state. Resume logic also recovers epoch/batch progress for dataloader continuation. ([GitHub][22])

## 6. Design pressure

> **Can different regions of the same token sequence carry different diffusion/noise times in the same forward pass?**

And:

> **Can attention topology express a hybrid factorization—causal across groups, bidirectional/diffusive within groups—without forcing the model into either an `autoregressive` or `diffusion` global category?**

And one that seems particularly relevant to the Trainer:

> **Can validation-derived algorithm state modify future training-example construction and be checkpointed as authoritative run state even though it belongs to neither the model nor optimizer?**

---

# Diffusion-LM takeaway

These recipes change the earlier “LLM training is basically next-token prediction” conclusion in an important way.

The **token domain** is shared, but the execution semantics do not have to be:

```text
AR:
clean prefix
  ↓ causal model
next token

masked diffusion:
clean sequence
  ↓ stochastic corruption
partially masked sequence
  ↓ bidirectional denoiser
same-position clean-token targets

block diffusion:
previous clean blocks + noisy current blocks
  ↓ hybrid attention/factorization
parallel within-block denoising
```

So I would not give the generic Trainer an inherently autoregressive LLM primitive. The model/recipe boundary needs to be able to define:

```text
how clean tokens become model observations
which positions are targets
how targets align with inputs
attention topology
noise/time state and its granularity
loss weighting
iterative/block generation semantics
```

`d3LLM` is also a useful follow-up for the **distillation** note because it transfers a teacher diffusion model's decoding order into a pseudo-trajectory and progressively changes noise/window schedules. `MMaDA` is more useful under a future **unified multimodal generation** topic because it applies masked diffusion across text and visual generation/understanding rather than adding a fundamentally new text-only corruption primitive.

---

Yep. I’d add it as the next concrete recipe in `llm-training.md`.

# Recipe 8 — DiffusionGemma Sudoku LoRA SFT: hybrid causal + uniform-state diffusion

**Implementation examined:** `google-deepmind/gemma`, commit `0513283af5afffa27390b6ede2facc35d0f16e08`, using the concrete Sudoku LoRA recipe in `gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py`.

Supporting diffusion primitives come from `google/hackable_diffusion`, commit `fb225b51132657ebebc294fc976418d0ecc30d7a`.

Primary sources:

[DiffusionGemma explanation](https://ai.google.dev/gemma/docs/diffusiongemma/explained?utm_source=chatgpt.com)
[Sudoku SFT config](https://github.com/google-deepmind/gemma/blob/0513283af5afffa27390b6ede2facc35d0f16e08/gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku.py?utm_source=chatgpt.com)
[SFTDiffusion implementation](https://github.com/google-deepmind/gemma/blob/0513283af5afffa27390b6ede2facc35d0f16e08/gemma/diffusion/hackable_diffusion_adapter/hd/sft_model.py?utm_source=chatgpt.com)
[attention/KV-cache helpers](https://github.com/google-deepmind/gemma/blob/0513283af5afffa27390b6ede2facc35d0f16e08/gemma/diffusion/hackable_diffusion_adapter/hd/mask_helpers.py?utm_source=chatgpt.com)
[Hackable Diffusion categorical corruption](https://github.com/google/hackable_diffusion/blob/fb225b51132657ebebc294fc976418d0ecc30d7a/hackable_diffusion/lib/corruption/discrete.py?utm_source=chatgpt.com)

**Claim labels:** **Observed** = directly established by source. **Inferred** = follows from implementation. **Unknown** = examined sources do not establish it.

## 1. Components

**Observed:** the concrete recipe starts from `DiffusionGemma_26B_A4B`, wraps it as a diffusion-capable Gemma network, and applies rank-8 LoRA to all linear modules.

There are not separate “AR model” and “diffusion model” objects. The **same physical Gemma backbone** participates in two execution roles:

```text
same Gemma backbone
      │
      ├── causal encoder / prefill role
      │       → AR logits
      │       → KV cache
      │
      └── diffusion denoiser role
              → bidirectional canvas logits
```

Google describes DiffusionGemma as one backbone dynamically switching between causal prefill and bidirectional denoising rather than using separate models. ([Google AI for Developers][24])

**Observed:** independently identifiable recipe components include:

```text
Gemma backbone
LoRA parameters
categorical corruption process
discrete corruption schedule
time sampler
KV-cache / attention-state machinery
diffusion loss
causal encoder loss
```

The corruption process and loss are not implemented as properties of the Gemma architecture itself.

## 2. Data and one training step

The concrete Sudoku config uses:

```text
vocabulary size = 262,144
prompt length   = 256
canvas size     = 256
num canvases    = 1
```

**Observed:** DiffusionGemma uses **Uniform State Diffusion**, not masked diffusion. Google describes the forward corruption as replacing tokens with random vocabulary tokens instead of introducing a dedicated `[MASK]` state. ([Google AI for Developers][24])

The actual Hackable Diffusion implementation confirms this. `uniform_process()` defines the invariant distribution as:

```text
p(token=i) = 1 / vocabulary_size
```

and `corrupt()` independently either preserves each clean token or replaces it with a sample from that uniform distribution according to the corruption schedule.

So the training corruption is approximately:

```text
clean canvas x₀
      │
sample diffusion time t
      │
      ├── retain token with probability α(t)
      └── otherwise replace with random vocabulary token
      │
      ▼
corrupted canvas xₜ
```

That differs materially from MDLM/LLaDA:

```text
masked diffusion:
token → [MASK]

DiffusionGemma:
token → arbitrary vocabulary token
```

### The full training forward is more unusual

**Observed:** `SFTDiffusion.__call__()` first samples the diffusion time and corrupts the clean target canvas.

It then runs a **causal encoder/prefill pass** using the clean sequence:

```text
prompt + clean target canvases
          ↓
causal Gemma execution
          ↓
encoder logits
+ positions
+ KV cache
```

The implementation subsequently adjusts the KV-cache write cursor so the denoiser sees the prompt and preceding canvases but does not treat future canvases as established history.

Then the same Gemma network runs as the diffusion denoiser:

```text
corrupted xₜ
+ diffusion time
+ prefilled KV cache
+ block/canvas attention mask
          ↓
denoiser pass 1
```

**Observed:** this first denoiser output is converted to clean-token predictions and then explicitly passed through `stop_gradient`.

Those predictions become the optional **self-conditioning** input for a second denoiser forward.

The default is:

```text
self_cond_prob = 0.5
```

so for each example the second pass receives either:

```text
first-pass prediction logits
```

or:

```text
zero logits
```

Importantly, **the first denoiser forward still happens either way**.

So the actual execution is:

```text
                         clean target
                              │
                    ┌─────────┴─────────┐
                    │                   │
             categorical corrupt   causal prefill
                    │                   │
                   xₜ               KV cache
                    │                   │
                    └─────────┬─────────┘
                              │
                       denoiser pass 1
                              │
                         predicted x₀
                              │
                        stop_gradient
                              │
                    50% prediction / zeros
                              │
                       denoiser pass 2
                              │
                    final diffusion logits
```

That is a genuinely different step structure from MDLM's single corrupted-sequence forward.

### Losses

**Observed:** the concrete recipe trains **two losses simultaneously**:

```text
diffusion_loss
+
encoder_loss
```

The diffusion loss is ordinary token cross-entropy between denoiser logits and clean tokens, restricted by the selected canvas mask.

The encoder loss is a separate causal next-token cross-entropy over the encoder/prefill output.

Both are configured with weight `1.0`.

So one physical backbone is simultaneously trained for:

```text
causal language modeling
AND
bidirectional token denoising
```

within the same training step.

## 3. Optimization and runtime

**Observed:** the examined recipe trains **rank-8 LoRA only**:

```text
target_modules = "all-linear"
```

The generic optimizer chain is:

```text
global gradient clip: 1.0
Adam:
    β1 = 0.95
    β2 = 0.99
    eps = 1e-8
weight decay = 1e-4
warmup + cosine LR schedule
peak LR = 1.5e-4
end LR = 1.5e-5
training steps = 2000
```

`partial_updates(... select("lora"))` restricts actual parameter updates to LoRA state.

The repository also supplies a **full-weight Sudoku recipe**, using Adafactor, but that is a separate concrete recipe and should not be conflated with this one.

**Observed:** the causal prefill is not just an inference convenience. Its KV-cache becomes an intermediate training representation consumed by the denoiser.

The recipe exposes:

```text
stop_gradient_from_denoiser_to_encoder
```

which controls whether gradients from the diffusion path may propagate back through that cached encoder computation.

For this Sudoku LoRA recipe:

```text
stop_gradient_from_denoiser_to_encoder = False
```

so the gradient boundary is intentionally open.

This gives a pretty striking relationship:

```text
causal encoder computation
         │
         ├── own AR loss
         │
         └── intermediate KV state
                    │
                    ▼
             diffusion denoiser
                    │
             diffusion loss
```

The **same execution can therefore be supervised directly and also act as differentiable conditioning for another role**.

## 4. Structural changes during a run

**Not present** as a stage transition in this concrete 2,000-step Sudoku recipe.

The model does not switch halfway through from AR to diffusion or replace components.

But there is substantial **per-step role variation**:

```text
one backbone
    causal execution
    ↓
    bidirectional execution
    ↓
    second bidirectional execution
```

and there is stochastic behavior from:

```text
diffusion time
token corruption
self-conditioning choice
```

The generic `SFTDiffusion` implementation also supports multiple canvases and samples one valid canvas per example for diffusion training. In this specific Sudoku configuration `num_canvases=1`, so that selection is trivially canvas 0.

For comparison, Google's documented long-form inference repeatedly finalizes 256-token canvases and appends them into causal history before diffusing the next canvas. ([Google AI for Developers][24])

## 5. Outputs and restoration

**Observed:** the LoRA recipe uses a Gemma-aware checkpoint formatter and exports **fused LoRA parameters**, rather than treating Hackable Diffusion's training wrapper itself as the deployable artifact.

The base DiffusionGemma checkpoint is therefore still required.

Inference additionally depends on the execution semantics around:

```text
causal prefill
KV-cache management
canvas denoising
uniform categorical diffusion
self-conditioning
multi-canvas AR orchestration
```

Google's adapter deliberately runs evaluation separately from training by loading a saved checkpoint and performing AR-diffusion sampling.

**Unknown:** the examined sources do not establish that the formatted inference artifact alone contains enough state for exact training continuation. Kauldron's ordinary training checkpoints carry additional optimizer/progress state, while the Gemma formatter is aimed at producing a Gemma-compatible model artifact.

## 6. Design pressure

This recipe gives us a few very useful constraints.

> **Can one physical model execute multiple semantic roles inside one training step, with different attention semantics and different objectives, without representing those roles as separate owned model components?**

That's probably the biggest one.

The second:

> **Can intermediate execution state—such as a KV cache produced by one role—become differentiable conditioning for another role, with the recipe explicitly controlling whether gradients cross that boundary?**

And another one that the self-conditioning path exposes:

> **Can a recipe request multiple forwards through the same role, where an earlier forward is computation-only, its outputs are detached, transformed or stochastically discarded, and a later forward produces the actual optimized prediction?**

I would also add a discrete-diffusion-specific constraint:

> **Token diffusion should not imply masked-token corruption. The corruption process may have a different invariant distribution, such as the full vocabulary, while the model still predicts clean token identities through cross-entropy.**

And finally:

> **Attention topology belongs to execution role, not necessarily model identity. The same Transformer may be causal during one forward and bidirectional/block-causal during another.**

That last one is probably the most useful new thing DiffusionGemma contributes beyond MDLM/LLaDA/BD3-LM.

---

# What about GLM-5.3 and regular Qwen3.5–3.8?

There is useful information, but I wouldn't manufacture full recipe writeups where the sources don't support them.

**GLM-5.3:** the official GLM repository explicitly says **GLM-5.3 uses the same base model as GLM-5.2 and all of the improvement comes from post-training**. Z.ai supports RL through its `slime` infrastructure, but the currently published 5.3 material doesn't expose enough of the exact 5.3 post-training sequence, losses and optimizer ownership to populate our six sections with the same confidence as the four cases above. ([GitHub][10])

That fact is itself useful:

```text
same base architecture
        ↓
different post-training procedure
        ↓
new released model
```

So a model/version should not imply a new architecture or new pretraining run.

**Qwen3.5 / 3.6 / regular 3.8:** their public material gives substantial architecture and RL-level descriptions, but **Qwen3.8-Flash-Next currently has the much stronger training-specific technical report**, including optimizer assignment, stability experiments and continued-pretraining transitions. Qwen's own repository describes the hybrid Gated-DeltaNet design introduced around Qwen3.5 as carrying through the later 3.x generations. ([GitHub][11])

So Flash-Next gives us better evidence than trying to reverse-engineer regular 3.8 from a model card.

---

# Cross-case findings

The enlarged set changes the answer a bit: **tokenized language is common, autoregressive training is not**.

| Assumption | Finding |
| --- | --- |
| LLM training necessarily means left-to-right next-token CE | **False.** MDLM/LLaDA use same-position masked denoising; BD3-LM mixes diffusion within blocks with AR dependence across blocks. |
| Therefore different architectures train identically | **No.** Optimizer assignment, corruption processes, auxiliary objectives, routing, attention and curriculum vary significantly. |
| Attention is globally causal for language training | **False.** masked diffusion can be bidirectional; block diffusion uses hybrid block-aware attention. |
| Tokenization fully determines the model input | **False.** diffusion recipes stochastically corrupt the token sequence after tokenization on every step. |
| One noise/time value applies to a whole example | **False.** BD3-LM can sample different diffusion times for different blocks of one sequence. |
| One model means one optimizer | **False.** Qwen3.8 and DeepSeek-V4 partition parameters between Muon and AdamW. |
| Trainability is fixed for the run | **False.** Qwen3-VL freezes only the merger initially, then unfreezes everything. |
| Architecture/execution semantics are fixed once training starts | **False.** Qwen and DeepSeek introduce sparse-attention machinery later; LLaDA2 changes diffusion block factorization across phases. |
| One sequence length defines the job | **False.** DeepSeek goes 4K→16K→64K→1M; Qwen3-VL goes 8K→32K→256K. |
| One loss defines the job | **False.** LM CE can coexist with MTP, MoE balance and other auxiliary objectives. |
| One sample produces one independent loss | **Very false in RL.** MiMo's rewards depend on groups of rollouts. |
| Dataset is necessarily static | **False.** RL environments generate trajectories; diffusion recipes also generate stochastic corrupted observations from clean examples. |
| All authoritative run state is model/optimizer/scheduler state | **False.** BD3-LM checkpoints an adaptive noise-sampling interval that changes future training data/objectives. |
| Every participant is a neural component | **False.** environments/verifiers/graders can determine the optimization signal. |

The main constraint I’d carry forward is:

> **A generic LLM Trainer should not assume an autoregressive execution primitive. The training recipe must own observation construction/corruption, target alignment, attention topology, loss weighting, parameter-group optimization, routing behavior, sequence/context curriculum, component trainability, and stage transitions. These cannot safely be inferred from “this is an LLM.”**

And after these newer reports, I’d add:

> **Generation factorization is part of the training contract. AR, full-sequence masked diffusion, and block diffusion can operate on the same basic token domain while requiring different corruption state, attention structure, targets, loss estimators, sampling state, and inference semantics.**

The other conclusion still holds:

> **Architecture and optimizer strategy are not independent configuration layers. Modern recipes co-design them: the model's parameter structure determines optimizer assignment, distributed sharding, stability mechanisms, and sometimes even when particular execution paths become active.**

So the forward path is not universally simpler than diffusion-style media training after all. **AR LMs often have a simple token-shifted objective; diffusion LMs move a surprising amount of complexity into observation construction, attention/factorization, and iterative-generation state.**

[1]: https://arxiv.org/abs/2608.30320 "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability"
[2]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next?utm_source=chatgpt.com "Qwen/Qwen3.8-Flash-Next · Hugging Face"
[3]: https://arxiv.org/abs/2606.19348 "[2606.19348] DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
[4]: https://arxiv.org/html/2606.19348 "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
[5]: https://arxiv.org/abs/2511.21631?utm_source=chatgpt.com "Qwen3-VL Technical Report"
[6]: https://neuroai-research.github.io/2.1_LLM/papers/05_Qwen3-VL_Technical_Report/?utm_source=chatgpt.com "5 2025 Qwen3-VL Technical Report - NeuroAI Research"
[7]: https://arxivlens.com/paperview/htmlversion/qwen3-vl-technical-report-9308-74b053e0?utm_source=chatgpt.com "- ArxivLens"
[8]: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/main/README.md "README.md · XiaomiMiMo/MiMo-V2.6-Flash-RL at main"
[9]: https://mimo.mi.com/docs/en-US/news/latest/v2-6?utm_source=chatgpt.com "Xiaomi MiMo Home"
[10]: https://github.com/zai-org/GLM-5/blob/main/README.md "GLM-5/README.md at main · zai-org/GLM-5 · GitHub"
[11]: https://github.com/QwenLM/Qwen3.8/blob/main/README.md?utm_source=chatgpt.com "Qwen3.8/README.md at main · QwenLM/Qwen3.8 · GitHub"

[12]: https://arxiv.org/abs/2406.07524 "Simple and Effective Masked Diffusion Language Models"
[13]: https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/README.md "MDLM README"
[14]: https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py "MDLM diffusion.py"
[15]: https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/scripts/train_owt_mdlm.sh "MDLM OpenWebText training recipe"
[16]: https://arxiv.org/abs/2512.15745 "LLaDA2.0: Scaling Up Diffusion Language Models to 100B"
[17]: https://github.com/inclusionAI/dFactory/blob/a385b14f91315cf43cf725fea1e3f45ef1b964bd/docs/source/algo/random_mask.rst "dFactory masked-diffusion objective"
[18]: https://github.com/inclusionAI/dFactory/blob/a385b14f91315cf43cf725fea1e3f45ef1b964bd/tasks/train_llada2_bd.py "dFactory LLaDA2 block-diffusion trainer"
[19]: https://github.com/inclusionAI/dFactory/blob/a385b14f91315cf43cf725fea1e3f45ef1b964bd/configs/sft/llada2_mini_bd_sft.yaml "dFactory LLaDA2-mini block-diffusion SFT config"
[20]: https://arxiv.org/abs/2503.09573 "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models"
[21]: https://github.com/kuleshov-group/bd3lms/blob/1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7/README.md "BD3-LM README"
[22]: https://github.com/kuleshov-group/bd3lms/blob/1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7/diffusion.py "BD3-LM diffusion.py"
[23]: https://github.com/kuleshov-group/bd3lms/blob/1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7/configs/config.yaml "BD3-LM default training config"
[24]: https://ai.google.dev/gemma/docs/diffusiongemma/explained "Diffusion in Text Generation Explained  |  Gemma  |  Google AI for Developers"
