# Precision and Quantization in Training

## Scope

Study how numerical representation affects training execution and ownership.

This is not a survey of numeric formats. The focus is on cases where storage
precision, compute precision, gradient precision, optimizer precision,
quantization state, or deployment precision differ and therefore place
requirements on the training system.

Topics include:
- ordinary FP16/BF16 mixed precision
- quantized frozen-base training
- native FP8/FP4 training
- quantization-aware training
- low-precision model state and checkpoint semantics

Claim labels:
Observed / Inferred / Unknown
```

---

## 1. Precision is not one model property

The central thing we want to establish is:

```text
weight storage dtype
    ≠
forward compute dtype
    ≠
backward compute dtype
    ≠
activation dtype
    ≠
gradient dtype
    ≠
gradient accumulation dtype
    ≠
optimizer/master-weight dtype
    ≠
saved artifact dtype
```

And with quantization there is additional state:

```text
quantized values
scale factors
zero points / codebooks
block geometry
amax history
rounding policy
quantization recipe
```

Transformer Engine explicitly describes mixed training this way: BF16/FP16 may be used for matrix operations while numerically sensitive operations stay higher precision; FP32 master weights can remain responsible for parameter updates. FP8/FP4 add per-tensor or per-block scaling state rather than behaving like a simple dtype substitution. ([NVIDIA Docs][1])

The first constraint should probably be:

> **Precision belongs to an operation/state role, not necessarily to an entire model or component.**

---

# Recipe 1 — Ordinary FP16/BF16 mixed-precision training

I’d include this even though it sounds boring, because everything else needs a baseline.

## 1. Components

No architectural component needs to change.

Instead, the same parameter may participate in several representations:

```text
FP32 master weight
       │
       ↓ cast
BF16 / FP16 execution weight
       │
       ↓
forward / backward
       │
       ↓
gradient
       │
       ↓
FP32 optimizer update
```

Transformer Engine documents FP32 master weights specifically because updates can be too small to survive directly in FP16/BF16 stored weights. ([NVIDIA Docs][1])

## 2. One training step

Autocast chooses low precision for suitable operations while keeping numerically sensitive work higher precision.

PyTorch AMP separates:

```text
autocast
    operation precision policy

GradScaler
    dynamic gradient scaling
```

rather than interpreting “mixed precision” as globally casting the model. ([PyTorch Documentation][2])

## 3. Optimization/runtime

FP16's smaller exponent range creates underflow/overflow pressure, so loss scaling is commonly needed.

BF16 retains FP32's exponent range and generally avoids that particular requirement, despite having fewer mantissa bits. ([NVIDIA Docs][1])

## 4. Structural changes

None necessarily.

But dynamic loss scaling introduces mutable training state:

```text
current loss scale
growth/backoff history
```

which is not model state yet matters for exact continuation.

## 5. Outputs/restoration

A deployable checkpoint need not contain AMP/loss-scaler state.

A resumable checkpoint may need:

```text
model/master state
optimizer
gradient scaler
training progress
```

## 6. Design pressure

> **Can execution dtype vary per operation without changing component identity?**

> **Can optimizer-owned higher-precision state exist for parameters represented differently during execution?**

> **Can precision-control state belong to resumable training without belonging to the deployable model?**

---

# Recipe 2 — QLoRA: frozen 4-bit base with higher-precision training computation

This is worth including even though PEFT itself is already covered. Here we'd study **only the precision topology**.

QLoRA's important relationship is:

```text
4-bit NF4 base weights
        │
        │ dequantize / low-precision matmul
        ↓
BF16-ish compute
        │
        ↓
higher-precision LoRA parameters
        │
        ↓
loss
```

The pretrained base stays frozen while gradients propagate **through its quantized execution** into the trainable adapter. QLoRA introduced NF4, double quantization of quantization constants, and paged optimizers primarily to reduce memory requirements. ([arXiv][3])

Bitsandbytes makes the separation explicit: `Linear4bit` can store weights using NF4/FP4 while separately selecting a `compute_dtype`; the parameters' quantized storage is therefore not the arithmetic format of every operation. ([Hugging Face][4])

This gives us:

```text
base storage       = NF4
base trainability  = frozen
matmul compute      = BF16, etc.
adapter storage     = higher precision
optimizer state     = adapter-related
```

### Design pressure

> **Can a frozen component participate differentiably even though its stored parameter representation is quantized and non-optimizable?**

And:

> **Can parameter storage format and execution compute format be independent properties?**

And importantly:

> **“Quantized training” does not imply that the quantized parameters themselves are being optimized.**

That distinction needs to survive the note.

---

# Recipe 3 — Transformer Engine FP8 / MXFP8 / NVFP4 native low-precision training

This should probably be the central case.

Transformer Engine treats FP8/FP4 not as ordinary dtypes but as **training recipes**.

For example, its default hybrid FP8 scheme uses:

```text
forward weights / activations
    E4M3

backward gradients
    E5M2
```

because forward values benefit from more precision while gradients benefit from larger dynamic range. ([NVIDIA Docs][5])

And even then:

```text
linear GEMMs
    ↓ low precision

attention
    ↓ usually high precision

normalization / softmax
    ↓ higher precision

some internal arithmetic
    ↓ fixed FP32
```

([NVIDIA Docs][1])

So this absolutely kills:

```text
model.dtype = fp8
```

as a sufficient description.

### Scaling state

FP8 additionally needs state such as:

```text
amax measurements
scale factors
scale history
scaling algorithm
format choice
```

Transformer Engine's delayed-scaling recipe literally carries an `amax_history_len` and algorithm used to derive future scales. ([NVIDIA Docs][6])

That means **numeric policy can itself have mutable state updated every training step**.

### NVFP4

NVFP4 gets even stranger:

```text
activations
    block scaling
    random Hadamard transform

weights
    16 × 16 2D scaling

gradients
    stochastic rounding

last sensitive layers
    recommended higher precision
```

Transformer Engine recommends keeping the final few LLM layers at higher precision rather than blindly applying NVFP4 everywhere. ([NVIDIA Docs][6])

### Design pressure

> **Can precision policy have its own mutable execution state without becoming a model component?**

> **Can different tensors within one linear layer use different quantization/scaling rules?**

> **Can precision selection depend on architectural position, such as keeping sensitive final layers higher precision?**

This also gives us:

> **Low-precision execution policy may depend on hardware capabilities and kernel constraints rather than model semantics alone.**

For example, Transformer Engine notes shape divisibility requirements for some FP8 operations. ([NVIDIA Docs][6])

---

# Recipe 4 — DeepSeek-V4 FP4 quantization-aware post-training

This one is extremely good because it is **actual QAT of selected parts of a production model**, rather than generic library capability.

DeepSeek applies FP4 QAT during post-training to:

```text
MoE expert weights
CSA indexer Q/K path
```

while other parts remain at different precisions. ([arXiv][7])

For the MoE experts, the training path is:

```text
FP32 master weight
       │
       ↓
quantize to FP4
       │
       ↓
dequantize to FP8
       │
       ↓
forward computation
       │
       ↓
backward wrt FP8 representation
       │
       ↓ STE through quantization
       │
       ↓
FP32 master weight update
```

DeepSeek explicitly states that gradients propagate to the FP32 master weights using the straight-through estimator through the quantization step. ([arXiv][7])

That's almost the perfect counterexample to thinking that the “parameter” has one representation.

### Training vs rollout

It gets better.

During gradient training:

```text
FP32 → simulated FP4 → FP8 compute
```

But during RL rollout/inference:

```text
native FP4 weights
```

DeepSeek does that so generated trajectories experience the same quantized behavior as deployment while actually obtaining inference speed/memory savings. ([arXiv][7])

So the same logical model has **different physical numerical representations depending on execution role**.

### Design pressure

> **Can a parameter have an authoritative optimization representation and a different execution/deployment representation simultaneously?**

> **Can simulated quantization be used for differentiable training while another execution role uses actual quantized storage?**

And this is a great one:

> **Can quantization policy target only selected internal paths rather than an entire component?**

Because DeepSeek quantizes expert weights and the indexer's QK path, not simply `"transformer = fp4"`.

---

# Recipe 5 — diffusion-pipe: LoRA directly over quantized diffusion-model weights

I'd definitely include this because it's directly adjacent to your actual trainer ecosystem.

Current `diffusion-pipe` supports LoRA training directly on quantized ComfyUI model weights for a growing set of models, while the adapter itself can remain BF16. Its example configuration explicitly has:

```text
base dtype          = bfloat16
transformer dtype   = float8
LoRA dtype          = bfloat16
```

for Hunyuan Video. ([GitHub][8])

Recent support also allows training directly from already-quantized ComfyUI weights rather than first expanding the base model into a normal BF16 checkpoint. ([GitHub][9])

Conceptually:

```text
quantized base transformer
        │
        │ frozen
        │
        ├────── LoRA BF16 parameters
        │
        ↓
mixed execution
        ↓
loss / backward
        ↓
update LoRA only
```

There is an especially useful checkpoint implication in the implementation: because quantized base weights cannot be trained in that path, diffusion-pipe strips their special quantized state-dict behavior for DeepSpeed and relies on the fact that **the unchanged base weights do not need to be saved/restored as training-owned updates**.

So:

```text
loaded dependency:
    quantized base checkpoint

resumable mutable training state:
    LoRA
    optimizer
    loader/progress/etc.

export artifact:
    LoRA
```

That is exactly the kind of ownership distinction we're looking for.

There is also a practical warning from HunyuanVideo: diffusion-pipe notes that certain FP8 model + LoRA combinations can behave poorly if the LoRA is **merged into the low-precision base for inference**, even though training works because the LoRA weights remain separate. ([GitHub][10])

That's actually a fantastic finding:

> **A numerical representation that is suitable for frozen-base training may not be suitable for merging the resulting update into that representation.**

So training compatibility and artifact-merging compatibility are separate questions.

---

# Cross-case findings

I'd finish with something like:

| Tempting assumption                                           | Evidence                                                                                   |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| A model has one dtype                                         | False. Storage, compute, gradient and optimizer representations may differ.                |
| Quantized training means quantized weights are optimized      | False. QLoRA and quantized diffusion LoRA keep the quantized base frozen.                  |
| Low precision is stateless                                    | False. FP8/FP4 training may maintain scales, amax history and other quantization metadata. |
| One precision policy applies to the whole model               | False. Sensitive operations/layers/paths may remain higher precision.                      |
| Stored weight representation equals arithmetic representation | False. NF4→BF16 and FP4→FP8 are concrete counterexamples.                                  |
| Optimizer representation equals model representation          | False. FP32 master weights may update lower-precision execution weights.                   |
| Quantization is only a load/export concern                    | False. QAT inserts quantization into the differentiable training path.                     |
| Train-time and rollout representations must match physically  | False. DeepSeek uses simulated quantization for training and native FP4 for rollouts.      |
| A training checkpoint must copy the entire loaded base model  | False for frozen quantized-base adapter training.                                          |
| If training works on quantized weights, merging does too      | False in at least some low-precision diffusion cases.                                      |

And I think the final architecture constraint should be:

> **Precision should not be modeled as a scalar property of a model or training job. A recipe may independently determine parameter storage, forward and backward compute, accumulation, optimizer/master representation, quantization transforms, scaling state, and export representation.**

Then another one specifically for ownership:

> **Quantization state may belong to the loaded artifact, the live execution policy, the optimizer, or the training recipe depending on how the model is being trained; those responsibilities should not be inferred solely from parameter dtype.**

And maybe the nastiest useful one:

> **The authoritative trainable parameter need not be the tensor actually consumed by the forward pass.**

DeepSeek's:

```text
FP32 master
   ↓ quantize
FP4 representation
   ↓ dequantize
FP8 forward weight
```

makes that painfully concrete. ([arXiv][7])

[1]: https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/introduction/introduction.html?utm_source=chatgpt.com "Introduction — Transformer Engine 2.19.0"
[2]: https://docs.pytorch.org/docs/stable/accelerator/amp.html?utm_source=chatgpt.com "Automatic Mixed Precision — PyTorch 2.14 documentation"
[3]: https://arxiv.org/abs/2305.14314?utm_source=chatgpt.com "QLoRA: Efficient Finetuning of Quantized LLMs"
[4]: https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit?utm_source=chatgpt.com "4-bit quantization · Hugging Face"
[5]: https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html?utm_source=chatgpt.com "FP8 Current Scaling — Transformer Engine 2.19.0"
[6]: https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html?utm_source=chatgpt.com "Using FP8 and FP4 with Transformer Engine — Transformer Engine 2.19.0"
[7]: https://arxiv.org/html/2606.19348 "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
[8]: https://github.com/tdrussell/diffusion-pipe/blob/main/examples/main_example.toml?utm_source=chatgpt.com "diffusion-pipe/examples/main_example.toml at main · tdrussell/diffusion-pipe · GitHub"
[9]: https://github.com/tdrussell/diffusion-pipe?utm_source=chatgpt.com "GitHub - tdrussell/diffusion-pipe: A pipeline parallel training script for diffusion models. · GitHub"
[10]: https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md?utm_source=chatgpt.com "diffusion-pipe/docs/supported_models.md at main · tdrussell/diffusion-pipe · GitHub"
