# Conditioning and Control Side Networks

## Scope

Study training recipes that add new conditioning or control capability around
a pretrained generative model through separately trainable side networks or
injected control paths.

The focus is not control quality or inference UX. It is the effect on:

- component and parameter ownership
- paired-data requirements
- execution and gradient topology
- representation/injection boundaries
- initialization from the base model
- frozen-base training
- conditioning dropout/augmentation
- caching and preprocessing
- saved artifact boundaries

Representative cases:

1. ControlNet — copied side backbone + zero-initialized residual injection
2. T2I-Adapter — lightweight multi-scale feature side network
3. IP-Adapter — image encoder + projection + decoupled attention injection
4. Anima ControlNet-LLLite — shared conditioning encoder + injected micro-modules

Claim labels:
Observed / Inferred / Unknown

---

# 1. Start with the actual distinction

I’d open with this, because “control network” otherwise sounds like one architecture:

```text
                 PRETRAINED GENERATOR
                         │
                         ▼
                  model execution
                         │
                         ▼
                       output

new condition can enter through:

A. residual features
B. intermediate feature pyramids
C. attention K/V
D. patched linear/module computation
E. possibly combinations of these
```

So:

> **A control component is defined more by its relationship to the base execution than by a particular model topology.**

That becomes very obvious across the four recipes.

---

# 2. Recipe — classic ControlNet

I’d use the original ControlNet paper plus the concrete `sd-scripts` ControlNet implementation.

Primary references:

* ControlNet paper: *Adding Conditional Control to Text-to-Image Diffusion Models* ([arXiv][1])
* upstream `sd-scripts/train_control_net.py`
* `library/sdxl_original_control_net.py`

### Components

Classic ControlNet starts with a pretrained diffusion backbone and creates a trainable control branch based heavily on its encoder/down path.

In current `sd-scripts` SDXL implementation:

```text
frozen base U-Net
       │
       │
       ├─────────────── normal latent/text path
       │
       ▼
      output


trainable ControlNet
       ↑
control image
       │
conditioning encoder
       │
copied U-Net input/down/mid structure
       │
zero-initialized projections
       │
       ├── residual 0
       ├── residual 1
       ├── residual 2
       └── ...
                │
                ▼
        injected into frozen U-Net
```

**Observed:** `SdxlControlNet.init_from_unet()` initializes the ControlNet's shared encoder/down/middle structure from the pretrained U-Net.

The side-output convolutions and the final conditioning projection are zero-initialized.

That mirrors the original ControlNet idea: begin with a side branch whose contribution is initially approximately zero, so installing it does not immediately perturb the pretrained generator. ([arXiv][1])

### One training step

Current `sd-scripts/train_control_net.py` does essentially:

```text
training image
    ↓ VAE
clean latent
    ↓ noise
noisy latent ──────────────────┐
                               │
control image ─→ ControlNet    │
                    │          │
             down/mid residuals
                    │          │
                    ▼          ▼
                 frozen U-Net
                       │
                 noise prediction
                       │
                      loss
```

Only `controlnet.parameters()` are given to the optimizer; the base U-Net and text encoder are explicitly frozen.

But this is important:

```text
frozen U-Net
    ≠
irrelevant to backward
```

The loss is after the U-Net.

Gradient has to propagate:

```text
loss
 ↓
frozen U-Net operations
 ↓
injected residuals
 ↓
ControlNet
```

The U-Net weights don't accumulate updates, but its operations remain part of the differentiable path.

### Design pressure

This gives a very strong constraint:

> **Frozen execution is not the same as detached or no-gradient execution.**

And:

> **A trainable component can receive its supervision only through a frozen downstream component.**

That's a useful case that ordinary independent-component training doesn't expose.

---

# 3. Initialization itself is part of the recipe

ControlNet adds another wrinkle we haven't seen quite this cleanly elsewhere.

The side network is neither:

```text
randomly initialized independent model
```

nor simply:

```text
another reference to the base model
```

It is:

```text
pretrained backbone state
       │
       ├── retained as frozen base
       │
       └── copied into new trainable branch
```

plus newly created zero modules.

So one accepted model artifact produces two runtime parameter identities:

```text
base.weight
control.weight

initially equal
later diverge
```

This gives:

> **Preparation may clone selected pretrained state into a newly owned trainable component without creating an ongoing weight-sharing relationship.**

That belongs in the note. It's relevant to model preparation and artifact ownership.

---

# 4. Recipe — T2I-Adapter

T2I-Adapter deliberately makes the side network much smaller than ControlNet. The pretrained T2I generator remains frozen; the adapter converts an external control into multiscale features that get added into the frozen U-Net. ([arXiv][2])

The training topology is more like:

```text
control image
     ↓
lightweight adapter
     │
     ├─ feature scale 1
     ├─ feature scale 2
     ├─ feature scale 3
     └─ feature scale 4
             │
             ▼
       frozen U-Net
             │
             ▼
           loss
```

The official TencentARC training implementation literally does:

```python
down_block_additional_residuals = adapter(edge)

model_pred = unet(
    ...,
    down_block_additional_residuals=...
)
```

and the optimizer is constructed only from:

```python
adapter.parameters()
```

The paper's core recipe freezes the original diffusion model and learns the adapter. ([arXiv][2])

### The interesting training wrinkle: condition production

Their sketch recipe does not require the stored source to already be the final control representation.

It takes the image and runs a pretrained sketch detector:

```text
training image
     │
     ├──────────────────────→ target image latent
     │
     └→ frozen sketch detector
             ↓
          sketch
             ↓
      threshold / masking
             ↓
          adapter
```

So we now have a distinction between:

```text
raw conditioning source
        ≠
model-facing control representation
```

And the processor producing that representation may itself be:

```text
frozen neural model
algorithmic transform
offline preprocessing
live training-time transform
```

### Design pressure

> **A control-input producer is not necessarily the trainable control model.**

And:

> **Condition preprocessing may itself be a model execution role without belonging to the optimization graph.**

Very relevant to your data/preparation boundary research.

---

# 5. Paired data becomes part of the training contract

Classic ControlNet and LLLite-style training require a relationship like:

```text
target image
control image
caption
```

rather than an ordinary image/caption sample.

Current `sd-scripts` represents this explicitly through `conditioning_data_dir`; conditioning images share basenames with their target images. The current LLLite docs also note that those controls are resized alongside the training image. ([GitHub][3])

So:

```text
dataset item
{
    generation target,
    control source,
    text condition
}
```

The control source must generally undergo **the same relevant spatial transforms** as the target.

For example:

```text
target image ── crop(x,y,w,h) ──→ target
control image ─ crop(x,y,w,h) ──→ control
```

Independent random cropping would destroy the correspondence.

That means:

> **Some augmentations are relational operations over multiple sample fields, not independent transforms per representation.**

This is a nice addition to the data-pipeline research.

---

# 6. Recipe — IP-Adapter

This is useful because it completely changes the injection mechanism.

IP-Adapter is not primarily spatial residual injection. It introduces an **image-prompt pathway through cross-attention** and explicitly decouples text and image attention. The base diffusion model remains frozen. ([arXiv][4])

The official training code uses:

```text
reference image
      ↓
frozen CLIP vision encoder
      ↓
image embedding
      ↓
trainable image projection
      ↓
image tokens
      │
      ▼
trainable IP attention processors
      │
      ▼
frozen U-Net
```

while text independently goes through the frozen text encoder.

Current official tutorial code freezes:

```text
U-Net
VAE
text encoder
CLIP image encoder
```

and optimizes only:

```text
image_proj_model
IP attention modules
```

Those attention modules are installed into the frozen U-Net's cross-attention sites.

### Initialization is again interesting

The IP attention K/V projections are initialized from the pretrained U-Net's existing text-attention K/V weights.

So again:

```text
existing parameter
   ↓ copied
new parameter identity
   ↓
trained independently
```

But unlike ControlNet, the copied state isn't a huge network branch. It's selected attention projection state.

### Design pressure

> **A side network does not need to remain topologically outside the base model. It may install trainable modules into specific execution sites of an otherwise frozen component.**

That's an important distinction.

---

# 7. External encoder and trainable bridge are separate ownership roles

IP-Adapter gives another nice pipeline:

```text
reference image
     ↓
CLIP Vision              frozen
     ↓
image representation
     ↓
ImageProjModel           trainable
     ↓
image prompt tokens
     ↓
IP attention modules     trainable
     ↓
U-Net                    frozen
```

So:

```text
conditioning encoder
    ≠
conditioning adapter
    ≠
injection mechanism
```

All three can have different trainability and artifact roles.

That's basically the image-side equivalent of the Anima LLM-adapter lesson.

---

# 8. Conditioning dropout is training semantics

Official IP-Adapter training randomly drops image conditioning.

The dataset supplies `drop_image_embed`, and when selected the CLIP image embedding is replaced with zero before the trainable projection path.

Conceptually:

```text
reference image
     ↓
image encoder
     ↓
image embedding
     │
     ├── keep
     └── replace with zero
             ↓
        image adapter
```

This is effectively teaching the model both conditioned and unconditioned image-prompt behavior.

Likewise, T2I-Adapter has condition augmentation, and LLLite can have its own condition transformations.

So:

> **Condition dropout/augmentation belongs to the conditioning recipe, not necessarily to generic dataset augmentation.**

Especially because it operates on a particular representation boundary.

---

# 9. Recipe — Anima ControlNet-LLLite

This one is particularly useful because current upstream `sd-scripts` has a much more modern DiT implementation.

The current Anima implementation freezes the entire DiT and trains only `ControlNetLLLiteDiT`. ([GitHub][5])

Instead of a parallel copied backbone:

```text
control image
     ↓
shared conditioning trunk
     ↓
shared cond_emb
     │
     ├───────────────┬───────────────┐
     ▼               ▼               ▼
LLLite module    LLLite module    LLLite module
     │               │               │
 patch q/k/v       patch q/k/v     patch MLP/etc.
     │               │               │
     └──────── frozen Anima DiT ─────┘
```

The side network scans the existing DiT and installs small trainable modules onto selected `Linear` operations.

Current v2 includes:

* shared conditioning-image trunk,
* per-module depth embedding,
* small down/mid/up path,
* FiLM modulation,
* zero-initialized FiLM/output paths,
* selectable target layer classes. ([GitHub][5])

### This is not merely LoRA

The original Linear still runs:

```text
y_base = original_linear(x)
```

and LLLite computes a condition-dependent correction:

```text
y = y_base + control_correction(x, cond)
```

So its trainable parameters depend simultaneously on:

```text
current base activation
+
external condition representation
```

That's different from ordinary low-rank weight adaptation.

### Design pressure

> **A side-control module may alter the execution of an existing operator rather than supply an explicit side-output tensor to the model API.**

This means the integration mechanism itself is part of the recipe.

---

# 10. Installation can change the runtime graph without changing base parameter ownership

ControlNet looks like:

```text
control_net(...)
   ↓ residuals
base_model(... residuals ...)
```

LLLite looks like:

```text
base_model.linear.forward
          ↓ patched/wrapped
base + control correction
```

IP-Adapter looks like:

```text
base attention processor
          ↓ replaced
text attention + image attention
```

These are three substantially different integration mechanisms:

```text
explicit input/output composition

operator replacement

operator wrapping/injection
```

So I'd capture:

> **A generic “control model slot” is probably too weak. Control training also requires an installation/injection contract describing where and how the side computation participates in base execution.**

That feels like the core architecture finding from this topic.

---

# 11. The base model can be frozen but still dominate training memory

This should definitely be recorded because it is counterintuitive.

In all these recipes, only the small control side may be optimized.

But:

```text
frozen base
    still needs forward activations
    still participates in gradient propagation
    still may need checkpointing
    still dominates parameter residency
```

Anima LLLite makes this painfully obvious: the current implementation supports gradient checkpointing over the frozen DiT, while only the LLLite parameters are optimizer-owned. Its documentation currently rules out several offload/DeepSpeed paths despite the tiny trainable artifact because the frozen DiT still participates in the live graph. ([GitHub][5])

So:

> **Trainable parameter count is not a reliable proxy for training execution memory.**

That's a very useful constraint.

---

# 12. Control strength is not necessarily learned state

Another subtle distinction:

Control systems frequently expose inference-time values such as:

```text
control strength
conditioning scale
step-range ratio
```

Current `sd-scripts` ControlNet inference, for example, lets users change both the multiplier and the fraction of denoising steps during which ControlNet applies. ([GitHub][6])

Those aren't necessarily learned parameters.

So:

```text
control network weights
        ≠
application policy
```

The same artifact can be applied at:

```text
0.3 strength
1.0 strength
first half of trajectory
whole trajectory
```

### Design pressure

> **Artifact state and runtime control policy should not be conflated.**

This is analogous to sampler configuration versus trained diffusion model state.

---

# 13. Multiple controls make the execution relation many-to-one

Both ControlNet and T2I-Adapter were explicitly designed with composability in mind. The original ControlNet paper demonstrates multiple conditions, while T2I-Adapter emphasizes composable adapters. ([arXiv][1])

That means inference may become:

```text
depth ControlNet ─────┐
edge ControlNet ──────┤
pose ControlNet ──────┼──→ one frozen generator
reference adapter ────┘
```

Even if each was trained separately.

So:

> **One base execution may consume zero, one, or several independently owned control artifacts.**

That argues against baking one specific control network into the identity of the base model.

---

# 14. Artifacts differ quite a bit

Classic ControlNet saves a substantial side network that contains copied pretrained structure plus learned control layers.

T2I-Adapter saves the compact adapter.

IP-Adapter's artifact contains at least:

```text
image projection weights
IP attention weights
```

while requiring the original image encoder and compatible base model separately.

Anima LLLite saves only:

```text
shared conditioning trunk
per-target LLLite modules
architecture metadata
```

The current sd-scripts format records things like target layers, embedding dimensions, inpaint configuration, and architecture version so inference can reconstruct the injected topology. ([GitHub][5])

That's important:

> **A control artifact may require structural metadata describing how to reinstall it into a base model, not merely a state dictionary.**

Very relevant to your model preparation/publication work.

---

# 15. Cross-case findings

I’d end with a compact table like this:

| Tempting assumption                                              | Evidence                                                                           |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| A control network is always a second full model                  | False; T2I-Adapter and LLLite can be much smaller.                                 |
| Side networks inject the same way                                | False; residuals, feature pyramids, attention K/V and patched operators all occur. |
| Frozen base means `no_grad()`                                    | False; gradient must often flow through the frozen base to the side network.       |
| Side network starts independently                                | False; ControlNet and IP-Adapter copy selected base weights as initialization.     |
| Control input is already model-ready                             | False; it may require edge/depth/vision encoders or other preprocessing.           |
| Condition preprocessing belongs to the trainable control network | False; preprocessors can be frozen or external.                                    |
| One training image is enough                                     | Often false; paired target/control correspondence is part of the training sample.  |
| Augmentations can be applied independently                       | False when target and control must stay spatially aligned.                         |
| Trainable parameter count predicts memory cost                   | False; the frozen base still executes in the differentiable graph.                 |
| Control artifact = base model modification                       | False; side artifacts may remain separately installable/composable.                |
| A control artifact is only weights                               | False; installation topology/metadata can be required.                             |
| One base model has one control                                   | False; independently trained controls may compose around one backbone.             |

Then I’d end the note with three strong architecture conclusions:

> **Control capability should not be modeled as one special model slot. A recipe may introduce a side component plus an explicit injection relationship into an existing model's execution graph.**

> **Trainability and gradient participation are separate: the base generator may be frozen while still remaining an essential differentiable path between the trainable control component and the loss.**

> **Condition representation, condition producer, trainable side network, injection mechanism, and runtime application policy are separate roles and should not be collapsed into “conditioning.”**

And perhaps the one most relevant to your current rework:

```text
control recipe
    owns:
        control representation
        side component(s)
        injection relationships
        relevant training transforms

base model strategy
    exposes:
        valid injection surfaces / representations

Trainer
    owns:
        optimization and lifecycle

backend
    realizes:
        placement / wrapping / distribution
```

---

[1]: https://arxiv.org/abs/2302.05543?utm_source=chatgpt.com "Adding Conditional Control to Text-to-Image Diffusion Models"
[2]: https://arxiv.org/abs/2302.08453?utm_source=chatgpt.com "T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models"
[3]: https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_lllite_README.md?utm_source=chatgpt.com "sd-scripts/docs/train_lllite_README.md at main · kohya-ss/sd-scripts · GitHub"
[4]: https://arxiv.org/abs/2308.06721?utm_source=chatgpt.com "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models"
[5]: https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_control_net_lllite.md?utm_source=chatgpt.com "sd-scripts/docs/anima_train_control_net_lllite.md at main · kohya-ss/sd-scripts · GitHub"
[6]: https://github.com/kohya-ss/sd-scripts/blob/main/docs/gen_img_README.md?utm_source=chatgpt.com "sd-scripts/docs/gen_img_README.md at main · kohya-ss/sd-scripts · GitHub"
