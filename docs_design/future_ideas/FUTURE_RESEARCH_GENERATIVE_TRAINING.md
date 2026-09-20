# Future Research Directions for Generative Image Training

> **Working research map — August 2026**  
> A watchlist of recent ideas that may matter over the next few years, with emphasis on what is actually relevant to a general-purpose image-training repository rather than only to frontier-scale model development.

---

## Working thesis

The next generation of image models probably will not be defined only by **larger MMDiTs**.

A lot of the more interesting recent work attacks the surrounding stack:

- the quality of the latent representation,
- how aggressively images are compressed,
- what hidden representations the generator is encouraged to learn,
- how the diffusion/flow path is parameterized,
- how training time/noise levels are sampled and weighted,
- how many network evaluations inference actually needs,
- whether image computation should stay flat or become multiscale again,
- and whether text, images, video, and other modalities should share one generative framework.

A plausible future stack looks less like:

```text
image
  ↓
old f8 VAE
  ↓
huge flat MMDiT
  ↓
rectified flow
  ↓
20–50 model evaluations
  ↓
image
```

and more like:

```text
image
  ↓
semantic / highly compressed representation
  ↓
multiscale or adaptive backbone
  ↓
representation-aware training
  ↓
better flow / trajectory objective
  ↓
1–4 model evaluations
  ↓
image
```

The individual ideas below should be treated as **research directions**, not as guaranteed winners.

---

# 1. Representation Alignment

## REPA

**REPA — Representation Alignment for Generation** adds an auxiliary objective that aligns hidden states inside a diffusion/flow model with representations from a strong frozen vision encoder.

Paper:

- [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://arxiv.org/abs/2410.06940)
- [Official REPA repository](https://github.com/sihyun-yu/REPA)

Conceptually:

```text
clean image ───────→ frozen vision encoder ───────→ target representation
                                                         │
                                                         │ alignment loss
                                                         ↓
noisy latent ──────→ denoiser ──────→ hidden state ───→ projector
                         │
                         └──────────────────────────────→ normal generative loss
```

The important idea is not merely "use DINO as another loss."

It is that a generative model otherwise spends a substantial amount of training learning useful visual representations **from the denoising objective alone**, even though self-supervised vision models may already contain better ones.

This changes an architectural question:

> How do we force the generator to learn good internal representations?

into:

> Which external representation should the generator inherit or align with?

The original REPA work reported extremely large convergence improvements on SiT/DiT-style models, including matching a much longer baseline training run in a fraction of the steps.

### Why this may become important

Representation learning and generation have historically been treated as related but separate problems.

REPA suggests that this separation may be wasteful.

Future generators may routinely be trained with:

- frozen semantic teachers,
- spatial representation teachers,
- multimodal teachers,
- domain-specific visual encoders,
- or jointly learned representation objectives.

### Training-repo relevance: **High**

REPA is unusually suitable as an actual trainer feature.

Likely requirements:

```text
auxiliary frozen model
    └── vision teacher

trainable auxiliary module
    └── representation projector

denoiser
    └── intermediate activation access

loss
    ├── normal diffusion / flow loss
    └── representation alignment loss

optional cache
    └── teacher features
```

A good trainer should not need to special-case "REPA" everywhere. The model/training strategy should be able to declare the auxiliary components and losses it needs.

---

## U-REPA

For U-Net training, **U-REPA** is arguably more directly relevant than the original paper.

Paper:

- [U-REPA: Aligning Diffusion U-Nets to ViTs](https://arxiv.org/abs/2503.18414)
- [Official U-REPA repository](https://github.com/YuchuanTian/U-REPA)

A U-Net has a very different hidden-state structure from an isotropic DiT:

```text
high-resolution encoder
        ↓
medium-resolution encoder
        ↓
low-resolution bottleneck
        ↓
medium-resolution decoder
        ↓
high-resolution decoder
```

The authors found that simply copying the DiT REPA setup is not ideal. Their method focuses alignment around the **middle stage / bottleneck**, handles resolution mismatch, and adds a manifold-style representation loss.

### Why this is interesting

This is evidence that representation alignment is not just a transformer trick.

It also raises much more interesting architectural questions:

- Which U-Net block should align with the teacher?
- Should multiple scales be supervised?
- Should teacher features be resized to match the U-Net or vice versa?
- Should projectors be convolutional?
- Should encoder and decoder blocks use different targets?
- Does the best alignment point change with skip-connection design?

### Repo implications

Intermediate-feature extraction should ideally not assume:

```text
model.blocks[n] -> one flat sequence
```

A general system may need named or strategy-defined feature taps:

```text
encoder.stage_2
middle
decoder.stage_1
transformer.block_18
```

---

## iREPA: spatial structure may matter more than "semantic strength"

Paper:

- [What matters for Representation Alignment: Global Information or Spatial Structure?](https://arxiv.org/abs/2512.10794)
- [iREPA project page](https://end2end-diffusion.github.io/irepa)

This is one of the more interesting follow-ups.

The authors compare many vision encoders and argue that **spatial organization of patch representations** is a stronger predictor of usefulness for generative alignment than conventional global semantic/classification strength.

Their iREPA modifications are deliberately simple:

- replace the standard MLP projector with a convolutional projection,
- spatially normalize the teacher representation.

### Research questions worth testing

Especially for illustration/anime/tag-heavy datasets:

- DINO vs SigLIP vs CLIP vs MAE,
- general-purpose vs domain-specific visual encoder,
- global semantic features vs spatial patch features,
- frozen teacher vs slowly updated teacher,
- one teacher vs multiple teachers,
- middle-block alignment vs multiscale alignment,
- representation alignment strength over timestep/noise level.

### Potential experiment

Keep everything fixed except the teacher:

```text
generator
dataset
optimizer
training schedule
REPA weight
projector
```

Compare:

```text
DINO
SigLIP
CLIP
MAE
domain-specific encoder
```

Track:

- convergence speed,
- reconstruction/detail quality,
- composition,
- anatomy,
- text/tag adherence,
- concept retention,
- diversity,
- behavior at fixed compute.

---

# 2. Flow Matching as a General Training Framework

The biggest conceptual topic worth understanding properly is probably **the relationship between diffusion, flow, parameterization, and trajectory design**.

Useful chain to study:

```text
DDPM
├── epsilon prediction
├── x0 prediction
└── v-prediction
        ↓
EDM / continuous noise parameterizations
        ↓
probability-flow ODEs
        ↓
Flow Matching
        ↓
Rectified Flow
        ↓
Consistency / flow-map / few-step methods
        ↓
MeanFlow-like objectives
```

The important abstraction is often:

```text
sample data x
sample source/noise z
sample time t
        ↓
choose interpolation path x_t
        ↓
construct training input
        ↓
define target field / prediction
        ↓
weight loss
```

Instead of thinking:

```python
if model == "sdxl":
    use_this_loss()
elif model == "sd3":
    use_that_loss()
elif model == "flux":
    use_another_loss()
```

a trainer can think in terms of components such as:

```text
path
time/noise sampler
prediction parameterization
target construction
loss weighting
conditioning dropout
```

### Training-repo relevance: **Very high**

This is probably more important to a general trainer than adding one fashionable backbone.

A clean implementation makes it much easier to support:

- epsilon prediction,
- v-prediction,
- x0 prediction,
- flow prediction,
- rectified flow,
- custom sigma distributions,
- custom timestep weighting,
- future few-step objectives.

### Recommended research order

1. DDPM parameterizations
2. EDM
3. Flow Matching
4. Rectified Flow
5. timestep/noise sampling
6. loss weighting
7. consistency-style training
8. MeanFlow / flow-map methods

---

# 3. MeanFlow and Few-Step Generation

Paper:

- [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)

Most current diffusion/flow systems learn a field that is repeatedly evaluated during inference:

```text
noise
 ↓ network
state
 ↓ network
state
 ↓ network
state
 ↓ ...
image
```

MeanFlow instead models an **average velocity across an interval**, allowing the model to make a much larger movement through the generative trajectory in one evaluation.

The paper reports strong one-step ImageNet generation **without requiring a pretrained teacher or distillation pipeline**.

### Why this matters

If this line of research scales successfully, it changes inference economics much more dramatically than a modest backbone optimization.

Approximate comparison:

```text
current:
12B model × 30 evaluations

possible future:
3–12B model × 1–4 evaluations
```

That matters enormously for:

- local generation,
- high-resolution workflows,
- video,
- interactive editing,
- agentic image generation,
- serving cost,
- iterative search.

### Training-repo relevance: **Medium now / potentially very high later**

It is worth designing training abstractions so objectives like this do not require rewriting the entire trainer.

But ordinary flow matching should be clean before implementing experimental one-step objectives.

---

# 4. Representation Autoencoders

Paper:

- [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690)
- [ICLR 2026 paper page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3c4141c12660ad3625eb4ae845e0a6f9-Abstract-Conference.html)

Traditional latent diffusion uses a VAE mostly as a **reconstruction-oriented compressor**.

```text
pixels
 ↓
VAE encoder
 ↓
small reconstruction latent
 ↓
generator
```

Representation Autoencoders ask whether the latent itself should already contain strong visual semantics.

```text
pixels
 ↓
pretrained representation encoder
 ↓
semantic visual latent
 ↓
generator
 ↓
trained decoder
```

The paper explores encoders including:

- DINO,
- SigLIP,
- MAE.

### Why this could matter

A stronger latent may reduce how much representational work the denoiser must learn from scratch.

Potential consequences:

- faster convergence,
- better semantics,
- smaller required generator,
- stronger transfer,
- better multimodal compatibility.

### Important tension

RAEs often create **higher-dimensional** latent representations.

That is almost the opposite of the "compress everything as much as possible" direction.

This creates an interesting unresolved axis:

```text
                  richer semantics
                       ↑
                       │
                       │        RAE
                       │
                       │
old VAE ───────────────┼────────────→ compression
                       │
                       │
                       │              DC-AE
```

A major future question is whether we can get both:

> **semantically strong + aggressively compressed latents**

### Training-repo relevance: **Medium to high**

Avoid hardcoding the latent stage as specifically:

```python
vae.encode(image).latent_dist.sample()
```

The more future-proof abstraction is:

```text
image -> latent representation
latent representation -> image
```

Different model families may implement this very differently.

---

# 5. Deep Compression Autoencoders

Paper:

- [Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](https://arxiv.org/abs/2410.10733)

The common Stable Diffusion-style VAE uses roughly **8× spatial compression**.

For a 1024×1024 image:

```text
f8:
1024 × 1024
      ↓
128 × 128 latent grid
```

That is still a large sequence for transformer processing.

DC-AE explores substantially stronger spatial compression, up to **128×** in the paper, while attempting to preserve useful reconstruction quality.

The authors report very large training/inference speedups in downstream latent generative models compared with an f8 VAE.

### Why this matters

For image transformers, token count is brutally expensive.

```text
128 × 128 = 16,384 positions
32 × 32   = 1,024 positions
16 × 16   =   256 positions
```

Better compression can therefore change the required size and compute of the generator itself.

Potential benefits:

- smaller image generators,
- much cheaper attention,
- higher native resolution,
- cheaper video,
- more realistic local training.

### Training-repo relevance: **Medium**

Mostly model-family territory, but generic code should not assume a particular latent resolution or channel count.

---

# 6. Multiscale Transformers and the Return of U-Net Ideas

One reason U-Nets remain attractive is that they naturally allocate different amounts of computation to different spatial scales.

```text
high resolution
      ↓
medium
      ↓
low / global
      ↓
medium
      ↓
high / detail
```

Flat DiTs largely discard this bias and process a large token grid repeatedly.

Recent work is increasingly reintroducing hierarchical computation.

---

## HDiT — Hourglass Diffusion Transformer

Paper:

- [Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers](https://arxiv.org/abs/2401.11605)

HDiT explicitly tries to bridge:

- the efficiency of convolutional U-Nets,
- the scaling behavior of transformers.

It uses an hourglass-style structure so expensive global processing does not have to occur at the full spatial token count everywhere.

---

## UDT — U-Net Diffusion Transformer

Paper:

- [UDT: Reconciling U-Nets and Diffusion Transformers with Data-Adaptive Token Reduction](https://arxiv.org/abs/2608.01298)

This is especially worth watching because it is extremely recent.

UDT combines DiT-style blocks with an explicit U-Net-like encoder/decoder hierarchy and **data-adaptive token merging** for down/up-sampling.

The direction is almost exactly:

```text
transformer
+
U-Net hierarchy
+
adaptive token reduction
+
REPA-compatible representation learning
```

### Why this matters

The eventual "post-flat-DiT" architecture may simply rediscover a lot of the reasons U-Nets were good:

- multiscale features,
- local detail processing,
- cheap global structure processing,
- skip connections,
- encoder/decoder asymmetry.

### Training-repo relevance: **Low directly / important architecturally**

The trainer should avoid assuming:

- all hidden states use one resolution,
- every block has the same token count,
- feature extraction is uniform,
- the denoiser has one simple sequential block list.

This also matters for REPA/U-REPA support.

---

# 7. Timestep / Sigma Sampling and Loss Weighting

This is less exciting than architecture papers and probably **more useful for actual trainer development**.

Training behavior can change substantially depending on:

- timestep distribution,
- sigma distribution,
- SNR,
- loss weighting,
- interpolation path,
- prediction parameterization,
- conditioning dropout,
- CFG dropout probability.

These choices are often buried inside model-specific training scripts even though they are fundamental training semantics.

### Research topics

Study the differences between:

- uniform timestep sampling,
- logit-normal sampling,
- EDM-style sigma sampling,
- min-SNR weighting,
- P2 weighting,
- flow-matching time distributions,
- shifted timestep distributions,
- resolution-dependent timestep shifts.

### Training-repo relevance: **Extremely high**

These are ideal candidates for reusable, declarative trainer components.

For example:

```text
TrainingStrategy
├── TimeSampler
├── NoisePath
├── PredictionTarget
├── LossWeighting
└── ConditioningDropout
```

The exact abstraction does not have to look like this, but these concerns should not be scattered through model-specific trainer code.

---

# 8. Continuous Autoregressive Image Generation

Paper:

- [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)
- [Official MAR repository](https://github.com/LTH14/mar)

Image autoregressive models usually use discrete VQ tokens.

MAR demonstrates that autoregression does not fundamentally require discrete image tokens.

Instead, a diffusion-style loss models the probability distribution of each **continuous** image token.

Conceptually:

```text
previous continuous patches
          ↓
autoregressive model
          ↓
distribution for next patch
          ↓
small diffusion process
          ↓
next continuous patch
```

### Why this matters

The traditional divide:

```text
autoregressive -> discrete tokens
diffusion       -> continuous latents
```

is not fundamental.

Future models may mix:

- autoregressive structure,
- continuous representations,
- diffusion/flow losses,
- masked modeling.

### Training-repo relevance: **Low now**

This is closer to a separate model family than a small trainer feature.

Still useful to understand because the distinction between "AR model" and "diffusion model" is becoming less clean.

---

# 9. Unified Continuous + Discrete Multimodal Models

## Transfusion

Paper:

- [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039)
- [ICLR 2025 paper page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/12678c3948153f4bc391f51e2082bd6e-Abstract-Conference.html)

Transfusion trains one transformer across:

- discrete text,
- continuous images.

Text uses next-token prediction while image regions use a diffusion objective.

```text
text tokens ──────→ language-model loss
       │
       ├──── shared transformer
       │
image patches ────→ diffusion loss
```

### Why this matters

It avoids forcing every modality into the same representation purely for architectural convenience.

A future multimodal model could naturally combine:

```text
text       -> categorical token loss
images     -> flow/diffusion loss
audio      -> continuous generative loss
video      -> structured temporal objective
```

while sharing large parts of the backbone.

### Training-repo relevance: **Medium in the long term**

This suggests a future trainer may eventually need multiple simultaneous objectives attached to different portions of one sequence/batch.

---

# 10. Diffusion Forcing and Sequence / World Models

Paper:

- [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://arxiv.org/abs/2407.01392)
- [Project page](https://boyuan.space/diffusion-forcing)

Normal diffusion commonly gives one example/state a shared noise level.

Diffusion Forcing allows **different tokens in a sequence to have different noise levels**.

Example:

```text
past                                future

clean      clean      clean     noisy     noisier     noise
██████     ██████     ██████    ▓▓▓▓▓     ▒▒▒▒▒      ░░░░░
```

This makes it possible to combine properties of:

- causal next-token prediction,
- full-sequence diffusion,
- variable-horizon generation,
- planning,
- long video rollout.

### Why this matters

This is more relevant to:

- video,
- robotics,
- agents,
- simulation,
- learned world models,

than to a conventional text-to-image trainer.

### Training-repo relevance: **Low for current image work**

Worth following if the project expands into temporal models.

---

# 11. Native-Resolution Training

Paper:

- [Native-Resolution Image Synthesis](https://arxiv.org/abs/2506.03131)

Most image generators are still conceptually tied to a small set of canonical training resolutions.

NiT instead models variable-length visual sequences and is explicitly trained across different native image resolutions and aspect ratios.

### Why this matters

Real image distributions are not naturally:

```text
1024 × 1024
1024 × 1024
1024 × 1024
```

That is largely a training convenience.

Future models may treat:

- aspect ratio,
- resolution,
- token count,

as normal variable dimensions rather than special cases.

### Training-repo relevance: **Medium**

This is especially relevant to:

- bucket design,
- batch construction,
- positional encodings,
- resolution conditioning,
- timestep shifts,
- token-budget-based batching.

A trainer should ideally not assume one fixed token count per batch forever.

---

# 12. Research Themes Worth Tracking Beyond Individual Papers

Individual papers come and go. These broader directions are more useful to keep on a long-term watchlist.

## Representation-guided generation

Questions:

- Can generation reuse self-supervised vision representations?
- Which representation properties matter?
- Can the teacher eventually be removed?
- Can representation learning and generative learning become one objective?

Watch:

- REPA
- U-REPA
- iREPA
- RAE

---

## Better latent spaces

Questions:

- How compressed should image latents be?
- Should latents prioritize reconstruction or semantics?
- Can we get semantic + highly compressed latents simultaneously?
- Should latent channels have explicit hierarchy/meaning?

Watch:

- RAE
- DC-AE
- learned semantic codecs
- hierarchical latents

---

## Fewer model evaluations

Questions:

- Is iterative numerical integration fundamentally necessary?
- Can models learn trajectory maps directly?
- Can one-step generation match full flow models without distillation?

Watch:

- MeanFlow
- consistency models
- flow-map approaches
- trajectory distillation

---

## Multiscale computation

Questions:

- Why process every token at equal cost?
- Can global structure be solved cheaply at low resolution?
- Can high-resolution compute be reserved for detail?
- Should token count change dynamically?

Watch:

- HDiT
- UDT
- hierarchical DiTs
- token merging
- dynamic-resolution models
- local/global attention hybrids

---

## Architecture hybrids

The future may not be:

```text
U-Net vs Transformer
```

but:

```text
convolution
+
local attention
+
global transformer
+
multiscale hierarchy
+
adaptive token reduction
```

The winning architecture may keep the best inductive biases of U-Nets while retaining transformer scaling and multimodal flexibility.

---

# 13. Direct Priorities for the Training Repository

## Priority 1 — Understand and cleanly model flow/diffusion training semantics

**Highest value.**

Research:

- prediction parameterizations,
- timestep/sigma sampling,
- interpolation paths,
- loss weighting,
- rectified flow,
- flow matching.

Potential architecture:

```text
TrainingStrategy
├── time/noise sampling
├── path construction
├── model input preparation
├── target construction
├── prediction interpretation
└── loss weighting
```

Goal:

> New training objectives should be possible without modifying the generic training loop everywhere.

---

## Priority 2 — Prototype REPA / U-REPA support

**Best experimental feature candidate.**

Needed:

- frozen vision encoder,
- activation extraction,
- projector,
- auxiliary loss,
- auxiliary checkpoint state.

Optional optimization:

### Cache teacher representations

For a frozen deterministic teacher:

```text
image
 ↓
teacher
 ↓
feature tensor
```

does not need to be recomputed every epoch.

Possible cache:

```text
dataset item
├── VAE latent
├── text embedding
└── teacher representation
```

This could make REPA-style training much more practical on local hardware.

---

## Priority 3 — Make latent encoders generic

Avoid assuming:

```text
autoencoder == Stable Diffusion VAE
```

Prefer a model-family interface capable of representing:

```text
image -> latent
latent -> image
```

without requiring the latent to be:

- Gaussian,
- four-channel,
- f8,
- spatially shaped like SD,
- sampled from `latent_dist`.

This prepares the repository for:

- RAE,
- DC-AE,
- future learned codecs,
- non-VAE latent spaces.

---

## Priority 4 — Treat timestep sampling and weighting as first-class configuration

Support experimentation without model-code edits.

Possible config concepts:

```yaml
time_sampling:
  type: logit_normal
  mean: 0.0
  std: 1.0

loss_weighting:
  type: min_snr
  gamma: 5.0
```

Exact naming is unimportant.

The important point is that these choices should be explicit rather than buried inside a model implementation.

---

## Priority 5 — Keep feature extraction architecture-agnostic

Future models may have:

- U-Net stages,
- transformer blocks,
- merged-token stages,
- multiscale bottlenecks,
- separate modality streams.

For representation losses, debugging, and research instrumentation, consider an abstraction like:

```text
request features:
    "middle"
    "encoder.2"
    "decoder.1"
    "transformer.18"
```

where the model strategy resolves those names.

---

# 14. Suggested Experimental Roadmap

## Experiment A — U-REPA on an existing U-Net

Goal:

> Determine whether representation alignment improves convergence on a real non-ImageNet dataset.

Compare:

```text
baseline U-Net
vs
U-Net + U-REPA
```

Control:

- identical data,
- identical optimizer,
- identical seed where possible,
- identical noise schedule,
- identical total training compute.

Measure quality periodically instead of only at the final checkpoint.

---

## Experiment B — Teacher comparison

Using the same REPA setup:

```text
DINO
SigLIP
CLIP
MAE
domain-specific encoder
```

Question:

> What representation characteristics actually help the target dataset?

This could be more informative than blindly copying the teacher used by the original paper.

---

## Experiment C — Timestep sampler sweep

Keep the model fixed.

Compare:

```text
uniform
logit-normal
custom sigma distribution
shifted distribution
```

Measure:

- loss by noise level,
- validation behavior,
- convergence speed,
- sample quality,
- prompt adherence.

This is cheap relative to architectural experiments and directly useful to trainer development.

---

## Experiment D — Loss-weighting sweep

Compare:

```text
unweighted
SNR-based
min-SNR
flow-specific weighting
```

Plot loss and quality against training time.

The goal is not merely finding one best number, but verifying the trainer exposes the correct abstractions.

---

## Experiment E — Latent-space comparison

Longer-term:

```text
classic VAE
vs
high-compression autoencoder
vs
semantic representation autoencoder
```

Keep the generator as comparable as possible.

Measure:

- reconstruction error,
- semantic retention,
- required generator capacity,
- training speed,
- memory usage,
- final image quality.

---

# 15. Reading Order

If only reading a small number of papers, this order gives a coherent progression.

### Representation learning

1. [REPA](https://arxiv.org/abs/2410.06940)
2. [U-REPA](https://arxiv.org/abs/2503.18414)
3. [iREPA](https://arxiv.org/abs/2512.10794)

### Training objectives

4. Flow Matching / Rectified Flow background
5. [MeanFlow](https://arxiv.org/abs/2505.13447)

### Latent representations

6. [Deep Compression Autoencoder](https://arxiv.org/abs/2410.10733)
7. [Representation Autoencoders](https://arxiv.org/abs/2510.11690)

### Architecture

8. [HDiT](https://arxiv.org/abs/2401.11605)
9. [UDT](https://arxiv.org/abs/2608.01298)

### Alternative generative formulations

10. [MAR — Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)
11. [Transfusion](https://arxiv.org/abs/2408.11039)
12. [Diffusion Forcing](https://arxiv.org/abs/2407.01392)
13. [Native-Resolution Image Synthesis](https://arxiv.org/abs/2506.03131)

---

# 16. Current Personal Research Priority

If the goal is **useful repo work rather than paper collecting**, the most valuable path currently looks like:

```text
1. Flow/diffusion training semantics
        ↓
2. timestep + loss weighting abstractions
        ↓
3. REPA / U-REPA implementation
        ↓
4. teacher representation caching
        ↓
5. generic latent encoder support
        ↓
6. RAE / DC-AE experiments
        ↓
7. multiscale / hybrid architecture experiments
        ↓
8. few-step objectives such as MeanFlow
```

The first four can improve the trainer even if none of the newer research directions ultimately becomes dominant.

---

# 17. Things I Would Not Rush to Implement Yet

### Giant MMDiT-specific machinery

Supporting existing MMDiT families is useful, but architecture should not assume that "future image model" means "larger flat transformer."

### Mamba / SSM image backbones

Interesting, but there is still much less evidence that they will displace transformer/U-Net hybrids than there is for improvements to representations, compression, and flow objectives.

### One-step generation before ordinary flow support is clean

MeanFlow is exciting, but implementing experimental objectives on top of messy training semantics creates technical debt.

### Hardcoding one representation teacher

REPA should ideally support a teacher interface rather than:

```python
teacher = DINOv2(...)
```

The teacher itself is an experimental variable.

---

# 18. Larger Prediction

The most likely future is not that one paper replaces diffusion transformers overnight.

It is that several ideas gradually merge:

```text
better visual encoder / latent representation
                 +
aggressive but intelligent compression
                 +
representation-aware training
                 +
multiscale / adaptive computation
                 +
better flow objective
                 +
few-step sampling
                 +
multimodal conditioning
```

At that point, calling the model simply a "DiT" or "U-Net" may stop being very informative.

The interesting research question is therefore not only:

> **What backbone wins?**

but:

> **Which parts of the generative stack are currently wasting learning capacity or compute, and can we remove that waste?**

That is probably the more useful lens for deciding what deserves implementation effort.
