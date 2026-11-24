# Research topic: Autoencoders / VAEs

*Claim labels:** **Observed** = directly established by source. **Inferred** = follows from the implementation but is not explicitly stated. **Unknown** = the examined sources do not establish it.

| Recipe                             | What it tests                                                                    |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| CompVis `AutoencoderKL`            | classic continuous KL-regularized image VAE + perceptual/GAN loss                |
| Taming Transformers ImageNet VQGAN | discrete learned codebook instead of Gaussian posterior                          |
| Open-Sora Plan WFVAE               | temporal/spatial video VAE + wavelet path + alternating GAN optimization         |
| Stable Audio 2.0 VAE               | waveform VAE + spectral losses + explicit encoder/decoder/bottleneck composition |

---

# Recipe 1 — CompVis KL image autoencoder, 32×32×4 latent

**Implementation examined:** `CompVis/latent-diffusion`, commit `a506df57`, config [`autoencoder_kl_32x32x4.yaml`](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/configs/autoencoder/autoencoder_kl_32x32x4.yaml), model [`ldm/models/autoencoder.py`](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/autoencoder.py), loss [`contperceptual.py`](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/losses/contperceptual.py).

The concrete recipe trains on 256×256 ImageNet images with:

```text
embed_dim = 4
downsampling factor = 8
batch size = 12
gradient accumulation = 2
KL weight = 1e-6
GAN starts at step 50001
```

## 1. Components

**Observed:** the primary reconstruction model consists of:

```text
encoder
  ↓
quant_conv
  ↓
DiagonalGaussianDistribution
  ↓ sample z
post_quant_conv
  ↓
decoder
```

There is no discrete quantizer despite the historical `quant_conv` naming.

**Observed:** the encoder produces twice the latent channel count because `double_z=True`; those channels represent the parameters of a diagonal Gaussian posterior.

**Inferred:** from the 256 input resolution, three downsampling levels and `embed_dim=4`, the sampled representation is approximately:

```text
[B, 4, 32, 32]
```

**Observed:** two additional loss-side neural components participate:

* frozen LPIPS perceptual network;
* trainable patch discriminator.

LPIPS parameters are explicitly `requires_grad=False`, but its forward remains part of the reconstruction-loss graph.

That gives a useful distinction:

```text
LPIPS weights:
    frozen

LPIPS computation:
    still differentiable w.r.t. reconstructed image
```

## 2. Data and one training step

**Observed:** the dataset supplies 256×256 images. `get_input()` rearranges the incoming image representation to:

```text
[B,C,H,W]
```

The exact numerical normalization range is not established by this model/config alone.

A forward is:

```text
image
 ↓ encoder
posterior parameters
 ↓ sample
z
 ↓ decoder
reconstruction
```

The generator-side loss combines:

```text
pixel L1
+ LPIPS perceptual loss
+ KL(posterior || N(0,1))
+ adversarial generator loss
```

The GAN contribution is gated by global training step.

**Observed:** the discriminator separately sees real images and detached reconstructions.

## 3. Optimization and runtime

**Observed:** there are **two Adam optimizers**.

Generator-side optimizer:

```text
encoder
decoder
quant_conv
post_quant_conv
```

Discriminator optimizer:

```text
discriminator
```

both use betas `(0.5, 0.9)`.

**Interesting observed mismatch:** `LPIPSWithDiscriminator` defines a learnable `logvar` parameter and `training_step()` comments that the autoencoder branch trains “encoder+decoder+logvar”, but `configure_optimizers()` does **not** include the loss module's `logvar` parameter.

So according to the implementation examined, `logvar` receives no optimizer step despite the comment.

**Observed:** LPIPS must remain loaded and execute during generator training despite having no optimized parameters.

## 4. Structural changes during a run

**Observed:** adversarial training effectively activates at step **50,001**.

Before then:

```text
reconstruction + perceptual + KL
```

Afterward:

```text
reconstruction + perceptual + KL + GAN
```

The discriminator component exists from initialization, but `disc_factor` is zero before the threshold.

Optimizer topology itself does not change.

## 5. Outputs and restoration

**Observed:** the useful downstream representation model consists of the encoder, decoder and latent projection layers. A latent diffusion model can subsequently use this autoencoder as its first-stage model.

The discriminator and LPIPS network are training-only dependencies; they are unnecessary to encode/decode images after training.

**Unknown:** the model-specific source does not establish the minimal exact state needed for bit-identical continuation beyond what PyTorch Lightning normally checkpoints.

## 6. Design pressure

> Can components participate only in the **training objective** without being part of the artifact's inference graph?

And:

> Can a frozen component remain inside the autograd path, so “not optimized” does not imply “run under `no_grad()`”?

That distinction is probably important for your ownership model.

---

# Recipe 2 — ImageNet VQGAN with 1024-entry codebook

**Implementation examined:** `CompVis/taming-transformers`, commit `3ba01b24`, config [`imagenet_vqgan.yaml`](https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/configs/imagenet_vqgan.yaml), [`taming/models/vqgan.py`](https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/models/vqgan.py), [`vqperceptual.py`](https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/losses/vqperceptual.py).

Concrete recipe:

```text
input = 256×256 RGB
embed_dim = 256
codebook entries = 1024
spatial compression = 16×
GAN starts = step 250001
```

## 1. Components

**Observed:**

```text
encoder
 ↓
quant_conv
 ↓
vector quantizer + learned codebook
 ↓
quantized latent vectors
 ↓
post_quant_conv
 ↓
decoder
```

plus:

```text
frozen LPIPS
trainable discriminator
```

Unlike Recipe 1, there is **no Gaussian posterior and no sampling from mean/variance**.

The latent bottleneck instead produces:

```text
continuous pre-quantized vectors
       ↓ nearest learned code
quantized vectors + integer code indices
```

## 2. Data and one training step

**Observed:** each 256×256 ImageNet image is encoded, quantized, and reconstructed.

The quantizer supplies both the quantized representation and a quantization/codebook loss.

Generator loss combines:

```text
pixel reconstruction
+ LPIPS
+ codebook/commitment loss
+ adversarial generator loss
```

The decoder reconstructs from the **quantized vector**, not directly from the raw encoder output.

The same representation can also be exposed as discrete code indices for another model.

So one encoding produces two conceptually useful views:

```text
quantized continuous vectors → decoder

integer code IDs            → token model
```

## 3. Optimization and runtime

**Observed:** two optimizers again exist.

The generator optimizer explicitly owns:

```text
encoder
decoder
vector quantizer/codebook
quant_conv
post_quant_conv
```

The second optimizer owns the discriminator.

Thus the bottleneck itself has trainable state, unlike a purely mathematical reparameterization.

## 4. Structural changes during a run

**Observed:** adversarial training is disabled until step **250,001**, then becomes active.

No component is created at that point, but its relationship to the loss changes from inactive to active.

## 5. Outputs and restoration

**Observed:** a usable VQGAN artifact requires the learned encoder, decoder **and codebook/quantizer**.

The discriminator and perceptual network are training-only.

This matters compared with the KL VAE:

```text
KL VAE artifact:
encoder + latent parameterization + decoder

VQ artifact:
encoder + codebook/quantizer + decoder
```

Losing the codebook means losing the latent representation's meaning.

## 6. Design pressure

> Can a “bottleneck” be an independently stateful, trainable component whose state is required for both encoding and decoding?

And:

> Can one encoding expose multiple representations—continuous embeddings and discrete token IDs—without forcing one of them to be designated the single canonical latent?

That one matters a lot once audio codecs/tokenizers enter the picture too.

---

# Recipe 3 — Open-Sora Plan WF-VAE 4×8×8 video VAE

**Implementation examined:** `PKU-YuanGroup/Open-Sora-Plan`, commit `f7fa604f`, concrete [`scripts/causalvae/train.sh`](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/f7fa604f4e3a523d6b973e4c89a5620ed1aff65a/scripts/causalvae/train.sh), config [`wfvae_4dim.json`](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/f7fa604f4e3a523d6b973e4c89a5620ed1aff65a/scripts/causalvae/wfvae_4dim.json), trainer [`train_causalvae.py`](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/f7fa604f4e3a523d6b973e4c89a5620ed1aff65a/opensora/train/train_causalvae.py), loss [`LPIPSWithDiscriminator3D`](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/f7fa604f4e3a523d6b973e4c89a5620ed1aff65a/opensora/models/causalvideovae/model/losses/perceptual_loss.py).

The script trains on:

```text
25 frames
256×256
batch size 1/GPU
8 GPUs
AdamW
lr = 1e-5
EMA = 0.999
wavelet loss
3D discriminator
```

The v1.3 WFVAE family is used downstream as a **4×8×8 temporal/spatial compressor**.

## 1. Components

**Observed:** the recipe contains:

```text
WFVAE encoder
WFVAE decoder
Gaussian posterior / VAE bottleneck
wavelet energy-flow paths
3D discriminator
LPIPS perceptual network
EMA copy of generator
```

The model config explicitly includes both 2D and 3D Haar wavelet transforms in its down/up paths.

Unlike the previous image VAE, spatial and temporal compression are not interchangeable:

```text
time   → compressed separately
height → compressed
width  → compressed
```

## 2. Data and one training step

**Observed:** a batch contains video:

```text
[B,C,T,H,W]
```

with the concrete recipe using:

```text
T = 25
H = W = 256
```

The generator produces:

```text
reconstructed video
posterior
optional wavelet coefficients
```

For reconstruction/perceptual loss, the implementation temporarily reshapes video:

```text
[B,C,T,H,W]
      ↓
[B*T,C,H,W]
```

and applies image-space reconstruction + LPIPS frame by frame.

It then restores the full five-dimensional representation for the **3D discriminator**.

So one loss intentionally ignores temporal grouping while another explicitly operates on temporal structure.

Generator loss combines:

```text
frame reconstruction
+ frame LPIPS
+ KL
+ 3D adversarial loss
+ wavelet loss
```

## 3. Optimization and runtime

**Observed:** there are two AdamW optimizers:

```text
generator optimizer:
    encoder + decoder
    lr 1e-5
    weight decay 1e-4

discriminator optimizer:
    3D discriminator
    lr 1e-5
    weight decay 1e-2
```

The implementation can optionally freeze the encoder, but the concrete script does **not** request `--freeze_encoder`, so both encoder and decoder are optimized.

**Observed:** generator and discriminator do not update on the same step.

The training loop explicitly alternates:

```text
even/global initial step → generator
odd step                 → discriminator
```

once the discriminator start condition is satisfied.

The concrete script has:

```text
disc_start = 0
```

so alternation begins immediately.

EMA updates occur after generator updates.

## 4. Structural changes during a run

**Not present** in this exact recipe: adversarial training is active from the start.

However, the actual execution/optimization role changes **every step**:

```text
step N:
  encoder/decoder train
  discriminator effectively frozen from optimizer update

step N+1:
  encoder/decoder requires_grad disabled
  discriminator trains
```

So optimization ownership is temporally alternating even though the component topology is fixed.

## 5. Outputs and restoration

**Observed:** checkpoints explicitly contain:

```text
generator weights
discriminator weights
generator optimizer state
discriminator optimizer state
AMP scaler state
sampler state
epoch
current step
EMA state
```

This is one of the cleanest examples we've seen of the difference between:

```text
deployable artifact
    VAE generator

same-run restoration artifact
    generator
    + discriminator
    + both optimizers
    + scaler
    + sampler
    + EMA
    + progress state
```

The documentation also explicitly distinguishes loading model weights from `resume_from_checkpoint`, which restores optimizer/training state.

## 6. Design pressure

> Can different losses consume different views of the same reconstructed object—for example frame-flattened `[B*T,C,H,W]` for LPIPS and full `[B,C,T,H,W]` for a temporal discriminator?

And:

> Can optimization ownership alternate between component groups from step to step, including temporarily disabling gradients on the other group?

Also:

> Is compression geometry represented as several semantic axes rather than one generic scalar “downsampling factor”?

A video VAE's `4×8×8` is meaningfully different from an image VAE's `8×`.

---

# Recipe 4 — Stable Audio 2.0 waveform VAE

**Implementation examined:** `Stability-AI/stable-audio-tools`, commit `3241adba`, concrete [`stable_audio_2_0_vae.json`](https://github.com/Stability-AI/stable-audio-tools/blob/3241adba4fc2a85cf5b29d9eb68d42f40a28e820/stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json), [`AutoencoderTrainingWrapper`](https://github.com/Stability-AI/stable-audio-tools/blob/3241adba4fc2a85cf5b29d9eb68d42f40a28e820/stable_audio_tools/training/autoencoders.py), and [`autoencoders.md`](https://github.com/Stability-AI/stable-audio-tools/blob/3241adba4fc2a85cf5b29d9eb68d42f40a28e820/docs/autoencoders.md).

Concrete model:

```text
44.1 kHz stereo
sample_size = 65536
Oobleck encoder
VAE bottleneck
Oobleck decoder

latent channels = 64
downsampling = 2048×
```

## 1. Components

**Observed:** stable-audio-tools explicitly separates three model roles:

```text
encoder
 ↓
bottleneck
 ↓
decoder
```

The bottleneck is itself configurable and isn't assumed to be part of either encoder or decoder.

For this recipe:

```text
encoder output channels = 128
         ↓ split
mean + scale
         ↓ VAE sample
64-channel latent sequence
         ↓
decoder
```

Additional training components are:

```text
multi-resolution STFT reconstruction losses
Encodec-style discriminator
EMA autoencoder copy
```

This recipe has no image perceptual network.

## 2. Data and one training step

**Observed:** input is stereo waveform data:

```text
[B,2,N]
```

at 44.1 kHz.

For the configured sample size:

```text
N = 65536
```

The encoder downsamples time by **2048×** into a 64-channel latent sequence.

The VAE bottleneck returns both:

```text
sampled latent
KL value
```

The decoder reconstructs waveform samples.

The generator objective includes:

```text
multi-resolution STFT reconstruction
+ KL * 1e-4
+ adversarial loss * 0.1
+ feature matching * 5.0
```

The configured raw waveform L1 weight is actually `0.0`.

That means “reconstruction loss” here is primarily **frequency-domain**, rather than elementwise sample matching.

## 3. Optimization and runtime

**Observed:** two AdamW optimizers have different learning rates:

```text
autoencoder:
    lr = 1.5e-4
    betas = [0.8,0.99]

discriminator:
    lr = 3e-4
    betas = [0.8,0.99]
```

Both also have separate inverse-LR schedulers.

**Observed:** the wrapper disables Lightning automatic optimization and manually chooses which optimizer runs.

With a discriminator enabled:

```text
one step → generator
next step → discriminator
...
```

The discriminator still executes on generator steps to provide adversarial/feature-matching gradients, but its own parameters have `requires_grad` disabled for that step.

Again:

```text
needed in differentiable execution
≠
receives parameter gradients
```

## 4. Structural changes during a run

For the exact Stable Audio 2.0 config, **no adversarial warmup transition is present** because:

```text
warmup_steps = 0
```

EMA state does have its own delayed update behavior in the current wrapper: its EMA helper is configured to begin after 2,000 steps and the wrapper updates it periodically rather than every batch.

The generic wrapper can also freeze the encoder after warmup, but **that is not enabled by this recipe**.

## 5. Outputs and restoration

**Observed:** `export_model()` intentionally strips training machinery and exports the autoencoder—using the EMA model when EMA exists.

So:

```text
training wrapper:
autoencoder
+ discriminator
+ losses
+ optimizers
+ schedulers
+ EMA machinery

export:
autoencoder only
```

This is a very explicit separation between **training system** and **deployable model**.

The exported autoencoder itself still contains the encoder, VAE bottleneck and decoder necessary to define its latent space.

**Unknown:** exact bit-for-bit restoration state was not established from the model config and wrapper alone; Lightning checkpoint handling supplies additional generic state.

## 6. Design pressure

> Can the autoencoder architecture itself be composition of independently selected encoder, bottleneck and decoder components instead of one indivisible `VAE` model slot?

And:

> Can reconstruction objectives operate in a representation different from both source and latent spaces—for example STFT-domain losses over waveform reconstructions?

That second one makes “loss space” another independent concern.

---

Yep — I’d add **FLUX.2 as Recipe 5**. It’s useful precisely because it’s a modern counterexample to treating “the VAE latent” as one obvious tensor.

---

# Recipe 5 — FLUX.2 autoencoder and downstream latent representation

**Implementation examined:** official `black-forest-labs/flux2`, commit `50fe5162`, [`src/flux2/autoencoder.py`](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/autoencoder.py), cross-checked against Diffusers `AutoencoderKLFlux2`, commit `8b3c707e`, [`autoencoder_kl_flux2.py`](https://github.com/huggingface/diffusers/blob/8b3c707ebd3ec4881f4190cf42931da07eaf3b65/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py).

This is primarily an examination of the **released autoencoder and the representation consumed by FLUX.2**, not a complete VAE pretraining recipe. Training-only details not established by the released sources are marked **unknown**.

## 1. Components

**Observed:** the official FLUX.2 autoencoder contains an encoder and decoder around a KL-style latent parameterization.

The encoder ends in twice the configured latent channel count:

```text
image
 ↓
encoder
 ↓
[mean | second posterior parameter]
```

with:

```text
z_channels = 32
```

The released FLUX.2 inference implementation then selects only the **mean** half:

```python
moments = encoder(x)
mean = torch.chunk(moments, 2, dim=1)[0]
```

So although the architecture can represent posterior parameters, the downstream FLUX.2 encoding path is deterministic.

**Observed:** another independently meaningful component is the latent normalization state. The official implementation contains a non-affine `BatchNorm2d` whose stored running mean/variance define the transform between autoencoder latents and the representation consumed by FLUX.2.

That normalization state is required even though it is not an encoder, decoder, or learned affine layer.

## 2. Data and one training/inference representation step

**Observed:** image-space input is encoded into a 32-channel spatial representation.

The autoencoder then performs an additional rearrangement that packs each `2×2` latent neighborhood into channels:

```text
[B, 32, H/8, W/8]
        ↓ 2×2 rearrangement
[B, 128, H/16, W/16]
```

Conceptually:

```text
image
 ↓
encoder
 ↓
posterior parameters
 ↓
take posterior mean
 ↓
32-channel VAE representation
 ↓
2×2 spatial packing
 ↓
128-channel representation
 ↓
normalization using stored statistics
 ↓
FLUX.2 model input
```

**Observed:** decoding performs the inverse operations:

```text
normalized 128-channel representation
 ↓
inverse normalization
 ↓
2×2 unpack
 ↓
32-channel VAE latent
 ↓
decoder
 ↓
image
```

Thus three tensors that could casually all be called “latents” are actually distinct:

```text
encoder/posterior representation

32-channel autoencoder latent

128-channel normalized FLUX.2 representation
```

## 3. Optimization and runtime

**Unknown:** the released BFL implementation examined does not establish the full original FLUX.2 autoencoder training objective, optimizer configuration, discriminator setup, update schedule, or exact set of auxiliary training components.

Therefore we should **not infer** that it used the same LPIPS/GAN/two-optimizer recipe as the older CompVis autoencoders.

**Observed:** when FLUX.2 itself is trained or fine-tuned with a frozen pretrained autoencoder, the encoder is still required to transform source images into the model's latent representation unless those representations have been cached.

The decoder is unnecessary for an ordinary latent-space training step unless reconstruction, preview generation, decoded-space loss, or another recipe operation requires it.

## 4. Structural changes during a run

**Unknown / not established.**

The released autoencoder implementation does not establish a VAE pretraining recipe with component activation or retirement transitions.

For ordinary downstream FLUX.2 training with cached latents, there can conceptually be a preparation/runtime distinction:

```text
preparation:
image → VAE encode → normalized/packed representation → cache

training:
cached representation → FLUX.2
```

but whether a particular training recipe actually does this must be established from that recipe rather than inferred from the autoencoder.

## 5. Outputs and restoration

**Observed:** using the FLUX.2 latent representation correctly requires more than decoder weights alone.

The representation contract includes:

```text
encoder/decoder architecture
32-channel latent geometry
posterior-mean selection
2×2 latent packing
normalization statistics
inverse transforms for decoding
```

The BatchNorm running mean/variance are therefore semantically part of the representation definition.

**Unknown:** the released inference implementation does not establish all state needed to resume the original autoencoder pretraining run exactly.

## 6. Design pressure

This gives a particularly useful modern constraint:

> **Can the system distinguish the autoencoder's internal latent representation from the downstream model's representation, even when both are casually called “latents”?**

And:

> **Can representation transformations such as posterior selection, packing, normalization, and inverse normalization be first-class parts of the representation path without requiring each one to be promoted into a separate trainable model component?**

Also:

> **Can non-parameter state such as stored normalization statistics be part of an artifact's required representation contract?**

---

# Cross-case findings

These recipes make “VAE support” look quite a bit broader than simply giving a Trainer `vae.encode()` and `vae.decode()`.

| Tempting assumption                                                    | What the recipes show                                                                          |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| A VAE is just encoder + decoder                                        | **False.** Bottleneck can have independent semantics/state.                                    |
| All bottlenecks are Gaussian                                           | **False.** VQGAN uses learned discrete codebooks.                                              |
| A latent is always continuous                                          | **False.** VQ can expose integer token IDs.                                                    |
| Bottlenecks are stateless transforms                                   | **False.** VQ codebooks are trained state.                                                     |
| Reconstruction loss compares input/output directly                     | **False.** LPIPS and STFT-domain objectives introduce additional representations.              |
| Frozen loss models can run under `no_grad()`                           | **False.** LPIPS must transmit gradients to reconstruction despite frozen weights.             |
| One optimizer is enough                                                | **False.** All four adversarial recipes involve distinct generator/discriminator optimization. |
| All model components update every step                                 | **False.** WFVAE and stable-audio alternate generator/discriminator steps.                     |
| GAN components belong to inference                                     | **False.** They're training-only and discarded from deployment.                                |
| Compression is one integer                                             | **False.** Image, waveform and video compression have different axis semantics.                |
| “VAE latent” uniquely identifies its representation                    | **False.** latent channels, scaling, stochasticity, quantization and geometry all matter.      |
| The artifact needed to resume equals the artifact needed for inference | **Very false.** WFVAE is a particularly clear counterexample.                                  |
| `vae.encode(x)` directly gives the tensor consumed by the generative model | **False.** FLUX.2 selects the posterior mean, spatially packs it, and normalizes it first.                                |
| Latent channel count is a single property of the model                     | **False.** FLUX.2 has a 32-channel VAE representation but a 128-channel packed downstream representation.                 |
| All representation state is trainable parameters                           | **False.** FLUX.2 depends on stored normalization statistics.                                                             |
| “Latent” names one uniquely defined representation                         | **False.** Posterior output, selected VAE latent, packed latent, and normalized model representation can all be distinct. |

The first general design constraint I’d take from this is:

> **An autoencoder path may contain distinct encoder, posterior/bottleneck, latent-selection, normalization, packing, and decoder stages. These are distinct representation roles even when several are implemented inside one model object rather than exposed as separate components.**

The second is probably even more important for your trainer:

> **“Representation” and “component” are separate concepts. The same autoencoder may expose posterior parameters, sampled continuous latents, quantized vectors, discrete indices, reconstructions, wavelet coefficients, or auxiliary bottleneck losses; a recipe must identify which representation each consumer expects.**

And one more pops out from the optimizer cases:

> **Trainability cannot just live on the model slot. Encoder, decoder, bottleneck/codebook, discriminator and auxiliary components may have different optimizer ownership and may become active, frozen, or update on different steps.**

That last part connects pretty neatly with the teacher/student research: in both cases, **component identity, execution participation, gradient participation, optimizer ownership, and artifact ownership are all different axes**. A Trainer architecture that collapses them into `trainable=True/False` is going to get painful pretty quickly.
