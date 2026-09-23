# Research topic: Pixel-space training

**Claim labels:** **Observed** = directly established by source; **Inferred** = follows from implementation; **Unknown** = source does not establish it.

## Recipe 1 — Improved Diffusion, class-conditional ImageNet-64

**Implementation examined:** `openai/improved-diffusion`, commit `1bc7bbbd`, released **270M class-conditional ImageNet-64** recipe, entrypoint `scripts/image_train.py`.

Official recipe:

```text
image_size=64
num_channels=192
num_res_blocks=3
learn_sigma=True
class_cond=True

diffusion_steps=4000
noise_schedule=cosine
rescale_learned_sigmas=False
rescale_timesteps=False

lr=3e-4
batch_size=2048
```

Primary sources: [`README.md`](https://github.com/openai/improved-diffusion/blob/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5/README.md), [`scripts/image_train.py`](https://github.com/openai/improved-diffusion/blob/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5/scripts/image_train.py), [`image_datasets.py`](https://github.com/openai/improved-diffusion/blob/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5/improved_diffusion/image_datasets.py), [`gaussian_diffusion.py`](https://github.com/openai/improved-diffusion/blob/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5/improved_diffusion/gaussian_diffusion.py), [`script_util.py`](https://github.com/openai/improved-diffusion/blob/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5/improved_diffusion/script_util.py), and the Improved DDPM paper. ([arXiv][1])

### 1. Components

**Observed:** the learned component is a class-conditional U-Net. There is **no VAE/autoencoder** in the training path. `create_model()` fixes `in_channels=3`; because `learn_sigma=True`, it uses `out_channels=6`: three channels for the denoising prediction and three for learned variance.

**Observed:** the diffusion process/schedule and schedule sampler participate but are algorithm state, not trainable neural networks. EMA copies of model parameters are also maintained by `TrainLoop`.

### 2. Data and one training step

**Observed:** images are cropped/resized, converted to CHW float arrays, and normalized directly to `[-1, 1]`. A batch is therefore approximately:

```text
image: [B, 3, 64, 64]
class: [B]
```

There is no intermediate encoding step.

**Observed:** a timestep `t` is sampled, Gaussian noise with the **same shape as the pixel tensor** is sampled, and `q_sample()` constructs noisy pixels `x_t`.

The U-Net receives:

```text
x_t + timestep + class
```

and predicts epsilon plus learned variance.

**Observed:** for this configuration the mean target is the sampled pixel-space noise. The loss contains pixel-shaped epsilon MSE plus the variational term used to train the learned variance.

So the essential path is:

```text
RGB pixels
   ↓
sample t + pixel-shaped noise
   ↓
noisy RGB tensor
   ↓
U-Net
   ↓
epsilon + variance prediction
   ↓
MSE + variance/VB loss
```

### 3. Optimization and runtime

**Observed:** the U-Net parameters receive gradients and are optimized with AdamW. EMA parameter copies are updated separately.

**Observed:** no frozen image encoder or decoder needs to be resident.

**Observed:** distributed training and microbatching are supported.

### 4. Structural changes during a run

**Not present.**

Timesteps, noise and examples vary, but component relationships and optimizer ownership remain fixed.

### 5. Outputs and restoration

**Observed:** the repository saves ordinary model checkpoints, EMA checkpoints and optimizer state separately. Resume restores the relevant model/EMA/optimizer states.

**Observed:** inference does **not** require a VAE or other learned image-space conversion component.

**Unknown:** the inspected checkpoint path does not establish bit-exact restoration of RNG and dataloader position.

### 6. Design pressure

> Can the training representation itself be the source media tensor, with no mandatory encoder/cache/latent stage between dataset preparation and noise construction?

And:

> Can prediction/target shapes be determined directly from media channels—for example a three-channel RGB target and a six-channel model output when variance is also learned—rather than from a model-family-specific latent format?

---

# Recipe 2 — EDM, class-conditional ImageNet-64

**Implementation examined:** `NVlabs/edm`, commit `008a4e53`, official `imagenet-64x64-cond-adm` recipe through `train.py`.

The released configuration is:

```text
--cond=1
--arch=adm
--duration=2500
--batch=4096
--lr=1e-4
--ema=50
--dropout=0.10
--augment=0
--fp16=1
--ls=100
--tick=200
```

Primary sources: [`README.md`](https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/README.md), [`train.py`](https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/train.py), [`training/loss.py`](https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py), [`training/training_loop.py`](https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/training_loop.py), [`training/dataset.py`](https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/dataset.py), and the EDM paper. EDM explicitly separates noise distribution, preconditioning, loss weighting and sampling design. ([arXiv][2])

### 1. Components

**Observed:** one trainable pixel-space ADM network plus a frozen EMA copy. No VAE participates.

An optional augmentation pipeline can participate in EDM generally; the exact ImageNet-64 recipe examined uses `--augment=0`.

### 2. Data and one training step

**Observed:** the dataset stores `uint8` images in NCHW form. The loop converts:

```text
uint8 [0,255]
      ↓
float32 / 127.5 - 1
      ↓
[B,C,H,W] in [-1,1]
```

For this recipe that is `[B,3,64,64]`.

**Observed:** `EDMLoss` samples one continuous noise scale per item:

```text
sigma: [B,1,1,1]
```

from a log-normal distribution.

It then performs:

```text
y = clean pixel image

n = randn_like(y) * sigma

input = y + n

prediction = net(input, sigma, class)

loss = weight(sigma) * (prediction - y)^2
```

So unlike Recipe 1 there is no discrete 4000-element diffusion-time target in the loss. The noisy object is still directly an RGB tensor.

### 3. Optimization and runtime

**Observed:** the network is optimized with Adam; EMA is a frozen deep copy updated from it. DDP and gradient accumulation are supported.

**Observed:** again, no encoder/decoder needs to be loaded merely to turn media into trainable latent tensors.

### 4. Structural changes during a run

**Not present.**

Continuous noise levels change per example, but model topology and optimizer ownership remain fixed.

### 5. Outputs and restoration

**Observed:** EDM intentionally distinguishes:

```text
network-snapshot-*.pkl
    → inference artifact

training-state-*.pt
    → training model + optimizer state
```

`--resume` consumes the training-state artifact and corresponding network snapshot.

**Unknown:** the dump shown does not establish restoration of every RNG/dataloader state needed for bit-identical continuation.

### 6. Design pressure

> Can noise level be a continuous per-example scalar/broadcast tensor rather than a model-family-specific integer timestep?

And more generally:

> Is “pixel-space” represented independently from the noise/process formulation, so DDPM-style discrete diffusion and EDM continuous-noise training can share the same media representation without being forced into the same scheduler semantics?

That distinction feels important: **pixel space describes the representation, not the training objective.**

---

# Recipe 3 — Guided Diffusion 64→256 pixel-space super-resolution

**Implementation examined:** `openai/guided-diffusion`, commit `22e0df81`, `scripts/super_res_train.py` / `SuperResModel`. The released model family includes the official `64x64 -> 256x256` ImageNet upsampler. The corresponding paper uses cascaded diffusion to successively produce higher-resolution images. ([arXiv][3])

Primary sources: [`scripts/super_res_train.py`](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/scripts/super_res_train.py), [`guided_diffusion/unet.py`](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py), [`README.md`](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/README.md).

### 1. Components

**Observed:** the recipe trains one `SuperResModel`, a modified U-Net.

**Observed:** there is still **no VAE**. Both the target and conditioning input are images.

The low-resolution condition is not produced by a separately trained encoder. During training it is generated directly from the high-resolution training image.

### 2. Data and one training step

**Observed:** `load_superres_data()` starts with the high-resolution batch:

```text
large_batch: [B,3,256,256]
```

and constructs:

```python
low_res = interpolate(large_batch, 64, mode="area")
```

giving approximately:

```text
low_res: [B,3,64,64]
```

The normal diffusion machinery noises the **256×256 target image**.

Inside `SuperResModel.forward()`, the 64×64 condition is bilinearly resized back to the noisy target's dimensions and concatenated with it:

```text
noisy target       [B,3,256,256]
upsampled low-res  [B,3,256,256]
                       ↓ concat channels
model input        [B,6,256,256]
```

The U-Net then predicts the high-resolution diffusion target.

### 3. Optimization and runtime

**Observed:** only the super-resolution U-Net receives optimizer updates. The low-resolution transform is ordinary interpolation, not a frozen neural component.

**Observed:** generic `TrainLoop` provides EMA, mixed precision, distributed execution and checkpointing.

### 4. Structural changes during a run

**Not present.**

The relationship:

```text
high-resolution target
        ↓ downsample
low-resolution condition
```

exists throughout training.

### 5. Outputs and restoration

**Observed:** the trained upsampler is its own generative artifact, but it is **conditionally useful**: inference requires a low-resolution image produced elsewhere.

So unlike the first two:

```text
base pixel diffusion
    → standalone generator

super-resolution pixel diffusion
    → depends on a lower-resolution source
```

**Unknown:** the H→L condition-generation state itself has no learned checkpoint state because it is interpolation.

### 6. Design pressure

> Can one training sample contain multiple representations of the same modality at different spatial resolutions, with one serving as the noisy target and another as conditioning?

And:

> Can a Trainer distinguish “pixel-space target,” “pixel-space condition,” and “model input after concatenation” instead of assuming there is one canonical image tensor per example?

---

# Recipe 4 — PixelFlow-XL ImageNet-256 four-stage pixel flow

This is probably the most interesting one for your trainer design.

**Implementation examined:** `ShoufaChen/PixelFlow`, commit `8805204d`, config `configs/pixelflow_xl_c2i.yaml`, entrypoint `train.py`.

PixelFlow explicitly targets **raw pixel-space generation without a pretrained VAE**, while using cascaded flow modeling to make that practical. ([arXiv][4])

Primary implementation locations: [`train.py`](https://github.com/ShoufaChen/PixelFlow/blob/8805204d1be8df22382280959b3063974ff3d77d/train.py), [`configs/pixelflow_xl_c2i.yaml`](https://github.com/ShoufaChen/PixelFlow/blob/8805204d1be8df22382280959b3063974ff3d77d/configs/pixelflow_xl_c2i.yaml), [`pixelflow/data_in1k.py`](https://github.com/ShoufaChen/PixelFlow/blob/8805204d1be8df22382280959b3063974ff3d77d/pixelflow/data_in1k.py), `PixelFlowScheduler`, and `PixelFlowModel`.

### 1. Components

**Observed:** the recipe has one trainable `PixelFlowModel` Transformer and one frozen EMA copy.

Config:

```text
in_channels=3
out_channels=3
patch_size=4
depth=28
num_classes=1000
num_stages=4
resolution=256
```

**Observed:** there is no VAE. The paper explicitly presents removing the pretrained VAE as a defining property of PixelFlow. ([arXiv][4])

### 2. Data and one training step

**Observed:** source ImageNet images begin as RGB `256×256` tensors normalized to `[-1,1]`.

A particularly important difference is that a batch does **not stay BCHW**.

Each example is assigned one of four flow stages. Depending on that stage, the image is represented at:

```text
256×256
128×128
64×64
32×32
```

**Observed:** for a stage, preprocessing constructs a higher-detail `pixel_values_end` and a lower-detail `pixel_values_start`. The lower-detail representation is downsampled further and then nearest-upsampled to the stage resolution.

The **same Gaussian noise** is mixed into both endpoints. Their difference becomes the flow target:

```text
target = pixel_values_end - pixel_values_start

x_t =
    t * pixel_values_end
  + (1-t) * pixel_values_start
```

Then the image tensors are patchified directly:

```text
[B,C,H,W]
    ↓ patch_size=4
[(all image tokens), C*4*4]
```

For RGB, each raw patch vector therefore has **48 values**.

**Observed:** examples at different stages have different token counts. The collator concatenates them into a packed sequence and provides:

```text
seqlen_list_q
cumsum_q_len
pos_embed
batch_latent_size
```

The unfortunately named `batch_latent_size` is computed from the **pixel patch sequence size**; there is no VAE latent involved.

The model predicts the packed flow target and uses MSE.

### 3. Optimization and runtime

**Observed:** all trainable model parameters are optimized with AdamW. Training uses DDP, BF16 autocast, gradient clipping at 1.0 and an EMA model.

**Observed:** only one optimizer is present.

**Observed:** no frozen image encoder/decoder participates in the training path.

### 4. Structural changes during a run

**Not present** as a component-topology change.

This distinction matters: PixelFlow has **four stages**, but they are sampled as training cases within one model/run. The Trainer does not successively replace a 32×32 model with a 64×64 model.

Stage selection is **per-example/per-step algorithm state**, not a lifecycle transition.

### 5. Outputs and restoration

**Observed:** checkpoints contain:

```text
model state
EMA state
optimizer state
args
```

and the run writes its YAML config separately.

**Observed:** `train.py` does not expose a full checkpoint-resume path analogous to EDM's `--resume`; `--pretrained-model` loads model weights only.

Therefore:

**Unknown:** exact same-run restoration is not established by this implementation even though optimizer state is written.

### 6. Design pressure

This one breaks several tempting assumptions at once:

> Can source media be pixel-shaped while the actual model-facing training representation is a packed, variable-length sequence of raw pixel patches?

> Can one batch contain examples at different spatial resolutions and therefore different sequence lengths?

> Can “stage” be an attribute of each training example rather than a global Trainer phase or a different model component?

And:

> Can naming/typing distinguish actual latent-space data from generic spatial/token-grid metadata, so something named `latent_size` in an implementation does not force the framework to classify the recipe as latent-space training?

---

# Cross-case findings

These four recipes give us a reasonably strong definition of the design pressure.

| Tempting assumption                            | What the recipes show                                                                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Training images must first become VAE latents  | **False.** All four optimize directly from pixel-space representations.                                                               |
| Pixel-space means DDPM epsilon prediction      | **False.** EDM uses continuous denoising; PixelFlow uses flow targets.                                                                |
| Pixel-space batches stay `[B,C,H,W]`           | **False.** PixelFlow turns raw pixels into packed variable-length patch sequences.                                                    |
| One sample has one image tensor                | **False.** Super-resolution has target + lower-resolution pixel condition; PixelFlow constructs multiple stage endpoints.             |
| Resolution is fixed for the run                | **False as a model-facing assumption.** PixelFlow mixes several spatial scales inside the same run/batch.                             |
| No VAE means no preprocessing                  | **False.** Cropping, normalization, downsampling, stage construction, noise construction and patchification can still be substantial. |
| Pixel-space determines the scheduler/objective | **False.** Representation space and training process are independent concerns.                                                        |

The most useful overall constraint for your design is probably:

> **The representation domain (“pixels,” “VAE latents,” codec tokens, etc.) should be independent from the training process (“discrete diffusion,” “continuous denoising,” “flow matching,” etc.). Pixel-space training does not imply one scheduler, target parameterization, tensor layout, or conditioning scheme.**

And I’d add:

> **Media-space ownership should not imply model-input shape. A recipe may begin from RGB pixels yet derive noisy BCHW tensors, multiresolution conditions, or packed variable-length pixel-patch sequences before the trainable model executes.**

That feels like the main thing worth carrying back into the `sd-scripts` architecture: **“pixel vs latent” belongs much earlier in the data/representation story than “what does the Trainer forward look like?”**

[1]: https://arxiv.org/abs/2102.09672?utm_source=chatgpt.com "Improved Denoising Diffusion Probabilistic Models"
[2]: https://arxiv.org/abs/2206.00364?utm_source=chatgpt.com "Elucidating the Design Space of Diffusion-Based Generative Models"
[3]: https://arxiv.org/abs/2105.05233?utm_source=chatgpt.com "Diffusion Models Beat GANs on Image Synthesis"
[4]: https://arxiv.org/abs/2504.07963?utm_source=chatgpt.com "PixelFlow: Pixel-Space Generative Models with Flow"
