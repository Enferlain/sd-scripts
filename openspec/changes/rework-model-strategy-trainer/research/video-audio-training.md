# Research topic: Video and audio training

**Claim labels:** **Observed** = directly established by the examined source. **Inferred** = follows from the implementation but is not explicitly stated. **Unknown** = the examined sources do not establish it.

I used three deliberately different recipes:

| Recipe                                    | What it tests                                                                |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| Wan2.2 T2V A14B low-noise, diffusion-pipe | ordinary latent video training + temporal dimension + split timestep experts |
| MiniMax-H3 LoRA, diffusion-pipe           | one model jointly training synchronized video **and audio**                  |
| YuE2, AI Toolkit                          | audio-native hybrid **AR token modeling + NAR flow matching**                |

---

# Recipe 1 — Wan2.2 T2V A14B low-noise training

**Implementation examined:** `tdrussell/diffusion-pipe`, commit `27eab430`, Wan2.2 T2V A14B **low-noise** model path.

The documented model configuration is:

```toml
[model]
type = 'wan'
ckpt_path = '.../Wan2.2-T2V-A14B'
transformer_path = '.../Wan2.2-T2V-A14B/low_noise_model'
dtype = 'bfloat16'
transformer_dtype = 'float8'
min_t = 0
max_t = 0.875
```

Primary sources: [`docs/supported_models.md`](https://github.com/tdrussell/diffusion-pipe/blob/27eab430ccf2dd82c1edaca834a44808d942de89/docs/supported_models.md), [`models/wan/wan.py`](https://github.com/tdrussell/diffusion-pipe/blob/27eab430ccf2dd82c1edaca834a44808d942de89/models/wan/wan.py), [`models/wan/configs.py`](https://github.com/tdrussell/diffusion-pipe/blob/27eab430ccf2dd82c1edaca834a44808d942de89/models/wan/configs.py).

### 1. Components

**Observed:** the recipe uses a Wan diffusion transformer, a frozen Wan video VAE and a frozen UMT5 text encoder. `WanPipeline.__init__` explicitly freezes the T5 model.

**Observed:** the VAE and text encoder can participate during cache construction rather than every optimizer step. The trainable transformer is loaded separately by `load_diffusion_model()`.

**Observed:** Wan2.2 A14B consists of independent **low-noise and high-noise transformer checkpoints**. The concrete recipe examined loads only the low-noise transformer. Diffusion-pipe documents the T2V inference boundary as `t=0.875`.

**Unknown:** the Wan2.2 model snippet does not choose LoRA versus full fine-tuning by itself; diffusion-pipe supports both. That choice belongs to the outer training configuration, which is not specified by this particular documented recipe.

### 2. Data and one training step

**Observed:** preprocessing supports actual video and fixes a model-specific framerate. VAE encoding produces five-dimensional video latents:

```text
[B, C, F, H, W]
```

`prepare_inputs()` samples one training noise level per example, restricted to the configured interval:

```text
0 <= t <= 0.875
```

and computes:

```text
x1 = video latent
x0 = Gaussian noise

x_t = (1 - t) * x1 + t * x0
target = x0 - x1
```

The timestep is then scaled to `[0,1000]` for the transformer.

**Observed:** spatial loss masks are resized to latent `H×W` and expanded so they broadcast across the temporal axis rather than having an independently specified value for every frame.

**Observed:** inside `InitialLayer`, the 3D latent grid is patch-embedded, flattened into a sequence, and padded to the longest sequence in the batch. `grid_sizes` and `seq_lens` preserve each video's original latent geometry.

So video is simultaneously represented as:

```text
media:
frames × height × width

VAE representation:
C × F × H × W

transformer representation:
sequence of spatiotemporal patches
```

### 3. Optimization and runtime

**Observed:** transformer blocks are the trainable execution path. The architecture supports pipeline parallelism and block swapping.

**Observed:** VAE and T5 are needed to derive cached training representations but do not themselves receive optimization in this recipe.

**Observed:** when text embeddings are cached, the training step consumes cached embeddings and sequence lengths rather than requiring T5 execution for every step.

**Unknown:** because the model documentation does not supply the outer adapter/full-finetune configuration, the exact optimized parameter set for this particular example is not established.

### 4. Structural changes during a run

**Not present** in the concrete low-noise run.

Importantly, Wan2.2 having two timestep experts does **not** mean this recipe swaps models at `t=.875` during training. The implementation loads one transformer and restricts sampled timesteps to that transformer's range.

Training both experts means training independent model components/runs rather than treating them as one dynamically switched training component.

### 5. Outputs and restoration

**Observed:** diffusion-pipe can save either a full Wan transformer as `model.safetensors` or a LoRA in ComfyUI-compatible format, depending on the generic training configuration.

The resulting artifact still depends on the Wan system's VAE/text-conditioning components for generation.

**Unknown:** exact resumable state for this concrete Wan run was not established from the model-specific sources alone; that would require auditing diffusion-pipe's generic checkpoint machinery.

### 6. Design pressure

> Can a training representation carry an explicit temporal axis and preserve `(F,H,W)` geometry even after being flattened into variable-length transformer sequences?

And:

> Can one model family contain multiple independent trainable components responsible for different regions of the same noise/time domain, without forcing the Trainer to treat them as one model that changes identity during a step?

That second one is especially relevant to fixed `model` slots.

---

# Recipe 2 — MiniMax-H3 joint video/audio LoRA

**Implementation examined:** `tdrussell/diffusion-pipe`, commit `27eab430`, [`examples/minimax_h3_example.toml`](https://github.com/tdrussell/diffusion-pipe/blob/27eab430ccf2dd82c1edaca834a44808d942de89/examples/minimax_h3_example.toml).

This is a concrete LoRA recipe:

```text
MiniMax-H3 joint diffusion model
video VAE
audio VAE
Qwen3-VL text encoder

LoRA rank = 32
AdamW
microbatch = 1
```

Primary implementation: [`models/minimax_h3.py`](https://github.com/tdrussell/diffusion-pipe/blob/27eab430ccf2dd82c1edaca834a44808d942de89/models/minimax_h3.py), [`minimax_h3_dataset.toml`](https://github.com/tdrussell/diffusion-pipe/blob/27eab430ccf2dd82c1edaca834a44808d942de89/examples/minimax_h3_dataset.toml), [`minimax_h3_notes.md`](https://github.com/tdrussell/diffusion-pipe/blob/27eab430ccf2dd82c1edaca834a44808d942de89/docs/minimax_h3_notes.md).

### 1. Components

**Observed:** independently identifiable components include:

```text
joint H3 transformer       → LoRA-trained
video VAE                  → frozen
audio VAE                  → frozen
Qwen3-VL text encoder      → frozen/caching
LoRA                       → optimized
```

**Observed:** unlike Wan, the trainable H3 transformer handles **video and audio together**.

The media encoder itself is explicitly two-part:

```text
video → video VAE → video latents
audio → audio VAE → audio latents
```

There is no requirement that those representations have the same shape or rank.

### 2. Data and one training step

**Observed:** preprocessing declares both:

```text
support_video = True
support_audio = True
audio_sample_rate = 32000
```

The dataset can also contain still images; the documented H3 frame buckets include `1` as a special image case and longer video buckets.

During caching, `vae_encode()` independently constructs video and audio latents.

At training time:

```text
video latents: [B, C, F, H, W]

audio latents:
a separate temporal latent tensor
```

Audio can be absent for an individual example. The implementation builds a `valid_audio` flag and uses an empty audio tensor when the whole batch has no audio.

**Observed:** video gets Gaussian noise and a sampled video flow timestep:

```text
video_target = video_noise - video_latents
```

Audio gets **different Gaussian noise** and its own flow target:

```text
audio_target = audio_noise - audio_latents
```

But its noise level is derived from the video's:

```text
video t
   ↓ fixed video→audio time mapping
audio t
```

So the modalities use different noise schedules while remaining synchronized through one shared underlying training time.

The transformer receives both noisy streams in the same forward.

**Observed:** loss is computed separately for video and audio, then combined according to the number of elements/tokens contributing to each. If no audio output exists, only video loss contributes.

### 3. Optimization and runtime

**Observed:** in the examined example only rank-32 LoRA parameters receive optimizer updates.

The base H3 transformer still executes the joint video/audio forward. The VAEs and text encoder must be available for cache construction but are not optimized.

**Observed:** the example uses aggressive block swapping and activation checkpointing because the frozen base itself is large.

This distinction matters:

```text
not optimized
≠
not required
```

### 4. Structural changes during a run

**Not present** as model topology.

But **modality participation changes per sample**:

```text
image only      → video/image stream
silent video    → video stream
video + audio   → video + audio streams
```

The same transformer and optimizer remain installed; `valid_audio` controls whether an audio target actually contributes.

### 5. Outputs and restoration

**Observed:** the examined recipe produces a LoRA rather than a standalone H3 model. It therefore depends on the compatible H3 base at deployment.

Inference also belongs to a system containing the appropriate video/audio decoding components.

**Unknown:** the model-specific sources do not establish every generic diffusion-pipe checkpoint state required for exact continuation.

### 6. Design pressure

This is probably the strongest temporal-media case:

> Can a single training example own multiple synchronized target streams with different tensor ranks, encoders, noise tensors, time mappings, validity states, and losses?

And:

> Can “modality absent for this sample” be represented as ordinary recipe state rather than requiring a different Trainer/model family?

That is much nastier than simply changing an image tensor from four dimensions to five.

---

# Recipe 3 — YuE2 hybrid AR + NAR audio training

**Implementation examined:** AI Toolkit `0.13.20`, commit `0bd34111`, current `YuE2AudioModel` training path.

There isn't a checked-in complete end-user YuE2 training YAML establishing every optimizer setting, so those parts are deliberately marked unknown.

Primary sources: [`yue2_model.py`](https://github.com/ostris/ai-toolkit/blob/0bd34111b85c3544f8fb29831a7bc985b35fb462/extensions_built_in/audio_models/yue2/yue2_model.py), [`src/model.py`](https://github.com/ostris/ai-toolkit/blob/0bd34111b85c3544f8fb29831a7bc985b35fb462/extensions_built_in/audio_models/yue2/src/model.py), [`src/tokenizer.py`](https://github.com/ostris/ai-toolkit/blob/0bd34111b85c3544f8fb29831a7bc985b35fb462/extensions_built_in/audio_models/yue2/src/tokenizer.py), [`src/vae.py`](https://github.com/ostris/ai-toolkit/blob/0bd34111b85c3544f8fb29831a7bc985b35fb462/extensions_built_in/audio_models/yue2/src/vae.py).

### 1. Components

**Observed:** YuE2 contains two major trainable experts:

```text
AR expert
    semantic/music-token generation

NAR expert
    flow-matching acoustic rendering
```

Both are Qwen3-shaped transformer experts inside the YuE2 model.

AI Toolkit targets **both `YuE2AR` and `YuE2NAR` with one LoRA network**.

Additional components include:

```text
48 kHz stereo waveform VAE
text tokenizer / AR embedding table

training-cache only:
    MERT-v2-FullSong
    community semantic tokenizer head
    SheetSage2       (default COT mode)
    optional source separator
```

**Observed:** because the official YuE2 audio-to-token encoder has not been released, AI Toolkit currently obtains training semantic tokens using the community `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4` tokenizer.

### 2. Data and one training step

**Observed:** source audio is:

```text
[B, 2, samples] @ 48 kHz
```

The VAE turns it into 64-channel acoustic latents at approximately 25 frames/sec:

```text
[B, T, 64]
```

Separately, the semantic tokenizer converts the same real song into discrete codec tokens:

```text
[B, T]
```

So one source waveform creates **two fundamentally different training representations**:

```text
continuous acoustic latents
discrete semantic/music tokens
```

The cached training item can also contain an ABC musical score plus prompt information such as style and lyrics.

**Observed:** the default training window is 1500 latent frames, approximately 60 seconds. For longer songs, one contiguous acoustic window is randomly selected each step.

The NAR path trains a flow objective on that selected VAE-latent window:

```text
noise - real audio latent
```

The AR path is different.

It receives:

```text
instruction
+ style/lyrics prefix
+ optional ABC score
+ codec tokens
```

and trains ordinary **next-token cross entropy** on the codec-token sequence.

If the NAR window starts in the middle of the song, the implementation deliberately trains the AR objective from the **start of the song** rather than treating the cropped window as an independent sequence.

Then:

```text
AR prefill
   ↓
KV cache
   ↓ detach()
NAR flow forward
```

**Observed:** the AR KV cache is explicitly detached before being supplied to the NAR model. Therefore the NAR flow loss does **not** backpropagate through the AR computation. AR is shaped by its own token loss.

The resulting step therefore contains two different objective families:

```text
AR:
cross entropy over discrete tokens

NAR:
flow-matching loss over continuous audio latents
```

### 3. Optimization and runtime

**Observed:** the model declares both AR and NAR experts as LoRA targets, allowing one LoRA artifact to train both.

**Observed:** `ar_lr_multiplier` can split AR LoRA parameters into a separate optimizer parameter group with a different learning rate.

**Observed:** the VAE and semantic/tokenization machinery do not receive normal training updates.

The semantic tokenizer itself is particularly interesting: it is required to produce training cache content but is unnecessary once that content exists.

**Unknown:** a complete checked-in YuE2 training config establishing the concrete optimizer type, LoRA rank and every optimizer hyperparameter was not found in the examined sources.

### 4. Structural changes during a run

**Observed.**

This case actually does retire components.

AI Toolkit tracks whether audio encoding is still occurring. Once training has proceeded for several calls without another encode request, it explicitly releases:

```text
semantic tokenizer / MERT
SheetSage2 transcriber
optional separator
```

and flushes them from memory.

If encoding is needed again later, those components are lazily recreated.

So there is a real lifecycle:

```text
cache/preparation phase
    encoder components loaded
          ↓
cached representations established
          ↓
training-only phase
    encoder components retired
```

This is not merely CPU/GPU offloading; those component references are actually discarded.

### 5. Outputs and restoration

**Observed:** the LoRA artifact can contain parameters for **both experts**. Before saving, AI Toolkit maps:

```text
YuE2 NAR LoRA → diffusion_model.* keys
YuE2 AR LoRA  → text_encoders.* keys
```

So one adapter artifact spans components that generic image trainers might otherwise classify into completely different slots.

The trained artifact depends on the YuE2 base model for inference.

**Observed:** the community real-audio semantic tokenizer is a **training/preparation dependency**, not a generation dependency: at inference the AR model itself generates semantic codec tokens.

**Unknown:** exact generic optimizer/scheduler/RNG/dataloader state required for bit-identical AI Toolkit resume was not established from this model-specific source.

### 6. Design pressure

This recipe raises several fairly brutal questions:

> Can one optimization step train different components using entirely different mathematical objectives—autoregressive token CE and continuous flow matching—while preserving explicit gradient boundaries between them?

> Can one cached media item contain continuous latents, discrete tokens, symbolic music, prompt embeddings and metadata whose temporal axes are related but not interchangeable?

And:

> Can preparation-only components have an explicit lifecycle—load, produce derived data, retire—without being mistaken for permanent members of the trainable model?

---

# Cross-case findings

These three are enough to break most of the assumptions inherited from image diffusion.

| Tempting assumption                                    | What the recipes show                                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Media is `[B,C,H,W]`                                   | **False.** Video is naturally `[B,C,F,H,W]`; waveform audio begins `[B,C,S]`; YuE2 latents become `[B,T,C]`.    |
| One media item gives one training representation       | **False.** H3 has video + audio latents; YuE2 has acoustic latents + semantic tokens + possibly symbolic score. |
| Every modality uses the same timestep                  | **False.** H3 derives an audio noise time from video time using a modality-specific mapping.                    |
| One model forward produces one prediction              | **False.** H3 jointly predicts video and audio.                                                                 |
| One step has one kind of loss                          | **False.** YuE2 combines AR cross-entropy and NAR flow matching.                                                |
| Temporal crops affect everything identically           | **False.** YuE2 may crop the NAR acoustic window while deliberately training AR from the beginning of the song. |
| Frozen preprocessing components remain part of runtime | **False.** YuE2 loads substantial audio-analysis components for caching and later retires them.                 |
| Missing modality means a different recipe              | **False.** H3 represents missing audio within the same recipe.                                                  |
| “Model family” means one denoiser                      | **False.** Wan2.2 A14B has independent low/high-noise transformers for different parts of inference time.       |
| One adapter belongs to one familiar model slot         | **False.** YuE2's one LoRA can span AR and NAR experts.                                                         |

The big overall constraint I’d carry into the trainer design is:

> **A training job cannot assume one canonical media tensor, one temporal coordinate system, one model prediction, or one loss. A recipe may own several related temporal representations with separate encoders, timebases, noise processes and objectives, while coordinating them as one training example and one optimizer step.**

And I think there’s a second equally important one:

> **Temporal metadata must have semantic meaning, not just tensor dimensions: video frames/FPS, waveform samples/sample-rate, VAE latent frames, semantic-token frames and symbolic-score positions are different coordinate systems even when several happen to be called “time.”**

That one feels especially relevant if the eventual trainer is supposed to grow from image training into video/audio rather than bolting `frames` onto an image-shaped batch later.
