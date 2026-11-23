# Trainer baseline

This folder pins the last revision where today's trainer is verified to train
and records what it does, so the new system has something to be checked
against. It belongs to `tasks.md` section 12 of the
`rework-model-strategy-trainer` change. Recorded on 2026-11-10.

## Scope

The user set the scope on 2026-11-10: record only the two configurations the
trainer has actually been run with, SDXL PEFT (LoRA) and SDXL fine-tune, on
the Windows machine where the model, dataset and caches live. SD, SD3, EDM2,
cache outputs, validation, sampling and save-and-resume are not recorded. A
later check that wants one of those reruns the pinned revision where it can
run.

No multi-GPU run is recorded. The pinned trainer cannot start one on this
machine (see Findings), and the user decided on 2026-11-10 not to force one
for the sake of a baseline.

## What later sections can use

| Section | Recorded here | Limit |
| --- | --- | --- |
| 16, minimal SDXL step | `steps/finetune/`: one step's batch, noise, timesteps, UNet call, loss and gradient summaries; the base model's hash. `reference/`: the same step made in this environment, with an exact digest of every gradient and update (Reference step, below) | The Windows recording's gradients and updates are summaries, comparable as a whole; the reference is exact, but only in this environment |
| 18, SDXL fine-tune run and cost check | `runs/finetune-run1..3`: loss series, step time, memory, saved-product structure; `cost_probe/` and the `rmst-cost-probe` tag | The checkpoints' tensor values were not kept; a value comparison reruns the pinned revision |
| 19, SDXL LoRA run | `runs/peft-run1..3`, `steps/peft/`, the saved LoRA files and the full initial LoRA weights on `D:` | Losses after the first step match only when the run starts from the stored initial weights |

## Pinned revision

| | |
| --- | --- |
| Tag | `rmst-baseline` |
| Commit | `d880bd2e31b8ac20906fdef4309b6cbb7ff85129` (2026-07-19) |
| Checkout | `D:\Projects\sd-scripts`, branch `adapter-layer`, no tracked changes |

How it was chosen:

- The run logs under `D:\Projects\sd-scripts\outputs` hold 526 runs. The last
  one started on 2026-07-17 18:33 and stopped at step 14, reported as failed
  with no error message; its cause is unknown. The last complete PEFT run was
  2026-07-16 03:07 and the last complete fine-tune run was 2026-05-14 16:46.
  Both used `tests/test_peft_resource_sampled`.
- Those runs were on `d79524f3` or earlier. The checkout was then pulled to
  `d880bd2e`, which had never been run.
- On 2026-11-10 both configurations completed 50 steps on `d880bd2e`. It is
  the newest revision with evidence that it trains, and the revision the only
  machine that runs the trainer holds.
- The 41 commits between it and the Python switch were never pulled to that
  checkout or run. From `33f73a4d` (2026-10-11) the repository declares
  Python 3.14t and torch 2.14.1 and no longer installs xformers, and today's
  trainer has never been run in that environment.

What a later reader should know about the pin:

- **The trainer code is almost the current code.** Between the pin and
  `c343cb84`, `library/` and `train.py` differ in nine files:
  `library/training/runners/trainer.py` by four lines, seven metadata and
  utility files, and the added research module
  `library/training/execution.py`, which the trainer does not use. The pin is
  in effect the Windows environment as much as the code.
- **Saved-product metadata may differ at later revisions**, because the
  metadata files changed.
- **Both tags are local to the WSL clone.** The Windows checkout does not
  have them. If it moves, `git -C /mnt/d/Projects/sd-scripts checkout d880bd2e31b8ac20906fdef4309b6cbb7ff85129`
  restores the revision.
- **The pin does not contain the cost probe.** See Cost probe.

## Environment

| | Training runs | Cost probe |
| --- | --- | --- |
| Machine | Ryzen 9 3950X, 64 GB | same |
| OS | Windows 11 Pro Insider Preview 10.0.26340 | WSL2, kernel 6.18.40.1 |
| Python | 3.13.5, GIL enabled | 3.14.5 free-threaded, GIL disabled |
| torch | 2.11.0+cu130, cuDNN 9.19 | not used |
| GPU | RTX 3090 24 GB, driver 621.14, power limit 262.5 W of 350 W | none |
| Revision | `rmst-baseline`, `d880bd2e` | `rmst-cost-probe`, `c343cb84` |
| Packages | [environment-windows.json](environment-windows.json) | [environment-wsl.json](environment-wsl.json) |

The RTX 3090 also drives the display. About 2.1 GB of its memory was in use
by the desktop before each run; `runs/<run>/gpu_before.csv` records the state.
The RTX 3080 (power limit 256 W of 320 W) was idle.

Inputs:

| | |
| --- | --- |
| Base model | `D:\stable-diffusion-webui-reForge\models\Stable-diffusion\21862-seele_pop3_v-pred.safetensors`, 6,938,046,572 bytes, SHA-256 `200311107e6bd1adf20fa902b7c726c2862a70a1e8bb3741dae211e3af22ba54` |
| Dataset | `D:\Projects\sd-scripts\tests\assets\training_images`, 550 images with 550 captions |
| Caches | `D:\Projects\sd-scripts\benchmark_cache`, latents and text-encoder outputs, reused and not rebuilt |

The dataset and caches are not hashed. The recorded steps carry their own
cached latents and text-encoder outputs, so a step comparison does not depend
on them.

## Recorded runs

Each configuration ran three times as a separate process, alternating between
the two. All runs use seed 42, batch size 2, 50 steps, bf16 mixed precision,
gradient checkpointing, xformers attention, AdamW8bit at 1e-5 on the UNet,
`objective.prediction: v_prediction`, and cached latents and text-encoder
outputs. The two configs also differ in `loss.v_parameterization`, true for
fine-tune and false for PEFT, as their source configs always have.

| Run | Seconds per step | Training loop | Peak allocated | Reserved | Final average loss |
| --- | --- | --- | --- | --- | --- |
| peft-run1 | 1.731 | 86.6 s | 7,342 MB | 8,258 MB | 0.075131 |
| peft-run2 | 1.935 | 96.8 s | 7,342 MB | 8,258 MB | 0.075137 |
| peft-run3 | 2.039 | 102.0 s | 7,342 MB | 8,258 MB | 0.075137 |
| finetune-run1 | 1.257 | 62.9 s | 15,743 MB | 16,678 MB | 0.083857 |
| finetune-run2 | 1.249 | 62.5 s | 15,743 MB | 16,678 MB | 0.083862 |
| finetune-run3 | 1.441 | 72.0 s | 15,743 MB | 16,678 MB | 0.083862 |

Peak allocated is the allocator peak during the training loop.

Run-to-run spread:

| | PEFT | Fine-tune |
| --- | --- | --- |
| Seconds per step, median (range) | 1.935 (1.731–2.039) | 1.257 (1.249–1.441) |
| Step-time spread | 15.9% of the median | 15.2% of the median |
| Peak allocated memory | identical | identical |
| Loss at step 0 | identical | identical |
| Largest per-step loss difference | 4.7e-5 | 7.3e-5 |
| Final average loss difference | 6e-6 | 5e-6 |

For scale, the historical runs took 1.89–2.03 s per step for PEFT on
`d79524f3` (2026-07-16) and 1.42 s for fine-tune on 2026-05-14.

`runs/<run>/` holds the composed config, the Hydra overrides, the trainer's
own benchmark report and `summary.json`. The summary has the timing
milestones, phase totals, memory, the per-step loss and learning-rate series
and a check of the saved product: size, SHA-256, tensor count, dtypes and
metadata keys. The saved products differ in hash from run to run. The PEFT
product is 198 MB with 2,382 tensors and the fine-tune product is 6.9 GB with
2,514 tensors, all fp16.

The raw outputs are in `D:\Projects\sd-scripts\outputs\rmst-baseline\`
(about 1.8 GB). The PEFT products are kept there. The fine-tune checkpoints
were deleted on the user's instruction, so for fine-tune only the structure
in `summary.json` survives; comparing tensor values needs a rerun of the
pinned revision, which takes about a minute and a half.

## Recorded steps

`steps/peft/` and `steps/finetune/` each hold the first three training steps
of a run, written by [record_steps.py](record_steps.py). The recorder runs the
pinned `train.py` and adds observers; it changes nothing the trainer computes.
The loss of the first recorded step equals the first loss of the 50-step
runs, and the next two agree with them to about 1e-5.

| File | Contents |
| --- | --- |
| `config.yaml`, `overrides.yaml` | The composed config and overrides of the recording run |
| `parameters.json` | Name, optimizer group, shape and dtype of every trainable tensor, in the order used by the per-parameter arrays (1,588 float32 tensors for PEFT, 1,680 bfloat16 tensors for fine-tune) |
| `initial.safetensors` | Per-parameter sum, L2 norm and 64 fixed strided samples before the first step |
| `step_NNN.safetensors` | The batch (latents, cached text-encoder outputs, loss weights), noise, noisy latents, timesteps, every argument and the output of the UNet call, target, per-sample loss, loss, final loss, and per-parameter sum, L2 norm and samples of the gradients at and of the parameters after the optimizer step |
| `step_NNN.json` | Shapes and dtypes, captions, image ids, bucket size, size and crop conditioning, learning rates, and which tensors were stored once because they are bit-identical to another (`same_as`) |
| `repeat_comparison.json` | The difference between this recording and a second one made the same way, written by [compare_steps.py](compare_steps.py) |

Limits of a recording:

- **Gradients are read after clipping.** The trainer clips to a norm of 1.0
  before the optimizer step. The recorded global gradient norms are 0.005 to
  0.007 for PEFT and 0.04 to 0.11 for fine-tune, so clipping never applied
  and the recorded gradients are the unclipped ones. The clipping path is not
  exercised by these steps.
- **Parameters and gradients are summaries.** A recording cannot initialize
  a model. For fine-tune the initial parameters are the base model. For PEFT
  the full initial LoRA weights are stored separately, below.
- **Optimizer state is not recorded.** Steps 1 and 2 can be compared only by
  a system that ran step 0 itself.
- **One batch per optimizer step.** The recorder refuses gradient
  accumulation.

### Initial LoRA weights

`D:\Projects\sd-scripts\outputs\rmst-baseline\peft-initial\steps\initial_full.safetensors`
holds every trainable LoRA tensor in full before the first step: 395,559,152
bytes, SHA-256
`75959430f1e8860ce6d90fcaec3c8bd56cc2407e68e8b2bca86731bc9dbe1cbe`. It is too
large for the repository. It comes from a third recording whose
`initial.safetensors` and `parameters.json` are byte-identical to the ones in
`steps/peft/`, so it is the initialization of the stored recording and of
every PEFT run above.

### Reproducibility of a step

Two recordings with the same seed agree bit for bit on the batch, the noise,
the timesteps, the UNet output and the loss of step 0. Their gradients differ
at rounding level, and from step 1 on the parameters and losses drift.

The trainer at this revision does not enable deterministic algorithms, so
the backward pass uses CUDA kernels whose accumulation order is not fixed.
Which operation is responsible has not been identified.

Agreement between the two recordings, from `repeat_comparison.json`:

| | PEFT step 0 | PEFT steps 1–2 | Fine-tune step 0 | Fine-tune steps 1–2 |
| --- | --- | --- | --- | --- |
| Largest gradient difference per sampled element | 1.2e-6 | 2.0e-6 to 3.1e-5 | 3.8e-6 | 3.2e-5 to 1.5e-4 |
| Global gradient norm, relative difference | 1.1e-3 | 2.2e-3 to 1.1e-2 | 1.7e-4 | 4.3e-3 to 1.5e-2 |
| Cosine of sampled gradients | 0.99999 | 0.9947 to 0.99997 | 0.999998 | 0.9997 to 0.9999 |
| Cosine of sampled parameter updates | 0.998 | 0.917 to 0.940 | 0.996 | 0.910 to 0.936 |
| Updated elements with opposite sign | 0.4% | 7% to 9% | 0.1% | 0.8% to 1.1% |

Parameter updates cannot be compared element by element. The first Adam
update of an element is the learning rate, 1e-5, with the sign of its
gradient, whatever the gradient's size. Rounding-level noise in a near-zero
gradient therefore flips a whole update, and the difference between two
recordings (up to 1.8e-5 per sampled element for PEFT) is larger than the
update itself. In the fine-tune recording the trainable weights are bfloat16
and only 8.4% of the sampled elements change at step 0, because most updates
of 1e-5 round away.

For comparisons this means:

- Inputs, noise, timesteps, model output and loss of the first step can be
  matched exactly.
- Gradients and parameter updates are compared as a whole, with the measures
  in the table. Each column comes from one pair of recordings, so it shows
  the scale of the baseline's own noise and is not a tested threshold.

## Cost probe

Three process runs of
`uv run python -m tests.unit.training.execution_cost_probe --iterations 10000 --repeats 7`
are in `cost_probe/`. Values are the median of the three process medians, in
microseconds per call, with the range of the process medians.

| Scope | Candidate | Matched direct Python | Paired excess |
| --- | --- | --- | --- |
| 1 cheap operation | 2.72 (2.59–2.73) | 1.53 (1.49–1.55) | 1.17 |
| 4 cheap operations | 3.92 (3.88–3.94) | 1.70 (1.70–1.74) | 2.19 |
| 16 cheap operations | 8.83 (8.80–9.25) | 2.42 (2.37–2.51) | 6.44 |

| Scope | Microseconds per call |
| --- | --- |
| Current-loop dispatch, 4 cheap operations | 2.46 (2.46–2.50) |
| Workflow tick | 11.11 (10.83–11.42) |
| Workflow tick with 256 dormant actions | 11.13 (11.07–11.47) |
| Workflow tick with 1,024 dormant actions | 11.36 (10.78–11.68) |

How these were taken:

- **Revision.** `c343cb842223d10dc293c558026f5fd8b6aa7ba5`, tagged
  `rmst-cost-probe`. `library/training/execution.py` was added on 2026-09-27 and the probe
  on 2026-10-01, so neither exists at `rmst-baseline`. No tracked file was
  modified at the time. The JSON files do not record the revision themselves.
- **Interpreter.** CPython 3.14.5 free-threaded with the GIL disabled,
  confirmed by importing the probe's module in the same environment.
- **The loop it measures is the pinned loop.** The probe hashes
  `library/training/phases/training_loop.py`; its `source_sha256`,
  `7da8804b…d88d`, equals the file's hash at both tags.

The 2026-10-01 numbers in `research/training-mechanism-sketch.md` and
`research/execution-cost-results.json` were taken on CPython 3.13.13 and are
unchanged. Their paired excess was 2.13, 3.79 and 10.83 microseconds, so the
excess is about 45% lower on this interpreter. Neither set is a threshold by
itself: G3.7 requires the retained probe and the new implementation to be
measured in the same environment when section 18 runs.

Sections 16 and 18 move `execution.py` out of the way before that check. To
rerun the probe then:

```bash
git worktree add <path> rmst-cost-probe
cd <path>
uv run python -m tests.unit.training.execution_cost_probe --iterations 10000 --repeats 7
```

## Pass criteria

From `tasks.md` 12.2, as the user settled it on 2026-11-11. No fixed
allowance is used: a fixed percentage says nothing against runs that vary
this much, and the history is logged in enough detail to compare against
directly.

- **Recorded steps** match exactly under the accepted numerical policy. With
  the findings above, that is an exact match for the first step's inputs,
  noise, timesteps, model output and loss. Gradients and updates are compared
  as a whole, read against the reproducibility table.
- **Loss** stays within the baseline's run-to-run spread.
- **Throughput and peak memory** are read against the recorded runs above and
  the historical runs below, using their logs and phase breakdowns.

Refined by the user on 2026-11-16, after the environment probe below:

- **The Windows baseline is for performance and resource usage.** Its losses
  and recorded steps show obvious problems, but they are not expected to
  match one to one across Windows and WSL/Linux.
- **Exact correctness is checked in this environment.** A reference step is
  made here by running today's model code on the recorded step's inputs with
  PyTorch's deterministic algorithms on, and the new system's step must equal
  it bit for bit: model output, loss, gradients and the optimizer update.
- **The historical runs are reference only**, for throughput and resource
  usage.
- **Record the GPUs' tensor-unlock state with every throughput measurement**,
  and compare like with like (see GPU state below). The baseline should be
  rechecked with the cards unlocked before it is compared for performance.

### Environment probe (2026-11-16)

Today's SDXL UNet, run in this repository's environment (CPython 3.14.5t,
torch 2.14.1, SDPA attention because xformers is not installed, autocast
bf16, the RTX 3080 unlocked) on the recorded fine-tune step 0:

| Against the Windows recording | This environment | Baseline's own repeat |
| --- | --- | --- |
| Model output | 46% of elements identical, relative L2 0.36%, cosine 0.999994 | identical |
| Loss | relative difference 1.25e-4 | identical |
| Global gradient norm | relative difference 0.99% | 1.7e-4 |
| Sampled gradients | cosine 0.99991, 2.1% opposite signs | 0.999998, 0.1% |

The differences are at bf16 rounding scale, which is what different kernels
give (attention, torch version, operating system); adding xformers alone
would not remove them. Without autocast the output is slightly further from
the recording, so the baseline ran under autocast. The recorded per-sample
loss is the mean squared error times the min-SNR (gamma 5) weight for
v-prediction. In this environment the forward and the loss repeat bit for
bit; the backward does too once `torch.use_deterministic_algorithms(True)`
is set with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and otherwise differs at about
1e-5. Forward and backward together peaked at 11.4 GB.

### Reference step (2026-11-23)

[reference_step.py](reference_step.py) makes the reference the new system's
step must equal bit for bit. It runs today's training computation for
fine-tune step 0 in this environment, with PyTorch's deterministic
algorithms on, from the recording's batch, noise, timesteps and
configuration. It calls today's functions in the order the trainer does:
the DDPM schedule with zero-terminal SNR and its noising, the cached text
conditioning, SDXL's size conditioning and the UNet call, the v-prediction
target, the float64 l2 loss with the loss weights and min-SNR weighting,
Accelerate's backward, clipping, and AdamW8bit's step. Two things differ
from today's trainer, both on purpose: attention is SDPA, since xformers is
not installed here, and gradient checkpointing is non-reentrant, the mode
the new system uses. Before the step, the loaded weights are checked against
the recording's initial summaries. The step runs twice from the same initial
state, and the script writes only when both runs agree bit for bit; with
`--check` it writes nothing and compares a run in a new process with the
stored reference.

`reference/finetune_step_000.safetensors` holds the UNet's inputs (noisy
latents, timesteps, text context and vector conditioning), its output (bf16
under autocast; today's trainer casts it to float32 through Accelerate's
wrapper, an exact cast), the target, the SNR at each timestep, the per-sample
squared error, the weighted per-sample loss, the loss and clipping's norm.
For every trainable tensor, in the recording's order, which is the UNet's
own parameter order, it keeps the recording's summaries (sum, L2 norm and 64
strided samples) of the gradient after clipping and before the optimizer
step, and of the value after the step. `reference/finetune_step_000.json`
has the SHA-256 of each of those tensors' exact bytes (contiguous, in its own
dtype, with shape and dtype listed beside it), the environment and the
comparisons below.

Made at `1faa2f55` on the RTX 3080 (torch 2.14.1+cu130, CPython 3.14.5t with
the GIL on through Triton, bitsandbytes 0.50.2, diffusers 0.41.0). What it
establishes:

- **The starting weights are the recording's.** The loaded weights' samples
  match the recording's initial summaries exactly, their sums are equal, and
  their norms differ by at most 2.2e-16.
- **It repeats bit for bit.** Two runs in one process agreed, and a run in a
  new process matched the stored reference. The step peaked at 15.0 GiB.
- **Clipping is not exercised.** Clipping's norm is 0.065 (in bf16, as torch
  computes it from bf16 gradients), below the limit of 1.0, so clipping
  multiplies every gradient by exactly 1. A step that clipped wrongly would
  still match; clipping needs its own check.

Against the Windows recording:

| | Identical elements | Relative L2 |
| --- | --- | --- |
| Text context and vector conditioning | all | 0 |
| Noisy latents | 99.994% | 3.5e-7 |
| Target | 70.8% | 5.0e-8 |
| Model output | 45.8% | 0.36% |
| Per-sample loss and loss | none | 7.6e-5 and 7.1e-5 |
| Global gradient norm | none | 1.4% |
| Sampled gradients | 3.0% | 1.4% |
| Sampled parameters after the step | 99.5% | 3.6e-6 |

The conditioning reproduces exactly, and the loss agrees to 7e-5, which
supports the forward computation. Noising and the target differ by single
roundings; the likely cause, not tested here, is the schedule's values
differing in the last bit between the two environments' library versions,
so a formulation that is to match the reference should compute the schedule
as this environment does. The model output differs as in the probe above,
which also includes SDPA in place of xformers. The comparison cannot confirm
the gradients: the recording's own repeat differs by up to 6% in a tensor's
gradient norm (`steps/finetune/repeat_comparison.json`). That the reference's
gradients are today's rests on the script calling today's code path, which a
review checked line by line against the trainer. Rerun with
`.venv-wsl/bin/python -I docs_design/models-strategy-trainer/baseline/reference_step.py --device cuda:1`
(add `--check` to compare without writing); a run takes about a minute and a
half, most of it loading the base model.

### GPU state

Both cards can carry a volatile tensor-rate unlock (applied before the OS
boots, lost on every restart), which raises BF16, FP16 and TF32 tensor
throughput by about 50-80% and leaves FP32 unchanged. It changes how fast
tensor instructions issue, not what they compute. The historical runs were
taken without it; whether the 2026-11-10 runs had it is not known. A BF16
8192x8192 GEMM tells the state in seconds: about 100 TFLOPS unlocked against
63 locked for the RTX 3080 at 256 W, and about 92-95 against 63.5 for the RTX
3090 at 262.5 W. On 2026-11-16 both cards were unlocked.

### Historical runs

`D:\Projects\sd-scripts\tests\test_output` and `benchmark_output` hold 212
benchmark reports. Each has the step time, the runtime trace, a per-phase
time and memory breakdown and the full composed config. Completed 50-step
runs of the baseline's configuration:

| | Runs | Dates | Seconds per step, median (range) |
| --- | --- | --- | --- |
| PEFT (`AdapterMode`) | 41 | 2026-05-05 to 2026-07-16 | 1.85 (1.68–2.32) |
| Fine-tune (`FineTuneMode`) | 14 | 2026-05-05 to 2026-05-14 | 1.42 (1.35–1.52) |

These span many revisions and whatever else the machine was doing at the
time, and not every run had an ideal setup. Read the range accordingly: a
slow run may only reflect its conditions, while a fast run shows a speed the
trainer can reach, so the fast end is never an outlier. The six runs recorded
here fall inside the PEFT range and at or below the fine-tune range.

One caveat applies to any comparison: these runs use Python 3.13.5 and torch
2.11 on Windows, and the repository now targets Python 3.14t and torch
2.14.1. A throughput or memory difference across that gap reflects the
environment as well as the code.

## Findings

- **The tracked fine-tune config does not compose at this revision.** Since
  2026-05-28 `benchmarks/benchmark_sdxl_finetune` composes `/_defaults/default`,
  which adds an `adapter` section, and fine-tune mode rejects that at
  validation. [configs/baseline_sdxl_finetune.yaml](configs/baseline_sdxl_finetune.yaml)
  restores the composition of the May runs with `adapter: null`. Its composed
  config differs from the 2026-05-14 run only in the output folders and in
  `resource_monitor.phase_summary`, whose schema changed from `true` to
  `default`. The PEFT config differs from the 2026-07-16 run only in the
  output folders.
- **Multi-GPU does not start on Windows.** `accelerate launch --multi_gpu`
  fails before any worker starts. The launcher's static rendezvous in torch
  2.11 creates its store with libuv, which the Windows build lacks, and it
  did so with `USE_LIBUV=0` set; [multi_gpu_attempt.log](multi_gpu_attempt.log)
  is the error. A second obstacle was read from the code and not reached:
  Accelerate 1.11 selects the NCCL backend whenever CUDA is available, the
  Windows build reports NCCL unavailable and Gloo available, and the trainer
  passes no backend override. No two-process run was found in the reports
  and resource-monitor logs under `tests/test_output`.
- **Step time is 15% noisy** on this machine at 50 steps, with the display on
  the training GPU and both GPUs power-limited.
- **The backward pass is not bitwise reproducible**, as described above.
- **Fine-tune trains the UNet in bfloat16**, where most updates at this
  learning rate round away, as described above.

## Running the pinned revision

The pinned revision is run from its own checkout, outside this working tree,
with that checkout's virtual environment. From WSL:

```bash
cd /mnt/d/Projects/sd-scripts
git status --short && git rev-parse HEAD      # expect d880bd2e, no tracked changes
mkdir -p outputs/rmst-baseline/harness/configs
cp <this folder>/record_steps.py outputs/rmst-baseline/harness/
cp <this folder>/configs/*.yaml outputs/rmst-baseline/harness/configs/
```

Then in PowerShell, from `D:\Projects\sd-scripts`:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'   # the RTX 3090
$out = 'D:/Projects/sd-scripts/outputs/rmst-baseline/<run>'
$dirs = "output.saving.output_dir=$out", "output.logging.logging_dir=$out/logs", "hydra.run.dir=outputs/rmst-baseline/<run>/hydra"

# PEFT run
.\.venv\Scripts\python.exe train.py --config-name=tests/test_peft_resource_sampled @dirs

# Fine-tune run
.\.venv\Scripts\python.exe train.py --config-dir outputs/rmst-baseline/harness/configs --config-name=baseline_sdxl_finetune @dirs

# Record three steps (either configuration)
$env:RMST_BASELINE_OUT = "$out/steps"; $env:RMST_BASELINE_STEPS = '3'
$env:RMST_BASELINE_FULL_INITIAL = '1'   # optional: also write the trainable tensors in full
.\.venv\Scripts\python.exe outputs\rmst-baseline\harness\record_steps.py --config-name=tests/test_peft_resource_sampled training.max_train_steps=3 @dirs
```

Use the virtual environment's Python directly. `uv run` would try to sync the
environment first.

Back in this repository:

```bash
uv run python docs_design/models-strategy-trainer/baseline/collect_runs.py <folder of runs> <destination>
uv run python docs_design/models-strategy-trainer/baseline/compare_steps.py <recording a> <recording b>
```
