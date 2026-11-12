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
| 16, minimal SDXL step | `steps/finetune/`: one step's batch, noise, timesteps, UNet call, loss and gradient summaries; the base model's hash | Gradients and updates are summaries, comparable as a whole and not element by element |
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
