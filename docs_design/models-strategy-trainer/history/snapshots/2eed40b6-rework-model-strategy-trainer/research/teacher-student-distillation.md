# Research topic: Teacher–student knowledge transfer / distillation

**Claim labels:** **Observed** = directly established by the examined source; **Inferred** = follows from the implementation but is not stated directly; **Unknown** = source does not establish it.

The cases are deliberately different:

| Recipe                                    | Why it matters                                           |
| ----------------------------------------- | -------------------------------------------------------- |
| Musubi MiniMax-H3 teacher matching        | teacher/student are **roles of the same physical model** |
| DeiT hard distillation                    | teacher and student are **different architectures**      |
| Seq-NAT offline KD                        | teacher is **not present during training at all**        |
| OpenAI progressive diffusion distillation | teacher/student relationship **changes during the run**  |
| veRL on-policy distillation               | teacher is a **remote inference service/resource pool**  |

---

## Recipe 1 — MiniMax-H3 T2VA LoRA endpoint teacher matching

**Implementation:** Musubi Tuner v0.3.5, commit `4e7c714`, `minimax_h3_train_network.py`, recipe `--task t2va --h3_teacher_matching --h3_teacher_conditions first,last`. v0.3.5 explicitly introduced H3 teacher-matching training. ([GitHub][1])

Primary sources: [MiniMax-H3 training guide](https://github.com/kohya-ss/musubi-tuner/blob/4e7c7149249e7715e9168920feb4c420423abba7/docs/minimax_h3.md?utm_source=chatgpt.com) · [advanced internals](https://github.com/kohya-ss/musubi-tuner/blob/4e7c7149249e7715e9168920feb4c420423abba7/docs/minimax_h3_advanced.md?utm_source=chatgpt.com) · [trainer implementation](https://github.com/kohya-ss/musubi-tuner/blob/4e7c7149249e7715e9168920feb4c420423abba7/src/musubi_tuner/minimax_h3_train_network.py?utm_source=chatgpt.com)

**1. Components.** **Observed:** one MiniMax-H3 transformer base and one trainable LoRA participate. The student forward uses the LoRA and T2VA/text-only conditioning. The teacher forward is the **same transformer with the LoRA disabled**, under no-grad, with privileged first/last-frame conditioning. Thus “teacher” is not another independently loaded model. The video/audio VAEs and Qwen3-VL encoder are primarily involved in preparing cached representations rather than being trained in this recipe. Musubi explicitly describes the student as T2VA and the teacher as the frozen base under richer conditions. ([GitHub][2])

**2. Data and one step.** **Observed:** the student consumes target video/audio latents plus ordinary text conditioning. The teacher additionally consumes cached endpoint conditions and its richer text presentation. The teacher-matching rows deliberately use a richer cache task than the student: endpoint latents are cached as FL2VA while training remains T2VA. One base timestep is sampled and converted into separate video/audio sigmas; then teacher and student predictions are compared. ([GitHub][2])

**3. Optimization/runtime.** **Observed:** only the LoRA receives optimizer updates. The frozen base still executes twice: student forward with LoRA active and one extra no-grad teacher forward with it disabled. The documented default LoRA targets attention qkv/output and MLP layers in the 50 main DiT blocks. Teacher matching therefore increases compute without requiring a second copy of the base weights. ([GitHub][2])

**4. Changes during the run.** **Not present structurally.** Sigma-dependent gates can change whether a step emphasizes privileged teaching or preservation, but components are not created/replaced and optimizer ownership does not change.

**5. Outputs/restoration.** **Observed:** the deployed artifact is a LoRA that depends on the MiniMax-H3 base. Teacher settings are written into LoRA metadata. **Unknown:** the H3-specific sources do not establish every state necessary for bit-exact continuation of a run.

**6. Design pressure.**

> Can one physical component execute multiple roles in one step with different conditioning, adapter state, and gradient mode, without requiring separate fixed `teacher_model` and `student_model` slots?

That already argues pretty strongly that **teacher/student should be roles or relationships, not necessarily component types**.

---

## Recipe 2 — DeiT-base hard distillation from RegNetY-16GF

**Implementation:** Facebook Research DeiT, commit `7e160fe`; official recipe trains `deit_base_distilled_patch16_224` using `regnety_160` as the hard-distillation teacher for 300 ImageNet epochs. ([GitHub][3])

Primary sources: [DeiT training README](https://github.com/facebookresearch/deit/blob/main/README_deit.md?utm_source=chatgpt.com) · [main.py](https://github.com/facebookresearch/deit/blob/main/main.py?utm_source=chatgpt.com) · [paper](https://proceedings.mlr.press/v139/touvron21a?utm_source=chatgpt.com)

**1. Components.** **Observed:** the student is a Vision Transformer with an additional distillation token/head. The teacher is a separately instantiated pretrained RegNetY convolutional network. The teacher is put in eval mode and is not optimized. This is genuine **cross-architecture distillation: CNN → Transformer**. The paper specifically reports RegNetY-16GF as its default teacher. ([Proceedings of Machine Learning Research][4])

**2. Data and one step.** **Observed:** an ImageNet image and class target enter the student. The same input is passed to the frozen teacher. The student produces a normal classification output plus a distillation output. In hard-distillation mode, the normal head is trained from the dataset label while the distillation head is trained against the teacher's argmax class; the two losses are mixed by `distillation_alpha`. The teacher forward occurs under no-grad. ([GitHub][5])

**3. Optimization/runtime.** **Observed:** gradients/update apply to the DeiT student only. The CNN teacher nevertheless must be resident and execute every batch. The optimizer sees the student parameters; model EMA is separately supported for the student. ([GitHub][5])

**4. Changes during the run.** **Not present.** Teacher/student roles remain fixed.

**5. Outputs/restoration.** **Observed:** the useful trained artifact is the distilled DeiT model; the RegNet teacher is not required for inference. The training checkpoint can include student model, optimizer, LR scheduler, epoch, EMA and AMP scaler state. ([GitHub][5])

**6. Design pressure.**

> Can teacher and student be entirely different model strategies, provided the recipe defines a common supervision space—in this case class logits/classes?

This answers your earlier “can model type 1 learn what model type 2 does?” question nicely: **yes; their internals need not line up at all.**

---

## Recipe 3 — Seq-NAT offline sequence-level knowledge distillation

**Implementation:** `ictnlp/Seq-NAT`, commit `6e00c955...`; WMT14 En–De NAT baseline pretraining from `data-bin/wmt14_en_de_distill`, via `training_scripts/pretrain.sh`.

Primary sources: [Seq-NAT repository/guide](https://github.com/ictnlp/Seq-NAT/tree/6e00c95532dc9a2ef678760c3581ea452fbc44cd?utm_source=chatgpt.com) · [paper](https://direct.mit.edu/coli/article/47/4/891/107176/Sequence-Level-Training-for-Non-Autoregressive?utm_source=chatgpt.com) · [pretrain.sh](https://github.com/ictnlp/Seq-NAT/blob/6e00c95532dc9a2ef678760c3581ea452fbc44cd/training_scripts/pretrain.sh?utm_source=chatgpt.com)

**1. Components.** **Observed:** the teacher is an autoregressive Transformer, while the student is a non-autoregressive Transformer. But crucially, they do **not coexist in the student training run**. The documented workflow is to train the AR teacher first, use it to decode the training corpus, then train the NAT model against those generated translations. ([MIT Press Direct][6])

So:

```text
AR teacher
    ↓ offline generation
distilled translation corpus
    ↓
NAT student training
```

**2. Data and one step.** **Observed:** once NAT training starts, a batch contains source/target token sequences from the distilled corpus. The criterion code describes outputs as `[batch, length, ...]`, targets and masks as `[batch, length]`. The student performs an ordinary NAT forward and its objectives operate against the materialized target sequences. There is **no teacher forward in the training step**.

**3. Optimization/runtime.** **Observed:** only the NAT model is needed during this run. `pretrain.sh` uses Adam, inverse-square-root LR scheduling, FP16, gradient accumulation and a label-smoothed NAT criterion. The AR teacher consumes **zero training-time VRAM** because its work has already been turned into data.

**4. Changes during the run.** **Not present.** The important transition occurs **before** the run: executable teacher → generated dataset. Nothing dynamically assumes the teacher role afterward.

**5. Outputs/restoration.** **Observed:** the output is a standalone NAT checkpoint. **Inferred:** reproducing the experiment also depends on the exact distilled corpus and preprocessing/dictionary, even though those are not part of the trained weights. **Unknown:** I have not audited this Fairseq fork deeply enough to claim bit-identical resume state.

**6. Design pressure.**

> Can “knowledge transfer” be represented when the teacher is an upstream producer of training artifacts rather than a runtime component at all?

This is a big one. A generic `teacher_model` field would completely miss this recipe.

---

## Recipe 4 — OpenAI progressive diffusion distillation

**Implementation:** `openai/consistency_models`, commit `e32b69e`, `CMTrainLoop` with `training_mode="progdist"`, using `KarrasDenoiser.progdist_losses`. The underlying Progressive Distillation method repeatedly halves the number of sampling steps. ([arXiv][7])

Primary sources: [progdist loss implementation](https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py) · [training loop](https://github.com/openai/consistency_models/blob/main/cm/train_util.py?utm_source=chatgpt.com) · [Progressive Distillation paper](https://arxiv.org/abs/2202.00512?utm_source=chatgpt.com)

**1. Components.** **Observed:** the implementation contains a trainable student, a frozen teacher, and additional target/EMA state. When starting from a teacher checkpoint, the student is initialized by copying the teacher weights. Teacher forwards are no-grad. This is not just “big model teaches small model”; teacher/student may initially have the same architecture and parameter count while differing in the number of sampling steps they represent.

**2. Data and one step.** **Observed:** a real image `x_start` is noised at a sampled noise level `t`. The student predicts a denoised result at `t`. The frozen teacher then executes **two no-grad Euler denoising steps**, `t → t₂ → t₃`; those two steps are converted into the one-step denoising target the student should reproduce. The implementation supports L1, L2 or LPIPS matching. ([GitHub][8])

Conceptually:

```text
student:
x_t ─────────────→ predicted target

teacher:
x_t → x_t2 → x_t3
       two steps
          ↓
derive equivalent one-step target
```

**3. Optimization/runtime.** **Observed:** the student is optimized with RAdam. Teacher execution is frozen/no-grad; target/EMA state is maintained separately. Both teacher and student therefore participate in runtime even though only the student receives optimizer updates. ([GitHub][9])

**4. Changes during the run.** **Observed — and this is the really interesting case.** When the configured number of scales changes, `reset_training_for_progdist()` copies the **current student weights into the teacher**, creates a **new RAdam optimizer**, resets the ordinary EMA states, can modify the LR-annealing horizon, returns the teacher to eval mode, and resets the stage-local step counter. ([GitHub][9])

So the relationship evolves:

```text
stage N:
teacher_N → student_N

boundary:
student_N → teacher_(N+1)

stage N+1:
teacher_(N+1) → student_(N+1)
```

**5. Outputs/restoration.** **Observed:** checkpoints save student weights, EMA state, optimizer state, target-model state and—specifically for progressive distillation—teacher-model state. The resume code also knows how to restore the teacher. ([GitHub][9]) **Unknown:** RNG/data-loader position sufficient for bit-identical continuation is not established by these save routines.

**6. Design pressure.**

> Can a recipe change component roles and replace optimizer/EMA state atomically at a stage boundary, rather than assuming model-role and optimizer ownership are immutable for the lifetime of a run?

This is probably the nastiest case for a static Trainer design.

---

## Recipe 5 — veRL Qwen3-VL 8B ← Qwen3-VL 32B on-policy distillation

**Implementation:** current veRL main examined at commit `4711cea...`; concrete script `run_qwen3_vl_8b_fsdp.sh`. Defaults are `Qwen3-VL-8B-Instruct` student and `Qwen3-VL-32B-Instruct` teacher, with vLLM rollout, FSDP student training, `k1` distillation and policy-gradient application. ([GitHub][10])

Primary sources: [concrete Qwen3-VL recipe](https://github.com/verl-project/verl/blob/main/examples/on_policy_distillation_trainer/run_qwen3_vl_8b_fsdp.sh) · [OPD design documentation](https://github.com/verl-project/verl/blob/main/docs/algo/opd.md) · [trainer overview](https://github.com/verl-project/verl/blob/main/examples/on_policy_distillation_trainer/README.md?utm_source=chatgpt.com)

**1. Components.** **Observed:** there is a trainable 8B student actor, student rollout inference through vLLM, and a frozen 32B teacher served via its own **Ray resource pool and vLLM inference server**. Teacher and student are independent models and can have different parameter counts, but this implementation requires them to share tokenizer/vocabulary for token-level distillation. ([GitHub][11])

**2. Data and one step.** **Observed:** Geo3K prompts/images enter the recipe. The student first generates its **own rollout**. The teacher is then queried on states along that student-generated trajectory and supplies next-token log probabilities. The distillation signal therefore follows states chosen by the student rather than trajectories generated by the teacher. The examined script disables task rewards, uses `k1`, and enables policy-gradient application of the distillation signal. ([GitHub][10])

**3. Optimization/runtime.** **Observed:** optimizer updates belong to the student actor. The teacher is frozen but occupies a separate GPU/resource pool and is accessed through an inference client/server path. The concrete script uses FSDP parameter and optimizer offload for the student and tensor parallelism for both rollout and teacher inference. ([GitHub][10])

**4. Changes during the run.** **Not present** in this single-teacher recipe. Teacher identity and student ownership remain fixed.

**5. Outputs/restoration.** **Observed:** the meaningful learned artifact is the student; the teacher is training infrastructure, not an inference dependency of the distilled student. **Unknown:** the OPD documentation examined here does not establish everything needed for bit-exact recovery of in-flight Ray/vLLM rollout state.

**6. Design pressure.**

> Can a teacher be an externally scheduled inference capability with independent resources/backend/lifetime, rather than a `torch.nn.Module` owned by the same Trainer process?

And:

> Can the recipe state an explicit compatibility requirement between teacher and student representations—here, shared tokenizer/vocabulary—rather than assuming every model pair can be directly compared?

---

# What the five cases actually establish

I think this is the useful result for your architecture work:

| Assumption a Trainer might make                   | Evidence                                                                                                                                           |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| “Teacher is another model slot.”                  | **False:** Musubi uses one physical model in two roles; Seq-NAT has no live teacher.                                                               |
| “Teacher and student have the same architecture.” | **False:** DeiT uses CNN → Transformer; Seq-NAT uses AR → NAT.                                                                                     |
| “Teacher lives beside the student.”               | **False:** veRL can serve it on a separate Ray/vLLM resource pool.                                                                                 |
| “Distillation compares logits.”                   | **False:** examples include classes, generated sequences, conditioned flow predictions, token distributions and multi-step denoising trajectories. |
| “Teacher is frozen forever.”                      | **False:** progressive distillation promotes the current student into the next teacher.                                                            |
| “The optimizer structure is fixed.”               | **False:** progressive distillation explicitly throws away/recreates optimizer and EMA state at stage transitions.                                 |
| “Teacher must ship with the trained artifact.”    | **Usually false in these cases:** the deployed result is the student or LoRA.                                                                      |
| “A batch has one canonical representation.”       | **False:** H3 needs student and privileged-teacher presentations of the same sample; OPD adds student-generated trajectories.                      |

So I would phrase the resulting overall constraint something like:

> **A training system should not assume that “teacher” and “student” identify fixed model components. Knowledge transfer may relate execution roles, independent architectures, upstream-generated artifacts, evolving model copies, or remote inference services. The recipe must own how those participants are observed, compared, updated, and transitioned.**

And one other constraint follows pretty cleanly:

> **The transferable thing is not universally a tensor of logits. A recipe needs an explicit comparison/supervision space: class outputs, token distributions, generated sequences, features, flow predictions, trajectories, decoded outputs, or another representation established by that recipe.**

That second point is probably the important answer to your earlier **“model type A learns what model type B does”** question. The architectural requirement isn't that A and B look alike. It's that the particular transfer recipe can establish **some meaningful bridge between their behaviors**.

For your trainer design research, I think these five are enough to establish the major shape of the problem. The next edge case I'd investigate separately is **cross-representation distillation where the teacher doesn't even provide a conventional target—e.g. score-distillation/SDS-style guidance**. That could tell us whether even “teacher produces supervision output” is too restrictive.

[1]: https://github.com/kohya-ss/musubi-tuner/releases?utm_source=chatgpt.com "Releases · kohya-ss/musubi-tuner · GitHub"
[2]: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/minimax_h3.md?utm_source=chatgpt.com "musubi-tuner/docs/minimax_h3.md at main · kohya-ss/musubi-tuner · GitHub"
[3]: https://github.com/facebookresearch/deit/blob/main/README_deit.md?utm_source=chatgpt.com "deit/README_deit.md at main · facebookresearch/deit · GitHub"
[4]: https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf?utm_source=chatgpt.com "Training data-efficient image transformers & distillation through attention"
[5]: https://github.com/facebookresearch/deit/blob/main/main.py?utm_source=chatgpt.com "deit/main.py at main · facebookresearch/deit · GitHub"
[6]: https://direct.mit.edu/coli/article/47/4/891/107176/Sequence-Level-Training-for-Non-Autoregressive?utm_source=chatgpt.com "Sequence-Level Training for Non-Autoregressive Neural Machine Translation | Computational Linguistics | MIT Press"
[7]: https://arxiv.org/abs/2202.00512?utm_source=chatgpt.com "Progressive Distillation for Fast Sampling of Diffusion Models"
[8]: https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py "consistency_models/cm/karras_diffusion.py at main · openai/consistency_models · GitHub"
[9]: https://github.com/openai/consistency_models/blob/main/cm/train_util.py "consistency_models/cm/train_util.py at main · openai/consistency_models · GitHub"
[10]: https://github.com/verl-project/verl/blob/main/examples/on_policy_distillation_trainer/run_qwen3_vl_8b_fsdp.sh "verl/examples/on_policy_distillation_trainer/run_qwen3_vl_8b_fsdp.sh at main · verl-project/verl · GitHub"
[11]: https://github.com/verl-project/verl/blob/main/docs/algo/opd.md?utm_source=chatgpt.com "verl/docs/algo/opd.md at main · verl-project/verl · GitHub"
