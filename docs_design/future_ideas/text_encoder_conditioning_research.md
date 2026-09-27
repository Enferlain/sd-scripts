---
title: "Text-Encoder Conditioning: Dynamic Captions, Caching, and Editable Representations"
date: 2026-09-25
status: "Research / hypotheses — not an implementation specification"
related:
  - "sae_embedding_aug.md — earlier SAE research; companion note, not superseded"
tags:
  - diffusion-training
  - text-encoder
  - embedding-cache
  - caption-augmentation
  - multi-caption
  - representation-learning
  - research
---

# Text-Encoder Conditioning: Dynamic Captions, Caching, and Editable Representations

> **Research objective:** Preserve arbitrary, dynamically constructed captions—including booru tags from multiple sources, natural-language descriptions, aliases, reordered tags, omissions, and combinations—without paying the full text-encoder (TE) cost for every change and without silently degrading the conditioning that the image model learns to use.
>
> **Scope:** Consolidates the current discussion into a *separate* research note. The earlier `sae_embedding_aug.md` remains the detailed reference for SAE reconstruction and feature-level augmentation. This note concerns the larger conditioning problem, the evidence available as of **2026-09-25**, and the experiments needed before choosing an architecture.

## 1. The actual problem

A conventional diffusion-training setup offers an awkward choice:

- **Live TE:** construct any caption at training time, tokenize and encode it, and retain complete caption flexibility. This incurs compute and possibly significant VRAM use. If the TE is frozen, its outputs are still recomputed for every *unseen* caption.
- **Precomputed TE outputs:** avoid repeated encoder execution (and potentially unload it from training GPU memory), but only captions whose *complete conditioning* was precomputed are faithfully available. New/reordered captions require another encode or an approximation.

Neither is conceptually mandatory. What the image model needs is **suitable conditioning**, not necessarily execution of the original TE. But suitable conditioning is learned relative to the model's original training distribution; a replacement that looks semantically reasonable to us may not work for the denoiser.

The strongest design constraint is therefore: **caption construction must not be dictated by the limitations of the embedding cache**. Storage, retrieval, representation approximations, and actual TE execution should be interchangeable conditioning mechanisms *when validated for the target model*, not assumptions built into dataset format.

### The questions to keep separate

1. Can we expose one image to multiple text representations without damaging rare concepts or introducing unnecessary image repetition?
2. What conditioning information does a particular denoiser actually use: individual token identity, word identity, position, contextual interactions, pooled semantics, etc.?
3. Can cached representations be edited or composed to approximate a *specific* modified caption, not merely perturbed for regularization?
4. If approximation is possible at inference, does training on those approximations preserve or improve the model's learning dynamics?
5. What is the true compute bottleneck for the target architecture: TE execution, its adapter, denoiser evaluations, memory transfer, or cache I/O?

**Do not equate:** caption augmentation, embedding augmentation, faithfully approximated re-encoding, and image-model semantic generalization. These solve different problems.

## 2. Established constraints and terminology

### 2.1 What is cached?

A TE can produce a sequence of contextualized hidden states, possibly alongside pooled outputs, attention masks or other encoder-specific data. A cache must cover the **actual conditioning contract** consumed downstream, not just a convenient tensor. Multi-encoder and adapter-based models may have extra tokenizer and conditioning requirements.

In a contextual transformer, the hidden state associated with a tag or word is generally a function of *the entire allowed context and its position*. Reordering words, substituting one word, or changing a nearby phrase can alter more than one output row. For causal LLM-style encoders, a token can depend on its preceding context; in bidirectional encoders, tokens may depend on both sides.

Consequences:

- Rearranging contextualized token rows is **not** the same operation as re-encoding rearranged text.
- Separately encoding individual tags and concatenating their outputs is **not**, in general, equivalent to jointly encoding the complete tag list.
- Deleting one hidden-state row may leave information about that concept in other rows.
- A pooled sentence vector loses token-by-token conditioning and cannot automatically replace sequence conditioning.
- Changing an encoder's weights invalidates representations computed under older weights. This particularly matters when training the TE itself.

Nothing prevents **approximate** use of these operations, but fidelity and downstream effects must be measured rather than assumed.

### 2.2 The information limit

No deterministic edit to one stored embedding can recover details the stored representation discarded. To represent arbitrary new text, the system must receive information about the change (the new text, token IDs, a structured edit, or other input), then perform *some* computation. The plausible goal is **much cheaper computation than the full original TE**, not unlimited caption editing at literally zero cost.

### 2.3 Reusing the visual side is a separate optimization

Cached VAE image latents can generally be reused for a fixed image and fixed image preprocessing, irrespective of caption variant. New spatial crops/augmentations may invalidate an image-latent cache, depending on where augmentation occurs. This reduces image preprocessing, **not** TE or denoiser costs.

## 3. Caption variants: what the training schedule does and does not do

Suppose each image has two captions, `C1` and `C2`:

| Schedule | Encounters | Interpretation |
| --- | --- | --- |
| Alternate captions between epochs | `I1+C1`, `I2+C1`, then `I1+C2`, `I2+C2` | One caption per image per epoch; both after two epochs. |
| Both captions every epoch | `I1+C1`, `I1+C2`, `I2+C1`, `I2+C2` | Twice as many image–caption examples per epoch. |
| Independent balanced sampling | Each image has its own shuffled queue of caption variants | Balanced coverage without synchronized caption distribution changes at epoch boundaries. |
| Occasional paired encounter | For selected images compute losses for two captions and combine them in one optimizer update | Allows within-image gradient comparison or normalization but still costs two denoiser evaluations. |

**Core result:** Two epochs of the first schedule and one epoch of the second can expose the model to the same image–caption pairs. Do not compare schemes only by epoch counts. Ordering may matter, but neither schedule *intrinsically* teaches equivalence between the captions. They both provide visual supervision for each representation.

### Balanced caption sampling as a baseline

Use a per-image caption queue, reshuffle it after each complete cycle, and log variant exposure. This avoids having every image use the same caption source in a given epoch. The queue need not be uniformly weighted if a clearly stated research question calls for unequal exposure.

**Rare-concept caveat:** If a concept appears explicitly only in one caption variant, uniform variant sampling lowers its supervised exposure compared with always selecting that variant. Track **concept appearances in sampled training captions**, not merely their frequency in the master annotation database. The existing alias/ontology research can help count exact aliases and related-but-not-equivalent labels separately.

### Paired caption losses

For one image and two captions, a straightforward grouped objective is:

```text
L_image = (L(I, C1) + L(I, C2)) / 2
```

The paired losses can be accumulated without keeping both graphs live simultaneously. For a particularly controlled diagnostic, use the same image augmentation, noise sample and diffusion timestep in both passes; the remaining conditioning difference then explains more of the gradient disagreement.

Paired processing **does not** halve denoiser compute. A proposed factor such as `0.8 * L_image` only changes *relative* influence when other examples or objectives use different weights. Multiplying the entire loss by 0.8 is not an overfitting remedy and may be largely normalized away by adaptive optimizers. Control and report **image exposure, denoiser evaluations, optimizer steps, and actual wall-clock/energy cost** separately.

An additional *consistency* loss between predictions for `C1` and `C2` is experimental; use it, if at all, only for genuinely equivalent descriptions. One caption may contain different or extra information, in which case forcing identical predictions can destroy useful conditional distinctions.

### Distinguish caption relationships

- **Exact equivalent:** verified paraphrase / exact alias under the intended meaning.
- **Partially overlapping:** shared concepts, but one description is broader, narrower, or contains additional information.
- **Related but not equivalent:** e.g. `female` vs `1girl` (gender versus gender **and** count); `split` vs `split_form` may require source-specific meaning checks.
- **Conflict or hallucination:** a generated sentence introduces an unsupported attribute or assigns an attribute to the wrong subject.

Do not indiscriminately apply consistency or alias substitution across these groups.

## 4. Dropout and natural-language flexibility

### Tag dropout

- **Percentage-based random dropout:** under independent dropping, each tag occurrence has the same specified removal probability; remaining caption lengths vary.
- **Fixed-count dropout:** if exactly `k` of `n` tags are chosen uniformly, the probability of dropping any *one* tag in that caption is `k/n`. There is **no inherent preferential removal of rare tags within that caption**. Problems arise from varying `n` and the fact that rare concepts have fewer total exposure opportunities.
- **Caption-level dropout for classifier-free guidance:** a distinct objective; multiple caption variants do not automatically make it unnecessary.

Multiple caption variants can reduce dependence on one exact caption string, but do not prove that conventional dropout is unnecessary. Evaluate the interaction rather than assuming the mechanisms are interchangeable.

### Natural-language dropout should preserve meaning

Deleting arbitrary NL tokens risks broken syntax, changing scope/negation, changing the referent of an attribute, or silently producing false supervision. Better research directions:

1. Maintain a **structured annotation internally** (entities, attributes, relationships, visibility/certainty, provenance), but **do not require one rigid text template in the model-facing captions**.
2. Sample coherent semantic units to omit and realize the remaining facts as valid, varied natural language.
3. Preserve original source-specific tag lists separately, including their taxonomies and true aliases.
4. Track what is *visible* and explicitly supervised in each selected caption.

This keeps annotation structure at the data-management level without asserting that the diffusion model should learn one particular writing format. More varied descriptions can also introduce mistakes; verify critical relationships and rare concepts against the image.

**Related evidence:** Caption design matters. An ICCV Workshops 2025 study found randomized caption lengths could balance alignment, aesthetics and diversity better than always using dense captions in its tested setup. The RECAP paper showed benefits from informative recaptioning; another NeurIPS 2023 study found randomizing/augmenting captions could mitigate training-image replication. These are reasons to test controlled variation, **not** proof that multi-source booru + NL captions or any particular dropout policy is optimal. [R1][R2][R3]

## 5. Why tag + natural-language supervision is interesting

A pretrained language model can represent far more descriptions than an image model has learned to render. A TE recognizing the difference between *holding an umbrella* and *standing under an umbrella* does not guarantee that:

- its adapter preserves that distinction;
- the denoiser knows how to use it;
- the visual training data contains adequate examples;
- the model generalizes to that combination rather than falling back on familiar visual patterns.

Tags are not merely fixed image lookups: their combinations and order can matter substantially. NL expands the convenient expression of relationships and unfamiliar combinations, but also exposes more ways the image model can fail. Thus, measure **linguistic representation quality** and **actual image-generation competence** separately.

A useful annotation/retrieval system would keep original Danbooru/e621/other tags intact, associate typed cross-source relationships (exact alias, narrower/broader, related, incompatible), and maintain independently verified NL descriptions. Alternative captions then provide overlapping supervision without pretending that all formulations are semantically identical.

**Model-specific caution:** The Anima model card documents training with tags, NL, mixtures, and random tag dropout; it also recommends descriptive NL prompts. Its behavior alone cannot establish whether instability versus another checkpoint is caused by NL conditioning, training duration, refinement, or the model/adapter. [R4]

## 6. Conditioning/computation options: which preserve arbitrary captions?

| Approach | Arbitrary newly generated captions? | Original TE required *during diffusion training*? | Main qualification |
| --- | --- | --- | --- |
| Full online encoding | Yes | Yes | Reference behavior; cost/VRAM. |
| Precompute complete variants | Only known variants, exactly | No, if TE frozen and cache complete | Storage and restricted online editing. |
| Lazy exact-embedding cache | Yes (encode cache misses) | Yes on misses | Flexible, but worst-case unique-captions workload remains expensive. |
| Mixed exact cache + live encoding | Yes | Yes on misses | Natural implementation baseline. |
| Shared-prefix reuse for causal LLMs | Yes | Yes, partial reuse | Only unchanged prefixes; edits early in caption invalidate most suffix work. Not general for bidirectional encoders. |
| Smaller distilled encoder | Yes | No original TE, after distillation | Requires training, downstream validation; retains compute but potentially much less. |
| Position-aware reusable word embeddings | Potentially, within covered vocabulary/positions | No original TE online for covered units | Approximate, architecture-sensitive; not established for training or long-tail vocabulary. |
| Learned embedding editor | Potentially, for learned edit distribution | No original TE online, after teacher-supervised training | New text/structured edit still needs interpretation; arbitrary unseen edits unproven. |
| SAE perturbation of stored embeddings | Only approximation/regularization, not arbitrary *faithful* caption edits | No | Separate older research objective. |
| Concurrent TE on second GPU | Yes | Yes, on separate device | Full semantics but hardware, transfer and scheduling costs. |

### Practical baseline that does not constrain the research

Keep the full, frozen TE as the **reference conditioning path**. Implement an exact lazy cache with optional persistent storage, and online encode any miss. Cache pre-tokenized captions if useful, but remember that tokenization usually costs far less than the TE. Evaluate low-precision TE inference and a second-device preparation queue *after measuring which stage actually dominates throughput*.

The cache key should include exact input text/token IDs and every relevant conditioning dependency: tokenizer version, special formatting, encoder checkpoint/weights, precision where it affects compatibility, output-layer choice, sequence length/padding/attention masks, and any adapter output that is also cached. Frozen encoder outputs can be cached even if a downstream adapter remains trainable; in that case, recompute the adapter every time from the cached encoder outputs. A trainable encoder cannot use stale detached cached outputs for current gradient-bearing passes.

With a causal encoder, shared-prefix KV caching can reuse computation of genuinely unchanged prefixes, but it is an inference/preparation technique, not a general method for arbitrary reordering or independent tag composition. [R5]

### GPU concurrency as a separate axis

If available hardware permits, prepare text conditioning on a second device while the first trains the denoiser. Move the resulting conditioning to the training device and keep a short prefetch queue. This retains arbitrary captions and the original encoder's behavior. It only helps end-to-end throughput when encoder work can overlap sufficient denoiser work and transfers are controlled. Benchmark **single GPU online, offline exact cache, lazy cache, and concurrent separate-device encoding** before committing to a more complex learned representation.

## 7. What recent literature suggests about the TE's *necessary* information

### 7.1 Contextless embeddings with word identity and position — promising, model-dependent

Spingarn et al., *Text-to-Image Models Need Less from Text Encoders Than You Think* (June 2026), construct:

- **BoT:** context-independent representations for tokens;
- **BoW:** preserves whole-word identity when words occupy multiple tokens;
- **BoPTW:** also preserves the word's **position** in the original prompt.

They obtain reusable representations by averaging original-TE outputs across **unrelated** sentences, with position-constrained averaging for BoPTW. This is **not** a demonstration that naïvely concatenated single-tag outputs reproduce contextual embeddings. On the evaluated DiTs (including SD3 and FLUX models), BoPTW often produced comparable inference outputs to the original full representations. This suggests the denoiser can sometimes perform a substantial portion of the linguistic composition itself. **But the results are not uniformly interchangeable even within DiTs:** on the paper's GenEval aggregate, FLUX.2 Klein scored **82.3** with full embeddings versus **71.3** with BoPTW; FLUX.1 scored **70.4** versus **70.7** in the same table. That architecture-specific difference matters when deciding what is safe to cache or approximate. [R6]

**Critical limits:**

- The result is primarily **inference on pretrained models**, not evidence that training from a pretrained starting point on BoPTW preserves all learning dynamics.
- Their tests with SD2.1 and SDXL **failed badly** using the simplified representations. Do not extrapolate DiT findings to U-Nets. [R6]
- Rare-word and misspelling categories were excluded from part of the benchmark; additional generated examples helped construct rare-word representations. This is material for booru long-tail vocabulary. [R6]
- They benchmarked particular prompts, architectures, and an automated evaluation procedure. Before adopting the method, test held-out artist names, source aliases, detailed attribute binding, multi-character scenes, and training outcomes.

**Follow-up experiment:** independently cached **position-aware words/phrases** for a chosen model, including an explicit rare-vocabulary fallback to online encoding. Compare to full TE encoding on *identical* captions and on caption permutations. Record cache size and miss rate as well as image quality.

### 7.2 Actual token interactions are not uniformly helpful

Kaplan et al., *Follow the Flow* (ACL 2026), used interventions to study information flow across textual tokens in TTI models. Their findings include lexical information concentrated in a subset of tokens and cases where contextualization leads to incorrect associations (their paper describes `pool` interpreted as `pool table` in a particular prompt). [R7]

This cautions against two opposite assumptions:

- **Wrong:** all full-encoder contextual interactions are indispensable and correct.
- **Also wrong:** all contextual interactions can safely be discarded for every architecture and task.

What matters is the **causal use of representation information by the target denoiser**. Track which interactions help attribute binding, relationships, and rare concept control, and which introduce mistakes.

### 7.3 Smaller TE through distillation

Wang et al., *Scaling Down Text Encoders of Text-to-Image Diffusion Models* (CVPR 2025), demonstrate vision-guided distillation of T5-XXL into much smaller T5-based encoders. Their T5-base student is approximately **50× smaller** with comparable image quality in the reported evaluations. Their method includes image-model-informed supervision, not just matching raw text embeddings. [R8]

This is an evidence-backed direction for models whose TE is much more expensive than the denoiser pipeline can economically accommodate. It may be less compelling for a model already using a relatively small LLM encoder. A new student must be checked on long-tail booru vocabulary, multilingual names, tag/NL equivalences, and relationships, not just headline quality.

### 7.4 Replacing or bridging encoders is possible, not transparent

ELLA (2024) trains an **efficient LLM adapter** to connect a language model to a frozen diffusion backbone. It is evidence that the mapping between language features and denoiser conditioning can be learned without retraining both large backbones. It does **not** eliminate the runtime cost of the LLM; use it as an architecture precedent for small conditioning bridges, not as a caching solution by itself. [R9]

## 8. Reusable semantic units and learned embedding edits

This is the more speculative but potentially valuable branch for the original flexibility problem.

### 8.1 Caption composition ≠ embedding composition

It's safe to *construct* a complete caption from structured entities/relationships and run the original TE. It is not safe to assume that pre-encoded descriptions can be concatenated to obtain the same conditioning tensor.

Three hypotheses worth separating:

1. **Word/position reuse:** Contextless but position-aware word representations may work for some denoisers, as [R6] suggests at inference. This may bypass most runtime TE computation for covered words.
2. **Composition network:** Independently cached descriptions plus explicit relation/position data feed a small trainable contextualizer that creates denoiser-ready conditioning.
3. **Embedding editor:** Given a **previous exact conditioning tensor** plus a specified caption edit, a small network learns to predict the TE's output for the *modified complete caption*.

The last approach is closer to replacing re-encoding than SAE feature dropout. For example, offline teacher pairs might be:

```text
E("1girl, red hair, sitting")
E("1girl, black hair, sitting")
edit instruction = substitute(red hair, black hair)

learn: editor(E(original), edit instruction, original text/positions?)
       -> approximately E(modified)
```

A fixed global `red_to_black` embedding-difference vector is only a baseline. Because the TE is contextual, the modification could alter the other output rows and sequence length. The editor may need access to the original tokens, full conditioning sequence, positions, attention mask, and a contextualized representation of the edit itself.

**Information/cost warning:** If arbitrary natural language modifications are allowed, a sufficiently general editor eventually needs substantial language-understanding capability. A cheap editor trained on substitutions might fail on negation, nested relations, or novel multi-subject descriptions. Its feasible scope must be demonstrated, not declared.

### 8.2 Do SAEs help?

The companion `sae_embedding_aug.md` explores SAE reconstruction, per-token versus whole-sequence analysis, feature dropout/reweighting/noise/mixup, and a learned reorder operator. Those are **augmentation and interpretability experiments**, not proof that an SAE can reconstruct the TE's response to arbitrary text edits.

Related outside work:

- **SpLiCE** decomposes CLIP embeddings into sparse combinations of interpretable concepts and demonstrates representation-editing applications. [R10]
- **SAEdit** (2025) uses an SAE to discover token-level directions for continuous image editing, manipulating text embeddings without altering the denoising algorithm. Its [official code](https://github.com/ronen94/SAEdit) includes embedding-data preparation and SAE training. This demonstrates *some targeted edits*, not arbitrary caption re-encoding. [R11]
- **SAEs Do Not Find Canonical Units of Analysis** (ICLR 2025) argues against interpreting an SAE dictionary as a complete, unique set of atomic concepts. Features may be incomplete, composite or dependent on SAE training choices. [R12]

**Necessary control experiment:** compare an SAE-based editor with a **direct residual/sequence editor trained in original embedding space**. Otherwise a useful learned editor could be incorrectly attributed to the SAE. Test a per-token SAE, a sequence-aware editor, and a hybrid only if the simpler baselines justify the extra complexity.

### 8.3 What constitutes success?

Discovery that a feature *correlates* with `red hair` is not discovery of a reliably editable `red hair` control. Require three distinct tests:

1. **Interpretability/probing:** does the feature consistently predict the concept across many contexts?
2. **Representation validity:** does reconstruction/editing preserve other information, appropriate norms, and the distribution the denoiser expects?
3. **Causal intervention:** when adjusted, does the denoiser produce the intended change **without changing unrelated attributes**?

Finally ask whether the result works with a **training** run, not only at inference.

## 9. Research plan and controls

### Stage 0 — Model-specific profiling (required before clever solutions)

Select a target conditioning path (e.g. SDXL or Anima) and document:

- tokenizer(s), actual output tensor(s), positional/padding conventions, attention masks, pooled outputs, adapter path and whether each component trains;
- measured TE wall time/VRAM, adapter time, denoiser time, encoder-output transfer time and disk-cache throughput at realistic lengths/batches;
- per-run distinct-caption rate and how it changes with multiple source captions, word-order randomization and semantic-unit omission;
- precomputed full-caption embedding footprint at practical data sizes and precisions.

This establishes whether the TE is a large enough bottleneck to justify approximations. Keep a fully dynamic original-TE reference path regardless.

### Stage 1 — Controlled, exact-conditioning caption experiments

Hold image set, number of denoiser evaluations, optimizer schedule and evaluation prompts as comparable as possible:

A. One stable caption per image (baseline).  
B. Per-image balanced sampling across verified captions.  
C. B plus carefully designed tag or semantic-unit omission.  
D. Occasional paired-caption gradient averaging (same noise/timestep for within-pair diagnostics).  
E. Only for verified equivalents, optional prediction-consistency loss.

Test separately for source tags, NL paraphrases, genuinely different detail levels, and intentionally order-sensitive prompts. Make sure annotations don't invent unseen features in the image.

**Do not conflate** the effect of caption diversity with simply processing each image more often. Log rare concept exposure and caption-source exposures explicitly.

### Stage 2 — Inference-only conditioning substitution (cheaper diagnostic)

Compare the same pretrained image model and identical noise/seeds/prompts with:

- fresh exact full-caption encoding;
- exact cached encoding (sanity check);
- raw token/row masking or reordering (deliberately crude control);
- contextless token/word/position-aware lookup;
- if available, SAE reconstruction **without edits**;
- targeted SAE features **with** edits;
- direct learned-editor prediction and SAE-based learned-editor prediction.

Evaluate both final images **and** how the denoiser predictions vary under the different conditionings at fixed image/noise/timestep. Include multiple timesteps; a method could work near the end of denoising and fail early (or vice versa).

Success at this stage justifies training experiments; it does not replace them.

### Stage 3 — Offline teacher-supervised editor

Generate a controlled teacher dataset of `(original caption, edit description, edited caption, original TE output, edited TE output)` across held-in and held-out contexts. Use **the original TE as teacher**. Compare:

1. A simple global difference vector for common attribute substitutions.
2. A small conditioned residual editor using full original conditioning plus edit specification.
3. A sequence-aware editor that can change positions/output length.
4. An SAE-derived editor versus the identical architecture without an SAE.
5. A small distilled student encoder which processes the entire edited caption as a competing alternative.

Train on some concepts and combinations; withhold others. Test substitutions, additions, removals, reordering, negation, multi-entity attribute binding, cross-source aliases, rare vocabulary and NL paraphrases **as separate generalization tasks**.

Consider a loss that combines representation matching with *teacher-denoiser response matching* over sampled noisy latents/timesteps. Pure embedding MSE is not sufficient if apparently small differences are semantically important to the denoiser. Avoid automatically forcing the teacher and student to be identical when an alternative conditioning contract is deliberately being learned.

### Stage 4 — Actual diffusion-training impact

Only candidates that pass Stage 2/3 should be used to fine-tune comparable image models. Match denoiser evaluations and record end-to-end compute/memory costs. Evaluate each resulting checkpoint using both its intended conditioning mechanism and the original full TE to determine compatibility and representation drift.

### Measurement matrix

| Axis | Examples / measurements |
| --- | --- |
| Compute | TE+adapter time, denoiser time, total step time, throughput, energy if available, transfer costs |
| Cache | exact hit rate, unique caption count, disk/RAM/VRAM use, read bandwidth, fallback frequency |
| Representation | norm/distribution shifts, reconstruction error, hidden-state similarity, denoiser-output differences |
| Semantics | counts, attribute binding, entity-reference correctness, spatial relations, negation, tag order sensitivity |
| Long tail | rare artists, obscure characters, infrequent tags, aliases and source-specific meanings |
| Generalization | held-out combinations, held-out phrasing, unseen edit sequences, paraphrases, novel relation structures |
| Training | loss curves, conditioning stability, memorization/near-copying, prompt robustness, sampled image quality |

Prefer targeted human inspection and scene-specific tests alongside automatic VLM metrics. A fluent VLM judge can miss subtle tag or artist distinctions.

## 10. Trainer architecture implications (research-level, not an API design)

Treat these as separate logical responsibilities; the code's final ownership/API remains an independent architecture decision:

```text
source annotations / provenance
          |
    caption constructor  <-- source caption / aliases / NL / omission policy
          |
  exact caption + metadata
          |
  conditioning provider  <-- original TE | exact cache | distilled TE
          |                  position-aware lookup | learned editor (experimental)
          v
 complete conditioning contract (sequence / pooled / masks / other model inputs)
          |
    model-specific adapter if applicable
          |
        denoiser
```

Operational boundaries the engine or preparation layer must be able to observe:

- variant selection and **actual semantic exposure** per image;
- exact-vs-approximate conditioning source and cache compatibility/provenance;
- ownership of trainable TE/adapter state and invalidation after weight updates;
- whether a paired-caption encounter represents **one image group but multiple denoiser evaluations**;
- normalization and optimizer-step boundaries for grouped losses;
- reproducible randomness for caption realization, noise, augmentation and fallback selection;
- checkpoint/restart behavior for sampler state, cache version, teacher/student revisions and experimental mixing ratios.

Avoid designing a training mechanism around an assumed fixed cached dataset. But also avoid building a generic semantic editor into the normal training path until experiments show what it can actually do.

## 11. Open questions (explicitly unresolved)

- How much of observed tag-order sensitivity comes from the tokenizer/TE, adapter, or denoiser, for each target model?
- Are BoPTW-style position-aware word caches sufficiently faithful for **training**, especially on rare multi-token booru/artist vocabulary?
- Are phrase-level reusable units materially better than individual word+position units for difficult relationships, and does that offset their cache growth?
- Does a student TE trained against image-model behavior preserve more useful conditioning than an editor trained for TE-output reconstruction?
- Can a sequence-aware editor learn *compositional* changes (including previously unseen interactions) from a tractable teacher dataset, or will it just memorize edit templates?
- Is an SAE actually useful for editing, beyond being a regularizer or interpretability tool? What does it add over direct learned residuals?
- Is any approximate conditioning method beneficial as a *training augmentation* even when it fails to perfectly reconstruct freshly encoded text?
- When an image has overlapping but non-equivalent source captions, what sampling policy preserves the supervision for the rarest true concepts?
- At realistic dataset and hardware scales, does TE compute dominate enough to justify novel modeling, or are denoiser forwards the binding cost?

## 12. Decision checkpoints / minimal near-term work

- [ ] **Keep** the original, dynamically executable conditioning path as the correctness reference.
- [ ] **Profile** target architectures and data distributions; do not assume the TE is the bottleneck.
- [ ] **Prototype** per-image balanced caption sampling with exact conditioning and concept-exposure logging.
- [ ] **Build a small evaluation set** with aliases, true non-equivalences, tag permutations, rare concepts and hard NL binding cases.
- [ ] **Reproduce** full-TE versus position-aware contextless lookup at inference for one DiT and, as a counterexample, one U-Net if feasible.
- [ ] **Test SAE reconstruction separately** using the companion note's protocol before attempting edits.
- [ ] **Collect teacher pairs** and compare simple embedding deltas, direct contextual editing and SAE-based editing.
- [ ] **Advance to diffusion fine-tuning only after inference diagnostics** and under matched training budgets.

---

## References and links

**Caption construction, varied supervision, copying**

- **[R1]** Brack et al. (ICCV Workshops 2025), *How to Train your Text-to-Image Model: Evaluating Design Choices for Synthetic Training Captions.* Randomized caption lengths and trade-offs with dense captions. https://openaccess.thecvf.com/content/ICCV2025W/CDEL/html/Brack_How_to_Train_your_Text-to-Image_Model_Evaluating_Design_Choices_for_ICCVW_2025_paper.html
- **[R2]** Segalis et al. (2023), *A Picture is Worth a Thousand Words: Principled Recaptioning Improves Image Generation* (RECAP). https://arxiv.org/abs/2310.16656
- **[R3]** Somepalli et al. (NeurIPS 2023), *Understanding and Mitigating Copying in Diffusion Models.* https://papers.neurips.cc/paper_files/paper/2023/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html
- **[R4]** CircleStone Labs, official *Anima* model card; tag ordering conventions, tag dropout, and mixed tag/NL prompting. https://huggingface.co/circlestone-labs/Anima

**Cache mechanics and model conditioning**

- **[R5]** vLLM documentation, *Automatic Prefix Caching.* Shared-prefix KV reuse for causal LLM inference. https://docs.vllm.ai/en/latest/design/prefix_caching/
- **[R6]** Spingarn et al. (June 2026), *Text-to-Image Models Need Less from Text Encoders Than You Think.* BoT, BoW, BoPTW, DiT results, U-Net failures, and rare-word evaluation limitations. https://arxiv.org/html/2606.03715 — Project: https://nsping13.github.io/contextless-TTI/
- **[R7]** Kaplan et al. (ACL 2026), *Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models.* https://aclanthology.org/2026.acl-long.1575/
- **[R8]** Wang et al. (CVPR 2025), *Scaling Down Text Encoders of Text-to-Image Diffusion Models.* https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Scaling_Down_Text_Encoders_of_Text-to-Image_Diffusion_Models_CVPR_2025_paper.html
- **[R9]** Hu et al. (2024), *ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment.* https://arxiv.org/abs/2403.05135 — Official code: https://github.com/TencentQQGYLab/ELLA
- **Implementation reference:** Hugging Face Diffusers, SDXL training documentation (including precomputed embedding flow and unloading the TE); useful as a baseline pattern, not a recommendation to constrain captions. https://huggingface.co/docs/diffusers/training/sdxl

**Embedding interpretation/editing; expand alongside the older SAE note**

- **[R10]** Bhalla et al. (2024), *Interpreting CLIP with Sparse Linear Concept Embeddings* (SpLiCE). https://arxiv.org/abs/2402.10376
- **[R11]** Kamenetsky et al. (2025), *SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder.* https://arxiv.org/abs/2510.05081 — Official code: https://github.com/ronen94/SAEdit
- **[R12]** Leask et al. (ICLR 2025), *Sparse Autoencoders Do Not Find Canonical Units of Analysis.* https://proceedings.iclr.cc/paper_files/paper/2025/hash/84ca3f2d9d9bfca13f69b48ea63eb4a5-Abstract-Conference.html

### Relation to previous research

- Companion user research: **`sae_embedding_aug.md`** — SAE feature dropout, reweighting, noise, feature mixing, reconstruction-first tests, per-token versus whole-sequence SAEs, and the earlier idea of a learned reorder operator.
- This note is broader. It **does not** replace that SAE document. It adds caption scheduling and semantics, exact/dynamic TE cost pathways, position-aware contextless lookup, distillation, the proposed teacher-supervised learned editor, and experiment/architecture boundaries.

*Status key: published findings above are attributed to their tested model and evaluation setting; the proposed editor, hybrid composition backend, and training-use of contextless representations are research hypotheses requiring controlled validation.*
