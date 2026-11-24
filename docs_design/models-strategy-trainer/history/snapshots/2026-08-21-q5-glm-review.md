# Architecture critique: model-strategy-trainer design docs (read-only)

**Method.** I read all three documents in full (`docs_design/models-strategy-trainer/strategy_system_direction.md`, `strategy_system_inventory.md`, `notes.md`), then spot-verified material claims against `library/models/components.py`, `library/strategies/base/contracts.py`, `library/strategies/base/features.py`, `library/training/runners/trainer.py`, `library/training/phases/{model_prep,training_loop}.py`, `library/training/modes/{base,adapter_mode,finetune_mode}.py`, family strategy folders, `library/optimization/{grouping,types}.py`, `library/adapters/runtime/{context,targets}.py`, and `train.py`/`library/strategies/factory.py`. No files were modified.

## 1. Evidence verification — the docs' code claims hold

I independently verified the load-bearing factual claims; none failed:

- **25-method aggregate, 19 abstract, 6 raising defaults** (Inventory M2): exact count confirmed in `contracts.py` — defaults are `load_denoiser_lazily`, `get_token_cache_encoder_names`, `build_te_cache_model_bundle`, `save_model_checkpoint`, `prepare_text_encoder_grad_ckpt_workaround`, `prepare_text_encoder_fp8`. No family overrides `load_denoiser_lazily` (grep), yet `model_prep.py:41-46` calls it whenever `trainer.denoiser is None` — the "contract gap via runtime-path assumption" claim is correct.
- **Backward path**: `training_loop.py:526-532` backpropagates `modifier_output.loss` (from `per_sample_loss`), and `batch_loss.loss` is only accumulated at line 570. The direction doc's "the current batch result is not the final optimization loss" section is accurate.
- **Whole-Trainer mutation**: `adapter_mode.py:243-290` replaces denoiser/TEs/adapter/optimizer/scheduler and sets `_grad_sync_handle`/`_primary_trainable`; `finetune_mode.py:84-158` similarly. `modes/base.py` declares exactly 15 methods + `checkpoint_artifact_role` (M6's count is exact).
- **EDM2 correction** (M9): confirmed. Active step/epoch/final flows call `loss_modifier.save_sidecar` directly (`training_loop.py:104, 289`; `trainer.py:926-954`); the `_edm2_loss_weights` branch in `finetune_mode.py:378-386` has no active production caller I could find. The doc's self-correction is honest and correct.
- **Persistence diversity**: `sdxl/checkpointing.py:49-121` unwraps denoiser + two TEs + VAE; `sd3/checkpointing.py:70-120` unwraps up to five components and returns paths, not a typed result; adapter export (`adapters/shared/state_io.py`, `LoadedAdapterRuntime.merge_into` at `runtime/context.py:81-93`) assembles capability-owned state. The Q4 pressure table faithfully describes real divergence.
- **Role-based projection writeback**: `trainer.py:1004-1034` setters rewrite `loaded_components` by role; `LoadedModelComponent.module` (components.py:42-69) is the single field that accelerator preparation overwrites — the "one module field conflates logical and prepared binding" diagnosis is correct.

Two claims I could not fully verify and mark as **plausible but unconfirmed**: (a) that the SD3 loader can return a `None` denoiser in practice (block-swap path — the seam is declared and `trainer.py:423` comments it, but I did not trace the loader branch); (b) M9's "only the main resource currently receives the supplied metadata" in SD3 `save_models`.

## 2. Coherence across the three documents

**Strong conclusion: the documents are mutually consistent and internally coherent.** The settled-semantics sections in the direction doc ("Settled Binding Semantics", "Settled Arrangement-Change Semantics", "Settled Artifact-Persistence Semantics") match Milestones 7-9 nearly verbatim with no drift I could find, and `notes.md` checkpoints track them accurately. Three structural issues remain:

1. **Duplicated settled statements.** Direction doc §"Settled *" sections and Inventory M7-9 are two copies of the same semantics. They are in sync today; before an OpenSpec turns them into requirements, one should become the single normative statement (the other links to it), or drift becomes likely.
2. **Stale vocabulary section.** Direction §"Working Vocabulary" predates Q3-Q5 and defines *contract concern, feature, capability, fulfillment, family integration, training strategy, model component* — but not *participant, relationship, arrangement, execution route, access view, projection, filing, binding authority, revision*, all of which are now load-bearing. Question numbering ("question 3", "question 4", "Question 5") is also used without ever being enumerated anywhere.
3. **A dropped blocker.** The 2026-07-30 checkpoint (notes.md, echoed in direction §"Code Pressure-Test Conclusions") names two blockers: binding state, and the **standard optimization-ownership profile / research extension profile**. Q1-Q4 settlement text resolves the first semantically; the second silently became "future design work" (direction §"Subsequent Design Work: Optimization exchange") and is no longer tracked as a blocker. It is still genuinely open, and nothing in the Q5 milestone touches it. The pre-OpenSpec sequencing narrative ("only Q5 remains") slightly overstates readiness.

## 3. Are settled Q1-Q4 semantics defensible?

**Strong conclusion: yes, with named soft spots.** Each settlement is grounded in real code pressure, and the derivation quality is high:

- **Q1 (logical identity)** — the criterion trio (shared live state → one component; derived-with-declared-refresh → a route; independently evolving → distinct component) is the right test, and the in-place-LoRA observation (object identity unchanged while authoritative execution semantics change — `adapter_mode.py` applies adapters by forward replacement without replacing the host object) is a genuinely well-derived reason revisions must exist. *Soft spot*: "what makes two bindings the *same* participant across replacement" is never given a criterion. Scenario E replaces autoencoder A with B as an identity-preserving `replace authoritative bound state`, but if A and B are semantically different encoders (different latent channels), it is unclear whether that is replacement or retire-and-amend. The taxonomy distinguishes the operations; nothing states when each applies.
- **Q2 (routes)** — minimal and correctly stingy ("training and validation share one route; caller intentions don't create routes"). *Soft spot*: the named-route mechanism has **zero current exercising code** — no family has a second prepared representation today. The hedge ("only when materially different preparation requires it") bounds it, but this is specification ahead of need; the only plausible near-term trigger is `torch.compile` producing a separate sampling artifact.
- **Q3 (arrangement taxonomy)** — the seven operations are clean and each is anchored (merge/fold to `merge_into`, materialization to the deferred-denoiser seam, attachment to `apply_to`). The three state axes and three persistence views are well separated.
- **Q4 (persistence)** — the strongest settlement. Plan-authoritative-for-intent / result-authoritative-for-emitted-facts, product/member/resource levels, and the orthogonal six dimensions all map one-to-one onto the five real products. *Soft spot*: the consistency invariant is only as strong as contributor-declared freshness, and no default freshness model is proposed for strategies that declare nothing.

## 4. Q5 target-first scenarios: sufficient and honest?

**Honest: yes, with two caveats. Sufficient: no — by their own construction they cannot decide the question they are aimed at.**

Honesty caveats:
- Scenario B's "current evidence" (declared-but-unbound SD3 participants) rests partly on a **latent, unimplemented seam**: `load_denoiser_lazily` is declared, called by generic phases, and implemented by nobody. The inventory admits this in M2/M5, but M10 presents deferred loading as pressure evidence without restating that the deferral path has never executed. The scenario is legitimate as a design probe; it is weaker as behavioral evidence.
- Scenario F is explicitly hypothetical and labeled as such — honest.

Insufficiency — this is my main finding on M10:
- The six scenarios were written at a semantic level that **all three candidate physical layouts satisfy by construction**. M10 itself concedes "those may be different object layouts for the same ownership semantics," then says comparing them "against these scenarios" is the remaining Q5 decision. That comparison is underdetermined: the scenarios narrow *scope* (strategy-contract scope; not Trainer fields, not metadata, not mixin attributes, not a shared dict — a useful negative result) but contain **no discriminating criterion** for layout. Deciding Q5 needs new criteria the scenarios don't supply (see §7).
- **Missing scenarios**: (a) **resume/restore** — after `accelerator.load_state()`, wrappers are re-created; how does the authority re-accept restored prepared bindings, and are revisions stable across resume? Neither M10's requirements list nor any scenario covers restore, despite resume being a first-class Trainer feature today. (b) **Rank consistency** — every rank holds its own authority; are identities/revisions per-rank, and must they agree? M9 pushes rank coordination to Trainer infrastructure but the authority-side story is absent. (c) **Transition failure** — requirement 3 says the authority "validates and accepts atomically," but no scenario exercises a rejected or half-failed materialization (Q4 got partial/failure semantics; Q3 transitions did not). (d) **Validation/sampling capability views** are only conceptually covered by Scenario D's "access view"; the validation-traversal relocation out of family strategies (a real behavioral move, per M4's placement table) appears in no scenario. (e) EMA — the canonical test of "independently evolving state → distinct identity" — is named in the semantics but never scenario-tested (Scenario F's teacher is analogous but not identical).

## 5. Hidden assumptions

1. **Pre-execution validation vs. post-load facts.** The contract promises incompatible selections fail "before expensive work," but several compatibility facts (latent channels, tokenizer/encoder pairing) only exist after materialization. The docs acknowledge this only as an open question ("Which compatibility facts are structural and testable now?"); the settled scenario-E semantics quietly assume post-transition re-validation is always possible and cheap.
2. **Opaque participants still prepare correctly.** M10 Scenario A requires Trainer to see "opaque identity, concrete object, requirements, joint-preparation group" — but today's DeepSpeed path keys composite kwargs by family names (`adapter_mode.py:253-259`: `denoiser`, `text_encoder1`…). The generic-authority-to-DeepSpeed-API translation is unaddressed (Scenario D covers the returned composite object, not the request naming).
3. **Identity strings are shared knowledge.** Cross-feature dependency declaration (cache codec depends on `representation.autoencoder`) assumes authors and features reference one stable identity vocabulary; nothing pins how features learn those identities.
4. **Single-process authority.** See rank consistency above.
5. **Scope exclusion vs. live artifact role.** The inventory excludes textual inversion from the active path, yet `sdxl/checkpointing.py:24-28` explicitly supports `artifact_role == "textual_inversion"`. A live artifact role exists whose producing path is outside the evidence base; M9's product table omits it.

## 6. Overengineering assessment

**Strong conclusion: the semantics are not overengineered, but the implied *machinery* has no stated staging, and that is the real overengineering risk.**

- The three-surface contract, the rejection of the required/optional split, and the lifecycle-orthogonality argument are all well-earned and evidence-backed; I would not push back on them.
- Scenario E's "every derived projection declares dependency set and source revision" is a build-system-grade dependency graph. The repository has exactly three real transition sources today (accelerator prepare, RamTorch substitution, adapter attach) and no hot-replacement of encoders at runtime. A staged alternative — v1 coarse invalidation (any accepted transition invalidates all derived projections; recompute is cheap because transitions are rare and lifecycle-bounded) with typed dependency sets as a later refinement — satisfies every current scenario and defers the graph. The docs write the requirements as if full precision were needed from day one; the OpenSpec should scope this explicitly.
- Similarly, Q4's six orthogonal dimensions + four stages + three containment levels are the right *semantics* but more than the first migration needs; the OpenSpec should mark which parts are v1-verifiable versus declared-for-completeness.
- Named execution routes (Q2) — see §3; keep the semantics, defer the mechanism until a second route actually exists.

## 7. Are the three physical layouts meaningfully different?

**Strong conclusion: semantically equivalent (by construction — the scenarios don't discriminate them), but practically different along at least five axes the docs don't name.** Treating the choice as a near-tie to be settled "against these scenarios" understates it:

| Axis | (a) contained in strategy | (b) paired Trainer-facing contract object | (c) separate contract-owned authority |
|---|---|---|---|
| State/behavior mixing | **Re-creates the diagnosed disease**: M4's complaint is that strategy aggregates mix behavior with run-derived state; containing the authority puts canonical mutable state back into the authored object | Good, if the paired object is the *only* thing Trainer receives | Best separation |
| One-writer enforceability | Weakest — strategy methods can mutate self arbitrarily | Medium — enforceable at the paired boundary | Strongest — narrow writer protocol |
| Risk echo from project history | The `PreparedTrainingProgram`/`ModelRuntime` rejections ("another synonym… same back-and-forth") — the docs must state *why this paired object differs* | Carries that named risk unless it is the sole Trainer-facing handle | Cleanest, but adds construction/lifetime questions ("established by strategy filing" — by whom, when?) |
| Conformance testability | Tests need real behavior to construct bound state | A paired state object can potentially be validated with stub behavior | Best — filing can be validated without executing strategy behavior |
| Trainer-facing surface | Smallest object count | One narrow handle | One handle + one shared authority interface for capabilities |

Layout (a) has a **named, already-diagnosed** risk; layout (b) has a **named, already-rejected** predecessor unless differentiated; layout (c) has precedent in the repo's own trajectory (`ObjectiveRuntime`, `OptimizationPlan` are typed run-scoped values) but the docs never connect that precedent. This table, not the six scenarios, is what should decide Q5.

## 8. Terminology problems

1. **component vs participant** — the most pervasive issue. Q1/Q2 sections say "logical component"; Q3/Q5 sections say "participant" (M8 even says "a logical component represents one independently addressable semantic *participant*", implying species/genus, then uses them interchangeably thereafter; M10 uses "participant" exclusively). Pick one or define the relation once.
2. **Two senses of bind/bound** — "the **binding** authority belongs to the **bound** strategy contract scope" (M10) uses "binding" (state-holding) and "bound" (lifecycle state) in one clause.
3. **vae vs autoencoder** — the user flagged this in notes.md ("decisions (vae vs autoencoder language…)"); the docs still use both: current code/roles say `vae` (`models/sd/__init__.py`, `components.py`), M10 scenario identities say `representation.autoencoder`. The CLIP section settles component-vs-feature ownership for CLIP; the VAE naming question is unsettled while scenario examples are already fossilizing `autoencoder`.
4. **"filing/filed"** as noun-jargon is defined by usage and mostly consistent, but "a named compatible filing" (M10) and "contract filing" stretch it; the glossary gap (§2.2) makes this worse.
5. **"arrangement revision R"** (Scenario B) introduces revision terminology never defined alongside the taxonomy.

## 9. What to discuss or rewrite before an OpenSpec

Ordered by leverage:

1. **Decide Q5 with discriminating criteria, not the scenarios.** Write the axes table (§7), add the failure-mode history (aggregate state-mixing; `ModelRuntime`/`PreparedTrainingProgram` rejections and *why a paired object would differ*), and answer: can conformance tests construct a validated filing with stub behavior? Can capabilities get narrow views without going through strategy behavior? Who constructs the authority and when? This is the only remaining Q5 work the scenarios can't do.
2. **Add the missing scenarios**: resume/restore of bound state; rank consistency; transition rejection/partial failure; validation/sampling capability views; EMA as the Q1 independently-evolving test.
3. **Stage the invalidation machinery**: v1 coarse invalidation, typed dependency sets later. Write this into the direction doc so the OpenSpec doesn't inherit a revision graph as a day-one requirement.
4. **Terminology pass**: merge component/participant; extend "Working Vocabulary" with the Q3-Q5 terms; enumerate Q1-Q5 once; decide vae/autoencoder naming before scenario identities harden.
5. **De-duplicate the settled statements** (direction §"Settled *" vs Inventory M7-9) — one normative copy before requirements are extracted.
6. **Re-flag the optimization-ownership blocker** explicitly in the sequencing, so the "only Q5 remains" narrative doesn't drop it; it is pre-OpenSpec work per the direction's own "four exchanges" plan.
7. **Note the textual-inversion artifact role** in M9's product table or explicitly exclude it with rationale.
8. Restate in M10 that Scenario B's deferral anchor is an unimplemented seam (one sentence; the inventory already concedes it elsewhere).

**Strong conclusions**: the direction (contract → authored strategy → trainer; three consumer-defined surfaces; TrainingMode dissolution; Q1-Q4 semantics) is defensible and well-evidenced; the docs are mutually consistent; the code claims I verified all held; the main Q5 gap is that the scenarios narrow scope but cannot discriminate the three layouts; the main overengineering risk is unstated staging of dependency-invalidation precision, not the semantics themselves.

**Uncertainties**: SD3 `None`-denoiser reachability (unverified branch); SD3 sidecar-metadata asymmetry detail; exact SD 23/25 concrete-method count (not recounted); whether a second execution route will ever be needed (my overengineering read could be wrong if `torch.compile` sampling is planned near-term).