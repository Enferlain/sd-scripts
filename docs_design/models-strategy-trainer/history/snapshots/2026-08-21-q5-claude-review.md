# Architecture Critique: Model-Strategy-Trainer Design Documents

## Scope

Read-only assessment of three design documents:
- [strategy_system_direction.md](file:///home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md) ("Direction")
- [strategy_system_inventory.md](file:///home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_inventory.md) ("Inventory")
- [notes.md](file:///home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/notes.md) ("Notes")

Against current source under `library/models`, `library/strategies`, `library/training`, `library/adapters`, and `library/optimization`.

---

## 1. Overall Coherence Assessment

### Strong conclusions

The three documents form an unusually rigorous design record. The central principle (Direction L112–118) — Contract defines acceptance, Strategy defines what/how, Trainer executes mechanism — is **coherent and internally consistent** across all three documents. Every subsequent decision flows from it. There are no structural contradictions between the documents.

The historical progression is honest: the documents record rejected routes, explain why they were rejected, and maintain those rejections consistently. The five-milestone inventory is evidence-grounded rather than aspirational. The pressure-test methodology (Milestone 10: target-first scenarios before choosing Python types) is architecturally sound.

### Uncertainties

The three documents have an **escalating abstraction gradient**. Direction establishes principles and vocabulary. Inventory applies them to current code. Notes records leanings and user intent. But as the sequence progresses from Q1 through Q5, the semantic precision increases while concreteness decreases. By Milestone 10, the scenarios are compelling but are arguing about ownership boundaries without a single concrete type signature, return type, or error type. This is deliberate ("derive exchanges before naming methods") but carries a risk documented below under §5.

---

## 2. Defensibility of Q1–Q4

### Q1: Logical Participant Identity — **Strong**

The claim that a logical component identity is stable, strategy-scoped, and distinct from Python object identity (Direction L1424–1430, Inventory M7 L1300–1317) is well-defended. The current `LoadedModelComponent.key` ([components.py:46](file:///home/imi/Projects/sd-scripts/library/models/components.py#L46)) already provides a stable key, and the documents correctly identify that its single `module` field conflates logical and prepared bindings (Direction L916–930). The Trainer's `vae`/`text_encoders`/`denoiser` property setters ([trainer.py:1009–1034](file:///home/imi/Projects/sd-scripts/library/training/runners/trainer.py#L1009-L1034)) replace modules by role via `update_loaded_component_modules_by_role`, confirming the pressure.

The one-route-per-logical-component default with named extension routes is defensible. The EMA/teacher/student → distinct identity rule (Direction L1469–1470) is correct and important.

**Minor concern:** The documents never address _when_ a logical identity is assigned. Is it declared in strategy authoring Python code? In config? At load time? The current `LoadedModelComponentSpec` is declared in model-package metadata (`LOADED_MODEL_COMPONENT_SPECS`), which is model-layer owned. The direction wants it strategy-scoped. This transfer of declaration authority hasn't been explicitly discussed.

### Q2: Execution Routes — **Defensible with a gap**

The claim that training and validation sharing one prepared wrapper do not justify separate routes (Direction L1449) is correct and well-argued. The current evidence is strong: validation's `process_val_batch()` never calls `process_batch()` (Inventory M3 L554), meaning the "is_train" parameter on `process_batch` is architecturally dishonest.

**Gap:** The documents don't discuss **compiled execution routes**. `torch.compile()` may produce different compiled traces for training vs. inference. If compilation is route-creating, the one-route default needs explicit handling. This isn't addressed in any scenario.

### Q3: Arrangement Transitions — **Strong**

The seven-operation taxonomy (Direction L1488–1496) is the most concrete and well-pressure-tested result. The LoRA in-place injection analysis (Inventory M8 L1335–1340) correctly identifies that Python object identity is insufficient for detecting semantic execution changes. The VeRA shared-bank analysis (Inventory M8 L1339) correctly proves that adapter identity ≠ Python container identity.

The separation of participant lifecycle / optimization status / operational relationship lifecycle (Direction L1530–1538) is clean and honest. The rule that "materialization is not addition" (Direction L1508) is crucial and well-argued.

**Strength:** This is the most implementable section. The taxonomy maps cleanly onto current code operations even though no API exists yet.

### Q4: Artifact Persistence — **Defensible but elaborate**

The four-stage model (declaration → request → plan → result) at Direction L1587–1601 is logically complete. The three-level containment (product / member / resource) at Direction L1610–1619 is correct — current SDXL/SD3 saves demonstrably produce multi-member products.

The six orthogonal dimensions (coverage, dependencies, transformations, representation, packaging, consistency) at Direction L1627–1645 are _individually_ correct but _collectively_ may be overspecified for pre-OpenSpec work. See §4.

---

## 3. Q5 Sufficiency and Honesty Assessment

### Scenario quality

The six scenarios (Inventory M10 L1706–1981) are well-chosen and honestly stress-tested:

| Scenario | Pressure quality | Concern |
| --- | --- | --- |
| A: SDXL fine-tune | Good baseline, uses real component names | No real controversy, mostly validates the obvious |
| B: Deferred SD3 | Valuable — tests binding-state progression | Correctly identifies that the current `None` component is semantically ambiguous |
| C: Adapter attachment | Critical — tests in-place semantic change | This is the strongest scenario; object-identity failure is proven |
| D: Distributed prep + persistence | Tests multi-view access | Correctly separates execution, original, artifact, and backend views |
| E: Replacement/invalidation | Tests dependency invalidation | This is the most architecturally important and least grounded scenario |
| F: Compound research | Tests beyond current families | Correctly pressures one-denoiser/one-trainable assumptions |

### Honest gaps in Q5

The documents are honest about what Q5 doesn't resolve (Inventory M10 L2037–2046): the physical layout remains open among three options (authority embedded in strategy, paired Trainer-facing object, or separate contract-owned object).

**However, there is a material honesty gap.** The ten requirements derived from all scenarios (Inventory M10 L1988–2013) are _semantic requirements_, but requirement #6 ("atomic snapshots and result application") and #7 ("dependency-aware freshness") imply a **revision-tracking runtime** that is significantly more machinery than anything in the current codebase. The documents don't quantify or acknowledge the implementation weight of these requirements. A revision-tracking, dependency-aware, atomically-snapshotting binding authority is a non-trivial runtime data structure. The closest current analogy in the codebase is `LoadedModelComponent` at [components.py:42–69](file:///home/imi/Projects/sd-scripts/library/models/components.py#L42-L69), which is a frozen dataclass with no revision, no dependency tracking, and no atomicity.

### Missing scenario: error recovery

No scenario models what happens when binding transitions _fail partway_. If adapter materialization succeeds but attachment fails, what is the binding authority's state? The requirements mention "partial/failure status" for persistence results (Direction L1600) but not for binding transitions themselves.

### Missing scenario: hot-reload / parameter surgery

No scenario models training-time parameter surgery (e.g., unfreezing a previously frozen encoder mid-training, progressive training). The optimization-status axis (selected/unselected, trainable/frozen) is declared (Direction L1530–1533) but never exercised in a scenario where it changes during a run.

---

## 4. Hidden Assumptions, Missing Scenarios, and Overengineering

### Hidden assumptions

1. **Single strategy per run.** Every document assumes exactly one strategy per training run. This is likely correct for the foreseeable future, but it's never stated as an explicit constraint. A multi-stage training run (e.g., first train the adapter, then fold and train the base) would either need strategy replacement or two runs. The binding authority's lifetime is "strategy-scoped" — what happens if a strategy is replaced mid-run?

2. **Synchronous binding transitions.** All scenarios assume binding transitions are synchronous and complete before the next lifecycle phase. Asynchronous model loading (streaming weights from remote storage) would break the declared/bound binary. This may be intentionally out of scope but is never stated.

3. **Trainer singleton.** The design assumes one Trainer instance orchestrating everything. Distributed training with model parallelism where different ranks host different components would need either a distributed binding authority or rank-local partial views. The documents mention "coordinate ranks" (Direction L1582) but don't explore this tension.

4. **Current families are representative.** The pressure tests use SD/SDXL/SD3. These are all latent diffusion models with the same broad skeleton. The Scenario F (compound research strategy) is the only non-latent-diffusion scenario, and it's speculative rather than grounded in real code. A real pressure test against, e.g., a GAN, autoregressive model, or consistency model would be more convincing.

### Missing scenarios

- **Multi-adapter:** Two LoRA adapters with overlapping targets. The relationship/effect separation (Direction L1520–1525) handles this conceptually, but no scenario exercises the ordering, priority, or composition semantics.
- **Component sharing across strategies:** If the codebase ever supports multi-strategy runs or strategy switching, shared component state would violate the "one authority" requirement.
- **Cache invalidation on arrangement change:** The documents mention that caches may depend on autoencoder realization (Inventory M10 L1924–1928), but no scenario walks through the actual invalidation and recomputation sequence.

### Terminology concerns

- **"Filing"** is used as a noun (Direction L237–239, L262) to mean "a strategy's particular selection for a contract concern." This is unusual jargon. In legal/bureaucratic English, a filing is a submitted document. The metaphor works but may confuse readers expecting software architecture terms. Consider whether "selection," "provision," or "registration" would be clearer.

- **"Concern"** is overloaded. It means both "a responsibility category the contract knows about" (Direction L498–503) and sometimes "a problem area." The former is a term of art; the latter is colloquial. The documents are usually careful but not always.

- **"Binding authority"** (Notes L294–297) is deliberately abstract, but it has been abstract for two milestones now. At some point, refusing to name it becomes a barrier to progress. The three candidate physical layouts (Inventory M10 L2039–2042) are sufficiently different that they would produce different API shapes, different testing strategies, and different migration paths. Delaying the decision further may not yield additional information.

### Potential overengineering

**Strong concern: the persistence product model.** The six-dimensional persistence description (Direction L1627–1645) with three containment levels, four stages, and explicit transformation ownership is **architecturally correct in the abstract** but may be more machinery than the repository needs in its first iteration. Current persistence is: save weights to a file, optionally convert, embed metadata. The gap between the current simple save and a four-stage artifact product pipeline with resolved plans and typed results is enormous.

> [!WARNING]
> The persistence model in Direction L1587–1667 risks becoming architecture astronautics if it tries to implement all six orthogonal dimensions, three containment levels, and four stages in the first OpenSpec. A staged approach — start with typed plan/result for the existing save operations, then add dimensions as real use cases demand — would be more defensible.

**Moderate concern: dependency-aware invalidation.** Requirement #7 (Inventory M10 L2004–2006) implies that every derived projection declares its input identities and revisions. This is a real requirement (the current code demonstrably has stale-state bugs), but building a general dependency-tracking system before having concrete exchange types is premature. The first implementation should probably invalidate conservatively (prepare everything again after any replacement) and only add fine-grained invalidation when profiling shows it matters.

---

## 5. Do the Three Physical Ownership Layouts Meaningfully Differ?

The three candidate layouts for binding authority (Inventory M10 L2039–2042):

```text
A. physically contained by the authored strategy object after filing
B. paired with strategy behavior in a small Trainer-facing contract object
C. a separate contract-owned object whose lifetime is established by strategy filing
```

### Assessment: they differ meaningfully, but the differences are implementational, not semantic

| Dimension | A (embedded) | B (paired) | C (separate) |
| --- | --- | --- | --- |
| Who can mutate bindings? | Strategy methods (through self) | Either strategy or contract object | Only the authority object |
| How does Trainer access it? | Through the strategy reference | Through its own reference to the paired object | Through its own reference to the authority |
| Testing: can the authority be tested without strategy behavior? | No | Partially | Yes |
| Migration: can it coexist with current code? | Easiest — strategy already exists | Medium — adds one new object | Hardest — needs new lifetime management |
| Encapsulation: can strategy internal features mutate bindings without going through the authority? | Yes (weakness) | Depends on interface | No (strength) |

**The meaningful difference is encapsulation.** Layout A permits the same multiple-inheritance self-mutation that the current architecture suffers from. Layout C enforces that all binding transitions go through one validated protocol. Layout B is Layout C with weaker enforcement.

**Recommendation for the pre-OpenSpec discussion:** Layout C (separate authority) is the strongest from a contract-enforcement perspective, but Layout A (embedded in strategy) is the most migration-compatible. The OpenSpec should pick one and justify the tradeoff rather than leaving it as three equivalent options.

---

## 6. What Should Be Discussed or Rewritten Before OpenSpec

### Must discuss

1. **Concrete exchange types for the four exchanges** (Direction L1714–1781). The documents correctly identify these as the next design step, but no work has started. Without at least sketch types (even as dataclass stubs), the OpenSpec will be specifying behavior against undefined nouns.

2. **Physical layout decision for binding authority.** The three layouts produce different code shapes. The OpenSpec needs one answer.

3. **Scope boundary for the first migration.** The documents simultaneously discuss near-term concerns (SD/SDXL/SD3 migration) and end-game concerns (arbitrary component catalogs, LLM participants, video tensors). The OpenSpec must draw a hard line at what is in scope for the first iteration and what is explicitly deferred.

4. **Validation timing and enforcement.** "Validate before expensive work" is stated repeatedly but never specified. When exactly does validation run? After strategy construction? After loading? Before each phase? The current `config_validation.py` ([config_validation.py:574–643](file:///home/imi/Projects/sd-scripts/library/config/config_validation.py#L574-L643) per Inventory M5 L933) runs at config time, but the design implies fulfillment validation at binding time. These are different and the OpenSpec needs to choose.

5. **`process_batch` decomposition.** The 14-positional-argument `process_batch` call ([training_loop.py:506–520](file:///home/imi/Projects/sd-scripts/library/training/phases/training_loop.py#L506-L520)) is correctly identified as a problem, but no candidate decomposition is proposed. The documents punt this to "derive exchanges" but this is the single most impactful API decision in the migration.

### Should rewrite

1. **Direction §"Settled Artifact-Persistence Semantics" (L1554–1687).** This section is 133 lines long, repeats the same concepts stated in Inventory M9, and introduces terminology (artifact state projection, contributor freshness relationship) that is used nowhere else in the documents. It should be condensed to the core statement (Direction L1678–1687) plus one worked example showing current SDXL save mapped onto the four stages. The full dimensional analysis can move to an appendix or downstream design document.

2. **Notes file chronological ordering.** The notes file reads bottom-to-top chronologically (the most recent Q5 checkpoint is at the top, user excerpts are at the bottom). This is confusing. Either reverse the order or add clear date headers.

3. **Inventory Milestone 10 scenario evidence links.** Several scenarios cite source paths but don't explain _what_ in those paths proves the scenario's pressure claim. For example, Scenario E cites `trainer.py:1010–1034` and `grouping.py:197–228` but doesn't say which specific lines demonstrate the invalidation failure. This makes the scenarios harder to verify independently.

### Should add

1. **A concrete worked example.** None of the documents traces a single complete lifecycle (load → bind → validate → prepare → step → save) through the proposed architecture using concrete types, even as pseudocode. One end-to-end example, even speculative, would make the abstract vocabulary concrete.

2. **Migration risk assessment.** The documents discuss what the target architecture should look like but not the migration path from current to target. How many files change? Can it be staged? What is the compatibility story during migration? These are OpenSpec prerequisites.

3. **The `live_plotter_process` problem** ([training_loop.py:195–201](file:///home/imi/Projects/sd-scripts/library/training/phases/training_loop.py#L195-L201)) is a symptom the documents mention (Inventory M4 L741–744) but don't classify. It's an observation concern that is neither a strategy contract method nor a Trainer mechanism — it's accidental coupling. The design should explicitly state where observation/monitoring hooks belong in the new architecture.

---

## 7. Summary of Findings

### Strong conclusions

- The central principle (Contract → Strategy → Trainer) is coherent and well-argued
- Q1–Q3 are settled at a level sufficient for OpenSpec design
- The seven-operation arrangement taxonomy is the strongest and most implementable result
- The rejection of `TrainingMode` as a parallel authority is correct and well-supported
- The inventory's evidence standard (graph-verified, source-confirmed) is rigorous
- Current `LoadedModelComponent` and `OptimizationPlan` are correctly identified as evolutionary bases

### Uncertainties

- Q4 persistence model may be overspecified for first iteration
- Q5 physical layout decision carries real implementation consequences that are being deferred
- Dependency-aware invalidation (requirement #7) needs scope-bounding
- No concrete exchange types exist yet — the OpenSpec will need them
- Migration risk, timing, and staging are unaddressed
- `torch.compile()` route implications, multi-adapter scenarios, mid-training parameter surgery, and error recovery in binding transitions are unexercised pressures

### Pre-OpenSpec readiness

The design is approximately **70% ready for OpenSpec**. The principle, vocabulary, semantic decisions, and pressure tests are solid. What's missing is: (a) the binding authority layout decision, (b) concrete exchange type sketches for the four exchanges, (c) a scope boundary for iteration 1, and (d) a migration strategy. These are tractable discussions that should happen before the OpenSpec rather than during it.
