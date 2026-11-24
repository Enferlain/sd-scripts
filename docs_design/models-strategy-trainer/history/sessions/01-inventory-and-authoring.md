# Inventory and explicit strategy authoring, 22-29 July 2026

Continuation of [chapter00](00-initial-direction.md): from the follow-up after the first direction document, through the production inventory and explicit-authoring clarification. The Lightning comparison begins the next reading.

Captured 2026-11-05T18:00:27.008450+00:00. 35 visible user/assistant messages. Bodies are preserved, including historical annotations, pasted material and line endings. Tool traffic, reasoning and generated context are excluded.

[Machine-readable provenance](01-inventory-and-authoring.json). [Conversation branch map](branch-map.md).

For each message, the first source is the earliest matching stable identity and content found in the inspected ancestry. The selected record shows the location from which this export was taken. An inherited record timestamp is not automatically its original conversation time.

## Message index

- [001 user: looking at the doc and otherwise, what's left to discuss and look into before doing anything?](#m001)
- [002 assistant: We’re ready to begin the research pass, but not implementation. The remaining work is much narrower than ](#m002)
- [003 user: then let's do the inventory before the other points. the working docs folder for this will be the area do](#m003)
- [004 assistant: I’ll start the model–strategy–trainer inventory and treat `docs_design/models-strategy-trainer/` as the d](#m004)
- [005 assistant: The first milestone will be a factual map, not a redesign: current entrypoints and object construction, t](#m005)
- [006 assistant: Milestone 1 confirms the current system is already “one strategy object” only at construction time. At ru](#m006)
- [007 assistant: Milestone 2 found a sharper issue than “the contract is too large”: it conflates two different interfaces](#m007)
- [008 assistant: Milestone 3 shows that the families are not three unrelated training systems. Their core step flow is alr](#m008)
- [009 assistant: The first three inventory milestones are already recorded in the new working area. I’m continuing with th](#m009)
- [010 assistant: The placement pass is exposing a useful distinction: the model loaders themselves currently mix component](#m010)
- [011 assistant: The placement and contract-seam milestones are now written down. The important result is that this is not](#m011)
- [012 assistant: The inventory is complete and the Beads task is closed. Since this is a substantial architecture handoff,](#m012)
- [013 assistant: The inventory is complete and recorded at each milestone in \[strategy_system_inventory.md\](/home/imi/Proj](#m013)
- [014 user: my current stance as we discussed before is that I like the contract idea centralizing strategy (your tra](#m014)
- [015 assistant: Your stance is supported by the inventory. Today the trainer stores VAE/text-encoder/denoiser projections](#m015)
- [016 user: 1 let's try to make the contracts like this:](#m016)
- [017 assistant: That is the right derivation method. We should design a narrow, honest v1 contract from three sources—not](#m017)
- [018 user: the "run" doesn't select things, these are defined in the strategy build the same as today, just under th](#m018)
- [019 assistant: That correction materially changes the framing. “Fulfillment” should mean that a strategy author has expl](#m019)
- [020 user: yes. there is essentially no noob mode on strategy besides "trainer needs something to train chief)" and ](#m020)
- [021 assistant: I understand the distinction: the contract should provide a rigorous construction language, not a safety ](#m021)
- [022 assistant: Yes. The contract should constrain interoperability, not permissible research.](#m022)
- [023 user: basically we didn't change the overall goal, because we are still the ones that put the strategies togeth](#m023)
- [024 assistant: I’m using the OpenSpec exploration workflow to tighten the direction document without turning these archi](#m024)
- [025 assistant: The document already preserves the original architecture at the top, but its contract section still conta](#m025)
- [026 assistant: The correction is architectural, not cosmetic: strategy authors—including us for the maintained family de](#m026)
- [027 assistant: The direction now says the complete strategy is authored in Python, validated against the contract, and t](#m027)
- [028 assistant: Updated \[strategy_system_direction.md\](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/](#m028)
- [029 user: "onclusion: treat pixel/latent behavior as available feature implementations](#m029)
- [030 assistant: Mostly, but with one important distinction:](#m030)
- [031 user: nice. what are the next questions to decide/clarify?](#m031)
- [032 assistant: The next central question is:](#m032)
- [033 user: well this should be dictated by the trainer no? which also dictates the contract. idk if this means strat](#m033)
- [034 assistant: I’m staying in the OpenSpec exploration stance here because this is the contract boundary decision, not i](#m034)
- [035 assistant: Yes. The trainer—or more precisely, the shared training pipeline we want the trainer to embody—dictates t](#m035)

<a id="m001"></a>
## 001 User - 2026-07-22T20:12:00.324Z

Source: [019f86d4-43a2-73a2-a37a-ba7c5a26a4f5, raw line 11422](/home/imi/.codex/archived_sessions/rollout-2026-07-21T22-38-05-019f86d4-43a2-73a2-a37a-ba7c5a26a4f5.jsonl:11422).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11422, 2026-07-29T02:06:14.956Z.

looking at the doc and otherwise, what's left to discuss and look into before doing anything?


<a id="m002"></a>
## 002 Assistant (final_answer) - 2026-07-22T20:12:42.396Z

Source: [019f86d4-43a2-73a2-a37a-ba7c5a26a4f5, raw line 11427](/home/imi/.codex/archived_sessions/rollout-2026-07-21T22-38-05-019f86d4-43a2-73a2-a37a-ba7c5a26a4f5.jsonl:11427).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11427, 2026-07-29T02:06:14.957Z.

We’re ready to begin the research pass, but not implementation. The remaining work is much narrower than the discussion we just finished.

## Must resolve before implementation

1. **Inventory the current contract**

For every method in `base/contracts.py`, record:

- Who calls it.
- Whether the caller is trainer, mode, objective, or another strategy facet.
- What state it consumes and mutates.
- Whether it represents a contract concern, reusable feature, family-local helper, or misplaced behavior.
- How SD, SDXL, and SD3 actually fulfill it.

This will expose which current abstractions are real and which merely reflect historical extraction.

2. **Define the minimum training-contract vocabulary**

We need to identify the concerns that every valid run must file, without encoding SD anatomy. Likely candidates include:

```text
model/component loading
input representation
conditioning
prediction
training subjects
objective interaction
step result
preparation
persistence
```

Then distinguish:

- Exactly-one choices.
- Composable choices.
- Requirements activated by configuration or another selection.
- Behaviors that are merely internal implementation details.

3. **Settle the final participant relationship**

The largest conceptual question remains:

```text
family integration + features + mode + objective
```

Do these become one fulfilled training strategy before reaching the trainer, or remain separate objects jointly governed by one contract?

We do not need the final class structure yet, but we need a clear ownership rule so the trainer does not become the place that finishes model-family assembly accidentally.

4. **Resolve loaded-state ownership**

We need to map the lifecycle:

```text
unloaded
  → loaded components
  → training-mode modifications
  → accelerator wrapping/replacement
  → prepared execution state
  → saving/finalization
```

For each transition:

- Who owns the authoritative component collection?
- Who may replace component modules?
- Where do family loading facts such as SDXL checkpoint state live?
- How do modes select trainables without receiving the entire trainer?
- How does metadata observe this state without owning it?

The bound-strategy direction is promising, but it still needs to survive these cases.

5. **Define the minimum feature compatibility model**

Not a general dependency resolver—just enough vocabulary to validate current and immediately foreseeable compositions.

For example:

```text
LatentDiffusion
  requires: autoencoder encoding
  provides: latent clean state

SDXL prediction
  requires: SDXL-compatible latent and conditioning states

Distillation
  requires: teacher prediction + student prediction
```

We need to determine which compatibility facts are meaningfully typed and testable now, rather than relying on names or arbitrary tags.

6. **Write conformance scenarios before choosing code shapes**

The contract should be tested on paper against:

- Current SD, SDXL, and SD3 adapter/fine-tune runs.
- Cache-on and cache-off paths.
- Validation, sampling, and saving selections.
- Pixel diffusion.
- Alternate autoencoder/representation.
- Base model plus adapter training.
- Newly attached trainable components.
- Teacher/student distillation.
- Video/audio-shaped state.
- Integrated or external LLM components.

These pressure tests determine whether the vocabulary is useful. They do not require implementing all those arrangements.

## Important placement questions during the inventory

The research should specifically revisit:

- Whether model loading coordination belongs to models, strategy, or trainer infrastructure.
- Whether CLIP FP8/checkpointing quirks are component mechanics or training behavior.
- Whether `process_batch()` remains a meaningful public boundary.
- How training and validation share prediction/target behavior.
- What objective/loss work should leave family diffusion files.
- Whether caching codecs are strategy features while cache persistence remains data-owned.
- Whether sampling and checkpointing are contract filings, family integrations, or both.
- Which existing mixins are reusable features versus merely internal file organization.

## What does not need settling yet

These should stay deferred:

- Strategy graphs.
- Arbitrary user-configured component assembly.
- A general plugin/dependency-resolution framework.
- The final decorator or registration mechanism.
- Hydra target selection for features.
- Whether CLIP should be vendored.
- Formal contract version numbers.
- Exact package names beyond the current `features/` leaning.
- Supporting every future model category.

Those choices should follow the contract inventory rather than lead it.

## Project-tracking decision

Before implementation, we should also separate this from the active model-metadata change.

The clean relationship is probably:

```text
model/strategy contract work
        ↓ establishes stable model boundaries
model metadata continuation
        ↓ records identity, structure, provenance, and queries
```

This architecture work deserves its own Beads issue and OpenSpec rather than being folded into `expand-model-metadata-identity-and-structure`. The metadata change should record the dependency where relevant, without absorbing the strategy redesign.

## Practical next sequence

Tomorrow’s productive sequence would be:

1. Build the code-grounded contract/caller inventory.
2. Turn it into the contract-concern and ownership matrix.
3. Pressure-test that matrix against current and compound run scenarios.
4. Resolve the trainer/strategy state boundary.
5. Decide the bounded scope of the first implementation.
6. Write the governing OpenSpec and create/link the Beads work.
7. Only then begin code changes.

So we are no longer missing an overall direction. What remains is proving that direction against the actual code and converting the vocabulary into enforceable boundaries.

> Compaction boundary at selected raw line 11434 (2026-07-29T02:06:14.960Z); generated summary omitted.

<a id="m003"></a>
## 003 User - 2026-07-29T00:15:17.244Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11447](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11447).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11447, 2026-07-29T02:06:15.060Z.

then let's do the inventory before the other points. the working docs folder for this will be the area docs_design/models-strategy-trainer from now on

record the information between exploration milestones to avoid losing context from hitting context limit compaction


<a id="m004"></a>
## 004 Assistant (commentary) - 2026-07-29T00:15:22.685Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11451](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11451).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11451, 2026-07-29T02:06:15.061Z.

I’ll start the model–strategy–trainer inventory and treat `docs_design/models-strategy-trainer/` as the durable working area. I’m using the codebase-memory exploration workflow because this inventory depends on real call paths and ownership, and I’ll write findings at each meaningful milestone so compaction cannot erase the evidence or conclusions. I’ll work locally without subagents.

<a id="m005"></a>
## 005 Assistant (commentary) - 2026-07-29T00:15:30.938Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11457](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11457).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11457, 2026-07-29T02:06:15.062Z.

The first milestone will be a factual map, not a redesign: current entrypoints and object construction, trainer-facing calls, strategy-internal collaboration, state ownership, and the actual `models/` boundary. I’ll record that map before moving to contract evaluation, so later conclusions remain traceable to current code.

<a id="m006"></a>
## 006 Assistant (commentary) - 2026-07-29T00:17:02.053Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11531](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11531).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11531, 2026-07-29T02:06:15.072Z.

Milestone 1 confirms the current system is already “one strategy object” only at construction time. At runtime the trainer is the real integration hub: it owns the loaded-component tuple and projections, objective runtime, modes, caches, dtypes, accelerator state, optimization state, and lifecycle coordinates, then repeatedly hands those pieces back to strategy methods. I’m recording that distinction now before auditing the individual contracts.

<a id="m007"></a>
## 007 Assistant (commentary) - 2026-07-29T00:19:10.175Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11598](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11598).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11598, 2026-07-29T02:06:15.081Z.

Milestone 2 found a sharper issue than “the contract is too large”: it conflates two different interfaces. Twenty methods are called by trainer phases or modes, while five exist only so family mixins can find each other through the aggregate object. Six methods are non-abstract but still only raise `NotImplementedError`, so the type can instantiate without proving those conditionally required behaviors exist. That is nominal composition, not fulfillment validation.

<a id="m008"></a>
## 008 Assistant (commentary) - 2026-07-29T00:20:37.856Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11627](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11627).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11627, 2026-07-29T02:06:15.087Z.

Milestone 3 shows that the families are not three unrelated training systems. Their core step flow is already nearly the same: obtain a representation, resolve conditioning, ask the objective for corrupted input/target semantics, invoke the predictor, compute and shape per-sample loss. The meaningful variations are payload types, objective compatibility, model-forward arguments, VAE transforms, and persistence/sampling codecs. Validation duplicates the same skeleton in all three families instead of consuming one shared execution seam.

> Compaction boundary at selected raw line 11673 (2026-07-29T02:06:15.094Z); generated summary omitted.

<a id="m009"></a>
## 009 Assistant (commentary) - 2026-07-29T00:24:32.385Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11680](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11680).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11680, 2026-07-29T02:06:15.096Z.

The first three inventory milestones are already recorded in the new working area. I’m continuing with the ownership/placement audit now, then I’ll record the contract and compatibility gaps as the final milestone before we discuss any redesign.

<a id="m010"></a>
## 010 Assistant (commentary) - 2026-07-29T00:25:36.797Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11749](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11749).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11749, 2026-07-29T02:06:15.105Z.

The placement pass is exposing a useful distinction: the model loaders themselves currently mix component materialization with distributed and training-runtime policy, while strategy loading adds another policy layer on top. I’m recording this as a split responsibility—not as a verdict that either entire file should move.

<a id="m011"></a>
## 011 Assistant (commentary) - 2026-07-29T00:33:26.455Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11862](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11862).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11862, 2026-07-29T02:06:15.121Z.

The placement and contract-seam milestones are now written down. The important result is that this is not a “move strategies into models” cleanup: the current problem is mixed responsibilities and a missing typed fulfillment/state boundary, with several specific misplaced helpers on both sides. I’m doing a consistency pass now and recording the completed inventory in Beads.

<a id="m012"></a>
## 012 Assistant (commentary) - 2026-07-29T00:34:28.807Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11901](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11901).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11901, 2026-07-29T02:06:15.125Z.

The inventory is complete and the Beads task is closed. Since this is a substantial architecture handoff, I’m sending the documentation through the repository review step now; I’ll wait for the result and address any concrete findings before handing it back.

<a id="m013"></a>
## 013 Assistant (final_answer) - 2026-07-29T00:34:56.231Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11915](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11915).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11915, 2026-07-29T02:06:15.127Z.

The inventory is complete and recorded at each milestone in [strategy_system_inventory.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_inventory.md:1). The working area now contains both the direction document and production inventory.

Main conclusions:

- The model → strategy → trainer principle remains sound.
- The main problem is the absence of one typed fulfillment/bound-state boundary. The trainer currently fills that role implicitly.
- Model loaders mix component materialization with distributed, precision, memory, and performance policy.
- Strategy contracts mix trainer-facing operations with family-internal collaboration and late conditional failures.
- Compatibility rules already exist, but are scattered between config validation, family implementations, and runtime checks.
- Specific placement pressure and nine future conformance scenarios are documented.
- The next discussion is now bounded: contract concerns/results, state ownership, fulfillment, mode/objective participation, and conformance tests.

Bead `sd-scripts-syv` is closed. `git diff --check` passes; no tests were needed for documentation-only work.

The repository review tool was unavailable after discovery, so this has not received an external review result.

<a id="m014"></a>
## 014 User - 2026-07-29T01:39:17.563Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11923](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11923).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11923, 2026-07-29T02:06:15.131Z.

my current stance as we discussed before is that I like the contract idea centralizing strategy (your training "plan") and handing it to the active trainer layer so it doesn't need to track or know about individual parts, but that's not how it happens today, and the contracts despite being made "strict" are not really respected in the way we imagined it when we came up with the system. and the part about being able to grab features and eventually components for your strategy is pretty attractive as a goal. Then there's also decisions (vae vs autoencoder lanugage, specific component, names, handling, separation, etc basically same with "clip") about the models folder

I didn't read through the inventory fully yet, this is just my thoughts. we should work with our design docs (direction and now inventory) in general unless we decide on other details and actions. what should we do next?

<a id="m015"></a>
## 015 Assistant (final_answer) - 2026-07-29T01:40:43.331Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11930](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11930).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11930, 2026-07-29T02:06:15.132Z.

Your stance is supported by the inventory. Today the trainer stores VAE/text-encoder/denoiser projections and passes them back into the strategy, while the “strict” contracts mainly enforce method names. They do not prove that a concrete run’s features are compatible or fully implemented.

The intended architecture should be stated more firmly as:

```text
CONTRACT
defines what the training pipeline accepts

FAMILY INTEGRATIONS + FEATURES + COMPONENTS
fulfill the selected contract

FULFILLED TRAINING STRATEGY
is the bound, executable training plan

TRAINER
controls when things happen without understanding model parts
```

“One fulfilled strategy” should mean one narrow public façade—not one giant implementation class. Internally it can be assembled from the feature grab box and family/component integrations.

What I think we should do next:

1. Lock down the trainer-facing invariant in the direction document.

   The trainer should not retain semantic projections such as `vae`, `text_encoders`, or `denoiser`. It should call generic operations and receive generic results. It may provide infrastructure such as the accelerator, but the strategy should bind and update its own components.

2. Design the contract from the trainer outward.

   Walk through the trainer lifecycle and define the minimum semantic boundaries it needs:

   ```python
   strategy.bind(binding_context)
   strategy.prepare(preparation_context)

   result = strategy.training_step(step_request)
   evaluation = strategy.evaluation_step(evaluation_request)
   artifacts = strategy.sample(sample_request)
   saved = strategy.save(artifact_request)
   ```

   The names are provisional. The important part is deciding exactly what the trainer supplies and receives without exposing model anatomy.

   Optimization needs special attention: the strategy or mode should provide generic parameter groups/trainable state, rather than making the trainer discover components.

3. Separate structural fulfillment from runtime binding.

   Before expensive loading, fulfillment should determine that the selected representation, conditioning, objective, predictor, mode, caching, sampling, precision, and persistence choices are compatible. Runtime binding can then load components and attach accelerator-dependent state.

4. Define the first feature vocabulary.

   Start from what already exists rather than designing an unlimited plugin system:

   - latent representation
   - pixel representation
   - text conditioning
   - predictor invocation
   - DDPM integration
   - rectified-flow integration
   - cache codec
   - sampling
   - persistence
   - component preparation

   A selected feature is required for that run. “Feature” must not become another word for optional or unverified behavior.

5. Settle model/component vocabulary against those contract roles.

   The likely distinction is:

   ```text
   autoencoder / representation codec
       generic contract role

   VAE, SDVAE, AutoencoderKL
       concrete component identities

   text conditioning
       strategy concern

   CLIP-L, CLIP-G, T5, LLM
       concrete components

   CLIP adapter
       low-level component interaction

   weighted prompts / chunk assembly
       strategy features
   ```

   We should retain both generic roles and specific identities. Calling everything “autoencoder” would lose useful truth; calling the generic contract role “VAE” would encode one implementation into the pipeline.

6. Pressure-test the proposed contract against SD, SDXL, and SD3, then pixel diffusion and distillation. Those two extra cases expose the current fake-VAE and single-family/single-predictor assumptions without requiring us to design arbitrary component assembly immediately.

For documentation, I’d keep the roles simple:

- `strategy_system_inventory.md`: evidence about current production behavior.
- `strategy_system_direction.md`: decisions, leanings, rejected routes, and open questions.
- Only create another working document once we have a concrete candidate contract large enough to deserve one.

The immediate next session should therefore be the trainer-facing contract and bound-state discussion—not file movement, feature package layout, decorators, or an OpenSpec yet. Once that boundary is coherent, the feature catalog and model-folder decisions become much easier to judge.

<a id="m016"></a>
## 016 User - 2026-07-29T01:46:08.988Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11936](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11936).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11936, 2026-07-29T02:06:15.133Z.

1 let's try to make the contracts like this:

a basic idea built on what we have today, intuition about what we might have in the future extrapolated from the direction we want to move to, what "training a neural network" means, and what is needed to be able to train something in our repo today (this can be very narrow technically so let's start from where we are at)


<a id="m017"></a>
## 017 Assistant (final_answer) - 2026-07-29T01:48:33.741Z

Source: [019fab38-3678-72c3-b798-adb2e9e225b9, raw line 11946](/home/imi/.codex/archived_sessions/rollout-2026-07-29T00-13-35-019fab38-3678-72c3-b798-adb2e9e225b9.jsonl:11946).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11946, 2026-07-29T02:06:15.135Z.

That is the right derivation method. We should design a narrow, honest v1 contract from three sources—not attempt a universal model taxonomy.

| Source | What it contributes |
| --- | --- |
| Current production path | The operations required to train successfully in this repository today |
| “Training a neural network” | The irreducible semantics: trainable parameters, differentiable computation, loss, backward, and optimization |
| Intended direction | Enough separation and vocabulary to avoid permanently encoding SD’s VAE/CLIP/UNet anatomy |

The future intuition should influence the seams, but it should not make v1 pretend to support workflows we have never implemented.

### The irreducible contract

For the current trainer, a fulfilled strategy must be able to:

1. Validate that the concrete run selections form an executable training plan.
2. Bind all required model components and keep their live state authoritative.
3. Prepare those components for the selected precision, distributed runtime, mode, and training subject.
4. Supply optimizer-ready trainable parameter groups without requiring the trainer to understand which components they came from.
5. Accept a batch plus generic execution coordinates and produce a differentiable training result.
6. Support the required lifecycle transitions—training, evaluation, preparation, replacement, and finalization.
7. Fulfill every additional operation selected by the run, such as caching, validation, sampling, or persistence.

The trainer then owns:

- epochs, steps, and accumulation;
- dataloader traversal;
- backward;
- clipping;
- optimizer and scheduler advancement;
- trigger timing;
- interruption, observation, and cleanup.

It should not own:

- VAE/text-encoder/denoiser projections;
- conditioning assembly;
- representation conversion;
- predictor invocation;
- family-objective compatibility;
- knowledge of which component is being trained.

Its essential loop should eventually be conceptually close to:

```python
result = strategy.training_step(
    batch=batch,
    coordinates=step_coordinates,
)

accelerator.backward(result.loss)
optimizer.step()
```

`result.loss` should probably be the final differentiable loss for that training plan, not merely an intermediate family loss that requires the trainer to know about timesteps, objective modifiers, or model-family semantics. Metrics and observations can accompany it without becoming model parts.

### Three related contract levels

The current `TrainingStrategy` ABC collapses three different contracts. We should separate them conceptually before deciding Python classes.

1. Trainer-facing execution contract

   The narrow façade handed to the trainer:

   ```text
   bind
   prepare
   describe optimization inputs
   execute training step
   enter training/evaluation state
   execute selected lifecycle operations
   finalize
   ```

2. Strategy-fulfillment contract

   The internal rules used to assemble that façade:

   ```text
   selected representation
   selected conditioning
   selected predictor behavior
   selected objective integration
   selected training subject/mode
   selected persistence and auxiliary operations
   compatibility between all selections
   ```

   This is where the feature grab box belongs.

3. Model-component contract

   The concrete objects and mechanics used by those features:

   ```text
   component declarations
   loading and materialization
   component roles/capabilities
   low-level preparation quirks
   module replacement
   component state access
   ```

A family strategy becomes a tested default recipe that fulfills the second contract using known features and components. It is not the only imaginable assembly, but it remains the normal way SD, SDXL, and SD3 enter the system.

### Required versus selected concerns

We should preserve the distinction we reached earlier:

- The core training obligations are always required.
- Other concerns are selected by the concrete run.
- Once selected, they are fully required.

Caching is not inherently required to train a neural network. But if the run selects latent caching, the fulfilled strategy must contain compatible representation and cache-codec behavior. It cannot expose a method that merely raises later.

The same applies to:

- weighted prompts;
- validation;
- sampling;
- full-model persistence;
- adapter training;
- text-encoder training;
- FP8 preparation;
- deferred loading.

This avoids both extremes: forcing every strategy to implement everything and treating meaningful run requirements as vague “optional capabilities.”

### A deliberately narrow v1

We should explicitly allow v1 to mean:

- one trainer-controlled optimization step;
- one final differentiable loss per step;
- one optimizer plan, potentially containing parameters from multiple components;
- any number of bound model components;
- one fulfilled strategy façade;
- current SD, SDXL, and SD3 execution behavior.

That permits adapters and multiple trainable components without claiming immediate support for GAN-style alternating optimizers or arbitrary multi-stage training. Those can expand the contract later through an explicit version or extension.

### How model terminology fits

The generic contract vocabulary and concrete model vocabulary should coexist:

```text
contract concern: latent representation
concrete component: AutoencoderKL / SDVAE
component kind: VAE, deterministic autoencoder, tokenizer, etc.

contract concern: conditioning
concrete component: CLIP-L / CLIP-G / T5 / LLM
strategy feature: weighted prompts, chunk assembly, pooled-output handling
```

Therefore:

- “VAE” should not be the universal contract role.
- “Autoencoder” should not erase the fact that a component is specifically variational.
- “CLIP” should not be synonymous with conditioning.
- Specific component names remain model-owned facts.
- Generic concern names describe how the training plan uses them.

The next design milestone should be writing these principles into `strategy_system_direction.md`, followed by a first trainer-facing contract table: every operation, its semantic input, its result, its lifecycle state, and whether it is core or selected. We should derive that table from the inventory before choosing class names or modifying `contracts.py`.

<a id="m018"></a>
## 018 User - 2026-07-29T02:10:44.645Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 11955](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:11955).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11955, 2026-07-29T02:10:44.645Z.

the "run" doesn't select things, these are defined in the strategy build the same as today, just under the new contract. it might constrain or validate it (where this happens is not decided from what I've read so far). the contract defines what must be fulfilled and what is available to use whether it's a slot that needs to be filled or something that you're picking for yourself. we're not going for a super smart mind reader system that automatically picks things based on implication, at least not for now. strategies should be built with intent. we can validate that the choices are correct, but that's different from a feature dragging another with it, that should still be left to whoever is putting the strategy together. (and this is us, we make the default training strategies for models aka what you see currently)

This process is manual, it's not done by a hydra replacement or a config system, at least not initially anyways. It's the same as how model code is still written by hand. strategy is essentially "the model code" in this repo. there might be a config equivalent that can do the task of putting together a strategy in the future to make things more streamlined, but that's not current goal

<a id="m019"></a>
## 019 Assistant (final_answer) - 2026-07-29T02:11:56.034Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 11962](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:11962).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11962, 2026-07-29T02:11:56.034Z.

That correction materially changes the framing. “Fulfillment” should mean that a strategy author has explicitly implemented and assembled the contract—not that the runtime resolves a collection of requirements into a strategy.

The intended flow is:

```text
TRAINING CONTRACT
defines required slots, available feature interfaces, and execution rules
        │
        ▼
HANDWRITTEN STRATEGY DEFINITION
explicitly chooses and wires implementations
        │
        ▼
STRATEGY BUILD
constructs that known implementation, much as it does today
        │
        ├── validation may reject incompatible or incomplete wiring
        │
        ▼
FULFILLED TRAINING STRATEGY
handed to the trainer as one executable model-training implementation
```

There is no generic dependency resolver in that picture.

### What the contract does

The contract should tell a strategy author:

- what every strategy must provide;
- which slots admit different implementations;
- which reusable feature interfaces are available;
- what inputs and results those features use;
- what the trainer expects from the finished strategy;
- which combinations are semantically valid or invalid;
- which lifecycle rules implementations must respect.

It does not choose implementations.

For example, the contract may define a representation slot, a predictor slot, and an objective-integration slot. The SDXL strategy author explicitly wires:

```python
representation = SdxlLatentRepresentation(...)
conditioning = SdxlClipConditioning(...)
predictor = SdxlUnetPredictor(...)
objectives = SdxlObjectiveIntegration(
    ddpm=...,
    rectified_flow=...,
)
```

That example is conceptual; it does not settle composition versus inheritance.

Configuration may choose DDPM versus rectified flow only because the handwritten SDXL strategy deliberately supports and exposes both. Seeing rectified flow in the config must not cause the system to discover and attach some feature automatically.

Similarly:

- SD explicitly wires DDPM.
- SD3 explicitly wires rectified flow.
- A strategy using latent representation explicitly wires the needed representation behavior.
- If its cache feature needs a corresponding latent codec, the author explicitly provides it.
- Validation may detect that the author forgot the codec or connected incompatible tensor semantics.
- Validation does not add the codec.

That is the distinction between validating intent and inventing intent.

### What “available features” means

The grab box is a code organization and reuse system for strategy authors:

```text
features/
  representation/
    latent.py
    pixel.py
  diffusion/
    ddpm.py
    rectified_flow.py
  conditioning/
    clip.py
  persistence/
  caching/
```

A feature being available means:

> A strategy implementation can deliberately import, instantiate, inherit, or otherwise use this tested behavior.

It does not mean:

> The framework automatically activates this feature whenever its requirements appear to match.

The exact assembly mechanism remains open:

- explicit composition;
- current-style inheritance;
- decorators;
- imported factory helpers;
- a limited mixture.

But whichever mechanism we choose, the family strategy remains readable source code that shows its intent.

### Two kinds of choice

We should distinguish these clearly in the design:

1. Strategy-author choice

   The implementation author decides which components and features form the strategy and writes that choice into code.

2. Runtime configuration choice

   The strategy may deliberately expose a bounded choice to users—objective variant, caching enabled, precision mode, which components to train, and so on.

A runtime choice exists only because the strategy author implemented it. Configuration cannot summon arbitrary feature implementations that the strategy did not wire.

### What strictness should mean

A strict contract should not merely require method names. It should make handwritten strategy implementations provably coherent through:

- required typed slots or operations;
- typed feature inputs and results;
- explicit supported alternatives;
- compatibility validation;
- lifecycle/state rules;
- shared conformance tests;
- fail-fast rejection of an incomplete strategy build.

This remains ordinary authored software. The validation layer acts more like a type checker or construction invariant than a dependency injection system.

### Strategy as repository model code

Your description is accurate:

```text
library/models/
    implements or adapts the neural components themselves

library/strategies/
    implements how those components become a trainable model in this repository
```

The strategy is effectively the repository-specific executable model integration. That is why it must remain intentional and reviewable. Automatically assembling it from config would create a second implicit model implementation system and make behavior harder to understand.

A future declarative/config-based strategy assembler could eventually be syntax over the same contracts:

```text
declarative strategy description
        ↓
constructs the same explicit feature objects
        ↓
must satisfy the same contract and conformance tests
```

It is not part of the initial design.

This means the direction document should be corrected in several places:

- replace “run-selected features” with “strategy-authored features and explicitly exposed runtime choices”;
- define fulfillment as authored conformance, not resolution;
- state that features never pull dependencies into a strategy automatically;
- add manual strategy construction as an initial architectural invariant;
- list declarative/config-driven assembly as a possible future layer, not a current goal.

That correction should happen before drafting the actual contract table, because it changes who the contract is written for: primarily strategy implementers, with the finished contract surface consumed by the trainer.

<a id="m020"></a>
## 020 User - 2026-07-29T02:19:42.238Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 11968](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:11968).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11968, 2026-07-29T02:19:42.238Z.

yes. there is essentially no noob mode on strategy besides "trainer needs something to train chief)" and even that can be overridden if someone wants it, or allowed for mutated/adopted/custom strategies besides the base model training strategies as that would go against the spirit of experimentation and research

if you know this repo it takes a similar approach at a base level, although it's not for training [ljleb/sd-mecha](https://github.com/ljleb/sd-mecha)


<a id="m021"></a>
## 021 Assistant (commentary) - 2026-07-29T02:19:48.648Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 11972](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:11972).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11972, 2026-07-29T02:19:48.648Z.

I understand the distinction: the contract should provide a rigorous construction language, not a safety rail that narrows research to approved recipes. I’m checking `sd-mecha` directly because the useful comparison is likely its separation between an expressive low-level mechanism and authored compositions—not its specific inference/model-merging purpose.

<a id="m022"></a>
## 022 Assistant (final_answer) - 2026-07-29T02:21:11.835Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 11995](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:11995).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 11995, 2026-07-29T02:21:11.835Z.

Yes. The contract should constrain interoperability, not permissible research.

There is one boundary we should keep intellectually honest about: the active trainer needs some stable handshake. A custom strategy can avoid VAEs, predictors, conventional trainable components, or the standard feature assembly entirely, but it must still give the active trainer whatever that trainer requires. If someone also wants to replace backward/optimization semantics or the handshake itself, that becomes an explicit trainer-contract extension—not something we should disguise as satisfying the unchanged contract.

That suggests three levels.

### 1. Minimal trainer handshake

This is the genuinely hard floor for compatibility with the shared trainer:

```text
the strategy can be initialized/prepared
the trainer can ask it to perform the next training operation
the strategy returns enough information for the trainer to continue
the strategy participates in required lifecycle operations
```

Even the assumption that the result is one scalar loss should be treated as the narrow initial handshake, not a universal truth. A later experimental strategy might require multiple losses, optimizers, backward phases, or strategy-managed updates.

### 2. Standard strategy-construction contract

This is the organized route used by our handwritten model-family strategies:

```text
representation
conditioning
predictor invocation
objective integration
training-subject exposure
preparation
caching
validation
sampling
persistence
```

SD, SDXL, and SD3 would deliberately fill or override these slots. This is the paved architecture we maintain and test—not “noob mode” and not the only legal way to experiment.

### 3. Research escape surface

A custom strategy should be allowed to:

- replace a standard feature;
- override a whole slot;
- combine slots unusually;
- add a component dynamically;
- train only an attached component;
- use a model as a teacher or fixed instrument rather than the subject;
- implement the trainer-facing operation directly without using the standard internal decomposition;
- eventually target an extended trainer handshake where the shared trainer’s assumptions are insufficient.

So the standard feature system is compositional help, not a compulsory ontology.

The useful rule is:

```text
Default family strategies should use the standard contract vocabulary.

Custom strategies may bypass its internal decomposition,
but must still satisfy the active trainer handshake they claim to support.
```

That keeps the standard system coherent without making research ask permission from the architecture.

The `sd-mecha` comparison is apt. It provides model configurations, component/key structure, typed merge-method interfaces, recipe nodes, and an execution engine, but the actual recipe is explicitly authored. Its README describes recipes as exact executable instructions, and custom models and merge methods are extension points rather than things inferred from a beginner-facing configuration system. [`@merge_method` also validates declared parameter/return constraints and interface implementations](https://github.com/ljleb/sd-mecha/blob/main/sd_mecha/extensions/merge_methods.py#L598-L637), while [model configurations explicitly publish components, keys, tensor metadata, and aliases](https://github.com/ljleb/sd-mecha/blob/main/sd_mecha/extensions/model_configs.py#L65-L124). The framework supplies vocabulary and execution semantics; the author supplies intent. [sd-mecha README](https://github.com/ljleb/sd-mecha#usage)

What we should borrow:

- authored compositions rather than inferred compositions;
- structural contracts that make custom implementations interoperable;
- explicit component descriptions;
- validation of declared intent;
- a useful standard library that does not define the limit of experimentation;
- direct extensibility at both the operation and model-description levels.

What we should not copy automatically:

- its recipe graph, because state-dict merging naturally forms a lazy executable graph and we deliberately deferred a training graph;
- automatic interface dispatch;
- a registry-driven resolver as the primary strategy builder;
- its state-dict-centered model concept.

For us, the analogous structure is:

```text
sd-mecha                         this repository

model config                     model component declaration/adapter
merge method                     strategy feature implementation
authored recipe                  handwritten family/custom strategy
recipe validation/finalization   strategy conformance/compatibility validation
merge executor                   shared trainer
```

The next revision to the direction document should therefore add two explicit principles:

1. **Research freedom:** the standard strategy decomposition is the maintained default, not the boundary of permitted training behavior.
2. **Layered conformance:** a strategy may bypass standard feature slots, but it must satisfy the trainer handshake it claims to target.

Then the contract inventory table should include a column such as:

```text
standard slot
standard feature interface
direct strategy override allowed?
trainer-visible obligation
```

That will let us make the normal path strong without accidentally turning it into a gatekeeper.

> Compaction boundary at selected raw line 12000 (2026-07-29T02:26:17.001Z); generated summary omitted.

<a id="m023"></a>
## 023 User - 2026-07-29T02:26:17.101Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12009](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12009).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12009, 2026-07-29T02:26:17.101Z.

basically we didn't change the overall goal, because we are still the ones that put the strategies together, we just establish the mechanics around the contract (which covers the constraints as well) and the intended way to interact with the strategy system

if any updates should be made to [strategy_system_direction.md](docs_design/models-strategy-trainer/strategy_system_direction.md) now is a good time before context compacts


<a id="m024"></a>
## 024 Assistant (commentary) - 2026-07-29T02:26:24.277Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12013](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12013).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12013, 2026-07-29T02:26:24.277Z.

I’m using the OpenSpec exploration workflow to tighten the direction document without turning these architectural decisions into production code. I’ll first reread the current document and project workflow context, then revise only the language that still implies automatic strategy assembly or changes the original contract goal.

<a id="m025"></a>
## 025 Assistant (commentary) - 2026-07-29T02:26:56.645Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12039](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12039).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12039, 2026-07-29T02:26:56.645Z.

The document already preserves the original architecture at the top, but its contract section still contains one important drift: it describes “the run” as selecting and filing features. That suggests a resolver/config assembler we explicitly rejected. I’m checking the rest of the document for the same assumption before making a narrow correction.

<a id="m026"></a>
## 026 Assistant (commentary) - 2026-07-29T02:27:46.038Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12051](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12051).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12051, 2026-07-29T02:27:46.038Z.

The correction is architectural, not cosmetic: strategy authors—including us for the maintained family defaults—compose the strategy deliberately. The contract supplies vocabulary, required outcomes, lifecycle mechanics, constraints, and validation; it does not select features or infer dependencies. I’m recording that distinction, plus the separate research escape surface, so later OpenSpec work does not accidentally design an automatic assembler.

<a id="m027"></a>
## 027 Assistant (commentary) - 2026-07-29T02:29:21.513Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12078](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12078).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12078, 2026-07-29T02:29:21.513Z.

The direction now says the complete strategy is authored in Python, validated against the contract, and then handed to the trainer as one contract-facing object. It also records that reusable features are authoring tools—not units the run, Hydra, or a dependency resolver automatically assembles—and that custom research strategies can either meet the active trainer boundary directly or declare an extension. I’m doing the documentation checks now.

<a id="m028"></a>
## 028 Assistant (final_answer) - 2026-07-29T02:30:10.919Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12104](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12104).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12104, 2026-07-29T02:30:10.919Z.

Updated [strategy_system_direction.md](/home/imi/Projects/sd-scripts/docs_design/models-strategy-trainer/strategy_system_direction.md:122) to make the unchanged goal explicit:

- Strategies remain deliberately assembled in Python by repository developers.
- The contract defines mechanics, constraints, validation, and trainer-facing results.
- Configuration may only choose bounded alternatives intentionally exposed by a strategy.
- Validation rejects bad wiring; it never selects features or adds dependencies.
- The trainer receives the complete strategy rather than coordinating its components.
- Custom research strategies can satisfy the standard trainer contract directly or explicitly extend it.
- Graphs, automatic resolvers, and config-authored assembly remain out of scope.
- The `sd-mecha` comparison is recorded as an analogy for authored composition, not an implementation template.

I also updated the Beads issue with this milestone so it survives compaction. `git diff --check` passes; no tests were needed for documentation-only changes. `review-mcp` was not available in this session.

<a id="m029"></a>
## 029 User - 2026-07-29T02:37:41.523Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12110](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12110).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12110, 2026-07-29T02:37:41.523Z.

"onclusion: treat pixel/latent behavior as available feature implementations
that a strategy author can deliberately wire into a contract filing, not
necessarily as the family integration's class."

this means that the contract offers this and the author chooses it? I'm not 100% on what "deliberately wire into a contract filing" means


<a id="m030"></a>
## 030 Assistant (final_answer) - 2026-07-29T02:37:53.905Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12114](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12114).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12114, 2026-07-29T02:37:53.905Z.

Mostly, but with one important distinction:

- The contract defines a required **representation concern** and the interface/results it must provide.
- The feature catalog offers known implementations such as `PixelDiffusion` and `LatentDiffusion`.
- The strategy author explicitly chooses and connects an implementation.
- Validation checks that the chosen representation works with the other strategy parts.

So “contract filing” meant “the strategy’s concrete implementation choices,” but it is unnecessarily abstract wording.

A clearer version would be:

> Conclusion: treat pixel and latent diffusion as reusable representation implementations. A strategy author explicitly chooses and connects one when implementing the training contract; it does not need to define the family strategy’s class hierarchy.

The contract offers the **place and rules**; the feature catalog offers the **implementations**; the author performs the **wiring**.

<a id="m031"></a>
## 031 User - 2026-07-29T02:42:44.308Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12120](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12120).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12120, 2026-07-29T02:42:44.308Z.

nice. what are the next questions to decide/clarify?


<a id="m032"></a>
## 032 Assistant (final_answer) - 2026-07-29T02:43:21.477Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12125](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12125).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12125, 2026-07-29T02:43:21.477Z.

The next central question is:

> What must the trainer be able to ask of one completed strategy, without knowing how that strategy was assembled?

That should be decided before naming feature classes or moving files. I’d work through these in order:

1. Trainer-facing contract

   - Which lifecycle operations does every trainer-compatible strategy expose?
   - What are their typed inputs and results?
   - Does the trainer call broad operations such as `load`, `prepare`, `training_step`, `validate`, `sample`, and `save`?
   - Which current calls should disappear because they expose VAE/text-encoder/denoiser anatomy?

2. Shape of the fulfilled strategy

   The trainer should receive one public strategy surface, but internally it could contain:

   - one family integration;
   - representation and conditioning features;
   - a training mode;
   - an objective;
   - persistence and validation behavior.

   We still need to decide whether mode and objective are inside that object or remain separate collaborators hidden behind the same trainer-facing façade.

3. Loaded-state ownership

   If the strategy owns its loaded components, we must define:

   - how training mode selects trainable parameters;
   - how distributed preparation replaces/wraps modules;
   - how checkpointing accesses components;
   - how metadata observes them;
   - whether anything outside the strategy can replace a component.

4. Contract concerns

   We need the first concrete vocabulary. Likely candidates include:

   - training subject;
   - input representation;
   - conditioning;
   - prediction/forward execution;
   - objective and loss production;
   - component preparation;
   - validation;
   - sampling;
   - persistence.

   The important question is which are trainer-facing requirements and which are internal tools used by a strategy to satisfy those requirements.

5. Feature interfaces and compatibility

   For example, what exactly does a representation implementation provide?

   ```text
   batch input
       → clean training state
       → cacheable representation
       → shape/semantic description
   ```

   We need enough declared semantics for validation, without creating an automatic dependency resolver.

6. Model/component vocabulary

   This includes the unresolved naming and ownership questions:

   - generic `autoencoder` role versus concrete VAE implementations;
   - CLIP as a component versus CLIP-based conditioning behavior;
   - predictor role versus UNet/DiT implementation;
   - adapters and newly attached components;
   - what belongs in `models/` versus reusable strategy features.

7. Conformance and contract evolution

   - What shared tests prove that SD, SDXL, and SD3 genuinely fulfill the same contract?
   - When does an experiment still satisfy the existing contract?
   - When does it require an explicit extension or new contract version?
   - Where does validation execute: builder, strategy finalization, trainer startup, or some combination?

8. Migration shape

   Only after the above should we decide which current contract methods move, combine, disappear, or become feature interfaces.

I would start with question 1 and use the completed inventory to draft a concrete trainer-facing contract table:

```text
operation | trainer supplies | strategy returns | trainer must not know
```

That is the boundary everything else has to satisfy.

<a id="m033"></a>
## 033 User - 2026-07-29T02:52:16.745Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12132](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12132).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12132, 2026-07-29T02:52:16.745Z.

well this should be dictated by the trainer no? which also dictates the contract. idk if this means strategy passes a bundled item that the trainer then needs to unpack, or how this is handled in python/other repos normally

<a id="m034"></a>
## 034 Assistant (commentary) - 2026-07-29T02:52:35.610Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12137](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12137).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12137, 2026-07-29T02:52:35.610Z.

I’m staying in the OpenSpec exploration stance here because this is the contract boundary decision, not implementation yet. Your correction is right: the trainer’s execution needs define the contract; the strategy fulfills it. I’ll compare that conclusion against a few established Python training interfaces so we can distinguish a clean protocol from an opaque “bundle the trainer unpacks.”

<a id="m035"></a>
## 035 Assistant (final_answer) - 2026-07-29T02:54:33.261Z

Source: [019fab9f-4de5-7951-9cf5-d8acfe3d4e2f, raw line 12186](/home/imi/.codex/sessions/2026/07/29/rollout-2026-07-29T02-06-11-019fab9f-4de5-7951-9cf5-d8acfe3d4e2f.jsonl:12186).
Export record: `019fab9f-4de5-7951-9cf5-d8acfe3d4e2f`, raw line 12186, 2026-07-29T02:54:33.261Z.

Yes. The trainer—or more precisely, the shared training pipeline we want the trainer to embody—dictates the contract.

The dependency direction should be:

```text
TRAINER
  defines what it needs to execute training
          │
          ▼
TRAINING-STRATEGY CONTRACT
  expresses those needs as typed operations and results
          │
          ▼
STRATEGY
  implements the contract using its chosen features and components
```

One caveat: we should not simply copy everything the current `Trainer` happens to do. Some current requirements, such as directly holding a VAE and text-encoder list, are historical leaks. We first decide what the generic trainer legitimately owns, and then that becomes the contract.

## The trainer should not unpack the strategy

The clean Python pattern is that the strategy itself is the aggregate object. The trainer invokes its public contract:

```python
class TrainingStrategy(Protocol):
    def prepare(self, context: PreparationContext) -> None:
        ...

    def training_step(
        self,
        context: TrainingStepContext,
    ) -> TrainingStepResult:
        ...

    def save(self, context: SaveContext) -> None:
        ...
```

```python
class Trainer:
    def __init__(self, strategy: TrainingStrategy) -> None:
        self.strategy = strategy

    def run_step(self, batch: Batch) -> None:
        result = self.strategy.training_step(
            TrainingStepContext(batch=batch, step=self.global_step)
        )

        self.accelerator.backward(result.loss)
        self.advance_optimizer()
        self.record_metrics(result.metrics)
```

Internally, the strategy can be assembled however its author wants:

```python
class SdxlTrainingStrategy:
    def __init__(self, ...):
        self.representation = LatentDiffusion(...)
        self.conditioning = SdxlConditioning(...)
        self.predictor = SdxlPredictor(...)

    def training_step(self, context):
        clean = self.representation.prepare(context.batch)
        conditioning = self.conditioning.encode(context.batch)
        prediction = self.predictor.predict(clean, conditioning)
        return ...
```

The trainer never asks for `strategy.representation`, `strategy.autoencoder`, or `strategy.predictor`.

## Typed bundles are still useful

The problem is not bundling itself. The problem is exposing model anatomy.

Good boundary objects have trainer-level meaning:

```python
TrainingStepResult(
    loss=loss,
    metrics=metrics,
    observations=observations,
)
```

```python
TrainableGroup(
    parameters=parameters,
    learning_rate_role="denoiser",
)
```

Bad boundary objects merely package the current leakage:

```python
LoadedThings(
    vae=vae,
    text_encoders=text_encoders,
    denoiser=denoiser,
)
```

The first two tell the trainer what it needs to do generically. The last one makes it unpack and coordinate one specific model topology.

## There are two kinds of interaction

The likely answer is a hybrid:

1. Model-specific behavior stays behind strategy methods.

   ```python
   result = strategy.training_step(context)
   ```

2. Trainer-owned infrastructure receives generic typed descriptions where it genuinely must act.

   ```python
   trainables = strategy.describe_trainables()
   optimizer = trainer.build_optimizer(trainables)
   ```

The remaining design question is therefore narrower:

> For distributed preparation, optimizer construction, and component replacement, can the trainer operate on generic handles/plans, or should the strategy perform those operations using trainer-provided infrastructure services?

That is where some controlled strategy–trainer exchange may be unavoidable. It should exchange semantic plans and results, not SD component identities.

## How established frameworks handle it

PyTorch Lightning is close to this shape. A `LightningModule` is the authored training recipe: it owns `training_step()`, `validation_step()`, and optimizer configuration, while `Trainer` owns the loop, device movement, backward, and scheduling. The trainer receives the module and calls its contract rather than unpacking its internal neural modules. [LightningModule documentation](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), [Trainer execution overview](https://api.lightning.ai/docs/pytorch/stable/common/trainer.html)

Hugging Face’s `Trainer` uses a narrower, model-shaped contract: the model must accept particular inputs and return loss/output structures, while custom behavior can override `compute_loss()` or `training_step()`. Its documentation explicitly warns that non-Transformers models may behave unexpectedly if they do not follow those conventions. That is a useful warning for us: we should not make our trainer contract accidentally “SD-shaped” in the same way that HF Trainer is intentionally “Transformers-shaped.” [Hugging Face Trainer documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer)

The strongest direction for this repository is therefore:

```text
trainer owns:
  contract definition
  temporal execution
  optimization
  distributed/infrastructure policy

strategy owns:
  the complete authored fulfillment
  loaded model-family state
  feature/component coordination
  model-specific execution

boundary objects carry:
  generic requests
  generic plans
  generic results
```

This also suggests that the trainer-facing contract may ultimately belong under `library/training/`, while reusable feature interfaces belong under `library/strategies/`. The current `strategies/base/contracts.py` placement makes the provider appear to own a contract that is actually dictated by its consumer.
