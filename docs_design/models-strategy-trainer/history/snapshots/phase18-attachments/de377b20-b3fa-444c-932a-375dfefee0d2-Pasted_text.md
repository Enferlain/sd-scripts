**It looks like a substantive, carefully scoped experiment. I would continue with it.** It establishes considerably more than “we can express the candidate using xDSL.” There are a few concrete verification gaps to fix, and some architectural questions that the successful fixtures still leave open.

I reviewed commit [`6bea859`](https://github.com/Enferlain/sd-scripts/tree/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate), including the implementation, tests and retained reviews. **All 245 tests passed in my independent run** with xDSL 0.74.0, Python 3.14.8 and CPU PyTorch 2.15.0. My environment had the GIL enabled; this adds functional evidence, without extending the repository’s free-threading qualification. I made no repository changes.

**The strongest parts are worth preserving.**

- Accepted programs genuinely detach from Python authoring and reconstruct without the builder.
- Semantic acceptance and target readiness are separated, including accepting programs beyond the particular CPU target’s limits.
- Runtime payloads, live ownership and physical completion remain outside the structural IR.
- A and C reuse the existing ownership engine instead of quietly replacing its semantics.
- The independent arithmetic and association oracles catch shared implementation mistakes that agreement between lowerings would miss.
- C correctly distinguishes the static contribution edge from an actual, verified live handback.

Those are useful architectural results, particularly the last two. See the [grant adapter’s verification boundary](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/grant_adapter.py#L204-L242) and [independent C oracle](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/grant_conformance.py).

**1. The clearest fix: bring A and B’s prepared-plan verification up to C’s standard.**

I injected mutations into plan emission, following the approach already used by the candidate’s corruption tests:

| Mutation | Observed result |
|---|---|
| B: change the emitted horizon to `0` | Preparation succeeds and returns horizon zero. |
| B: change the emitted target to an unsupported name | Preparation succeeds. |
| B: change the emitted feedback update recipe | Preparation succeeds; runtime still follows its built-in policy update. |
| A: swap the two map implementation symbols, leaving their origins unchanged | Preparation succeeds; execution reports `stopped`, producing incorrect data. |

For the last case, sample 1’s left value becomes **4 instead of 3**: the runtime performs `offset → double` under the original stage identities. The independent oracle rejects the result, which is good, but preparation misses the broken correspondence.

The reason is specific: A checks each prepared binding’s role/effect/version/completion against the registry, without checking that **this origin is still bound to the implementation selected by the accepted program**. Both maps satisfy the same role/effect contract. B has similar correspondence gaps and checks its target/bounds before emission without adequately rechecking the decoded result. Sources: [A’s decoder](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/model_adapter.py#L414-L450), [B’s decoder](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/prepare.py#L334-L378).

These probes simulate a faulty lowering or pass; they do not show the normal generator emitting incorrect plans. **They become particularly relevant when adding fusion.**

I would add an explicit prepared-plan verifier covering:

- accepted-program correspondence and digest;
- target, contract, bounds and disposition;
- origin-to-implementation correspondence, completeness and duplicates;
- source ancestry;
- supported recipe/control semantics.

C already provides a useful local example. For future fusion, correspondence can become many-to-one; it must still account for every preserved semantic stage.

**2. Some prepared recipes currently describe behavior that the runtime does not consume.**

B emits selection, handoff and feedback recipes, but its decoder primarily extracts call chains and bounds. `ScalarSession` supplies the selection arithmetic, policy update, slot-retention and feedback protocol directly. A carries a semantic subtree into its prepared recipe and performs substantial interpretation during owner-program construction. See [B’s preparation](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/prepare.py) and [runtime](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/runtime.py).

That can be a legitimate fixed-protocol backend. The missing decision is **what authority each prepared field has**:

- An executable choice must affect realization.
- A required protocol identifier must be checked against the implementation.
- An explanatory annotation should be identified as such.

Otherwise an impressive prepared dump can overstate how much execution is determined by that representation. The feedback-recipe mutation exposes this directly.

I would settle this field-by-field before expanding the prepared vocabulary.

**3. Shared infrastructure is demonstrated; composable domain semantics are less demonstrated.**

The installed extension mechanism is useful, and keeping model dependencies optional is good. However, acceptance still selects one whole-session contract. A’s verifier expects one particular arrangement, including exactly two models and one occurrence of the main workflow operations. `local_maps()` accepts a closed `MapOp | LocalOp` vocabulary. Sources: [extension selection](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/prepare.py#L110-L152), [A’s verifier](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/model_contract.py#L27-L118).

This is reasonable fixture scoping. It leaves open whether new behavior can compose through common contracts, without adding another whole-workflow verifier and adapter.

A small revealing test would be an independently defined preprocessing operation with:

- its own payload/schema;
- declared resource access and possible outcomes;
- a supported lowering;
- participation in the existing local composition analysis.

The test should expose which common interfaces are genuinely useful. It should not require building a comprehensive plugin system first.

There is one concrete extensibility restriction already visible: **`wf.flow` hardcodes `record`, `score` and `feedback` in core verification**. An extension cannot introduce an embedding or audio payload contract through that type without changing core. I would explore an extensible schema/type parameter before treating this as the settled core. [Definition](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/dialects.py#L32-L39).

**4. The most important remaining workflow test is continuous feedback together with model ownership.**

The README accurately discloses this gap. B has a small scalar realization; A and C use the existing engine. Together they demonstrate several important mechanisms, but **they have not yet demonstrated one continuous data/model workflow under shared coordination**. [Explicit qualification limits](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/README.md#L358-L364).

A’s two preparation paths also run inside one generated producer’s `TaskGroup`. This exercises concurrent paths and atomic correlation, while leaving independently closing or failing producer lifetimes less explored. Its filtering is resolved from finite keys before execution. [Producer realization](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/model_adapter.py#L492-L558).

I would add one bounded mixed fixture:

> An open producer feeds model-assisted preprocessing and learning; completed learning changes subsequent selection; one branch excludes a key dynamically; stop occurs while physical work and a blocked handoff coexist.

That would pressure the actual uncertainty: whether admission, feedback, ownership and shutdown compose across the data/model boundary. Another standalone numerical fixture would provide less architectural information.

**5. Fusion needs to test observable failure and lifetime semantics, as well as values.**

It is correct that generic DCE/fusion has not been enabled just because a binding says `pure`.

B already flattens recursive local composition into direct local calls. Therefore the next comparison needs a real difference in execution boundaries. The existing owner engine’s `calls`/`inline` comparison does not by itself establish the proposed data-region fusion result.

For that experiment, I would require preservation of:

- stage entry, validation and failure attribution;
- whether later stages entered;
- publication and feedback application boundaries;
- retained resource lifetimes;
- cancellation/drain behavior;
- origin mapping after fusion.

A particularly useful negative case is **a supposedly pure stage returning an invalid value after entry**. The unfused and fused realizations must agree about validation and which later work was permitted to happen. Equality of final successful outputs is insufficient. The current [local-chain acceptance](https://github.com/Enferlain/sd-scripts/blob/6bea859b8884552670c021502794f117f443d093/docs_design/models-strategy-trainer/xdsl_candidate/prepare.py#L71-L92) and independent oracles are good starting points.

**6. Region meaning and resource contracts need to become queryable before generic passes grow.**

The candidate contains several different meanings of “region”:

- service declarations;
- ordered local transformation;
- repeated contribution windows;
- granted phase protocols.

The verifiers understand those meanings today. A transformation should not have to rediscover them from operation names, string properties and fixture-specific Python branches.

Similarly, `read_source`, `current_read`, `scoped_write`, `pure` and `may_effect` are useful initial labels, but do not yet form a common resource/effect contract for analyses. Unit membership is also inferred by the adapters from the subject and advancement parameter; multi-unit overlap cannot yet be assessed from a general declared membership model.

I would introduce only the interfaces required by the next pass: region execution meaning, relevant resource uses, observable outcomes and fusion eligibility. Avoid prematurely constructing an elaborate universal effect system.

**My next-work order would be:**

1. Fix A/B’s prepared correspondence checks and retain the new corruption regressions.
2. Complete one genuine unfused/fused data-region comparison.
3. Run the small continuous feedback + model ownership fixture.
4. Add the independent operation/schema extension test.

The experiment already strengthens the case for xDSL as machinery infrastructure. The next evidence should establish that the representation supports **sound transformation and composition beyond the three selected protocols**. Those are the remaining tests most likely to change the architectural decision.