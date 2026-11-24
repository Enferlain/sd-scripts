Reviewed [`8103760`](https://github.com/Enferlain/sd-scripts/commit/810376027dda4ac1501fe0a2ed0479171565e1db) against parent `5218fdb`. **Four findings reproduced.** Code, specs, tasks and issue status remain unchanged.

The graph-derived mechanism works for the covered fixtures. The findings concern accepted view behavior and insufficient verification of isolated sources.

1. **P1 — Isolated copies can silently change member alias correspondence.**  
   Location: [`state.py:503–514`](https://github.com/Enferlain/sd-scripts/blob/810376027dda4ac1501fe0a2ed0479171565e1db/docs_design/models-strategy-trainer/execution_candidate/state.py#L503-L514).

   **Reproduction:** One participant has `x is y`, with independent `z`; all values are `1`. Unit U selects paths `(x, y, z)`. Its copy instead has `x is z`, with independent `y`. Both representations contain two distinct parameters, have equal `state_dict` values and use disjoint storage.

   Execute an isolated retained sequence computing `L = x + 2*y + 3*z`, differentiating U, releasing and advancing at learning rate `0.1`. Correct current destinations receive gradients `(3,3)`, producing `x=z=0.7`. The candidate accepts the copy and produces **`x=0.6, z=0.8`**.

   **Expected:** Reject changed alias correspondence before invoking retained computation. Comparing deduplicated parameter counts cannot establish correspondence; the accepted path-to-member associations must survive copying. This follows the optimization and runtime-preparation member-correspondence requirements.

   **Why tests miss it:** Existing tests cover initial alias consolidation and copies borrowing current storage. They do not change the alias partition while preserving its cardinality.

2. **P1 — Isolated-source verification omits registered nonpersistent buffers.**  
   Location: [`state.py:490–501`](https://github.com/Enferlain/sd-scripts/blob/810376027dda4ac1501fe0a2ed0479171565e1db/docs_design/models-strategy-trainer/execution_candidate/state.py#L490-L501).

   **Reproduction:** Replace frozen F in the isolated routed fixture with `F(a)=weight*a+offset`, registering `offset` with `persistent=False`. Current `weight=1, offset=0`; its copy recreates `offset=10`.

   The candidate checks storage separation, but compares contents through `state_dict()`, which excludes that buffer. It accepts the copy: B’s gradient becomes **144 instead of 4**, and `b` becomes **−11.4 instead of 2.6**.

   **Expected:** Verify every registered source value needed by retained execution, including nonpersistent buffers, or reject unsupported copying before computation. This is registered participant state, within the README’s claimed checking scope—not an excluded private Python attribute.

   **Why tests miss it:** Snapshot tests use simple parameter-only models and a copy returning the original object. They do not test changed registered buffers absent from `state_dict`.

3. **P2 — `gradients=False` does not establish an intentional gradient cut.**  
   Location: [`state.py:123–137`](https://github.com/Enferlain/sd-scripts/blob/810376027dda4ac1501fe0a2ed0479171565e1db/docs_design/models-strategy-trainer/execution_candidate/state.py#L123-L137).

   **Reproduction:** Use `nn.Identity` for F, with a `View(..., gradients=False)`. Compute `h = target(a)` and `L = h*a`, with `a=2`.

   Disabling gradient recording returns the original input unchanged: `h is a`, and `h.requires_grad` remains true. Differentiation produces **4 instead of the cut-path gradient 2**; at learning rate `0.1`, `a` becomes **1.6 instead of 1.8**.

   **Expected:** A recording switch must not certify the required no-gradient target/intentional-cut role. Enforce the requested cut on returned tensors, or explicitly distinguish recording-only behavior and reject its use where a cut is required.

   **Why tests miss it:** Their target models multiply the input, creating a new output under disabled recording. Identity/pass-through outputs behave differently.

4. **P2 — Call-level evaluation silently excludes named views.**  
   Location: [`prepare.py:412–422`](https://github.com/Enferlain/sd-scripts/blob/810376027dda4ac1501fe0a2ed0479171565e1db/docs_design/models-strategy-trainer/execution_candidate/prepare.py#L412-L422), following the projection split at lines 104–110.

   **Reproduction:** An ordinary `Call(evaluation=True)` declares only `Use("f", view="named")`; that view has its default `evaluation=False`. Let F return `x` during training and `−x` during evaluation.

   Acceptance and preparation succeed. Calling the named view with `x=2` returns **+2**, although the enclosing call requests evaluation. Mode capture and `eval()` iterate only `Context.models`, which excludes named-view participants.

   **Expected:** Apply the accepted evaluation behavior to named-view participants, or reject this combination before loading. Silently ignoring the flag changes selected computation.

   **Why tests miss it:** Broad evaluation tests use raw models; named-view tests use view-level evaluation. Retained broad evaluation is rejected, but this ordinary-call combination remains accepted.

**Connected-mechanism evidence**

- Preparation derives executable order, cross-block locals and coordination from the accepted graph; covered execution does not rediscover static wiring.
- Named target/conduit attachments share one participant and refresh on replacement. Mixed-mode restoration works in the tested return/failure cases.
- Fixture B produces **(3,4)** across live/isolated and shared/recomputed variants. Pending gradients receive independent storage.
- Live old-state protection blocks premature advancement; borrowed-value use after release and publication with unfinished retained/contribution work are rejected.
- Covered producer cases preserve owner identity, actual source stamps, demand-end disposition and cleanup through repeated cancellation.
- An additional combined failure probe preserved **A optimizer/reset returned**, **B optimizer uncertain**, **B reset not attempted**, plus the separate lifetime-cleanup failure. No rollback or replay result was inferred.

**Verification**

All four requested commands pass: **128 tests**, Ruff lint, Ruff format and strict `ty`. All eight examples execute. Verification used Python 3.12.14, PyTorch 2.11.0, Ruff 0.15.17 and ty 0.0.49.

**Experimental limitations and unestablished claims**

The explicitly rejected shapes remain candidate limitations: multiple contributions per Sequence unit, dynamic/repeated Sequences, multiple retained sources per derivative, higher-order work, retained input waits and Sequence two-pass grants.

The evidence does not establish arbitrary module-copy fidelity, GPU/asynchronous completion, distributed or threaded safety, general producer-worker conformance, durable continuation, pending-gradient migration, safe replay, performance gates, production architecture adoption or G5 completion.