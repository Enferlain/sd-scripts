I now have enough evidence to complete the review. Here is my assessment.

# Review: rework-model-strategy-trainer planning update

## Verdict

The update is coherent with the three settled point-5 conclusions and systematically removes the monolithic adapter-capability wording. I found **one substantive incoherence** (an over-mandatory SHALL in `adapter-system`), **two minor coverage gaps**, and a few nits. No contradictions with the recorded direction were found; the Q1–Q5 direction record (strategy_system_direction.md, "Settled Binding-Authority Ownership") and the notes' TrainingMode-dissolution conclusions are faithfully carried.

## Verification against the three conclusions

**1. Capability facts are distinct** — properly recorded.
- Design D3's new paragraph states all four facts. The capability spec covers them: availability→provision non-inference (Req 1 "explicit rather than inferred from method presence, `hasattr()`…"), provision/request separation (Req 2 + the new "Provided capability is not requested" scenario), and readiness (Req 4 "Capability readiness derives from accepted state"). The adapter-system delta adds the availability-vs-provision rule for PEFT methods specifically ("repository availability alone SHALL NOT silently add support").

**2. Artifacts vs. exact restoration** — consistent across all three layers.
- Design D11's addition, the capability spec's expanded "Persistence and restoration remain distinct capabilities" (coordinator paragraph, "Product and snapshot have different outcomes", "Contributor participates in exact restoration"), and the pre-existing `training-artifact-persistence` / `run-participant-state` / `training-optimization` identity requirements agree. Ownership attribution is unambiguous everywhere: Trainer/delegated pipeline coordinates; contributors expose owned state and "MUST NOT become the coordinator or identity authority". The adapter-system resume-state scenario matches.

**3. TrainingMode dissolution / PEFT integration** — coherent.
- D12's expanded table, the PEFT-integration paragraph (shared behavior enumerated; "a particular method supplies only its method-specific implementation… and genuinely unique operations"), bounded per-strategy method support, the FineTuneMode paragraph, and the "no mode value or mode object crosses the accepted-arrangement boundary" rule are all mutually consistent, and mirrored in the proposal, `adapter-system` (integration sentence + "Method uses the shared PEFT integration" and "Undeclared PEFT method is selected" scenarios), the lora/vera deltas, the `loha` seams change, and tasks 2.2/5.5.
- Grep confirms **zero remaining "adapter capability" phrasing** in the change; the post-step scenario now correctly splits "accepted operation at its declared lifecycle point" vs. "named capability request when genuinely pipeline-recognized", matching D12's row exactly.
- The retained "fine-tune mode"/"adapter mode" scenario headings in `optimization-target-refs` are explicitly documented in the design as delta-identity retention, with bodies assigning no mode ownership. I also verified delta coverage against the main specs: every `AdapterMode`-invoking requirement/scenario in `adapter-system`, `adapter-module-targeting`, and `repo-owned-lora-method` is either MODIFIED or REMOVED by this change; `component-qualified-selectors` and `optimizer-group-lr-behavior` encode no mode authority, so their absence from the modified list is correct.

## Findings

**F1 (moderate) — Over-mandatory method selection in `adapter-system`.**
`specs/adapter-system/spec.md`, "Authored adapter intent…": "Strategy construction **SHALL** select a method from that declared set for a run." Design D12 says "**may** select one of those methods for a run". As written, the spec requires every run of a strategy that adopts the PEFT integration to select a method — including a direct-training run that never realizes an adapter participant. That (a) diverges from the design's permissive wording, (b) is in tension with conclusion 1's provision/request distinction you just codified ("a strategy may provide a capability that a particular run never requests"; the new "Provided capability is not requested" scenario), and (c) conflicts with D12's own "direct training, PEFT training, or their contract-compatible combination" not being mutually exclusive. Recommend scoping: "a run that realizes an adapter participant SHALL select its method from that declared set."

**F2 (minor) — "No replacement mode" is normatively weaker in the spec than in the design.**
`adapter-system` "Pipeline coordination replaces mode ownership" says only "No `TrainingMode` replacement SHALL receive **unrestricted** whole-Trainer mutation access" — literally permitting a mode object with bounded access. The settled rule ("no mode value or mode object crosses the accepted-arrangement boundary into Trainer") is enforced only indirectly via `training-contract`'s "Normal Trainer construction cannot bypass strategy fulfillment" ("not … a separately assembled strategy/mode/objective set"). Consider strengthening the adapter-system sentence to prohibit any replacement mode authority, not just unrestricted access.

**F3 (minor) — FineTuneMode dissolution / artifact-choice independence is design-and-tasks only.**
"No semantic replacement for FineTuneMode" and "emitting a full model is a separate product choice" appear in D12 and task 5.5 but in no delta requirement/scenario. Partial coverage exists (training-optimization "Adapter and base parameters are selected" for composition; persistence "Merged adapter product is requested"; capability "Full-model persistence is unavailable" showing product availability follows declaration, not mode), but there is no scenario asserting that a direct-subject run may declare a full-model product independent of training-subject choice. Since no main spec encodes FineTuneMode-gated products (verified: no persistence main spec exists), this is a forward-looking gap; one added scenario in `training-artifact-persistence` (or an explicit item in task 1.2's audit, which currently lists only `AdapterMode`-gated scenarios) would make the settled conclusion normative rather than prose-only.

**F4 (nit) — D5 restore row is stale relative to D11.**
Design D5's "Restore runtime" input column lists "Trainer/backend/optimization and capability contributors", while D11 and the capability spec add accepted **operations** as contributors. Not contradictory (D5 is slated for expansion in G1/task 1.4), but syncing it now avoids propagating the shorter list into the task-1.4 table.

**F5 (confirmation) — vera main spec malformed; already tracked.**
`openspec/specs/repo-owned-vera-method/spec.md` is delta-shaped ("## ADDED Requirements", no spec title/`## Requirements` header), so the MODIFIED delta cannot strictly resolve against it until repaired. Task 1.2 explicitly records this — good; just ensure the repair lands before the task-1.2 strict-validation claim is checked off.

## Minor observations (no action required)

- The general capability-availability fact (repo has it vs. strategy provides it) is normative only via the no-method-presence-inference rule plus the PEFT-specific adapter-system scenario. Acceptable, but a general "repository gains a capability implementation" scenario would mirror conclusion 1's first fact as directly as the others.
- `loaded-model-components`, `model-family-metadata`, `training-observability`, `run-participant-state`, `training-contract`, and the other untouched specs in this diff remain consistent with the new D3/D4/D10/D11/D12 text (checked: D2/D4/D10 additions match the pre-existing `training-contract` and `accepted-training-execution` requirements; the "four kinds of change" map onto existing contract/execution requirements including the contract-version escape hatch).

Overall: approve the direction; address F1 before the next milestone review, and F2/F3 as spec edits or explicit audit items in task 1.2.

---
## Review Trace
- Workspace: /home/imi/Projects/sd-scripts
- Diff target: unstaged
- Context entries requested: 2
- Context files loaded: 18
- Files selected for initial diff: 9
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 38221 prompt tokens, 3648 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 41899 prompt tokens, 334 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 42741 prompt tokens, 307 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool call: search_repository
- Model call: API usage 43144 prompt tokens, 197 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 44477 prompt tokens, 275 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: search_repository
- Model call: API usage 45097 prompt tokens, 129 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 46384 prompt tokens, 114 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_file
- Model call: API usage 48044 prompt tokens, 5184 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 54805 prompt tokens, 346 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 55197 prompt tokens, 59 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_files
- Model call: API usage 57209 prompt tokens, 612 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_files
- Model call: API usage 61617 prompt tokens, 448 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_files
- Model call: API usage 63067 prompt tokens, 1945 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 65036 prompt tokens, 52 completion tokens
- Model call: model requested 1 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 65112 prompt tokens, 95 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: list_repository_tree
- Model call: API usage 65277 prompt tokens, 533 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Tool result deduplicated: search_repository
- Tool call: search_repository
- Model call: API usage 65856 prompt tokens, 59 completion tokens
- Model call: model requested 1 tool call(s)
- Tool call: read_files
- Model call: API usage 67587 prompt tokens, 2809 completion tokens
- Review complete after 18 model call(s)