## Review Verdict

I traced every fix listed in the task through both files (no uncommitted diff exists; I reviewed the files as they stand). **No correctness findings.** All four fixes are implemented and sound; the stale-result and artifact guards remain intact. Verification details and three non-blocking observations follow.

## Fix Verification

**1. Relationship transitions blocked while either endpoint is destructively PREPARING — verified.**
All four revision-carrying transitions call `_ensure_relationship_endpoints_not_preparing` (`exchange.py:732, 744, 754, 764`), which checks both `source` and `target` states via `_ensure_not_preparing`. The indirect relationship-revision paths are equally guarded: `replace_binding` (`:679`) and `preview_retirement` (`:700`) both call `_ensure_related_relationships_not_preparing`. `_declare_relationship` only runs at establishment, before any bound/prepared state can exist. Exercised by `test_relationship_cannot_change_while_an_endpoint_is_mutating` (`scenarios.py:593`).

**2. All relationship revision paths withdraw both endpoint prepared views — verified.**
`_invalidate_relationship_endpoints` (`exchange.py:1047-1051`) invalidates both endpoints and is invoked from every path that advances a relationship revision or lifecycle: resolve/activate/deactivate/detach (`:737, 747, 757, 767`), endpoint replacement via `_invalidate_relationships_for` (`:1045`, resetting ACTIVE/RESOLVED/INACTIVE to DECLARED while leaving DETACHED closed), and endpoint retirement (`:718`). I confirmed `_invalidate_prepared_state` does not double-bump `preparation_revision` when called twice for the same participant (second call is a no-op once views are cleared). Covered by tests at `scenarios.py:540`, `:645`, `:700`.

**3. `backend_state_for` jointly checks authority PREPARED status — verified.**
`exchange.py:398-413` checks every member of the exact backend group, not just requested refs. This correctly closes the gap where the authority invalidates a participant whose backend group survives a runtime retirement preview untouched (e.g. retiring a relationship endpoint that sits in a *different* group — `scenarios.py:744-747`, and replacement of a co-member — `:692-698`). The invariant "valid routes ⇒ status PREPARED" holds everywhere (`_install_preparation` sets them together; `_invalidate_prepared_state` clears them together), so the status check is sufficient.

**4. Overlapping groups merge/expand but cannot omit members — verified.**
Publish-time `_preview_install` (`exchange.py:349-359`) rejects any partial overlap (`existing_group <= participant_group` required) and otherwise drops intersecting groups, permitting merge/expansion. The laminar invariant is self-maintaining: two overlapping non-nested groups can never form, since every insertion re-checks all existing groups. Destructive jobs are additionally rejected at `begin_preparation` (`:793-796`) *before* any state mutation — `test_destructive_job_cannot_partially_cover_a_joint_backend_group` confirms routes and statuses are untouched after rejection.

**Stale optimistic results — sound.** `preview_preparation` (`exchange.py:852-898`) checks binding revision, lifecycle, relationship dependency revisions, cited source revisions, and — per behavior — attempt ownership (IN_PLACE) or withdrawn guarantees + `preparation_revision` drift (REPLACEMENT_ONLY). I specifically probed the corner where an unprepared, relationship-free participant is invalidated without a revision bump (e.g. co-eviction via backend-group retirement): that state is unreachable, because runtime co-membership only exists for previously published (hence revision-carrying) participants. `abandon_preparation` validates attempt ownership for all entries before mutating (`:838-850`), and `_issued_attempt` prevents republication of failed/published attempts.

**Artifact guards — sound.** `capture_artifact` (`exchange.py:1197-1226`) rejects PREPARING/INVALID/`preparing_attempt`, requires BOUND lifecycle and the PERSIST use, and records per-member revisions; mid-mutation and post-destructive-failure captures are both tested (`scenarios.py:640-641`, `:841-842`, `:856-857`).

## Non-blocking Observations

1. **Test coverage asymmetry (not a defect):** only `deactivate_relationship` with a PREPARING *target* endpoint is exercised. `resolve`/`activate`/`detach` and a PREPARING source endpoint rely on the same shared helper — implemented correctly, just untested.
2. **Conservative over-blocking:** `_ensure_related_relationships_not_preparing` (`exchange.py:1053-1056`) does not skip DETACHED relationships, so replacing/retiring an endpoint is blocked while a *closed* relationship's partner is mutating. Unnecessary but safe, and consistent with the design's explicit conservatism.
3. **Coverage check trust boundary:** the destructive begin-time group check depends on `TrainingStrategy.prepare` supplying the runtime's groups; a direct `authority.begin_preparation(..., behavior=IN_PLACE)` call with empty `inseparable_groups` would admit a partial destructive job. This is backstopped at publish (`_preview_install`) and at reads (`backend_state_for`), so no unsound state is observable — only wasted destructive work followed by a rejected publication. Acceptable for the spike; worth keeping the `TrainingStrategy.prepare` entry point as the only sanctioned path if this graduates beyond the spike.

---
## Review Trace
- Workspace: /home/imi/Projects/sd-scripts
- Diff target: unstaged
- Context entries requested: 1
- Context files loaded: 1
- Files selected for initial diff: 2
- Max review context: provider token window (no character safety override)
- Max tool result: 20000 chars
- Model call: API usage 15028 prompt tokens, 27 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: get_uncommitted_changes
- Tool call: list_changed_files
- Model call: API usage 15066 prompt tokens, 105 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 22715 prompt tokens, 90 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result truncated: read_file from 21210 to 20000 chars
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 31671 prompt tokens, 87 completion tokens
- Model call: model requested 2 tool call(s)
- Tool result truncated: read_file from 22671 to 20000 chars
- Tool call: read_file
- Tool result truncated: read_file from 21300 to 20000 chars
- Tool call: read_file
- Model call: API usage 40954 prompt tokens, 87 completion tokens
- Model call: model requested 2 tool call(s)
- Tool call: read_file
- Tool call: read_file
- Model call: API usage 43629 prompt tokens, 22657 completion tokens
- Review complete after 6 model call(s)