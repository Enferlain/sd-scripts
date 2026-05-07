## 1. Observability Contracts

- [ ] 1.1 Define the target observability ownership buckets under `library/logging/` (`console`, `metrics`, `summaries`, `reports`, `resource_monitor`) and document what each one owns.
- [ ] 1.1a Preserve stdlib `logging` plus Rich-backed console handling as the canonical console stack and align early config/setup logging with the same user-facing console surface.
- [ ] 1.2 Introduce shared structured summary/event models for startup diagnostics, optimizer summaries, and report/tracker reuse.
- [ ] 1.3 Introduce the repo-owned `TrainingObserver` seam plus the narrower backend sink contract for run init, metric persistence, and run finalization so existing Accelerate trackers remain supported while future repo-owned observability back-ends can plug in cleanly.

## 2. Startup Diagnostics And Adapter Breakdown

- [ ] 2.1 Move startup diagnostics behind structured observability rows and shared rendering/sink paths instead of sink-local ad hoc derivation.
- [ ] 2.1a Implement the preferred console presentation split: sectioned startup blocks for dense summaries and tagged lifecycle lines for standalone canonical status messages.
- [ ] 2.2 Derive adapter diagnostics from repo-owned adapter provenance instead of collapsing adapter mode into a single generic `adapter` component by default.
- [ ] 2.3 Define user-facing component naming rules for diagnostics so public labels remain the default display names while internal component keys stay available for shared grouping and sink/report alignment.

## 3. Tracker And Report Alignment

- [ ] 3.1 Align per-step tracker emission with the new observability contracts without breaking current metric series or Accelerate tracker behavior.
- [ ] 3.2 Update benchmark/run report generation to reuse the shared startup/report summary data rather than rebuilding overlapping component/resource views independently.
- [ ] 3.3 Preserve resource monitoring as a distinct sub-concern while defining the interface points where it contributes observability events and summaries to the broader system.
- [ ] 3.4 Land the first-pass ownership moves directly in `library/logging/console.py`, `metrics.py`, `summaries.py`, and `reports.py` rather than adding transitional stopgap modules.

## 4. Validation And Follow-up

- [ ] 4.1 Add or update focused tests for structured diagnostics rows, adapter breakdown derivation, and report/tracker reuse of the shared observability data.
- [ ] 4.2 Document the observability architecture boundaries so future logging, dashboard, or experiment-system work extends the same contracts instead of creating new side paths.
- [ ] 4.3 Reconcile the related beads issue(s) and implementation notes so adapter-reporting work and broader observability work stay connected but not conflated.
