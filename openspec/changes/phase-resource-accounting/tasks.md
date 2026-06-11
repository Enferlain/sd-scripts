## 1. Remove the separate attribution report surface

- [x] 1.1 Remove the experimental `resource_attribution` report payload and Markdown section.
- [x] 1.2 Remove the `output.logging.resource_monitor.provenance` config knob and validation.
- [x] 1.3 Remove tests that assert the separate attribution surface, replacing them with guardrails that it is absent.

## 2. Define the phase accounting model

- [x] 2.1 Define a small accounting record shape under `library/logging/resource_monitor/` that is keyed by phase/event identity.
- [x] 2.2 Define the first owner labels and a separate accounting-gap status.
- [x] 2.3 Define how startup estimates and observed phase counters combine into accounting rows without rewriting observed facts.
- [x] 2.4 Preserve the distinction between structural accounting for known object sizes and scope accounting for observed movement during declared operations.

## 3. Add declared owner scopes

- [x] 3.1 Add a lightweight owner-scope API or context manager that can attach known work to the active phase/event context.
- [ ] 3.2 Instrument the first narrow code paths in order: cache phases, checkpoint save, model/component loading or registration, optimizer setup, first training step boundaries, then accelerator preparation.
- [ ] 3.3 Keep any heavier measurement behind explicit diagnostic configuration.

## 4. Upgrade report output

- [ ] 4.1 Add phase accounting rows to the existing resource report path without creating a separate user-facing subsystem.
- [ ] 4.2 Render accounted resource changes and accounting gaps in plain report language.
- [ ] 4.3 Add focused tests proving phase counters remain observed facts and accounting stays report/debug-derived.
