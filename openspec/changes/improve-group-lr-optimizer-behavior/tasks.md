## 1. Optimizer Factory Behavior

- [x] 1.1 Update `library/optimization/optimizer_factory.py` so grouped optimizer construction treats explicit execution-group LRs as authoritative instead of blindly forwarding `optimizer.learning_rates.base` into every constructor path.
- [x] 1.2 Add a clear repo-owned error path for grouped LR configurations that still cannot be satisfied for a selected optimizer backend, instead of leaking backend-specific constructor type/value errors.

## 2. LR Semantics Alignment

- [x] 2.1 Align grouping and trainability helpers so `null` means inherit or “no fallback,” `0` means frozen baseline path, and positive values mean train.
- [x] 2.2 Update config dataclass help text, defaults comments, and nearby validation messaging to use the same frozen-versus-inherited terminology consistently.

## 3. Regression Coverage

- [x] 3.1 Add unit coverage for grouped fine-tune configs with `base: null` plus explicit named groups, including the optimizer-factory boundary.
- [x] 3.2 Add backend-focused regression coverage for the currently failing grouped optimizer path and for clear repo-owned failure behavior when a backend cannot support a grouped LR configuration.
- [x] 3.3 Add config/trainability coverage for the standardized `null` versus `0` semantics so inheritance and frozen baseline behavior stay stable.
