## 1. Plan-aware contract cleanup

- [x] 1.1 Remove legacy optimizer train/eval callback fields from the plan-aware `OptimizerBuildResult` contract and fine-tune build path.
- [x] 1.2 Update the shared optimizer-phase normalization seam so legacy tuple callers still work while plan-aware results no longer rely on callback-pair state.

## 2. Verification

- [x] 2.1 Refresh focused unit tests for fine-tune optimizer building and optimizer-phase normalization around the cleaned-up contract.
- [x] 2.2 Run focused validation and record the cleanup in `CHANGELOG.md`.
