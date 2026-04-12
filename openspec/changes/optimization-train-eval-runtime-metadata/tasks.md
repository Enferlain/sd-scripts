## 1. Optimizer runtime metadata modeling

- [x] 1.1 Extend `library/optimization/types.py` with explicit optimizer train/eval runtime metadata for plan-aware paths.
- [x] 1.2 Add focused unit coverage for optimizer runtime metadata and helper behavior.

## 2. Orchestration integration

- [x] 2.1 Add shared optimizer train/eval runtime resolvers/helpers in `library/optimization/optimizer_utils.py`.
- [x] 2.2 Update training orchestration call sites to use the shared helper instead of directly calling callback pairs.

## 3. Validation

- [x] 3.1 Run focused runtime-mode/orchestration tests and record the change in `CHANGELOG.md` if the implementation lands in this working session.
