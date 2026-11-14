# GPT-6.1 Sol medium fallback review

Read-only native reviewer: `/root/products_review_sol`.
The required review-mcp call failed HTTP429 before assessment; this user-authorized fallback is separate.
The initial source snapshot is `review-source-hashes.json`.

Review outcome: **changes requested** for two confirmed findings.

- **P2 — Retained admission remains open after cancellation** at target.py:1155. Calling `runtime.cancel(run_task)` from the `make/call_complete` checkpoint can return normally without populating `interruption`. The target then admits the product despite `self.cancelled`. Reproduction raised `CancelledError` but emitted `product_retained` and recorded a cancelled `make` receipt. The reference explicitly rejects admission while closing. Check `self.cancelled or self.failed` immediately before retained admission; regress cancellation and failure at this checkpoint.
- **P3 — Malformed derived dependencies bypass attributed validation** at verify.py:600. Setting the derived fixture's `derive.retains` to `(9,)` makes `prepare()` raise raw `IndexError: tuple index out of range`. The parent product inspects the borrower's operands before that borrower's contract is validated. Validate all retained dependency/result shapes before traversing dependency edges, and regress malformed child positions with a source-attributed semantic rejection.

Independent verification: **55 product tests passed**. Both findings required additional boundary probes. No further actionable issues within the declared experimental scope. No files edited.
