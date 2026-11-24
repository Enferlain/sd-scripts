# Scoped products and retained physical uses

This milestone addresses a concrete boundary: a completed call can leave a
value whose later use still depends on existing storage or state. Deep copying
ordinary values does not establish that boundary for memory views or non-leaf
PyTorch tensors.

The accepted language distinguishes detached payloads from `lang.retained`
products. Actual storage and autograd objects stay outside IR. A retained call
names the argument positions on which its one result depends; the installed
binding declares the same positions. Dependencies are issued capabilities or
previous retained products. Verification and preparation follow actual SSA
wiring, access issuers and resource identities.

```text
protected owner access
  make(capability) -> product [retains capability]
  derive(product) -> child  [retains product]
  consume(child, capability) -> detached result
  release(child)
  release(product)
  yield(detached result)
access settles
```

| Boundary | Contract and implementation |
| --- | --- |
| Owner | The defining ordered body owns the product and exactly one explicit normal release. |
| Borrow | Physical calls may read the actual payload repeatedly. A live borrower pins the product until physical completion and validation, including interruption. |
| Derived product | It retains its parent and transitive capabilities; the child must release before the parent. |
| Mutation | Borrowing grants no update authority. Known conflicting updates reject before factories; runtime update checks retain the live-product guard. Trusted bindings must honor their declarations. |
| Aliases | A retained result may share its declared dependencies. The buffer witness checks actual view/storage identity; the Torch witness preserves its actual derivative graph. General alias discovery is not implemented. |
| Escape | This first profile rejects yields, nested captures and publication of retained products. It supplies no implicit ownership transfer beyond the defining body. |
| Protection | Live-state dependencies retain their access protection. Snapshot dependencies retain the isolated snapshot and its actual observed version while the live owner may advance. |
| Release | Explicit release requires no live borrower or child product, clears its strong payload/dependency references and records the actual dependency versions. The enclosing access still ends at its own settlement boundary. |
| Failure/cancellation | Admission closes; entered consumers drain; cleanup releases remaining local products in reverse dependency order. A late physical failure can leave the resource unusable after physical settlement. |

Ordinary calls keep their detached-result isolation convention. A selected
closed numerical call can compute internally and return detached numbers.
Returning an unmarked non-leaf tensor through that boundary fails; the runtime
does not silently detach its graph. Selected Python implementations remain
trusted, as in the existing target. Read-only borrowing is an authority
contract, not Python sandboxing or automatic enforcement of every tensor alias.

The new fields are explicit in new artifacts. Absent retention fields retain
historical descriptor spelling. Cold loading rederives dependency and
consumer/release correspondence from accepted IR and installed contracts;
forged prepared claims cannot authorize factories or work. The existing local
rewrite accepts detached unary stages only, so these boundaries remain barriers.

## Two domain witnesses

The buffer owner contains `bytearray(b"abcdef")`. A physical call returns the
actual read-only view `[1:4]`; its consumer verifies that the view's storage is
the protected owner's buffer. The independent result is `98 + 99 + 100 = 297`.
A derived view selects its first two bytes and produces `197`.

The Torch owner contains a trainable scalar `a=2`, frozen base `b=3`, and input
`x=4`. One physical call returns the non-leaf activation `b * (a * x) = 24`.
Another constructs `loss = (activation - 1)^2 / 2` and uses real autograd to
obtain `d(loss)/da = 276`. A derived activation doubles the original and yields
gradient `1128`. The consumer explicitly retains the graph for repeated borrows;
release drops the final product references. The frozen base does not acquire
gradients, and `.grad` accumulation is not used. This qualifies CPU
storage/derivative retention, not an optimizer or production model path.

The Torch-free reference uses independent arithmetic and logical products;
it does not call target implementations or need actual tensor/view objects.
Target controls prove actual storage aliasing, graph traversal and dropping
product references with weak references after release.

## Qualification

`test_products.py` checks semantic/preparation rejection, dependency ordering,
versioned receipts, cold corruption, logical interruption and historical field
compatibility. `test_product_target.py` checks actual views/graphs, repeat
borrows, release before replacement, snapshot independence, factory exclusion,
diagnostic erasure and repeated cancellation with held physical work and late
failure. Expected values are independent of both runtime outputs.

`product_qualification.py` saves detached artifacts and fresh normal/optimized
children. The child cannot import construction fixtures, tests or numerical
stand-ins. It erases histories, reconstructs results and release receipts from
owner state, and records actual imports, GIL state, physical hold/drain
conditions and independent negative controls. It refuses to overwrite evidence
or accept sources changing during qualification.

```bash
PYTHONPATH=docs_design/models-strategy-trainer:/tmp/sd-scripts-xdsl-b-env/lib/python3.14t/site-packages \
  .venv-wsl/bin/python -m xdsl_language.product_qualification /tmp/products-qualification-new
```

The profile deliberately leaves cross-region/flow ownership transfer, mutable
borrowed aliases, arbitrary escaped autograd graphs, GPU/device ordering,
optimizer routing, performance and production APIs open. Completed receipts are
finite-case evidence. Reclaiming historical bookkeeping for continuous
production is separate from releasing a product's live storage dependencies.

The [retained evidence](evidence_products/validation.json) records 510
owner/language tests, 445 Torch-free tests with four optional skips, and 68
fresh normal/optimized cold children. GIL state remained false before and after
ordinary imports. The initial 52-child qualification is preserved under
`evidence_products/initial/`: it passed but did not cover the two edge cases
subsequently found by review. Both findings were reproduced, fixed and
independently confirmed; the expanded cold matrix includes completion-checkpoint
closure. The required provider review failed quota before assessment; the
user-authorized GPT-6.1 Sol medium fallback and its follow-up are retained
separately. All 940 protected prior source/evidence/report files remain unchanged.
