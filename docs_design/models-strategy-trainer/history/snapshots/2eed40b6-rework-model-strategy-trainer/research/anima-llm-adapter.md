# Anima LLM Adapter

## Scope

Study Anima's built-in LLM Adapter as a native model component.

This is not a PEFT study. LoRA/LoHa/LoKr support is mentioned only where it
reveals ownership or trainability boundaries of the underlying component.

Sources:
- kohya-ss/sd-scripts
  - library/anima_models.py
  - library/strategy_anima.py
  - library/anima_train_utils.py
  - anima_train.py
- tdrussell/diffusion-pipe
  - models/cosmos_predict2.py
  - models/llm_adapter.py
  - docs/supported_models.md

Claim labels:
Observed / Inferred / Unknown
```

---

## 1. Component topology

This should establish that the conditioning system is not simply:

```text
text encoder → DiT
```

but:

```text
caption
   │
   ├─ Qwen3 tokenizer
   │      ↓
   │   Qwen3
   │      ↓
   │   source hidden states ────────┐
   │                                │
   └─ T5 tokenizer                  │
          ↓                         │
       token IDs                    │
          ↓                         │
   adapter-owned embedding          │
          ↓                         │
   target/query representation      │
          └─────────────────────────┤
                                    ↓
                              LLM Adapter
                                    ↓
                        DiT conditioning space
                                    ↓
                                   DiT
```

And explicitly note:

* Qwen3 is an external text encoder.
* There is **no T5 encoder** in this path.
* T5 token IDs are consumed by the LLM Adapter's own learned embedding.
* `LLMAdapter` is a native model component.
* The DiT consumes the adapter output through its normal conditioning/cross-attention path.

That distinction is the whole reason the case is useful.

---

## 2. What the adapter actually is

Document the real structure rather than calling it a projection layer.

Something like:

```text
LLMAdapter
├─ learned target embedding
├─ optional input projection
├─ rotary embedding
├─ 6 × LLMAdapterTransformerBlock
│    ├─ self-attention
│    ├─ cross-attention to Qwen3 states
│    └─ MLP
├─ output projection
└─ RMSNorm
```

Each adapter block effectively does:

```text
target representation
       ↓
self-attention
       ↓
cross-attention
       ↑
Qwen3 hidden states
       ↓
MLP
```

So this is much stronger than:

```text
Qwen hidden_dim → Linear → DiT hidden_dim
```

It is a small transformer subsystem that **constructs a new conditioning representation**.

That gives us a useful general observation:

> A representation adapter may itself be a substantial neural subsystem rather than a shape conversion.

---

## 3. Representation roles

This section is probably the most important for your architecture.

Keep the representations distinct:

```text
caption text

Qwen token IDs
T5 token IDs

Qwen hidden states

adapter target embeddings

adapter intermediate states

adapter output / DiT conditioning representation
```

The temptation would be to call all of this `"text_embeddings"`.

Anima proves that's too vague.

Especially:

```text
Qwen hidden state
    ≠
DiT conditioning embedding
```

even though both are 1024-dimensional in this implementation.

The **semantic representation matters independently of shape**.

I'd extract:

> Equal tensor dimensionality does not imply representation compatibility.

and:

> The model-facing text representation may be produced by a trainable model component downstream of the nominal text encoder.

---

## 4. Training ownership

Here I'd use **full Anima training**, not PEFT, because it's cleaner evidence.

`anima_train.py` separates real model parameters into groups including:

```text
base
self_attn
cross_attn
mlp
AdaLN modulation
LLM Adapter
```

with a dedicated:

```text
llm_adapter_lr
```

and `llm_adapter_lr = 0` freezes it.

So:

```text
component ownership:
    LLMAdapter belongs to Anima

optimization ownership:
    may be active
    may be frozen
    may use its own LR
```

This gives the useful constraint:

> Component membership does not determine optimization policy.

And conversely:

> A nested subcomponent may require independent optimization policy without becoming a separately owned model.

I'd mention LoRA only briefly here:

```text
The network-training path can additionally place PEFT overlays on the
LLMAdapter, demonstrating that native-component identity and trainable
parameter topology are separate concerns.
```

Then move on.

---

## 5. The caching boundary

This deserves its own section because it is the really unusual part.

Anima caches:

```text
Qwen prompt embeddings
Qwen attention mask
T5 input IDs
T5 attention mask
```

Not:

```text
final LLMAdapter output
```

Therefore:

```text
                   PRECOMPUTED
caption → Qwen → Qwen hidden state ─────┐
                                        │
caption → T5 tokenizer → token IDs ─────┤
                                        │
                   LIVE TRAINING        ↓
                                  LLM Adapter
                                       ↓
                                      DiT
```

This means the Qwen model can be removed from the live training graph while the adapter still trains.

That's a killer constraint:

> **Caching can cut through the middle of a logical conditioning pipeline.**

And:

> **A cached upstream representation does not imply that downstream conditioning computation is frozen.**

Also:

> Cache validity depends on which upstream producers are trainable.

If Qwen itself changes:

```text
Qwen weights change
    ↓
cached Qwen hidden states become stale
```

If only the LLM Adapter changes:

```text
cached Qwen states remain valid
    ↓
adapter recomputes from them every step
```

That's extremely relevant to your preparation architecture.

---

## 6. Execution ownership

I'd explicitly show that the adapter executes as part of the DiT forward path.

Something like:

```text
Anima.forward(...)
    ↓
_preprocess_text_embeds(...)
    ↓
LLMAdapter(...)
    ↓
forward_mini_train_dit(...)
```

The implementation deliberately keeps the adapter inside the model forward for training/DDP correctness.

So there's another distinction:

```text
logical conditioning preparation
        ≠
necessarily preparation-time execution
```

Some conditioning work belongs **inside live differentiable execution**.

Which means your architecture can't decide:

```text
"text conditioning" → preparation phase
```

purely from its semantic category.

It depends on trainability and gradient requirements.

---

## 7. Artifact and loading semantics

Anima can obtain the adapter weights:

```text
embedded in the DiT checkpoint

or

from a separate adapter weights file
```

while at execution time the adapter behaves as part of the Anima model.

That gives another nice separation:

```text
storage artifact boundary
        ≠
runtime component boundary
```

A component may be:

```text
separately stored
separately replaceable
separately trainable
```

without becoming a separate top-level model role.

I'd phrase the constraint as:

> **Loading boundaries, execution boundaries, optimization boundaries, and artifact boundaries need not coincide.**

That's been showing up everywhere in this research.

---

## 8. Design pressure

I'd finish with maybe these five, rather than a giant findings table:

> **Can a native model contain a learned representation adapter that remains independently freezeable/trainable without being promoted to a separate top-level model?**

> **Can a cache boundary occur between an upstream encoder and a downstream trainable adapter?**

> **Can cache validity be derived from the trainability of representation producers rather than from broad labels such as “text encoder caching”?**

> **Can two tensors with identical shape remain distinct representations with incompatible semantics?**

> **Can storage, execution, optimization, and model-component ownership remain separate axes?**

And then one compact conclusion:

```text
Anima's conditioning path is not:

text encoder → cached embedding → DiT

It is:

text
 ↓
multiple tokenizations
 ↓
Qwen representation
 ↓
trainable representation bridge
 ↓
DiT-facing representation

with a legal cache boundary between Qwen and the bridge.
```

That is the bit I'd want preserved for the trainer design.
