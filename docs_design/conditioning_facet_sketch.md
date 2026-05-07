# Conditioning Facet Sketch

This note sketches a possible `conditioning` strategy facet based on the
current active families (`sd`, `sdxl`, `sd3`) plus the reference `flux` and
`anima` strategy code.

This is not a final decision. The goal is to test whether conditioning is a
real shared strategy concern with a contract-worthy responsibility.

## Main conclusion

Conditioning looks like a real strategy concern.

Across model families, the same high-level job keeps appearing:

- load cached conditioning when available
- otherwise build live conditioning from tokenized text and encoder models
- merge cached/live paths when needed
- apply family-specific conditioning rules
- return the family conditioning object that diffusion / denoiser code uses

That shared responsibility is stronger than the implementation differences.

## Important boundary

This note does **not** argue for a large shared conditioning helper API.

The contract-worthy part is only the high-level responsibility:

- resolve model-family conditioning for a batch

The helper steps inside that responsibility are still family-local:

- what the token payload looks like
- what the encoded text payload looks like
- whether extra micro-conditioning metadata is involved
- how cache and live paths merge
- how denoiser-ready tensors are assembled

## Why this looks contract-worthy

### SD

Current conditioning flow is embedded in `library/strategies/sd/diffusion.py`:

- `_get_cached_text_conds()`
- `_encode_live_text_conds()`
- `_merge_text_conds()`
- `_get_text_conds()`

The final object is effectively text-conditioning only:

- hidden states from CLIP

### SDXL

Current conditioning flow is also embedded in
`library/strategies/sdxl/diffusion.py`:

- `_get_cached_text_conds()`
- `_encode_live_text_conds()`
- `_merge_text_conds()`
- `_get_text_conds()`

The final denoiser inputs combine:

- text conditioning
- pooled text output
- micro-conditioning metadata from `SdxlConditioning`

So SDXL already shows that "conditioning" is broader than "text encoding".

### SD3

Current conditioning flow is embedded in
`library/strategies/sd3/diffusion.py` and uses local payload helpers in
`library/strategies/sd3/encoding.py`.

The final denoiser inputs combine:

- CLIP-L / CLIP-G hidden states
- pooled CLIP output
- optional T5 outputs
- attention masks used by the family conditioning path

### Flux reference

In `strategy_flux.py`, the family keeps a distinct token payload and encoded
payload:

- tokens: `[clip_l_ids, t5_ids, t5_attn_mask]`
- encodings: `[l_pooled, t5_out, txt_ids, t5_attn_mask]`

### Anima reference

In `strategy_anima.py`, the family also keeps a distinct token payload and
encoded payload:

- tokens: `[qwen_ids, qwen_mask, t5_ids, t5_mask]`
- encodings: `[prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]`

These references reinforce the same point:

- family-specific conditioning payloads differ
- but the family-owned conditioning *responsibility* is stable

## Proposed facet shape

If we promote conditioning into a real strategy facet, keep the contract very
small.

Suggested required method:

```python
class ConditioningStrategy(ABC):
    @abstractmethod
    def resolve_conditioning(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        weight_dtype: torch.dtype,
        *,
        train_text_encoder: bool,
        is_train: bool,
    ) -> Any:
        ...
```

The return type should remain family-specific for now.

Possible future type:

- `ModelConditioning`

but only if the returned object really becomes the one stable family-owned
conditioning object across families.

## What should move if we do this

### Move into `strategies/<family>/conditioning.py`

Per family:

- `_get_cached_text_conds()`
- `_encode_live_text_conds()`
- `_merge_text_conds()`
- `_get_text_conds()`

plus any small family-local helpers directly tied to conditioning resolution.

### Keep in `encoding.py`

- token-to-encoder-output transformation
- encoder-specific pooling / masking work
- family-local payload helpers if they are only supporting encoding/conditioning

### Keep in `denoiser.py`

- final denoiser call shape
- any last-step assembly that is truly specific to the denoiser call

### Remove from `diffusion.py`

Diffusion should stop owning the condition-resolution flow directly.

Instead it should ask the conditioning facet for the resolved family object and
then continue with noise/target/loss logic.

## Why not put helper steps on the contract

These are implementation details, not stable shared obligations:

- `_get_cached_*`
- `_encode_live_*`
- `_merge_*`
- family-local payload conversion helpers

If those were put on the base contract, it would freeze the current
decomposition instead of the real shared responsibility.

## Relationship to `ModelConditioning`

There is already a `ModelConditioning` marker in
`library/strategies/base/contracts.py`, and the data layer already transports
cache-side conditioning via `CacheData.conditioning`.

That existing type is useful, but it currently only covers part of the story:

- cache-side model conditioning metadata (for example `SdxlConditioning`)

If we adopt a real conditioning facet, we should revisit whether
`ModelConditioning` becomes:

- the common family-owned conditioning object returned by the facet

or whether we want a separate marker for:

- cache metadata conditioning
- resolved denoiser conditioning

No decision is needed yet.

## Near-term recommendation

Do **not** implement this immediately in all families as a blind move.

Instead:

1. treat conditioning as a likely real strategy facet candidate
2. validate the single-method contract shape above against SD / SDXL / SD3
3. only then move the family conditioning flows out of diffusion

The key question for the next pass is:

- can `resolve_conditioning(...)` cleanly describe what all current families
  are already doing without dragging helper-level details into the base
  contract?
