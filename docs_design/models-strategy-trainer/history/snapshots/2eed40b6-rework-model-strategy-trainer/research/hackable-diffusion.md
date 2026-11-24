# Architecture research: Hackable Diffusion

**Implementation examined:** `google/hackable_diffusion`, commit `fb225b51132657ebebc294fc976418d0ecc30d7a` (2026-09-21).

Hackable Diffusion is a JAX/Flax research toolbox built around “composition over configuration” and explicitly describes its goal as separating architecture, corruption, inference, training loss, and sampling. The repository is under Google but its README states that it is **not an officially supported Google product**.

Primary sources:

[Repository](https://github.com/google/hackable_diffusion)
[Architecture overview](https://github.com/google/hackable_diffusion/blob/fb225b51132657ebebc294fc976418d0ecc30d7a/docs/index.md)
[Core protocols — `hd_api.py`](https://github.com/google/hackable_diffusion/blob/fb225b51132657ebebc294fc976418d0ecc30d7a/hackable_diffusion/lib/hd_api.py)
[Kauldron training integration — `kdiff/core.py`](https://github.com/google/hackable_diffusion/blob/fb225b51132657ebebc294fc976418d0ecc30d7a/hackable_diffusion/kdiff/core.py)
[Multimodal composition — `multimodal.py`](https://github.com/google/hackable_diffusion/blob/fb225b51132657ebebc294fc976418d0ecc30d7a/hackable_diffusion/lib/multimodal.py)
[Diffusion networks](https://github.com/google/hackable_diffusion/blob/fb225b51132657ebebc294fc976418d0ecc30d7a/hackable_diffusion/lib/diffusion_network.py)
[Training documentation](https://github.com/google/hackable_diffusion/blob/fb225b51132657ebebc294fc976418d0ecc30d7a/docs/training.md)
[Sampling documentation](https://github.com/google/hackable_diffusion/blob/fb225b51132657ebebc294fc976418d0ecc30d7a/docs/sampling.md)

**Claim labels:** **Observed** = directly established by source. **Inferred** = follows from the architecture but is not stated as a design rule. **Unknown** = source does not establish it.

---

## 1. Core decomposition

The most important architectural choice is that Hackable Diffusion does **not** represent “diffusion” as one giant scheduler/model object.

Its Tier-1 API defines separate contracts for:

```text
CorruptionProcess
CorruptionSchedule

InferenceFn

DiffusionLoss

SamplerStep
SampleFn

StepInfo
DiffusionStep
```

**Observed:** `hd_api.py` deliberately calls these the high-level contracts between diffusion components and leaves concrete schedules, architectures, and algorithms outside that Tier-1 interface.

A `CorruptionProcess` owns:

```python
corrupt(x0, time)
sample_from_invariant(data_spec)
convert_predictions(prediction, xt, time)
```

An `InferenceFn` owns approximately:

```python
prediction = inference(time, xt, conditioning)
```

A loss knows:

```python
loss(predictions, targets, time)
```

while a sampling algorithm owns:

```python
initialize(...)
update(...)
finalize(...)
```

The resulting dependency structure is much closer to:

```text id="hdcore1"
                     training time sampler
                              │
                              ▼
clean data ──→ CorruptionProcess ──→ x_t + target information
                                      │
                                      ▼
                                DiffusionNetwork
                                      │
                                      ▼
                                  prediction
                                      │
                  CorruptionProcess.convert_predictions()
                                      │
                                      ▼
                                DiffusionLoss
```

rather than:

```text
ModelStrategy.do_everything()
```

### Design pressure

> **Are corruption, target construction, model execution, prediction parameterization and loss actually one responsibility?**

Hackable Diffusion's answer is very explicitly **no**.

That lines up rather well with the conclusion we've been reaching from the individual recipes.

---

# 2. It separates three things that are usually all called “scheduler”

This might be the single most useful thing in the repo for your trainer.

Hackable Diffusion distinguishes:

```text id="hdsched1"
CorruptionSchedule
    what α(t), σ(t), etc. mean at arbitrary t

Training TimeSampler
    which t values training examples are drawn at

Sampling TimeSchedule
    which sequence of t values inference traverses
```

These are not the same object.

### Corruption schedule

For example:

```text
CosineSchedule
RFSchedule
LinearDiffusionSchedule
LinearDiscreteSchedule
...
```

defines the mathematical forward process.

### Training time sampler

Training separately decides how to choose `t`.

Available examples include:

```text
UniformTimeSampler
LogitNormalTimeSampler
UniformStratifiedTimeSampler
UnbalancedTimestepSampler
```

**Observed:** even the **shape** of training time is configurable. `UniformTimeSampler.axes` decides which axes get independent time values.

The normal image case might be:

```text id="hdsched2"
image: [B,H,W,C]

t:     [B,1,1,1]
```

but the abstraction does not hard-code “one timestep per batch item.”

### Sampling time schedule

Inference then separately decides which sequence of times to visit:

```text
UniformTimeSchedule
EDMTimeSchedule
...
```

The reverse solver consumes that sequence independently.

### Design pressure

> **Should “scheduler” really be one configuration slot?**

The evidence here says probably not.

At minimum there are three distinct semantics:

> **forward-process parameterization, training-time sampling distribution, and inference-time discretization.**

That feels like one of the strongest things worth stealing conceptually.

---

# 3. The corruption process owns the representation mathematics

**Observed:** Hackable Diffusion supports several fundamentally different corruption domains behind the same small protocol:

```text id="hdcorr1"
GaussianProcess
    continuous Euclidean data

CategoricalProcess
    integer/discrete states

SimplicialProcess
    probability simplex

RiemannianProcess
    manifold-valued data
```

So the generic training orchestration doesn't need to know:

```text
"this is an image"
"this is a language token"
"this is a rotation"
```

It asks the process to:

```text
produce x_t
produce training targets
describe invariant noise
convert model predictions
```

That last operation is particularly interesting.

For a Gaussian process, one model might directly predict:

```text
epsilon
```

but the process can convert that into:

```text id="hdcorr2"
x0
epsilon
score
velocity
v
```

when mathematically possible.

So:

```text
network prediction type
       ≠
loss comparison space
       ≠
sampler-required prediction type
```

The training docs even allow a network predicting epsilon while constructing a loss mathematically equivalent to one in `x0` space.

### Design pressure

> **Should conversion among equivalent target parameterizations live in every model strategy or every loss implementation?**

Hackable Diffusion instead associates that knowledge with the **corruption/process mathematics**.

That seems quite sensible.

---

# 4. The generic training object is surprisingly small

Their Kauldron integration makes the boundary especially clear.

`kdiff/core.py::Diffusion` does essentially:

```python id="hdtrain1"
time = time_sampler(x0)

xt, targets = corruption_process.corrupt(x0, time)

prediction = network(
    time=time,
    xt=xt,
    conditioning=cond,
)

prediction = corruption_process.convert_predictions(
    prediction, xt, time
)

return {
    "output": prediction,
    "target": targets,
    "xt": xt,
    "noise_info": schedule.evaluate(time),
}
```

That's pretty much it.

**Observed:** the actual optimizer/training engine is **Kauldron**, not this diffusion object. Losses are installed separately into the Kauldron context.

So there are really two layers:

```text id="hdtrain2"
Hackable Diffusion
    constructs diffusion computation semantics

Kauldron
    owns generic training machinery
    optimizer
    checkpointing
    distributed execution
    metrics
    context
```

That distinction is interesting for your project because it avoids making the diffusion abstraction become a second Trainer.

### Design pressure

> **Could a training method describe how to derive an optimized result while leaving stepping, accumulation, checkpointing and backend mechanics owned by the generic Trainer?**

Hackable Diffusion is a pretty clean real-world example of that separation.

---

# 5. Targets and predictions are structured information, not one tensor

The common interface uses dictionaries such as `TargetInfo`.

For Gaussian diffusion, target information may contain several mathematically related representations.

For categorical diffusion it includes things like:

```text id="hdtarget1"
x0
logits
is_corrupted
is_unused
```

The network likewise returns something like:

```python
{"epsilon": ...}
```

or:

```python
{"logits": ...}
```

rather than just returning an anonymous tensor whose meaning must be inferred elsewhere.

`convert_predictions()` then establishes additional equivalent representations.

### Design pressure

This supports something we've already seen repeatedly:

> **A model output is not sufficiently described by its tensor shape. Its semantic representation—epsilon, velocity, clean data, logits, score, etc.—needs to survive the execution boundary.**

For a generic trainer, something structurally equivalent to:

```text
Prediction
Target
```

being **typed/semantic collections** rather than bare tensors looks increasingly justified.

Not necessarily dictionaries specifically, but the concept seems solid.

---

# 6. Network architecture is deliberately downstream of process semantics

`StandardDiffusionNetwork` has a fairly narrow concern:

```text id="hdnetwork1"
time
x_t
conditioning
    │
    ├─ optional time rescaling
    ├─ optional input rescaling
    │
    ▼
conditioning encoder
    │
    ▼
backbone
    │
    ▼
declared prediction representation
```

It contains:

```text
backbone_network
conditioning_encoder
prediction_type
input_rescaler
time_rescaler
```

but **not**:

```text
corruption implementation
training timestep distribution
loss
sampling solver
optimizer
```

The library also treats schedule-dependent preprocessing as explicit roles:

```text
InputRescaler
TimeRescaler
```

For example EDM-like preconditioning can therefore affect model input/time representation without redefining the underlying corruption process.

### Design pressure

> **Does model-specific preparation need to mean model ownership?**

Not necessarily.

A transformation can be part of the network-facing representation contract without becoming a separate model component.

This is pretty similar to what FLUX.2 showed us with latent packing/normalization.

---

# 7. Conditioning has its own composition layer

Hackable Diffusion also doesn't make the backbone interpret every raw condition itself.

`StandardConditioningEncoder` independently takes:

```text
diffusion time
class labels
text embeddings
other conditions
```

and routes their embeddings into mechanisms such as:

```text id="hdcond1"
adaptive_norm
cross_attention
concatenate
sum
```

Multiple condition embeddings can then be merged before reaching the backbone.

So conceptually:

```text id="hdcond2"
raw condition A ─→ encoder ─┐
raw condition B ─→ encoder ─┼─→ injection mechanism ─→ backbone
time            ─→ encoder ─┘
```

### Design pressure

> **Can source/encoding and injection location be separate choices?**

That's useful for models where the same condition appears through several routes—for example a text representation injected through cross-attention plus pooled conditioning through AdaLN.

Again, I wouldn't necessarily clone their exact abstraction, but the separation is worth noting.

---

# 8. Multimodality is implemented by lifting the same operations over a data tree

This is probably the coolest part.

Hackable Diffusion represents multimodal state as a JAX PyTree, for example:

```python id="hdmulti1"
{
    "video": video_state,
    "audio": audio_state,
    "text": text_state,
}
```

It then has `Nested*` wrappers that mirror that structure.

Examples include:

```text id="hdmulti2"
NestedProcess
NestedDiffusionLoss
NestedTimeSampler
JointNestedTimeSampler

NestedSamplerStep
NestedTimeSchedule
NestedGuidanceFn
NestedProjectionFn

NestedTimeEmbedder
NestedFlaxLinenInferenceFn
```

So different leaves can get different semantics.

For example:

```python id="hdmulti3"
process = {
    "image": GaussianProcess(...),
    "text": CategoricalProcess(...),
}

loss = {
    "image": GaussianLoss(...),
    "text": DiscreteLoss(...),
}

prediction_type = {
    "image": "velocity",
    "text": "logits",
}
```

The generic operations are mapped leaf-wise.

This means **multimodality is primarily composition of independently meaningful representations**, rather than a hard-coded list of allowed combinations.

That's strikingly close to what H3 was telling us.

---

# 9. Modalities don't even have to share the same diffusion time

There are actually several choices.

### Independent modality times

`NestedTimeSampler` gives each leaf its own key/sampler:

```text id="hdtime1"
video → t_video
audio → t_audio
text  → t_text
```

### Correlated/joint times

`JointNestedTimeSampler` intentionally calls the modality samplers with the **same random key**, allowing correlated samples.

And the library contains an even more specialized example, `UnbalancedTimestepSampler`, based on JointDiT. It jointly constructs two modality times and sometimes enforces a relationship:

```text id="hdtime2"
with probability p_equal:

t₂ = 1 - t₁
```

rather than simply saying:

```text
t_image == t_depth
```

### This is an important nuance

The `Nested*` mechanism handles **structural composition**, but cross-modal relationships sometimes require a specialized object.

That's actually a useful warning for us.

It would be easy to overlearn:

```text
just make everything modality dictionaries
```

but H3 already showed:

```text
video t
   ↓ nonlinear mapping
audio t
```

and Hackable Diffusion itself needs `UnbalancedTimestepSampler` when the modalities' times have a more complex relationship.

### Design pressure

> **Can modalities own different state while the recipe still defines relationships across those states?**

A pure per-modality plugin model isn't sufficient by itself.

---

# 10. Multimodal network output follows the media structure too

`MultiModalDiffusionNetwork` assumes the PyTree structures of:

```text id="hdmulti4"
x_t
time
prediction_type
data_dtype
input rescaler
time rescaler
```

correspond.

A shared backbone can return a tree of outputs, after which each leaf is tagged with its own prediction semantics.

So something like:

```text id="hdmulti5"
input
{
  image: ...
  label: ...
}

       ↓ shared/joint backbone

{
  image: {"velocity": ...},
  label: {"logits": ...},
}
```

is valid.

This is different from simply training two independent diffusion models beside one another.

The **data/process semantics are per-leaf**, while the **neural computation may still be joint**.

That distinction matters a lot.

### Design pressure

> **Per-modality representation ownership should not imply per-modality model ownership.**

Exactly the same issue we saw with MiniMax-H3.

---

# 11. Self-conditioning is modeled as an execution pattern

Their `SelfConditioningDiffusionNetwork` explicitly performs:

```text id="hdsc1"
x_t + zero prediction
       ↓
forward #1
       ↓
prediction
       ↓ stop_gradient
optionally replace with zero
       ↓
x_t + detached prediction
       ↓
forward #2
       ↓
optimized prediction
```

During training the previous prediction is used probabilistically; during inference self-conditioning is always applied.

The nested version performs the same operation across a PyTree of discrete modalities.

This is useful because self-conditioning is **not a second model**.

It's a particular multi-forward execution graph involving the same model.

### Design pressure

> **Can training methodology request repeated execution of the same component with transformed/detached results from an earlier execution?**

That's the exact problem we found independently with DiffusionGemma and teacher/student recipes.

---

# 12. Sampling is split into orchestration, model prediction, and numerical transition

Sampling has a similarly clean decomposition:

```text id="hdsample1"
TimeSchedule
     │
     ▼
DiffusionSampler
     │
     ├── calls InferenceFn
     │
     └── calls SamplerStep
                 │
                 ▼
             next state
```

`DiffusionSampler` doesn't need to know how the model works.

`InferenceFn` doesn't need to know how the next diffusion state is calculated.

`SamplerStep` doesn't need to own the neural model.

That allows things such as:

```text id="hdsample2"
same trained model
    +
DDIM

same trained model
    +
SDE step

same trained model
    +
Heun
```

where mathematically compatible.

### `DiffusionStep`

The changing reverse-process state is explicit:

```text
xt
StepInfo
aux
```

and `StepInfo` separately contains precomputable/static information:

```text
step number
time
RNG
```

That's another decent ownership distinction:

> **static trajectory plan vs dynamic trajectory state.**

---

# 13. Inference is also compositional rather than synonymous with “model.forward”

`InferenceFn` means:

> produce the prediction needed for one reverse-process step.

It can be just the model.

But the provided guided implementation is:

```text id="hdinfer1"
conditional model call ─┐
                        ├→ guidance → projection → prediction
unconditional model call┘
```

So one `InferenceFn` invocation can itself mean **multiple physical forwards**.

Guidance and projection remain independent components.

### Design pressure

> **A logical “prediction” operation does not necessarily correspond to exactly one neural forward.**

This keeps showing up across everything we've researched.

Teacher/student, CFG, self-conditioning, DiffusionGemma—all break a one-forward-per-step abstraction.

---

# 14. Sampling conditioning can evolve during the trajectory

`DiffusionSampler` also accepts an optional:

```text
update_conditioning_fn
```

which is invoked from the current diffusion state before a prediction.

So conditioning is not necessarily immutable input metadata established at the beginning of generation.

It can conceptually be:

```text id="hdupdate1"
current diffusion state
       ↓
update condition
       ↓
model prediction
       ↓
next diffusion state
```

### Design pressure

> **Can conditioning be derived or updated from current execution state rather than treated as immutable batch input?**

That's another useful edge case for a general execution model.

---

# 15. Autoregressive diffusion is implemented one level above ordinary diffusion

Their newer `AutoregressiveDiffusionSampler` is particularly interesting after looking at DiffusionGemma.

It does not modify the generic diffusion sampler.

Instead:

```text id="hdar1"
AR state
   ↓
initialize noisy canvas
   ↓
ordinary DiffusionSampler
   ↓
finished canvas
   ↓
ARStateHandler.update_ar_state()
   ↓
next canvas
```

The generic loop knows nothing about Gemma or KV caches.

Model-specific state management is delegated to:

```text id="hdar2"
ARStateHandler

init_ar_state()
create_conditioning_from_state()
update_ar_state()
finalize_ar_state()
```

For DiffusionGemma, the concrete handler interprets that state as prompt/KV-cache/cursor state.

For another model it could mean something else.

### Design pressure

This is a good example of **composition above a complete algorithm**:

> **A new generation topology does not necessarily require making the lower-level sampler understand the new model.**

That's probably worth remembering before adding special cases to a generic Trainer too.

---

# 16. Early stopping belongs to control flow, not the model

The standard `DiffusionSampler` uses a fixed `jax.lax.scan`.

A separate `DiffusionSamplerWithEarlyStopping` uses a `while_loop` and tracks a per-example `done` mask.

Completed batch elements are frozen while unfinished elements continue.

So:

```text id="hdearly1"
sample A done at step 28 ─── frozen
sample B still active     ─── continues
sample C done at step 31 ─── frozen
```

This is another separation:

```text
model prediction
≠
trajectory termination policy
```

Again, useful if a future Trainer ever has iterative inner procedures.

---

# 17. What I would *not* copy blindly

There are a few important limits to this design.

**Observed:** much of multimodality assumes that process/loss/time/prediction configuration follows the same PyTree structure as the data. That's elegant when the relationship really is leaf-wise.

But some recipes we've already examined are more relational:

```text id="hdlimit1"
H3:
video time → mapped audio time

YuE2:
AR execution → KV cache → detached → NAR execution

teacher/student:
teacher output → adapter/common comparison space → student loss
```

Those aren't naturally just:

```python
jax.tree.map(fn, modalities)
```

Hackable Diffusion itself effectively acknowledges this with specialized cross-modal time samplers and DiffusionGemma's custom SFT wrapper.

So I wouldn't conclude:

> “Everything should become a nested modality tree.”

A safer conclusion is:

> **Tree composition works well for independent or structurally corresponding operations; recipes still need somewhere to express relationships across leaves, forwards, components and states.**

There's another limitation: `StandardDiffusionNetwork` is still oriented around a fairly conventional:

```text
x_t + time + conditioning → backbone → prediction
```

When DiffusionGemma needed causal prefill + KV state + two denoiser passes + AR loss, Google **built a custom `SFTDiffusion` orchestration object** rather than squeezing it into the generic simple path.

I actually think that's a positive signal.

It suggests the base abstractions are allowed to stop being universal when a recipe genuinely has a richer execution graph.

---

# 18. The architecture pattern I think is most relevant to your trainer

I wouldn't copy their class hierarchy, but there are four separations here that look unusually robust across everything we've researched.

```text id="hdsummary1"
REPRESENTATION / PROCESS
    What is x?
    How is it corrupted?
    What target representations exist?
    How do prediction forms convert?

EXECUTION
    Which component runs?
    How many times?
    With what inputs / conditioning / gradient boundary?

OBJECTIVE
    Which outputs and targets are compared?
    In what representation?
    How are losses combined?

OPTIMIZATION / TRAINER
    Which parameters update?
    Which optimizer owns them?
    accumulation / distributed / checkpoint / lifecycle
```

Hackable Diffusion has especially strong evidence for keeping the first three distinct.

And the most useful thing it adds to our existing research is probably this:

> **“Training strategy” should not have to own every piece of diffusion mathematics. Reusable process-level concepts—corruption, target conversion, timestep sampling, prediction semantics and reverse-step algorithms—can exist independently, while the recipe owns how they are composed for a particular training method.**

That seems like a much better direction than either extreme:

```text id="hdsummary2"
bad extreme A:
Trainer knows DDPM / flow / discrete diffusion / EDM / etc.

bad extreme B:
ModelStrategy knows literally everything.

more promising:
Trainer
    generic execution + optimization machinery

Recipe / objective
    composition and relationships

Process / representation capabilities
    reusable mathematical operations

Model strategy
    model-specific execution and representation adapters
```

I would **not turn that into an API yet**, though. The value of Hackable Diffusion here is that it gives us evidence that those boundaries can work in a real codebase—not that their exact protocols are automatically the right ones for `sd-scripts`.

One especially good constraint to preserve in the architecture notes would be:

> **Do not collapse corruption schedule, training timestep sampling, and inference timestep scheduling into one “scheduler” abstraction. They describe different decisions and may vary independently.**

And another:

> **Do not require multimodality to imply multiple models. Per-modality process, time, loss, representation and prediction semantics may coexist around one jointly executing backbone.**

Those two are probably the biggest new contributions from this research.
