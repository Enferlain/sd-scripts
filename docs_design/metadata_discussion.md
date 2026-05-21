# Discussion about metadata

## About metadata system shape (the shape appears as the conversation progresses towards the end)

Agent:
We probably don’t need two files at this stage. The split made architectural sense on paper: one file builds training metadata facts, the other adapts facts into provider/projection export. But in practice, both are small, both are training-checkpoint-facing, and the names are too similar. So it creates mental overhead without buying us much yet.

I’d merge `library/training/metadata_providers.py` into `library/training/metadata.py` for now. Then `library/training/metadata.py` becomes the single training metadata home:

```text
library/training/metadata.py
  build training run facts
  hold TrainingMetadataState
  define training metadata providers
  project checkpoint metadata
```

Later, if it grows too large, we split by clearer concerns, not by vague names. For example:
- `library/training/run_metadata.py`
- `library/training/checkpoint_metadata.py`
- `library/training/metadata_export.py`

But right now? One file is friendlier and easier to follow.

User:
yeah I do not want to let the system get to a stage where it needs different files just for the metadata in just one corner of the library

Agent:
Collapsed it into one training metadata module.

`library/training/metadata.py` now owns:
- training metadata build context/bundle/state
- training-owned provider wrappers
- checkpoint metadata projection helper

Removed:
- `library/training/metadata_providers.py`

Updated:
- test import in `tests/unit/training/test_metadata_providers.py`
- docs/changelog/roadmap references so they don’t point future work back toward a split module

Verification:
- `ruff check` passed.
- `ty check` passed.
- Focused provider test assertions passed (`2 passed`), though pytest still hit the WSL wrapper timeout during process teardown.

Closed bead `sd-scripts-bvk`.

User:
Where did you get this idea from in the first place? did anything we talk about point to this?

Agent:
It came from my interpretation, not from something you explicitly asked for.

The chain in my head was:

`main metadata backend <-> adapter layer <-> domain/family metadata surfaces`

So I mapped that into “domain builds facts” plus “provider wrappers feed the backend/projections.” That made me split:

- `training/metadata.py` for training facts/state
- `training/metadata_providers.py` for provider/backend/projection glue

But that was me over-applying the backbone architecture inside one small domain. What we talked about pointed to a **provider concept at the system level**, not necessarily separate provider files per corner of the repo. You were also pretty clear that this should stay friendly and config/dataclass-like, and that domains should own their metadata. A single `library/training/metadata.py` matches that better right now.

So, short answer: no, the two-file split was not directly asked for. It was an architectural extrapolation from the backend/provider/projection design. In this repo, at this stage, it was too much ceremony.

User:
Can you explain a bit more? From what I remember I said that metadata would live in library/metadata/dataclasses/concern.py so they can be followed easier and match the existing config system we have.

You said your chain was "main metadata backend <-> adapter layer <-> domain/family metadata surfaces", but that is about the adapter metadata specifically

Agent:
You’re right. I conflated two different ideas.

What you described was mostly:

`library/metadata/dataclasses/<concern>.py`

Meaning the **metadata shapes/facts** should live in one easy-to-follow metadata area, similar to the config dataclasses system. That gives us discoverability and keeps the schema readable.

The chain I cited:

`main metadata backend <-> adapter layer <-> adapter family metadata surfaces`

was specifically about **adapter metadata**, not a general mandate to make every domain grow local provider files. I overgeneralized it into a broader “each domain has provider glue near itself” pattern. That’s where the extra `library/training/metadata_providers.py` came from.

A cleaner interpretation of what we agreed on is:

```text
library/metadata/
  dataclasses/
    run.py
    model.py
    artifact.py
    future_adapter_method_or_family.py
  backends.py
  providers.py
  projections.py
  keys.py

library/training/
  metadata.py maybe only orchestration/assembly if needed
```

So the “canonical shapes” should be under `library/metadata/dataclasses/`, not scattered into `library/training/` as little metadata sub-systems. Training can assemble or provide runtime context, but it shouldn’t become another schema home.

Where I think the current code is still slightly off: `TrainingMetadataState` and the provider wrappers live in `library/training/metadata.py`. That’s better than two training metadata files, but it may still be more training-local than your original direction. A stricter version would move the stable shapes/providers closer to `library/metadata/`, and leave training with only the builder that gathers facts from `cfg`, manifests, and strategy hooks.

User:
That's not what I said either. Adapter method/family metadata is local, not central

Agent:
You’re right again. I misstated it.

The correct version is:

- **Shared/stable cross-cutting shapes** can live in `library/metadata/dataclasses/<concern>.py`.
- **Adapter method/family metadata** should stay local to the adapter method/family, because it is not shared and should not be centralized.
- The metadata backbone should provide the infrastructure: records, keys, projections, validation, backend/storage, provider contracts.
- Domain/family/method code owns the metadata details that are actually domain/family/method-specific.

So for adapter metadata specifically, it should look more like:

```text
library/metadata/
  dataclasses/
    run.py
    model.py
    artifact.py

library/adapters/methods/peft/vera/
  metadata.py   # VeRA-owned metadata facts/keys/provider if needed

library/adapters/methods/peft/lora/
  metadata.py   # LoRA-owned metadata if needed
```

Not:

```text
library/metadata/dataclasses/adapter.py
```

That was exactly why we deleted the generic adapter dataclass earlier.

For the current training work, the key question is whether `TrainingMetadataState` / provider wrappers are shared enough to belong in `library/metadata`, or training-owned enough to stay in `library/training/metadata.py`. I think they are training-owned assembly/state, so local is okay. But the stable fact dataclasses they use, like run/model/artifact, belong in `library/metadata/dataclasses/`.

User:
okay can you please clearly explain what the issue is

Agent:
The issue is that I blurred the ownership boundaries.

What you wanted was:

```text
library/metadata/
  backbone infrastructure
  shared dataclasses for truly shared concerns

domain/family/method modules
  own their specific metadata details
```

So:

```text
library/metadata/dataclasses/run.py
library/metadata/dataclasses/model.py
library/metadata/dataclasses/artifact.py
```

is fine because those are shared concerns.

But:

```text
library/metadata/dataclasses/adapter.py
```

was wrong because adapter metadata is not one shared concern. It belongs to adapter methods/families locally.

Then I made a second mistake: I over-applied the “provider/backend” architecture inside training and created:

```text
library/training/metadata.py
library/training/metadata_providers.py
```

That was not directly grounded in your requested shape. It made one small repo corner feel like it had its own mini metadata subsystem. We already fixed that by merging back into one `library/training/metadata.py`.

The remaining design tension is:

`library/training/metadata.py` currently contains both:
- training-owned assembly/state, which is okay locally
- provider wrappers for run/model/artifact facts, which may or may not be too generic-looking

So the concrete issue is not “metadata.py exists.” The issue is making sure we do not accidentally centralize local metadata or scatter tiny metadata subsystems everywhere. Shared shapes go central. Local details stay local. Training should only assemble training-run/checkpoint metadata, not become a second metadata framework.

Agent:
I don't think you're clear on what the goal is, or maybe we're not clear on what's possible with the plans written swo far

User:
You’re right. The problem is not just implementation drift; the written plan itself now has mixed signals.

The clearest issue I see is this:

The design says two things that can conflict if we do not define boundaries tightly:

1. `library/metadata/` should be the repo-owned backbone, with shared dataclasses/protocols/projections/storage.
2. Domains/families/methods should own their metadata details.

That means we need to define what counts as **shared backbone shape** versus **domain-owned metadata**. Without that line, I keep making guesses, and those guesses can create weird structures like extra training metadata files or centralizing something that should stay local.

A better clarified goal might be:

`library/metadata/` owns:
- generic record/event/edge/backbone infrastructure
- shared projection machinery
- shared keys/namespaces
- dataclasses only for genuinely cross-domain concepts, like run, artifact, model identity, maybe provenance

Domain/local code owns:
- training-run assembly from config/manifests/runtime
- SD/SDXL/SD3-specific model facts
- adapter method/family metadata
- cache/data sample facts
- optimizer/logging-specific metadata facts

So the issue is that “metadata dataclasses live in `library/metadata/dataclasses/concern.py`” is true only for concerns that are actually shared/repo-level. It should not mean every metadata thing gets centralized. And “provider layer” is an integration pattern, not a reason to split every domain into provider files.

I think before more code work, we should rewrite the metadata design into a short ownership contract:

- What belongs in `library/metadata/`
- What must stay local
- What shape providers should have
- When a local metadata file is justified
- How compatibility keys like `ss_*` and `modelspec.*` relate to typed facts

That would give us a stable rulebook instead of continuing to infer from partially conflicting notes.

User:
These 2 things can't be true at the same time. Something can't be "domain" owned and also be in library/metadata/dataclasses, and having 2 different things at the same time is confusing.

"shared" is not a word we're using in the metadata work. The idea was to do the same thing that config dataclasses are doing. either every metadata item dataclass that is recorded is kept in library/metadata/dataclasses/concern.py or none of it is. Right now the only exception is adapters because that part of the repo genuinely functions as a plugin facet. Maybe you can argue model implementations fall under this, and that could be a discussion, but we need to actually have them. And I'm still not 100% on board with having metadata.py in every folder all over the repo yet.

Agent:
You’re right. My “domain-owned but in `library/metadata/dataclasses`” framing is contradictory, and it’s the source of the confusion.

The cleaner rule is:

`library/metadata/dataclasses/<concern>.py` is the canonical schema catalog for recorded metadata dataclasses, like `library/config/dataclasses/`.

So if a metadata item is part of the repo’s recorded metadata system, its dataclass belongs there. Not “shared only,” not “domain-owned but central.” Just: recorded metadata schema lives in the metadata dataclass catalog.

Then the exceptions need to be explicit, not fuzzy:
- Adapter methods can own local metadata because adapters behave like plugin facets.
- Model implementations might also qualify later, but only after we make that decision deliberately.
- We should not casually create `metadata.py` in every folder.

That means the current direction should be corrected to something like:

```text
library/metadata/
  dataclasses/
    run.py
    artifact.py
    model.py
    training.py maybe, if training-specific recorded facts exist
    data.py later
    cache.py later
    optimization.py later
  providers/projections/backends/etc.
```

And local training code should not become a separate metadata mini-system. If it only gathers runtime inputs, that can be a collector/builder concern, but the recorded dataclasses should still live under `library/metadata/dataclasses/`.

So the issue is: I introduced a “local metadata ownership” interpretation that does not match the config-dataclass analogy you were aiming for. We should update the docs and then adjust the code toward one consistent rule.

User:

Other agent's input:

You’re pointing at the core contradiction correctly:

> “Dataclasses live in `library/metadata/dataclasses/<concern>.py` like config”
>
> and
>
> “domains own their metadata details”

Those cannot both mean “metadata dataclasses live in both places.” That creates exactly the confusion you hit: `library/training/metadata.py`, `metadata_providers.py`, maybe later `library/data/metadata.py`, `library/optimization/metadata.py`, etc. Then suddenly the metadata system is scattered all over the repo, while pretending to be centralized.

The cleaner rule should be:

```text
Recorded metadata dataclasses live in library/metadata/dataclasses/.
Domain code gathers/builds/emits those dataclasses.
```

So training can own **assembly logic**, but not its own little metadata schema island.

Example:

```text
library/metadata/dataclasses/
  run.py
  artifact.py
  model.py
  checkpoint.py
  cache.py
  optimization.py
  observability.py

library/training/metadata.py
  gathers cfg/runtime/manifest facts
  builds central metadata dataclasses
  asks backend/projection layer to export checkpoint metadata
```

That means `library/training/metadata.py` is allowed to exist only as a **producer/orchestrator**, not as a second place where “training metadata types” are invented independently.

The earlier `metadata_providers.py` split was bad because it made training look like it had its own mini metadata subsystem:

```text
training metadata facts here
training metadata providers there
generic metadata elsewhere
```

That is too much for one corner of the repo, and it weakens the whole “metadata is like config dataclasses” idea.

The wording that caused the damage is probably **“domain-owned metadata facts.”** That should be rewritten. Better wording:

```text
Domains own metadata production.
library/metadata owns recorded metadata schemas.
```

or even stricter:

```text
A metadata item that is recorded by the repo metadata backend must have its dataclass in library/metadata/dataclasses/.

Domain/family/method code may collect facts and construct those records, but it should not define parallel recorded metadata dataclasses locally, unless the area is explicitly plugin-like.
```

Then exceptions have to be named, not implied.

Current reasonable exceptions:

```text
Adapter methods:
  local metadata is okay because adapter methods behave like plugin facets.

Maybe model families:
  only if model families are treated as plugin-like implementations too,
  but that should be an explicit decision, not accidental drift.
```

So I’d update the design around this ownership contract:

### `library/metadata/` owns

```text
dataclasses for recorded metadata items
backend
record/event/edge concepts
validation
projection machinery
storage interfaces
key namespace helpers
```

### domain code owns

```text
fact collection
runtime context gathering
calling provider/backend APIs
constructing central dataclasses
domain-specific validation before emission
```

### domain code does not own, by default

```text
separate recorded metadata schemas
parallel metadata dataclass trees
extra metadata_providers.py files just because providers exist
```

And yeah, I’d avoid the word **shared** in this work. It makes the boundary fuzzy. The better axis is:

```text
central recorded schema
vs
local producer/plugin implementation
```

For the immediate situation, I think the correction is:

1. Keep `library/training/metadata.py` only if it is just training-side assembly/export orchestration.
2. Do not recreate `metadata_providers.py`.
3. Move any stable recorded metadata dataclasses out of training and into `library/metadata/dataclasses/...`.
4. Make the design doc explicitly say local `metadata.py` files are not the default pattern.
5. List adapters as a real exception because they are plugin-like method implementations.

The short version of the problem: **the implementation treated “domains own producing metadata” as “domains own metadata schema files.” That is not the same thing, and it fights the config-dataclass-style goal.**

No — **not `metadata.py` in every folder by default.**

The current design doc accidentally points in that direction because it lists examples like:

```text
library/training/metadata.py
library/models/sdxl/metadata.py
library/adapters/methods/peft/vera/metadata.py
library/data/metadata.py
library/optimization/metadata.py
library/logging/metadata.py
```

and says domain-owned provider implementations can live in their own packages. 

That’s the part I think needs to be corrected, or at least heavily constrained.

A cleaner rule would be:

```text
library/metadata/
  owns the metadata system:
  - dataclasses for recorded metadata items
  - backend
  - validation
  - projections
  - storage
  - provider contracts

normal domain folders:
  do not get metadata.py by default

plugin-like / genuinely isolated implementation folders:
  may get local metadata.py only when needed
```

So:

```text
library/training/metadata.py
```

should probably **not** become the start of a pattern where every area gets its own metadata module. At most, training might have a small orchestration/helper file while the actual recorded dataclasses live centrally.

Better target shape:

```text
library/metadata/dataclasses/
  run.py
  artifact.py
  checkpoint.py
  model.py
  cache.py
  optimization.py
  observability.py

library/metadata/
  backend.py
  validation.py
  projections/
  storage/
  providers/
```

Then training code just builds/emits those central records.

The exception is adapters/methods because they are closer to plugins. For example:

```text
library/adapters/methods/peft/vera/metadata.py
```

could be okay if VeRA has method-local artifact reconstruction facts that should not pollute the central metadata dataclasses. Same maybe for model families, but only if you explicitly decide model families are plugin-like too.

So the answer is: **no, not metadata.py everywhere.** The design doc currently implies that too much, and that’s probably the thing to fix before implementation drifts into “metadata soup.” The doc already says the backend should not know model-family or PEFT internals, and providers should not become arbitrary dict dumping grounds, so the stricter version is very compatible with the existing intent.

Agent:
Updated the docs/specs to use the cleaner ownership contract:

```text
library/metadata/dataclasses/
  owns recorded metadata schemas

domain code
  owns production/emission: gathering facts, constructing central dataclasses,
  validating local invariants, calling provider/backend APIs

local metadata schemas
  not default; only explicit plugin-like exceptions, currently adapter methods
```

I changed:
- [metadata_system_design.md](/mnt/d/Projects/sd-scripts/docs_design/metadata_system_design.md:42)
- [metadata_legacy_migration_control.md](/mnt/d/Projects/sd-scripts/docs_design/metadata_legacy_migration_control.md:31)
- [openspec metadata-backbone design.md](/mnt/d/Projects/sd-scripts/openspec/changes/metadata-backbone/design.md:1)
- [openspec metadata-backbone spec.md](/mnt/d/Projects/sd-scripts/openspec/changes/metadata-backbone/specs/metadata-backbone/spec.md:1)
- [openspec metadata-backbone tasks.md](/mnt/d/Projects/sd-scripts/openspec/changes/metadata-backbone/tasks.md:1)
- `CHANGELOG.md`

Also removed/rewrote the fuzzy wording around “shared” and “domain-owned facts,” and added explicit language that normal domain folders should not get `metadata.py` just because metadata exists.

Verification: `ruff check` passed for the touched docs. Closed bead `sd-scripts-0hc`.

User:
What I'm trying to say makes sense no? Like having a catalog like items but usage is local to the concern. or is there anything that looks like a hurdle to you with this idea?

Agent:
Yes, it makes sense. The clean mental model is:

```text
Central catalog:
  What recorded metadata shapes exist?

Local usage:
  Who knows how to fill those shapes?
```

That is very close to the config system:

```text
library/config/dataclasses/
  schema catalog

training/data/models/etc.
  consume/build/use config objects
```

For metadata:

```text
library/metadata/dataclasses/
  recorded metadata schema catalog

training/data/models/adapters/etc.
  gather facts, construct catalog dataclasses, emit them
```

The main hurdle is not conceptual. It’s just boundary discipline.

Potential hurdles:

- Some metadata is currently just exported compatibility strings, like `ss_*` and `modelspec.*`. We need to decide which of those become real catalog dataclasses versus projection-only compatibility keys.
- Some areas are plugin-like, especially adapter methods. If they need method-private persistence facts, centralizing every tiny method fact could make the catalog noisy. Your adapter exception handles that.
- Model families may be borderline. SD/SDXL/SD3 are repo-owned today, so central catalog is fine unless we intentionally treat model families as plugins later.
- The central catalog can get large. That’s okay if it’s organized by concern, but we should avoid huge grab-bag files.
- We need to avoid putting producer logic in the catalog. Dataclasses should stay mostly shape/validation/conversion, not “go inspect the trainer/model/cache.”

So I think your idea is strong. The rule I’d use is:

```text
If the repo metadata backend records it as a durable/queryable concept,
its dataclass belongs in library/metadata/dataclasses/.

If a domain merely knows how to observe or produce it,
that production logic stays local.

If a plugin-like implementation has private persistence metadata,
it may keep a local schema.
```

That’s coherent and workable.

User:
well compatibility is a utility/miscellaneous concern, not a first party metadata thing. you're going out of your way to support or keep track of something that's not an official thing you're doing. but even then it's probably still clearest if they are in a visible place and not hidden somewhere

adapters is an exception, because there is broad adapter level metadata concerns, and family or method level concerns. it's abstraction based. for models the same thing applies to configs and metadata too really, at least from where I am now, because it's not an integral part of how the repo like training/optimization/data etc

model families having central catalog doesn't make much sense

what's producer logic

Agent:
Yes, that distinction helps.

Compatibility metadata is not “first-party metadata.” It is a projection/export utility concern. So we should not design the central catalog around `ss_*` or `modelspec.*` as if those are the native schema. But the compatibility mapping should still be visible, probably under something like:

```text
library/metadata/compatibility/
library/metadata/projections/
library/metadata/keys.py
```

Not hidden inside training or model helpers.

On adapters/models: your abstraction point makes sense. They have layered concerns:

```text
adapter-level metadata
adapter family/method metadata

model-level metadata
model family/implementation metadata
```

So model families may also be an exception like adapter methods, because they are implementation facets rather than core training/data/optimization concerns. That means central catalog should probably cover repo-level model/artifact concepts, but not every SDXL/SD3-specific field.

By “producer logic” I mean the code that **fills metadata values from real runtime/context**.

Example dataclass/catalog shape:

```python
@dataclass
class RunMetadata:
    session_id: str
    started_at: float
    seed: int
```

Producer logic is the code that knows:

```python
RunMetadata(
    session_id=str(trainer.session_id),
    started_at=trainer.training_started_at,
    seed=cfg.training.seed,
)
```

So:
- The dataclass is the recorded shape.
- Producer logic is the local code that gathers values from `cfg`, `trainer`, manifests, adapters, models, cache entries, etc.

Producer logic should stay near the concern because it knows the runtime objects. The catalog should not import `Trainer` or inspect manifests itself if we can avoid it.

User:
Please show me in code the result of the previous wording vs the new one

Agent:
Previous wording led to this kind of shape:

```python
# library/training/metadata.py

@dataclass(frozen=True)
class TrainingRunFacts:
    session_id: str
    seed: int
    optimizer: str

class TrainingRunMetadataProvider:
    def collect(self) -> TrainingRunFacts:
        ...

def build_checkpoint_metadata(...):
    ...
```

```python
# library/data/metadata.py

@dataclass(frozen=True)
class SampleFacts:
    image_path: str
    caption: str
    bucket: tuple[int, int]

class DataMetadataProvider:
    def collect(self) -> SampleFacts:
        ...
```

```python
# library/optimization/metadata.py

@dataclass(frozen=True)
class OptimizerFacts:
    optimizer_name: str
    scheduler_name: str

class OptimizationMetadataProvider:
    def collect(self) -> OptimizerFacts:
        ...
```

That is the “domain owns metadata facts” interpretation. It spreads recorded schemas into every corner.

The corrected wording points more like this:

```python
# library/metadata/dataclasses/run.py

@dataclass(frozen=True)
class RunMetadata:
    session_id: str
    started_at: float
    seed: int
```

```python
# library/metadata/dataclasses/data.py

@dataclass(frozen=True)
class SampleMetadata:
    image_path: str
    caption: str
    bucket_resolution: tuple[int, int]
```

```python
# library/metadata/dataclasses/optimization.py

@dataclass(frozen=True)
class OptimizerMetadata:
    optimizer_name: str
    scheduler_name: str
```

Then local code only fills those central shapes:

```python
# library/training/metadata.py or a training setup module

from library.metadata.dataclasses.run import RunMetadata

def build_run_metadata(trainer, cfg) -> RunMetadata:
    return RunMetadata(
        session_id=str(trainer.session_id),
        started_at=trainer.training_started_at,
        seed=cfg.training.seed,
    )
```

```python
# library/data/manifest.py or a data producer module

from library.metadata.dataclasses.data import SampleMetadata

def sample_metadata_from_entry(entry) -> SampleMetadata:
    return SampleMetadata(
        image_path=entry.image_path,
        caption=entry.caption,
        bucket_resolution=entry.bucket_resolution,
    )
```

Compatibility stays visible but not native:

```python
# library/metadata/projections/kohya_ss.py

def project_run_to_ss(metadata: RunMetadata) -> dict[str, str]:
    return {
        "ss_session_id": metadata.session_id,
        "ss_seed": str(metadata.seed),
    }
```

Plugin-like exception:

```python
# library/adapters/methods/peft/vera/metadata.py

@dataclass(frozen=True)
class VeraProjectionBankMetadata:
    save_projection: bool
    projection_prng_key: int
```

So the practical difference is:

```text
Old wording:
  library/training/metadata.py defines TrainingRunFacts

New wording:
  library/metadata/dataclasses/run.py defines RunMetadata
  library/training/... only builds/emits RunMetadata
```

User:
if we have central dataclasses for things, we still don't know where the code that uses/fills them goes in each place

Agent:
Exactly. The central catalog only answers **what shape exists**. It does not answer **where construction happens**.

We need a second rule for usage/filling code.

A practical rule could be:

```text
The code that fills a metadata dataclass lives at the nearest place that already owns the source values.
```

Examples:

```text
Run/session metadata
  source values: Trainer/session/cfg
  builder location: training orchestration area

Dataset/sample metadata
  source values: CacheEntry/manifest/scanner
  builder location: data pipeline area

Optimizer metadata
  source values: optimizer plan/scheduler runtime
  builder location: optimization area

Artifact/checkpoint metadata
  source values: checkpoint save request, path, step, epoch
  builder location: checkpoint/save boundary

Compatibility projection
  source values: central metadata dataclasses
  builder location: metadata projection/compatibility package
```

Code shape:

```python
# central shape
# library/metadata/dataclasses/run.py

@dataclass(frozen=True)
class RunMetadata:
    session_id: str
    started_at: float
    seed: int
```

```python
# local producer
# library/training/run_metadata.py maybe, or library/training/metadata_builders.py

def build_run_metadata(trainer: Trainer) -> RunMetadata:
    return RunMetadata(
        session_id=str(trainer.session_id),
        started_at=trainer.training_started_at,
        seed=trainer.cfg.training.seed,
    )
```

The file name question is separate. To avoid `metadata.py` everywhere, we can name producer files by lifecycle/action instead of by “metadata”:

```text
library/training/run_context.py
library/training/checkpointing.py
library/data/manifest.py
library/optimization/optimizer_factory.py
library/logging/reports.py
```

So not every domain needs a new metadata file. Often the builder can live in an existing module that already owns the lifecycle.

Rule of thumb:

```text
If metadata construction is one small part of an existing lifecycle, keep it in that lifecycle module.

If metadata construction becomes large enough to obscure that lifecycle, extract a named producer module:
  run_metadata_builder.py
  checkpoint_metadata_builder.py
  cache_metadata_builder.py

Avoid generic local metadata.py unless it is truly the domain’s metadata orchestration file.
```

---

Agent:
Yes, this fits the repo better than the previous “local producers everywhere” wording.

The important tweak is: **central collectors should not become god-code that deeply knows every domain.** They should be thin translation code from existing repo objects into central metadata dataclasses.

I’d describe the shape as:

```text
library/metadata/dataclasses/
  what recorded metadata exists

library/metadata/collectors/
  how repo runtime objects are translated into those records

library/metadata/projections/
  how records become ss_ / modelspec. / kuro. / safetensors metadata

domain code
  owns the source objects and calls collectors at lifecycle boundaries
```

So the current messy file:

```text
library/training/metadata.py
```

is doing too many things at once:

```text
training runtime context
training metadata state
run fact construction
dataset ss_* compatibility collection
adapter ss_* compatibility collection
model source ss_* compatibility collection
checkpoint projection assembly
provider wrappers
```

Under the cleaner split, I’d expect something closer to:

```text
library/metadata/collectors/run.py
  collect_training_run_metadata(...)

library/metadata/collectors/data.py
  collect_dataset_metadata_from_manifest(...)

library/metadata/collectors/optimization.py
  collect_optimizer_metadata_from_plan(...)

library/metadata/collectors/artifact.py
  collect_checkpoint_artifact_metadata(...)

library/metadata/projections/kohya_ss.py
  ss_* compatibility export rules

library/metadata/projections/modelspec.py
  modelspec.* export rules

library/metadata/projections/kuro.py
  kuro.* export rules
```

The main hurdle is import direction. I would avoid collectors that take the whole `Trainer` unless we really need it, because then `library/metadata` starts depending on all of training. Better:

```python
collect_training_run_metadata(
    cfg=cfg,
    manifest=train_manifest,
    session_id=session_id,
    started_at=training_started_at,
    optimizer_name=optimizer_name,
    ...
)
```

or a narrow context dataclass. That keeps the collector central without making it omnivorous.

So my take: **central dataclasses + central collectors is probably the right correction.** Local `metadata.py` files should be rare. Adapter methods and maybe model-family implementations can still have local hooks because they are plugin/facet-like, but normal training/data/optimization should call central collectors rather than invent local metadata islands.

User:
it's either this or metadata.py everywhere. If we still need them, then the collectors idea doesn't really serve a purpose

Agent:
The collectors idea only serves a purpose if we make it a real rule:

```text
No normal domain-level metadata.py files.

Normal domains call library/metadata/collectors/* at lifecycle boundaries.
```

So yes, I’d treat it as basically either/or.

If we choose central collectors, then `library/training/metadata.py` should not survive as the pattern. Its contents should be split into central metadata homes:

```text
library/metadata/collectors/run.py
library/metadata/collectors/data.py
library/metadata/collectors/optimization.py
library/metadata/collectors/artifact.py
library/metadata/projections/kohya_ss.py
library/metadata/session.py  # or state.py, if we need active metadata state
```

Then trainer code just does orchestration:

```python
run_facts = collect_training_run_metadata(...)
dataset_facts = collect_dataset_metadata(...)
checkpoint_facts = collect_checkpoint_artifact_metadata(...)

metadata = project_checkpoint_metadata(...)
```

No `library/training/metadata.py`.

The only local metadata files should be explicit plugin/facet exceptions, not routine domains:

```text
library/adapters/.../metadata.py        # okay when method/family-local
library/models/<family>/metadata.py     # maybe, if model families are treated similarly
```

So I agree with your framing. If we still need `library/training/metadata.py`, `library/data/metadata.py`, `library/optimization/metadata.py`, etc., then central collectors are mostly fake organization. The cleaner commitment is: **central catalog, central collectors, central projections; domains own source objects and call sites, not metadata modules.**

User:
is that system able to defend itself fro criticism? we're still looking for the best method

Agent:
Yes, it can defend itself, but only with constraints. Without those constraints, it collapses into “metadata.py everywhere, but under a different folder.”

The strongest defense is this:

```text
Metadata is a cross-cutting catalog/export/storage system.
So the code that defines recorded records, collects them, validates them, and projects them should be findable in one place.
```

That matches the config-system analogy better than local domain files do. Config dataclasses are not scattered into every feature folder; metadata dataclasses probably shouldn’t be either.

The system defends itself well on:

- **Discoverability:** if someone asks “what metadata can this repo record/export?”, they inspect `library/metadata/`.
- **Consistency:** collectors can share projection rules, validation, key naming, schema versions, minimum/full policies, and backend behavior.
- **Migration control:** old `ss_*` compatibility can be visibly quarantined under metadata projection/compatibility code instead of leaking through training/data/model files.
- **Avoiding local schema drift:** domains cannot accidentally invent their own recorded metadata shapes.
- **Future SQLite/export support:** centralized collectors make it easier to write records into a durable store later.

The criticism is also real:

- **Dependency risk:** central collectors may import too much from training/data/models and become a spiderweb.
- **Ownership fuzziness:** if metadata collectors deeply understand domain internals, then domain logic has moved away from the domain.
- **Plugin friction:** adapters/model families may need local metadata behavior because central collectors cannot know every method/family detail.
- **God package risk:** `library/metadata/collectors/` can become a second repo inside the repo if every concern dumps logic there.

The way it survives that criticism is by setting a hard rule:

```text
Collectors translate already-owned domain objects into metadata records.
Collectors do not perform domain decisions.
```

So this is good:

```python
collect_optimizer_metadata(plan: OptimizationPlan) -> OptimizerMetadata
```

This is suspicious:

```python
collect_optimizer_metadata(cfg, model, trainable_params, accelerator)
```

And this is bad:

```python
collect_optimizer_metadata(...)  # decides optimizer grouping or scheduler behavior
```

My current judgment: **central collectors are probably the better default for this repo**, because metadata is meant to become a backbone with storage/projections/validation. But it must be paired with explicit escape hatches:

```text
Default:
  dataclasses + collectors + projections live in library/metadata/

Normal domains:
  call collectors, do not define metadata.py

Plugin/facet domains:
  may expose local metadata hooks when central collection cannot know their surface

Forbidden:
  local recorded metadata schema islands
  local metadata.py files by convenience
  central collectors that make domain decisions
```

So I’d defend it as the best method if we phrase it as **central metadata assembly, local domain ownership of source truth**, not “central metadata owns every detail.”

User:
why are we struggling with figuring out what to do? how do other people do it if we're stuck?

Agent:
We’re struggling because metadata is not one thing.

It looks like one concern, but it is actually several concerns wearing the same coat:

```text
1. What facts exist?
2. Who knows those facts?
3. When are they captured?
4. Where are they stored?
5. How are they exported?
6. Which facts are compatibility junk vs repo-owned truth?
7. Which parts are plugin/family-specific?
```

Most systems get messy when they answer those with one folder or one abstraction.

Other projects usually solve this in one of a few ways:

```text
MLflow / experiment trackers:
  Central run store, but user/domain code logs facts explicitly.

DVC / data lineage tools:
  Metadata sits beside data artifacts and pipelines, often file/storage oriented.

Django / Rails style apps:
  Schemas/models are central-ish, but business logic lives near app domains.

Compiler/plugin systems:
  Core defines interfaces, plugins provide local metadata/details.

Config systems:
  Central schemas, local consumers.
```

Our repo is awkward because it wants parts of all of these:

```text
experiment/run metadata
checkpoint/export metadata
data/cache lineage
future SQLite-ish registry
compatibility projections
plugin-like adapters/model families
```

So there is no obvious industry-standard answer like “just put it in X.”

The practical answer is usually a hybrid:

```text
Central:
  schemas, registry/backend, validation, storage, projections

Near lifecycle owner:
  the call site that says “now capture metadata”

Central collectors:
  reusable translation from repo objects to metadata records

Plugin/family local:
  details that only a method/family implementation can know
```

The reason we keep circling is that each simple slogan has a trap:

```text
“Domains own metadata”
  sounds right, but becomes metadata.py everywhere.

“Central metadata owns everything”
  sounds clean, but can become a god package.

“Adapters/families own details”
  sounds extensible, but can leak exceptions everywhere.

“Compatibility is projection”
  sounds right, but old ss_* facts are currently mixed into run construction.
```

So no, we’re not stuck because this is impossible. We’re stuck because we’re trying to design the boundary before the system has enough real examples. That is genuinely hard.

My recommendation is to pick a rule that is easy to enforce and revise later:

```text
Default to central metadata package:
  dataclasses/
  collectors/
  projections/
  backend/storage/validation

No normal domain metadata.py files.

Domain code only owns call sites and source objects.

Plugin-like areas can expose local metadata hooks by explicit exception.
```

That is not perfect, but it is defensible, inspectable, and much less likely to turn into soup than the alternatives.

User:
I looked at what other repos do and it doesn't seem like they care much about this topic, they usually have the metadata code next to the actual code

Agent:
That sounds right. Most repos put metadata next to the code because for them metadata is usually **incidental output**, not a first-class subsystem.

Typical pattern:

```text
save_model.py
  builds metadata dict
  writes safetensors/checkpoint
```

or:

```text
dataset.py
  writes manifest/cache metadata next to cache logic
```

That works when metadata is mostly:

```text
small
local
export-only
not queried later
not versioned carefully
not shared across run/data/artifact systems
```

Our case is different only if we really mean “metadata system backbone.”

Because we want metadata to eventually cover:

```text
checkpoint exports
run provenance
data/cache accounting
streamed/async data readiness
artifact lineage
future SQLite lookup
kuro.* namespace
legacy ss_* compatibility
modelspec projection
adapter/model-family extension points
```

If we follow the common repo pattern, we probably get a normal practical result:

```text
metadata lives near code
things are easy locally
global inventory gets harder
SQLite/backend later needs adapters over scattered code
compatibility keys leak everywhere
```

That may still be acceptable. It is the simpler industry-default path.

The central collectors path is more “designed system”:

```text
metadata is discoverable centrally
projections/storage/validation have one home
domain files stay less metadata-heavy
but collectors must avoid becoming a god package
```

So I think the honest choice is:

```text
Option A: Common repo style
Metadata code lives near the thing producing it.
Best if metadata remains mostly artifact/export/helper behavior.

Option B: Backbone style
Recorded schemas, collectors, projections, backend live centrally.
Best if metadata becomes durable infrastructure across repo areas.
```

Given your goal has consistently been “metadata system backbone,” I still lean Option B. But the reason other repos do Option A is not because they solved this better; it is usually because they never elevated metadata into a real subsystem.

User:

I think if we don't do metadata.py in each section that wants it we need to treat it as an entire central system that things go through like we wanted initially, even if it's a "god module". that would kind of be the point in that case

some suggestions agents had:

Then I’d pick **central system that things interact with**.

Given your constraint:

> no metadata-related helper functions sprinkled inside normal domain files

then this:

```text
library/data/manifest.py
  sample_metadata_from_entry(...)
```

is out.

And honestly, fair. It keeps “normal domain logic” cleaner.

So the realistic choices become:

## Option 1: local `metadata.py` per domain

```text
library/training/metadata.py
library/data/metadata.py
library/optimization/metadata.py
library/models/metadata.py
```

This is understandable, but it has the sprawl risk you already dislike. Even if it starts clean, it encourages every area to grow its own metadata mini-layer.

## Option 2: central metadata system

```text
library/metadata/
  dataclasses/
  emitters/
  projections/
  backend.py
  validation.py
  storage/
```

Domain code interacts with it at boundaries.

I think **Option 2 is better**.

The important tweak: don’t call the central code “collectors” if that sounds like crawling/importing everything. Call them **emitters**, **builders**, or **recorders**.

A shape like this makes sense:

```text
library/metadata/
  dataclasses/
    run.py
    data.py
    model.py
    optimization.py
    artifact.py
    checkpoint.py

  emitters/
    training.py
    data.py
    model.py
    optimization.py
    checkpoint.py

  projections/
    kohya_ss.py
    modelspec.py
    kuro.py
    safetensors.py

  backend.py
  validation.py
  storage/
```

Then normal files stay clean:

```python
# library/training/checkpointing.py

from library.metadata.emitters.checkpoint import build_checkpoint_metadata

metadata = build_checkpoint_metadata(...)
```

```python
# library/data/cache.py

from library.metadata.emitters.data import emit_sample_cache_metadata

emit_sample_cache_metadata(metadata_backend, cache_entry)
```

The difference from `metadata.py everywhere` is that the metadata-related code is centralized and discoverable:

```text
where is metadata logic?
  library/metadata/
```

not scattered:

```text
library/training/metadata.py
library/data/metadata.py
library/optimization/metadata.py
...
```

The rule should be:

```text
Normal domain files call metadata APIs.
They do not define metadata helper functions inline.
Recorded metadata dataclasses live centrally.
Metadata assembly/export helpers live centrally.
Plugin-like exceptions may keep local metadata files.
```

So yeah: with your preference included, I’d stop trying to force “local emission where facts become true” as actual helper placement. The **call site** can be local, but the metadata code should live centrally.

Final verdict:

```text
Best fit:
  central library/metadata system that domains call into

Acceptable fallback:
  one local metadata.py only when a domain/plugin genuinely needs isolated behavior

Avoid:
  metadata helper functions mixed into normal domain implementation files
  metadata.py in every folder by habit
```

Agent:
Yes, I think this is the cleanest conclusion:

```text
If we reject metadata.py per domain, then metadata has to become a real central subsystem.
```

At that point “god module” is not automatically bad. The bad version is a god module that makes training/data/model decisions. The good version is a central system that owns metadata concerns end-to-end:

```text
library/metadata/
  dataclasses/      # recorded metadata shapes
  emitters/         # build/emit records from domain-owned inputs
  projections/      # ss_, modelspec., kuro., safetensors
  backend.py        # composition/validation/query surface
  storage/          # sqlite later
  validation.py
  keys.py
```

I like **emitters** better than collectors here. “Collector” sounds like it might rummage through the repo and discover facts. “Emitter” makes the boundary clearer:

```text
domain code reaches a lifecycle boundary
domain code passes known objects/facts into metadata
metadata system emits records/projections
```

So normal domain code would contain call sites, not metadata helper logic:

```python
from library.metadata.emitters.checkpoint import build_checkpoint_metadata

metadata = build_checkpoint_metadata(
    cfg=cfg,
    run_state=run_state,
    manifest=train_manifest,
    model_metadata=model_metadata,
)
```

And the rule becomes enforceable:

```text
Default:
  Metadata code lives in library/metadata/.

Normal domains:
  Call metadata APIs at lifecycle boundaries.

Not allowed by default:
  library/training/metadata.py
  library/data/metadata.py
  library/optimization/metadata.py
  inline sample_metadata_from_entry helpers in normal domain files

Allowed exceptions:
  plugin/facet areas like adapter methods, and possibly model families if we explicitly bless them.
```

The key nuance: central metadata can be “central” without owning source truth. Training still owns training runtime state. Data still owns manifests/cache entries. Optimization still owns optimizer plans. Metadata owns the act of turning those things into recorded/exported metadata.

So I’d revise our direction to:

```text
central metadata catalog
central metadata emitters
central projections/storage/backend
local lifecycle call sites
explicit plugin/family exceptions only
```

That feels like the first version that actually matches your preference and is defensible as a backbone.

User:
Do you want to go through the metadata docs/openspec and update the language where it's lacking? it's fine if you don't lock down everything 100%, but it would be nice if the setup goal is stated in them so we know what to do when we look at it and don't go through the confusion again.

Agent:
Updated the metadata docs and OpenSpec language to lock in the clearer target:

- `library/metadata/` is the central system: recorded dataclasses, emitters/builders, projections, backend/storage, validation.
- Normal domains keep lifecycle call sites and source objects, but should not grow `metadata.py` modules by default.
- `library/training/metadata.py` is now described as transitional, not the pattern.
- Adapter methods, and possibly model families later, are explicit plugin/facet exceptions.
- Compatibility keys are projection/export concerns, not first-party schema ownership.

Validation passed:

```text
openspec validate metadata-backbone --strict
git diff --check
```

I also added the clarification to `CHANGELOG.md` / `ROADMAP.md` and closed bead `sd-scripts-cd3`. I did not commit because the worktree already contains broader existing metadata/code changes and unrelated dirty files.
