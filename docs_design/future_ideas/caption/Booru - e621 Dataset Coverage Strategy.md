# Booru / e621 Dataset Coverage Strategy

## Goal

When adapting an already-pretrained diffusion model to Danbooru/e621-style captions, distinguish two different objectives:

1. **Teach the model the tag language**
   - Learn that tags such as `blue_hair`, `female`, `split_form`, etc. correspond to particular visual concepts.
   - This alone does **not** require training on the entire database.

2. **Transfer the visual knowledge contained in the database**
   - Rare anatomy
   - Unusual poses
   - Species/subspecies
   - Fine-grained clothing concepts
   - Transformations
   - Objects
   - Interactions
   - Long-tail visual concepts

For objective #2, naïvely selecting a small "representative" subset is dangerous because it can remove exactly the long-tail concepts worth learning from booru datasets.

The desired dataset should therefore be **coverage-aware rather than simply random, popularity-weighted, or quality-filtered**.

---

## Core principle

Use the **entire database metadata/tag index** when constructing the training dataset, even if the final training set uses only a subset of the images.

Dataset selection should optimize for:

- tag coverage;
- long-tail preservation;
- diversity within each tag;
- diversity of tag combinations;
- image quality;
- reduction of duplicates / near-duplicates;
- avoiding excessive dominance by common tags.

The important quantity is not simply:

```text
number of images selected
```

but:

```text
visual concept coverage
+
examples per concept
+
diversity of examples
```

---

# Initial tag-frequency policy

A reasonable starting policy for a pretrained diffusion model:

| Total database occurrences | Suggested treatment | Desired distinct selected examples |
|---:|---|---:|
| 1–9 | Opportunistic only | Whatever naturally survives |
| 10–24 | Preserve useful examples | ~10–24 |
| 25–49 | Soft coverage target | Prefer essentially all good examples |
| 50–99 | Explicit coverage target | Prefer essentially all |
| 100–299 | Strong preservation | ~100–250 |
| 300–999 | Enough for robust learning | ~250–500 |
| 1k–10k | Diversity-oriented subsampling | ~500–1,000 |
| 10k–100k | Heavy subsampling | ~750–1,500 |
| 100k+ | Strong cap | ~1,000–3,000 |

These are starting heuristics, not fixed truths.

### Practical thresholds

```text
Main guaranteed coverage threshold: ~50 occurrences

Soft coverage threshold: ~25 occurrences

Below 25:
    retain opportunistically rather than intentionally removing
```

Do **not** interpret `<50` as "delete the tag."

A 15-example concept may still be useful if:

- the base model already understands the concept;
- the tag merely teaches an alias/new textual representation;
- its examples naturally enter the dataset for other reasons.

---

# How much data is needed to learn a concept?

For concepts genuinely missing from the base model, a useful rough scale is:

```text
<25       highly uncertain / experimental
25–50     weak signal
50–100    plausible
100–300   useful
300–500   comfortable
500–1000  strong
1000+     potentially useful for highly diverse concepts
```

This means **distinct examples**, not training exposures.

Repeating 50 images ten times does not produce the same concept diversity as 500 different images.

---

# Existing concept vs genuinely new concept

Not every tag requires the same amount of training.

## Existing concepts

Example:

```text
red hair
long hair
chair
boots
standing
clouds
```

The base model probably already has these concepts.

Teaching:

```text
red_hair
long_hair
chair
boots
standing
clouds
```

may primarily be a **conditioning remapping problem**.

Tens to low hundreds of tagged examples may therefore be sufficient.

## New or unusually fine-grained concepts

Example categories:

```text
rare anatomy
specific transformation state
obscure species morphology
fine-grained pose
uncommon interaction
specialized clothing construction
```

These require actual visual knowledge transfer.

For these, hundreds of diverse examples are much more desirable.

---

# Do not sample based on tag count alone

Two tags can both have 800 occurrences while requiring very different amounts of data.

Example:

```text
red_necktie        800 examples
transformation     800 examples
```

`red_necktie` is visually narrow.

`transformation` may contain many distinct visual modes.

Therefore the ideal target depends on:

```text
frequency
×
visual complexity
×
visual diversity
×
whether base model already understands it
```

Frequency is only the easiest first approximation.

---

# Prefer diversity over raw sample count

For each tag, selected examples should span as many independent contexts as possible.

Prefer variation across:

- artist;
- character;
- species;
- pose;
- composition;
- background;
- style;
- viewpoint;
- lighting;
- interacting concepts;
- image embedding clusters.

Example:

```text
500 examples
from 100 artists
```

may teach a concept better than:

```text
2,000 examples
from 3 artists
```

even though the latter has four times as many images.

---

# Important: avoid concept quarantine

A concept can become accidentally tied to an artist, character, or other correlated tag.

Bad situation:

```text
artist_x
rare_anatomy
character_y
```

appear together in nearly every example.

The model may learn:

```text
artist_x → rare_anatomy
```

rather than independently learning:

```text
rare_anatomy → rare anatomy
```

For important concepts, seek examples across:

```text
concept X
 ├─ artist A
 ├─ artist B
 ├─ artist C
 ├─ character A
 ├─ character B
 ├─ multiple species
 ├─ multiple compositions
 └─ multiple styles
```

Diversity is therefore also a **disentanglement mechanism**.

---

# Treat tag namespaces differently

Not every tag category should receive the same coverage priority.

Potential policy:

```text
General visual tags
    → high priority

Anatomy tags
    → high priority

Species / morphology tags
    → high priority

Pose / action tags
    → high priority

Object / clothing tags
    → high priority

Interaction / relationship tags
    → high priority

Character tags
    → separate budget

Artist tags
    → separate budget

Copyright / franchise tags
    → intermediate / separate budget

Administrative / metadata tags
    → normally exclude
```

Artist and character tags can consume enormous training capacity without contributing as much reusable visual ontology knowledge.

They should probably not determine general-concept coverage targets.

---

# Common tags should saturate

Do not allow extremely common tags to dominate the selected dataset.

Something like:

```text
blue_hair: 900,000 examples
```

does not require 900,000 images merely for the model to understand `blue_hair`.

Common tags should hit a saturation point.

Approximate desired shape:

```text
selected examples
     │
2000 ─────────────────────────
     │                  ______
1000 ──────────────____/
     │          ___/
 500 ────────__/
     │      /
 100 ─────/
     │
     └──────────────────────── tag frequency
         100  1k  10k  100k+
```

A possible simple heuristic:

```text
rare tags:
    keep most/all

medium tags:
    gradually subsample

common tags:
    cap around ~1k–2k diverse examples

very diverse concepts:
    optionally raise cap toward ~3k+
```

---

# Smooth target function

Instead of fixed buckets, an implementation could use a continuous target curve.

For example:

\[
T(f)=\min(f,\ 20\sqrt{f})
\]

where:

- \(f\) = total database frequency;
- \(T(f)\) = desired selected examples.

A practical implementation should additionally include:

- special handling below ~100 occurrences;
- a maximum cap;
- category-specific overrides.

Possible conceptual version:

\[
T(f)=
\min
\left(
f,\;
\max(100,\;25\sqrt f),\;
2000
\right)
\]

Do not treat this exact formula as final. The important property is:

```text
rare concepts → retain nearly everything
common concepts → diminishing returns
very common concepts → saturation
```

---

# Dataset selection is a set-cover problem

Images contain many tags simultaneously.

One image might provide coverage for:

```text
female
anthro
wolf
canine
digitigrade
claws
white_fur
blue_eyes
tail
standing
outdoors
...
```

Therefore:

```text
50,000 tags × 300 desired examples
```

does **not** imply 15 million unique images.

A good image can satisfy many tag deficits at once.

This means dataset construction can be treated as an approximate **weighted set-cover / coverage optimization problem**.

---

# Basic coverage-aware sampler

Maintain for every tag:

```text
frequency[t]       # total occurrences in database
target[t]          # desired selected examples
selected_count[t]  # selected examples so far
```

For every candidate image, calculate how much uncovered knowledge it contributes.

Simplified score:

\[
S_i =
\sum_{t\in i}
w_t
\frac{\max(0,T_t-C_t)}{T_t}
\]

where:

- \(T_t\) = desired coverage target;
- \(C_t\) = current selected coverage;
- \(w_t\) = importance weight for the tag.

Interpretation:

```text
image contains badly under-covered tags
    → high score

image contains already-saturated tags
    → little/no coverage benefit
```

Then iteratively:

```text
1. Compute tag coverage deficits.

2. Score candidate images according to
   how many useful deficits they satisfy.

3. Add diversity / quality terms.

4. Select high-value image.

5. Increment coverage counts for all its tags.

6. Recalculate deficits.

7. Repeat until targets/budget are satisfied.
```

---

# Better image-selection score

Eventually the image score should probably combine:

\[
S_i =
S_\text{coverage}
+
\alpha S_\text{diversity}
+
\beta S_\text{quality}
-
\gamma S_\text{duplicate}
-
\delta S_\text{undesirable correlations}
\]

Possible components:

### Coverage

Rewards under-covered tags.

### Diversity

Rewards images unlike already-selected examples for the same concepts.

Possible signals:

- image embedding distance;
- different artists;
- different characters;
- different co-occurring tags.

### Quality

Avoid keeping bad examples solely because they contain a rare tag.

### Duplicate penalty

Penalize:

- exact duplicates;
- crops;
- alternate resolutions;
- tiny modifications;
- very similar image embeddings.

### Correlation penalty

Potentially penalize selecting more samples when a rare concept is already overly concentrated under one artist/character.

---

# Important distinction: selection vs training frequency

The dataset selection policy and actual training sampler do not have to be identical.

Example:

```text
Selected dataset:
    preserve long-tail concepts

Training sampler:
    can rebalance exposures separately
```

A rare concept might have:

```text
50 distinct selected images
```

but receive somewhat more training exposure.

This increases learning pressure without pretending repeated examples create additional visual diversity.

Keep these two concepts separate:

```text
distinct-example coverage
≠
training exposure
```

---

# Tag combinations matter too

Individual tag coverage alone is insufficient.

A model can know:

```text
horse
tail
female
standing
```

without necessarily learning more specific combinations or relationships involving them.

Eventually consider measuring:

- tag pair coverage;
- selected higher-order combinations;
- concept + context diversity.

Full combinatorial coverage is impossible, so prioritize:

- meaningful relationships;
- rare combinations;
- combinations corresponding to known ontology concepts.

---

# Multi-caption training

If the goal is to support multiple caption dialects, the same image can be presented under different captions on different exposures.

Example:

```text
Natural language:
"A white-haired female character with animal ears
wearing a swimsuit on a pool float."

Danbooru:
1girl, white_hair, animal_ears, swimsuit, pool_float

e621:
female, white_hair, animal_ears, swimsuit, pool_float
```

Conceptually:

```text
natural language ─┐
Danbooru tags     ├──→ same visual semantics
e621 tags         ┘
```

This can teach multiple interfaces without requiring them to collapse into one caption format.

Whether the text encoder should also be trained depends on how well it represents unusual/opaque tags.

---

# Text encoder considerations

A frozen text encoder may be sufficient for:

```text
blue_hair
long_hair
standing
white_fur
```

because the underlying words already have useful representations.

More specialized tags may produce poor embeddings:

```text
split_form
obscure species terminology
opaque character name
artist identifier
booru-specific jargon
```

The denoiser can learn associations with arbitrary embeddings, but eventually this becomes inefficient.

Possible later approaches:

1. frozen text encoder + denoiser training;
2. conservative text-encoder fine-tuning;
3. separate tag encoder;
4. adapter/projector between tag representation and diffusion model.

Do not automatically assume all booru adaptation requires text-encoder training.

---

# What not to do

Avoid:

```text
Randomly sample N million images
```

without checking what knowledge disappeared.

Avoid:

```text
Only keep popular/high-score images
```

because this can destroy rare-concept coverage.

Avoid:

```text
Keep every common image because the tag is frequent
```

because common concepts quickly reach diminishing returns.

Avoid:

```text
frequency < 50 → discard
```

because very rare tags may still contain useful knowledge.

Avoid:

```text
500 exposures = 500 examples
```

when those exposures are repetitions of a small number of images.

Avoid optimizing tag counts while ignoring:

```text
artist diversity
character diversity
visual diversity
tag correlations
near-duplicates
```

---

# Initial experimental configuration

A sensible first implementation when this becomes relevant:

```text
Coverage threshold:
    hard-ish target: >= 50 database occurrences
    soft target:     >= 25
    retain <25 opportunistically

Rare tags:
    preserve nearly every useful distinct example

100–300 occurrence tags:
    aim for ~100–250 examples

300–1000:
    aim for ~250–500 diverse examples

1k–10k:
    ~500–1000 diverse examples

10k+:
    aggressively subsample

Common-tag cap:
    initially ~1000–2000

Highly variable concepts:
    allow ~3000+ where justified
```

Then evaluate empirically rather than assuming these numbers are optimal.

---

# Experiments to run later

When the training pipeline exists, compare several dataset policies.

## Experiment A — Random subset

Baseline.

```text
same total image count
randomly sampled
```

## Experiment B — Frequency-balanced

Use inverse/log/sqrt frequency weighting.

## Experiment C — Explicit coverage targets

Use the target-count scheme above.

## Experiment D — Coverage + diversity

Add embedding/co-tag/artist diversity during selection.

## Experiment E — Full database

Useful reference point if computationally feasible.

Evaluate not only image quality but:

- rare-tag prompt adherence;
- common-tag prompt adherence;
- compositionality;
- disentanglement from artist/character tags;
- generalization to combinations not directly seen;
- natural-language capability retention;
- catastrophic forgetting;
- concept coverage across frequency buckets.

---

# Useful evaluation split

Create evaluation buckets based on original database frequency:

```text
1–24
25–49
50–99
100–299
300–999
1k–10k
10k–100k
100k+
```

Then measure tag understanding separately in each bucket.

Otherwise excellent performance on thousands of common tags can hide complete failure on the long tail.

Also create special evaluation groups:

```text
rare anatomy
species
poses
clothing
objects
interactions
transformations
artist tags
character tags
```

This will show which parts of the ontology actually transfer.

---

# Main takeaway

The purpose of using Danbooru/e621 should not merely be:

```text
teach model comma-separated tags
```

The valuable part is:

```text
transfer a huge, unusually detailed visual ontology
```

Therefore dataset construction should preserve the long tail intentionally.

The initial working rule is:

> Use the complete metadata database to determine coverage. Explicitly cover meaningful tags occurring at least ~50 times, preserve tags around 25–50 aggressively, keep rarer examples opportunistically, aim for a few hundred diverse examples for learnable concepts, and cap extremely common tags at roughly 1–2k examples unless their visual diversity justifies more.

The exact thresholds should later be tuned from controlled training experiments rather than treated as fixed constants.