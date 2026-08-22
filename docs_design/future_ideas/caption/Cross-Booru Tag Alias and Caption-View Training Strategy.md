# Cross-Booru Tag Alias and Caption-View Training Strategy

## Goal

When combining Danbooru and e621 training data, do **not** treat the two tag sets as either:

```text
completely separate vocabularies
```

or:

```text
one flat merged vocabulary
```

The useful structure is somewhere in between.

The dataset contains:

- tags unique to Danbooru;
- tags unique to e621;
- images that exist on only one source;
- images that exist on both sources;
- tags that are effectively aliases;
- tags that are only partially equivalent;
- tags with broader/narrower relationships;
- tags that are merely related.

The training pipeline should exploit those relationships without destroying information.

The main objective is:

> Teach the model that equivalent or closely related tags refer to overlapping visual concepts, while preserving distinctions that carry additional semantic information.

---

# Main principle

Do **not** use a single undifferentiated alias table.

This is too simplistic:

```text
split      → split_form
1boy       → male
2boys      → male
```

because these relationships are not the same.

Instead, represent relationships explicitly.

For example:

```text
split <== equivalent ==> split_form

1boy  -- implies --> male
2boys -- implies --> male
3boys -- implies --> male
```

The type of relationship determines how the training caption is allowed to be transformed.

---

# Relationship types

A useful ontology should distinguish at least:

```text
EQUIVALENT
BROADER_THAN
NARROWER_THAN
RELATED
SOURCE_SPECIFIC
UNKNOWN
```

Potentially also:

```text
INCOMPATIBLE
```

where two tags should explicitly not be substituted for one another.

---

# 1. True equivalents

Example:

```text
Danbooru:
split

e621:
split_form
```

If both describe effectively the same visual concept, treat them as an equivalence class:

```text
{split, split_form}
```

The desired behavior is:

```text
split      → concept X
split_form → concept X
```

Both tags should independently evoke approximately the same visual concept.

---

# Do not routinely place equivalent aliases together

Avoid:

```text
female, split, split_form, outdoors
```

if `split` and `split_form` mean the same thing.

This teaches primarily:

```text
split and split_form often co-occur
```

rather than providing the strongest possible evidence that each individually predicts the same visual feature.

It also wastes conditioning capacity.

Prefer alternate substitution:

```text
Exposure A:
female, split, outdoors

Exposure B:
female, split_form, outdoors
```

Across training:

```text
image 1 → split
image 1 → split_form
image 2 → split_form
image 3 → split
...
```

For true aliases:

> Use one surface form at a time and vary which one appears across exposures.

---

# Why alias substitution is stronger than alias co-occurrence

Suppose an image visually contains concept X.

Training:

```text
image X + "split"
```

directly teaches:

```text
split → X
```

Training another exposure as:

```text
image X + "split_form"
```

directly teaches:

```text
split_form → X
```

By comparison:

```text
image X + "split, split_form"
```

only tells the model that the combined conditioning corresponds to X.

The model can still learn both terms, but the supervision is less clean.

Therefore:

```text
alias substitution
```

is generally preferable to:

```text
alias duplication
```

for genuinely equivalent tags.

---

# 2. Broader / narrower relationships

Example:

```text
1boy → male
2boys → male
3boys → male
```

These are **not true aliases**.

`1boy` contains at least two pieces of information:

```text
male
+
count = 1
```

Similarly:

```text
2boys =
male
+
count = 2
```

while:

```text
male =
male presence
+
count unspecified
```

Conceptually:

```text
                 male
              /    |    \
           1boy  2boys  3boys ...
```

The narrower tags contain information that the broader tag does not.

---

# Never blindly normalize narrow tags into broad tags

Bad:

```text
1boy → male
2boys → male
3boys → male
```

as unconditional replacement.

That destroys count information.

The model would lose the ability to distinguish:

```text
1boy
2boys
3boys
```

even though Danbooru explicitly encoded that distinction.

---

# Better treatment for broader/narrower tags

Keep the precise tag most of the time.

Example original:

```text
1boy, black_hair, standing
```

Occasionally produce a broader caption view:

```text
male, black_hair, standing
```

Conceptually:

```text
most exposures:
    retain 1boy

some exposures:
    substitute or augment toward male
```

A possible experimental starting point:

```text
60–80%:
    precise source tag

20–40%:
    broader representation
```

These percentages are not final recommendations; they should be evaluated experimentally.

The important asymmetry is:

```text
1boy can provide evidence for male
```

but:

```text
male must not automatically become 1boy
```

because the broader tag does not specify count.

---

# Natural distribution can teach the hierarchy

Suppose training data contains:

```text
1boy:
    only one male

2boys:
    exactly two males

male:
    one, two, three, etc.
```

The model can infer:

```text
1boy = male concept + count 1

2boys = male concept + count 2

male = common visual component
```

This is desirable.

Do not collapse such tags prematurely through preprocessing.

---

# 3. Source-specific tags

If e621 contains a useful tag with no Danbooru equivalent:

```text
e621_unique_tag
```

retain it normally.

Likewise:

```text
danbooru_unique_tag
```

should remain available.

Do not force every concept into a shared cross-source vocabulary.

The model can learn both vocabularies simultaneously.

Conceptually:

```text
                 shared concepts
                /               \
       Danbooru terms         e621 terms
             |                    |
             +---- aliases -------+
             |
      source-specific terms

Danbooru-only                e621-only
```

The desired model behaves more like a multilingual model than a model with one canonical vocabulary.

---

# 4. Images present on both sources

Duplicate images across Danbooru and e621 are especially valuable.

Example:

```text
Same image

Danbooru:
1boy, split, blue_hair

e621:
male, split_form, blue_hair
```

Do not treat these only as two unrelated training images.

Instead, treat them as:

```text
one visual sample
+
multiple valid caption views
```

Conceptually:

```text
                        ┌→ Danbooru caption
same image pixels ──────┤
                        └→ e621 caption
```

This provides unusually strong supervision because the pixels are identical while the textual representation changes.

---

# Paired duplicate images are useful cross-vocabulary supervision

For:

```text
split
```

versus:

```text
split_form
```

the same pixels provide direct evidence that the two terms correspond to the same visual feature.

For:

```text
1boy
```

versus:

```text
male
```

the same image provides evidence that the concepts overlap, while the rest of the dataset preserves the broader statistical difference between them.

This is exactly the behavior desired.

---

# 5. Caption views

Rather than creating one permanent merged caption per image, preserve the raw source information and generate alternate caption views.

Underlying structure:

```text
IMAGE
  |
  ├─ raw Danbooru tags
  ├─ raw e621 tags
  └─ cross-source ontology / relationship graph
              |
              v
       caption-view generator
```

Possible output views:

```text
View A:
    Danbooru-native

View B:
    e621-native

View C:
    merged/enriched

View D:
    alias-randomized

View E:
    cross-source hybrid

View F:
    natural language, if desired
```

Training can randomly choose one view per exposure.

---

# Why caption views are preferable to permanently duplicating data

Do not necessarily create:

```text
image_001_caption_a
image_001_caption_b
image_001_caption_c
image_001_caption_d
```

as independent dataset entries.

Instead store:

```text
image
+
raw annotations
+
relationship metadata
```

and generate captions dynamically.

Example:

```text
Epoch/exposure 1:
1boy, split

Epoch/exposure 2:
1boy, split_form

Epoch/exposure 3:
male, split

Epoch/exposure 4:
1boy, split
```

This changes textual supervision without falsely pretending the dataset contains additional visual diversity.

Keep separate:

```text
number of visual samples
```

and:

```text
number of conditioning views
```

---

# 6. Cross-source hybrid captions

This is potentially one of the strongest techniques.

Suppose the same image has:

```text
Danbooru:
1boy, split, muscular

e621:
male, split_form, muscular
```

Using only source-native captions produces:

```text
1boy + split
male + split_form
```

The model may accidentally learn source-level correlations:

```text
1boy tends to occur with split

male tends to occur with split_form
```

even though the vocabulary choices are semantically independent.

Hybrid captions break this correlation.

Generate:

```text
1boy, split
1boy, split_form
male, split
male, split_form
```

Now the model receives evidence that:

```text
1boy / male
```

and:

```text
split / split_form
```

are independent axes of variation.

This should improve disentanglement between the two tag dialects.

---

# Hybridization should be relationship-aware

Do **not** randomly swap every tag between sources.

Only perform transformations justified by the ontology.

Example:

```text
split ↔ split_form
```

can be freely substituted.

But:

```text
1boy → male
```

must be treated asymmetrically.

Therefore hybrid generation should operate on typed semantic relationships rather than string mappings.

---

# 7. Mixed captions

A caption containing both Danbooru and e621 vocabulary is not inherently bad.

For example:

```text
danbooru_unique_tag,
e621_unique_tag,
blue_hair,
split_form
```

is reasonable if all tags genuinely describe the image.

This can be useful when combining annotations from multiple sources.

The distinction is:

```text
mixing useful independent annotations
    → good

duplicating exact synonyms in the same caption
    → usually unnecessary
```

---

# Suggested transformation rules

## Equivalent

Example:

```text
split == split_form
```

Training rule:

```text
randomly select one alias
```

Possible:

```text
split
```

or:

```text
split_form
```

Avoid routinely producing:

```text
split, split_form
```

---

## Narrower → broader

Example:

```text
1boy → male
```

Training rule:

```text
retain precise term most of the time
occasionally expose broader representation
```

Never treat the reverse direction as guaranteed.

---

## Related but non-equivalent

Example:

```text
A RELATED_TO B
```

Training rule:

```text
do not substitute automatically
```

Let natural co-occurrence and visual evidence teach the relationship unless there is a specific augmentation scheme.

---

## Source-specific

Example:

```text
e621_unique_tag
```

Training rule:

```text
keep normally
```

Do not normalize away.

---

## Unknown

Training rule:

```text
preserve original tag
```

Do not invent semantic mappings merely because two terms look similar.

---

# Proposed ontology representation

A practical relationship table might look conceptually like:

```text
tag_a
tag_b
relation
confidence
source_a
source_b
```

Example:

```text
split
split_form
EQUIVALENT
high
danbooru
e621
```

```text
1boy
male
NARROWER_THAN
high
danbooru
e621
```

Potential additional metadata:

```text
manual_verified
automatic_match_score
notes
replacement_allowed
replacement_probability
```

This allows the caption generator to behave differently depending on relationship type.

---

# Alias equivalence classes

For true aliases, it may be useful to build equivalence groups.

Example:

```text
concept_id: split_pose

aliases:
    Danbooru: split
    e621: split_form
```

Another hypothetical concept:

```text
concept_id: X

aliases:
    source_A: tag_a
    source_B: tag_b
    source_C: tag_c
```

Training can sample one alias from the equivalence class.

This preserves all accepted interfaces while grounding them against the same visual concept.

---

# Do not necessarily expose canonical concept IDs to the model

The internal dataset system may use:

```text
concept_id = split_pose
```

for bookkeeping.

That does not mean the training caption needs to contain:

```text
split_pose
```

The canonical representation can exist only inside the preprocessing/ontology layer.

The model can still see:

```text
split
```

and:

```text
split_form
```

as independent surface forms.

---

# Alias balancing

Raw source frequency may be highly imbalanced.

Example:

```text
split:
    50,000 examples

split_form:
    2,000 examples
```

If trained directly, the model may strongly prefer one spelling.

For true aliases, dynamic substitution can deliberately balance exposure:

```text
50% split
50% split_form
```

or some softer ratio such as:

```text
70% common form
30% rare form
```

depending on desired behavior.

This lets alias learning be decoupled from original database frequency.

---

# Important distinction: concept frequency vs surface-form frequency

Suppose:

```text
split:
    50k

split_form:
    2k
```

The concept itself may have:

```text
52k+ usable visual samples
```

even though one textual alias is rare.

Alias substitution means both text forms can benefit from the combined visual evidence.

Conceptually:

```text
all split-related images
         |
         ├→ "split"
         └→ "split_form"
```

This is much stronger than forcing `split_form` to learn only from its original 2,000 source examples.

---

# Cross-source enrichment

When an image exists only on one source, aliases can still generate alternate valid captions.

Example:

```text
e621 image:
female, split_form, blue_hair
```

If:

```text
split_form == split
```

then another valid view is:

```text
female, split, blue_hair
```

Even without a matching Danbooru duplicate, the known ontology can transfer supervision across vocabularies.

This allows one source to help teach terminology from the other.

---

# Handling count tags carefully

Tags such as:

```text
1boy
2boys
3boys
1girl
2girls
3girls
```

should generally be treated as structured information rather than aliases for:

```text
male
female
```

A useful conceptual decomposition is:

```text
1boy =
gender/male concept
+
count = 1

2boys =
gender/male concept
+
count = 2
```

Eventually, the preprocessing system may want explicit semantic components for analysis:

```text
entity_type: male
count: 2
```

while still preserving the original surface tag:

```text
2boys
```

This can help determine legal transformations without throwing away the source vocabulary.

---

# Preserve source identity in metadata

Even if the final captions are hybridized, retain:

```text
original source
original tags
original caption
```

This is useful for:

- debugging;
- ablations;
- evaluating whether hybridization helps;
- source-specific training modes;
- reproducing experiments;
- detecting bad mappings.

Never irreversibly overwrite the raw annotations with normalized tags.

---

# Suggested stored structure

Conceptually:

```text
image_id

sources:
    danbooru:
        original_tags: [...]
        post_id: ...

    e621:
        original_tags: [...]
        post_id: ...

semantic_relations:
    ...

duplicate_group_id:
    ...

caption_views:
    generated dynamically
```

The raw data remains authoritative.

The transformation layer remains configurable.

---

# Training modes worth comparing

## A. Source-native only

```text
Danbooru images → Danbooru captions
e621 images     → e621 captions
```

Baseline.

---

## B. Merged captions

Combine all valid annotations.

Useful baseline, but may create synonym co-occurrence.

---

## C. Alias substitution

True equivalents randomly alternate.

Example:

```text
split ↔ split_form
```

Expected to strengthen independent alias grounding.

---

## D. Native + alias substitution

Mostly preserve source-native captions while dynamically substituting true aliases.

Likely strong conservative option.

---

## E. Cross-source hybrid views

Randomly mix valid equivalent forms across tag families.

Example:

```text
1boy + split
1boy + split_form
male + split
male + split_form
```

Potentially strongest for breaking source correlations.

---

## F. Hierarchy-aware augmentation

Additionally use broader/narrower relationships such as:

```text
1boy → male
```

with controlled probabilities.

More aggressive and should be evaluated separately.

---

# Evaluation questions

For equivalent tags:

```text
Does "split" produce the same concept quality as "split_form"?
```

Test both independently.

Also test whether one alias has become much stronger than another.

---

For hierarchical tags:

```text
1boy
2boys
male
```

Check that:

```text
1boy → one male
2boys → two males
male → male concept without strict count binding
```

If `male` always produces one person, the hierarchy has collapsed incorrectly.

If `1boy` loses count control, augmentation was too aggressive.

---

For unique tags:

Check that source-specific concepts remain promptable after cross-source training.

---

For hybridization:

Check whether the model handles combinations never native to either database, such as:

```text
Danbooru-specific term
+
e621-specific term
```

without degradation.

---

# Failure modes

## 1. Alias co-dependence

If equivalent tags always appear together:

```text
split, split_form
```

the model may become less reliable when only one appears.

Solution:

```text
alternate them independently
```

---

## 2. Semantic collapse

If:

```text
1boy
2boys
male
```

are treated as equal, count information disappears.

Solution:

```text
typed hierarchy, not alias replacement
```

---

## 3. Source-language clustering

If every Danbooru term only co-occurs with Danbooru vocabulary and every e621 term only with e621 vocabulary, the model may learn two partially separated caption dialects.

Solution:

```text
controlled cross-source hybrid views
```

---

## 4. Rare alias undertraining

If an alias exists only rarely on its native source, it may remain weak despite describing a common concept.

Solution:

```text
sample the rare alias across images belonging to the entire equivalence class
```

---

## 5. Destroying raw annotations

If all tags are normalized into one vocabulary before training, useful distinctions may become impossible to recover.

Solution:

```text
preserve raw tags permanently
apply transformations dynamically
```

---

## 6. Incorrect automatic aliases

String similarity or database mapping may claim equivalence where semantics actually differ.

Solution:

```text
relationship confidence
manual verification for important mappings
safe default = preserve originals
```

---

# Recommended initial implementation

Start conservatively.

```text
1. Preserve original Danbooru and e621 annotations.

2. Detect exact duplicate / cross-posted images.

3. Build a typed cross-source relationship graph.

4. Initially classify mappings as:
       EQUIVALENT
       BROADER/NARROWER
       SOURCE_SPECIFIC
       UNKNOWN

5. For EQUIVALENT:
       randomly choose one alias per exposure.

6. For BROADER/NARROWER:
       preserve precise tag by default;
       optionally generate broader views at low/moderate probability.

7. Preserve unique tags normally.

8. For images appearing on both sites:
       use both original captions as alternate views.

9. Optionally generate cross-source hybrids.

10. Never permanently overwrite the raw source tags.
```

---

# Conservative first training policy

A reasonable first version:

```text
True equivalent aliases:
    50/50 randomized surface form

Source-native captions:
    remain a major portion of exposures

Broader/narrower substitutions:
    disabled initially
    OR low probability

Cross-source hybrid captions:
    moderate probability

Unique tags:
    unchanged
```

Then add hierarchy-aware augmentation only after evaluating the simpler alias system.

This makes it easier to determine what actually improves training.

---

# Likely strongest practical strategy

The promising overall structure is:

```text
                     raw image
                         |
             ┌───────────┴───────────┐
             |                       |
       Danbooru tags             e621 tags
             |                       |
             └───────────┬───────────┘
                         |
                 relationship graph
                         |
                 caption generator
             ┌───────────┼───────────┐
             |           |           |
          native      alias view    hybrid
             |           |           |
             └───────────┴───────────┘
                         |
                      training
```

The relationship graph acts as the semantic control layer.

The training model does not need to know that such a graph exists.

It simply receives many valid textual descriptions of the same underlying visual concepts.

---

# Main takeaway

For true aliases such as:

```text
split ↔ split_form
```

the strongest straightforward supervision is likely:

> Train the same visual concept under either surface form on different exposures, rather than routinely placing both aliases in the same caption.

For relationships such as:

```text
1boy → male
```

do **not** treat them as aliases.

Preserve the narrower tag's extra information and use the broader term only as controlled alternative supervision.

The combined Danbooru/e621 system should therefore be based on a **typed tag relationship graph + dynamic caption views**, not a flat alias replacement table.

This gives the model the chance to learn:

```text
different words can mean the same thing

different words can mean overlapping things

some words encode additional information

both tag dialects can be mixed compositionally

source-specific concepts remain usable
```

without sacrificing the semantic distinctions present in either database.