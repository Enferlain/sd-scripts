# Extension — Scaling Cross-Booru Ontology to Multiple Databases + LLM-Assisted Mapping

This extends the previous **Cross-Booru Tag Alias and Caption-View Training Strategy**.

The core idea continues to work as more image databases are added, but the representation should change from:

```text
database A tag ↔ database B tag
```

into:

```text
source-specific tags
        ↓
shared semantic concepts
```

This avoids an explosion of pairwise mappings and makes it possible to incorporate arbitrarily many annotation systems.

---

# 1. Do not scale with pairwise alias tables

With two databases, this is manageable:

```text
Danbooru: split
    ↔
e621: split_form
```

With many databases, a pairwise system becomes increasingly messy:

```text
DB1: split ↔ DB2: split_form
DB1: split ↔ DB3: doing_splits
DB1: split ↔ DB4: legs_split

DB2: split_form ↔ DB3: doing_splits
DB2: split_form ↔ DB4: legs_split
...
```

For \(N\) vocabularies, maintaining relationships pair-by-pair scales badly and creates duplicated semantic information.

Instead, introduce an internal semantic concept:

```text
                 [CONCEPT: split pose]
                  /      |       \
                 /       |        \
Danbooru: split   e621: split_form   DB3: doing_splits
                                      \
                                       DB4: legs_split
```

The semantic concept is primarily an internal bookkeeping object.

The diffusion model does **not** necessarily need to see the canonical concept identifier.

---

# 2. Source vocabularies attach to shared concepts

The general structure becomes:

```text
                     SHARED ONTOLOGY
                 /       |       |       \
                /        |       |        \
          Danbooru      e621     DB3      DB4
             |           |        |        |
          source       source   source   source
           tags         tags     tags     tags
```

Each source keeps its own vocabulary.

A tag may be:

- equivalent to a shared concept;
- narrower than a shared concept;
- broader than one;
- related to one;
- composed of multiple concepts;
- source-specific;
- unresolved.

This is preferable to forcing everything into one canonical surface vocabulary.

---

# 3. The ontology should represent semantic structure, not merely aliases

Example:

```text
split
split_form
doing_splits
```

may all attach as:

```text
EQUIVALENT → pose.split
```

But:

```text
1boy
```

should not simply alias:

```text
male
```

Instead:

```text
1boy
    ├── IMPLIES → gender.male
    └── IMPLIES → count.male.1
```

while:

```text
male
    └── MAPS_TO → gender.male
```

Likewise:

```text
2boys
    ├── IMPLIES → gender.male
    └── IMPLIES → count.male.2
```

This lets additional databases map naturally.

For example:

```text
DB3: one_male
```

could map to:

```text
gender.male
+
count.male.1
```

rather than incorrectly being treated as an exact alias of generic `male`.

---

# 4. Concept decomposition may be useful

Some source tags encode multiple semantic components simultaneously.

Example:

```text
1boy
```

can conceptually be decomposed into:

```text
entity:
    male

count:
    1
```

while preserving the original training tag:

```text
1boy
```

This decomposition is useful internally because it helps determine:

- whether two tags are genuinely equivalent;
- whether substitution loses information;
- which direction an implication goes;
- which caption augmentations are legal.

Do not necessarily replace compound source tags with decomposed forms during training.

The semantic decomposition can remain metadata.

---

# 5. More databases can improve ontology quality

Additional databases are not only more data.

They can reveal that an earlier supposed alias was actually a broader/narrower relationship.

Example:

Initially:

```text
DB1: foo
DB2: bar

foo ≈ bar
```

Then another database provides a more specific distinction:

```text
DB3:
    baz = subtype X
    qux = subtype Y
```

and empirical evidence shows:

```text
foo covers X + Y
bar only covers X
```

The ontology should then be revised to:

```text
bar NARROWER_THAN foo
```

rather than preserving the earlier `EQUIVALENT` assumption.

Therefore relations should be:

```text
typed
versioned
confidence-scored
traceable to evidence
```

rather than hardcoded permanently.

---

# 6. Duplicate and crossposted images become increasingly valuable

When the same image exists across several databases, the captions form a kind of parallel visual-language corpus.

Example:

```text
same image:

Danbooru:
    1girl, split, swimsuit

e621:
    female, split_form, swimwear

DB3:
    woman, doing_splits, swimsuit

DB4:
    1_female, legs_split, bathing_suit
```

Because the pixels are identical, this is unusually strong evidence for possible correspondences.

Likely candidate group:

```text
split
split_form
doing_splits
legs_split
```

Potential relationship:

```text
EQUIVALENT → pose.split
```

Meanwhile:

```text
1girl
female
woman
1_female
```

may require more careful treatment because the source semantics around count, age, or entity type may differ.

---

# 7. Crosspost clusters can help discover ontology mappings

Crossposted-image groups can be used to generate candidate correspondences automatically.

Across many duplicated images, calculate relationships such as:

```text
P(tag_B | tag_A on same-image crossposts)
```

and bidirectional variants.

Example:

```text
Danbooru split
    ↔ e621 split_form

3,842 matched image pairs
very high conditional overlap
```

This becomes strong evidence that the two terms should be investigated as aliases.

The system can propose:

```text
candidate_relation:
    split ↔ split_form

evidence:
    high duplicate-image correspondence
```

rather than requiring all mappings to be manually discovered.

---

# 8. Do not treat source databases as independent evidence automatically

Some databases may:

- mirror another site;
- scrape another site's captions;
- inherit its ontology;
- copy tags during uploads;
- use shared metadata sources.

Therefore:

```text
DB1 says X
DB2 says X
DB3 says X
```

does not always equal three independent confirmations.

Store source lineage where known:

```text
source_family:
    danbooru-derived

source_family:
    e621-derived

source_family:
    independently annotated
```

Evidence confidence should take this into account.

Likewise:

> The same image mirrored across five databases is still one visual sample.

It should not count as five independent examples of visual diversity.

---

# 9. Caption-view training continues to scale

Suppose the shared concept has:

```text
Danbooru: split
e621: split_form
DB3: doing_splits
DB4: legs_split
```

Then images containing the concept can be captioned on different exposures as:

```text
female, split
```

or:

```text
female, split_form
```

or:

```text
female, doing_splits
```

or:

```text
female, legs_split
```

All valid surface forms can therefore benefit from the combined visual evidence associated with the shared concept.

---

# 10. Rare textual aliases can borrow visual evidence from common aliases

Suppose source counts are:

```text
split             100,000
split_form          8,000
doing_splits          700
legs_split              90
```

If all four are genuinely equivalent, there may be ~108k visual examples of the underlying concept.

The rare alias:

```text
legs_split
```

does not need to learn only from its original 90 examples.

Some samples from the broader concept pool can be re-captioned as:

```text
legs_split
```

during dynamic caption generation.

Thus:

```text
visual concept frequency
```

and:

```text
surface-form frequency
```

can be decoupled.

This is especially useful for rare but valid source vocabularies.

---

# 11. Preserve native caption modes

As more databases are mixed, do not generate only maximally hybrid captions.

The model should still understand clean source-native captions.

Maintain several caption modes:

```text
SOURCE_NATIVE

ALIAS_RANDOMIZED

CROSS_SOURCE_HYBRID

MERGED / ENRICHED
```

Possible experimental mixture:

```text
40% source-native
30% alias-randomized
20% cross-source hybrid
10% enriched
```

These numbers are only placeholders for experimentation.

The important principle is:

> Hybrid training should teach interoperability without destroying each source's native caption dialect.

---

# 12. Avoid artificial "booru Esperanto"

If every caption becomes:

```text
DB1_tag,
DB3_tag,
DB7_tag,
DB2_tag,
DB5_tag,
...
```

the model may become excellent at an artificial preprocessing language that no real user ever writes.

Preserve realistic caption modes alongside mixed ones.

Desired capability:

```text
raw Danbooru caption → works

raw e621 caption → works

raw DB3 caption → works

hybrid caption → also works
```

rather than:

```text
only custom merged captions work well
```

---

# 13. Recommended multi-database architecture

```text
                     SOURCE DATABASES
         ┌─────────┬─────────┬─────────┬─────────┐
         │         │         │         │
      Danbooru    e621      DB3       DB4      ...
         │         │         │         │
         └─────────┴────┬────┴─────────┘
                        │
                 raw annotation DB
                        │
               duplicate/crosspost graph
                        │
                source tag registry
                        │
                semantic concept graph
                 /        |        \
          equivalent   hierarchy   related
                 \        |        /
                        │
               caption-view generator
            ┌───────────┼────────────┐
            │           │            │
         native      alias       hybrid/enriched
            │           │            │
            └───────────┴────────────┘
                        │
                     training
```

---

# 14. LLM-assisted ontology construction

An LLM can make this system far more practical, but it should act as an **analysis and proposal layer**, not as the unquestioned source of truth.

The LLM can help classify tag relationships using:

- source definitions;
- wiki descriptions;
- neighboring tags;
- frequency statistics;
- crosspost correspondence;
- example captions;
- representative images where available;
- existing manually verified mappings.

The resulting decisions should be stored with evidence and confidence.

---

# 15. LLM input for a candidate relationship

Example input bundle:

```text
Tag A:
    split

Source:
    Danbooru

Frequency:
    182,331


Tag B:
    split_form

Source:
    e621

Frequency:
    9,142


Same-image crossposts containing both:
    3,822

Typical neighboring tags for A:
    ...

Typical neighboring tags for B:
    ...

Source definition A:
    ...

Source definition B:
    ...

Representative examples:
    ...
```

Ask the LLM to classify the relationship as one of:

```text
EQUIVALENT
NARROWER_THAN
BROADER_THAN
IMPLIES
RELATED
CONFLICTS_WITH
UNRELATED
UNCERTAIN
```

and provide:

```text
confidence
reasoning summary
evidence used
warnings / ambiguity
```

---

# 16. Statistics should propose; LLM should interpret

A useful division of labor:

```text
statistical system:
    "these two tags correspond unusually often"

LLM:
    "their meanings appear equivalent"

or:

    "they overlap, but one contains count information"

human/manual review:
    resolves important ambiguous cases
```

This is stronger than relying purely on either:

```text
string similarity
```

or:

```text
LLM intuition
```

The empirical dataset provides grounding.

The LLM provides semantic interpretation.

---

# 17. Suggested relation record

Example:

```text
relation_id: ...

source_tag_a:
    danbooru:split

source_tag_b:
    e621:split_form

concept_id:
    pose.split

relation:
    EQUIVALENT

confidence:
    0.98

evidence:
    source_wiki_definition
    duplicate_image_overlap
    cooccurrence_statistics
    llm_semantic_analysis

manual_review:
    false

ontology_version:
    12
```

For a weaker case:

```text
relation:
    RELATED

confidence:
    0.68

replacement_allowed:
    false
```

This lets training transforms depend on confidence.

---

# 18. Confidence-controlled augmentation

Possible policy:

```text
EQUIVALENT
confidence >= 0.95
    → free alias substitution

EQUIVALENT
confidence 0.80–0.95
    → conservative substitution / review

RELATED
    → no direct substitution

NARROWER/BROADER
    → hierarchy-aware augmentation only

UNCERTAIN
    → preserve original source tag

CONFLICT
    → prevent invalid combinations
```

The exact thresholds can later be tuned.

---

# 19. Store provenance for every inferred relationship

Do not store only:

```text
split == split_form
```

Store why the system believes that.

Possible provenance:

```text
manual mapping

source-defined alias

wiki definition comparison

crosspost statistics

tag co-occurrence analysis

LLM classification

training behavior evidence
```

This becomes very useful when an assumption later turns out to be wrong.

---

# 20. Version the ontology

The ontology will almost certainly evolve.

Use explicit versions:

```text
ontology_v1
ontology_v2
ontology_v3
...
```

A mapping may change from:

```text
EQUIVALENT
```

to:

```text
NARROWER_THAN
```

after more databases or evidence are added.

Training runs should record which ontology version produced their captions.

Example:

```text
training_run:
    dataset_version: 14
    ontology_version: 9
    caption_generator_version: 5
```

This makes experiments reproducible.

---

# 21. Keep raw annotations immutable

Never overwrite:

```text
original Danbooru tags

original e621 tags

original DB3 tags
```

with normalized interpretations.

Instead:

```text
RAW DATA
    immutable

SEMANTIC LAYER
    derived

CAPTION VIEWS
    generated
```

If a mapping is corrected later, all captions can be regenerated without re-scraping the source data.

---

# 22. Suggested database layers

A useful architecture might contain:

## Image table

```text
image_id
hash
perceptual_hash
embedding
duplicate_group_id
```

---

## Source post table

```text
source_post_id
source
source_native_id
image_id
original_caption
score
rating
metadata
```

---

## Source tag table

```text
tag_id
source
surface_form
category
source_frequency
definition
```

---

## Post-tag relation

```text
source_post_id
tag_id
```

---

## Semantic concept table

```text
concept_id
concept_name
description
concept_category
```

---

## Tag-concept relation

```text
tag_id
concept_id
relation_type
confidence
provenance
review_status
ontology_version
```

---

## Tag-tag interaction table

```text
tag_a
tag_b
relation_type
confidence
evidence
```

---

# 23. Track interactions beyond aliases

The same framework can capture more than cross-source equivalence.

Potential interaction types:

```text
EQUIVALENT

IMPLIES

REQUIRES

BROADER_THAN

NARROWER_THAN

COUNT_OF

PART_OF

MODIFIES

ROLE_OF

RELATED

MUTUALLY_EXCLUSIVE

CONFLICTS_WITH
```

Examples:

```text
1boy
    IMPLIES male
```

```text
2boys
    COUNT_OF male = 2
```

```text
holding_sword
    IMPLIES sword
```

Potential contradictory combinations can also be tracked.

This gives the caption generator semantic awareness rather than treating all tags as independent strings.

---

# 24. LLM-assisted interaction discovery

The LLM can inspect sets of tags and definitions to propose relationships.

Example:

```text
closed_eyes
blue_eyes
```

It might classify these as:

```text
not inherently contradictory,
but visible-eye-color evidence may be absent when eyes are closed
```

That is more nuanced than a hard:

```text
CONFLICT
```

Likewise:

```text
1boy
2boys
```

is a much clearer mutually incompatible count description under normal whole-image semantics.

The relationship system therefore benefits from expressive relation types rather than only boolean alias/conflict flags.

---

# 25. LLM-generated proposals should be reviewable

The LLM should produce candidate records rather than silently modifying ontology truth.

Workflow:

```text
candidate discovered
        ↓
statistics gathered
        ↓
LLM classification
        ↓
proposed relation
        ↓
automatic acceptance if very high confidence
        OR
manual review queue
        ↓
ontology update
```

Critical/high-frequency mappings deserve stricter review than obscure low-impact ones.

---

# 26. Use cross-source frequencies intelligently

Suppose:

```text
DB1 tag A:
    100k samples

DB2 alias B:
    500 samples
```

If:

```text
A == B
```

with high confidence, concept-level coverage may be:

```text
~100.5k visual examples
```

while the surface-form frequency remains:

```text
A = common
B = rare
```

The caption generator can deliberately expose B on selected A-associated images.

This helps rare textual aliases without requiring repeated native examples.

---

# 27. Generated caption provenance

For debugging, it can be useful to record how a caption was generated.

Example:

```text
image_id:
    3921

original:
    1boy, split, muscular

generated:
    male, split_form, muscular

transformations:

    1boy → male
        relation:
            NARROWER_TO_BROADER
        confidence:
            0.99

    split → split_form
        relation:
            EQUIVALENT
        confidence:
            0.99

ontology_version:
    12

caption_recipe:
    cross_source_hybrid_v3
```

This does not necessarily need to be stored for every training exposure forever, but the system should at least be capable of reproducing the transformation.

---

# 28. Caption recipes

Instead of storing millions of generated captions, define reproducible recipes.

Example:

```text
recipe: native

recipe: alias_randomized
    equivalent_substitution_probability: 0.5

recipe: hybrid
    source_mix_probability: 0.4

recipe: hierarchy_augmented
    broader_substitution_probability: 0.2
```

Training can generate captions dynamically using:

```text
raw annotations
+
ontology version
+
recipe
+
random seed
```

This keeps the dataset flexible.

---

# 29. Training results can feed back into the ontology

Eventually, model behavior itself can become another source of evidence.

Suppose the ontology claims:

```text
A == B
```

but evaluation shows:

```text
prompt A → concept X reliably

prompt B → visibly different concept Y
```

That relationship should be flagged for review.

Potential loop:

```text
ontology
    ↓
training
    ↓
evaluation
    ↓
alias-equivalence tests
    ↓
unexpected discrepancy
    ↓
relation review
```

The goal is **not** to assume model behavior proves semantics.

Instead, inconsistent model behavior is a useful diagnostic that either:

- the mapping is wrong;
- the caption exposure is imbalanced;
- the text encoder represents the terms differently;
- training failed to align the aliases.

---

# 30. LLM-assisted visual-language ontology pipeline

Long-term architecture:

```text
                        SOURCE DATABASES
             ┌─────────┬─────────┬─────────┐
             │         │         │         │
          Danbooru    e621      DB3       DB4...
             │         │         │         │
             └─────────┴────┬────┴─────────┘
                            │
                     immutable raw DB
                            │
                 duplicate / crosspost graph
                            │
                    tag statistics layer
                            │
                    candidate mappings
                            │
             ┌──────────────┴──────────────┐
             │                             │
        statistical evidence           definitions /
                                          metadata
             │                             │
             └──────────────┬──────────────┘
                            │
                       LLM analysis
                            │
                  proposed semantic graph
                            │
                confidence / review system
                            │
                     versioned ontology
                            │
                  caption-view compiler
              ┌─────────────┼─────────────┐
              │             │             │
           native        aliases        hybrid
              │             │             │
              └─────────────┴─────────────┘
                            │
                         training
                            │
                        evaluation
                            │
                    ontology diagnostics
```

---

# 31. Core design principle

The system should increasingly resemble a:

> **visual-language ontology + dataset compiler**

rather than a pile of source-specific caption preprocessing scripts.

The databases supply:

```text
images
+
annotations
+
different textual views of visual concepts
```

The ontology layer determines:

```text
what those annotations mean relative to one another
```

The caption compiler determines:

```text
which valid textual view the model sees on each exposure
```

And the LLM helps automate the difficult semantic interpretation needed to maintain that ontology.

---

# Practical initial implementation

A first usable version does not need the entire ambitious graph system.

Start with:

```text
1. Import raw captions/tags from every source unchanged.

2. Deduplicate images and create crosspost groups.

3. Create a global source-tag registry.

4. Generate candidate tag correspondences from:
       exact strings
       known aliases
       crosspost statistics
       definitions

5. Use an LLM to classify candidates into:
       EQUIVALENT
       BROADER
       NARROWER
       RELATED
       UNKNOWN

6. Store:
       relation
       confidence
       evidence
       ontology version

7. Automatically use only high-confidence
   EQUIVALENT mappings for alias randomization.

8. Preserve source-native captions.

9. Add cross-source hybrid captions as a separate recipe.

10. Add hierarchy-aware transformations later.
```

This gives most of the immediate benefit without requiring a perfect universal ontology before training can start.

---

# Main takeaway

The two-database strategy extrapolates cleanly to many databases, but the abstraction should evolve from:

```text
tag A aliases tag B
```

to:

```text
many source-specific surface forms
        ↓
typed relationships
        ↓
shared semantic concepts
```

As more annotation databases are added, they provide not only more training data but also more evidence for understanding the semantics of the tagging systems themselves.

An LLM can make this tractable by helping interpret candidate mappings and interactions, especially when combined with:

```text
crossposted-image evidence
tag statistics
source definitions
co-occurrence structure
manual review
```

The LLM should propose and explain ontology relations; it should **not erase or rewrite the authoritative raw annotations**.

The resulting long-term system is effectively a versioned, evidence-backed visual-language ontology that compiles heterogeneous source annotations into controlled training caption views.