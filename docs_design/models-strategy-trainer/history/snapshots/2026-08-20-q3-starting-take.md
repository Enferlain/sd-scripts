Question 3: how are additions and replacements represented?

This now follows pretty naturally.

There are actually three different operations hiding behind what we'd casually call “replace a component”:

A. rebind execution
B. replace logical implementation/source
C. add/remove a logical participant

They need different semantics.

A. Execution rebinding

This is what Accelerator/DDP does.

Before preparation:

denoiser
  logical binding  → SdxlUNet
  execution        → SdxlUNet

After:

denoiser
  logical binding  → SdxlUNet
  execution.train  → DDP(SdxlUNet)

Nothing about the logical arrangement changed.

This should be represented as an update to execution state under the same component identity.

No new component. No provenance change.

ComponentKey("denoiser")
    remains ComponentKey("denoiser")

This is exactly the thing the current LoadedModelComponent.module cannot represent cleanly because it gets overwritten during preparation. The design docs explicitly called for retaining distinct logical/original and prepared bindings.

B. Replacing what fills an existing logical slot

External VAE is the obvious example.

Start:

autoencoder
  source → base checkpoint VAE
  logical binding → integrated VAE instance

Then the authored strategy chooses an external VAE:

autoencoder
  source → external VAE checkpoint
  logical binding → external VAE instance

This can retain the same logical identity because the strategy still says:

this participant is the autoencoder used by this training definition

What changes is:

source
implementation/binding
possibly capabilities
possibly persistence relationship

I'd call this logical rebinding or slot replacement, although we don't need to lock vocabulary yet.

The important rule is:

The identity may remain stable when a new implementation replaces the occupant of the same authored semantic slot and the old occupant is no longer concurrently meaningful.

That gives us:

autoencoder
    old binding ──replaced_by──> external AutoencoderKL

rather than:

autoencoder
autoencoder2

just because the checkpoint source changed.

However, replacement must cause capability/compatibility validation again.

Suppose the strategy expects:

latent representation:
    channels = 4
    scale behavior = SDXL

and someone replaces the autoencoder with something producing 16 channels.

Same slot name does not magically mean valid replacement.

The arrangement needs to be revalidated before preparation.

C. Adding a genuinely new participant

This is the adapter case.

Before:

denoiser
autoencoder
conditioning.clip_l
conditioning.clip_g

After attachment:

denoiser
autoencoder
conditioning.clip_l
conditioning.clip_g
adapter.style

The adapter needs a new logical identity because it:

owns distinct parameters;
can have an independent source;
can be trained/frozen separately;
may persist independently;
has a relationship to the denoiser rather than replacing it.

Conceptually:

adapter.style
    attached_to → denoiser

This is exactly why the prior direction said adapter training should not imply a new adapter-specific strategy class. The strategy can deliberately add/select that participant and the Trainer operates over the resulting declared arrangement.

Deferred loading fits somewhere between declaration and binding

This needs a bit of care.

We could have:

declared component
    teacher.denoiser


binding
    not materialized yet

That is different from the component not existing.

So the lifecycle should be able to distinguish:

declared identity
    exists semantically


logical binding
    absent/deferred


prepared execution binding
    absent

Then later:

teacher.denoiser
    logical binding → loaded teacher module

This is not “addition” because the participant was already part of the validated arrangement.

That distinction seems very useful:

declare
    establishes participant identity and expected semantics


materialize/bind
    fills it with an object


add
    changes the arrangement itself

And the direction document already warned us not to collapse declaring, materializing, relationship binding, and establishing authoritative identity into one giant bind() operation.

The slightly uncomfortable question: can the arrangement mutate after validation?

I think yes, but only through explicit lifecycle transitions.

Adapters already prove that we may need:

validated base arrangement
    ↓ attachment capability
expanded bound arrangement

But the generic Trainer shouldn't observe an arbitrary list suddenly changing because strategy code appended to it.

Something has to say:

AttachmentResult
  added:
    adapter.style


  relationships:
    adapter.style attached_to denoiser

Then the authoritative binding system applies that result and validates whatever downstream invariants are affected.

Likewise for replacement:

ReplacementResult
  component:
    autoencoder


  old source:
    integrated


  new source:
    external


  logical binding:
    new_module

I'm deliberately describing semantics here rather than proposing those actual dataclasses.

I'd use these rules
execution preparation/wrapping
    → same component
    → update execution binding only


new object occupying same semantic slot
    → same logical identity may remain
    → update logical binding/source
    → revalidate affected constraints


previously declared deferred component becomes available
    → same logical identity
    → establish missing logical binding


genuinely new semantic participant
    → new logical identity
    → declare relationships/capabilities explicitly


old and new participant both remain meaningful
    → they cannot share one logical identity


removing a participant
    → explicit arrangement mutation
    → invalidate any dependencies/optimization bindings referring to it

The particularly strong invariant I'd add is:

A logical identity cannot silently change semantic meaning.

So this is valid:

autoencoder:
    integrated VAE → external compatible VAE

because it still fills autoencoder.

But this is nonsense:

denoiser:
    UNet → LoRA adapter

even if some Python API could technically stuff the object into the slot.

That should fail contract validation.