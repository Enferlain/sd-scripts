1. Ordinary SDXL denoiser

When the strategy is authored, we need to know:

key:
    denoiser

meaning/obligations:
    this is the participant the strategy uses for its
    denoising/training behavior

execution need:
    it requires the normal training execution route

materialization:
    expected to be available for this run

We do not need to know yet:

checkpoint path
Python class
actual nn.Module
device
dtype
Accelerate/DDP wrapper
optimizer membership

Suppose later we replace:

raw SDXL module
    ↓
another compatible implementation

It can still be the same participant if it continues satisfying whatever the strategy declared for denoiser.

But replacing it with something that cannot fulfill that strategy role should fail.

The important part is that the authority doesn't need a universal enum saying:

kind = DENOISER

It needs a way to establish:

this candidate still fulfills the declaration that the strategy authored.

2. Declared-but-deferred denoiser

This is almost exactly the same declaration.

The meaningful difference is:

materialization:
    may initially be absent
    must be materialized before some later operation

So:

authored strategy
    declares denoiser

authority established
    denoiser exists semantically
    but has no concrete module yet

later
    module loaded
    ↓
    materialization accepted

That strongly supports having some declaration-level binding/materialization requirement.

But I would avoid making “deferred” a participant category. It's just one allowed state of fulfillment.

3. LoRA-like adapter

Here the test gets useful.

The strategy authors:

adapter.style

and separately says something like:

adapter.style
    augments
denoiser

The adapter has meaningful state, but it has no independent forward execution route.

So if ParticipantDeclaration contains:

independently_executable: false

and also:

execution_route_requirements: []

we've encoded the same fact twice.

I'd therefore drop the boolean.

Instead:

zero required execution routes
    → state-bearing participant with no independent execution

For the denoiser:

normal training route required
    → execution-capable participant

That falls out naturally.

The adapter example also shows why its relationship shouldn't be stuffed into its participant identity:

adapter.style
    remains the same participant

while:
    attachment active
    attachment inactive
    target resolved
    target detached

Those are relationship changes, not participant identity changes.

4. Teacher and student from the same checkpoint

This is probably the cleanest identity test.

Imagine:

student.denoiser
teacher.denoiser

Both are initially loaded from exactly:

foo.safetensors

That source tells us essentially nothing about their logical identity.

They're different because the strategy authored two independently meaningful participants.

Later:

student
    trains

teacher
    remains frozen

But trainability isn't what made them different either. Even if both were frozen temporarily, they're still teacher and student.

Likewise:

teacher source checkpoint changes

could still be an identity-preserving replacement if it remains the participant declared as the teacher.

So this scenario strongly confirms:

source ≠ identity
trainability ≠ identity
Python object ≠ identity

The declaration is what gives each one its semantic place in the strategy.

What I think this leaves in the minimum declaration

After those four cases, I'd reduce the agent's proposed boundary slightly.

Probably intrinsic to the participant declaration:

authored semantic address
    ParticipantKey

semantic fulfillment requirements
    what must remain true for a realization to fulfill
    this participant declaration

materialization requirement
    what binding state is acceptable at establishment
    and what must eventually become available

Then associated with the participant, but possibly better modeled as separate declarations rather than fields inside ParticipantDeclaration:

execution requirements
    zero or more required named execution purposes

relationships
    how this participant relates to other participants

That would make the conceptual model:

ParticipantDeclaration
    "this meaningful participant exists"

ExecutionRequirement
    "this participant must support this execution purpose"

RelationshipDeclaration
    "this participant has this semantic relationship to another"

rather than stuffing everything into one giant declaration record.

And I would specifically remove:

whether independently executable

because the execution requirements already answer it.

The hard part: “semantic obligations”

I think the escape hatch from a universal taxonomy is:

The binding authority does not itself need to understand what a denoiser, VAE, text encoder, LoRA, teacher, etc. is. The strategy and its selected capability/feature contracts declare the requirements that a realization must fulfill; the authority records those requirements and enforces that transitions are validated against the same declaration.

So instead of:

ParticipantDeclaration:
    kind = DENOISER

the conceptual structure is more like:

ParticipantDeclaration:
    key = student.denoiser

    fulfills:
        requirements contributed by
        this strategy's authored behavior/features

For example, one strategy might require of a participant:

can supply training denoising execution
supports this conditioning/input contract
participates in this prediction objective

Another strategy could have something completely different without extending a central enum.

The authority's generic job is then:

candidate proposed for participant P
        ↓
does it satisfy P's existing declared obligations?
        ↓
yes
    identity-preserving materialization/replacement may proceed

no
    cannot silently redefine P
    arrangement change/new declaration required

That keeps the semantic meaning structured enough to validate without teaching the central runtime every model-family noun in existence.

One thing I would not settle yet

The agent currently has:

separate relationship declaration
    typed relationship identity

I'd stop one word earlier:

separate relationship declaration

We know relationships need to be declared and transitioned. I'm not sure these four examples yet prove that every relationship requires its own independently stable identity analogous to ParticipantRef.

Maybe it does eventually. But that's exactly the sort of thing these scenarios should derive rather than assume.

So my outcome from the pressure test would be:

minimum participant declaration
    authored key
    semantic fulfillment requirements
    materialization/binding requirement

separate associated declarations
    execution requirements
    relationships

definitely runtime state, not declaration
    ParticipantRef
    realization/module
    source facts
    access views
    route bindings
    revisions
    trainability
    optimizer membership
    active relationship state

The nice part is that this is already substantially smaller than the initial list. The four scenarios suggest that “participant declaration” itself can stay pretty boring. Most of the complexity belongs in the contracts around the participant, rather than becoming twenty fields on ParticipantDeclaration.