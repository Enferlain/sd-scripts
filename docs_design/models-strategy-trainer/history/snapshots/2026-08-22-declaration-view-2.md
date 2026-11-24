The four scenarios did exactly what they were supposed to do. They ruled out the tempting shortcuts—role, executability, source, initial presence, contract/facet identity—without inventing a universal model taxonomy. 
I also like where the minimum ended up:
```
ParticipantDeclaration
    authored key/address
    semantic compatibility filing
    execution-route requirements
```
rather than a huge struct full of current implementation facts. 
And separating readiness from identity was the right call. A deferred model and an eagerly loaded model can be the same declaration; they simply reach the required state at different times. 
The ParticipantKey / ParticipantRef section is also now pretty clean. The key is the authored address; the ref is the accepted run incarnation; revisions describe changing state rather than changing identity. 
The materialization table currently says:
moves a declared participant from unbound/deferred to bound

But earlier the document deliberately decided that deferred/required/etc. are readiness constraints, not participant or binding-state identities:
```
binding state
    declared/unbound -> bound
```

So I'd change that table entry to simply:
moves a declared participant from unbound to bound
“Deferred” describes why remaining unbound is currently acceptable, not a different binding state.
Tiny wording change, but it preserves a distinction you just deliberately established.
The document now uses:
```
relationship references
relationship revision
relationship reference + expected revision
```
throughout the binding exchange. 
That may very well be correct. In fact, once operational relationships can independently transition through resolved/active/inactive/detached states, a stable run-scoped reference is quite plausible.
But earlier we were intentionally careful not to assume that every relationship necessarily requires ParticipantRef-style identity before deriving it.
Here the document effectively has made that choice, but it isn't listed in the working-decision register. 
I'd do one of two things:
```
either

WORKING DECISION:
declared operational relationships receive
run-scoped RelationshipRefs because their state
and revisions evolve independently

or

temporarily say "relationship declaration/reference"
where exact identity mechanics aren't needed yet
```
I lean toward the first now, because the binding exchange is starting to demonstrate an actual reason for it. But it should be an explicit working decision rather than something that sneaks in through table vocabulary.
The coarse freshness rule says:
```
snapshot R17
    ↓
any later binding-affecting accepted transition
    ↓
R18
```

But the authority also owns relationship and route state. A relationship transition can invalidate a derived projection without changing a participant's state binding.
I'd use something more like:
any accepted transition that changes snapshot-visible authoritative state
or shorter:
any relevant authority-state transition
Then the conservative rule is unambiguous:
```
projection depends on whole snapshot R17

anything represented by that snapshot changes
    → new snapshot
    → projection stale
```
Fine-grained dependencies can then exempt unrelated participant/relationship/route changes.
This table is probably the most valuable part of the file now:
```
establish
materialize
replace
relationship transition
arrangement amendment
route rebinding
merge/fold
```

It turns the earlier Q3 taxonomy into actual state-changing operations and, crucially, forces each operation to say:
-  what goes in; 
-  what gets accepted; 
-  what canonical state changes; 
-  what can fail. 
That's exactly the bridge we needed between “architecture” and APIs.
The replacement section is also nice because the rule is understandable without knowing the implementation:
You can replace the object while keeping the participant identity if the new thing still fulfills the same thing the strategy declared.

That's much more useful than rules involving class identity or module shape.
The ownership split is very clean:
```
prepared model execution routes
    → binding authority

optimizer/scheduler runtime
    → Trainer optimization runtime

DeepSpeed / sync / coordination handles
    → Trainer runtime infrastructure
```

That stops accelerator.prepare() from dictating the architecture just because one backend API happens to return several things together.
And the stale-result rule is exactly the sort of correctness guarantee missing from the current system:
Something prepared from snapshot R17 cannot silently be installed after the strategy has moved to R18.

The in-place-mutation section also looks correctly cautious now: it states the invariant without prematurely deciding that the answer must be a “lease” or a PREPARING enum. 
There is a nice upcoming tension hiding there, though:
```
preparation reads snapshot R17
    ↓
must withdraw freshness before in-place mutation
    ↓
authority state changes
    ↓
is the preparation still considered based on R17?
```
That's not a flaw in the document. That's exactly why the preparation-coordination protocol is still open.  I think that'll be one of the next genuinely substantive design problems.
I wouldn't let preparation completely define joint optimizer/scheduler behavior before the optimization exchange is designed.
The document currently includes prepared optimizer/scheduler objects in the preparation result.  That's reasonable as backend pressure—Accelerate/DeepSpeed really can prepare them jointly—but the exact way they enter the preparation plan should probably remain subordinate to the later optimization ownership decision.
In other words:
```
known already:
    they do not belong to BindingAuthority
    backend may need to prepare them jointly

not yet necessarily known:
    exact optimization request/result types
    exact point at which optimizer exists
    exact preparation coordination API
```
The file mostly respects that already; I'd just keep an eye on it.