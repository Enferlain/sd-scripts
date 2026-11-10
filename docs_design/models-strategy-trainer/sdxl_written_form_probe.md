# SDXL strategy in the written form: a reading probe

Written 2026-11-11 for `sd-scripts-syv.35.5`. This is a design probe, not a
strategy and not authority. Nothing checks it: there is no composed contract,
no front end and no verifier for the production vocabulary. Its purpose is to
find where the governing design
([design.md](../../openspec/changes/rework-model-strategy-trainer/design.md),
D16–D21) does not yet say enough to write a real SDXL run.

Method: take what today's SDXL run does (`library/strategies/sdxl/`, the
training phases, and the design's G2 worked cases 1 and 2) and write it with
D16's words and written-form conventions only. Implementation names after `@`
are placeholders, as in D17's outline. Spellings of planned words are guesses.
A number in brackets marks a place where I had to stop; the findings below
use the same numbers.

The outline can be deleted once the findings are handled.

Status, 2026-11-11: findings 1–8 were settled with the user the same day and
written into the governing design (D15–D20), two specs and the task list.
Findings 9–12 are spellings that the milestones building those words decide.
The outline below is as first written: it shows the nesting that finding 1
removed, and it does not use the evaluation-point route from finding 3.

## The run written out

Configuration: direct fine-tune of the UNet and both text encoders in one
unit, EMA on, latents read from a cache with a live encode on a miss, text
conditioning computed live, validation, sampling and saving on step triggers,
and a final save.

```text
iro.strategy @sdxl_finetune
    uses(@load_sdxl v1, @next_items v1, @read_latent_cache v1, @encode_latent v1,
         @sample_timesteps v1, @encode_sdxl_text v1, @sdxl_denoise v1,
         @ddpm_loss v1, @detach v1, @observe_loss v1, @val_loss v1,
         @sdxl_generate v1, @write_samples v1) {

  // All four participants come from one released checkpoint.              [5]
  %unet = iro.resource @unet : participant materialize(@load_sdxl)
  %te1  = iro.resource @text_encoder_1 : participant materialize(@load_sdxl)
  %te2  = iro.resource @text_encoder_2 : participant materialize(@load_sdxl)
  %vae  = iro.resource @vae : participant materialize(@load_sdxl)

  // Objective state with its own owner, like @policy in D16's data example.
  %sampler = iro.resource @timestep_sampler

  // One unit with three logical groups; EMA as a choice on the unit.
  %unit = iro.train.unit @main of [%unet, %te1, %te2] optimizer @adamw ema(0.9999)

  %batches = iro.stream @batches : latent_batch
               bounds(outstanding 4, ready 2)
               admit any_with_provenance(@vae)
  %heldout = iro.stream @heldout : latent_batch
               bounds(outstanding 2, ready 1)
               admit any_with_provenance(@vae)

  iro.parallel @activities capture(...) {

    // Latents from the cache, encoded live on a miss. Captions stay tokens:
    // both text encoders are trained, so they must run inside the window.
    iro.producer @data into %batches capture(%vae) {
    ^(%index: index):
      %items   = iro.call @next_items(%index) -> item_batch
      %cached  = iro.call @read_latent_cache(%items) -> optional latents
      %latents = iro.choice %cached {                                      // [9]
        // hit: use %cached
        // miss:
        iro.access %vae snapshot published initial {
        ^(%lease):
          iro.call @encode_latent(%lease, %items) -> latents
        }
      }
      iro.yield batch(%items, %latents)
    }
    iro.producer @heldout_data into %heldout capture(%vae) { ... as @data ... }

    iro.parallel @consumers capture(...) {

      iro.repeat @training until @max_steps
          capture(%batches, %unet, %te1, %te2, %unit, %sampler) {
      ^(%step: index):
        iro.train.window %unit members 4 {
          %claim = iro.take %batches
          %b     = iro.borrow %claim
          %t = iro.access %sampler protected_current current_at_start {
          ^(%s):
            iro.call @sample_timesteps(%s, %b) -> timesteps
          }

          // The loss carries a graph through all three participants. Their
          // leases have to last until the contribution has run backward.   [1]
          %obs = iro.access %te1 protected_current current_at_start {
          ^(%l1):
            iro.access %te2 protected_current current_at_start {
            ^(%l2):
              iro.access %unet protected_current current_at_start {       // [12]
              ^(%lu):
                %cond = iro.call @encode_sdxl_text(%l1, %l2, %b) -> retained conditioning
                %pred = iro.call @sdxl_denoise(%lu, %b, %cond, %t) -> retained prediction
                %loss = iro.call @ddpm_loss(%pred, %b, %t) -> retained loss
                iro.train.contribute %loss to %unit
                %seen = iro.call @detach(%loss) -> per_sample_loss
                iro.release %loss
                iro.release %pred
                iro.release %cond
                iro.yield %seen
              }
            }
          }

          // Feedback to the adaptive timestep sampler, by its owner.
          iro.access %sampler owner_update {
          ^(%s):
            iro.call @observe_loss(%s, %obs, %t)
          }
          // Step loss also has to reach the observation owner.            [6]
          iro.dispose %claim consumed
        }

        %val_due = iro.train.trigger @every_500 on %unit
        iro.choice %val_due {
          iro.request @validation(%step) from @training at settled_success  // [10]
        }
        // ... the same for @sampling and @save, each with its trigger ...
        // Epoch-based triggers have no source here.                       [7]
      }

      iro.handler @validation sources [@training] max 1
          capture(%heldout, %unet, %te1, %te2) {
      ^(%step: index):
        iro.repeat @val_items 64 times {
          %claim = iro.take %heldout
          %b     = iro.borrow %claim
          // Weights as they were at the request. With EMA on, the run wants
          // the averaged weights of that point instead.                    [3]
          iro.access %te1 protected_current at_request {
          ^(%l1): ... %te2 ... %unet ...
            iro.call @val_loss(%l1, %l2, %lu, %b) -> scalar
          }
          iro.dispose %claim consumed
        }
        // The reduced result has to reach the observation owner.          [6]
      }

      iro.handler @sampling sources [@training] max 1
          capture(%unet, %te1, %te2, %vae) {
      ^(%step: index):
        // Text encoders and UNet at the request; the VAE is frozen.    [3] [4]
        %images = ... iro.call @sdxl_generate(%l1, %l2, %lu, %lv) -> images
        iro.call @write_samples(%images, %step)                           // [11]
      }

      iro.handler @save sources [@training] max 1
          capture(%unet, %te1, %te2, %vae) {
      ^(%step: index):
        iro.persist @sdxl_checkpoint of [%unet, %te1, %te2, %vae] at_request
      }
    }

    iro.stop @data after @consumers
    iro.stop @heldout_data after @consumers
  }

  // Final save, once everything above has settled.
  iro.persist @sdxl_checkpoint of [%unet, %te1, %te2, %vae] current_at_start

  // Not written: save and exit on an interrupt.                           [8]
}
```

The adapter variant changes three things, and two of them could not be
written:

```text
  %lora   = iro.resource @lora : participant materialize(@build_lora)     // [5]
  %attach = iro.relationship @lora_on_unet from %lora to %unet
  %unit   = iro.train.unit @adapter of %lora optimizer @adamw

  // In the window the call still goes through the host only:
  iro.access %unet protected_current current_at_start {
  ^(%lu):
    iro.call @sdxl_denoise(%lu, %b, %cond, %t) -> retained prediction     // [2]
  }
```

## What could be said without trouble

- Input production as a producer into a bounded stream, with frozen sources
  read as snapshots and admission by provenance.
- Objective state as a resource with its own owner, read in the step and
  updated from the step's observation. This is the same shape as D16's
  data-only example.
- One unit over several participants, with groups and EMA as attributes.
- Accumulation as a window, with advancement left to the Trainer.
- Step triggers that request validation, sampling and saving, each served by
  a handler.
- The version at the request for the handlers, including a save that needs it
  only for its capture.
- Producer stops tied to the consumers, and a final save after they settle.

## Findings

### Collide with text the design already has

1. **A loss that still carries its graph leaves the access scope.** D16's
   training example, D17's outline and D18's
   `unit.contribute(denoise(unet.protected_current, batch))` all compute the
   loss inside an access and contribute it after the lease has ended. D14's
   retained-product rule keeps a result that depends on leased state inside
   its defining body until it is released, and `accepted-training-execution`
   requires protection to cover backward. Written to those rules, the whole
   step from text encoding to the contribution sits inside every trained
   participant's access scope, three deep for this run, with retained results
   and explicit releases. Either `contribute` is given a rule that carries
   the loss's dependencies to the end of backward, or the nesting is what
   authors write. D18's reading rule for how long a lease from
   `.protected_current` lasts depends on the answer.

2. **A host access does not cover an attached adapter.** With LoRA attached,
   the denoise call reads the adapter's weights through the host's route. The
   call holds a lease only on the host. Leases, versions and reservations are
   per participant, so nothing protects the adapter's weights during the
   call, the call's reads are not what its lease types say, and a validation
   at the request reserves the host's version but not the adapter's. D19
   gives a route its dependencies, but no rule says that using a host's route
   also covers the other end of its relationships.

3. **Nothing reads a unit's evaluation point.** D20 says sampling and
   validation read averaged weights "as the unit's evaluation point", and that
   a schedule-free optimizer's switch is an owner update. No word selects
   that point in an access. If it is realized as a swap of the participant's
   weights, the swap is an update that the handler's own reservation makes
   wait.

### Missing statements

4. **When participants load, and where they sit while idle.** The host
   materializes participants before execution, and a later materialization
   is a transition, but nothing says which is which for a given strategy.
   Today the VAE and text encoders leave the device when their outputs are
   cached and come back for sampling. D15's capacity paragraph covers work
   being admitted onto a device, not participants resident on it between
   uses.

5. **Materializers that span or depend on participants.** One released
   checkpoint materializes four participants, and an adapter's
   materialization needs its host's loaded structure first. `materialize` is
   written per resource, with no form for either. D19 has the run-time
   mechanics for both.

6. **Reporting a value as an observation.** Step loss and a validation
   result have to reach the observation owner. No word does that, and D18's
   handler `return` does not say where the value goes.

7. **The end of a producer's pass.** Epoch-based triggers need to know that
   a pass over the data ended. A stream carries items only; the outline
   could express it only by putting an epoch number on every item.

8. **Stopping on an interrupt and still saving.** `stop` closes one
   activity's admission when another settles, and cancellation closes
   admission for everything. Neither lets an outside signal end training in
   order and then run the final save.

### Spelling-level, for the milestone that builds the words

9. Whether `choice` yields a value, needed for cached-or-live with a miss.
10. Reading a unit's progress as a value. The outline passes the repeat's
    index, which is not the unit's progress once accumulation or resume is
    involved.
11. Whether sample images are written through `persist` or as an
    implementation's declared output.
12. An access over several resources at once. Per-participant scopes put the
    step seven levels deep in this run.
