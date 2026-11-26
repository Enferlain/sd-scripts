"""Live sketch of an SDXL fine-tune strategy, built during the authoring discussion (bead sd-scripts-syv.77).

Not run, and not part of the new system yet. Names marked FAKE do not exist; the
point is the mechanics. OPEN marks a question we have not settled, PROPOSED a
proposal not yet agreed.

What topic 1 settled, as it shows here:

- Existing library code is imported and called as ordinary Python. Nothing in the
  libraries knows about this strategy or the IR.
- The wiring in an activity is each piece's role: which values go into a call and
  where its result goes.
- What the system needs to know about a piece of code comes from what it already
  knows about that code, kept by the code's identity, or from what this strategy
  states. Where it knows nothing and the strategy says nothing, it asks the author.
- The strategy may still state a piece's intended role, overriding what the system
  knows.

Topic 2, what the strategy states because the wiring cannot show it. The foundation:

- The strategy states what it computes, and can state requirements on how it runs
  when it genuinely needs to (how is OPEN, below). A requirement means the same
  wherever it is stated.
- Everything else is delegated to the system, which decides it within those
  requirements and the run settings' limits.
- Whatever the system chooses has to be established to preserve what the run
  requires. An alternative is never assumed equivalent because its mathematical
  intent is the same.
- The system records its decisions, what they were based on, and evidence of what
  actually ran, not only what it planned.

Within it:

- Precision (agreed): a recipe chosen per participant. Which recipes fit follows
  from how the strategy uses the participant.
- Module mode (D20, unchanged): a call runs a module in training mode only where
  the module belongs to a unit whose loss the call reaches, and in evaluation mode
  otherwise, so a frozen network on the gradient path stays in evaluation mode.
  Annotations can ask for other behavior, such as dropout at inference.
- Gradients (PROPOSED): they follow the data flow from what a unit trains to what
  it contributes, and the system works out from that which calls record them. Only
  a value that depends on something trained but should count as a constant is
  marked, with iro.train.fixed.
- Random numbers: that a call draws them is known about its code. Keeping each
  drawing call's numbers independent of the others is a capability we want. OPEN:
  how a stream gets its identity, how repeated samples and resumes are keyed, how
  library code that keeps its own generator state fits, and how calls are isolated.
- Which weights a reader sees: the strategy states the version (`current` and
  `at_request` below are placeholder names); how the system provides it is
  delegated.
- OPEN: how an author expresses an execution requirement, and what one can be
  about. It needs a precise meaning and has to work through the system's general
  understanding of activities, their dependencies and resources, not one flag per
  pair of activities. Nothing shows yet that this strategy needs one.
- OPEN: how the system chooses among the ways it could run something (no way is a
  privileged fallback; a run is refused only when none satisfies the
  requirements), what equivalence and reproducibility guarantees can be given and
  at what scope, and how execution adapts during a run, is coordinated across
  devices and is recorded.
"""

from library.iro import frontend as iro  # FAKE: strategy, producer, activity, handler, stream, parallel, stop, random, report and train come with section 17

# Existing code, used as it is. Piece 1's versions still carry IR declarations;
# they become plain functions when piece 1 is redone.
from library.losses.elementwise import l2_f64
from library.losses.reductions import per_sample_mean, sample_mean, weigh
from library.losses.weightings import min_snr_v_weight
from library.objectives.formulations.ddpm import ddpm_noise, ddpm_snr, ddpm_v_target
from library.objectives.schedules import zero_terminal_snr

from library.data import cached  # FAKE: batches of cached latents, text conditioning and loss weights
from library.data.noise import gaussian_like  # FAKE: noise shaped like the latents
from library.models.assemblies import sdxl  # FAKE: SDXL's UNet, its released noise schedule and its conditioning layout (piece 2)
from library.optimization import OptimizerChoice, adamw8bit  # FAKE: the optimization library's settings type for a chosen optimizer, and bitsandbytes' AdamW8bit as one choice
from library.precision import Precision, bf16_autocast  # FAKE: the role for a precision recipe, and one recipe: weights stored in bf16, calls under bf16 autocast
from library.strategies.features import AdapterChoice, attach_adapter  # FAKE: a chosen adapter method with its own settings, and the reusable piece that attaches it
from library.strategies.roles import ElementwiseLoss, TimestepSampler  # FAKE: parameters configuration fills with any code the system knows for that role
from library.timesteps import uniform  # FAKE: one training timestep per sample


@iro.producer
def produce(dataset):
    for batch in cached.batches(dataset):
        yield batch


@iro.activity
def train(batches, unet, unit, loss, timesteps, gamma, steps, validate_every):
    schedule = zero_terminal_snr(sdxl.schedule())  # SDXL's released schedule, rescaled to zero terminal SNR: a value every call below reads
    for step in range(steps):
        for batch in unit.window(batches, members=1):
            # Both calls draw random numbers, which is known about their code, so nothing is marked. Keeping their numbers
            # independent, so that swapping the timestep sampler never changes the noise, is a capability we want; the
            # mechanism is OPEN.
            t = timesteps(batch.size, schedule)
            noise = gaussian_like(batch.latents)
            noisy = ddpm_noise(schedule, batch.latents, noise, t)
            # Gradients follow the data flow (PROPOSED): denoise and every call after it that takes its result record
            # them, because they lie between what the unit trains and what it contributes. The samplers, the noising,
            # the target and the min-SNR weight lie off that path and don't. Module mode is separate (D20): the UNet's
            # modules run in training mode because they belong to the unit whose loss this call reaches. With an adapter
            # only the adapter's modules do, and the frozen UNet stays in evaluation mode while gradients pass through it.
            # `current` is the version the unit is working on.
            prediction = sdxl.denoise(unet.current, noisy, t, batch.conditioning)
            # No mark: nothing trained feeds the target, so no gradient can reach it. A value that does depend on
            # something trained but should count as a constant is marked, for example a prediction from the weights
            # being trained, used as a target (self-distillation): iro.train.fixed(sdxl.denoise(unet.current, ...)).
            # A teacher that is a separate frozen copy needs no mark.
            target = ddpm_v_target(schedule, batch.latents, noise, t)
            losses = per_sample_mean(loss(prediction, target))
            losses = weigh(losses, batch.loss_weights)
            losses = weigh(losses, min_snr_v_weight(ddpm_snr(schedule, t), gamma))
            unit.contribute(sample_mean(losses))  # a loss with its mass, the number of samples
        if step % validate_every == 0:
            validate.request(step)  # OPEN (later topic): how work like validation is requested


@iro.handler
def validate(step, unet, validation_data, loss, timesteps):
    schedule = zero_terminal_snr(sdxl.schedule())
    # A requirement on results: the same timesteps and noise every time validation runs, and in every run, so its losses
    # compare across steps and runs. How the streams are identified and kept apart is OPEN, so this form is a placeholder.
    with iro.random.streams("validation", seed=0, repeat=True):
        for batch in cached.batches(validation_data):
            t = timesteps(batch.size, schedule)
            noise = gaussian_like(batch.latents)
            noisy = ddpm_noise(schedule, batch.latents, noise, t)
            # The weights of the step that asked: that is the requirement. Whether training waits meanwhile or a copy is
            # kept is the system's decision, and a copy has to hold the full state the call reads (buffers, precision
            # scales), not only the parameters. Evaluation mode: no module this call runs belongs to a unit whose loss
            # it reaches (D20).
            prediction = sdxl.denoise(unet.at_request, noisy, t, batch.conditioning)
            target = ddpm_v_target(schedule, batch.latents, noise, t)
            iro.report("validation_loss", sample_mean(per_sample_mean(loss(prediction, target))), step=step)  # OPEN (later topic): reporting


@iro.strategy
def train_sdxl(
    model_path,
    dataset,
    validation_data,  # images kept out of training, so their loss shows how the model does on data it doesn't learn from
    steps: int = 2000,
    validate_every: int = 250,
    optimizer: OptimizerChoice = adamw8bit(lr=1e-5, weight_decay=0.01, betas=(0.9, 0.999)),
    gamma: float = 5.0,
    loss: ElementwiseLoss = l2_f64,
    timesteps: TimestepSampler = uniform,
    adapter: AdapterChoice | None = None,
    precision: Precision = bf16_autocast,
):
    # The parameters are all configuration can fill (PROPOSED): plain values, picks among named options,
    # and roles (loss, timesteps, adapter, optimizer, precision) filled with any code the system knows for that role.
    # A settings type such as the optimizer's belongs to its library, so a setup saved in the config
    # fits any strategy that takes that type.
    # Configuration never adds wiring. How much a strategy offers as parameters, and when to write a
    # separate strategy instead, is the author's call; configuration does not care about either.
    unet = sdxl.unet(model_path, precision=precision)  # declares the participant and how it loads; nothing loads here
    # Which precision recipes fit follows from how this strategy uses the UNet: trained in a full fine-tune, frozen with
    # gradients passing through it when an adapter is attached. Config picks among those; any other is refused, naming why.
    if adapter is None:
        trained = unet  # full fine-tune
    else:
        # Any method the system knows, with that method's own settings. The methods named here are the ones
        # this strategy maintains; another method that fits the UNet still runs, reported as outside that set.
        trained = attach_adapter(unet, adapter, maintained=("lora", "locon", "lokr"))  # the UNet stays frozen
    unit = iro.train.unit("denoiser", of=trained, optimizer=optimizer, max_grad_norm=1.0)
    # OPEN (later topic): what gets saved, the full model or the adapter file.
    # No prefetch depth: how far the producer runs ahead, whether a frozen part leaves the GPU between uses, whether
    # activities share a GPU, recomputing activations and keeping or releasing intermediate values are delegated to the
    # system. It may choose any way it can establish preserves what the run requires: data order, freshness and the
    # producer's state, random state, precision and the full state of whatever it moves. The strategy or the run
    # settings may still state requirements about any of them.
    batches = iro.stream("batches")
    with iro.parallel():
        produce(dataset, into=batches)
        train(batches, unet, unit, loss, timesteps, gamma, steps, validate_every)
        validate.serve(unet=unet, validation_data=validation_data, loss=loss, timesteps=timesteps)
        iro.stop(produce, after=train)
