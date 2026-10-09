"""Readable whole-run examples using one CPU target, not model-family engines.

Run with:
uv run --no-sync python -m docs_design.models-strategy-trainer.execution_candidate.examples

The tensors are real, but tiny. These are not SDXL, GAN, SAM or backend
integrations. Their differing relationships are the subject of this probe.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import torch
from torch import nn

from .graph import (
    Advance,
    Block,
    Call,
    Channel,
    Choice,
    Contract,
    Differentiate,
    Join,
    Output,
    OwnedState,
    Participant,
    Producer,
    Repeat,
    Replace,
    Requested,
    Run,
    Together,
    TwoPass,
    Unit,
    Use,
)
from .prepare import Context, Engine, GrantedAccess, ProviderContext, Ready, prepare
from .state import Product, Projection, Replacement


class Scale(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.weight * value


def scale(name: str, value: float) -> Participant:
    # Routine compatibility knowledge belongs to this maintained integration,
    # not a validator repeated by every author who selects it.
    def verify(model: nn.Module) -> None:
        if not isinstance(model, Scale) or model.weight.shape != torch.Size([]):
            raise ValueError(f"{name}: incompatible scale implementation")

    return Participant(name, lambda: Scale(value), verify)


@dataclass
class LearningState:
    limit: int
    observations: list[float] = field(default_factory=list)
    factors_used: list[float] = field(default_factory=list)
    factor: float = 1.0


@dataclass
class SourceState:
    started: bool = False
    produced: list[int] = field(default_factory=list)
    finished: bool = False


@dataclass
class ValidationState:
    results: list[tuple[int, float]] = field(default_factory=list)


@dataclass(frozen=True)
class Conditioning:
    sample: int
    caption_revision: int
    value: torch.Tensor


def correlated(product: Product, projection: Projection) -> bool:
    """A selected input policy, not universal caption fields in the engine."""
    if isinstance(product.value, Conditioning):
        return (
            product.value.sample == product.key
            and product.value.caption_revision == 1
            and product.provenance == (projection.stamps["encoder"],)
        )
    return isinstance(product.value, torch.Tensor) and not product.provenance


async def pixels(context: ProviderContext) -> None:
    state = context.states["pixels"]
    assert isinstance(state, SourceState)
    state.started = True
    # Completion order need not be consumption order. Capacity admits this
    # finite window; this is not an unbounded out-of-order queue algorithm.
    for key in (2, 1, 3):
        await context.publish("image", Product(key, torch.tensor(float(key))))
        state.produced.append(key)
        await asyncio.sleep(0)
    state.finished = True


async def captions(context: ProviderContext) -> None:
    state = context.states["captions"]
    assert isinstance(state, SourceState)
    state.started = True

    async def worker(key: int) -> None:
        async with context.use_source() as source:
            with torch.no_grad():
                value = source.models["encoder"](torch.tensor(key * 0.25)).detach().clone()
            actual = (source.stamps["encoder"],)
        # Completion/publication can be later than actual source use. The
        # product retains the actual provenance, not a version looked up later.
        await asyncio.sleep(0)
        await context.publish("conditioning", Product(key, Conditioning(key, 1, value), actual))
        state.produced.append(key)

    # Worker topology belongs to the provider, not the semantic run graph.
    async with asyncio.TaskGroup() as tasks:
        for key in (3, 1, 2):
            tasks.create_task(worker(key))
    state.finished = True


def diffusion_objective(context: Context, image: torch.Tensor, condition: Conditioning):
    state = context.states["adaptive"]
    assert isinstance(state, LearningState)
    state.factors_used.append(state.factor)
    prediction = context.models["denoiser"](image + condition.value)
    loss = (prediction - image * 0.5).square() * state.factor
    return loss, float(loss.detach())


def observe(context: Context, error: float) -> None:
    state = context.states["adaptive"]
    assert isinstance(state, LearningState)
    state.observations.append(error)
    state.factor = 1.0 / (1.0 + error)


def choose_learning(counts, state) -> Choice | None:
    assert isinstance(state, LearningState)
    count = counts.get("learn", 0)
    return Choice("learn", count + 1) if count < state.limit else None


def validate(context: Context) -> None:
    state = context.states["validation"]
    assert isinstance(state, ValidationState)
    with torch.no_grad():
        value = context.models["denoiser"](torch.tensor(1.0))
    assert isinstance(context.choice.payload, int)
    state.results.append((context.choice.payload, float(value)))


def diffusion(*, validation: bool = True) -> Run:
    learn = Block(
        "learn",
        (
            # Deliberately not listed in execution order. Value/ordering links
            # produce input -> objective -> feedback -> backward -> advancement.
            Advance("advance", "gradient", ("weights",)),
            Differentiate("backward", "loss", ("weights",), "gradient", after=("observe",)),
            Call("observe", observe, ("error",), states=("adaptive",)),
            Call(
                "objective",
                diffusion_objective,
                ("image", "condition"),
                (Output("loss", torch.Tensor), Output("error", float)),
                (Use("denoiser"),),
                ("adaptive",),
            ),
            Join(
                "correlate",
                ("image", "conditioning"),
                (Output("image", torch.Tensor), Output("condition", Conditioning)),
                correlated,
                (Use("encoder"),),
            ),
        ),
    )
    evaluation = Block("evaluate", (Call("measure", validate, uses=(Use("denoiser"),), states=("validation",), evaluation=True),))
    return Run(
        Together(
            "run",
            (
                Together(
                    "inputs",
                    (
                        Producer("pixels", ("image",), pixels, states=("pixels",)),
                        Producer("captions", ("conditioning",), captions, (Use("encoder"),), ("captions",)),
                    ),
                ),
                Repeat("learning", (learn,), choose_learning, "adaptive"),
                Requested(
                    "validation",
                    "learning",
                    evaluation,
                    lambda done: Choice("evaluate", done.key, done.position) if done.position % 2 == 0 else None,
                    enabled=validation,
                ),
            ),
        ),
        (scale("denoiser", 0.8), scale("encoder", 1.5)),
        (Unit("weights", (("denoiser", "weight"),)),),
        (
            OwnedState("adaptive", lambda: LearningState(3)),
            OwnedState("pixels", SourceState),
            OwnedState("captions", SourceState),
            OwnedState("validation", ValidationState),
        ),
        (Channel("image", 4), Channel("conditioning", 4)),
    )


def discriminator(context: Context, real: torch.Tensor) -> torch.Tensor:
    # Detached fake input means this computation advances D, not G. This is
    # algorithm-specific behavior; the engine does not know what D or G means.
    fake = context.models["generator"](torch.tensor(1.0)).detach()
    d = context.models["discriminator"]
    return (d(real) - 1).square() + d(fake).square()


def generator(context: Context) -> torch.Tensor:
    # Gradients pass THROUGH D to G, but only G belongs to this window.
    fake = context.models["generator"](torch.tensor(1.0))
    return (context.models["discriminator"](fake) - 1).square()


def alternating(counts, state) -> Choice | None:
    assert isinstance(state, LearningState)
    d, g = counts.get("D", 0), counts.get("G", 0)
    if g == state.limit:
        return None
    return Choice("D", d + 1) if d == g else Choice("G", g + 1)


def adversarial() -> Run:
    d = Block(
        "D",
        (
            Join("real", ("image",), (Output("real", torch.Tensor),), correlated),
            Call("D-objective", discriminator, ("real",), (Output("D-loss", torch.Tensor),), (Use("generator"), Use("discriminator"))),
            Differentiate("D-backward", "D-loss", ("D-unit",), "D-gradient"),
            Advance("D-advance", "D-gradient", ("D-unit",)),
        ),
    )
    g = Block(
        "G",
        (
            Call("G-objective", generator, outputs=(Output("G-loss", torch.Tensor),), uses=(Use("generator"), Use("discriminator"))),
            Differentiate("G-backward", "G-loss", ("G-unit",), "G-gradient"),
            Advance("G-advance", "G-gradient", ("G-unit",)),
        ),
    )
    return Run(
        Together(
            "run", (Producer("pixels", ("image",), pixels, states=("pixels",)), Repeat("alternating", (d, g), alternating, "alternation"))
        ),
        (scale("generator", 0.6), scale("discriminator", 0.5)),
        (Unit("D-unit", (("discriminator", "weight"),)), Unit("G-unit", (("generator", "weight"),))),
        (OwnedState("pixels", SourceState), OwnedState("alternation", lambda: LearningState(3))),
        (Channel("image", 4),),
    )


def joint() -> Run:
    def objective(context: Context):
        a, b = context.models["a"], context.models["b"]
        return (a(torch.tensor(2.0)) + b(torch.tensor(3.0)) - 1).square()

    return Run(
        Block(
            "joint",
            (
                Call("objective", objective, outputs=(Output("loss", torch.Tensor),), uses=(Use("a"), Use("b"))),
                Differentiate("shared-backward", "loss", ("A", "B"), "gradient"),
                Advance("joint-advance", "gradient", ("A", "B")),
            ),
        ),
        (scale("a", 0.5), scale("b", 0.2)),
        (Unit("A", (("a", "weight"),), learning_rate=0.01), Unit("B", (("b", "weight"),), learning_rate=0.02)),
    )


def two_pass() -> Run:
    def selected(access: GrantedAccess, models):
        model = models["model"]
        access.backward((model(torch.tensor(2.0)) - 1).square())
        access.perturb(0.1)
        access.reset()
        access.backward((model(torch.tensor(2.0)) - 1).square())
        access.restore()

    return Run(
        Block(
            "two-pass",
            (
                TwoPass("selected-two-pass", selected, (), "weights", "handback", (Use("model", "write"),)),
                Advance("retained-final-step", "handback", ("weights",)),
            ),
        ),
        (scale("model", 0.8),),
        (Unit("weights", (("model", "weight"),)),),
    )


def replacement() -> Run:
    def objective(context: Context):
        return (context.models["model"](torch.tensor(2.0)) - 1).square()

    def propose(context: Context):
        return Replacement("model", 1, Scale(0.4))

    def choose(counts, state):
        if counts.get("before", 0) == 0:
            return Choice("before")
        if counts.get("change", 0) == 0:
            return Choice("change")
        if counts.get("after", 0) == 0:
            return Choice("after")
        return None

    def learning(name: str):
        return Block(
            name,
            (
                Call(f"{name}-objective", objective, outputs=(Output("loss", torch.Tensor),), uses=(Use("model"),)),
                Differentiate(f"{name}-backward", "loss", ("weights",), "gradient"),
                Advance(f"{name}-advance", "gradient", ("weights",)),
            ),
        )

    change = Block(
        "change",
        (
            Call("propose", propose, outputs=(Output("proposal", Replacement),)),
            Replace("publish", "proposal", "model"),
        ),
    )
    return Run(
        Repeat("stages", (learning("before"), change, learning("after")), choose, "stages"),
        (scale("model", 0.8),),
        (Unit("weights", (("model", "weight"),)),),
        (OwnedState("stages", lambda: None),),
    )


async def execute(filing: Run, *, contract: Contract | None = None) -> Ready:
    accepted = (contract or Contract()).accept(filing)
    ready = prepare(accepted).publish()
    await Engine(ready).run()
    return ready


async def main() -> None:
    for name, build, contract in (
        ("diffusion-style", diffusion, Contract()),
        ("alternating D/G", adversarial, Contract()),
        ("joint units", joint, Contract()),
        ("granted two-pass", two_pass, Contract(two_pass=True)),
        ("compatible replacement", replacement, Contract()),
    ):
        ready = await execute(build(), contract=contract)
        assert ready.state.image is not None
        print(f"{name}: generation={ready.state.generation}; units={ {key: unit.steps for key, unit in ready.state.image.units.items()} }")
        print(f"  activity positions={ready.state.coordinates}")
        if name == "diffusion-style":
            print("  Lowered learn block:\n" + ready.source["learn"])


if __name__ == "__main__":
    asyncio.run(main())
