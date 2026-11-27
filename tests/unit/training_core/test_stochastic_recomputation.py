"""Stochastic recomputation keeps RNG ownership (task 16.2, bead sd-scripts-syv.29; D15, D16).

A call that declares randomness draws from its activity's own random state on
the device. With gradient checkpointing, a block's forward runs again inside
the backward, which here runs inside a later call, the contribution. The
recompute must draw exactly what the forward drew, and neither draw from nor
advance the activity's state, nor the state of the call it runs in.

The oracle is the same window run serially by hand, without the runtime and
without checkpointing, from the activity's initial random state.

The replay itself is PyTorch's: its checkpoint stashes the generator state
the forward ran with and restores it around the recompute. What the runtime
provides is that this state is the activity's own, so the recompute replays
the activity's draws inside a call that has none, and that no other state
moves. The negative control runs the same window with the replay turned off,
which shows the oracle tells the two apart.
"""

import copy
from typing import Any

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from library.iro import gil
from library.iro.devices import Host
from library.iro.devices.torch_context import RandomStates
from library.iro.frontend import implementation
from library.iro.prepare import HOST, DeviceFacts, Target
from library.iro.seams import Effect
from library.losses.reductions import Loss
from library.runstate.authority import NORMAL
from library.training.trainer import Trainer
from library.training.units import DERIVATIVE
from library.training.units.standard import installed as standard
from library.training.words import contract
from tests.unit.iro.accepted_runs import accepted_run

pytestmark = [pytest.mark.requires_gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]

KEEP_GIL_OFF = gil.enabled() is False
OWNER = "@step/@once/@learner.window"  # the window, an activity of its own, whose random state the forward draws from

STEP = """iro.strategy @step uses(@batch, @predict, @objective, @standard.contribute, @standard.advance) {
  %net = iro.resource @net : participant
  %unit = iro.train.unit @learner of %net optimizer @adamw(lr 0.01, weight_decay 0.0)
  iro.sequence @once capture(%net, %unit) {
    iro.train.window %unit members 1 {
      %batch = iro.call @batch() -> batch
      %loss = iro.access %net protected_current {
      ^(%n):
        %prediction = iro.call @predict(%n, %batch) -> retained prediction
        iro.call @objective(%prediction, %batch) -> retained loss
      }
      iro.train.contribute %loss to %unit
    }
  }
}
"""


class Noisy(torch.nn.Module):
    """Two layers with dropout between them, in one block that may recompute inside the backward."""

    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(8, 64)
        self.drop = torch.nn.Dropout(0.5)
        self.second = torch.nn.Linear(64, 2)
        self.checkpointed = False
        self.replays = True  # PyTorch's checkpoint restores the forward's generator state for its recompute
        self.recompute_fails = False
        self.runs = 0

    def block(self, x: torch.Tensor) -> torch.Tensor:
        self.runs += 1
        dropped = self.drop(torch.relu(self.first(x)))
        if self.recompute_fails and self.runs > 1:  # after the recompute drew its dropout mask
            raise RuntimeError("the recompute failed")
        return self.second(dropped)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpointed and torch.is_grad_enabled():
            return checkpoint(self.block, x, use_reentrant=False, preserve_rng_state=self.replays)
        return self.block(x)


BATCH: list[tuple[torch.Tensor, torch.Tensor]] = []  # module state, which no code record covers


def batch():
    return BATCH[0]


def predict(network, b):
    return network(b[0])


def objective(prediction, b):
    return Loss(torch.nn.functional.mse_loss(prediction, b[1]), float(b[1].shape[0]))


def installed() -> dict:
    derivative = {DERIVATIVE: [(0, 0)]}
    return {
        "batch": implementation(outputs=("batch",), name="batch")(batch),
        "predict": implementation(
            inputs=("protected_current lease", "batch"),
            outputs=("retained prediction",),
            effects=[Effect.DEVICE, Effect.RANDOMNESS],
            name="predict",
            relations=derivative,
        )(predict),
        "objective": implementation(
            inputs=("retained prediction", "batch"),
            outputs=("retained loss",),
            effects=[Effect.DEVICE],
            name="objective",
            relations=derivative,
        )(objective),
        **standard(),
    }


def index() -> int:
    return 1 if torch.cuda.device_count() > 1 else 0


def model() -> Noisy:
    torch.manual_seed(0)
    return Noisy().to(torch.device("cuda", index()))


def through_the_trainer(net: Noisy, checkpointed: bool, replays: bool = True) -> tuple[dict[str, torch.Tensor], Any]:
    """One window through the Trainer; the device it ran on, for its random states."""
    from library.iro.devices.cuda import Cuda

    device = Cuda(index())
    generator = torch.Generator().manual_seed(1)
    BATCH[:] = [(torch.randn(16, 8, generator=generator).to(device.device), torch.randn(16, 2, generator=generator).to(device.device))]

    def prepare(network: object) -> object:
        assert isinstance(network, Noisy)
        network.checkpointed = checkpointed
        network.replays = replays
        return network

    implementations = installed()
    run = accepted_run(STEP, contract(), installed=implementations)
    trainer = Trainer(
        run,
        contract(),
        implementations,
        Target((HOST, DeviceFacts.of(device)), device=device.name),
        {"host": Host(), device.name: device},
        supplied={"net": net},
        preparations={("net", NORMAL): prepare},
        keep_gil_off=KEEP_GIL_OFF,
    )
    outcome = trainer.train()
    assert outcome.status == "succeeded", outcome.error
    return {name: p.detach().clone() for name, p in net.named_parameters()}, device


def by_hand(net: Noisy) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The oracle: the same window serially, without checkpointing, the forward drawing from the activity's initial state."""
    cuda = torch.device("cuda", index())
    states = RandomStates(cuda, seed=0)
    x, y = BATCH[0]
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0.0)
    net.train()
    with states.drawing(OWNER):
        prediction = net(x)
    torch.nn.functional.mse_loss(prediction, y).backward()
    optimizer.step()
    return {name: p.detach().clone() for name, p in net.named_parameters()}, states.states[OWNER]


def test_a_recomputed_stochastic_forward_draws_what_the_forward_drew_and_takes_no_one_s_random_progress() -> None:
    start = model()
    default_before = torch.cuda.get_rng_state(torch.device("cuda", index()))
    checkpointed, device = through_the_trainer(copy.deepcopy(start), checkpointed=True)
    default_after = torch.cuda.get_rng_state(torch.device("cuda", index()))
    plain, _ = through_the_trainer(copy.deepcopy(start), checkpointed=False)
    expected, owner_after = by_hand(copy.deepcopy(start))

    assert all(torch.equal(checkpointed[n], expected[n]) for n in expected), "the recompute drew exactly what the forward drew"
    assert all(torch.equal(plain[n], expected[n]) for n in expected), "and checkpointing changes nothing"
    assert set(device.random.states) == {OWNER}, "only the activity whose call declares randomness has random progress"
    assert torch.equal(device.random.states[OWNER], owner_after), (
        "the activity's state advanced by the forward's draws only: the recompute neither drew from it nor advanced it"
    )
    assert torch.equal(default_before, default_after), "the device's default state, which the contribution runs with, is untouched"


def test_without_the_replay_the_recompute_draws_other_numbers_and_the_oracle_says_so() -> None:
    """The negative control: the same window, its recompute drawing from whatever state is current in the call it runs in."""
    start = model()
    unreplayed, _ = through_the_trainer(copy.deepcopy(start), checkpointed=True, replays=False)
    expected, _ = by_hand(copy.deepcopy(start))
    assert not all(torch.equal(unreplayed[n], expected[n]) for n in expected)


def test_a_recompute_that_fails_leaves_the_activity_the_random_progress_its_forward_made() -> None:
    """The recompute draws its mask from the replayed state and then fails, inside the contribution's backward."""
    from library.iro.devices.cuda import Cuda

    start = model()
    device = Cuda(index())
    generator = torch.Generator().manual_seed(1)
    BATCH[:] = [(torch.randn(16, 8, generator=generator).to(device.device), torch.randn(16, 2, generator=generator).to(device.device))]
    implementations = installed()
    net = copy.deepcopy(start)
    net.checkpointed = True
    net.recompute_fails = True
    run = accepted_run(STEP, contract(), installed=implementations)
    default_before = torch.cuda.get_rng_state(device.device)
    outcome = Trainer(
        run,
        contract(),
        implementations,
        Target((HOST, DeviceFacts.of(device)), device=device.name),
        {"host": Host(), device.name: device},
        supplied={"net": net},
        keep_gil_off=KEEP_GIL_OFF,
    ).train()
    assert outcome.status == "failed" and "the recompute failed" in (outcome.error or "")
    assert net.runs == 2, "the forward, then the recompute that failed"
    _, owner_after = by_hand(copy.deepcopy(start))
    assert torch.equal(device.random.states[OWNER], owner_after), (
        "the forward's draws stay the activity's progress, and the recompute's do not count"
    )
    assert torch.equal(torch.cuda.get_rng_state(device.device), default_before), "and nothing else's random state moved"
    assert all(torch.equal(a, b) for a, b in zip(net.parameters(), start.parameters(), strict=True)), "nothing stepped"
