"""Trainer optimization on the host: a window of contributions and its advancement, run through the Trainer (D17, D20).

A small Torch network is trained for one window of two members, through the
whole host sequence, and compared exactly with the same step written by
hand. A frozen encoder on the gradient path shows that module mode follows
trainability, not the flow of gradients.
"""

import copy
import re
from contextlib import contextmanager

import pytest
import torch

from library.iro.frontend import implementation
from library.iro.prepare import Target
from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.diagnostics import Rejection
from library.iro.verify import CORE_EXECUTION
from library.runstate.authority import RunAuthority
from library.runstate.generation import ExecutableRun, Publication
from library.runstate.owners import Refused, Stale
from library.training.trainer import Trainer
from library.iro.seams import CallContext
from library.training.units import TRAINER, ModuleModes, OptimizationPlan, Outcome, Progress, TrainerOptimization
from library.training.units.standard import installed as standard
from library.training.words import contract
from tests.unit.iro.accepted_runs import accepted_run

STEP = """iro.strategy @step uses(@batch v1, @encode v1, @predict v1, @objective v1, @standard.contribute v1, @standard.advance v1) {
  %net = iro.resource @net : participant
  %enc = iro.resource @enc : participant
  %unit = iro.train.unit @learner of %net optimizer @adamw(lr 0.1, weight_decay 0.0) max_grad_norm 0.5
  iro.sequence @once capture(%net, %enc, %unit) {
    iro.train.window %unit members 2 {
      %batch = iro.call @batch() -> batch
      %cond = iro.access %enc protected_current {
      ^(%e):
        iro.call @encode(%e, %batch) -> retained cond
      }
      %loss = iro.access %net protected_current {
      ^(%n):
        %prediction = iro.call @predict(%n, %cond) -> retained prediction
        iro.call @objective(%prediction, %batch) -> retained loss
      }
      iro.train.contribute %loss to %unit
    }
  }
}
"""

BATCHES = [
    (torch.randn(3, 4, generator=torch.Generator().manual_seed(i)), torch.randn(3, 2, generator=torch.Generator().manual_seed(10 + i)))
    for i in range(2)
]


def networks() -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Dropout(0.0)), torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Dropout(0.0)
    )


def implementations(seen: dict[str, list]) -> dict:
    taken = iter(BATCHES)

    @implementation(version=1, outputs=("batch",), name="batch")
    def batch():
        return next(taken)

    @implementation(version=1, inputs=("protected_current lease", "batch"), outputs=("retained cond",), name="encode")
    def encode(encoder, batch):
        seen["encoder"].append((encoder.training, torch.is_grad_enabled()))
        return encoder(batch[0])

    @implementation(
        version=1,
        inputs=("protected_current lease", "retained cond"),
        outputs=("retained prediction",),
        name="predict",
        differentiable=(1,),
    )
    def predict(network, cond):
        seen["network"].append((network.training, torch.is_grad_enabled()))
        return network(cond)

    @implementation(version=1, inputs=("retained prediction", "batch"), outputs=("retained loss",), name="objective", differentiable=(0,))
    def objective(prediction, batch):
        return torch.nn.functional.mse_loss(prediction, batch[1])

    return {"batch": batch, "encode": encode, "predict": predict, "objective": objective, **on_host()}


def on_host() -> dict:
    """The standard profile's implementations declared without device use, so a host-only target runs them."""
    found = {}
    for name, made in standard().items():
        declared = made.declaration
        found[name] = implementation(
            version=declared.version,
            inputs=declared.inputs,
            outputs=declared.outputs,
            name=name,
            differentiates=declared.differentiates,
            differentiable=declared.differentiable,
        )(made.function)
    return found


def trainer(text: str = STEP, net: torch.nn.Module | None = None, enc: torch.nn.Module | None = None, seen=None) -> Trainer:
    seen = seen if seen is not None else {"encoder": [], "network": []}
    installed = implementations(seen)
    run = accepted_run(text, contract(), installed=installed)
    default_net, default_enc = networks()
    supplied = {"net": net if net is not None else default_net, "enc": enc if enc is not None else default_enc}
    return Trainer(run, contract(), installed, Target(), {"host": Host()}, supplied=supplied)


def by_hand(net: torch.nn.Module, enc: torch.nn.Module) -> torch.nn.Module:
    """The same window written directly: two accumulated losses, clipping, one AdamW step."""
    for parameter in enc.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(net.parameters()), lr=0.1, weight_decay=0.0)
    for x, y in BATCHES:
        torch.nn.functional.mse_loss(net(enc(x)), y).backward()
    torch.nn.utils.clip_grad_norm_(list(net.parameters()), 0.5)
    optimizer.step()
    return net


def test_one_window_through_the_trainer_matches_the_same_step_by_hand() -> None:
    net, enc = networks()
    expected = by_hand(copy.deepcopy(net), copy.deepcopy(enc))
    seen: dict[str, list] = {"encoder": [], "network": []}
    made = trainer(net=net, enc=enc, seen=seen)
    outcome = made.train()
    assert outcome.status == "succeeded", outcome.error
    for got, wanted in zip(net.parameters(), expected.parameters(), strict=True):
        assert torch.equal(got, wanted)
    assert all(p.grad is None for p in net.parameters()), "the window end zeroes the gradients"
    assert all(not p.requires_grad for p in enc.parameters()) and all(p.requires_grad for p in net.parameters())
    assert seen["network"] == [(True, True)] * 2, "the trained network runs in train mode, with grad mode"
    assert seen["encoder"] == [(False, True)] * 2, "a frozen network on the gradient path keeps evaluation mode (D20)"
    optimization = made.optimization
    assert optimization is not None and made.authority is not None
    live = optimization.live("learner")
    assert live.progress == Progress(1, 1) and live.position == 0
    assert live.last is not None and live.last.contributions == 2 and live.last.optimizer == Outcome.RETURNED
    assert optimization.current("learner").ordinal == 3, "two contributions and one window end each updated the unit"
    assert made.authority.current("net").ordinal == 1 and made.authority.current("enc").ordinal == 0, (
        "only the trained participant advanced"
    )


def test_the_unit_s_prepared_run_shows_grad_mode_and_contribution() -> None:
    made = trainer()
    executable = made.realize()
    text = executable.prepared.text()
    assert "@encode (grad True, derivatives to @learner)" in text and "@batch (grad False)" in text
    assert executable.readiness.judges == ("run authority", TRAINER)


def test_an_executable_run_runs_once() -> None:
    executable = trainer().realize()
    assert executable.execute().status == "succeeded"
    with pytest.raises(Refused, match="single-use"):
        executable.execute()


def test_a_unit_whose_members_resolve_to_no_parameters_is_refused() -> None:
    with pytest.raises(Refused, match="@learner: its members resolve to no parameters"):
        trainer(net=torch.nn.Identity()).realize()


def test_a_unit_that_cannot_build_its_optimizer_is_refused_with_why() -> None:
    text = STEP.replace("@adamw(lr 0.1, weight_decay 0.0)", "@adamw(lr 0.1, momentum 0.9)")
    with pytest.raises(Refused, match="its optimizer @adamw could not be built"):
        trainer(text).realize()


def test_a_unit_has_no_prepared_unit_before_preparation() -> None:
    installed = implementations({"encoder": [], "network": []})
    run = accepted_run(STEP, contract(), installed=installed)
    authority = RunAuthority.seed(run)
    plan = OptimizationPlan.derive(cold_load(run, contract(), installed=installed).strategy, authority.incarnation)
    optimization = TrainerOptimization(authority.generations, plan, run.obligations)
    reasons = optimization.readiness(CORE_EXECUTION)
    assert reasons and all("@learner has no prepared unit" in reason for reason in reasons)


def test_weights_tied_within_a_unit_are_one_member_with_its_other_names() -> None:
    shared = torch.nn.Linear(4, 4)
    made = trainer(net=torch.nn.Sequential(shared, torch.nn.ReLU(), shared, torch.nn.Linear(4, 2)))
    made.realize()
    assert made.optimization is not None and made.optimization.section is not None
    members = {m.name: m.aliases for m in made.optimization.section.units["learner"].members}
    assert members["@net.0.weight"] == ("@net.2.weight",) and "@net.2.weight" not in members


TWO_UNITS = """iro.strategy @two {
  %net = iro.resource @net : participant
  %enc = iro.resource @enc : participant
  %a = iro.train.unit @a of %net optimizer @adamw(lr 0.1)
  %b = iro.train.unit @b of %enc optimizer @adamw(lr 0.1)
}
"""


def test_weights_two_units_resolve_to_are_refused_as_overlapping_ownership() -> None:
    shared = torch.nn.Linear(4, 4)
    made = trainer(TWO_UNITS, net=torch.nn.Sequential(shared), enc=torch.nn.Sequential(shared))
    with pytest.raises(Refused, match=re.escape("@b: @enc.0.weight is also trained by @a as @net.0.weight")):
        made.realize()


def test_a_unit_of_two_participants_is_refused_at_its_window_end_until_declarations_repeat_lease_inputs() -> None:
    text = STEP.replace("iro.train.unit @learner of %net optimizer", "iro.train.unit @learner of %net, %enc optimizer")
    with pytest.raises(Rejection, match=r"iro-prep-001(.|\n)*@standard\.advance"):
        trainer(text).realize()


def test_a_window_end_with_nothing_contributed_steps_nothing() -> None:
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    assert made.optimization is not None
    before = [p.detach().clone() for p in net.parameters()]
    record = made.optimization.live("learner").advance((net,))
    assert record.optimizer == Outcome.NOT_ATTEMPTED and record.progress == Progress(0, 0)
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))


def test_preparation_that_stops_partway_leaves_no_in_place_registration() -> None:
    def interrupted(module):
        raise KeyboardInterrupt

    made = trainer()
    made.preparations[("net", "normal")] = interrupted
    with pytest.raises(KeyboardInterrupt):
        made.realize()
    assert made.authority is not None and made.authority._in_place == {}


class Counted(torch.nn.Module):
    """A module tree that counts how often its mode is set."""

    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
        self.head = torch.nn.Dropout(0.5)
        self.sets = 0

    def train(self, mode: bool = True) -> Counted:
        self.sets += 1
        return super().train(mode)


def test_module_mode_follows_the_units_a_call_s_derivatives_go_to_and_changes_only_where_it_differs() -> None:
    whole, part = ModuleModes({"learner": ("",)}), ModuleModes({"learner": ("block",)})
    model = Counted()
    model.sets = 0
    evaluating, learning, elsewhere = (
        CallContext(),
        CallContext(grad=True, derivatives_to=("learner",)),
        CallContext(grad=True, derivatives_to=("other",)),
    )
    whole.establish(evaluating, model)
    assert not model.training and not model.block.training, "a call whose derivatives go nowhere runs the participant in evaluation mode"
    whole.establish(evaluating, model)
    assert model.sets == 1, "the mode is set only where the previous call left a different one"
    whole.establish(learning, model)
    assert model.training and model.block[1].training and model.head.training, "a participant its unit trains whole runs in train mode"
    whole.establish(elsewhere, model)
    assert not model.training, "derivatives going to another unit leave this participant in evaluation mode"
    part.establish(learning, model)
    assert model.block.training and model.block[1].training, "the part the unit trains runs in train mode"
    assert not model.training and not model.head.training, "and the rest of the participant in evaluation mode"
    whole.establish(evaluating, model)
    assert not model.block.training, "another route's call finds what this one left and sets it back"
    model.train(True)
    whole.establish(evaluating, model)
    assert not model.training, "a mode changed outside a call is set again"
    whole.establish(learning, object())  # a handle that is no module has no mode


class Plain:
    """A section of a stand-in owner."""

    owner = "elsewhere"

    def __init__(self, revision: int) -> None:
        self.revision = revision

    def revision_of(self, key: str) -> int | None:
        return None


class Admits:
    @contextmanager
    def admit(self, publication, current, next):
        yield


def test_an_executable_run_never_executes_after_its_generation_moves_on() -> None:
    made = trainer()
    executable = made.realize()
    generations = executable.generations
    generations.register("elsewhere", Admits())
    generations.publish(Publication(generations.basis("o"), (Plain(1),)))
    with pytest.raises(Stale, match=f"realized on generation {executable.generation}"):
        executable.execute()


def test_a_run_whose_owners_are_not_ready_is_refused_with_every_reason() -> None:
    made = trainer()
    executable = made.realize()

    class NotReady:
        def readiness(self, checkpoint) -> tuple[str, ...]:
            return ("@learner has no prepared unit",)

    with pytest.raises(Refused) as refused:
        ExecutableRun(executable.generations, executable.prepared, {}, {}, {"host": Host()}, {"later": NotReady()})
    assert "@net has no owner to serve it" in str(refused.value) and "later: @learner has no prepared unit" in str(refused.value)
