"""Trainer optimization on the host: a window of contributions and its advancement, run through the Trainer (D17, D20).

A small Torch network is trained for one window of two members, through the
whole host sequence, and compared exactly with the same step written by
hand. A frozen encoder before the trained network runs without grad mode,
since no trained state reaches it, and in evaluation mode, since module mode
follows trainability (D15, D20).
"""

import copy
import dataclasses
import re
import threading
import time
from contextlib import contextmanager
from types import MappingProxyType

import pytest
import torch

from library.iro.frontend import implementation
from library.iro.prepare import Target
from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.diagnostics import Rejection
from library.iro.operation_contracts import AccessMode, VersionSelection
from library.iro.verify import CORE_EXECUTION
from library.runstate.authority import RunAuthority, Unavailable
from library.runstate.generation import ExecutableRun, Publication
from library.runstate.owners import Refused, Stale
from library.training.trainer import Trainer
from library.iro.seams import Use, annotate
from library.training.units import (
    TRAINER,
    ModuleModes,
    OptimizationPlan,
    OptimizationSection,
    Outcome,
    Progress,
    TrainerOptimization,
)
from library.training.units import DERIVATIVE, UNITS
from library.training.units.standard import installed as standard
from library.training.words import contract
from tests.unit.iro.accepted_runs import accepted_run

STEP = """iro.strategy @step uses(@batch v1, @encode v1, @predict v1, @objective v1, @standard.contribute v2, @standard.advance v2) {
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

    @implementation(
        version=1, inputs=("protected_current lease", "batch"), outputs=("retained cond",), name="encode", relations={DERIVATIVE: [(0, 0)]}
    )
    def encode(encoder, batch):
        seen["encoder"].append((encoder.training, torch.is_grad_enabled()))
        return encoder(batch[0])

    @implementation(
        version=1,
        inputs=("protected_current lease", "retained cond"),
        outputs=("retained prediction",),
        name="predict",
        relations={DERIVATIVE: [(0, 0), (1, 0)]},
    )
    def predict(network, cond):
        seen["network"].append((network.training, torch.is_grad_enabled()))
        return network(cond)

    @implementation(
        version=1, inputs=("retained prediction", "batch"), outputs=("retained loss",), name="objective", relations={DERIVATIVE: [(0, 0)]}
    )
    def objective(prediction, batch):
        return Massive(torch.nn.functional.mse_loss(prediction, batch[1]), float(batch[1].numel()))  # its mean divides by every element

    @implementation(version=1, inputs=("protected_current lease",), outputs=("int",), name="measure")
    def measure(network):
        seen.setdefault("measure", []).append((network.training, torch.is_grad_enabled()))
        return 1

    return {"batch": batch, "encode": encode, "predict": predict, "objective": objective, "measure": measure, **on_host()}


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
            relations=dict(declared.relations),
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
    """The same window written directly: two accumulated losses of equal mass divided once by the window's, clipping, one AdamW step."""
    for parameter in enc.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(net.parameters()), lr=0.1, weight_decay=0.0)
    for x, y in BATCHES:
        torch.nn.functional.mse_loss(net(enc(x)), y).backward()
    for parameter in net.parameters():
        assert parameter.grad is not None
        parameter.grad.mul_(0.5)  # one batch of both: the two losses have the same mass (D20)
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
    assert seen["encoder"] == [(False, False)] * 2, (
        "a frozen encoder that no trained state reaches runs without grad mode, and keeps evaluation mode (D15, D20)"
    )
    optimization = made.optimization
    assert optimization is not None and made.authority is not None
    live = optimization.live("learner")
    assert live.progress == Progress(1, 1) and live.position == 0
    assert live.last is not None and live.last.contributions == 2 and live.last.optimizer == Outcome.RETURNED
    assert live.last.mass == 12.0, "two losses, each a mean over six elements"
    assert optimization.current("learner").ordinal == 3, "two contributions and one window end each updated the unit"
    assert made.authority.current("net").ordinal == 1 and made.authority.current("enc").ordinal == 0, (
        "only the trained participant advanced"
    )


def test_the_unit_s_prepared_run_shows_what_training_derived() -> None:
    made = trainer()
    executable = made.realize()
    text = executable.prepared.text()
    assert re.search(r"derived by\n  @training\.derive v1 code \w+\n  @training\.grad_mode v1 code \w+\n", text)
    assert "@predict, @objective (setups @training.grad_mode v1, units learner, hold @net (learner))" in text
    assert re.search(r": @encode \(hold @enc \(\)\)\n", text), "the frozen encoder's route needs no unit's modules in train mode"
    assert re.search(r": @batch\n", text), "it leases nothing and no trained state reaches it, so nothing is derived for it"
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


WINDOW_BODY = """      %batch = iro.call @batch() -> batch
      %cond = iro.access %enc protected_current {
      ^(%e):
        iro.call @encode(%e, %batch) -> retained cond
      }
      %loss = iro.access %net protected_current {
      ^(%n):
        %prediction = iro.call @predict(%n, %cond) -> retained prediction
        iro.call @objective(%prediction, %batch) -> retained loss
      }"""

TWO_UNITS = f"""iro.strategy @two uses(@batch v1, @encode v1, @predict v1, @objective v1, @standard.contribute v2, @standard.advance v2) {{
  %net = iro.resource @net : participant
  %enc = iro.resource @enc : participant
  %a = iro.train.unit @a of %net optimizer @adamw(lr 0.1)
  %b = iro.train.unit @b of %enc optimizer @adamw(lr 0.1)
  iro.sequence @once capture(%net, %enc, %a, %b) {{
    iro.train.window %a members 1 {{
{WINDOW_BODY}
      iro.train.contribute %loss to %a
    }}
    iro.train.window %b members 1 {{
{WINDOW_BODY}
      iro.train.contribute %loss to %b
    }}
  }}
}}
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


def test_a_value_that_is_not_finite_is_kept_on_the_device_and_read_only_where_the_host_reads() -> None:
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    assert made.optimization is not None
    live = made.optimization.live("learner")
    for parameter in net.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    live.position, live._mass, live._basis = 1, 1.0, 1.0  # as one contribution of mass 1 leaves it
    first = live.advance((net,))
    assert first.optimizer == Outcome.RETURNED and first.progress == Progress(1, 1), "the window end reads nothing, so it steps"
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    live.position, live._mass, live._basis = 1, 1.0, 1.0  # as one contribution of mass 1 leaves it
    second = live.advance((net,))
    assert second.optimizer == Outcome.RETURNED, "nor does the next: the host reads only where it reads the device already"
    assert live.check() == (
        "@learner: the gradient norm of window 1 was not finite; its step was taken, so the unit's parameters may no longer be finite",
    ), "the record keeps the first window that was not finite"


@pytest.mark.parametrize(
    ("text", "what"), [(STEP, "gradient norm"), (STEP.replace(" max_grad_norm 0.5", ""), "loss")], ids=["clipped", "unclipped"]
)
def test_a_run_whose_gradient_is_not_finite_fails_once_its_work_has_ended(text: str, what: str) -> None:
    net, enc = networks()
    with torch.no_grad():
        next(net.parameters()).view(-1)[0] = float("nan")  # the network's output, its loss and its gradients are no longer finite
    outcome = trainer(text, net=net, enc=enc).train()
    assert outcome.status == "failed"
    assert f"trainer optimization: @learner: the {what} of window 1 was not finite" in (outcome.error or ""), (
        "without clipping the check reads the window's loss rather than add a pass over the gradients"
    )


def test_a_newly_prepared_unit_takes_over_the_record() -> None:
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    optimization = made.optimization
    assert optimization is not None
    live = optimization.live("learner")
    for parameter in net.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    live.position, live._mass, live._basis = 1, 1.0, 1.0  # as one contribution of mass 1 leaves it
    live.advance((net,))
    optimization.generations.publish(republished(optimization))
    replaced = optimization.live("learner")
    assert replaced is not live and replaced.progress == Progress(1, 1)
    (reason,) = optimization.settle()
    assert reason.startswith("@learner: the gradient norm of window 1 was not finite")


def test_settling_is_reported_after_the_run_s_facts_and_a_settle_that_raises_fails_the_run() -> None:
    net, enc = networks()
    with torch.no_grad():
        next(net.parameters()).view(-1)[0] = float("nan")
    executable = trainer(net=net, enc=enc).realize()
    outcome = executable.execute()
    kinds = [f.kind for f in executable.facts]
    assert outcome.status == "failed" and kinds[-1] == "settle_failed", "execution reported its own end; the settle comes after"
    (settled,) = [f for f in executable.facts if f.kind == "settle_failed"]
    assert settled.owner == TRAINER and "the gradient norm of window 1 was not finite" in str(settled.values)
    made = trainer()
    executable = made.realize()
    assert made.optimization is not None

    def raising() -> tuple[str, ...]:
        raise RuntimeError("the device was lost")

    made.optimization.settle = raising  # ty: ignore[invalid-assignment]
    outcome = executable.execute()
    assert outcome.status == "failed"
    assert "trainer optimization: settling raised RuntimeError: the device was lost" in (outcome.error or "")


@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_record_on_a_device_waits_for_nothing_and_is_read_once_its_work_has_ended() -> None:
    made = trainer()
    made.realize()
    assert made.optimization is not None
    live = made.optimization.live("learner")
    device = torch.device("cuda", 1 if torch.cuda.device_count() > 1 else 0)
    with torch.cuda.device(device):
        nan, two = torch.tensor(float("nan"), device=device), torch.tensor(2.0, device=device)
        live._fold(0, two * 1)  # load these kernels now: CUDA loads a kernel lazily, waiting for the device
        torch.cuda.synchronize(device)
        live._bad = None
        torch.cuda._sleep(500_000_000)  # one kernel that keeps the device busy for a noticeable time
        live._fold(1, nan * 1)
        live._fold(2, two * 1)
        assert not torch.cuda.current_stream(device).query(), "folding queued its work after the busy kernel, and nothing waited for it"
    assert made.optimization.settle() == (
        "@learner: the gradient norm of window 1 was not finite; its step was taken, so the unit's parameters may no longer be finite",
    )


class Massive:
    """A loss with its mass, as a loss whose reduction divides by its sample count reports it (D20)."""

    def __init__(self, value: torch.Tensor, mass: float | torch.Tensor) -> None:
        self.value, self.mass = value, mass


def test_a_window_s_gradient_is_one_batch_of_its_contributions_whatever_their_masses() -> None:
    """Three windows: unequal masses, then one like it (scaled before backward), then a shorter one (divided at its end)."""
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    assert made.optimization is not None
    live = made.optimization.live("learner")
    samples = torch.randn(6, 4, generator=torch.Generator().manual_seed(3))
    targets = torch.randn(6, 2, generator=torch.Generator().manual_seed(4))
    reference = copy.deepcopy(net)
    windows = [[(0, 2), (2, 6)], [(0, 1), (1, 6)], [(0, 3)]]
    for window in windows:
        for start, end in window:
            loss = torch.nn.functional.mse_loss(net(samples[start:end]), targets[start:end], reduction="none").mean(dim=1).mean()
            live.contribute(Massive(loss, float(end - start)))
        whole = [s for start, end in window for s in range(start, end)]
        batch_loss = torch.nn.functional.mse_loss(reference(samples[whole]), targets[whole], reduction="none").mean(dim=1).mean()
        expected = torch.autograd.grad(batch_loss, list(reference.parameters()))
        got = [p.grad.clone() for p in net.parameters() if p.grad is not None]  # before the window end: the contributions' sum
        assert live._basis is not None
        factor = live._basis / live._mass
        for g, e in zip(got, expected, strict=True):
            torch.testing.assert_close(g * factor, e, rtol=1e-5, atol=1e-6)
        live.advance((net,))
        reference.load_state_dict(net.state_dict())
    assert live._expected == 3.0, "the next window expects the last one's total"


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


def test_module_mode_follows_the_units_a_call_is_annotated_with_and_changes_only_where_it_differs() -> None:
    whole, part = ModuleModes({"learner": ("",)}), ModuleModes({"learner": ("block",)})
    model = Counted()
    model.sets = 0
    evaluating, learning, elsewhere = (), annotate({UNITS: ["learner"]}), annotate({UNITS: ["other"]})
    whole.hold(evaluating, model)
    assert not model.training and not model.block.training, "a call that serves no unit runs the participant in evaluation mode"
    whole.hold(evaluating, model)
    assert model.sets == 1, "the mode is set only where the previous call left a different one"
    whole.hold(learning, model)
    assert model.training and model.block[1].training and model.head.training, "a participant its unit trains whole runs in train mode"
    whole.hold(elsewhere, model)
    assert not model.training, "a call serving another unit leaves this participant in evaluation mode"
    part.hold(learning, model)
    assert model.block.training and model.block[1].training, "the part the unit trains runs in train mode"
    assert not model.training and not model.head.training, "and the rest of the participant in evaluation mode"
    whole.hold(evaluating, model)
    assert not model.block.training, "another route's call finds what this one left and sets it back"
    model.train(True)
    whole.hold(evaluating, model)
    assert not model.training, "a mode changed outside a call is set again"
    whole.hold(learning, object())  # a handle that is no module has no mode


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


def republished(optimization: TrainerOptimization) -> Publication:
    """A publication of the next optimization section, each prepared unit replaced by an equal new one."""
    current = optimization.section
    assert current is not None
    units = MappingProxyType({name: dataclasses.replace(unit) for name, unit in current.units.items()})
    return Publication(optimization.generations.basis("o"), (OptimizationSection(current.revision + 1, units),))


def test_a_publication_changing_a_unit_is_refused_while_its_lease_is_out_and_installs_after() -> None:
    made = trainer()
    made.realize()
    optimization = made.optimization
    assert optimization is not None and optimization.section is not None
    before = optimization.section.units["learner"]
    use = Use("learner", AccessMode.OWNER_UPDATE, VersionSelection.CURRENT_AT_START)
    (lease,) = optimization.lease([use])
    lease.updated()
    with pytest.raises(Refused, match="@learner has a lease out"):
        optimization.generations.publish(republished(optimization))
    assert optimization.section.units["learner"] is before
    lease.release()
    optimization.generations.publish(republished(optimization))
    after = optimization.section.units["learner"]
    assert after is not before
    (lease,) = optimization.lease([use])
    assert lease.handle.prepared is after, "a lease taken after the publication reaches the newly prepared unit"
    assert lease.version.ordinal == 1, "the unit's update count carries over to the newly prepared unit"
    lease.release()


def test_a_prepared_unit_never_changes_under_a_lease_held_on_another_thread() -> None:
    """Without the GIL: update leases taken on other threads and publications that replace the prepared unit never overlap."""
    made = trainer()
    made.realize()
    optimization = made.optimization
    assert optimization is not None
    generations = optimization.generations
    use = Use("learner", AccessMode.OWNER_UPDATE, VersionSelection.CURRENT_AT_START)
    mismatches: list[str] = []
    errors: list[BaseException] = []
    held: list[int] = []
    done = threading.Event()

    def hold() -> None:
        count = 0
        try:
            while not done.is_set():
                try:
                    (lease,) = optimization.lease([use])
                except Unavailable:
                    continue  # another thread holds the unit's one update lease
                count += 1
                for _ in range(3):
                    section = optimization.section
                    if section is None or section.units["learner"] is not lease.handle.prepared:
                        mismatches.append("the prepared unit changed under a held lease")
                    time.sleep(0.00005)
                lease.release()
                time.sleep(0.0001)  # a gap in which a publication can install
        except BaseException as failed:
            errors.append(failed)
        held.append(count)

    workers = [threading.Thread(target=hold) for _ in range(3)]
    for worker in workers:
        worker.start()
    published = refused = attempts = 0
    deadline = time.monotonic() + 10  # a slow machine takes longer; the outcomes do not change
    while (attempts < 300 or not (published and refused)) and time.monotonic() < deadline:
        attempts += 1
        try:
            generations.publish(republished(optimization))
            published += 1
        except Refused:
            refused += 1
        time.sleep(0.0001)
    done.set()
    for worker in workers:
        worker.join()
    assert errors == [] and mismatches == []
    assert published > 0 and refused > 0 and sum(held) > 0, (published, refused, held)
    assert not optimization._outstanding and not optimization._writers


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


# What training derives, and what it refuses (D15, D16).


def realized(text: str, extra: dict) -> ExecutableRun:
    installed = {**extra, **on_host()}
    run = accepted_run(text, contract(), installed=installed)
    net, _ = networks()
    return Trainer(run, contract(), installed, Target(), {"host": Host()}, supplied={"net": net}).realize()


def said(rejection: Rejection) -> str:
    """A rejection's text with its line wrapping undone."""
    return " ".join(str(rejection).split())


SAMPLED = """iro.strategy @sampled uses(@sample v1, @judge v1, @standard.contribute v2, @standard.advance v2) {
  %net = iro.resource @net : participant
  %unit = iro.train.unit @learner of %net optimizer @adamw(lr 0.1)
  %items = iro.stream @items : prediction
             bounds(outstanding 1, ready 1)

  iro.parallel @all capture(%items, %net, %unit) {
    iro.producer @sampling into %items capture(%net) {
    ^(%index: int):
      %p = iro.access %net protected_current {
      ^(%n):
        iro.call @sample(%n, %index) -> prediction
      }
      iro.yield %p
    }

    iro.sequence @training capture(%items, %net, %unit) {
      iro.train.window %unit members 1 {
        %c = iro.take %items
        %p = iro.borrow %c
        JUDGED
        iro.train.contribute %loss to %unit
        iro.dispose %c consumed
      }
    }

    iro.stop @sampling after @training
  }
}
"""

ONLY_SAMPLED = "%loss = iro.call @judge(%p) -> retained loss"
ALSO_SCORED = """%loss = iro.access %net protected_current {
        ^(%n):
          iro.call @judge(%n, %p) -> retained loss
        }"""


def sampling(judge_inputs: tuple[str, ...]) -> dict:
    sample = implementation(
        version=1, inputs=("protected_current lease", "int"), outputs=("prediction",), name="sample", relations={DERIVATIVE: [(0, 0)]}
    )(lambda n, index: None)
    judge = implementation(
        version=1,
        inputs=judge_inputs,
        outputs=("retained loss",),
        name="judge",
        relations={DERIVATIVE: [(i, 0) for i in range(len(judge_inputs))]},
    )(lambda *args: None)
    return {"sample": sample, "judge": judge}


def test_a_trained_participant_whose_forward_reaches_the_loss_only_from_another_activity_is_refused() -> None:
    with pytest.raises(Rejection) as rejected:
        realized(SAMPLED.replace("JUDGED", ONLY_SAMPLED), sampling(("prediction",)))
    assert rejected.value.codes == ("iro-train-002",)
    assert "reaches @learner's loss only from another activity" in said(rejected.value)
    assert "run @net's forward in the activity that contributes to @learner" in said(rejected.value)


def test_the_trained_model_sampling_in_another_activity_runs_without_grad_mode_while_its_forward_in_the_window_has_it() -> None:
    text = realized(SAMPLED.replace("JUDGED", ALSO_SCORED), sampling(("protected_current lease", "prediction"))).prepared.text()
    assert re.search(r"in @sampled/@all/@sampling: @sample \(hold @net \(\)\)\n", text), (
        "sampling's outputs are data to the training activity"
    )
    assert "@judge (setups @training.grad_mode v1, units learner, hold @net (learner))" in text


def test_a_trained_participant_that_reaches_its_loss_by_no_path_is_refused() -> None:
    def without_weights(seen):
        found = implementations(seen)
        declared = found["predict"].declaration
        found["predict"] = implementation(
            version=1, inputs=declared.inputs, outputs=declared.outputs, name="predict", relations={DERIVATIVE: [(1, 0)]}
        )(found["predict"].function)
        return found

    installed = without_weights({"encoder": [], "network": []})
    run = accepted_run(STEP, contract(), installed=installed)
    net, enc = networks()
    made = Trainer(run, contract(), installed, Target(), {"host": Host()}, supplied={"net": net, "enc": enc})
    with pytest.raises(Rejection) as rejected:
        made.realize()
    assert rejected.value.codes == ("iro-train-002",) and "reaches @learner's loss by no path" in said(rejected.value)


def test_a_call_on_a_path_from_trained_state_that_holds_a_lease_must_declare_its_result_retained() -> None:
    text = STEP.replace("iro.call @predict(%n, %cond) -> retained prediction", "iro.call @predict(%n, %cond) -> prediction").replace(
        "iro.call @objective(%prediction, %batch) -> retained loss", "iro.call @objective(%prediction, %batch) -> retained loss"
    )

    def plain(seen):
        found = implementations(seen)
        found["predict"] = implementation(
            version=1,
            inputs=("protected_current lease", "retained cond"),
            outputs=("prediction",),
            name="predict",
            relations={DERIVATIVE: [(0, 0), (1, 0)]},
        )(found["predict"].function)
        found["objective"] = implementation(
            version=1, inputs=("prediction", "batch"), outputs=("retained loss",), name="objective", relations={DERIVATIVE: [(0, 0)]}
        )(found["objective"].function)
        return found

    installed = plain({"encoder": [], "network": []})
    run = accepted_run(text, contract(), installed=installed)
    net, enc = networks()
    made = Trainer(run, contract(), installed, Target(), {"host": Host()}, supplied={"net": net, "enc": enc})
    with pytest.raises(Rejection) as rejected:
        made.realize()
    assert rejected.value.codes == ("iro-train-003",) and "while holding a lease on @net" in said(rejected.value)


def test_a_window_of_no_mass_steps_nothing_wrong_and_leaves_the_next_window_its_own_basis() -> None:
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    assert made.optimization is not None
    live = made.optimization.live("learner")
    x, y = BATCHES[0]
    before = [p.detach().clone() for p in net.parameters()]
    live.contribute(
        Massive(torch.nn.functional.mse_loss(net(x), y) * 0, 0.0)
    )  # a batch whose loss keeps nothing, such as a fully masked one
    empty = live.advance((net,))
    assert empty.optimizer == Outcome.NOT_ATTEMPTED and empty.progress == Progress(0, 0) and empty.mass == 0.0, "nothing to optimize"
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True)), "not even weight decay moved anything"
    assert all(p.grad is None for p in net.parameters()) and live._expected is None, (
        "its zero gradients are cleared, and it sets no expectation"
    )
    live.contribute(Massive(torch.nn.functional.mse_loss(net(x), y), 3.0))
    record = live.advance((net,))
    assert record.mass == 3.0 and record.progress == Progress(1, 1) and all(torch.isfinite(p).all() for p in net.parameters())


def test_a_loss_that_declares_no_mass_is_refused() -> None:
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    assert made.optimization is not None
    x, y = BATCHES[0]
    with pytest.raises(TypeError, match="declares its mass"):
        made.optimization.live("learner").contribute(torch.nn.functional.mse_loss(net(x), y))


def test_a_unit_that_no_contribution_offers_a_loss_is_refused() -> None:
    text = STEP.replace("  iro.sequence @once", "  %idle = iro.train.unit @idle of %enc optimizer @adamw(lr 0.1)\n  iro.sequence @once")
    with pytest.raises(Rejection) as rejected:
        trainer(text).realize()
    assert rejected.value.codes == ("iro-train-004",) and "no contribution offers @idle a loss" in said(rejected.value)


def test_a_window_whose_masses_are_on_the_device_steps_nothing_when_they_add_to_zero() -> None:
    """The same semantics wherever a mass is held: the host reads a device total at the window's end (D20)."""
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    assert made.optimization is not None
    live = made.optimization.live("learner")
    x, y = BATCHES[0]
    before = [p.detach().clone() for p in net.parameters()]
    live.contribute(Massive(torch.nn.functional.mse_loss(net(x), y) * 0, torch.tensor(0.0)))
    empty = live.advance((net,))
    assert empty.optimizer == Outcome.NOT_ATTEMPTED and empty.progress == Progress(0, 0) and empty.mass == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))
    live.contribute(Massive(torch.nn.functional.mse_loss(net(x), y), torch.tensor(6.0)))
    record = live.advance((net,))
    assert record.progress == Progress(1, 1) and record.mass == 6.0 and live._expected == 6.0, (
        "a device total read once is known on the host"
    )


def test_a_contribution_of_no_mass_contributes_nothing_even_where_its_loss_is_undefined() -> None:
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    assert made.optimization is not None
    live = made.optimization.live("learner")
    x, y = BATCHES[0]
    reference = copy.deepcopy(net)
    live.contribute(Massive(torch.nn.functional.mse_loss(net(x), y), 6.0))
    live.contribute(Massive(net(x).sum() / 0.0, 0.0))  # a fully masked batch: its mean divides by nothing
    expected = torch.autograd.grad(torch.nn.functional.mse_loss(reference(x), y), list(reference.parameters()))
    for got, wanted in zip(net.parameters(), expected, strict=True):
        assert got.grad is not None and torch.allclose(got.grad, wanted)
    assert live.advance((net,)).mass == 6.0


def test_a_first_contribution_of_no_mass_does_not_set_the_scale_the_window_s_others_use() -> None:
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    assert made.optimization is not None
    live = made.optimization.live("learner")
    x, y = BATCHES[0]
    live.contribute(Massive(torch.nn.functional.mse_loss(net(x), y) * 0, 0.0))
    live.contribute(Massive(torch.nn.functional.mse_loss(net(x), y), 6.0))
    assert live._basis == 6.0 and live.position == 2, "the first contribution with mass sets the basis, so its share is one"


def test_a_backward_that_fails_counts_no_contribution() -> None:
    net, enc = networks()
    made = trainer(net=net, enc=enc)
    made.realize()
    assert made.optimization is not None
    live = made.optimization.live("learner")
    with pytest.raises(RuntimeError):
        live.contribute(Massive(torch.tensor(1.0), 6.0))  # a loss that carries no graph cannot be differentiated
    assert live.position == 0 and live._mass == 0.0, "the window still holds nothing, so its end steps nothing rather than hide the failure"


MEASURED_FIRST = STEP.replace("@objective v1,", "@objective v1, @measure v1,").replace(
    "        %prediction = iro.call @predict(%n, %cond) -> retained prediction",
    "        %seen = iro.call @measure(%n) -> int\n        %prediction = iro.call @predict(%n, %cond) -> retained prediction",
)


def test_a_measurement_before_the_forward_in_one_access_runs_in_evaluation_mode_and_the_forward_in_train_mode() -> None:
    seen: dict[str, list] = {"encoder": [], "network": []}
    outcome = trainer(MEASURED_FIRST, seen=seen).train()
    assert outcome.status == "succeeded", outcome.error
    assert seen["measure"] == [(False, False)] * 2 and seen["network"] == [(True, True)] * 2, (
        "the open access keeps nothing that needs a mode, so the two calls take turns (D16)"
    )


def test_a_measurement_while_the_forward_s_result_is_retained_is_refused_since_it_would_wait_on_itself() -> None:
    text = STEP.replace("@objective v1,", "@objective v1, @measure v1,").replace(
        "        iro.call @objective(%prediction, %batch) -> retained loss",
        "        %seen = iro.call @measure(%n) -> int\n        iro.call @objective(%prediction, %batch) -> retained loss",
    )
    with pytest.raises(Rejection) as rejected:
        trainer(text).realize()
    assert rejected.value.codes == ("iro-prep-016",) and "would wait on itself" in said(rejected.value)
