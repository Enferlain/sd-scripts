"""A participant's route realizations, such as mixed precision (D15, D16).

The participant declares ``realize(@mixed_precision, "bfloat16")``; preparation
builds the realization installed under that name with those settings as the
route's call setup, which the runtime enters around every call through the
route, and only those calls.
"""

import pytest
import torch
from library.iro import gil
from library.iro.devices import Host
from library.iro.diagnostics import Rejection
from library.iro.frontend import implementation
from library.iro.prepare import Target
from library.iro.text import parse, print_strategy
from library.losses.reductions import Loss
from library.performance.precision import mixed_precision
from library.runstate.owners import Refused
from library.training.trainer import Trainer
from library.training.units import DERIVATIVE
from library.training.units.standard import installed as standard
from library.training.words import contract
from tests.unit.iro.accepted_runs import accepted_run

KEEP_GIL_OFF = gil.enabled() is False

TEXT = """iro.strategy @step uses(@batch, @predict, @objective, @standard.contribute, @standard.advance) {
  %net = iro.resource @net : participant REALIZE
  %unit = iro.train.unit @learner of %net optimizer @adamw(lr 0.1, weight_decay 0.0) max_grad_norm 0.5
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

SEEN: dict[str, object] = {}  # what the calls saw; module state, which no code record covers


def batch():
    return torch.randn(3, 4), torch.randn(3, 2)


def predict(network, b):
    SEEN["predict"] = (torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"))
    return network(b[0])


def objective(prediction, b):
    SEEN["objective"] = torch.is_autocast_enabled("cpu")
    return Loss(torch.nn.functional.mse_loss(prediction.float(), b[1]), 3.0)


def installed() -> dict:
    derivative = {DERIVATIVE: [(0, 0)]}
    found = {
        "batch": implementation(outputs=("batch",), name="batch")(batch),
        "predict": implementation(
            inputs=("protected_current lease", "batch"), outputs=("retained prediction",), name="predict", relations=derivative
        )(predict),
        "objective": implementation(
            inputs=("retained prediction", "batch"), outputs=("retained loss",), name="objective", relations=derivative
        )(objective),
    }
    for name, made in standard().items():  # declared without device use, so the host runs them
        declared = made.declaration
        found[name] = implementation(inputs=declared.inputs, outputs=declared.outputs, name=name, relations=dict(declared.relations))(
            made.function
        )
    return found


def trainer(realize: str, realizations: dict | None = None) -> Trainer:
    implementations = installed()
    run = accepted_run(TEXT.replace("REALIZE", realize), contract(), installed=implementations)
    supplied = {"net": torch.nn.Linear(4, 2)}
    return Trainer(
        run,
        contract(),
        implementations,
        Target(),
        {"host": Host()},
        supplied=supplied,
        keep_gil_off=KEEP_GIL_OFF,
        realizations=realizations,
    )


def test_only_the_calls_through_the_route_run_under_its_mixed_precision() -> None:
    SEEN.clear()
    outcome = trainer('realize(@mixed_precision, "bfloat16")', {"mixed_precision": mixed_precision}).train()
    assert outcome.status == "succeeded", outcome.error
    assert SEEN["predict"] == (True, torch.bfloat16), "the call the lease is passed to runs under the route's realization"
    assert SEEN["objective"] is False, "a call outside the route keeps its own precision"


def test_a_route_without_a_realization_runs_as_it_is() -> None:
    SEEN.clear()
    assert trainer("").train().status == "succeeded"
    assert SEEN["predict"][0] is False  # ty: ignore[not-subscriptable]


def test_a_realization_nothing_installs_is_refused_naming_it() -> None:
    with pytest.raises(Refused, match="@net is realized by @fp4, and no realization is installed under that name"):
        trainer('realize(@fp4, "e2m1")').realize()


def test_a_realization_its_settings_cannot_build_is_refused_with_why() -> None:
    with pytest.raises(Refused, match=r"realization @mixed_precision could not be built from \('float8',\)"):
        trainer('realize(@mixed_precision, "float8")', {"mixed_precision": mixed_precision}).realize()


def test_the_written_form_keeps_realizations_and_the_seed_carries_them() -> None:
    text = TEXT.replace("REALIZE", 'realize(@mixed_precision, "bfloat16") realize(@other, 3, true)')
    printed = print_strategy(parse(text, contract()), contract())
    assert 'realize(@mixed_precision, "bfloat16") realize(@other, 3, true)' in printed
    run = accepted_run(text, contract())
    (net,) = run.seed.participants
    assert net.realizations == (("mixed_precision", ("bfloat16",)), ("other", (3, True)))


def test_a_realization_is_named_once_and_realizes_only_a_participant() -> None:
    with pytest.raises(Rejection, match="a realization is named once"):
        parse(TEXT.replace("REALIZE", 'realize(@mixed_precision, "bfloat16") realize(@mixed_precision, "float16")'), contract())
    with pytest.raises(Rejection, match="only a participant's routes are realized"):
        parse('iro.strategy @s {\n  %w = iro.resource @w : weights realize(@mixed_precision, "bfloat16")\n}\n', contract())


def test_a_refused_realization_refuses_before_any_route_is_prepared() -> None:
    """Preparation may change the bound object in place, so nothing is prepared for an attempt a realization already refuses."""
    prepared = []
    made = trainer('realize(@mixed_precision, "float8")', {"mixed_precision": mixed_precision})
    made.preparations = {("net", "normal"): lambda network: prepared.append(network) or network}
    net = made.supplied["net"]
    before = [p.requires_grad for p in net.parameters()]  # ty: ignore[unresolved-attribute]
    with pytest.raises(Refused, match="could not be built"):
        made.realize()
    assert prepared == [], "no route preparation ran"
    assert [p.requires_grad for p in net.parameters()] == before, "and trainability was not rewritten"  # ty: ignore[unresolved-attribute]


def test_a_participant_without_a_route_has_its_realizations_checked_too() -> None:
    text = TEXT.replace("REALIZE", "").replace(
        "  %unit =", '  %spare = iro.resource @spare : participant realize(@fp4, "e2m1")\n  %unit =', 1
    )
    implementations = installed()
    run = accepted_run(text, contract(), installed=implementations)
    made = Trainer(
        run,
        contract(),
        implementations,
        Target(),
        {"host": Host()},
        supplied={"net": torch.nn.Linear(4, 2), "spare": torch.nn.Linear(4, 2)},
        keep_gil_off=KEEP_GIL_OFF,
    )
    with pytest.raises(Refused, match="@spare is realized by @fp4, and no realization is installed under that name"):
        made.realize()


class _Meta:
    device = torch.device("meta")


def test_mixed_precision_is_built_for_the_device_the_calls_run_on_and_refuses_one_without_it() -> None:
    assert mixed_precision(Host(), "bfloat16").device_type == "cpu", "the host runs Torch on the CPU"
    with pytest.raises(ValueError, match="no mixed precision on meta"):
        mixed_precision(_Meta(), "bfloat16")


def test_a_realization_or_materializer_setting_that_is_not_a_literal_fails_verification() -> None:
    """The reader only makes literals; IR built another way is checked as well, before its seed would read the settings."""
    from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
    from xdsl.utils.exceptions import VerifyException

    from library.iro.words.model import ResourceOp

    module = parse(TEXT.replace("REALIZE", 'realize(@mixed_precision, "bfloat16")'), contract())
    (resource,) = [op for op in module.walk() if isinstance(op, ResourceOp)]
    resource.realize = ArrayAttr([ArrayAttr([SymbolRefAttr("mixed_precision"), ArrayAttr([StringAttr("bfloat16")])])])
    with pytest.raises(VerifyException, match="a realization's settings are literals"):
        resource.verify()
    resource.realize = None
    resource.materialize = SymbolRefAttr("load")
    resource.settings = ArrayAttr([ArrayAttr([])])
    with pytest.raises(VerifyException, match="a materializer's settings are literals"):
        resource.verify()


def test_float16_runs_as_declared_and_scaling_its_loss_is_not_the_realization_s_choice() -> None:
    SEEN.clear()
    outcome = trainer('realize(@mixed_precision, "float16")', {"mixed_precision": mixed_precision}).train()
    assert outcome.status == "succeeded", outcome.error
    assert SEEN["predict"] == (True, torch.float16)


class _OnCard:
    def __init__(self, index: int):
        self.device = torch.device("cuda", index)


@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_bfloat16_support_is_judged_on_the_adapter_s_card_not_the_current_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """PyTorch's check reads the current card; here only card 1 'computes in bfloat16', and card 0 stays current."""
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: torch.cuda.current_device() == 1)
    with torch.cuda.device(0):
        assert mixed_precision(_OnCard(1), "bfloat16").device_type == "cuda"
        with pytest.raises(ValueError, match="cuda:0 does not compute in bfloat16"):
            mixed_precision(_OnCard(0), "bfloat16")
        assert torch.cuda.current_device() == 0, "and the current card is left as it was"
