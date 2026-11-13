"""The CUDA device adapter on a real GPU (14.2): completion, hand-off with events, protection and call settings.

These run on one GPU, the second when there are two, so the display card is
left alone. Each needs a CUDA device and is marked ``requires_gpu``.
"""

import asyncio

import pytest
import torch

from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.frontend import implementation
from library.iro.prepare import HOST, DeviceFacts, Target, prepare
from library.iro.runtime import Coordinator, Fact
from library.iro.seams import CallContext
from tests.unit.iro.accepted_runs import accepted_run

pytestmark = [pytest.mark.requires_gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]


def gpu_index() -> int:
    return 1 if torch.cuda.device_count() > 1 else 0


def busy(x: torch.Tensor, rounds: int = 300) -> torch.Tensor:
    """Queues enough work on the current stream to take a noticeable time."""
    for _ in range(rounds):
        x = torch.tanh(x @ x) * 0.5
    return x


def test_a_completion_is_done_only_when_the_device_has_finished() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    with device.context(CallContext(), (), ()):
        x = torch.randn(2048, 2048, device=device.device)
        busy(x)
        completion = device.completion()
    assert not completion.done(), "returning from the call queued the work; it has not finished"
    completion.wait()
    assert completion.done()


def test_work_on_another_stream_follows_its_completion_on_the_device() -> None:
    from library.iro.devices.cuda import Cuda

    first, second = Cuda(gpu_index()), Cuda(gpu_index())
    with first.context(CallContext(), (), ()):
        x = torch.randn(2048, 2048, device=first.device)
        y = busy(x)
        y.fill_(7.0)
        handed = first.completion()
    with second.context(CallContext(), (handed,), ()):
        total = y.sum()
        done = second.completion()
    done.wait()
    assert total.item() == 7.0 * y.numel(), "the other stream waited on the event before reading"


UPDATE_THEN_READ = """iro.strategy @s uses(@update v1, @read v1) {
  %m = iro.resource @m : participant
  iro.sequence @once capture(%m) {
    iro.access %m owner_update {
    ^(%lease):
      %u = iro.call @update(%lease) -> int
    }
    iro.access %m protected_current {
    ^(%lease):
      %v = iro.call @read(%lease) -> float
    }
  }
}
"""


class Weights:
    """An owner holding one CUDA tensor, versioned by update ordinal."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor, self.ordinal = tensor, 0

    def resources(self):
        return ("m",)

    def current(self, resource):
        from tests.unit.iro.runtime.harness import Seen

        return Seen("m", 1, "initial", self.ordinal)

    def lease(self, uses, reserved=None):
        from tests.unit.iro.runtime.harness import Granted

        return [Granted(use, self.current("m"), self) for use in uses]


def test_a_host_read_after_device_work_waits_for_its_completion_not_for_python_to_return() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    owner = Weights(torch.zeros(2048, 2048, device=device.device))

    def update(weights):
        busy(torch.randn(2048, 2048, device=weights.tensor.device), 200)
        weights.tensor.fill_(3.0)  # queued after the busy work on the device's stream
        weights.ordinal += 1
        return 0

    def read(weights):
        return float(weights.tensor[0, 0].item())

    installed = {
        "update": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), effects=["device"], name="update")(update),
        "read": implementation(version=1, inputs=("protected_current lease",), outputs=("float",), name="read")(read),
    }
    run = accepted_run(UPDATE_THEN_READ, installed=installed)
    target = Target((HOST, DeviceFacts(device.name, "cuda")), device=device.name)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    facts: list[Fact] = []
    coordinator = Coordinator(prepared, {"m": owner}, installed, {"host": Host(), device.name: device}, observe=facts.append, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    read_value = next(f.values[1] for f in facts if f.kind == "call_completed" and f.owner.endswith("/1/0"))
    assert read_value == (3.0,), "the host read followed the device update's completion"
    assert not prepared.group_of(prepared.groups[0].calls[0]).sync, "the update's result stays on the device: no wait"


def test_a_call_establishes_its_settings_on_the_device_and_restores_them() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index(), seed=5)
    module = torch.nn.Dropout(0.5).to(device.device)
    module.train(False)
    seen = {}
    with torch.enable_grad():
        with device.context(CallContext(grad=False, module_mode="train", precision="bfloat16", random="a"), (), (module,)):
            seen["grad"] = torch.is_grad_enabled()
            seen["training"] = module.training
            seen["dtype"] = (torch.ones(4, 4, device=device.device) @ torch.ones(4, 4, device=device.device)).dtype
            first = torch.rand(4, device=device.device)
        assert torch.is_grad_enabled() and not module.training, "the previous settings are restored"
    with device.context(CallContext(random="b"), (), ()):
        other = torch.rand(4, device=device.device)
    with device.context(CallContext(random="a"), (), ()):
        second = torch.rand(4, device=device.device)
    assert seen == {"grad": False, "training": True, "dtype": torch.bfloat16}
    fresh = Cuda(gpu_index(), seed=5)
    with fresh.context(CallContext(random="a"), (), ()):
        again = torch.rand(4, device=device.device)
    assert torch.equal(first.cpu(), again.cpu()), "an owner's draws do not depend on other owners' draws in between"
    assert not torch.equal(first.cpu(), second.cpu()) and not torch.equal(first.cpu(), other.cpu())


TWO_UPDATES_THEN_READ = """iro.strategy @s uses(@first v1, @second v1, @read v1) {
  %m = iro.resource @m : participant
  iro.sequence @once capture(%m) {
    iro.access %m owner_update {
    ^(%lease):
      %u = iro.call @first(%lease) -> int
    }
    iro.access %m owner_update {
    ^(%lease):
      %v = iro.call @second(%lease) -> int
    }
    iro.access %m protected_current {
    ^(%lease):
      %r = iro.call @read(%lease) -> float
    }
  }
}
"""


def test_device_work_after_device_work_is_ordered_on_the_device_without_a_wait_on_the_host() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    owner = Weights(torch.zeros(2048, 2048, device=device.device))
    seen = {}

    def first(weights):
        busy(torch.randn(2048, 2048, device=weights.tensor.device), 200)
        weights.tensor.fill_(1.0)
        weights.ordinal += 1
        return 0

    def second(weights):
        # Had the worker waited for the first call's completion, the stream would be idle by now.
        seen["stream_busy"] = not torch.cuda.current_stream(weights.tensor.device).query()
        weights.tensor.add_(1.0)
        weights.ordinal += 1
        return 0

    update = ("owner_update lease",)
    installed = {
        "first": implementation(version=1, inputs=update, outputs=("int",), effects=["device"], name="first")(first),
        "second": implementation(version=1, inputs=update, outputs=("int",), effects=["device"], name="second")(second),
        "read": implementation(version=1, inputs=("protected_current lease",), outputs=("float",), name="read")(
            lambda weights: float(weights.tensor[0, 0].item())
        ),
    }
    run = accepted_run(TWO_UPDATES_THEN_READ, installed=installed)
    target = Target((HOST, DeviceFacts(device.name, "cuda")), device=device.name)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    facts: list[Fact] = []
    coordinator = Coordinator(prepared, {"m": owner}, installed, {"host": Host(), device.name: device}, observe=facts.append, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    assert seen["stream_busy"], "the second call was queued behind the first on the device, not after a host wait"
    read = next(f.values[1] for f in facts if f.kind == "call_completed" and f.owner.endswith("/2/0"))
    assert read == (2.0,), "the host read followed both updates' completion"


def test_the_cuda_adapter_reports_its_kind_what_it_offers_and_its_memory() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    facts = DeviceFacts.of(device)
    assert (facts.name, facts.kind, facts.offers) == (f"cuda:{gpu_index()}", "cuda", frozenset({"ordering", "memory"}))
    memory = device.capacity()
    assert 0 < memory["free_bytes"] <= memory["total_bytes"]


def test_threads_entering_one_adapter_at_once_each_get_its_stream_and_restore_their_own() -> None:
    import threading

    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    wrong: list[str] = []

    def enter_many() -> None:
        own = torch.cuda.current_stream(device.device).stream_id
        for _ in range(300):
            with device.context(CallContext(), (), ()):
                if torch.cuda.current_stream(device.device).stream_id != device.stream.stream_id:
                    wrong.append("not the adapter's stream inside")
            if torch.cuda.current_stream(device.device).stream_id != own:
                wrong.append("the thread's stream was not restored")

    threads = [threading.Thread(target=enter_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not wrong, wrong[:3]
