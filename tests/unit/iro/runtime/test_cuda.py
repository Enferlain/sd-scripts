"""The CUDA device adapter on a real GPU (14.2): completion, hand-off with events, protection and call settings.

These run on one GPU, the second when there are two, so the display card is
left alone. Each needs a CUDA device and is marked ``requires_gpu``.
"""

import asyncio
import time

import pytest
import torch

from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.frontend import implementation
from library.iro.prepare import HOST, DeviceFacts, Prepared, Target, prepare
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
    with device.context(CallContext(), ()):
        x = torch.randn(2048, 2048, device=device.device)
        busy(x)
        completion = device.completion()
    assert not completion.done(), "returning from the call queued the work; it has not finished"
    completion.wait()
    assert completion.done()


def test_work_on_another_stream_follows_its_completion_on_the_device() -> None:
    from library.iro.devices.cuda import Cuda

    first, second = Cuda(gpu_index()), Cuda(gpu_index())
    with first.context(CallContext(), ()):
        x = torch.randn(2048, 2048, device=first.device)
        y = busy(x)
        y.fill_(7.0)
        handed = first.completion()
    with second.context(CallContext(), (handed,)):
        total = y.sum()
        done = second.completion()
    done.wait()
    assert total.item() == 7.0 * y.numel(), "the other stream waited on the event before reading"


UPDATE_THEN_READ = """iro.strategy @s uses(@update, @read) {
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

    def advanced(self):
        self.ordinal += 1
        return self.current("m")

    def lease(self, uses, reserved=None):
        from tests.unit.iro.runtime.harness import Granted

        return [Granted(use, self.current("m"), self, self.advanced) for use in uses]


def test_a_host_read_after_device_work_waits_for_its_completion_not_for_python_to_return() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    owner = Weights(torch.zeros(2048, 2048, device=device.device))

    def update(weights):
        busy(torch.randn(2048, 2048, device=weights.tensor.device), 200)
        weights.tensor.fill_(3.0)  # queued after the busy work on the device's stream
        return 0

    def read(weights):
        return float(weights.tensor[0, 0].item())

    installed = {
        "update": implementation(inputs=("owner_update lease",), outputs=("int",), effects=["device"], name="update")(update),
        "read": implementation(inputs=("protected_current lease",), outputs=("float",), name="read")(read),
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


def test_a_call_starts_from_a_neutral_baseline_and_its_own_random_state_on_the_device() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index(), seed=5)
    seen = {}
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with device.context(CallContext(random="a"), ()):
            seen["grad"] = torch.is_grad_enabled()
            seen["dtype"] = (torch.ones(4, 4, device=device.device) @ torch.ones(4, 4, device=device.device)).dtype
            first = torch.rand(4, device=device.device)
        assert torch.is_grad_enabled() and torch.is_autocast_enabled("cuda"), "the thread's previous settings are restored"
    with device.context(CallContext(random="b"), ()):
        other = torch.rand(4, device=device.device)
    with device.context(CallContext(random="a"), ()):
        second = torch.rand(4, device=device.device)
    assert seen == {"grad": False, "dtype": torch.float32}, "a call never inherits the thread's grad mode or autocast (D16)"
    fresh = Cuda(gpu_index(), seed=5)
    with fresh.context(CallContext(random="a"), ()):
        again = torch.rand(4, device=device.device)
    assert torch.equal(first.cpu(), again.cpu()), "an owner's draws do not depend on other owners' draws in between"
    assert not torch.equal(first.cpu(), second.cpu()) and not torch.equal(first.cpu(), other.cpu())


TWO_UPDATES_THEN_READ = """iro.strategy @s uses(@first, @second, @read) {
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
        return 0

    def second(weights):
        # Had the worker waited for the first call's completion, the stream would be idle by now.
        seen["stream_busy"] = not torch.cuda.current_stream(weights.tensor.device).query()
        weights.tensor.add_(1.0)
        return 0

    update = ("owner_update lease",)
    installed = {
        "first": implementation(inputs=update, outputs=("int",), effects=["device"], name="first")(first),
        "second": implementation(inputs=update, outputs=("int",), effects=["device"], name="second")(second),
        "read": implementation(inputs=("protected_current lease",), outputs=("float",), name="read")(
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
    assert (facts.name, facts.kind, facts.offers) == (f"cuda:{gpu_index()}", "cuda", frozenset({"ordering", "copies", "memory"}))
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
            with device.context(CallContext(), ()):
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


TWO_USES = """iro.strategy @s uses(@encode, @pause) {
  %vae = iro.resource @vae : participant may_leave_device
  iro.sequence @once capture(%vae) {
    iro.access %vae protected_current {
    ^(%lease):
      %a = iro.call @encode(%lease) -> float
    }
    %p = iro.call @pause() -> int
    iro.access %vae protected_current {
    ^(%lease):
      %b = iro.call @encode(%lease) -> float
    }
  }
}
"""


class Frozen:
    """An owner holding a frozen participant's weights on the device; it names that storage for the runtime."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def resources(self):
        return ("vae",)

    def current(self, resource):
        from tests.unit.iro.runtime.harness import Seen

        return Seen("vae", 1, "initial", 0)

    def lease(self, uses, reserved=None):
        from tests.unit.iro.runtime.harness import Granted

        return [Granted(use, self.current("vae"), self.tensor) for use in uses]

    def storage(self, resource):
        return (self.tensor,)


def test_a_frozen_participant_leaves_the_device_between_uses_and_comes_back_exact() -> None:
    from library.iro.devices.cuda import Cuda

    moves: list[str] = []

    class Counting(Cuda):
        def empty(self, storage):
            moves.append("leave")
            super().empty(storage)

        def fill(self, storage):
            moves.append("return")
            return super().fill(storage)

    def run(text: str, tensor: torch.Tensor, device) -> tuple[list[tuple[object, ...]], Prepared]:
        installed = {
            "encode": implementation(inputs=("protected_current lease",), outputs=("float",), effects=["device"], name="encode")(
                lambda weights: float(weights.sum().item())
            ),
            "pause": implementation(outputs=("int",), name="pause")(lambda: time.sleep(0.05) or 0),
        }
        accepted = accepted_run(text, installed=installed)
        target = Target((HOST, DeviceFacts.of(device)), device=device.name)
        prepared = prepare(cold_load(accepted, installed=installed), installed, target)
        facts: list[Fact] = []
        coordinator = Coordinator(
            prepared, {"vae": Frozen(tensor)}, installed, {"host": Host(), device.name: device}, observe=facts.append, checked=True
        )
        outcome = asyncio.run(coordinator.run())
        assert outcome.status == "succeeded", outcome.error
        encoded = [v for f in facts if f.kind == "call_completed" and isinstance(v := f.values[1], tuple) and isinstance(v[0], float)]
        return encoded, prepared

    device = Counting(gpu_index())
    size = 64 << 20  # 64 Mi floats, 256 MiB
    weights = torch.rand(size, device=device.device)
    expected = float(weights.sum().item())
    same = weights.clone()  # for the run that keeps it on the device: after its last use, it has left
    torch.cuda.synchronize(device.device)
    before = torch.cuda.memory_allocated(device.device)
    values, prepared = run(TWO_USES, weights, device)
    torch.cuda.synchronize(device.device)
    freed = before - torch.cuda.memory_allocated(device.device)
    assert prepared.leaving == (("vae", device.name),)
    assert moves == ["leave", "return", "leave"], "it left after each use's last lease and came back for the second use"
    assert values == [(expected,), (expected,)], "it came back exact"
    assert freed >= size * 4, f"its device memory was freed after its last use ({freed} bytes)"
    staying, _ = run(TWO_USES.replace(" may_leave_device", ""), same, Cuda(gpu_index()))
    assert staying == values, "leaving changes no result"
    moves.clear()
    back_to_back = TWO_USES.replace("    %p = iro.call @pause() -> int\n", "")
    run(back_to_back, torch.rand(size, device=device.device), Counting(gpu_index()))
    assert moves == ["leave"], "leased again before its move off began, it stays; it leaves after its last use"


def test_storage_that_leaves_comes_back_whole_for_views_offsets_and_shared_storage() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    base = torch.rand(1 << 20, device=device.device)
    expected = base.clone()
    first, second = base[: 1 << 19], base[(1 << 19) + 7 :]  # two views of one storage, one with an offset
    alone = torch.rand(1000, device=device.device)[10:20]  # a slice of a storage nothing else names
    alone_expected = alone.untyped_storage().nbytes(), alone.clone()
    torch.cuda.synchronize(device.device)
    device.empty((first, second, alone))
    torch.cuda.synchronize(device.device)
    assert base.untyped_storage().nbytes() == 0 and alone.untyped_storage().nbytes() == 0
    device.fill((first, second, alone)).wait()
    assert torch.equal(base, expected), "the whole shared storage came back, not only the views"
    assert alone.untyped_storage().nbytes() == alone_expected[0] and torch.equal(alone, alone_expected[1])
    device.empty((first,))
    import pytest

    with pytest.raises(RuntimeError, match="already empty"):
        Cuda(gpu_index()).empty((first,))


def test_a_capture_copies_whole_storage_after_the_work_queued_and_before_later_work() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    base = torch.zeros(1 << 20, device=device.device)
    torch.cuda.synchronize(device.device)
    # Written on the calling thread's own stream behind busy work, as loading weights would be.
    busy(torch.randn(2048, 2048, device=device.device), 200)
    base.fill_(4.0)
    first, second = base[: 1 << 19], base[(1 << 19) + 7 :]  # two views of one storage, one with an offset
    copies, done = device.capture((first, second), ())
    with device.context(CallContext(), ()):
        base.fill_(9.0)  # the next update, queued on the device's stream after the copy
    done.wait()
    torch.cuda.synchronize(device.device)
    assert all(torch.all(c == 4.0) for c in copies), "the copy followed the write before it and preceded the one after"
    assert [c.shape for c in copies] == [first.shape, second.shape] and copies[1].storage_offset() == second.storage_offset()
    assert copies[0].untyped_storage().data_ptr() == copies[1].untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
    assert not any(c.requires_grad for c in copies)


SNAPSHOT_BETWEEN_UPDATES = """iro.strategy @s uses(@first, @look, @second) {
  %m = iro.resource @m : participant
  iro.sequence @once capture(%m) {
    iro.access %m owner_update {
    ^(%lease):
      %u = iro.call @first(%lease) -> int
    }
    iro.access %m snapshot {
    ^(%lease):
      %v = iro.call @look(%lease) -> int
    }
    iro.access %m owner_update {
    ^(%lease):
      %w = iro.call @second(%lease) -> int
    }
  }
}
"""


class Captured(Weights):
    """Weights whose snapshot lease asks the runtime to copy the tensor; the handle is the copy itself."""

    def lease(self, uses, reserved=None):
        from library.iro.operation_contracts import AccessMode
        from library.iro.seams import Capture
        from tests.unit.iro.runtime.harness import Granted

        return [
            Granted(use, self.current("m"), Capture((self.tensor,), lambda copies: copies[0]))
            if use.mode == AccessMode.SNAPSHOT
            else Granted(use, self.current("m"), self, self.advanced)
            for use in uses
        ]


def test_a_snapshot_of_device_state_is_a_copy_queued_after_the_update_before_it_without_a_host_wait() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    owner = Captured(torch.zeros(2048, 2048, device=device.device))
    seen = {}

    def first(weights):
        busy(torch.randn(2048, 2048, device=weights.tensor.device), 200)
        weights.tensor.fill_(1.0)
        return 0

    def look(snapshot):
        seen["stream_busy"] = not torch.cuda.current_stream(snapshot.device).query()
        seen["snapshot"] = snapshot
        return 0

    def second(weights):
        weights.tensor.fill_(5.0)
        return 0

    update = ("owner_update lease",)
    installed = {
        "first": implementation(inputs=update, outputs=("int",), effects=["device"], name="first")(first),
        "look": implementation(inputs=("snapshot lease",), outputs=("int",), effects=["device"], name="look")(look),
        "second": implementation(inputs=update, outputs=("int",), effects=["device"], name="second")(second),
    }
    run = accepted_run(SNAPSHOT_BETWEEN_UPDATES, installed=installed)
    target = Target((HOST, DeviceFacts.of(device)), device=device.name)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    coordinator = Coordinator(prepared, {"m": owner}, installed, {"host": Host(), device.name: device}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    torch.cuda.synchronize(device.device)
    assert seen["stream_busy"], "the copy and the call reading it were queued behind the first update, not after a host wait"
    assert torch.all(seen["snapshot"] == 1.0), "the snapshot holds the version it leased"
    assert torch.all(owner.tensor == 5.0), "the later update went to the live state, not the snapshot"


def test_a_capture_keeps_strides_copies_empty_tensors_and_refuses_storage_it_does_not_hold() -> None:
    from library.iro.devices.cuda import Cuda

    device = Cuda(gpu_index())
    matrix = torch.rand(64, 32, device=device.device)
    transposed = matrix.t()  # not contiguous
    empty = torch.empty(0, device=device.device)
    copies, done = device.capture((transposed, empty), ())
    done.wait()
    assert not copies[0].is_contiguous() and copies[0].stride() == transposed.stride() and torch.equal(copies[0], transposed)
    assert copies[1].numel() == 0 and copies[1].device == device.device
    with pytest.raises(ValueError, match="kept on the host between uses"):
        device.capture((torch.zeros(3),), ())
    with pytest.raises(TypeError, match="named by tensors"):
        device.capture(([1.0, 2.0],), ())
