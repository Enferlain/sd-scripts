"""The coordinator's own guarantees: worker threads, fused groups, failure and cancellation, device turns."""

import asyncio
import contextlib
import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from library.iro.accepted import cold_load
from library.iro.devices import Host, Returned
from library.iro.frontend import implementation
from library.iro.prepare import HOST, DeviceFacts, Target, prepare
from library.iro.prepare.derive import Derived
from library.iro.runtime import Coordinator, Fact
from library.iro.seams import annotate
from tests.unit.iro.accepted_runs import accepted_run
from tests.unit.iro.runtime.harness import Memory

CUDA = Target((HOST, DeviceFacts("cuda:0", "cuda")), workers=4)


@dataclass
class FakeDevice:
    """A device that records how many admissions' work overlaps on it."""

    name: str = "cuda:0"
    kind: str = "cuda"
    offers: frozenset[str] = frozenset()
    active: int = 0
    most: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @contextmanager
    def context(self, call, after):
        with self.lock:
            self.active += 1
            self.most = max(self.most, self.active)
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1

    def completion(self):
        return Returned()

    def capacity(self):
        return {}


def execute(text: str, installed: dict, *, resources=None, target: Target = Target(), devices=None, before: Callable | None = None):
    run = accepted_run(text, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    owner = Memory(resources or {})
    facts: list[Fact] = []
    coordinator = Coordinator(
        prepared,
        dict.fromkeys(resources or {}, owner),
        installed,
        {"host": Host(), **(devices or {})},
        observe=facts.append,
        checked=True,
    )
    if before is not None:
        before(coordinator)
    outcome = asyncio.run(coordinator.run())
    return outcome, facts, coordinator


CHAIN = """iro.strategy @s uses(@a v1, @b v1, @c v1) {
  iro.sequence @once {
    %x = iro.call @a() -> int
    %y = iro.call @b(%x) -> int
    %z = iro.call @c(%y) -> int
  }
}
"""


def test_calls_run_on_worker_threads_and_a_fused_group_on_one() -> None:
    seen: list[tuple[str, int]] = []
    loop_thread = threading.get_ident()

    def make(name, inputs, body):
        return implementation(version=1, inputs=inputs, outputs=("int",), name=name)(
            lambda *args: (seen.append((name, threading.get_ident())), body(*args))[1]
        )

    installed = {"a": make("a", (), lambda: 1), "b": make("b", ("int",), lambda x: x + 1), "c": make("c", ("int",), lambda y: y * 10)}
    outcome, facts, _ = execute(CHAIN, installed, target=Target(workers=4))
    assert outcome.status == "succeeded"
    assert [f.values[1] for f in facts if f.kind == "call_completed"] == [(1,), (2,), (20,)]
    threads = {thread for _, thread in seen}
    assert loop_thread not in threads, "selected calls never run on the coordinator's thread"
    assert len(threads) == 1, "a fused group is one submission on one worker"


PARALLEL = """iro.strategy @s uses(@slow v1, @boom v1) {
  iro.parallel @both {
    iro.sequence @left {
      %x = iro.call @slow() -> int
      %y = iro.call @slow() -> int
    }
    iro.sequence @right {
      %z = iro.call @boom() -> int
    }
  }
}
"""


def test_a_failure_fails_the_run_and_closes_admission_for_its_siblings() -> None:
    def slow():
        time.sleep(0.05)
        return 1

    def boom():
        raise ValueError("selected code failed")

    installed = {
        "slow": implementation(version=1, outputs=("int",), name="slow")(slow),
        "boom": implementation(version=1, outputs=("int",), name="boom")(boom),
    }
    outcome, facts, coordinator = execute(PARALLEL, installed, target=Target(workers=4, fuse=False))
    assert outcome.status == "failed" and "selected code failed" in (outcome.error or "")
    assert [f.values[1] for f in facts if f.kind == "call_failed"] == ["selected code failed"]
    statuses = {f.owner.rsplit("/", 1)[-1]: f.values[0] for f in facts if f.kind == "outcome"}
    assert statuses["@right"] == "failed" and statuses["@left"] in ("cancelled", "succeeded")
    assert not coordinator.outcomes and not coordinator.protections.leases


def test_cancellation_closes_admission_and_the_run_settles() -> None:
    holder: list[Coordinator] = []

    def first():
        holder[0].cancel()
        return 1

    installed = {
        "a": implementation(version=1, outputs=("int",), name="a")(first),
        "b": implementation(version=1, inputs=("int",), outputs=("int",), name="b")(lambda x: x),
        "c": implementation(version=1, inputs=("int",), outputs=("int",), name="c")(lambda x: x),
    }
    outcome, facts, coordinator = execute(CHAIN, installed, target=Target(fuse=False), before=holder.append)
    assert outcome.status == "cancelled"
    assert [f.owner.rsplit("/", 1)[-1] for f in facts if f.kind == "call_completed"] == ["0"], "only the first call ran"
    assert not coordinator.outcomes


TURNS = """iro.strategy @s uses(@work v1, @keep v1, @look v1, @solo v1) {
  %m = iro.resource @m : participant
  %n = iro.resource @n : participant
  iro.parallel @both capture(%m, %n) {
    iro.repeat @left 3 times capture(%m) {
    ^(%i: int):
      %r = iro.access %m protected_current {
      ^(%lease):
        iro.call @keep(%lease) -> retained int
      }
      %u = iro.call @work() -> int
      %v = iro.call @look(%r) -> int
    }
    iro.repeat @right 3 times capture(%n) {
    ^(%j: int):
      iro.access %n protected_current {
      ^(%lease):
        iro.call @solo() -> int
      }
    }
  }
}
"""


def test_device_work_of_different_activities_takes_turns_and_a_turn_lasts_until_the_release() -> None:
    device = FakeDevice()
    order: list[str] = []

    def keep(lease):
        order.append("keep")
        time.sleep(0.01)
        return 1

    def work():
        order.append("work")
        time.sleep(0.01)
        return 2

    installed = {
        "keep": implementation(version=1, inputs=("protected_current lease",), outputs=("retained int",), effects=["device"], name="keep")(
            keep
        ),
        "work": implementation(version=1, outputs=("int",), effects=["device"], name="work")(work),
        "look": implementation(version=1, inputs=("retained int",), outputs=("int",), effects=["device"], name="look")(
            lambda r: (order.append("look"), r)[1]
        ),
        "solo": implementation(version=1, outputs=("int",), effects=["device"], name="solo")(lambda: (order.append("solo"), 3)[1]),
    }
    outcome, facts, coordinator = execute(TURNS, installed, resources={"m": 1, "n": 2}, target=CUDA, devices={"cuda:0": device})
    assert outcome.status == "succeeded", outcome.error
    assert device.most == 1, "one admission's work at a time on a device"
    held = False
    for name in order:
        held = (held or name == "keep") and name != "look"
        assert not (held and name == "solo"), f"the other activity ran while a turn was held until its release: {order}"
    assert order.count("solo") == 3 and order.count("look") == 3
    released = [i for i, f in enumerate(facts) if f.kind == "product_released"]
    assert len(released) == 3
    assert all(t.holder is None and not t.waiting for t in coordinator.turns.by_device.values()) and not coordinator.turns.admissions


def test_a_fused_group_that_fails_midway_reports_only_the_calls_that_ran() -> None:
    def boom(x):
        raise ValueError("b failed")

    installed = {
        "a": implementation(version=1, outputs=("int",), name="a")(lambda: 1),
        "b": implementation(version=1, inputs=("int",), outputs=("int",), name="b")(boom),
        "c": implementation(version=1, inputs=("int",), outputs=("int",), name="c")(lambda y: y),
    }
    outcome, facts, coordinator = execute(CHAIN, installed)
    calls = [(f.owner.rsplit("/", 1)[-1], f.kind) for f in facts if f.kind.startswith("call_")]
    assert calls == [("0", "call_entered"), ("0", "call_completed"), ("1", "call_entered"), ("1", "call_failed")]
    assert outcome.status == "failed" and not coordinator.outcomes


TWO_UPDATES = """iro.strategy @s uses(@bump v1) {
  %m = iro.resource @m : participant
  iro.sequence @once capture(%m) {
    iro.access %m owner_update {
    ^(%lease):
      %x = iro.call @bump(%lease) -> int
      %y = iro.call @bump(%lease) -> int
    }
  }
}
"""


def test_each_owner_update_is_observed_after_the_call_that_made_it() -> None:
    def bump(cell):
        cell.update(cell.value + 1)
        return cell.value

    installed = {"bump": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), name="bump")(bump)}
    outcome, facts, _ = execute(TWO_UPDATES, installed, resources={"m": 0})
    assert outcome.status == "succeeded"
    assert [f.values[0].ordinal for f in facts if f.kind == "updated"] == [1, 2]


LEASE_AND_TURN = """iro.strategy @s uses(@keep v1, @write v1, @read v1, @look v1) {
  %m = iro.resource @m : participant
  %n = iro.resource @n : participant
  iro.parallel @both capture(%m, %n) {
    iro.sequence @holder capture(%m, %n) {
      %r = iro.access %n protected_current {
      ^(%lease):
        iro.call @keep(%lease) -> retained int
      }
      iro.access %m owner_update {
      ^(%lease):
        %w = iro.call @write(%lease) -> int
      }
      %v = iro.call @look(%r) -> int
    }
    iro.sequence @reader capture(%m) {
      iro.access %m protected_current {
      ^(%lease):
        %x = iro.call @read(%lease) -> int
      }
    }
  }
}
"""


def test_an_access_never_waits_for_the_device_while_holding_its_leases() -> None:
    def keep(lease):
        time.sleep(0.02)
        return 1

    def write(cell):
        cell.update(cell.value + 1)
        return 0

    device = ["device"]
    installed = {
        "keep": implementation(version=1, inputs=("protected_current lease",), outputs=("retained int",), effects=device, name="keep")(
            keep
        ),
        "write": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), effects=device, name="write")(write),
        "read": implementation(version=1, inputs=("protected_current lease",), outputs=("int",), effects=device, name="read")(lambda c: 2),
        "look": implementation(version=1, inputs=("retained int",), outputs=("int",), effects=device, name="look")(lambda r: r),
    }
    for _ in range(5):
        outcome, _, coordinator = execute(
            LEASE_AND_TURN, installed, resources={"m": 0, "n": 0}, target=CUDA, devices={"cuda:0": FakeDevice()}
        )
        assert outcome.status == "succeeded", outcome.error
        assert not coordinator.turns.admissions and all(t.holder is None for t in coordinator.turns.by_device.values())


KEPT_PAST_ITS_ACCESS = """iro.strategy @s uses(@predict v1, @derive v1, @report v1, @hold v1, @bump v1) {
  %m = iro.resource @m : participant
  iro.parallel @both capture(%m) {
    iro.sequence @trainer capture(%m) {
      %loss = iro.access %m protected_current {
      ^(%lease):
        %pred = iro.call @predict(%lease) -> retained int
        iro.call @derive(%pred) -> retained int
      }
      %r = iro.call @report(%loss) -> int
    }
    iro.sequence @writer capture(%m) {
      %h = iro.call @hold() -> int
      iro.access %m owner_update {
      ^(%lease):
        %b = iro.call @bump(%lease) -> int
      }
    }
  }
}
"""


def test_a_result_that_leaves_its_access_keeps_the_state_protected_after_what_it_keeps_is_released() -> None:
    derived = threading.Event()

    holder: list[Coordinator] = []

    def derive(pred):
        derived.set()
        return pred + 1

    def report(loss):
        # The loss is still alive here: hold it until the writer's access waits for @m.
        deadline = time.monotonic() + 5
        while not holder[0].kernel._waiters.get(("resource", "m")):
            assert time.monotonic() < deadline, "the writer's update waits for @m while the loss is alive"
            time.sleep(0.001)
        return loss

    def hold():
        assert derived.wait(5), "the trainer derives its loss before the writer asks for its update"
        return 0

    def bump(cell):
        cell.update(cell.value + 1)
        return cell.value

    installed = {
        "predict": implementation(version=1, inputs=("protected_current lease",), outputs=("retained int",), name="predict")(
            lambda lease: lease.value
        ),
        "derive": implementation(version=1, inputs=("retained int",), outputs=("retained int",), name="derive")(derive),
        "report": implementation(version=1, inputs=("retained int",), outputs=("int",), name="report")(report),
        "hold": implementation(version=1, outputs=("int",), name="hold")(hold),
        "bump": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), name="bump")(bump),
    }
    outcome, facts, coordinator = execute(
        KEPT_PAST_ITS_ACCESS, installed, resources={"m": 0}, target=Target(workers=4), before=holder.append
    )
    assert outcome.status == "succeeded", outcome.error
    made = [f.owner for f in facts if f.kind == "product_retained"]
    released = [i for i, f in enumerate(facts) if f.kind == "product_released"]
    assert [facts[i].owner for i in released] == made, "the prediction is released inside the access, before the loss"
    protected = next(i for i, f in enumerate(facts) if f.kind == "lease_released")
    updated = next(i for i, f in enumerate(facts) if f.kind == "updated")
    assert released[1] < protected < updated, "the loss keeps the prediction's lease, and the update waits for its release"
    assert not coordinator.protections.leases and not coordinator.protections.products


def test_cancellation_while_queued_for_a_turn_leaves_no_turn_behind() -> None:
    holder: list[Coordinator] = []

    def keep(lease):
        holder[0].cancel()
        return 1

    installed = {
        "keep": implementation(version=1, inputs=("protected_current lease",), outputs=("retained int",), effects=["device"], name="keep")(
            keep
        ),
        "work": implementation(version=1, outputs=("int",), effects=["device"], name="work")(lambda: 2),
        "look": implementation(version=1, inputs=("retained int",), outputs=("int",), effects=["device"], name="look")(lambda r: r),
        "solo": implementation(version=1, outputs=("int",), effects=["device"], name="solo")(lambda: 3),
    }
    outcome, _, coordinator = execute(
        TURNS, installed, resources={"m": 1, "n": 2}, target=CUDA, devices={"cuda:0": FakeDevice()}, before=holder.append
    )
    assert outcome.status == "cancelled"
    assert not coordinator.turns.admissions and all(t.holder is None and not t.waiting for t in coordinator.turns.by_device.values())
    assert not coordinator.protections.leases and not coordinator.protections.products and not coordinator.outcomes


def test_a_run_where_every_activity_waits_reports_a_stall_instead_of_hanging() -> None:
    installed = {"a": implementation(version=1, outputs=("int",), name="a")(lambda: 1)}
    text = "iro.strategy @s uses(@a v1) {\n  iro.sequence @once {\n    %x = iro.call @a() -> int\n  }\n}\n"
    run = accepted_run(text, installed=installed)
    coordinator = Coordinator(prepare(cold_load(run, installed=installed), installed, Target()), {}, installed, {"host": Host()})

    async def wait_forever():
        await coordinator.kernel.until(lambda: False)

    import pytest

    from library.iro.runtime import RunFailure

    with pytest.raises(RunFailure, match="no progress"):
        asyncio.run(wait_forever())


SHARED = """iro.strategy @s uses(@slow v1, @fast v1, @late v1, @pause v1) {
  %m = iro.resource @m : participant
  iro.parallel @all capture(%m) {
    iro.sequence @one capture(%m) {
      iro.access %m protected_current {
      ^(%lease):
        %a = iro.call @slow(%lease) -> int
      }
    }
    iro.sequence @two capture(%m) {
      iro.access %m protected_current {
      ^(%lease):
        %b = iro.call @fast(%lease) -> int
      }
    }
    iro.sequence @three capture(%m) {
      %p = iro.call @pause() -> int
      iro.access %m protected_current {
      ^(%lease):
        %c = iro.call @late(%lease) -> int
      }
    }
  }
}
"""


def test_calls_not_safe_for_concurrent_use_run_one_at_a_time_on_a_participant_even_beside_safe_ones() -> None:
    lock, unsafe = threading.Lock(), {"now": 0, "most": 0}

    def tracked(seconds):
        def run(lease):
            with lock:
                unsafe["now"] += 1
                unsafe["most"] = max(unsafe["most"], unsafe["now"])
            time.sleep(seconds)
            with lock:
                unsafe["now"] -= 1
            return 0

        return run

    lease = ("protected_current lease",)
    installed = {
        "slow": implementation(version=1, inputs=lease, outputs=("int",), name="slow")(tracked(0.3)),
        # A call declared safe for concurrent use finishing must not let another unsafe call in.
        "fast": implementation(version=1, inputs=lease, outputs=("int",), concurrency_safe=True, name="fast")(lambda lease: 0),
        "late": implementation(version=1, inputs=lease, outputs=("int",), name="late")(tracked(0.05)),
        "pause": implementation(version=1, outputs=("int",), name="pause")(lambda: time.sleep(0.1) or 0),
    }
    outcome, _, coordinator = execute(SHARED, installed, resources={"m": 0}, target=Target(workers=4))
    assert outcome.status == "succeeded", outcome.error
    assert unsafe["most"] == 1, "two calls not safe for concurrent use ran at once on @m"
    assert not coordinator.calls.busy


TWO_WRITERS = """iro.strategy @s uses(@write v1) {
  %m = iro.resource @m : participant
  iro.parallel @both capture(%m) {
    iro.sequence @left capture(%m) {
      iro.access %m owner_update {
      ^(%lease):
        %a = iro.call @write(%lease) -> int
      }
    }
    iro.sequence @right capture(%m) {
      iro.access %m owner_update {
      ^(%lease):
        %b = iro.call @write(%lease) -> int
      }
    }
  }
}
"""


def test_checked_mode_fails_a_run_that_loses_a_wake() -> None:
    def write(cell):
        time.sleep(0.02)
        cell.update(cell.value + 1)
        return 0

    installed = {"write": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), name="write")(write)}
    from library.iro.runtime.kernel import Kernel

    original = Kernel.changed

    def losing(self, *keys):
        return original(self, *(key for key in keys if key != ("resource", "m")))

    Kernel.changed = losing
    try:
        outcome, _, coordinator = execute(TWO_WRITERS, installed, resources={"m": 0})
    finally:
        Kernel.changed = original
    assert outcome.status == "failed"
    assert any("without a change on the keys it waits on" in error for error in coordinator.errors), coordinator.errors


def test_a_host_call_that_loads_torch_itself_fails_instead_of_running_under_the_threads_defaults(monkeypatch) -> None:
    import sys

    import pytest

    from library.iro.seams import CallContext

    torch = pytest.importorskip("torch")
    host = Host()
    monkeypatch.delitem(sys.modules, "torch")
    with pytest.raises(RuntimeError, match="loaded Torch itself"), host.context(CallContext(), ()):
        sys.modules["torch"] = torch  # what importing it inside the call does
    with torch.enable_grad(), host.context(CallContext(), ()):
        assert not torch.is_grad_enabled(), "once Torch is loaded, the host establishes the call's baseline"


def test_the_coordinator_refuses_adapters_that_differ_from_what_the_run_was_prepared_for() -> None:
    import pytest

    installed = {"a": implementation(version=1, outputs=("int",), name="a")(lambda: 1)}
    text = "iro.strategy @s uses(@a v1) {\n  iro.sequence @once {\n    %x = iro.call @a() -> int\n  }\n}\n"
    run = accepted_run(text, installed=installed)
    target = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"ordering"}))))
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    with pytest.raises(ValueError, match="no device adapter realizes it"):
        Coordinator(prepared, {}, installed, {"host": Host()})
    with pytest.raises(ValueError, match="prepared for a cuda device offering"):
        Coordinator(prepared, {}, installed, {"host": Host(), "cuda:0": FakeDevice()})
    with pytest.raises(ValueError, match="the adapter given for cuda:0 is cuda:1"):
        Coordinator(prepared, {}, installed, {"host": Host(), "cuda:0": FakeDevice(name="cuda:1", offers=frozenset({"ordering"}))})
    with pytest.raises(ValueError, match="not prepared for: cuda:1"):
        Coordinator(prepared, {}, installed, {"host": Host(), "cuda:0": FakeDevice(offers=frozenset({"ordering"})), "cuda:1": FakeDevice()})
    offering = FakeDevice(offers=frozenset({"ordering"}))
    assert Coordinator(prepared, {}, installed, {"host": Host(), "cuda:0": offering}).devices["cuda:0"] is offering


LEAVING = """iro.strategy @s uses(@encode v1) {
  %vae = iro.resource @vae : participant may_leave_device
  iro.sequence @once capture(%vae) {
    iro.access %vae protected_current {
    ^(%lease):
      %x = iro.call @encode(%lease) -> int
    }
  }
}
"""


def test_a_participant_that_may_leave_needs_an_owner_that_names_its_storage_and_an_adapter_that_copies_it() -> None:
    import pytest

    installed = {
        "encode": implementation(version=1, inputs=("protected_current lease",), outputs=("int",), effects=["device"], name="encode")(
            lambda lease: 1
        )
    }
    run = accepted_run(LEAVING, installed=installed)
    target = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"copies"}))))
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    assert prepared.leaving == (("vae", "cuda:0"),)
    device = FakeDevice(offers=frozenset({"copies"}))
    with pytest.raises(ValueError, match="its owner does not say what its device storage is"):
        Coordinator(prepared, {"vae": Memory({"vae": 0})}, installed, {"host": Host(), "cuda:0": device})

    class Named(Memory):
        def storage(self, resource):
            return ()

    with pytest.raises(ValueError, match="its adapter does not copy storage"):
        Coordinator(prepared, {"vae": Named({"vae": 0})}, installed, {"host": Host(), "cuda:0": device})


SNAPSHOT_BETWEEN_UPDATES = """iro.strategy @s uses(@first v1, @look v1, @second v1) {
  %m = iro.resource @m : participant
  iro.sequence @once capture(%m) {
    iro.access %m owner_update {
    ^(%lease):
      %u = iro.call @first(%lease) -> int
    }
    iro.access %m SELECTION {
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


@dataclass
class Copier(FakeDevice):
    """A device that copies storage for snapshots, logging when, as a list's items stand in for device storage."""

    offers: frozenset[str] = frozenset({"copies"})
    log: list = field(default_factory=list)

    def capture(self, storage, after):
        self.log.append("capture")
        return [list(box) for box in storage], Returned()

    def empty(self, storage):
        raise AssertionError("nothing leaves here")

    def fill(self, storage):
        raise AssertionError("nothing leaves here")


class OnDevice:
    """An owner of one participant whose state is device storage: a snapshot lease asks the runtime to copy it."""

    def __init__(self, log: list, name: str = "m"):
        self.box: list[object] = [0]  # the storage, which a call may move within its use
        self.ordinal, self.log, self.name = 0, log, name

    def resources(self):
        return (self.name,)

    def current(self, resource):
        from tests.unit.iro.runtime.harness import Seen

        return Seen(self.name, 1, "initial", self.ordinal)

    def lease(self, uses, reserved=None):
        from library.iro.operation_contracts import AccessMode, VersionSelection
        from library.iro.seams import Capture
        from tests.unit.iro.runtime.harness import Granted, Seen

        granted = []
        for use in uses:
            if use.mode == AccessMode.SNAPSHOT:
                initial = use.selection == VersionSelection.PUBLISHED_INITIAL
                version = Seen(self.name, 1, "initial", 0) if initial else self.current(self.name)
                granted.append(Granted(use, version, Capture((self.box,), lambda copies: ("snapshot", copies[0]))))
            else:
                granted.append(Granted(use, self.current(self.name), self, self.advanced))
        return granted

    def advanced(self):
        self.ordinal += 1
        return self.current(self.name)

    def update(self, value):
        self.log.append("update")
        self.box[0] = value


def _between_updates(selection: str = "snapshot"):
    seen = []

    def look(handle):
        seen.append(handle)
        return 0

    installed = {
        "first": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), effects=["device"], name="first")(
            lambda owner: owner.update(1) or 0
        ),
        "look": implementation(version=1, inputs=("snapshot lease",), outputs=("int",), effects=["device"], name="look")(look),
        "second": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), effects=["device"], name="second")(
            lambda owner: owner.update(2) or 0
        ),
    }
    log: list = []
    owner = OnDevice(log)
    device = Copier(log=log)
    run = accepted_run(SNAPSHOT_BETWEEN_UPDATES.replace("SELECTION", selection), installed=installed)
    target = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"copies"}))), workers=4)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    coordinator = Coordinator(prepared, {"m": owner}, installed, {"host": Host(), "cuda:0": device}, checked=True)
    outcome = asyncio.run(coordinator.run())
    return outcome, prepared, log, seen, owner


def test_a_snapshot_of_device_storage_is_copied_by_the_runtime_between_the_updates_around_it() -> None:
    outcome, prepared, log, seen, owner = _between_updates()
    assert outcome.status == "succeeded", outcome.error
    assert prepared.placed == (("m", "cuda:0"),)
    assert log == ["update", "capture", "update"], "the copy follows the update before it and precedes the one after"
    assert seen == [("snapshot", [1])], "the call received the owner's handle over the copy, at the version it leased"
    assert owner.box == [2]


def test_a_capture_copies_live_storage_so_an_older_version_of_device_storage_fails_with_the_reason() -> None:
    outcome, _, log, seen, _ = _between_updates("snapshot published initial")
    assert outcome.status == "failed" and "the snapshot's version is not the current one" in (outcome.error or "")
    assert "capture" not in log and not seen


def _host_only(text: str, owner) -> tuple:
    installed = {
        "look": implementation(version=1, inputs=("snapshot lease",), outputs=("int",), name="look")(lambda handle: handle),
        "bump": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), name="bump")(lambda o: o.update(5) or 0),
    }
    installed = {k: v for k, v in installed.items() if f"@{k} v1" in text}
    run = accepted_run(text, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, Target())
    facts: list[Fact] = []
    coordinator = Coordinator(prepared, {"m": owner}, installed, {"host": Host()}, observe=facts.append, checked=True)
    return asyncio.run(coordinator.run()), coordinator, prepared, facts


ONLY_A_SNAPSHOT = """iro.strategy @s uses(@look v1) {
  %m = iro.resource @m : participant
  iro.sequence @once capture(%m) {
    iro.access %m snapshot {
    ^(%lease):
      %v = iro.call @look(%lease) -> int
    }
  }
}
"""

UPDATED_THEN_SNAPSHOT = ONLY_A_SNAPSHOT.replace("uses(@look v1)", "uses(@bump v1, @look v1)").replace(
    "    iro.access %m snapshot {",
    "    iro.access %m owner_update {\n    ^(%u):\n      %b = iro.call @bump(%u) -> int\n    }\n    iro.access %m snapshot {",
)


def test_a_snapshot_of_updated_device_storage_needs_the_one_device_whose_work_uses_it() -> None:
    outcome, coordinator, _, _ = _host_only(UPDATED_THEN_SNAPSHOT, OnDevice([]))
    assert outcome.status == "failed" and "on no device or on several" in (outcome.error or "")
    assert not coordinator.protections.leases


def test_a_snapshot_of_a_participant_the_run_never_updates_is_its_live_storage_without_a_copy() -> None:
    log: list = []
    owner = OnDevice(log)
    outcome, _, prepared, facts = _host_only(ONLY_A_SNAPSHOT, owner)
    assert outcome.status == "succeeded", outcome.error
    assert prepared.updated == () and log == [], "nothing updates @m, so no copy is made"
    (handle,) = next(f.values[1] for f in facts if f.kind == "call_completed")
    assert handle[0] == "snapshot" and handle[1] is owner.box, "the owner's handle over the live storage itself"


def test_the_prepared_form_marks_participants_the_run_never_updates() -> None:
    installed = {
        "first": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), effects=["device"], name="first")(
            lambda owner: 0
        ),
        "look": implementation(version=1, inputs=("snapshot lease",), outputs=("int",), effects=["device"], name="look")(lambda handle: 0),
        "second": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), effects=["device"], name="second")(
            lambda owner: 0
        ),
    }
    target = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"copies"}))))
    updated = prepare(
        cold_load(accepted_run(SNAPSHOT_BETWEEN_UPDATES.replace("SELECTION", "snapshot"), installed=installed), installed=installed),
        installed,
        target,
    )
    assert updated.updated == ("m",) and "  @m on cuda:0\n" in updated.text()
    frozen = {"look": installed["look"]}
    prepared = prepare(cold_load(accepted_run(ONLY_A_SNAPSHOT, installed=frozen), installed=frozen), frozen, target)
    assert prepared.updated == () and "  @m on cuda:0, frozen\n" in prepared.text()


REQUESTED_SNAPSHOTS = """iro.strategy @s uses(@step v1, @look v1) {
  %m = iro.resource @m : participant
  iro.parallel @all capture(%m) {
    iro.repeat @train 3 times capture(%m) {
    ^(%k: int):
      iro.access %m owner_update {
      ^(%lease):
        %w = iro.call @step(%lease, %k) -> int
      }
      iro.request @sample(%k) from @train at settled_success
    }
    iro.handler @sample sources [@train] max 1 capture(%m) {
    ^(%x: int):
      iro.access %m snapshot at_request {
      ^(%lease):
        %v = iro.call @look(%lease, %x) -> int
      }
    }
  }
}
"""


TWO_LEASES = (
    REQUESTED_SNAPSHOTS.replace("@step v1", "@step2 v1")
    .replace("%m = iro.resource @m : participant", "%m = iro.resource @m : participant\n  %n = iro.resource @n : participant")
    .replace("iro.repeat @train 3 times capture(%m)", "iro.repeat @train 3 times capture(%m, %n)")
    .replace("iro.parallel @all capture(%m)", "iro.parallel @all capture(%m, %n)")
    .replace(
        "iro.access %m owner_update {\n      ^(%lease):\n        %w = iro.call @step(%lease, %k) -> int",
        "iro.access %n protected_current, %m owner_update {\n      ^(%read, %lease):\n        %w = iro.call @step2(%read, %lease, %k) -> int",
    )
)


@pytest.mark.parametrize("shape", ["unsafe update", "safe update", "update beside another lease"])
def test_an_update_after_a_request_waits_until_its_snapshot_of_device_storage_is_copied(shape: str) -> None:
    seen = []

    def look(handle, x):
        seen.append((x, handle))
        return 0

    update = ("owner_update lease", "int")
    installed = {
        "step": implementation(
            version=1, inputs=update, outputs=("int",), effects=["device"], concurrency_safe=shape == "safe update", name="step"
        )(lambda owner, k: owner.update(k + 1) or 0),
        "step2": implementation(version=1, inputs=("protected_current lease", *update), outputs=("int",), effects=["device"], name="step2")(
            lambda other, owner, k: owner.update(k + 1) or 0
        ),
        # On the host, so the handler holds no device turn the loop's update would wait behind instead.
        "look": implementation(version=1, inputs=("snapshot lease", "int"), outputs=("int",), name="look")(look),
    }
    text = TWO_LEASES if shape == "update beside another lease" else REQUESTED_SNAPSHOTS
    installed = {k: v for k, v in installed.items() if f"@{k} v1" in text}
    log: list = []
    owner = OnDevice(log)
    run = accepted_run(text, installed=installed)
    target = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"copies"}))), workers=4)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    owners = {"m": owner, "n": OnDevice([], "n")} if "%n" in text else {"m": owner}
    coordinator = Coordinator(prepared, owners, installed, {"host": Host(), "cuda:0": Copier(log=log)}, checked=True)
    copy = coordinator.storage.capture

    async def late(lease, capture, adapter):
        # Queued late, as behind its storage coming back: the loop's next update is ready long before.
        await asyncio.sleep(0.05)
        await copy(lease, capture, adapter)

    coordinator.storage.capture = late  # ty: ignore[invalid-assignment]
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    assert sorted(seen) == [(0, ("snapshot", [1])), (1, ("snapshot", [2])), (2, ("snapshot", [3]))], "each request saw its own version"
    assert log == ["update", "capture"] * 3, "each update after a request waited for that request's copy"


def _requested(text: str, owners: dict, device) -> Coordinator:
    installed = {
        "step": implementation(version=1, inputs=("owner_update lease", "int"), outputs=("int",), effects=["device"], name="step")(
            lambda owner, k: owner.update(k + 1) or 0
        ),
        "look": implementation(version=1, inputs=("snapshot lease", "int"), outputs=("int",), name="look")(lambda handle, x: 0),
        "both": implementation(version=1, inputs=("snapshot lease", "snapshot lease", "int"), outputs=("int",), name="both")(
            lambda first, second, x: 0
        ),
        "bump": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), name="bump")(
            lambda owner: owner.update(0) or 0
        ),
    }
    installed = {k: v for k, v in installed.items() if f"@{k} v1" in text}
    run = accepted_run(text, installed=installed)
    target = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"copies"}))), workers=4)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    return Coordinator(prepared, owners, installed, {"host": Host(), "cuda:0": device}, checked=True)


def _clean(coordinator: Coordinator) -> None:
    assert not coordinator.protections.leases, "no lease is left behind"
    assert (
        not coordinator.storage.capturing and not any(coordinator.storage.using.values()) and not any(coordinator.storage.holding.values())
    )


def test_a_request_whose_second_snapshot_cannot_be_captured_releases_the_first() -> None:
    # @n is updated, on the host only, so its snapshot needs a copy and no device holds its storage.
    text = (
        REQUESTED_SNAPSHOTS.replace("@look v1", "@look v1, @both v1, @bump v1")
        .replace(
            "      iro.request @sample(%k) from @train at settled_success",
            "      iro.access %n owner_update {\n      ^(%u):\n        %c = iro.call @bump(%u) -> int\n      }\n"
            "      iro.request @sample(%k) from @train at settled_success",
        )
        .replace("%m = iro.resource @m : participant", "%m = iro.resource @m : participant\n  %n = iro.resource @n : participant")
        .replace("capture(%m)", "capture(%m, %n)")
        .replace(
            "iro.access %m snapshot at_request {\n      ^(%lease):\n        %v = iro.call @look(%lease, %x) -> int",
            "iro.access %m snapshot at_request, %n snapshot at_request {\n      ^(%a, %b):\n        %v = iro.call @both(%a, %b, %x) -> int",
        )
    )
    assert "@both(%a, %b, %x)" in text
    log: list = []
    coordinator = _requested(text, {"m": OnDevice(log), "n": OnDevice(log, "n")}, Copier(log=log))
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "failed" and "@n: " in (outcome.error or "") and "on no device or on several" in (outcome.error or "")
    _clean(coordinator)


def test_a_capture_that_fails_on_the_device_fails_the_run_and_leaves_nothing_behind() -> None:
    class Failing(Copier):
        def capture(self, storage, after):
            raise RuntimeError("out of memory")

    log: list = []
    coordinator = _requested(REQUESTED_SNAPSHOTS, {"m": OnDevice(log)}, Failing(log=log))
    outcome = asyncio.run(coordinator.run())
    assert (
        outcome.status == "failed" and "capturing its snapshot failed" in (outcome.error or "") and "out of memory" in (outcome.error or "")
    )
    _clean(coordinator)


MOVED_WITHIN_A_USE = """iro.strategy @s uses(@swap v1, @pause v1, @look v1) {
  %m = iro.resource @m : participant
  iro.parallel @both capture(%m) {
    iro.sequence @reading capture(%m) {
      iro.access %m protected_current {
      ^(%lease):
        %r = iro.call @swap(%lease) -> int
      }
    }
    iro.sequence @snapshotting capture(%m) {
      %p = iro.call @pause() -> int
      iro.access %m snapshot {
      ^(%lease):
        %v = iro.call @look(%lease) -> int
      }
    }
  }
}
"""


def test_a_capture_copies_storage_where_a_use_that_moves_it_leaves_it() -> None:
    """A read whose realization moves the storage within its use, as block swapping does, is never copied mid-move."""
    seen = []

    def swap(owner):
        owner.box[0] = "elsewhere"  # moved within the use...
        time.sleep(0.1)
        owner.box[0] = 0  # ...and back where it was by the end of it
        return 0

    installed = {
        "swap": implementation(version=1, inputs=("protected_current lease",), outputs=("int",), effects=["device"], name="swap")(swap),
        "pause": implementation(version=1, outputs=("int",), name="pause")(lambda: time.sleep(0.03) or 0),
        # On the host, so the snapshot's access needs no device turn and is taken while the read runs.
        "look": implementation(version=1, inputs=("snapshot lease",), outputs=("int",), name="look")(
            lambda handle: seen.append(handle) or 0
        ),
    }
    log: list = []
    run = accepted_run(MOVED_WITHIN_A_USE, installed=installed)
    target = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"copies"}))), workers=4)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    coordinator = Coordinator(prepared, {"m": OnDevice(log)}, installed, {"host": Host(), "cuda:0": Copier(log=log)}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    assert seen == [("snapshot", [0])], "the copy waited for the read already being queued, so it holds what the read left"


def test_a_capture_is_refused_for_an_adapter_without_copies_and_for_a_lease_that_is_not_a_snapshot() -> None:
    from library.iro.operation_contracts import AccessMode
    from library.iro.seams import Capture
    from tests.unit.iro.runtime.harness import Granted

    installed = {
        "first": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), effects=["device"], name="first")(
            lambda owner: 0
        ),
        "look": implementation(version=1, inputs=("snapshot lease",), outputs=("int",), effects=["device"], name="look")(lambda handle: 0),
        "second": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), effects=["device"], name="second")(
            lambda owner: 0
        ),
    }
    run = accepted_run(SNAPSHOT_BETWEEN_UPDATES.replace("SELECTION", "snapshot"), installed=installed)
    bare = Target((HOST, DeviceFacts("cuda:0", "cuda")), workers=4)
    prepared = prepare(cold_load(run, installed=installed), installed, bare)
    coordinator = Coordinator(prepared, {"m": OnDevice([])}, installed, {"host": Host(), "cuda:0": FakeDevice()}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "failed" and "needs storage copies, and cuda:0's adapter does not offer them" in (outcome.error or "")
    _clean(coordinator)

    class CapturesEverything(OnDevice):
        def lease(self, uses, reserved=None):
            return [Granted(use, self.current("m"), Capture((self.box,), lambda copies: copies)) for use in uses]

    copying = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"copies"}))), workers=4)
    prepared = prepare(cold_load(run, installed=installed), installed, copying)
    coordinator = Coordinator(prepared, {"m": CapturesEverything([])}, installed, {"host": Host(), "cuda:0": Copier()}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert (
        outcome.status == "failed"
        and f"only a snapshot lease is captured, and the owner asked to capture a {AccessMode.OWNER_UPDATE} lease" in (outcome.error or "")
    )
    _clean(coordinator)


SNAPSHOTS_OF_A_LEAVING_PARTICIPANT = """iro.strategy @s uses(@look v1, @pause v1) {
  %m = iro.resource @m : participant may_leave_device
  iro.sequence @once capture(%m) {
    iro.access %m snapshot {
    ^(%lease):
      %a = iro.call @look(%lease) -> int
    }
    %p = iro.call @pause() -> int
    iro.access %m snapshot {
    ^(%lease):
      %b = iro.call @look(%lease) -> int
    }
  }
}
"""


@dataclass
class Moving(Copier):
    """A copying device whose storage also leaves and comes back, logging each move."""

    def empty(self, storage):
        self.log.append("leave")

    def fill(self, storage):
        self.log.append("return")
        return Returned()


class LeavingOnDevice(OnDevice):
    def storage(self, resource):
        return (self.box,)


def _leaving_snapshots() -> tuple[Coordinator, list]:
    installed = {
        "look": implementation(version=1, inputs=("snapshot lease",), outputs=("int",), effects=["device"], name="look")(lambda handle: 0),
        "pause": implementation(version=1, outputs=("int",), name="pause")(lambda: time.sleep(0.05) or 0),
    }
    log: list = []
    run = accepted_run(SNAPSHOTS_OF_A_LEAVING_PARTICIPANT, installed=installed)
    target = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"copies"}))), workers=4)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    assert prepared.leaving == (("m", "cuda:0"),)
    owner = LeavingOnDevice(log)
    return Coordinator(prepared, {"m": owner}, installed, {"host": Host(), "cuda:0": Moving(log=log)}, checked=True), log


def test_a_snapshot_of_a_participant_that_may_leave_reads_its_storage_like_a_live_use() -> None:
    """Only frozen storage may leave, so its snapshot needs no copy: the storage comes back for the use and leaves after."""
    coordinator, log = _leaving_snapshots()
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    assert log == ["leave", "return", "leave"], "no copy; it left after each use and came back for the second"
    _clean(coordinator)


def test_a_capture_cancelled_before_it_begins_leaves_nothing_behind() -> None:
    from library.iro.operation_contracts import AccessMode, VersionSelection
    from library.iro.seams import Use

    log: list = []
    coordinator = _requested(REQUESTED_SNAPSHOTS, {"m": OnDevice(log)}, Copier(log=log))  # @m is updated: its snapshot copies

    async def cancel_at_once() -> None:
        (granted,) = coordinator.owners["m"].lease([Use("m", AccessMode.SNAPSHOT, VersionSelection.CURRENT_AT_START)])
        lease = coordinator.protections.hold("m", AccessMode.SNAPSHOT, granted)
        assert lease.capture is not None and coordinator.storage.capturing
        lease.capture.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert lease.capture.cancelled() and log == [], "it never began"
        assert not coordinator.storage.capturing, "its entry went with it, so later updates of @m are not held back"

    asyncio.run(cancel_at_once())


MOVED_THROUGH_A_RETAINED_RESULT = """iro.strategy @s uses(@hold v1, @move v1, @pause v1, @look v1) {
  %m = iro.resource @m : participant
  iro.parallel @both capture(%m) {
    iro.sequence @reading capture(%m) {
      %r = iro.access %m protected_current {
      ^(%lease):
        iro.call @hold(%lease) -> retained int
      }
      %x = iro.call @move(%r) -> int
    }
    iro.sequence @snapshotting capture(%m) {
      %p = iro.call @pause() -> int
      iro.access %m snapshot {
      ^(%lease):
        %v = iro.call @look(%lease) -> int
      }
    }
  }
}
"""


def test_a_capture_waits_for_a_call_reaching_the_storage_through_a_retained_result() -> None:
    """As a backward pass reaches the weights through the prediction it is given, and may move them as it goes."""
    seen = []
    owner = OnDevice([])

    def move(retained):
        owner.box[0] = "elsewhere"
        time.sleep(0.1)
        owner.box[0] = 0
        return 0

    installed = {
        "hold": implementation(version=1, inputs=("protected_current lease",), outputs=("retained int",), effects=["device"], name="hold")(
            lambda lease: 0
        ),
        "move": implementation(version=1, inputs=("retained int",), outputs=("int",), effects=["device"], name="move")(move),
        "pause": implementation(version=1, outputs=("int",), name="pause")(lambda: time.sleep(0.03) or 0),
        "look": implementation(version=1, inputs=("snapshot lease",), outputs=("int",), name="look")(
            lambda handle: seen.append(handle) or 0
        ),
    }
    run = accepted_run(MOVED_THROUGH_A_RETAINED_RESULT, installed=installed)
    target = Target((HOST, DeviceFacts("cuda:0", "cuda", frozenset({"copies"}))), workers=4)
    prepared = prepare(cold_load(run, installed=installed), installed, target)
    coordinator = Coordinator(prepared, {"m": owner}, installed, {"host": Host(), "cuda:0": Copier()}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    assert seen == [("snapshot", [0])], "the copy waited for the call the retained result's lease reaches the storage through"


# Setups and annotations (D16).

DIFFERENTIATING = """iro.strategy @s uses(@predict v1, @learn v1, @step v1) {
  %net = iro.resource @net : participant
  %w = iro.resource @w
  iro.sequence @once capture(%net, %w) {
    %loss = iro.access %net protected_current {
    ^(%n):
      iro.call @predict(%n) -> retained loss
    }
    iro.access %w owner_update {
    ^(%u):
      iro.call @learn(%u, %loss)
    }
    iro.access %w owner_update, %net owner_update {
    ^(%u, %m):
      iro.call @step(%u, %m)
    }
  }
}
"""


@dataclass(frozen=True)
class Recording:
    """A call setup that records the annotations and handles it is entered with, and on which thread."""

    seen: list
    name: str = "test.recording"
    version: str = "1"

    def entered(self, annotations, handles):
        self.seen.append(("entered", annotations, tuple(handles), threading.current_thread().name))
        return contextlib.nullcontext()


@dataclass(frozen=True)
class Annotating:
    """A deriver that gives the calls of the named implementations the annotations and setups listed for them."""

    given: dict
    name: str = "test.annotating"
    version: str = "1"

    def derive(self, program):
        return {call: self.given[program.implementation(call)] for call in program.calls() if program.implementation(call) in self.given}


class HeldRecording:
    """A held setup that records what it is applied with."""

    name = "test.held"
    version = "1"

    def __init__(self, seen: list) -> None:
        self.seen = seen

    def key(self, annotations):
        return annotations

    def hold(self, annotations, handle) -> None:
        self.seen.append(("hold", annotations, handle, threading.current_thread().name))


@dataclass
class Holding(Memory):
    """A memory owner whose leases on ``holding`` carry one held setup, as a route's realization does."""

    holding: tuple[str, ...] = ()
    seen: list = field(default_factory=list)
    setup: HeldRecording | None = None

    def lease(self, uses, reserved=None):
        granted = super().lease(uses, reserved)
        if self.setup is None:
            self.setup = HeldRecording(self.seen)
        # As the run authority's routes do: an update runs nothing through a route, so no route setup covers it.
        return [
            HoldingGrant(lease, self.setup) if lease.use.resource in self.holding and lease.use.mode != "owner_update" else lease
            for lease in granted
        ]


class HoldingGrant:
    def __init__(self, lease, setup) -> None:
        self._lease, self.setup = lease, setup

    def __getattr__(self, name: str):
        return getattr(self._lease, name)


def differentiating_installed(order: list[str] | None = None, *, safe: bool = False) -> dict:
    def predict(n):
        if order is not None:
            order.append("predict")
        return 1.0

    return {
        "predict": implementation(
            version=1, inputs=("protected_current lease",), outputs=("retained loss",), name="predict", concurrency_safe=safe
        )(predict),
        "learn": implementation(version=1, inputs=("owner_update lease", "retained loss"), name="learn")(lambda u, loss: None),
        "step": implementation(version=1, inputs=("owner_update lease", "owner_update lease"), name="step")(lambda u, m: None),
    }


def test_setups_are_entered_on_the_worker_with_the_call_s_annotations_and_handles() -> None:
    installed = differentiating_installed()
    run = accepted_run(DIFFERENTIATING, installed=installed)
    seen: list = []
    annotated = Derived(annotate({"units": ["w"]}), (Recording(seen),))
    prepared = prepare(cold_load(run, installed=installed), installed, Target(), [Annotating({"predict": annotated})])
    assert "@predict (setups @test.recording v1, units w)" in prepared.text()
    assert re.search(r"derived by\n  @test\.annotating v1 code \w+\n  @test\.recording v1 code \w+\n", prepared.text()), (
        "the deriver and the call setup are pinned by version and code identity"
    )
    owner = Holding({"net": 1, "w": 0}, holding=("net",), seen=seen)
    coordinator = Coordinator(prepared, {"net": owner, "w": owner}, installed, {"host": Host()}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    assert [(kind, annotations) for kind, annotations, *_ in seen] == [("entered", (("units", ("w",)),)), ("hold", (("units", ("w",)),))], (
        "the forward enters its call setup and applies its lease's held setup under its annotations; the update's lease carries none"
    )
    assert seen[0][2] == (seen[1][2],), "a setup receives the handles the calls receive"
    assert all(thread != threading.main_thread().name for *_, thread in seen), "it runs on the worker, as the calls do"


TWO_READERS = """iro.strategy @s uses(@read v1) {
  %net = iro.resource @net : participant
  iro.parallel @both capture(%net) {
    iro.sequence @left capture(%net) {
      iro.access %net protected_current {
      ^(%n):
        %a = iro.call @read(%n) -> int
      }
    }
    iro.sequence @right capture(%net) {
      iro.access %net protected_current {
      ^(%n):
        %b = iro.call @read(%n) -> int
      }
    }
  }
}
"""


@dataclass
class HoldingOnce(Holding):
    """A memory owner whose first lease carries a held setup and whose others do not."""

    hold: bool = True
    granted: int = 0

    def lease(self, uses, reserved=None):
        granted = Memory.lease(self, uses, reserved)
        self.granted += 1
        if self.setup is None:
            self.setup = HeldRecording(self.seen)
        return [HoldingGrant(granted[0], self.setup)] if self.hold and self.granted == 1 else granted


@pytest.mark.parametrize("hold", [True, False])
def test_a_group_that_applies_a_held_setup_runs_alone_even_beside_calls_safe_for_concurrent_use(hold: bool) -> None:
    spans: list[tuple[float, float]] = []
    both = threading.Barrier(2, timeout=0.5)

    def read(n):
        began = time.perf_counter()
        with contextlib.suppress(threading.BrokenBarrierError):
            both.wait()  # passes only when the two calls run at once
        spans.append((began, time.perf_counter()))
        return 1

    installed = {
        "read": implementation(version=1, inputs=("protected_current lease",), outputs=("int",), name="read", concurrency_safe=True)(read)
    }
    run = accepted_run(TWO_READERS, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, Target(workers=4))
    owner = HoldingOnce({"net": 1}, hold=hold)
    coordinator = Coordinator(prepared, {"net": owner}, installed, {"host": Host()}, checked=True)
    assert asyncio.run(coordinator.run()).status == "succeeded"
    (first, second) = sorted(spans)
    overlapped = second[0] < first[1]
    assert overlapped is not hold, "a group applying a held setup runs alone; two safe calls otherwise run together"
    assert len(owner.seen) == (1 if hold else 0)


HELD_PAST_ITS_ACCESS = """iro.strategy @s uses(@predict v1, @learn v1, @read v1) {
  %net = iro.resource @net : participant
  %w = iro.resource @w
  iro.parallel @both capture(%net, %w) {
    iro.sequence @training capture(%net, %w) {
      %loss = iro.access %net protected_current {
      ^(%n):
        iro.call @predict(%n) -> retained loss
      }
      iro.access %w owner_update {
      ^(%u):
        iro.call @learn(%u, %loss)
      }
    }
    iro.sequence @evaluation capture(%net) {
      iro.access %net protected_current {
      ^(%n):
        %b = iro.call @read(%n) -> int
      }
    }
  }
}
"""


def test_a_held_setup_holds_until_the_retained_result_releases_the_lease_and_another_waits() -> None:
    events: list = []

    def learn(u, loss):
        time.sleep(0.05)  # gives the other activity every chance to apply its setup in between
        events.append("learned")

    def read(n):
        events.append("read")
        return 1

    installed = {
        "predict": implementation(version=1, inputs=("protected_current lease",), outputs=("retained loss",), name="predict")(
            lambda n: events.append("predicted") or 1.0
        ),
        "learn": implementation(version=1, inputs=("owner_update lease", "retained loss"), name="learn")(learn),
        "read": implementation(version=1, inputs=("protected_current lease",), outputs=("int",), name="read", concurrency_safe=True)(read),
    }
    run = accepted_run(HELD_PAST_ITS_ACCESS, installed=installed)
    deriver = Annotating({"predict": Derived(annotate({"units": ["w"]}))})
    prepared = prepare(cold_load(run, installed=installed), installed, Target(workers=4), [deriver])
    owner = Holding({"net": 1, "w": 0}, holding=("net",), seen=events)
    coordinator = Coordinator(prepared, {"net": owner, "w": owner}, installed, {"host": Host()}, checked=True)
    assert asyncio.run(coordinator.run()).status == "succeeded"
    order = [event if isinstance(event, str) else f"hold {dict(event[1]).get('units', ())}" for event in events]
    assert order.count("read") == 1 and order.count("learned") == 1
    assert "hold ()" not in order[order.index("hold ('w',)") : order.index("learned")], (
        "the evaluation's setup waits until the loss, which keeps the forward's lease, is released"
    )


class SameState(HeldRecording):
    """A held setup object of its own that needs the same state as another."""

    def key(self, annotations):
        return "train"


def test_separate_setup_objects_that_need_the_same_state_share_a_hold() -> None:
    """Two routes' setups on one participant: a retained result of the first must not make the second wait on itself (D16)."""
    events: list = []
    installed = {
        "predict": implementation(version=1, inputs=("protected_current lease",), outputs=("retained loss",), name="predict")(
            lambda n: 1.0
        ),
        "learn": implementation(version=1, inputs=("owner_update lease", "retained loss"), name="learn")(lambda u, loss: None),
        "read": implementation(version=1, inputs=("protected_current lease",), outputs=("int",), name="read")(lambda n: 1),
    }
    text = """iro.strategy @s uses(@predict v1, @read v1, @learn v1) {
  %net = iro.resource @net : participant
  %w = iro.resource @w
  iro.sequence @once capture(%net, %w) {
    %loss = iro.access %net protected_current {
    ^(%n):
      iro.call @predict(%n) -> retained loss
    }
    %again = iro.access %net protected_current {
    ^(%n):
      iro.call @read(%n) -> int
    }
    iro.access %w owner_update {
    ^(%u):
      iro.call @learn(%u, %loss)
    }
  }
}
"""
    run = accepted_run(text, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, Target())

    @dataclass
    class TwoRoutes(Memory):
        granted: int = 0

        def lease(self, uses, reserved=None):
            granted = Memory.lease(self, uses, reserved)
            self.granted += 1
            return [HoldingGrant(lease, SameState(events)) if lease.use.resource == "net" else lease for lease in granted]

    owner = TwoRoutes({"net": 1, "w": 0})
    coordinator = Coordinator(prepared, {"net": owner, "w": owner}, installed, {"host": Host()}, checked=True)
    outcome = asyncio.run(asyncio.wait_for(coordinator.run(), timeout=10))
    assert outcome.status == "succeeded", outcome.error
    assert len(events) == 2, "each lease's own setup object applied the state, under one hold"


class Undoing(HeldRecording):
    """A held setup that would undo its change when its hold ends, which the runtime does not realize yet."""

    def undo(self, handle) -> None: ...


def test_a_held_setup_that_would_undo_its_change_is_refused_until_the_evaluation_point_swap_brings_it() -> None:
    installed = differentiating_installed()
    run = accepted_run(DIFFERENTIATING, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, Target())
    owner = Holding({"net": 1, "w": 0}, holding=("net",), setup=Undoing([]))
    outcome = asyncio.run(Coordinator(prepared, {"net": owner, "w": owner}, installed, {"host": Host()}, checked=True).run())
    assert outcome.status == "failed" and "come with the evaluation-point swap" in (outcome.error or "")
