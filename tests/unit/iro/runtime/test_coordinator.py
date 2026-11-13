"""The coordinator's own guarantees: worker threads, fused groups, failure and cancellation, device turns."""

import asyncio
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

from library.iro.accepted import cold_load
from library.iro.devices import Host, Returned
from library.iro.frontend import implementation
from library.iro.prepare import HOST, DeviceFacts, Target, prepare
from library.iro.runtime import Coordinator, Fact
from tests.unit.iro.accepted_runs import accepted_run
from tests.unit.iro.runtime.harness import Memory

CUDA = Target((HOST, DeviceFacts("cuda:0", "cuda")), workers=4)


@dataclass
class FakeDevice:
    """A device that records how many admissions' work overlaps on it."""

    name: str = "cuda:0"
    active: int = 0
    most: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @contextmanager
    def context(self, call, after, reaches):
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
    assert not coordinator.outcomes and not coordinator.leases


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
    assert all(t.holder is None and not t.waiting for t in coordinator.turns.values()) and not coordinator.admissions


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
        assert not coordinator.admissions and all(t.holder is None for t in coordinator.turns.values())


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
    assert not coordinator.admissions and all(t.holder is None and not t.waiting for t in coordinator.turns.values())
    assert not coordinator.leases and not coordinator.products and not coordinator.outcomes


def test_a_run_where_every_activity_waits_reports_a_stall_instead_of_hanging() -> None:
    installed = {"a": implementation(version=1, outputs=("int",), name="a")(lambda: 1)}
    text = "iro.strategy @s uses(@a v1) {\n  iro.sequence @once {\n    %x = iro.call @a() -> int\n  }\n}\n"
    run = accepted_run(text, installed=installed)
    coordinator = Coordinator(prepare(cold_load(run, installed=installed), installed, Target()), {}, installed, {"host": Host()})

    async def wait_forever():
        await coordinator._until(lambda: False)

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
    assert not coordinator._busy


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
    original = Coordinator._changed

    def losing(self, *keys):
        return original(self, *(key for key in keys if key != ("resource", "m")))

    Coordinator._changed = losing  # ty: ignore[invalid-assignment]
    try:
        outcome, _, coordinator = execute(TWO_WRITERS, installed, resources={"m": 0})
    finally:
        Coordinator._changed = original  # ty: ignore[invalid-assignment]
    assert outcome.status == "failed"
    assert any("without a change on the keys it waits on" in error for error in coordinator.errors), coordinator.errors


def test_a_host_call_that_loads_torch_itself_fails_instead_of_running_under_the_threads_defaults(monkeypatch) -> None:
    import sys

    import pytest

    from library.iro.seams import CallContext

    torch = pytest.importorskip("torch")
    host = Host()
    monkeypatch.delitem(sys.modules, "torch")
    with pytest.raises(RuntimeError, match="loaded Torch itself"), host.context(CallContext(), (), ()):
        sys.modules["torch"] = torch  # what importing it inside the call does
    with host.context(CallContext(grad=False), (), ()):
        assert not torch.is_grad_enabled(), "once Torch is loaded, the host establishes the call's settings"
