"""Preparation: matching, placement, fusion, device admissions, the manifest and loading a saved prepared form."""

from pathlib import Path
from typing import Any

import pytest

from library.iro.accepted import cold_load
from library.iro.diagnostics import Rejection
from library.iro.frontend import implementation
from library.iro.operation_contracts import symbol
from library.iro.prepare import HOST, DeviceFacts, Target, load_prepared, prepare
from library.iro.words.calls import CallOp
from library.iro.words.types import written
from tests.unit.iro.accepted_runs import accepted_run

WIDE = (Path(__file__).parent / "wide.iro").read_text()
CUDA = Target((HOST, DeviceFacts("cuda:0", "cuda"), DeviceFacts("cuda:1", "cuda")))


def installed_for(text: str, effects: dict[str, list[str]] | None = None, **declared):
    """Implementations declared as the strategy calls them, with the given effects; ``declared`` overrides one."""
    loaded = cold_load(accepted_run(text))
    shapes: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for op in loaded.normalized.strategy.walk():
        if isinstance(op, CallOp):
            shapes[symbol(op.implementation)] = (tuple(written(v.type) for v in op.operands), tuple(written(r.type) for r in op.results))
    installed = {}
    for name, version in zip((s.root_reference.data for s in loaded.strategy.uses or ()), loaded.strategy.use_versions or (), strict=True):
        inputs, outputs = shapes.get(name, ((), ()))
        options: dict[str, Any] = {
            "version": version.data,
            "inputs": inputs,
            "outputs": outputs,
            "effects": (effects or {}).get(name, ()),
            "name": name,
        }
        options |= declared.get(name, {})
        installed[name] = implementation(**options)(lambda *args: None)
    return installed


def prepared(text: str, target: Target = CUDA, effects=None, **declared):
    installed = installed_for(text, effects, **declared)
    run = accepted_run(text, installed=installed)
    return prepare(cold_load(run, installed=installed), installed, target), installed


DEVICE = {"encode_latent": ["device"], "encode_text": ["device"], "denoise": ["device"], "measure": ["device"]}


def test_calls_are_placed_by_their_declared_device_use() -> None:
    result, _ = prepared(WIDE, effects=DEVICE)
    placed = {name: g.device for g in result.groups for name in g.implementations}
    assert placed == {
        "read_latent_cache": "host",
        "encode_latent": "cuda:0",
        "pack": "host",
        "keep": "host",
        "encode_text": "cuda:0",
        "denoise": "cuda:0",
        "measure": "cuda:0",
    }
    assert sum(len(g.calls) for g in result.groups) == sum(1 for op in result.loaded.normalized.strategy.walk() if isinstance(op, CallOp))


def test_an_activity_can_be_placed_on_another_device() -> None:
    target = Target(CUDA.devices, placement=(("evaluation", "cuda:1"),))
    result, _ = prepared(WIDE, target, effects=DEVICE)
    assert {g.device for g in result.groups if "measure" in g.implementations} == {"cuda:1"}
    assert {g.device for g in result.groups if "denoise" in g.implementations} == {"cuda:0"}


def test_each_access_body_with_device_work_is_one_admission_ending_at_its_releases() -> None:
    result, _ = prepared(WIDE, effects=DEVICE)
    training = "@wide/@activities/@consumers/@training"
    assert [(a.begins, a.ends) for a in result.admissions] == [
        ("@wide/@activities/@data/4/1.0", ("at the end of its work",)),
        (f"{training}/5", (f"{training}/8",)),  # the text encoders' access, ended by `iro.release %cond`
        (f"{training}/6", (f"{training}/7",)),  # the denoiser's access, ended by `iro.release %prediction`
        ("@wide/@activities/@evaluation/0", ("at the end of its work",)),
    ]
    assert all(a.device == "cuda:0" for a in result.admissions)


CHAIN = """iro.strategy @s uses(@a v1, @b v1, @c v1) {
  iro.sequence @once {
    %x = iro.call @a() -> int
    %y = iro.call @b(%x) -> int
    %z = iro.call @c(%y) -> int
  }
}
"""


def test_consecutive_calls_on_one_device_form_one_group() -> None:
    result, _ = prepared(CHAIN)
    assert [g.implementations for g in result.groups] == [("a", "b", "c")]


@pytest.mark.parametrize(
    ("target", "effects", "expected"),
    [
        (CUDA, {"b": ["external"]}, [("a",), ("b",), ("c",)]),
        (CUDA, {"b": ["collectives"]}, [("a",), ("b",), ("c",)]),
        (CUDA, {"b": ["device"]}, [("a",), ("b",), ("c",)]),
        (Target(fuse=False), {}, [("a",), ("b",), ("c",)]),
    ],
    ids=["external effect", "collectives", "another device", "fusion off"],
)
def test_a_boundary_or_another_device_interrupts_a_chain(target, effects, expected) -> None:
    result, _ = prepared(CHAIN, target, effects)
    assert [g.implementations for g in result.groups] == expected


def test_an_implementation_with_its_own_completion_is_not_fused() -> None:
    result, _ = prepared(CHAIN, b={"version": "1", "inputs": ("int",), "outputs": ("int",), "completion": "own", "name": "b"})
    assert [g.implementations for g in result.groups] == [("a",), ("b",), ("c",)]


def test_a_call_that_disagrees_with_its_implementation_is_invalid_meaning() -> None:
    installed = installed_for(CHAIN)
    installed["b"] = implementation(version=1, inputs=("float",), outputs=("int",), name="b")(lambda x: x)
    run = accepted_run(CHAIN, installed=installed)
    with pytest.raises(Rejection) as raised:
        prepare(cold_load(run, installed=installed), installed, CUDA)
    assert raised.value.codes == ("iro-prep-001",) and not raised.value.target_limits_only


def test_device_work_on_a_target_without_a_device_is_a_target_limit() -> None:
    with pytest.raises(Rejection) as raised:
        prepared(WIDE, Target(), effects=DEVICE)
    assert set(raised.value.codes) == {"iro-prep-002"} and raised.value.target_limits_only


def test_collectives_on_several_ranks_are_a_target_limit_for_now() -> None:
    ranked = CHAIN.replace(" -> int\n", " -> int rank_rule(replicated_deterministic)\n")
    installed = installed_for(ranked, {"b": ["collectives"]})
    run = accepted_run(ranked, installed=installed, replicated=True)
    with pytest.raises(Rejection) as raised:
        prepare(cold_load(run, installed=installed), installed, CUDA)
    assert raised.value.codes == ("iro-prep-003",) and raised.value.target_limits_only


TAKE_INSIDE = """iro.strategy @s uses(@make v1, @use v1, @more v1) {
  %m = iro.resource @m : participant
  %items = iro.stream @items : item bounds(outstanding 1, ready 1)
  iro.parallel @all capture(%items, %m) {
    iro.producer @source into %items {
    ^(%index: int):
      %item = iro.call @make(%index) -> item
      iro.yield %item
    }
    iro.sequence @consume capture(%items, %m) {
      iro.access %m protected_current {
      ^(%lease):
        iro.call @use(%lease)
        %claim = iro.take %items
        %value = iro.borrow %claim
        iro.call @more(%lease, %value)
        iro.dispose %claim consumed
      }
    }
    iro.stop @source after @consume
  }
}
"""


def test_an_access_body_that_takes_after_its_device_work_is_a_target_limit() -> None:
    with pytest.raises(Rejection) as raised:
        prepared(TAKE_INSIDE, effects={"use": ["device"], "more": ["device"]})
    assert "iro-prep-006" in raised.value.codes and raised.value.target_limits_only


def test_a_saved_prepared_form_loads_only_when_it_derives_again() -> None:
    result, installed = prepared(WIDE, effects=DEVICE)
    loaded = result.loaded
    assert load_prepared(result.text(), loaded, installed, CUDA) == result
    forged = result.text().replace("groups 7;", "groups 6, 7;")
    assert forged != result.text(), "the forged form claims the two training accesses share one admission"
    with pytest.raises(Rejection) as raised:
        load_prepared(forged, loaded, installed, CUDA)
    assert raised.value.codes == ("iro-prep-004",)
    with pytest.raises(Rejection):
        load_prepared(result.text(), loaded, installed, Target(CUDA.devices, device="cuda:1"))


def test_the_manifest_keys_on_code_not_on_the_implementation_version() -> None:
    result, installed = prepared(CHAIN)
    replaced = dict(installed)
    replaced["b"] = type(installed["b"])(installed["b"].declaration, len)
    again = prepare(result.loaded, replaced, CUDA)
    codes = {m.name: m.code for m in result.manifest}
    assert {m.name: m.code for m in again.manifest}["b"] != codes["b"], "same declaration and version, other code"
    assert [m.digest for m in again.manifest] == [m.digest for m in result.manifest]
    with pytest.raises(Rejection):
        load_prepared(result.text(), result.loaded, replaced, CUDA)


def test_the_printout_shows_groups_devices_and_admissions() -> None:
    text = prepared(WIDE, effects=DEVICE)[0].text()
    assert text.startswith("prepared @wide\n")
    assert "on cuda:0" in text and "admissions\n  cuda:0 from" in text and "@denoise v1 digest" in text


def test_preparing_twice_gives_the_same_form() -> None:
    assert prepared(WIDE, effects=DEVICE)[0].text() == prepared(WIDE, effects=DEVICE)[0].text()


BOUNDED = """iro.strategy @s uses(@a v1, @b v1, @c v1, @d v1) {
  %m = iro.resource @m : participant
  iro.sequence @once capture(%m) {
    %x = iro.call @a() -> int
    %r = iro.access %m protected_current {
    ^(%lease):
      %y = iro.call @b(%lease) -> int
      %z = iro.call @c(%y) -> int
      %w = iro.call @d(%lease, %z) -> int
      iro.yield %w
    }
    %v = iro.call @a() -> int
  }
}
"""


def test_an_access_closes_a_group_and_one_access_is_one_admission_over_its_groups() -> None:
    result, _ = prepared(BOUNDED, effects={"b": ["device"], "d": ["device"]})
    assert [(g.implementations, g.device) for g in result.groups] == [
        (("a",), "host"),
        (("a",), "host"),
        (("b",), "cuda:0"),
        (("c",), "host"),
        (("d",), "cuda:0"),
    ]
    (admission,) = result.admissions
    assert admission.groups == (3, 5) and admission.begins.endswith("@once/1")


def test_one_access_body_on_two_devices_is_a_target_limit() -> None:
    split = BOUNDED.replace(
        """      %w = iro.call @d(%lease, %z) -> int
      iro.yield %w""",
        """      %w = iro.sequence @inner capture(%lease, %z) {
        %u = iro.call @d(%lease, %z) -> int
        iro.yield %u
      }
      iro.yield %w""",
    )
    target = Target(CUDA.devices, placement=(("inner", "cuda:1"),))
    with pytest.raises(Rejection) as raised:
        prepared(split, target, effects={"b": ["device"], "d": ["device"]})
    assert raised.value.codes == ("iro-prep-005",) and raised.value.target_limits_only


def test_the_innermost_placed_activity_decides() -> None:
    target = Target(CUDA.devices, placement=(("activities", "cuda:1"), ("training", "cuda:0")))
    result, _ = prepared(WIDE, target, effects=DEVICE)
    devices = {name: g.device for g in result.groups for name in g.implementations if g.device != "host"}
    assert devices == {"encode_latent": "cuda:1", "encode_text": "cuda:0", "denoise": "cuda:0", "measure": "cuda:1"}


def test_a_placement_that_names_no_activity_is_refused() -> None:
    with pytest.raises(Rejection) as raised:
        prepared(WIDE, Target(CUDA.devices, placement=(("evalution", "cuda:1"),)), effects=DEVICE)
    assert raised.value.codes == ("iro-prep-007",)


def test_a_take_inside_an_activity_of_its_own_is_not_the_accesss_take() -> None:
    nested = TAKE_INSIDE.replace(
        """        iro.call @use(%lease)
        %claim = iro.take %items
        %value = iro.borrow %claim
        iro.call @more(%lease, %value)
        iro.dispose %claim consumed""",
        """        iro.call @use(%lease)
        iro.repeat @inner 2 times capture(%items) {
        ^(%i: int):
          %claim = iro.take %items
          iro.dispose %claim consumed
        }""",
    ).replace("uses(@make v1, @use v1, @more v1)", "uses(@make v1, @use v1)")
    result, _ = prepared(nested, effects={"use": ["device"]})
    assert len(result.admissions) == 1


@pytest.mark.parametrize(
    "bad",
    [
        lambda: Target((DeviceFacts("cuda:0", "cuda"),)),
        lambda: Target((HOST, DeviceFacts("cuda:0", "cuda"), DeviceFacts("cuda:0", "cuda"))),
        lambda: Target((HOST,), device="host"),
        lambda: Target(CUDA.devices, placement=(("a", "cuda:0"), ("a", "cuda:1"))),
        lambda: Target(workers=0),
        lambda: Target((HOST, DeviceFacts("cuda:0", "xpu"))),
        lambda: Target((HOST, DeviceFacts("host", "cuda"))),
        lambda: Target((HOST, DeviceFacts("Cuda:0", "Cuda"))),
        lambda: Target((HOST, DeviceFacts("cuda:01", "cuda"))),
    ],
    ids=[
        "no host",
        "duplicate device",
        "device work on the host",
        "placed twice",
        "no worker",
        "name of another kind",
        "host of a kind",
        "not a device type",
        "a second name for one device",
    ],
)
def test_a_target_describes_itself_consistently(bad) -> None:
    with pytest.raises(ValueError):
        bad()


def test_a_device_of_any_pytorch_backend_is_named_by_its_device_type_and_shows_what_it_offers() -> None:
    target = Target((HOST, DeviceFacts("xpu:0", "xpu", frozenset({"ordering", "memory"}))))
    assert target.device_work == "xpu:0"
    assert "xpu:0 (xpu; memory; ordering)" in target.describe()


class Callable:
    def __call__(self, *args):
        return None


def test_code_identity_follows_the_code_and_never_merges_unknown_code() -> None:
    import functools

    from library.iro.prepare import code_identity

    def local(x):
        return x

    assert code_identity(functools.partial(local, 1)) == code_identity(local), "a partial is its function's code"
    assert code_identity(Callable()) == code_identity(local), "a callable object is identified by its class's module"
    assert code_identity(len) != code_identity(abs), "builtins have no source and never share an identity"
    assert code_identity(len) == code_identity(len)


RETAINED_ACROSS_TAKE = """iro.strategy @s uses(@make v1, @keep v1, @look v1) {
  %m = iro.resource @m : participant
  %items = iro.stream @items : item bounds(outstanding 1, ready 1)
  iro.parallel @all capture(%items, %m) {
    iro.producer @source into %items {
    ^(%index: int):
      %item = iro.call @make(%index) -> item
      iro.yield %item
    }
    iro.sequence @consume capture(%items, %m) {
      %r = iro.access %m protected_current {
      ^(%lease):
        iro.call @keep(%lease) -> retained int
      }
      %claim = iro.take %items
      %value = iro.borrow %claim
      %v = iro.call @look(%r, %value) -> int
      iro.dispose %claim consumed
    }
    iro.stop @source after @consume
  }
}
"""


def test_a_take_while_a_device_holding_result_is_live_is_a_target_limit() -> None:
    with pytest.raises(Rejection) as raised:
        prepared(RETAINED_ACROSS_TAKE, effects={"keep": ["device"], "look": ["device"]})
    assert raised.value.codes == ("iro-prep-009",) and raised.value.target_limits_only
    assert prepared(RETAINED_ACROSS_TAKE, effects={"look": ["device"]})[0], "host work holds no device"


REQUEST_IN_ADMISSION = """iro.strategy @s uses(@work v1, @see v1) {
  %m = iro.resource @m : participant
  iro.parallel @all capture(%m) {
    iro.sequence @driver capture(%m) {
      iro.access %m protected_current {
      ^(%lease):
        %x = iro.call @work(%lease) -> int
        iro.request @check(%x) from @driver at settled_success
      }
    }
    iro.handler @check sources [@driver] max 1 {
    ^(%value: int):
      %y = iro.call @see(%value) -> int
    }
  }
}
"""


def test_a_request_made_while_an_admission_holds_the_device_is_a_target_limit() -> None:
    with pytest.raises(Rejection) as raised:
        prepared(REQUEST_IN_ADMISSION, effects={"work": ["device"]})
    assert raised.value.codes == ("iro-prep-010",) and raised.value.target_limits_only
    assert prepared(REQUEST_IN_ADMISSION)[0], "without device work there is no admission to hold"


CARRY = """iro.strategy @s uses(@first v1, @peek v1, @step v1) {
  iro.sequence @once {
    %f = iro.call @first() -> int
    %last = iro.repeat @loop 3 times carry(%c = %f) {
    ^(%i: int):
      %p = iro.call @peek(%c) -> int
      %n = iro.call @step(%p) -> int
      iro.yield %n
    }
  }
}
"""


def test_a_device_result_a_repeat_carries_to_host_work_in_its_next_iteration_is_waited_for() -> None:
    result, _ = prepared(CARRY, effects={"step": ["device"]})
    step = next(g for g in result.groups if g.implementations == ("step",))
    assert step.device == "cuda:0"
    assert step.sync, "the next iteration's host @peek reads the carried device result"


def test_a_device_result_a_repeat_carries_only_to_device_work_on_its_device_is_not_waited_for() -> None:
    result, _ = prepared(CARRY, effects={"step": ["device"], "peek": ["device"], "first": ["device"]})
    group = next(g for g in result.groups if "step" in g.implementations)
    assert not group.sync, "the carried value only feeds device work on the same device"


LEAVING = """iro.strategy @s uses(@encode v1) {
  %vae = iro.resource @vae : participant may_leave_device
  iro.sequence @first capture(%vae) {
    iro.access %vae protected_current {
    ^(%lease):
      %x = iro.call @encode(%lease) -> int
    }
  }
  iro.sequence @second capture(%vae) {
    iro.access %vae protected_current {
    ^(%lease):
      %y = iro.call @encode(%lease) -> int
    }
  }
}
"""
COPIES = frozenset({"ordering", "copies"})


def leaving_target(*devices: str, offers=COPIES, placement=()) -> Target:
    return Target((HOST, *(DeviceFacts(d, "cuda", offers) for d in devices)), placement=placement)


def test_a_frozen_participant_used_on_one_device_that_copies_storage_may_leave_it() -> None:
    result, _ = prepared(LEAVING, leaving_target("cuda:0"), effects={"encode": ["device"]})
    assert result.leaving == (("vae", "cuda:0"),)
    assert "leaving the device\n  @vae from cuda:0\n" in result.text()


def test_a_participant_no_device_work_uses_has_nothing_to_leave() -> None:
    result, _ = prepared(LEAVING, leaving_target("cuda:0"))
    assert result.leaving == ()


@pytest.mark.parametrize(
    ("text", "target", "code"),
    [
        (
            LEAVING.replace("uses(@encode v1)", "uses(@encode v1, @tune v1)").replace(
                "  iro.sequence @second",
                "  iro.access %vae owner_update {\n  ^(%lease):\n    %u = iro.call @tune(%lease) -> int\n  }\n  iro.sequence @second",
            ),
            leaving_target("cuda:0"),
            "iro-prep-011",
        ),
        (LEAVING, leaving_target("cuda:0", "cuda:1", placement=(("first", "cuda:0"), ("second", "cuda:1"))), "iro-prep-012"),
        (LEAVING, leaving_target("cuda:0", offers=frozenset({"ordering"})), "iro-prep-013"),
    ],
    ids=["the strategy updates it", "used on two devices", "the adapter does not copy storage"],
)
def test_a_permission_to_leave_the_device_that_cannot_be_realized_is_a_target_limit(text: str, target: Target, code: str) -> None:
    with pytest.raises(Rejection) as rejected:
        prepared(text, target, effects={"encode": ["device"], "tune": ["device"]})
    assert rejected.value.codes == (code,) and rejected.value.target_limits_only
