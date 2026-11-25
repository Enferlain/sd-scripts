"""Preparation: matching, placement, fusion, device admissions, the manifest and loading a saved prepared form."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from library.iro.accepted import cold_load
from library.iro.diagnostics import Problems, Rejection, Stage
from library.iro.frontend import implementation
from library.iro.operation_contracts import symbol
from library.iro.prepare import HOST, DeviceFacts, Target, load_prepared, prepare
from library.iro.prepare.derive import Derived, Program
from library.iro.seams import annotate
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
    for name in (s.root_reference.data for s in loaded.strategy.uses or ()):
        inputs, outputs = shapes.get(name, ((), ()))
        options: dict[str, Any] = {
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


CHAIN = """iro.strategy @s uses(@a, @b, @c) {
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
    result, _ = prepared(CHAIN, b={"inputs": ("int",), "outputs": ("int",), "completion": "own", "name": "b"})
    assert [g.implementations for g in result.groups] == [("a",), ("b",), ("c",)]


def test_a_call_that_disagrees_with_its_implementation_is_invalid_meaning() -> None:
    installed = installed_for(CHAIN)
    installed["b"] = implementation(inputs=("float",), outputs=("int",), name="b")(lambda x: x)
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


TAKE_INSIDE = """iro.strategy @s uses(@make, @use, @more) {
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


def test_preparation_refuses_other_code_under_the_declaration_the_run_recorded() -> None:
    result, installed = prepared(CHAIN)
    records = {m.name: m for m in result.manifest}
    assert records["b"].declaration == installed["b"].declaration.text and records["b"].source is not None, "a record keeps its parts"
    replaced = dict(installed)
    replaced["b"] = type(installed["b"])(installed["b"].declaration, len)
    with pytest.raises(Rejection) as rejected:
        prepare(result.loaded, replaced, CUDA)
    assert rejected.value.codes == ("iro-run-003",) and "its source" in str(rejected.value), "the same declaration is not the same code"
    with pytest.raises(Rejection):
        load_prepared(result.text(), result.loaded, replaced, CUDA)


def test_the_printout_shows_groups_devices_and_admissions() -> None:
    text = prepared(WIDE, effects=DEVICE)[0].text()
    assert text.startswith("prepared @wide\n")
    assert "on cuda:0" in text and "admissions\n  cuda:0 from" in text and "  @denoise\n    declares name denoise\n" in text


def test_preparing_twice_gives_the_same_form() -> None:
    assert prepared(WIDE, effects=DEVICE)[0].text() == prepared(WIDE, effects=DEVICE)[0].text()


BOUNDED = """iro.strategy @s uses(@a, @b, @c, @d) {
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
    ).replace("uses(@make, @use, @more)", "uses(@make, @use)")
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


RETAINED_ACROSS_TAKE = """iro.strategy @s uses(@make, @keep, @look) {
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


REQUEST_IN_ADMISSION = """iro.strategy @s uses(@work, @see) {
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


CARRY = """iro.strategy @s uses(@first, @peek, @step) {
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


LEAVING = """iro.strategy @s uses(@encode) {
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
            LEAVING.replace("uses(@encode)", "uses(@encode, @tune)").replace(
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


# Reachability over named relations, and what domains derive (D16).

LEARNING = """iro.strategy @s uses(@batch, @encode, @measure, @predict, @objective, @learn) {
  %net = iro.resource @net : participant
  %enc = iro.resource @enc : participant
  %w = iro.resource @w
  iro.sequence @once capture(%net, %enc, %w) {
    %batch = iro.call @batch() -> batch
    %cond = iro.access %enc protected_current {
    ^(%e):
      iro.call @encode(%e, %batch) -> retained cond
    }
    %loss = iro.access %net protected_current {
    ^(%n):
      %seen = iro.call @measure(%n, %batch) -> int
      %prediction = iro.call @predict(%n, %cond, %batch) -> retained prediction
      iro.call @objective(%prediction, %batch) -> retained loss
    }
    iro.access %w owner_update {
    ^(%u):
      iro.call @learn(%u, %loss)
    }
  }
}
"""

# A relation name the core has never heard of: it compares names and never interprets them.
FLOWS = {
    "encode": {"relations": {"flows": [(0, 0)]}},  # the encoder's weights into its conditioning
    "predict": {"relations": {"flows": [(0, 0), (1, 0)]}},  # the network's weights and the conditioning, not the batch
    "objective": {"relations": {"flows": [(0, 0)]}},  # the prediction, not the batch
}


def program_of(text: str, **declared):
    installed = installed_for(text, None, **declared)
    loaded = cold_load(accepted_run(text, installed=installed), installed=installed)
    problems = Problems(Stage.PREPARE)
    program = Program(loaded.normalized.strategy, lambda call: installed[symbol(call.implementation)].declaration, problems)
    return program, problems


def call_named(program: Program, name: str) -> CallOp:
    return next(call for call in program.calls() if program.implementation(call) == name)


def reached_names(program: Program, reached) -> dict[str, set[str]]:
    return {program.implementation(call): set(keys) for call, keys in reached.calls.items()}


def learning_sink(program: Program) -> dict[str, list]:
    return {"w": [call_named(program, "learn").operands[1]]}


def test_reach_follows_declared_edges_into_a_sink_through_bodies() -> None:
    program, _ = program_of(LEARNING, **FLOWS)
    reached = program.reach("flows", learning_sink(program))
    assert reached_names(program, reached) == {"encode": {"w"}, "predict": {"w"}, "objective": {"w"}}, (
        "the batch reaches only inputs with no edge, the measurement reaches nothing, and the learner's own results reach nothing"
    )
    assert program.reach("other", learning_sink(program)).calls == {}, "a relation no implementation declares reaches nothing"


def test_reach_from_sources_keeps_only_the_calls_a_source_leads_into() -> None:
    program, _ = program_of(LEARNING, **FLOWS)
    leases = {resource: value for value, resource, _ in program.leases()}
    reached = program.reach("flows", learning_sink(program), [leases["net"]])
    assert reached_names(program, reached) == {"predict": {"w"}, "objective": {"w"}}, "the encoder lies on no path from the network"
    assert reached.sources == {"w": frozenset({leases["net"]})}
    assert program.reach("flows", learning_sink(program), [leases["enc"], leases["net"]]).sources["w"] == {leases["enc"], leases["net"]}


def test_an_input_with_no_edge_stops_the_path() -> None:
    program, _ = program_of(LEARNING, **(FLOWS | {"objective": {}}))
    assert program.reach("flows", learning_sink(program)).calls == {}, "without the objective's edge nothing reaches the loss"


def test_a_relation_declaration_names_positions_that_exist_and_its_digest_covers_it() -> None:
    with pytest.raises(ValueError, match="declares 1 inputs"):
        implementation(inputs=("retained loss",), outputs=("loss",), relations={"flows": [(1, 0)]})(lambda *a: None)
    with pytest.raises(ValueError, match="declares 1 outputs"):
        implementation(inputs=("owner_update lease",), outputs=("loss",), relations={"flows": [(0, 1)]})(lambda *a: None)
    plain = implementation(inputs=("retained loss",), outputs=("loss",))(lambda *a: None).declaration
    marked = implementation(inputs=("retained loss",), outputs=("loss",), relations={"flows": [(0, 0)]})(lambda *a: None)
    assert plain.digest != marked.declaration.digest, "the declaration's digest covers its relations"
    assert marked.declaration.edges("flows") == ((0, 0),) and marked.declaration.edges("other") == ()


CARRIED_INTO_LEARNING = """iro.strategy @s uses(@first, @step, @learn) {
  %w = iro.resource @w
  iro.sequence @once capture(%w) {
    %f = iro.call @first() -> loss
    %last = iro.repeat @loop 2 times capture(%w) carry(%c = %f) {
    ^(%i: int):
      %n = iro.call @step(%c) -> loss
      iro.yield %n
    }
    iro.access %w owner_update {
    ^(%u):
      iro.call @learn(%u, %last)
    }
  }
}
"""


def test_reach_follows_a_value_into_a_repeat_as_its_carry_and_out_as_its_result() -> None:
    program, _ = program_of(CARRIED_INTO_LEARNING, step={"relations": {"flows": [(0, 0)]}})
    sinks = learning_sink(program)
    assert reached_names(program, program.reach("flows", sinks)) == {"step": {"w"}}
    first = call_named(program, "first").results[0]
    assert program.reach("flows", sinks, [first]).sources == {"w": frozenset({first})}, (
        "the initial carry reaches the loss through the repeat"
    )


FILTERED_INTO_LEARNING = """iro.strategy @s uses(@make, @keep, @predict, @learn) {
  %w = iro.resource @w
  %items = iro.stream @items : item bounds(outstanding 1, ready 1)
  iro.parallel @all capture(%items, %w) {
    iro.producer @source into %items {
    ^(%index: int):
      %item = iro.call @make(%index) -> item
      iro.yield %item
    }
    iro.sequence @consume capture(%items, %w) {
      %claim = iro.take %items
      %kept = iro.filter %claim {
      ^(%value: item):
        %keep = iro.call @keep(%value) -> bool
        iro.yield %keep
      }
      %loss = iro.call @predict(%kept) -> loss
      iro.access %w owner_update {
      ^(%u):
        iro.call @learn(%u, %loss)
      }
    }
    iro.stop @source after @consume
  }
}
"""

THROUGH_A_STREAM = {name: {"relations": {"flows": [(0, 0)]}} for name in ("make", "keep", "predict")}


def test_reach_crosses_a_stream_unless_the_query_keeps_to_one_activity() -> None:
    program, _ = program_of(FILTERED_INTO_LEARNING, **THROUGH_A_STREAM)
    sinks = learning_sink(program)
    assert reached_names(program, program.reach("flows", sinks)) == {"make": {"w"}, "predict": {"w"}}, (
        "the core cuts no relation at a stream; the filter's decision is no result, so the keep call reaches nothing"
    )
    assert reached_names(program, program.reach("flows", sinks, within_activity=True)) == {"predict": {"w"}}, (
        "a query that keeps to one activity follows no value out of the producer"
    )


@dataclass(frozen=True)
class Annotating:
    given: dict
    name: str = "test.annotating"

    def derive(self, program):
        return {call: self.given[program.implementation(call)] for call in program.calls() if program.implementation(call) in self.given}


def test_calls_fuse_only_where_what_was_derived_for_them_is_the_same() -> None:
    effects = {name: ["device"] for name in ("encode", "measure", "predict", "objective", "learn")}
    installed = installed_for(LEARNING, effects)
    run = accepted_run(LEARNING, installed=installed)
    serving = Derived(annotate({"units": ["w"]}))
    deriver = Annotating({"predict": serving, "objective": serving})
    result = prepare(cold_load(run, installed=installed), installed, CUDA, [deriver])
    (measure,) = [g for g in result.groups if "measure" in g.implementations]
    (predict,) = [g for g in result.groups if "predict" in g.implementations]
    assert measure.index != predict.index, "calls with different annotations are not fused"
    assert predict.implementations == ("predict", "objective") and predict.annotations == (("units", ("w",)),)
    assert "@predict, @objective (units w, sync)" in result.text() and "derived by\n  @test.annotating\n" in result.text()


def test_two_derivers_never_give_one_call_the_same_annotation() -> None:
    installed = installed_for(LEARNING)
    run = accepted_run(LEARNING, installed=installed)
    serving = Derived(annotate({"units": ["w"]}))
    derivers = [Annotating({"predict": serving}), Annotating({"predict": serving}, name="test.again")]
    with pytest.raises(Rejection) as rejected:
        prepare(cold_load(run, installed=installed), installed, CUDA, derivers)
    assert rejected.value.codes == ("iro-prep-014",)


def test_one_name_claimed_by_two_derivers_or_by_setups_with_different_code_is_refused() -> None:
    installed = installed_for(LEARNING)
    run = accepted_run(LEARNING, installed=installed)
    loaded = cold_load(run, installed=installed)
    twice = [Annotating({"predict": Derived(annotate({"a": ["x"]}))}), Annotating({"objective": Derived(annotate({"b": ["x"]}))})]
    with pytest.raises(Rejection) as rejected:
        prepare(loaded, installed, CUDA, twice)
    assert rejected.value.codes == ("iro-prep-015",), "two derivers under one name"
    elsewhere = type("Recording", (), {"name": "test.setup", "entered": lambda self, a, h: None, "__module__": "elsewhere"})
    setups = Annotating({"predict": Derived((), (Setup(),)), "objective": Derived((("c", ("x",)),), (elsewhere(),))})  # ty: ignore[invalid-argument-type]
    with pytest.raises(Rejection) as rejected:
        prepare(loaded, installed, CUDA, [setups])
    assert rejected.value.codes == ("iro-prep-015",), "two call setups claiming one name with different code"


@dataclass(frozen=True)
class Setup:
    name: str = "test.setup"

    def entered(self, annotations, handles):
        return None
