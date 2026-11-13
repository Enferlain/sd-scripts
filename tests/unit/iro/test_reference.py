"""The reference semantics: lifecycles run to settlement with stand-ins, and the facts they record."""

import subprocess
from collections.abc import Mapping
import sys
from pathlib import Path

from library.iro.normalize import normalize
from library.iro.reference import CallResult, Reference, StandIn, Update, Version, returning
from library.iro.text import parse

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
WIDE = (HERE / "wide.iro").read_text()
LEASE = "protected_current lease"


def reference(text: str, resources: Mapping[str, object], standins: Mapping[str, StandIn]) -> Reference:
    return Reference(normalize(parse(text)), resources, standins)


def wide_standins(cache=lambda index: None) -> dict[str, StandIn]:
    return {
        "read_latent_cache": StandIn(("int", "str", "float", "bool"), ("optional latents",), lambda a: CallResult((cache(a[0]),))),
        "encode_latent": StandIn(("snapshot lease", "int"), ("latents",), lambda a: CallResult((f"encoded {a[1]}",))),
        "pack": StandIn(("latents", "count"), ("item",), lambda a: CallResult(({"latents": a[0]},))),
        "keep": StandIn(("item",), ("bool",), returning(True)),
        "encode_text": StandIn((LEASE, LEASE, "item"), ("retained conditioning",), returning("conditioning"), retains=(0, 1)),
        "denoise": StandIn(
            (LEASE, "item", "retained conditioning", "bool"), ("retained prediction",), returning("prediction"), retains=(0, 2)
        ),
        "measure": StandIn((LEASE,), (), returning()),
    }


def increment(given: tuple[object, ...]) -> CallResult:
    value = given[0]
    assert isinstance(value, int)
    return CallResult((), (Update(0, value + 1),))


def ordinal(version: object) -> int:
    assert isinstance(version, Version)
    return version.ordinal


PARTICIPANTS = {"vae": "vae", "te1": "te1", "te2": "te2", "unet": "unet", "lora": "lora"}


# The fixture.


def test_the_fixture_runs_to_settlement() -> None:
    run = reference(WIDE, PARTICIPANTS, wide_standins())
    run.run()
    assert run.settled and run.errors == []
    settled = run.settlements("items")
    trained = [d for d in settled.values() if d == ("forwarded", "consumed")]
    assert len(trained) == 4, "@training forks each item: the filter forwards one fork, the other is consumed"
    assert ("cancelled",) in settled.values(), "@turns carries its item and cancels it"
    assert [f.values[1] for f in run.facts_of("request_settled")] == ["completed"] * 4
    assert len(run.facts_of("product_released")) == 8
    assert all(o.status == "succeeded" for o in run.outcomes.values())


def test_items_whose_production_did_not_observe_the_admission_owner_are_rejected() -> None:
    run = reference(WIDE, PARTICIPANTS, wide_standins(cache=lambda index: None if index % 2 else "cached"))
    run.run()
    assert run.settled and run.errors == []
    rejected = [key for key, d in run.settlements("items").items() if d == ("rejected",)]
    assert rejected and all(key % 2 == 0 for key in rejected), "a cache hit never reads @vae, so @vae's lag cannot be judged"


# Requests and the version at the request.

REQUESTED = """iro.strategy @s uses(@step v1, @measure v1) {
  %m = iro.resource @m : participant

  iro.parallel @all capture(%m) {
    iro.repeat @training 3 times capture(%m) {
    ^(%i: int):
      iro.access %m owner_update {
      ^(%lease):
        iro.call @step(%lease)
      }
      iro.request @evaluation(%i) from @training at settled_success
    }

    iro.handler @evaluation sources [@training] max 3 capture(%m) {
      iro.access %m MODE at_request {
      ^(%lease):
        iro.call @measure(%lease)
      }
    }
  }
}
"""


def requested(mode: str) -> Reference:
    standins = {
        "step": StandIn(("owner_update lease",), (), increment, updates=(0,)),
        "measure": StandIn((f"{mode} lease",), (), returning()),
    }
    return reference(REQUESTED.replace("MODE", mode), {"m": 0}, standins)


def handler_reads(run: Reference) -> list[int]:
    return [ordinal(f.values[2]) for f in run.facts_of("leased") if f.owner.endswith("@evaluation/0") and f.values[0] == "m"]


def test_requested_work_reads_the_version_current_at_its_request() -> None:
    for mode in ("snapshot", "protected_current"):
        run = requested(mode)
        run.run()
        assert run.settled and run.errors == []
        reserved = [ordinal(f.values[0]) for f in run.facts_of("reserved")]
        assert reserved == [1, 2, 3], "each iteration's request follows its own update"
        assert handler_reads(run) == reserved


def updates_while_requested(mode: str) -> int:
    run = requested(mode)
    run.run()
    pending, count = 0, 0
    for fact in run.facts:
        if fact.kind == "reserved":
            pending += 1
        elif fact.kind == "request_settled":
            pending -= 1
        elif fact.kind == "updated" and pending:
            count += 1
    return count


def test_a_live_reservation_makes_owner_updates_wait_and_a_snapshot_does_not() -> None:
    assert updates_while_requested("protected_current") == 0
    assert updates_while_requested("snapshot") > 0, "a snapshot at the request is detached, so updates proceed"


# Claims, choices and retained results.

FORKED = """iro.strategy @s uses(@work v1, @keep v1) {
  %items = iro.stream @items : item
             bounds(outstanding 1, ready 1)

  iro.parallel @all capture(%items) {
    iro.producer @data into %items {
    ^(%index: int):
      %item = iro.call @work(%index) -> item
      iro.yield %item
    }

    iro.repeat @consumer 2 times capture(%items) {
      %c = iro.take %items
      %a, %b = iro.fork %c
      iro.parallel @both capture(%a, %b) {
        iro.dispose %a consumed

        iro.dispose %b filtered_out
      }
    }

    iro.stop @data after @consumer
  }
}
"""


def test_forks_captured_into_parallel_branches_settle_the_item_once_all_have() -> None:
    run = reference(FORKED, {}, {"work": StandIn(("int",), ("item",), lambda a: CallResult((a[0],))), "keep": StandIn((), (), returning())})
    run.run()
    assert run.settled and run.errors == []
    settled = run.settlements("items")
    assert [settled[0], settled[1]] == [("consumed", "filtered_out")] * 2


CHOSEN = """iro.strategy @s uses(@look v1, @make v1, @use v1) {
  iro.sequence @once {
    %found = iro.call @look() -> optional latents
    %latents = iro.choice %found {
    ^(%hit: latents):
      iro.yield %hit
    } else {
      iro.call @make() -> latents
    }
    iro.call @use(%latents)
  }
}
"""


def test_a_choice_on_an_optional_gives_the_first_alternative_the_value() -> None:
    for found, expected in (("cached", "cached"), (None, "made")):
        used: list[object] = []

        def use(given: tuple[object, ...], used: list[object] = used) -> CallResult:
            used.append(given[0])
            return CallResult(())

        standins = {
            "look": StandIn((), ("optional latents",), returning(found)),
            "make": StandIn((), ("latents",), returning("made")),
            "use": StandIn(("latents",), (), use),
        }
        run = reference(CHOSEN, {}, standins)
        run.run()
        assert run.errors == [] and used == [expected]


RETAINED = """iro.strategy @s uses(@objective v1, @report v1, @step v1) {
  %m = iro.resource @m : participant
  iro.sequence @train capture(%m) {
    %loss = iro.access %m protected_current {
    ^(%lease):
      iro.call @objective(%lease) -> retained loss
    }
    iro.call @report(%loss)
    iro.access %m owner_update {
    ^(%lease):
      iro.call @step(%lease)
    }
  }
}
"""


def test_a_retained_result_keeps_its_lease_after_the_access_until_its_release() -> None:
    standins = {
        "objective": StandIn((LEASE,), ("retained loss",), lambda a: CallResult((float(str(a[0])),)), retains=(0,)),
        "report": StandIn(("retained loss",), (), returning()),
        "step": StandIn(("owner_update lease",), (), increment, updates=(0,)),
    }
    run = reference(RETAINED, {"m": 1}, standins)
    run.run()
    assert run.settled and run.errors == []
    kinds = [f.kind for f in run.facts if f.kind in ("call_completed", "product_released", "lease_released", "updated")]
    released = kinds.index("product_released")
    assert kinds.index("lease_released") > released, "the protected lease outlives its access body until the release"
    assert kinds.index("updated") > released


# Failures, holds and cancellation.


def test_a_failed_call_settles_its_claims_and_reports_once() -> None:
    run = reference(FORKED, {}, {"work": StandIn(("int",), ("item",), lambda a: CallResult((a[0],))), "keep": StandIn((), (), returning())})
    run.fail("@s/@all/@data/0", "out of memory", ordinal=1)
    run.run()
    assert run.settled
    assert len(run.errors) == 1 and "out of memory" in run.errors[0]
    assert all(o.state == "terminal" for o in run.outcomes.values())


def test_a_held_completion_is_not_enabled_until_completed() -> None:
    run = reference(
        CHOSEN,
        {},
        {
            "look": StandIn((), ("optional latents",), returning(None)),
            "make": StandIn((), ("latents",), returning("x")),
            "use": StandIn(("latents",), (), returning()),
        },
    )
    run.hold("@s/@once/0")
    run.run()
    assert not run.settled and run.enabled() == ()
    run.complete("@s/@once/0")
    run.run()
    assert run.settled and run.errors == []


def test_cancellation_closes_admission_and_everything_settles() -> None:
    run = reference(WIDE, PARTICIPANTS, wide_standins())
    for _ in range(40):
        run.step()
    run.cancel()
    run.run()
    assert run.settled
    assert all(o.state == "terminal" for o in run.outcomes.values())
    assert not run.streams["items"].outstanding, "every item is settled"


def test_the_language_half_imports_no_torch() -> None:
    code = (
        "import sys\n"
        "import library.iro.accepted, library.iro.reference, library.iro.seams, library.iro.verify\n"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith('torch'))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# Regressions from the section review.


def test_cancellation_at_any_point_leaves_no_lease_or_retained_result() -> None:
    for after in range(0, 400, 7):
        run = reference(WIDE, PARTICIPANTS, wide_standins())
        for _ in range(after):
            if not run.step():
                break
        run.cancel()
        run.run()
        assert run.settled and run.errors == [], after
        assert not run.products and all(not lease.active for lease in run.leases.values()), after
        assert all(not r.readers and r.writer is None for r in run.resources.values()), after


UPDATED_WHILE_REQUESTED = """iro.strategy @s uses(@step v1, @measure v1) {
  %m = iro.resource @m : participant

  iro.parallel @p capture(%m) {
    iro.sequence @updater capture(%m) {
      iro.access %m owner_update {
      ^(%u):
        iro.call @step(%u)
      }
    }

    iro.repeat @train 1 times capture(%m) {
    ^(%i: int):
      iro.request @h(%i) from @train at settled_success
    }

    iro.handler @h sources [@train] max 1 capture(%m) {
      iro.access %m protected_current at_request {
      ^(%l):
        iro.call @measure(%l)
      }
    }
  }
}
"""


def test_a_request_made_during_an_update_reserves_the_updated_version() -> None:
    standins = {
        "step": StandIn(("owner_update lease",), (), increment, updates=(0,)),
        "measure": StandIn((LEASE,), (), returning()),
    }
    run = reference(UPDATED_WHILE_REQUESTED, {"m": 1}, standins)
    run.hold("@s/@p/@updater/0/0")
    for _ in range(200):
        if not run.step():
            break
    assert not run.facts_of("reserved"), "the request waits while the update is in flight"
    run.complete("@s/@p/@updater/0/0")
    run.run()
    assert run.settled and run.errors == []
    assert [ordinal(f.values[0]) for f in run.facts_of("reserved")] == [1]


UNTAKEN_SOURCE = """iro.strategy @s uses(@flag v1, @measure v1) {
  iro.parallel @p {
    iro.sequence @gate {
      %go = iro.call @flag() -> bool
      iro.choice %go {
        iro.repeat @train 1 times {
        ^(%i: int):
          iro.request @h(%i) from @train at settled_success
        }
      }
    }

    iro.handler @h sources [@train] max 1 {
      iro.call @measure()
    }
  }
}
"""


def test_a_handler_whose_source_never_starts_settles_once_it_cannot_start() -> None:
    for taken in (True, False):
        standins = {"flag": StandIn((), ("bool",), returning(taken)), "measure": StandIn((), (), returning())}
        run = reference(UNTAKEN_SOURCE, {}, standins)
        run.run()
        assert run.settled and run.errors == [], taken
        assert len(run.facts_of("request_settled")) == (1 if taken else 0)
