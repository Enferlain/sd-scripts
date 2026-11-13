"""The accepted run: what it pins, its authority seed, and cold loading that refuses parts that do not correspond."""

from dataclasses import replace
from pathlib import Path

import pytest

from library.iro.accepted import DeclaredParticipant, DeclaredRelationship, cold_load, implementation_versions
from library.iro.composition import Composition, Definition, core
from library.iro.diagnostics import Rejection
from library.iro.verify import Obligations
from tests.unit.iro.accepted_runs import accepted_run

WIDE = (Path(__file__).parent / "wide.iro").read_text()


def test_an_accepted_run_pins_its_contract_and_implementations() -> None:
    run = accepted_run(WIDE)
    assert (run.contract, run.contract_version) == ("kuroiro", core().version)
    assert implementation_versions(run)["load_sdxl"] == "1"
    assert run.written == WIDE, "a strategy read from a written form without locations prints back the same"
    assert run.normalized.startswith("iro.strategy @wide normalized")
    assert run.identity == accepted_run(WIDE).identity, "two runs of one strategy share its content identity"


def test_the_authority_seed_declares_every_participant_and_relationship_unbound() -> None:
    seed = accepted_run(WIDE).seed
    assert [p.name for p in seed.participants] == ["vae", "te1", "te2", "unet", "lora"]
    assert DeclaredParticipant("lora", "participant", "build_lora", (("unet", "inspection"),)) in seed.participants
    assert seed.relationships == (DeclaredRelationship("lora_on_unet", "attachment", "lora", "unet"),)
    assert seed.obligations == accepted_run(WIDE).obligations.revision


def test_locations_survive_into_the_written_form_and_the_seed() -> None:
    located = WIDE.replace(
        "  %vae  = iro.resource @vae : participant materialize(@load_sdxl)",
        '  %vae  = iro.resource @vae : participant materialize(@load_sdxl) loc("strategies/sdxl/finetune.py":12)',
    )
    run = accepted_run(located)
    assert 'loc("strategies/sdxl/finetune.py":12)' in run.written
    assert run.seed.participants[0].origin == "strategies/sdxl/finetune.py:12"
    assert cold_load(run).run is run


def test_a_saved_run_cold_loads_to_the_same_obligations() -> None:
    run = accepted_run(WIDE)
    loaded = cold_load(run)
    assert loaded.verification.obligations == run.obligations
    assert loaded.normalized.text() == run.normalized


@pytest.mark.parametrize(
    "tamper",
    [
        lambda run: replace(run, implementations=(("load_sdxl", "2"), *run.implementations[1:])),
        lambda run: replace(run, normalized=run.normalized.replace("iro.release %cond\n", "")),
        lambda run: replace(run, obligations=Obligations(run.obligations.entries[1:])),
        lambda run: replace(run, seed=replace(run.seed, participants=run.seed.participants[1:])),
        lambda run: replace(run, written=run.written.replace("4 times", "5 times")),
    ],
    ids=["implementations", "normalized form", "obligations", "authority seed", "strategy"],
)
def test_a_saved_run_whose_parts_do_not_correspond_is_refused(tamper) -> None:
    with pytest.raises(Rejection) as raised:
        cold_load(tamper(accepted_run(WIDE)))
    assert set(raised.value.codes) == {"iro-run-002"}


def test_a_run_pinned_to_another_contract_is_refused() -> None:
    from library.iro.words.calls import CallOp

    newer = Composition([*(d for d in core().definitions.values() if d.name != "iro.call"), Definition(CallOp, "calls", version=2)])
    with pytest.raises(Rejection) as raised:
        cold_load(accepted_run(WIDE), newer)
    assert raised.value.codes == ("iro-run-001",)


def test_a_run_accepted_for_several_ranks_cold_loads_with_its_rank_obligations() -> None:
    text = (
        "iro.strategy @s uses(@work v1) {\n  iro.sequence @once {\n    iro.call @work() rank_rule(decided_once_and_distributed)\n  }\n}\n"
    )
    run = accepted_run(text, replicated=True)
    assert run.replicated and any(o.evidence.startswith("decided once and distributed") for o in run.obligations)
    assert cold_load(run).verification.obligations == run.obligations
    assert run.identity != accepted_run(text).identity, "the rank setting is part of what the run pins"
