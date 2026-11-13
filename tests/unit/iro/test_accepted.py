"""The accepted run: what it pins, its authority seed, and cold loading that refuses parts that do not correspond."""

from dataclasses import replace
from pathlib import Path

import pytest

from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, prop_def, var_result_def

from library.iro.accepted import DeclaredParticipant, DeclaredRelationship, check_implementations, cold_load, implementation_versions
from library.iro.composition import Composition, Definition, core
from library.iro.diagnostics import Rejection
from library.iro.frontend import implementation
from library.iro.seams import Declaration
from library.iro.verify import Obligations
from library.iro.words.calls import CallOp
from tests.unit.iro.accepted_runs import accepted_run

WIDE = (Path(__file__).parent / "wide.iro").read_text()
ONE_CALL = "iro.strategy @s uses(@work v1) {\n  iro.sequence @once {\n    iro.call @work()\n  }\n}\n"


def test_an_accepted_run_pins_the_definitions_it_uses_and_its_implementations() -> None:
    run = accepted_run(WIDE)
    pinned = {name: (version, digest) for name, version, digest in run.definitions}
    assert pinned["iro.call"] == (1, core()["iro.call"].digest)
    assert "iro.access" in pinned and "iro.lease" in pinned, "words and the types they carry are pinned"
    small = {name for name, _, _ in accepted_run(ONE_CALL).definitions}
    assert small == {"iro.strategy", "iro.sequence", "iro.call", "iro.yield"}, "a definition the strategy does not use is not pinned"
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
        lambda run: replace(run, definitions=run.definitions[1:]),
        lambda run: replace(run, implementations=(("load_sdxl", "2", ""), *run.implementations[1:])),
        lambda run: replace(run, normalized=run.normalized.replace("iro.release %cond\n", "")),
        lambda run: replace(run, obligations=Obligations(run.obligations.entries[1:])),
        lambda run: replace(run, seed=replace(run.seed, participants=run.seed.participants[1:])),
        lambda run: replace(run, written=run.written.replace("4 times", "5 times")),
    ],
    ids=["definitions", "implementations", "normalized form", "obligations", "authority seed", "strategy"],
)
def test_a_saved_run_whose_parts_do_not_correspond_is_refused(tamper) -> None:
    with pytest.raises(Rejection) as raised:
        cold_load(tamper(accepted_run(WIDE)))
    assert set(raised.value.codes) == {"iro-run-002"}


def _replacing_call(definition: Definition) -> Composition:
    return Composition([*(d for d in core().definitions.values() if d.name != "iro.call"), definition])


def test_a_run_whose_pinned_definition_has_a_new_version_is_refused() -> None:
    with pytest.raises(Rejection) as raised:
        cold_load(accepted_run(WIDE), _replacing_call(Definition(CallOp, "calls", version=2)))
    assert raised.value.codes == ("iro-run-001",)


@irdl_op_definition
class ChangedCallOp(IRDLOperation):
    """A call word whose declaration changed: it lost its operands."""

    name = "iro.call"
    implementation = prop_def(SymbolRefAttr)
    outputs = var_result_def()


@pytest.mark.parametrize(
    "changed",
    [Definition(ChangedCallOp, "calls"), Definition(CallOp, "calls", form=_replacing_call)],
    ids=["structure", "written-form code"],
)
def test_a_pinned_definition_whose_declaration_changed_without_a_new_version_is_refused(changed) -> None:
    with pytest.raises(Rejection) as raised:
        cold_load(accepted_run(WIDE), _replacing_call(changed))
    assert raised.value.codes == ("iro-run-001",) and "differs" in str(raised.value)


def test_words_that_only_an_expansion_or_a_body_introduces_are_pinned() -> None:
    run = accepted_run(WIDE)
    loaded = cold_load(run)
    pinned = {name for name, _, _ in run.definitions}
    assert {op.name for op in loaded.normalized.strategy.walk()} <= pinned
    assert {op.name for op in loaded.strategy.walk()} <= pinned


@irdl_op_definition
class UnrelatedOp(IRDLOperation):
    """A word some later vocabulary adds."""

    name = "iro.unrelated"


def test_definitions_the_run_does_not_use_may_be_added_or_changed() -> None:
    run = accepted_run(WIDE)
    assert cold_load(run, core().extend([Definition(UnrelatedOp, "later")], name="later")).run is run
    small = accepted_run(ONE_CALL)
    without_filter = Composition([d for d in core().definitions.values() if d.name != "iro.filter"])
    assert cold_load(small, without_filter).run is small


def test_a_run_accepted_for_several_ranks_cold_loads_with_its_rank_obligations() -> None:
    text = (
        "iro.strategy @s uses(@work v1) {\n  iro.sequence @once {\n    iro.call @work() rank_rule(decided_once_and_distributed)\n  }\n}\n"
    )
    run = accepted_run(text, replicated=True)
    assert run.replicated and any(o.evidence.startswith("decided once and distributed") for o in run.obligations)
    assert cold_load(run).verification.obligations == run.obligations
    assert run.identity != accepted_run(text).identity, "the rank setting is part of what the run pins"


def _implementations(*declarations: Declaration):
    installed = {}
    for declared in declarations:
        installed[declared.name] = implementation(
            version=declared.version, inputs=declared.inputs, outputs=declared.outputs, name=declared.name
        )(lambda *args: None)
    return installed


def test_acceptance_with_installed_implementations_pins_their_declarations() -> None:
    installed = _implementations(Declaration("work", "1"))
    run = accepted_run(ONE_CALL, installed=installed)
    assert run.implementations == (("work", "1", installed["work"].declaration.digest),)
    check_implementations(run, installed)
    assert cold_load(run, installed=installed).run is run


@pytest.mark.parametrize(
    "installed",
    [{}, _implementations(Declaration("work", "2")), _implementations(Declaration("work", "1", outputs=("int",)))],
    ids=["missing", "other version", "other declaration"],
)
def test_the_trainer_refuses_implementations_that_do_not_match_the_pins(installed) -> None:
    run = accepted_run(ONE_CALL, installed=_implementations(Declaration("work", "1")))
    with pytest.raises(Rejection) as raised:
        check_implementations(run, installed)
    assert raised.value.codes == ("iro-run-003",)


def test_a_run_accepted_without_installed_implementations_never_passes_the_trainer_check() -> None:
    with pytest.raises(Rejection) as raised:
        check_implementations(accepted_run(ONE_CALL), _implementations(Declaration("work", "1")))
    assert raised.value.codes == ("iro-run-003",)


def test_acceptance_refuses_a_pinned_implementation_that_is_not_installed_at_its_version() -> None:
    with pytest.raises(Rejection) as raised:
        accepted_run(ONE_CALL, installed=_implementations(Declaration("work", "2")))
    assert raised.value.codes == ("iro-run-003",)
