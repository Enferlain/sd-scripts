"""The accepted run: the evidence of its code, its authority seed, and cold loading that refuses parts that do not correspond."""

from dataclasses import replace
from pathlib import Path

import pytest

from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, prop_def, var_result_def

from library.iro.accepted import DeclaredParticipant, DeclaredRelationship, check_implementations, cold_load, implementation_evidence
from library.iro.composition import Composition, Definition, core
from library.iro.diagnostics import Rejection
from library.iro.evidence import CodeEvidence, source_evidence
from library.iro.frontend import implementation
from library.iro.seams import Declaration
from library.iro.verify import Obligations
from library.iro.words.calls import CallOp
from tests.unit.iro.accepted_runs import accepted_run

WIDE = (Path(__file__).parent / "wide.iro").read_text()
ONE_CALL = "iro.strategy @s uses(@work) {\n  iro.sequence @once {\n    iro.call @work()\n  }\n}\n"


def test_an_accepted_run_records_the_definitions_it_uses_and_its_implementations() -> None:
    run = accepted_run(WIDE)
    recorded = {record.name: record for record in run.definitions}
    assert recorded["iro.call"] == core()["iro.call"].evidence, "a definition is recorded by its declaration"
    assert recorded["iro.call"].declaration.startswith("iro.call\n") and recorded["iro.call"].source == source_evidence(CallOp)
    assert "iro.access" in recorded and "iro.lease" in recorded, "words and the types they carry are recorded"
    small = {record.name for record in accepted_run(ONE_CALL).definitions}
    assert small == {"iro.strategy", "iro.sequence", "iro.call", "iro.yield"}, "a definition the strategy does not use is not recorded"
    assert run.implementations[0] == CodeEvidence("load_sdxl"), "accepted without installed code, the run holds only the names"
    assert run.written == WIDE, "a strategy read from a written form without locations prints back the same"
    assert run.normalized.startswith("iro.strategy @wide normalized")
    assert run.identity == accepted_run(WIDE).identity, "two runs of one strategy share its content identity"


def test_the_authority_seed_declares_every_participant_and_relationship_unbound() -> None:
    seed = accepted_run(WIDE).seed
    assert [p.name for p in seed.participants] == ["vae", "te1", "te2", "unet", "lora"]
    assert DeclaredParticipant("lora", "participant", "build_lora", (("unet", "inspection"),)) in seed.participants
    assert seed.relationships == (DeclaredRelationship("lora_on_unet", "attachment", "lora", "unet"),)
    assert seed.obligations == accepted_run(WIDE).obligations.revision


def test_the_seed_carries_each_materializer_s_settings_as_values() -> None:
    text = (
        "iro.strategy @s uses(@load_unet) {\n"
        '  %unet = iro.resource @unet : participant materialize(@load_unet, "models/sdxl.safetensors", 2, false)\n'
        "}\n"
    )
    run = accepted_run(text)
    (unet,) = run.seed.participants
    assert unet.settings == ("models/sdxl.safetensors", 2, False)
    assert cold_load(run).run is run, "the settings are derived from the written form like the rest of the seed"


def test_locations_survive_into_the_written_form_and_the_seed() -> None:
    located = WIDE.replace(
        "  %vae  = iro.resource @vae : participant materialize(@load_sdxl)",
        '  %vae  = iro.resource @vae : participant materialize(@load_sdxl) loc("strategies/sdxl/finetune.py":12)',
    )
    run = accepted_run(located)
    assert 'loc("strategies/sdxl/finetune.py":12)' in run.written
    assert run.seed.participants[0].origin == "strategies/sdxl/finetune.py:12"
    assert cold_load(run).run is run


def test_a_strategy_edited_only_where_it_was_authored_keeps_its_identity() -> None:
    def authored_at(line: int) -> str:
        return WIDE.replace(
            "  %vae  = iro.resource @vae : participant materialize(@load_sdxl)",
            f'  %vae  = iro.resource @vae : participant materialize(@load_sdxl) loc("strategies/sdxl/finetune.py":{line})',
        )

    here, moved, bare = accepted_run(authored_at(12)), accepted_run(authored_at(14)), accepted_run(WIDE)
    assert here.written != moved.written, "the written form keeps where each statement was authored"
    assert here.seed.strategy == moved.seed.strategy == bare.seed.strategy, "the strategy's revision leaves locations out"
    assert here.identity == moved.identity == bare.identity
    assert cold_load(moved).run is moved
    te1 = "  %te1  = iro.resource @te1 : participant materialize(@load_sdxl)\n"
    te2 = "  %te2  = iro.resource @te2 : participant materialize(@load_sdxl)\n"
    assert te1 + te2 in WIDE
    reordered = accepted_run(WIDE.replace(te1 + te2, te2 + te1))
    assert reordered.identity != bare.identity, "an operation moved relative to the others is a change in structure, not in layout"
    changed = accepted_run(WIDE.replace("bounds(outstanding 4, ready 2)", "bounds(outstanding 5, ready 2)"))
    assert changed.seed.strategy != bare.seed.strategy and changed.identity != bare.identity, "a change in meaning changes both"


def test_a_saved_run_cold_loads_to_the_same_obligations() -> None:
    run = accepted_run(WIDE)
    loaded = cold_load(run)
    assert loaded.verification.obligations == run.obligations
    assert loaded.normalized.text() == run.normalized


@pytest.mark.parametrize(
    "tamper",
    [
        lambda run: replace(run, definitions=run.definitions[1:]),
        lambda run: replace(run, implementations=(CodeEvidence("unlisted"), *run.implementations[1:])),
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
def test_a_recorded_definition_whose_declaration_changed_is_refused(changed) -> None:
    with pytest.raises(Rejection) as raised:
        cold_load(accepted_run(WIDE), _replacing_call(changed))
    assert raised.value.codes == ("iro-run-001",) and "changed since the run" in str(raised.value)


class VerifiedOtherwise(CallOp):
    """The call word with the same declaration and another verifier: other code."""

    def verify_(self) -> None:
        return None


def test_a_definition_with_the_same_declaration_and_other_code_is_refused() -> None:
    assert Definition(VerifiedOtherwise, "calls").text == core()["iro.call"].text, "the declarations agree"
    with pytest.raises(Rejection) as raised:
        cold_load(accepted_run(WIDE), _replacing_call(Definition(VerifiedOtherwise, "calls")))
    assert raised.value.codes == ("iro-run-001",) and "its source" in str(raised.value)


def test_words_that_only_an_expansion_or_a_body_introduces_are_recorded() -> None:
    run = accepted_run(WIDE)
    loaded = cold_load(run)
    recorded = {record.name for record in run.definitions}
    assert {op.name for op in loaded.normalized.strategy.walk()} <= recorded
    assert {op.name for op in loaded.strategy.walk()} <= recorded


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
    text = "iro.strategy @s uses(@work) {\n  iro.sequence @once {\n    iro.call @work() rank_rule(decided_once_and_distributed)\n  }\n}\n"
    run = accepted_run(text, replicated=True)
    assert run.replicated and any(o.evidence.startswith("decided once and distributed") for o in run.obligations)
    assert cold_load(run).verification.obligations == run.obligations
    assert run.identity != accepted_run(text).identity, "the rank setting is part of the run's identity"


def _implementations(*declarations: Declaration):
    installed = {}
    for declared in declarations:
        installed[declared.name] = implementation(inputs=declared.inputs, outputs=declared.outputs, name=declared.name)(lambda *args: None)
    return installed


def test_acceptance_with_installed_implementations_records_their_declarations_and_sources() -> None:
    installed = _implementations(Declaration("work"))
    run = accepted_run(ONE_CALL, installed=installed)
    (record,) = run.implementations
    assert record == implementation_evidence("work", installed["work"])
    assert record.declaration == installed["work"].declaration.text
    assert record.source is not None and record.source.module == __name__, "the source is where the function is defined"
    assert "@work" in record.printout() and "declares name work" in record.printout(), "a record prints in full"
    check_implementations(run, installed)
    assert cold_load(run, installed=installed).run is run


def _other_work(*args: object) -> None:
    """The same declaration, written somewhere else: other code."""


@pytest.mark.parametrize(
    ("installed", "changed"),
    [
        ({}, "not installed"),
        ({"work": implementation(name="work")(_other_work)}, "its source"),
        (_implementations(Declaration("work", outputs=("int",))), "what it takes or returns"),
    ],
    ids=["missing", "same declaration, other code", "other declaration"],
)
def test_the_trainer_refuses_implementations_that_do_not_give_the_recorded_evidence(installed, changed) -> None:
    run = accepted_run(ONE_CALL, installed=_implementations(Declaration("work")))
    with pytest.raises(Rejection) as raised:
        check_implementations(run, installed)
    assert raised.value.codes == ("iro-run-003",) and changed in str(raised.value)


def test_a_run_accepted_without_installed_implementations_never_passes_the_trainer_check() -> None:
    with pytest.raises(Rejection) as raised:
        check_implementations(accepted_run(ONE_CALL), _implementations(Declaration("work")))
    assert raised.value.codes == ("iro-run-003",)


def test_a_bare_compiled_function_is_refused_as_an_implementation() -> None:
    compiled = {"work": implementation(name="work")(len)}
    with pytest.raises(Rejection) as raised:
        accepted_run(ONE_CALL, installed=compiled)
    assert raised.value.codes == ("iro-run-004",) and "compiled code without source" in str(raised.value)
    run = accepted_run(ONE_CALL, installed=_implementations(Declaration("work")))
    with pytest.raises(Rejection) as raised:
        check_implementations(run, compiled)
    assert raised.value.codes == ("iro-run-004",), "the host check refuses it too"


def test_an_implementation_that_captures_mutable_state_is_refused() -> None:
    state: list[int] = []

    @implementation(name="work")
    def work() -> None:
        state.append(1)

    with pytest.raises(Rejection) as raised:
        accepted_run(ONE_CALL, installed={"work": work})
    assert raised.value.codes == ("iro-run-004",) and "closes over state" in str(raised.value)


class Scale:
    def __init__(self, factor: int) -> None:
        self.factor = factor

    def __call__(self) -> int:
        return self.factor

    def apply(self) -> int:
        return self.factor


@pytest.mark.parametrize("code", [Scale(2), Scale(2).apply], ids=["callable object", "bound method"])
def test_an_implementation_with_state_its_source_does_not_show_is_refused(code) -> None:
    with pytest.raises(Rejection) as raised:
        accepted_run(ONE_CALL, installed={"work": implementation(name="work")(code)})
    assert raised.value.codes == ("iro-run-004",) and "whose state the system cannot see" in str(raised.value).replace("\n  ", " ")


def test_acceptance_refuses_an_implementation_the_strategy_uses_that_is_not_installed() -> None:
    with pytest.raises(Rejection) as raised:
        accepted_run(ONE_CALL, installed={})
    assert raised.value.codes == ("iro-run-003",)
