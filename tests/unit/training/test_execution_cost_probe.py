"""Check that the G3.7 comparisons retain equivalent observable work."""

import pytest

import library.training.execution as execution
from library.training.execution import InvalidExecution, OperationFailed, Value
from tests.unit.training.execution_cost_probe import action_pair, current_loop_dispatch, workflow_probe


@pytest.mark.unit
@pytest.mark.training
@pytest.mark.parametrize("count", (1, 4, 16))
def test_direct_reference_preserves_addressed_values_observations_and_inputs(count):
    candidate, direct = action_pair(count)
    for value in (0, 3, 25):
        inputs = {direct.input: value}
        candidate_result = candidate.execute(inputs)
        assert candidate_result == direct.execute(inputs)
        assert candidate_result.optimization[0].value == value + count
        assert candidate_result.optimization[0].unit == "main"
        assert candidate_result.observations[direct.source] == value + count
    for inputs in ({}, {direct.input: 1, Value("extra"): 2}):
        for path in (candidate, direct):
            with pytest.raises(InvalidExecution):
                path.execute(inputs)


@pytest.mark.unit
@pytest.mark.training
def test_both_paths_keep_post_invocation_failure_and_possible_effects():
    effects = []

    def fail(value):
        effects.append(value)
        raise RuntimeError("changed state before failing")

    candidate, direct = action_pair(1, fail)
    for path in (candidate, direct):
        with pytest.raises(InvalidExecution):
            path.execute({})
        assert effects == []
        with pytest.raises(OperationFailed) as failure:
            path.execute({direct.input: 1})
        assert failure.value.operation == "op_0"
        assert isinstance(failure.value.__cause__, RuntimeError)
        assert effects == [1]
        effects.clear()


@pytest.mark.unit
@pytest.mark.training
def test_prepared_candidate_keeps_dynamic_selected_state_without_compiling_again(monkeypatch):
    state = {"calls": 0}

    def adaptive(value):
        state["calls"] += 1
        return value + state["calls"]

    candidate, direct = action_pair(1, adaptive)

    def unexpected_compilation(*_args, **_kwargs):
        raise AssertionError("static construction reached repeated execution")

    monkeypatch.setattr(execution, "_compile_actions", unexpected_compilation)
    monkeypatch.setattr(execution, "_compile_action", unexpected_compilation)
    observed = [candidate.execute({direct.input: 10}).observations[direct.source] for _ in range(3)]
    assert observed == [11, 12, 13]
    assert state == {"calls": 3}


@pytest.mark.unit
@pytest.mark.training
def test_current_loop_dispatch_fixture_runs_the_same_selected_computation():
    _, direct = action_pair(4)
    dispatch, source, source_hash = current_loop_dispatch(direct)
    assert dispatch(7) == direct.execute({direct.input: 7})
    assert "strategies.process_batch" in source
    assert "global_step=trainer.global_step" in source
    assert len(source_hash) == 64


@pytest.mark.unit
@pytest.mark.training
def test_workflow_cost_fixture_correlates_unique_attempts_without_claiming_steps():
    run, tick, lookup, admission, correlation = workflow_probe()
    try:
        reports = [tick()[0] for _ in range(3)]
        assert [report.attempt for report in reports] == [1, 2, 3]
        assert [report.work for report in reports] == [1, 2, 3]
        assert [report.completed_actions for report in reports] == [1, 2, 3]
        assert all(report.outcomes[0].status == "contribution_recorded" for report in reports)
        assert all(report.produced_dependencies == report.requested_dependencies for report in reports)
        assert run.reports == []
        assert lookup()[0].name == "train"
        assert admission() is True
        assert correlation().outcomes == reports[0].outcomes
        assert run.reports == []
    finally:
        run.stop()
