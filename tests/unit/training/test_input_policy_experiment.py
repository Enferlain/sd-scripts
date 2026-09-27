"""Stateful input selection and packing experiment; not a production input API."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from library.training.execution import Action, Operation, Value, compile_run


@dataclass(frozen=True, slots=True)
class _Example:
    selection_id: int
    source: str
    record: int
    stage: int
    tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Packed:
    examples: tuple[_Example, ...]
    tokens: tuple[int, ...]
    boundaries: tuple[int, ...]
    catalog_revision: int


@dataclass(frozen=True, slots=True)
class _PolicyState:
    catalog_revision: int
    stage: int
    pending_stage: int | None
    schedule_offset: int
    positions: tuple[tuple[str, int], ...]
    buffer: tuple[_Example, ...]
    next_selection_id: int
    ready: frozenset[tuple[str, int]]


class _InputPolicy:
    """One selected policy owns source cursors, a pack buffer, and stage choices."""

    def __init__(self, *, token_budget: int = 5, catalog_revision: int = 1) -> None:
        self.catalog = {
            "a": ((1, 2), (3, 4, 5), (6,)),
            "b": ((10, 11),),
        }
        self.schedules = {0: ("a", "a"), 1: ("b", "a")}
        self.token_budget = token_budget
        self.catalog_revision = catalog_revision
        self.stage = 0
        self.pending_stage: int | None = None
        self.schedule_offset = 0
        self.positions = {"a": 0, "b": 0}
        self.buffer: list[_Example] = []
        self.next_selection_id = 0
        self.ready: set[tuple[str, int]] = {("a", 0), ("a", 1), ("a", 2)}

    def request_stage(self, stage: int) -> None:
        if stage not in self.schedules:
            raise ValueError("stage is outside accepted policy choices")
        if self.pending_stage is not None:
            raise RuntimeError("a stage change is already pending")
        if stage == self.stage:
            return
        if self.buffer:
            self.pending_stage = stage
        else:
            self.stage = stage
            self.schedule_offset = 0

    def make_ready(self, source: str, record: int) -> None:
        self.ready.add((source, record))

    def select(self) -> _Example:
        source = self.schedules[self.stage][self.schedule_offset]
        record = self.positions[source]
        if record >= len(self.catalog[source]):
            raise RuntimeError("selected source is exhausted")
        if (source, record) not in self.ready:
            raise RuntimeError("selected record is not ready; no fallback was authorized")
        example = _Example(self.next_selection_id, source, record, self.stage, self.catalog[source][record])
        if len(example.tokens) > self.token_budget:
            raise ValueError("one example exceeds the accepted token budget")
        self.positions[source] += 1
        self.schedule_offset = (self.schedule_offset + 1) % len(self.schedules[self.stage])
        self.next_selection_id += 1
        self.buffer.append(example)
        return example

    def pack(self) -> _Packed:
        if not self.buffer:
            raise RuntimeError("no selected examples are buffered")
        count = 0
        size = 0
        for example in self.buffer:
            if size + len(example.tokens) > self.token_budget:
                break
            size += len(example.tokens)
            count += 1
        examples = tuple(self.buffer[:count])
        del self.buffer[:count]
        boundaries = [0]
        tokens: list[int] = []
        for example in examples:
            tokens.extend(example.tokens)
            boundaries.append(len(tokens))
        packed = _Packed(examples, tuple(tokens), tuple(boundaries), self.catalog_revision)
        if not self.buffer and self.pending_stage is not None:
            self.stage = self.pending_stage
            self.pending_stage = None
            self.schedule_offset = 0
        return packed

    def state(self) -> _PolicyState:
        return _PolicyState(
            self.catalog_revision,
            self.stage,
            self.pending_stage,
            self.schedule_offset,
            tuple(sorted(self.positions.items())),
            tuple(self.buffer),
            self.next_selection_id,
            frozenset(self.ready),
        )

    @classmethod
    def from_state(cls, state: _PolicyState, *, token_budget: int = 5, catalog_revision: int = 1) -> _InputPolicy:
        policy = cls(token_budget=token_budget, catalog_revision=catalog_revision)
        if state.catalog_revision != catalog_revision:
            raise ValueError("source catalog revision changed")
        if state.stage not in policy.schedules or (state.pending_stage is not None and state.pending_stage not in policy.schedules):
            raise ValueError("saved stage is outside accepted policy choices")
        if state.pending_stage is not None and not state.buffer:
            raise ValueError("pending stage has no buffered old-stage work")
        if len(state.positions) != len(policy.catalog) or set(dict(state.positions)) != set(policy.catalog):
            raise ValueError("saved source positions do not match accepted sources")
        if not 0 <= state.schedule_offset < len(policy.schedules[state.stage]):
            raise ValueError("saved schedule offset is outside the selected stage")
        if any(not 0 <= position <= len(policy.catalog[source]) for source, position in state.positions):
            raise ValueError("saved source position is outside its catalog")
        if any(source not in policy.catalog or not 0 <= record < len(policy.catalog[source]) for source, record in state.ready):
            raise ValueError("saved readiness refers to an unknown record")
        if any(example.stage != state.stage for example in state.buffer):
            raise ValueError("buffer crosses policy stages")
        policy.stage = state.stage
        policy.pending_stage = state.pending_stage
        policy.schedule_offset = state.schedule_offset
        policy.positions = dict(state.positions)
        policy.buffer = list(state.buffer)
        policy.next_selection_id = state.next_selection_id
        policy.ready = set(state.ready)
        return policy


def _admit(packed: _Packed, *, catalog_revision: int) -> _Packed:
    """Selected input contract checks domain boundaries before action delivery."""

    if packed.catalog_revision != catalog_revision:
        raise ValueError("packed input has stale source dependencies")
    if not packed.examples or len({example.selection_id for example in packed.examples}) != len(packed.examples):
        raise ValueError("packed input has invalid logical example identities")
    if len({example.stage for example in packed.examples}) != 1:
        raise ValueError("packed input crosses policy stages")
    expected_boundaries = [0]
    expected_tokens: list[int] = []
    for example in packed.examples:
        expected_tokens.extend(example.tokens)
        expected_boundaries.append(len(expected_tokens))
    if packed.tokens != tuple(expected_tokens) or packed.boundaries != tuple(expected_boundaries):
        raise ValueError("packed input lost example boundaries or contents")
    return packed


def _prepared_consumer(calls: list[tuple[int, ...]]):
    input_value = Value("training_input")
    observed = Value("per_example_values")

    def consume(microbatch: tuple[_Packed, ...]) -> tuple[int, ...]:
        calls.append(tuple(example.selection_id for packed in microbatch for example in packed.examples))
        return tuple(sum(packed.tokens[start:end]) for packed in microbatch for start, end in zip(packed.boundaries, packed.boundaries[1:]))

    action = Action(
        "train",
        inputs=(input_value,),
        operations=(Operation("selected_model_consumer", consume, (input_value,), (observed,)),),
        observations=(observed,),
    )
    prepared = compile_run((action,), due=lambda _: ("train",))
    return prepared.actions["train"], input_value, observed


@pytest.mark.training
@pytest.mark.unit
def test_bounded_policy_change_and_packing_keep_logical_examples_visible():
    policy = _InputPolicy()
    calls: list[tuple[int, ...]] = []
    action, input_value, observed = _prepared_consumer(calls)
    first = policy.select()
    before_same_stage = policy.state()
    policy.request_stage(policy.stage)
    assert policy.state() == before_same_stage
    policy.request_stage(1)
    second = policy.select()  # Already buffered old-stage work drains under stage 0.
    packed_old = policy.pack()
    assert (first.source, second.source) == ("a", "a")
    assert packed_old.boundaries == (0, 2, 5)
    assert len(packed_old.examples) == 2  # One packed sequence is not one logical example.
    assert policy.stage == 1 and policy.pending_stage is None
    assert calls == []  # Selection and packing do not prove model consumption.

    old_result = action.execute({input_value: (_admit(packed_old, catalog_revision=1),)})
    assert old_result.observations[observed] == (3, 12)

    before_unready = policy.state()
    with pytest.raises(RuntimeError, match="not ready; no fallback"):
        policy.select()  # Stage 1 requests source B; silently substituting A changes the policy.
    assert policy.state() == before_unready
    policy.make_ready("b", 0)
    third = policy.select()
    fourth = policy.select()
    packed_new = policy.pack()
    assert (third.source, fourth.source) == ("b", "a")
    assert packed_new.boundaries == (0, 2, 3)
    new_result = action.execute({input_value: (_admit(packed_new, catalog_revision=1),)})
    assert new_result.observations[observed] == (21, 6)
    assert calls == [(first.selection_id, second.selection_id), (third.selection_id, fourth.selection_id)]
    with pytest.raises(ValueError, match="outside accepted"):
        policy.request_stage(2)


@pytest.mark.training
@pytest.mark.unit
def test_pack_buffer_and_pending_stage_restore_independently_of_training_step():
    policy = _InputPolicy(token_budget=4)
    first = policy.select()
    second = policy.select()
    policy.request_stage(1)
    saved = policy.state()
    restored = _InputPolicy.from_state(saved, token_budget=4)
    assert saved.buffer == (first, second)
    assert saved.pending_stage == 1

    first_pack = policy.pack()
    assert first_pack.examples == (first,)
    assert policy.stage == 0 and policy.buffer == [second]
    second_pack = policy.pack()
    assert second_pack.examples == (second,)
    assert policy.stage == 1
    assert (restored.pack(), restored.pack()) == (first_pack, second_pack)
    assert restored.state() == policy.state()

    # One physical delivery can hold two packed sequences and two logical
    # examples. The selected consumer, not the generic action runner, knows
    # what those boundaries mean.
    calls: list[tuple[int, ...]] = []
    action, input_value, observed = _prepared_consumer(calls)
    microbatch = (_admit(first_pack, catalog_revision=1), _admit(second_pack, catalog_revision=1))
    result = action.execute({input_value: microbatch})
    assert result.observations[observed] == (3, 12)
    assert calls == [(first.selection_id, second.selection_id)]

    with pytest.raises(ValueError, match="catalog revision changed"):
        _InputPolicy.from_state(saved, token_budget=4, catalog_revision=2)
    with pytest.raises(ValueError, match="pending stage has no buffered"):
        _InputPolicy.from_state(replace(saved, buffer=()), token_budget=4)
    with pytest.raises(ValueError, match="saved stage is outside accepted"):
        _InputPolicy.from_state(replace(saved, stage=2), token_budget=4)
    with pytest.raises(ValueError, match="source positions do not match"):
        _InputPolicy.from_state(replace(saved, positions=(("a", 2),)), token_budget=4)
    with pytest.raises(ValueError, match="schedule offset is outside"):
        _InputPolicy.from_state(replace(saved, schedule_offset=2), token_budget=4)
    with pytest.raises(ValueError, match="source position is outside"):
        _InputPolicy.from_state(replace(saved, positions=(("a", 4), ("b", 0))), token_budget=4)
    with pytest.raises(ValueError, match="readiness refers to an unknown"):
        _InputPolicy.from_state(replace(saved, ready=frozenset({("b", 1)})), token_budget=4)
    with pytest.raises(ValueError, match="buffer crosses policy stages"):
        _InputPolicy.from_state(replace(saved, buffer=(replace(first, stage=1),)), token_budget=4)


@pytest.mark.training
@pytest.mark.unit
def test_missing_boundaries_or_stale_source_state_fail_before_model_work():
    policy = _InputPolicy()
    policy.select()
    policy.select()
    packed = policy.pack()
    calls: list[tuple[int, ...]] = []
    action, input_value, _ = _prepared_consumer(calls)

    with pytest.raises(ValueError, match="lost example boundaries"):
        _admit(replace(packed, boundaries=(0, len(packed.tokens))), catalog_revision=1)
    with pytest.raises(ValueError, match="lost example boundaries"):
        _admit(replace(packed, tokens=tuple(reversed(packed.tokens))), catalog_revision=1)
    with pytest.raises(ValueError, match="stale source dependencies"):
        _admit(packed, catalog_revision=2)
    with pytest.raises(ValueError, match="invalid logical example identities"):
        _admit(replace(packed, examples=()), catalog_revision=1)
    with pytest.raises(ValueError, match="invalid logical example identities"):
        _admit(replace(packed, examples=(packed.examples[0], packed.examples[0])), catalog_revision=1)
    with pytest.raises(ValueError, match="crosses policy stages"):
        _admit(replace(packed, examples=(packed.examples[0], replace(packed.examples[1], stage=1))), catalog_revision=1)
    assert calls == []

    # The runner accepts the domain payload as one changing value; its generic
    # Value address deliberately contains no token, image, or pack fields.
    action.execute({input_value: (_admit(packed, catalog_revision=1),)})
    assert calls == [(packed.examples[0].selection_id, packed.examples[1].selection_id)]
