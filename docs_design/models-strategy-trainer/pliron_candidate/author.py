"""Tiny Python author emits Pliron's registered structural syntax, no closures."""

import json


def attributes(**values: str) -> str:
    return "[" + ", ".join(f"{key}: builtin.string {json.dumps(value)}" for key, value in values.items()) + "]"


def build(*, horizon: int = 2) -> str:
    """B's exact case meaning. This is a fixture, not a proposed language API."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("positive integer horizon required")
    session = attributes(
        source_id="data-session", contract="scalar-feedback-v1", version="1", horizon=str(horizon), stop_rule="settle_selected"
    )
    entities = [
        f"{name} = wf.entity () [] {attributes(source_id='owner.' + name, kind=name)} : <() -> (wf.entity)>"
        for name in ("cursor", "policy", "publication")
    ]
    flows = [
        f"{name} = wf.flow () [] {attributes(source_id=name)} : <() -> (wf.{payload})>"
        for name, payload in (("selected", "record"), ("mapped", "record"), ("scored", "score"), ("feedback", "feedback"))
    ]
    select = f"wf.select (cursor, policy, feedback, selected) [] {attributes(source_id='selector', application='ack_before_next_selection')} : <(wf.entity, wf.entity, wf.feedback, wf.record) -> ()>"
    double = f"doubled = wf.map (input) [] {attributes(source_id='data.double', implementation='double_v1', effect='pure')} : <(wf.record) -> (wf.record)>"
    offset = f"offset = wf.map (nested_input) [] {attributes(source_id='data.offset', implementation='offset_v1', effect='pure')} : <(wf.record) -> (wf.record)>"
    local = f"""nested = wf.local (doubled) [] {attributes(source_id="data.nested")} : <(wf.record) -> (wf.record)> {{
      ^nested_block(nested_input: wf.record):
        {offset};
        wf.yield (offset) [] [] : <(wf.record) -> ()>
    }}"""
    compose = f"""wf.compose (selected, mapped) [] {attributes(source_id="data.composed")} : <(wf.record, wf.record) -> ()> {{
      ^local_block(input: wf.record):
        {double};
        {local};
        wf.yield (nested) [] [] : <(wf.record) -> ()>
    }}"""
    score = f"wf.score (mapped, scored) [] {attributes(source_id='scorer', implementation='square_v1', effect='pure')} : <(wf.record, wf.score) -> ()>"
    publish = f"wf.publish (scored, feedback, publication) [] {attributes(source_id='publisher', implementation='receipt_v1', effect='publication', acknowledgement='returned_receipt')} : <(wf.score, wf.feedback, wf.entity) -> ()>"
    # Session graph order is not control order. The local SSA chains are ordered.
    operations = ";\n    ".join([*entities, *flows, select, compose, score, publish])
    return f"wf.session () [] {session} : <() -> ()> {{\n  ^workflow():\n    {operations}\n}}\n"
