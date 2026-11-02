"""Small A authoring adapter; native contracts own acceptance and preparation."""

from __future__ import annotations

from .author import attributes


def build(*, count: int = 2, horizon: int = 4, prefix: str = "") -> str:
    def attrs(source, **values):
        return attributes(source_id=prefix + source, **values)

    return f"""ac.session () [] {attrs("model-session", version="1", horizon=str(horizon), stop_rule="drain_entered_close_unadmitted")} : <() -> ()> {{
^session():
  encoder = ac.model () [] {attrs("model.encoder", participant="encoder", implementation="scale_v1", initial="1.5", treatment="frozen")} : <() -> (ac.model)>;
  learner = ac.model () [] {attrs("model.learner", participant="learner", implementation="scale_v1", initial="0.8", treatment="trainable")} : <() -> (ac.model)>;
  unit = ac.unit (learner) [] {attrs("owner.weights", unit="weights", parameter="weight")} : <(ac.model) -> (ac.unit)>;
  raw = ac.source () [] {attrs("source", keys="[1,2,3,4,5,6]")} : <() -> (ac.stream)>;
  selected = ac.filter (raw) [] {attrs("filter", excluded="[3,6]")} : <(ac.stream) -> (ac.stream)>;
  left_in, right_in = ac.fanout (selected) [] {attrs("fanout")} : <(ac.stream) -> (ac.stream, ac.stream)>;
  left = ac.compose (left_in) [] {attrs("data.composed")} : <(ac.stream) -> (ac.stream)> {{
  ^compose(record: wf.record):
    doubled = wf.map (record) [] {attrs("data.double", implementation="double_v1", effect="pure")} : <(wf.record) -> (wf.record)>;
    mapped = wf.local (doubled) [] {attrs("data.nested")} : <(wf.record) -> (wf.record)> {{
    ^nested(value: wf.record):
      offset = wf.map (value) [] {attrs("data.offset", implementation="offset_v1", effect="pure")} : <(wf.record) -> (wf.record)>;
      wf.yield (offset) [] {attrs("data.nested.yield")} : <(wf.record) -> ()>
    }};
    wf.yield (mapped) [] {attrs("data.composed.yield")} : <(wf.record) -> ()>
  }};
  right = ac.encode (right_in, encoder) [] {attrs("data.encode", implementation="encode_v1", protection="frozen_source_through_physical_completion")} : <(ac.stream, ac.model) -> (ac.stream)>;
  correlated = ac.correlate (left, right, encoder) [] {attrs("admission", membership="same_sample_and_slot_before_any_claim", provenance="current_source_at_admission")} : <(ac.stream, ac.stream, ac.model) -> (ac.correlated)>;
  events, learned = ac.window (correlated, learner, unit) [] {attrs("window", activity="learning", count=str(count), accumulation="sum")} : <(ac.correlated, ac.model, ac.unit) -> (ac.events, ac.done)> {{
  ^window(input: ac.correlated, subject: ac.model, weights: ac.unit):
    gradient = ac.contribute (input, subject, weights) [] {attrs("objective", implementation="loss_v1", normalization=str(count))} : <(ac.correlated, ac.model, ac.unit) -> (ac.gradient)>;
    advanced = ac.advance (gradient, weights) [] {attrs("advance", learning_rate="0.05", authority="retained_owner_only_after_all_contributions")} : <(ac.gradient, ac.unit) -> (ac.done)>;
    ac.window_end (advanced) [] {attrs("window.end")} : <(ac.done) -> ()>
  }};
  evaluated = ac.evaluate (events, learner) [] {attrs("evaluation", owner="evaluation", implementation="measure_v1", after_window="1", consistency="current_at_start", protection="current_read_through_physical_return_validation_and_mode_restore")} : <(ac.events, ac.model) -> (ac.done)>;
  ac.finish (learned, evaluated) [] {attrs("run.finish")} : <(ac.done, ac.done) -> ()>
}}
"""
