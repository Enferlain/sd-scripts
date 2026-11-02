"""Mixed graph/SSA authoring fixture; native verification owns its contracts."""

from __future__ import annotations

from .author import attributes


def build(*, horizon: int = 3, count: int = 1, after_windows: int = 3, capacity: int = 1, prefix: str = "") -> str:
    def attrs(source, **values):
        return attributes(source_id=prefix + source, **values)

    stop = "external_at_checkpoint; close_selection; discard_unclaimed; drain_entered"
    policy = "post_advance_weight; bias=int(abs(weight)*10)%5; revision+=1; retire_once"
    return f"""mixed.session () [] {attrs("mixed-session", version="1", horizon=str(horizon), stop_rule=stop)} : <() -> ()> {{
^session():
  learner = ac.model () [] {attrs("model.learner", participant="learner", implementation="scale_v1", initial="0.8", treatment="trainable")} : <() -> (ac.model)>;
  unit = ac.unit (learner) [] {attrs("owner.weights", unit="weights", parameter="weight")} : <(ac.model) -> (ac.unit)>;
  records, drained = mixed.producer (learner, feedback, stop) [] {attrs("producer", activity="mixed-data", capacity=str(capacity), publication="ordered_bounded_ports_after_model_release", settlement="close_selection_discard_unclaimed_join_private_physical_work")} : <(ac.model, mixed.feedback, mixed.stop) -> (ac.correlated, ac.done)> {{
  ^producer(subject: ac.model, learned_input: mixed.feedback, closing: mixed.stop):
    cursor = mixed.cursor () [] {attrs("owner.cursor", owner="cursor", start="0")} : <() -> (mixed.cursor)>;
    policy = mixed.policy () [] {attrs("owner.policy", owner="policy", bias="0", revision="0")} : <() -> (mixed.policy)>;
    raw = mixed.select (cursor, policy, acknowledged, closing) [] {attrs("select", application="ack_before_next_selection")} : <(mixed.cursor, mixed.policy, mixed.ack, mixed.stop) -> (ac.stream)>;
    selected = mixed.filter (raw, policy) [] {attrs("filter", implementation="dynamic_filter_v1")} : <(ac.stream, mixed.policy) -> (ac.stream)>;
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
    right = ac.encode (right_in, subject) [] {attrs("data.encode", implementation="encode_v1", protection="same_binding_read_through_physical_completion")} : <(ac.stream, ac.model) -> (ac.stream)>;
    correlated = ac.correlate (left, right, subject) [] {attrs("admission", membership="same_ticket_and_ordinal", provenance="same_binding_protected_at_production")} : <(ac.stream, ac.stream, ac.model) -> (ac.correlated)>;
    acknowledged = mixed.update (learned_input, policy) [] {attrs("policy.update", implementation="learned_bias_v1", protocol=policy)} : <(mixed.feedback, mixed.policy) -> (mixed.ack)>;
    mixed.publish (correlated) [] {attrs("producer.output")} : <(ac.correlated) -> ()>
  }};
  feedback, learned = mixed.repeat (records, learner, unit) [] {attrs("window", activity="learning", count=str(count), accumulation="sum")} : <(ac.correlated, ac.model, ac.unit) -> (mixed.feedback, ac.done)> {{
  ^window(input: ac.correlated, subject: ac.model, weights: ac.unit):
    gradient = ac.contribute (input, subject, weights) [] {attrs("objective", implementation="mixed_loss_v1", normalization=str(count))} : <(ac.correlated, ac.model, ac.unit) -> (ac.gradient)>;
    advanced = ac.advance (gradient, weights) [] {attrs("advance", learning_rate="0.001", authority="retained_owner_only_after_all_contributions")} : <(ac.gradient, ac.unit) -> (ac.done)>;
    observation = mixed.learned (advanced, input, subject) [] {attrs("learned.observe", observation="returned_advance_current_weight_and_input_associations")} : <(ac.done, ac.correlated, ac.model) -> (mixed.feedback)>;
    mixed.window_end (observation) [] {attrs("window.end")} : <(mixed.feedback) -> ()>
  }};
  stop, settled = mixed.checkpoint (learned) [] {attrs("stop.checkpoint", after_windows=str(after_windows), protocol=stop)} : <(ac.done) -> (mixed.stop, ac.done)>;
  mixed.finish (settled, drained) [] {attrs("run.finish")} : <(ac.done, ac.done) -> ()>
}}
"""
