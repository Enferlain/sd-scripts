"""Fixture authoring for the bounded composition case, without live implementations."""

from .author import attributes


def build(*, windows: int = 2, horizon: int = 3, after_window: int = 2, prefix: str = "") -> str:
    for value in (windows, horizon, after_window):
        if type(value) is not int or value < 1:
            raise ValueError("positive integer case bounds required")

    def a(source: str, **values: str) -> str:
        return attributes(source_id=prefix + source, **values)

    encoder = f"enc = lc.model () [] {a('encoder', implementation='scale_v1', initial='2.0', treatment='frozen')} : <() -> (lc.model)>"
    learner = f"learn = lc.model () [] {a('learner', implementation='scale_v1', initial='0.8', treatment='trainable')} : <() -> (lc.model)>"
    producer = f"""records, producer_done = lc.producer (enc, group_done) [] {a("producer", implementation="encode_v1", horizon=str(horizon), ready="2", workers="1", remaining="discard_unclaimed", failure="close_then_drain")} : <(lc.model, lc.done) -> (lc.stream, lc.done)>"""
    window = []
    for slot in range(2):
        take = f"r{slot} = lc.take (window_stream) [] {a(f'training.take.{slot}', selection=f'slot:{slot}', provenance='actual_frozen_source')} : <(lc.stream) -> (lc.sample)>"
        operands = f"r{slot}, window_model" + (", g0" if slot else "")
        inputs = "lc.sample, lc.model" + (", lc.gradient" if slot else "")
        contribution = f"g{slot} = lc.contribute ({operands}) [] {a(f'training.objective.{slot}', implementation='objective_v1', normalization='2')} : <({inputs}) -> (lc.gradient)>"
        window.extend((take, contribution))
    window.extend(
        (
            f"advanced = lc.advance (g1, window_model) [] {a('training.advance', parameter='weight', learning_rate='0.01', authority='ordinary_owner')} : <(lc.gradient, lc.model) -> (lc.completion)>",
            f"lc.window_end (advanced) [] {a('training.window_end')} : <(lc.completion) -> ()>",
        )
    )
    repeat = f"""window_events, training_done = lc.repeat (records, learn) [] {a("training", max_windows=str(windows), admission="policy_choice_before_first_claim", stop="finish_selected_window")} : <(lc.stream, lc.model) -> (lc.events, lc.done)> {{
      ^window(window_stream: lc.stream, window_model: lc.model):
        {"; ".join(window)}
    }}"""
    request = f"due = lc.request (window_events) [] {a('request', after_window=str(after_window), multiplicity='once_per_matching_completion', pending='keeps_group_alive', max_pending='1', cancellation='cancel_or_unfulfilled')} : <(lc.events) -> (lc.request)>"
    evaluation = f"""evaluation_done = lc.evaluator (due, records, learn) [] {a("evaluation", completion="physical_and_cleanup")} : <(lc.request, lc.stream, lc.model) -> (lc.done)> {{
      ^measurement(cause: lc.request, evaluation_stream: lc.stream, evaluation_model: lc.model):
        sample = lc.take (evaluation_stream) [] {a("evaluation.take", selection="after_trigger_members", provenance="actual_frozen_source")} : <(lc.stream) -> (lc.sample)>;
        result = lc.read (sample, evaluation_model) [] {a("evaluation.read", implementation="measure_v1", state="current_at_start", mode="evaluation", completion="physical_validate_restore")} : <(lc.sample, lc.model) -> (lc.measurement)>;
        lc.evaluation_end (result) [] {a("evaluation.end")} : <(lc.measurement) -> ()>
    }}"""
    group = f"""group_done = lc.group () [] {a("consumers", completion="children_physically_settled", pending="include_requested", failure="cancel_siblings_and_drain")} : <() -> (lc.done)> {{
      ^consumers():
        {repeat};
        {request};
        {evaluation};
        lc.settle (training_done, evaluation_done) [] {a("consumers.settle")} : <(lc.done, lc.done) -> ()>
    }}"""
    finish = f"lc.finish (group_done, producer_done) [] {a('run.finish')} : <(lc.done, lc.done) -> ()>"
    return f"""lc.run () [] {a("run", version="1", failure="cancel_siblings_and_drain")} : <() -> ()> {{
      ^activities():
        {"; ".join((encoder, learner, producer, group, finish))}
    }}\n"""
