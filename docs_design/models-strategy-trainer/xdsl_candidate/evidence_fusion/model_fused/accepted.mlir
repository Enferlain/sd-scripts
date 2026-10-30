"builtin.module"() ({
  "wf.session"() <{source_id = "model-session", contract = "correlated-model-v1", version = 1 : i64, horizon = 4 : i64, stop_rule = "drain_entered_close_unadmitted"}> ({
    "wf.entity"() <{sym_name = "encoder", source_id = "owner.encoder", kind = "participant"}> : () -> ()
    "wf.entity"() <{sym_name = "learner", source_id = "owner.learner", kind = "participant"}> : () -> ()
    "wf.entity"() <{sym_name = "weights", source_id = "owner.weights", kind = "unit"}> : () -> ()
    "wf.entity"() <{sym_name = "evaluation", source_id = "owner.evaluation", kind = "observation"}> : () -> ()
    %0 = "wf.endpoint"() <{source_id = "raw"}> : () -> !wf.flow<"record">
    %1 = "wf.endpoint"() <{source_id = "selected"}> : () -> !wf.flow<"record">
    %2 = "wf.endpoint"() <{source_id = "left.input"}> : () -> !wf.flow<"record">
    %3 = "wf.endpoint"() <{source_id = "right.input"}> : () -> !wf.flow<"record">
    %4 = "wf.endpoint"() <{source_id = "left.output"}> : () -> !wf.flow<"record">
    %5 = "wf.endpoint"() <{source_id = "right.output"}> : () -> !wf.flow<"record">
    %6 = "wf.endpoint"() <{source_id = "correlated"}> : () -> !wf.flow<"record">
    "model_data.source"(%0) <{source_id = "source", keys = [1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]}> : (!wf.flow<"record">) -> ()
    "model_data.filter"(%0, %1) <{source_id = "filter", excluded = [3 : i64, 6 : i64]}> : (!wf.flow<"record">, !wf.flow<"record">) -> ()
    "model_data.fanout"(%1, %2, %3) <{source_id = "fanout"}> : (!wf.flow<"record">, !wf.flow<"record">, !wf.flow<"record">) -> ()
    "wf.compose"(%2, %4) <{source_id = "data.composed"}> ({
    ^bb0(%7: !wf.flow<"record">):
      %8 = "data.map"(%7) <{source_id = "data.double", implementation = @double_v1, effect = "pure"}> : (!wf.flow<"record">) -> !wf.flow<"record">
      %9 = "wf.local"(%8) <{source_id = "data.nested"}> ({
      ^bb0(%10: !wf.flow<"record">):
        %11 = "data.map"(%10) <{source_id = "data.offset", implementation = @offset_v1, effect = "pure"}> : (!wf.flow<"record">) -> !wf.flow<"record">
        "wf.yield"(%11) : (!wf.flow<"record">) -> ()
      }) : (!wf.flow<"record">) -> !wf.flow<"record">
      "wf.yield"(%9) : (!wf.flow<"record">) -> ()
    }) : (!wf.flow<"record">, !wf.flow<"record">) -> ()
    "model_data.model"() <{source_id = "model.encoder", entity = @encoder, implementation = @scale_v1, initial = 1.500000e+00 : f64, treatment = "frozen"}> : () -> ()
    "model_data.model"() <{source_id = "model.learner", entity = @learner, implementation = @scale_v1, initial = 8.000000e-01 : f64, treatment = "trainable"}> : () -> ()
    "model_data.apply"(%3, %5) <{source_id = "data.encode", model = @encoder, implementation = @encode_v1}> : (!wf.flow<"record">, !wf.flow<"record">) -> ()
    "model_data.correlate"(%4, %5, %6) <{source_id = "admission", membership = "same_sample_and_slot", provenance = "current_source_at_admission"}> : (!wf.flow<"record">, !wf.flow<"record">, !wf.flow<"record">) -> ()
    "model_data.window"(%6) <{source_id = "window", sym_name = "learning", count = 2 : i64}> ({
    ^bb0(%12: !wf.flow<"record">):
      "model_data.contribute"(%12) <{source_id = "objective", model = @learner, unit = @weights, implementation = @loss_v1, normalization = 2 : i64}> : (!wf.flow<"record">) -> ()
      "model_data.advance"() <{source_id = "advance", unit = @weights, learning_rate = 5.000000e-02 : f64, parameter = "weight"}> : () -> ()
    }) : (!wf.flow<"record">) -> ()
    "model_data.evaluate"() <{source_id = "evaluation", owner = @evaluation, model = @learner, trigger = @learning, after_window = 1 : i64, implementation = @measure_v1, consistency = "current_at_start"}> : () -> ()
  }) : () -> ()
}) : () -> ()
