"builtin.module"() ({
  "exec.plan"() <{accepted_digest = "cc20b727c8e338fbe3d70a86952c794ddfdcd7a56d0fa3ad6fdec3ba39ccba26", target = "continuous-owner-cpu-v1", contract = "continuous-learning-feedback-v1", horizon = 3 : i64, capacity = 1 : i64, stop_rule = "external_at_checkpoint; close_selection; discard_unclaimed; drain_entered"}> ({
    "model_data.recipe"() <{origin = "mixed-session", role = "mixed-owner", protocol = "open producer; owner feedback receiver; ordered bounded ports; same-binding production snapshots; external checkpoint"}> ({
      "wf.session"() <{source_id = "mixed-session", contract = "continuous-learning-feedback-v1", version = 1 : i64, horizon = 3 : i64, stop_rule = "external_at_checkpoint; close_selection; discard_unclaimed; drain_entered"}> ({
        "wf.entity"() <{sym_name = "learner", source_id = "owner.learner", kind = "participant"}> : () -> ()
        "wf.entity"() <{sym_name = "weights", source_id = "owner.weights", kind = "unit"}> : () -> ()
        %0 = "wf.endpoint"() <{source_id = "raw"}> : () -> !wf.flow<"record">
        %1 = "wf.endpoint"() <{source_id = "selected"}> : () -> !wf.flow<"record">
        %2 = "wf.endpoint"() <{source_id = "left.input"}> : () -> !wf.flow<"record">
        %3 = "wf.endpoint"() <{source_id = "right.input"}> : () -> !wf.flow<"record">
        %4 = "wf.endpoint"() <{source_id = "left.output"}> : () -> !wf.flow<"record">
        %5 = "wf.endpoint"() <{source_id = "right.output"}> : () -> !wf.flow<"record">
        %6 = "wf.endpoint"() <{source_id = "correlated"}> : () -> !wf.flow<"record">
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
        "model_data.model"() <{source_id = "model.learner", entity = @learner, implementation = @scale_v1, initial = 8.000000e-01 : f64, treatment = "trainable"}> : () -> ()
        "model_data.apply"(%3, %5) <{source_id = "data.encode", model = @learner, implementation = @encode_v1}> : (!wf.flow<"record">, !wf.flow<"record">) -> ()
        "model_data.correlate"(%4, %5, %6) <{source_id = "admission", membership = "same_ticket_and_ordinal", provenance = "same_binding_protected_at_production"}> : (!wf.flow<"record">, !wf.flow<"record">, !wf.flow<"record">) -> ()
        "model_data.window"(%6) <{source_id = "window", sym_name = "learning", count = 1 : i64}> ({
        ^bb0(%12: !wf.flow<"record">):
          "model_data.contribute"(%12) <{source_id = "objective", model = @learner, unit = @weights, implementation = @mixed_loss_v1, normalization = 1 : i64}> : (!wf.flow<"record">) -> ()
          "model_data.advance"() <{source_id = "advance", unit = @weights, learning_rate = 1.000000e-03 : f64, parameter = "weight"}> : () -> ()
          "mixed.learned"(%13) <{source_id = "learned.observe", model = @learner, observation = "returned_advance_current_weight_and_input_associations"}> : (!wf.flow<"feedback">) -> ()
        }) : (!wf.flow<"record">) -> ()
        "wf.entity"() <{source_id = "owner.cursor", sym_name = "cursor", kind = "cursor"}> : () -> ()
        "wf.entity"() <{source_id = "owner.policy", sym_name = "policy", kind = "policy"}> : () -> ()
        %13 = "wf.endpoint"() <{source_id = "learned"}> : () -> !wf.flow<"feedback">
        %14 = "wf.endpoint"() <{source_id = "acknowledged"}> : () -> !wf.flow<"feedback">
        "data.select"(%14, %0) <{source_id = "select", cursor = @cursor, policy = @policy, application = "ack_before_next_selection"}> : (!wf.flow<"feedback">, !wf.flow<"record">) -> ()
        "mixed.dynamic_filter"(%0, %1) <{source_id = "filter", policy = @policy, implementation = @dynamic_filter_v1}> : (!wf.flow<"record">, !wf.flow<"record">) -> ()
        "mixed.policy_update"(%13, %14) <{source_id = "policy.update", policy = @policy, implementation = @learned_bias_v1, protocol = "post_advance_weight; bias=int(abs(weight)*10)%5; revision+=1; retire_once"}> : (!wf.flow<"feedback">, !wf.flow<"feedback">) -> ()
        "mixed.stop_checkpoint"() <{source_id = "stop.checkpoint", activity = @learning, after_windows = 3 : i64, protocol = "external_at_checkpoint; close_selection; discard_unclaimed; drain_entered"}> : () -> ()
      }) : () -> ()
    }) : () -> ()
    "exec.call"() <{origin = "model.learner", ancestors = ["mixed-session"], implementation = @scale_v1, binding_version = "1", role = "model", effect = "load", completion = "protected_physical_return_then_validate"}> : () -> ()
    "exec.call"() <{origin = "data.double", ancestors = ["mixed-session", "data.composed"], implementation = @double_v1, binding_version = "1", role = "map", effect = "pure", completion = "protected_physical_return_then_validate"}> : () -> ()
    "exec.call"() <{origin = "data.offset", ancestors = ["mixed-session", "data.composed", "data.nested"], implementation = @offset_v1, binding_version = "1", role = "map", effect = "pure", completion = "protected_physical_return_then_validate"}> : () -> ()
    "exec.call"() <{origin = "data.encode", ancestors = ["mixed-session"], implementation = @encode_v1, binding_version = "1", role = "encode", effect = "read_source", completion = "protected_physical_return_then_validate"}> : () -> ()
    "exec.call"() <{origin = "objective", ancestors = ["mixed-session", "window"], implementation = @mixed_loss_v1, binding_version = "1", role = "objective", effect = "build_loss", completion = "protected_physical_return_then_validate"}> : () -> ()
    "exec.call"() <{origin = "filter", ancestors = ["mixed-session"], implementation = @dynamic_filter_v1, binding_version = "1", role = "filter", effect = "read_policy", completion = "protected_physical_return_then_validate"}> : () -> ()
    "exec.call"() <{origin = "policy.update", ancestors = ["mixed-session"], implementation = @learned_bias_v1, binding_version = "1", role = "policy", effect = "derive_update", completion = "protected_physical_return_then_validate"}> : () -> ()
  }) : () -> ()
}) : () -> ()
