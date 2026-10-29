"builtin.module"() ({
  "wf.session"() <{source_id = "grant-session", contract = "granted-model-v1", version = 1 : i64, stop_rule = "drain_entered_preserve_uncertain"}> ({
    "wf.entity"() <{sym_name = "model", source_id = "owner.model", kind = "participant"}> : () -> ()
    "wf.entity"() <{sym_name = "weights", source_id = "owner.weights", kind = "unit"}> : () -> ()
    "model_data.model"() <{source_id = "model.binding", entity = @model, implementation = @scale_v1, initial = 8.000000e-01 : f64, treatment = "trainable"}> : () -> ()
    %0 = "grant.execute"() <{source_id = "selected-two-pass", model = @model, unit = @weights, implementation = @two_pass_v1, profile = "restored-two-pass-v1", delegated = ["backward", "temporary_edit", "intermediate_reset", "restore"], retained = ["clipping", "final_advancement", "final_reset"]}> ({
      "grant.phase"() <{source_id = "phase.0.backward", phase = "backward"}> : () -> ()
      "grant.phase"() <{source_id = "phase.1.temporary_edit", phase = "temporary_edit"}> : () -> ()
      "grant.phase"() <{source_id = "phase.2.intermediate_reset", phase = "intermediate_reset"}> : () -> ()
      "grant.phase"() <{source_id = "phase.3.backward", phase = "backward"}> : () -> ()
      "grant.phase"() <{source_id = "phase.4.restore", phase = "restore"}> : () -> ()
    }) : () -> !grant.contribution<@weights>
    "grant.advance"(%0) <{source_id = "retained-final-step", unit = @weights, parameter = "weight", learning_rate = 5.000000e-02 : f64}> : (!grant.contribution<@weights>) -> ()
  }) : () -> ()
}) : () -> ()
