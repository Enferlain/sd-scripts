"builtin.module"() ({
  "life_exec.plan"() <{schema = "1", target = "existing-owner-composition-cpu-v1", manifest = "{\22bindings\22:{\22encode_v1\22:{\22completion\22:\22physical_return_before_release\22,\22effect\22:\22read_source\22,\22role\22:\22produce\22,\22version\22:\221\22},\22measure_v1\22:{\22completion\22:\22physical_validate_restore\22,\22effect\22:\22protected_read\22,\22role\22:\22evaluate\22,\22version\22:\221\22},\22objective_v1\22:{\22completion\22:\22return_before_backward\22,\22effect\22:\22build_loss\22,\22role\22:\22objective\22,\22version\22:\221\22},\22scale_v1\22:{\22completion\22:\22before_install\22,\22effect\22:\22load\22,\22role\22:\22model\22,\22version\22:\221\22}},\22schema\22:1,\22target\22:\22existing-owner-composition-cpu-v1\22}"}> ({
    "life.run"() <{source_id = "run", policy = {version = "1", failure = "cancel_siblings_and_drain"}}> ({
      %0 = "life.model"() <{source_id = "encoder", policy = {implementation = "scale_v1", initial = "2.0", treatment = "frozen"}}> : () -> !life.model
      %1 = "life.model"() <{source_id = "learner", policy = {implementation = "scale_v1", initial = "0.8", treatment = "trainable"}}> : () -> !life.model
      %2, %3 = "life.producer"(%0, %4) <{source_id = "producer", policy = {implementation = "encode_v1", horizon = "3", ready = "2", workers = "1", remaining = "discard_unclaimed", failure = "close_then_drain"}}> : (!life.model, !life.done) -> (!life.stream, !life.done)
      %4 = "life.group"() <{source_id = "consumers", policy = {completion = "children_physically_settled", pending = "include_requested", failure = "cancel_siblings_and_drain"}}> ({
        %5, %6 = "life.repeat"(%2, %1) <{source_id = "training", policy = {max_windows = "2", admission = "policy_choice_before_first_claim", stop = "finish_selected_window"}}> ({
        ^bb0(%7: !life.stream, %8: !life.model):
          %9 = "life.take"(%7) <{source_id = "training.take.0", policy = {selection = "slot:0", provenance = "actual_frozen_source"}}> : (!life.stream) -> !life.sample
          %10 = "life.contribute"(%9, %8) <{source_id = "training.objective.0", policy = {implementation = "objective_v1", normalization = "2"}}> : (!life.sample, !life.model) -> !life.gradient
          %11 = "life.take"(%7) <{source_id = "training.take.1", policy = {selection = "slot:1", provenance = "actual_frozen_source"}}> : (!life.stream) -> !life.sample
          %12 = "life.contribute"(%11, %8, %10) <{source_id = "training.objective.1", policy = {implementation = "objective_v1", normalization = "2"}}> : (!life.sample, !life.model, !life.gradient) -> !life.gradient
          %13 = "life.advance"(%12, %8) <{source_id = "training.advance", policy = {parameter = "weight", learning_rate = "0.01", authority = "ordinary_owner"}}> : (!life.gradient, !life.model) -> !life.completion
          "life.window_end"(%13) <{source_id = "training.window_end", policy = {}}> : (!life.completion) -> ()
        }) : (!life.stream, !life.model) -> (!life.events, !life.done)
        %7 = "life.request"(%5) <{source_id = "request", policy = {after_window = "2", multiplicity = "once_per_matching_completion", pending = "keeps_group_alive", max_pending = "1", cancellation = "cancel_or_unfulfilled"}}> : (!life.events) -> !life.request
        %8 = "life.evaluator"(%7, %2, %1) <{source_id = "evaluation", policy = {completion = "physical_and_cleanup"}}> ({
        ^bb0(%9: !life.request, %10: !life.stream, %11: !life.model):
          %12 = "life.take"(%10) <{source_id = "evaluation.take", policy = {selection = "after_trigger_members", provenance = "actual_frozen_source"}}> : (!life.stream) -> !life.sample
          %13 = "life.read"(%12, %11) <{source_id = "evaluation.read", policy = {implementation = "measure_v1", state = "current_at_start", mode = "evaluation", completion = "physical_validate_restore"}}> : (!life.sample, !life.model) -> !life.measurement
          "life.evaluation_end"(%13) <{source_id = "evaluation.end", policy = {}}> : (!life.measurement) -> ()
        }) : (!life.request, !life.stream, !life.model) -> !life.done
        "life.settle"(%6, %8) <{source_id = "consumers.settle", policy = {}}> : (!life.done, !life.done) -> ()
      }) : () -> !life.done
      "life.finish"(%4, %3) <{source_id = "run.finish", policy = {}}> : (!life.done, !life.done) -> ()
    }) : () -> ()
  }, {
    "life_exec.declaration"() <{role = "scope", values = {origin = "run", parent = "", kind = "life.run", version = "1", failure = "cancel_siblings_and_drain"}}> : () -> ()
    "life_exec.declaration"() <{role = "call", values = {origin = "encoder", symbol = "scale_v1", version = "1", binding_role = "model", effect = "load", completion = "before_install"}}> : () -> ()
    "life_exec.declaration"() <{role = "call", values = {origin = "learner", symbol = "scale_v1", version = "1", binding_role = "model", effect = "load", completion = "before_install"}}> : () -> ()
    "life_exec.declaration"() <{role = "scope", values = {origin = "producer", parent = "run", kind = "life.producer", implementation = "encode_v1", horizon = "3", ready = "2", workers = "1", remaining = "discard_unclaimed", failure = "close_then_drain"}}> : () -> ()
    "life_exec.declaration"() <{role = "call", values = {origin = "producer", symbol = "encode_v1", version = "1", binding_role = "produce", effect = "read_source", completion = "physical_return_before_release"}}> : () -> ()
    "life_exec.declaration"() <{role = "access", values = {origin = "producer", participant = "encoder", mode = "read_source", through = "physical_return"}}> : () -> ()
    "life_exec.declaration"() <{role = "stop_link", values = {source = "consumers", target = "producer", relation = "normal_group_end_requests_stop", failure = "enclosing_cancel_then_physical_drain", run_completion = "producer_and_group_settled"}}> : () -> ()
    "life_exec.declaration"() <{role = "port", values = {producer = "producer", port_role = "slot:0", channel = "producer.slot.0", identity = "actual_record_and_source", ready_budget = "aggregate_provider_budget", settlement = "actual_use_then_notification"}}> : () -> ()
    "life_exec.declaration"() <{role = "port", values = {producer = "producer", port_role = "slot:1", channel = "producer.slot.1", identity = "actual_record_and_source", ready_budget = "aggregate_provider_budget", settlement = "actual_use_then_notification"}}> : () -> ()
    "life_exec.declaration"() <{role = "port", values = {producer = "producer", port_role = "evaluation", channel = "producer.evaluation", identity = "actual_record_and_source", ready_budget = "aggregate_provider_budget", settlement = "actual_use_then_notification"}}> : () -> ()
    "life_exec.declaration"() <{role = "port", values = {producer = "producer", port_role = "unused", channel = "producer.unused", identity = "actual_record_and_source", ready_budget = "aggregate_provider_budget", settlement = "actual_use_then_notification"}}> : () -> ()
    "life_exec.declaration"() <{role = "scope", values = {origin = "consumers", parent = "run", kind = "life.group", completion = "children_physically_settled", pending = "include_requested", failure = "cancel_siblings_and_drain"}}> : () -> ()
    "life_exec.declaration"() <{role = "scope", values = {origin = "training", parent = "consumers", kind = "life.repeat", max_windows = "2", admission = "policy_choice_before_first_claim", stop = "finish_selected_window"}}> : () -> ()
    "life_exec.declaration"() <{role = "call", values = {origin = "training.objective.0", symbol = "objective_v1", version = "1", binding_role = "objective", effect = "build_loss", completion = "return_before_backward"}}> : () -> ()
    "life_exec.declaration"() <{role = "access", values = {origin = "training.objective.0", participant = "learner", mode = "read_gradient", through = "backward"}}> : () -> ()
    "life_exec.declaration"() <{role = "call", values = {origin = "training.objective.1", symbol = "objective_v1", version = "1", binding_role = "objective", effect = "build_loss", completion = "return_before_backward"}}> : () -> ()
    "life_exec.declaration"() <{role = "access", values = {origin = "training.objective.1", participant = "learner", mode = "read_gradient", through = "backward"}}> : () -> ()
    "life_exec.declaration"() <{role = "access", values = {origin = "training.advance", participant = "learner", mode = "write", through = "update_outcome"}}> : () -> ()
    "life_exec.declaration"() <{role = "request", values = {origin = "request", source = "training", after_window = "2", multiplicity = "once_per_matching_completion", pending = "keeps_group_alive", max_pending = "1", cancellation = "cancel_or_unfulfilled"}}> : () -> ()
    "life_exec.declaration"() <{role = "scope", values = {origin = "evaluation", parent = "consumers", kind = "life.evaluator", completion = "physical_and_cleanup"}}> : () -> ()
    "life_exec.declaration"() <{role = "call", values = {origin = "evaluation.read", symbol = "measure_v1", version = "1", binding_role = "evaluate", effect = "protected_read", completion = "physical_validate_restore"}}> : () -> ()
    "life_exec.declaration"() <{role = "access", values = {origin = "evaluation.read", participant = "learner", mode = "read_evaluation", through = "physical_and_restore"}}> : () -> ()
  }) : () -> ()
}) : () -> ()
