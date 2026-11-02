"builtin.module"() ({
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
}) : () -> ()
