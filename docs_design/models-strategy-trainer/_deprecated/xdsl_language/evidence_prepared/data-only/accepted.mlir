"builtin.module"() ({
  "lang.program"() <{origin = "run", version = 1 : i64, ancestry = ["run"], stage = "accepted"}> ({
    %0 = "lang.resource_decl"() <{origin = "policy", version = 1 : i64, ancestry = ["policy"], sym_name = "policy"}> : () -> !lang.resource<"policy">
    %1 = "lang.stream"() <{origin = "records", version = 1 : i64, ancestry = ["records"], sym_name = "records", horizon = 3 : i64, ready_bound = 2 : i64, admission = "any_with_provenance", owners = [@policy], metric = "owner_update_ordinal_within_lineage", comparison_boundary = "item_admission", max_lag = 0 : i64, reproducibility = "unknown", rank_rule = "local", assumptions = []}> : () -> !lang.endpoint<!lang.payload<"record">>
    "lang.parallel"(%1, %0) <{origin = "activities", version = 1 : i64, ancestry = ["activities"], sym_name = "activities", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
    ^bb0(%2: !lang.endpoint<!lang.payload<"record">>, %3: !lang.resource<"policy">):
      "lang.producer"(%2, %3) <{origin = "producer", version = 1 : i64, ancestry = ["producer"], sym_name = "producer", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%4: !lang.resource<"policy">, %5: !lang.payload<"index">):
        %6 = "lang.access"(%4, %5) <{origin = "select.access", version = 1 : i64, ancestry = ["select.access"], sym_name = "select.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "protected_current", freshness = "current_at_start", version_selector = "current", profile = "ordinary_owner"}> ({
        ^bb0(%7: !lang.capability<!lang.resource<"policy">, "protected_current">, %8: !lang.payload<"index">):
          %9 = "lang.call"(%7, %8) <{origin = "select", version = 1 : i64, ancestry = ["select"], binding = @select, binding_version = "1"}> : (!lang.capability<!lang.resource<"policy">, "protected_current">, !lang.payload<"index">) -> !lang.payload<"record">
          "lang.yield"(%9) <{origin = "select.access.yield", version = 1 : i64, ancestry = ["select.access.yield"]}> : (!lang.payload<"record">) -> ()
        }) : (!lang.resource<"policy">, !lang.payload<"index">) -> !lang.payload<"record">
        "lang.yield"(%6) <{origin = "production.body.yield", version = 1 : i64, ancestry = ["production.body.yield"]}> : (!lang.payload<"record">) -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"policy">) -> ()
      "lang.yield"() <{origin = "producer.branch.yield", version = 1 : i64, ancestry = ["producer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%4: !lang.endpoint<!lang.payload<"record">>, %5: !lang.resource<"policy">):
      "lang.repeat"(%4, %5) <{origin = "consumer", version = 1 : i64, ancestry = ["consumer"], sym_name = "consumer", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 4 : i64, carry_count = 0 : i64, policy = @bounded_count}> ({
      ^bb0(%6: !lang.endpoint<!lang.payload<"record">>, %7: !lang.resource<"policy">, %8: !lang.payload<"index">):
        %9 = "lang.take"(%6) <{origin = "consume.take", version = 1 : i64, ancestry = ["consume.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
        %10 = "lang.borrow"(%9) <{origin = "consume.take.borrow", version = 1 : i64, ancestry = ["consume.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
        %11 = "lang.recipe"(%10) <{origin = "score.recipe", version = 1 : i64, ancestry = ["score.recipe"], sym_name = "score.recipe", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
        ^bb0(%12: !lang.payload<"record">):
          %13 = "lang.call"(%12) <{origin = "score", version = 1 : i64, ancestry = ["score"], binding = @score, binding_version = "1"}> : (!lang.payload<"record">) -> !lang.payload<"scalar">
          "lang.yield"(%13) <{origin = "score.recipe.yield", version = 1 : i64, ancestry = ["score.recipe.yield"]}> : (!lang.payload<"scalar">) -> ()
        }) : (!lang.payload<"record">) -> !lang.payload<"scalar">
        %12 = "lang.access"(%7, %11) <{origin = "feedback.access", version = 1 : i64, ancestry = ["feedback.access"], sym_name = "feedback.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
        ^bb0(%13: !lang.capability<!lang.resource<"policy">, "owner_update">, %14: !lang.payload<"scalar">):
          %15 = "lang.call"(%13, %14) <{origin = "feedback", version = 1 : i64, ancestry = ["feedback"], binding = @feedback, binding_version = "1"}> : (!lang.capability<!lang.resource<"policy">, "owner_update">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
          "lang.yield"(%15) <{origin = "feedback.access.yield", version = 1 : i64, ancestry = ["feedback.access.yield"]}> : (!lang.payload<"scalar">) -> ()
        }) : (!lang.resource<"policy">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
        "lang.retire"(%9) <{origin = "consume.retire", version = 1 : i64, ancestry = ["consume.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
        "lang.yield"() <{origin = "consume.body.yield", version = 1 : i64, ancestry = ["consume.body.yield"]}> : () -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"policy">) -> ()
      "lang.yield"() <{origin = "consumer.branch.yield", version = 1 : i64, ancestry = ["consumer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%6: !lang.endpoint<!lang.payload<"record">>, %7: !lang.resource<"policy">):
      "lang.stop_after"() <{origin = "stop", version = 1 : i64, ancestry = ["stop"], source = @consumer, target = @producer}> : () -> ()
      "lang.yield"() <{origin = "control.branch.yield", version = 1 : i64, ancestry = ["control.branch.yield"]}> : () -> ()
    }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"policy">) -> ()
  }) : () -> ()
}) : () -> ()
