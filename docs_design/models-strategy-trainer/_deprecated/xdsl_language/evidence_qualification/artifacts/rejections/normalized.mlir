"builtin.module"() ({
  "lang.program"() <{origin = "run", version = 1 : i64, ancestry = ["run"], stage = "normalized"}> ({
    %0 = "lang.resource_decl"() <{origin = "policy", version = 1 : i64, ancestry = ["run", "policy"], sym_name = "policy"}> : () -> !lang.resource<"policy">
    %1 = "lang.stream"() <{origin = "records", version = 1 : i64, ancestry = ["run", "records"], sym_name = "records", horizon = 3 : i64, ready_bound = 2 : i64, admission = "bounded_lag", owners = [@policy], metric = "owner_update_ordinal_within_lineage", comparison_boundary = "item_admission", max_lag = 0 : i64, reproducibility = "unknown", rank_rule = "local", assumptions = []}> : () -> !lang.endpoint<!lang.payload<"record">>
    "lang.parallel"(%1, %0) <{origin = "activities", version = 1 : i64, ancestry = ["run", "activities"], sym_name = "activities", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
    ^bb0(%2: !lang.endpoint<!lang.payload<"record">>, %3: !lang.resource<"policy">):
      "lang.producer"(%2, %3) <{origin = "producer", version = 1 : i64, ancestry = ["run", "activities", "producer"], sym_name = "producer", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%4: !lang.resource<"policy">, %5: !lang.payload<"index">):
        %6 = "lang.access"(%4, %5) <{origin = "select.access", version = 1 : i64, ancestry = ["run", "activities", "producer", "select.access"], sym_name = "select.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "snapshot", freshness = "published", version_selector = "initial", profile = "ordinary_owner"}> ({
        ^bb0(%7: !lang.capability<!lang.resource<"policy">, "snapshot">, %8: !lang.payload<"index">):
          %9 = "lang.call"(%7, %8) <{origin = "select", version = 1 : i64, ancestry = ["run", "activities", "producer", "select.access", "select"], binding = @select, binding_version = "1"}> : (!lang.capability<!lang.resource<"policy">, "snapshot">, !lang.payload<"index">) -> !lang.payload<"record">
          "lang.yield"(%9) <{origin = "select.access.yield", version = 1 : i64, ancestry = ["run", "activities", "producer", "select.access", "select.access.yield"]}> : (!lang.payload<"record">) -> ()
        }) : (!lang.resource<"policy">, !lang.payload<"index">) -> !lang.payload<"record">
        "lang.yield"(%6) <{origin = "production.body.yield", version = 1 : i64, ancestry = ["run", "activities", "producer", "production.body.yield"]}> : (!lang.payload<"record">) -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"policy">) -> ()
      "lang.yield"() <{origin = "producer.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "producer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%4: !lang.endpoint<!lang.payload<"record">>, %5: !lang.resource<"policy">):
      "lang.repeat"(%4, %5) <{origin = "consumer", version = 1 : i64, ancestry = ["run", "activities", "consumer"], sym_name = "consumer", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 1 : i64, carry_count = 0 : i64, policy = @bounded_count}> ({
      ^bb0(%6: !lang.endpoint<!lang.payload<"record">>, %7: !lang.resource<"policy">, %8: !lang.payload<"index">):
        %9 = "lang.take"(%6) <{origin = "consume.take", version = 1 : i64, ancestry = ["run", "activities", "consumer", "consume.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
        %10 = "lang.borrow"(%9) <{origin = "consume.take.borrow", version = 1 : i64, ancestry = ["run", "activities", "consumer", "consume.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
        %11 = "lang.sequence"(%10) <{origin = "score.recipe", version = 1 : i64, ancestry = ["run", "activities", "consumer", "score.recipe"], sym_name = "score.recipe", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
        ^bb0(%12: !lang.payload<"record">):
          %13 = "lang.call"(%12) <{origin = "score", version = 1 : i64, ancestry = ["run", "activities", "consumer", "score.recipe", "score"], binding = @score, binding_version = "1"}> : (!lang.payload<"record">) -> !lang.payload<"scalar">
          "lang.yield"(%13) <{origin = "score.recipe.yield", version = 1 : i64, ancestry = ["run", "activities", "consumer", "score.recipe", "score.recipe.yield"]}> : (!lang.payload<"scalar">) -> ()
        }) : (!lang.payload<"record">) -> !lang.payload<"scalar">
        "lang.retire"(%9) <{origin = "consume.retire", version = 1 : i64, ancestry = ["run", "activities", "consumer", "consume.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
        "lang.yield"() <{origin = "consume.body.yield", version = 1 : i64, ancestry = ["run", "activities", "consumer", "consume.body.yield"]}> : () -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"policy">) -> ()
      "lang.yield"() <{origin = "consumer.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "consumer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%6: !lang.endpoint<!lang.payload<"record">>, %7: !lang.resource<"policy">):
      "lang.stop_after"() <{origin = "stop", version = 1 : i64, ancestry = ["run", "activities", "stop"], source = @consumer, target = @producer}> : () -> ()
      "lang.yield"() <{origin = "control.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "control.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%8: !lang.endpoint<!lang.payload<"record">>, %9: !lang.resource<"policy">):
      %10 = "lang.constant"() <{origin = "write.value.1", version = 1 : i64, ancestry = ["run", "activities", "write.value.1"], literal = "1"}> : () -> !lang.payload<"scalar">
      %11 = "lang.access"(%9, %10) <{origin = "write.access.1", version = 1 : i64, ancestry = ["run", "activities", "write.access.1"], sym_name = "write.access.1", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
      ^bb0(%12: !lang.capability<!lang.resource<"policy">, "owner_update">, %13: !lang.payload<"scalar">):
        %14 = "lang.call"(%12, %13) <{origin = "write.1", version = 1 : i64, ancestry = ["run", "activities", "write.access.1", "write.1"], binding = @feedback, binding_version = "1"}> : (!lang.capability<!lang.resource<"policy">, "owner_update">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
        "lang.yield"(%14) <{origin = "write.body.1.yield", version = 1 : i64, ancestry = ["run", "activities", "write.access.1", "write.body.1.yield"]}> : (!lang.payload<"scalar">) -> ()
      }) : (!lang.resource<"policy">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
      %12 = "lang.constant"() <{origin = "write.value.2", version = 1 : i64, ancestry = ["run", "activities", "write.value.2"], literal = "2"}> : () -> !lang.payload<"scalar">
      %13 = "lang.access"(%9, %12) <{origin = "write.access.2", version = 1 : i64, ancestry = ["run", "activities", "write.access.2"], sym_name = "write.access.2", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
      ^bb0(%14: !lang.capability<!lang.resource<"policy">, "owner_update">, %15: !lang.payload<"scalar">):
        %16 = "lang.call"(%14, %15) <{origin = "write.2", version = 1 : i64, ancestry = ["run", "activities", "write.access.2", "write.2"], binding = @feedback, binding_version = "1"}> : (!lang.capability<!lang.resource<"policy">, "owner_update">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
        "lang.yield"(%16) <{origin = "write.body.2.yield", version = 1 : i64, ancestry = ["run", "activities", "write.access.2", "write.body.2.yield"]}> : (!lang.payload<"scalar">) -> ()
      }) : (!lang.resource<"policy">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
      "lang.yield"() <{origin = "writer.yield", version = 1 : i64, ancestry = ["run", "activities", "writer.yield"]}> : () -> ()
    }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"policy">) -> ()
  }) : () -> ()
}) : () -> ()
