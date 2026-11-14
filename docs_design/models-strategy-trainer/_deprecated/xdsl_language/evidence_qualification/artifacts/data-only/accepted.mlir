"builtin.module"() ({
  "lang.program"() <{origin = "run", version = 1 : i64, ancestry = ["run"], stage = "accepted"}> ({
    %0 = "lang.resource_decl"() <{origin = "policy", version = 1 : i64, ancestry = ["policy"], sym_name = "policy"}> : () -> !lang.resource<"policy">
    %1 = "lang.stream"() <{origin = "records", version = 1 : i64, ancestry = ["records"], sym_name = "records", horizon = 1 : i64, ready_bound = 1 : i64, admission = "any_with_provenance", owners = [@policy], metric = "owner_update_ordinal_within_lineage", comparison_boundary = "item_admission", max_lag = 0 : i64, reproducibility = "unknown", rank_rule = "local", assumptions = []}> : () -> !lang.endpoint<!lang.payload<"record">>
    %2 = "lang.resource_decl"() <{origin = "products", version = 1 : i64, ancestry = ["products"], sym_name = "products"}> : () -> !lang.resource<"products">
    "lang.parallel"(%1, %0, %2) <{origin = "activities", version = 1 : i64, ancestry = ["activities"], sym_name = "activities", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
    ^bb0(%3: !lang.endpoint<!lang.payload<"record">>, %4: !lang.resource<"policy">, %5: !lang.resource<"products">):
      "lang.producer"(%3, %4) <{origin = "producer", version = 1 : i64, ancestry = ["producer"], sym_name = "producer", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%6: !lang.resource<"policy">, %7: !lang.payload<"index">):
        %8 = "lang.access"(%6, %7) <{origin = "select.access", version = 1 : i64, ancestry = ["select.access"], sym_name = "select.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "protected_current", freshness = "current_at_start", version_selector = "current", profile = "ordinary_owner"}> ({
        ^bb0(%9: !lang.capability<!lang.resource<"policy">, "protected_current">, %10: !lang.payload<"index">):
          %11 = "lang.call"(%9, %10) <{origin = "select", version = 1 : i64, ancestry = ["select"], binding = @select, binding_version = "1"}> : (!lang.capability<!lang.resource<"policy">, "protected_current">, !lang.payload<"index">) -> !lang.payload<"record">
          "lang.yield"(%11) <{origin = "select.access.yield", version = 1 : i64, ancestry = ["select.access.yield"]}> : (!lang.payload<"record">) -> ()
        }) : (!lang.resource<"policy">, !lang.payload<"index">) -> !lang.payload<"record">
        "lang.yield"(%8) <{origin = "production.body.yield", version = 1 : i64, ancestry = ["production.body.yield"]}> : (!lang.payload<"record">) -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"policy">) -> ()
      "lang.yield"() <{origin = "producer.branch.yield", version = 1 : i64, ancestry = ["producer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%6: !lang.endpoint<!lang.payload<"record">>, %7: !lang.resource<"policy">, %8: !lang.resource<"products">):
      "lang.repeat"(%6, %7, %8) <{origin = "consumer", version = 1 : i64, ancestry = ["consumer"], sym_name = "consumer", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 4 : i64, carry_count = 0 : i64, policy = @bounded_count}> ({
      ^bb0(%9: !lang.endpoint<!lang.payload<"record">>, %10: !lang.resource<"policy">, %11: !lang.resource<"products">, %12: !lang.payload<"index">):
        %13 = "lang.take"(%9) <{origin = "consume.take", version = 1 : i64, ancestry = ["consume.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
        %14 = "lang.borrow"(%13) <{origin = "consume.take.borrow", version = 1 : i64, ancestry = ["consume.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
        %15 = "lang.recipe"(%14) <{origin = "score.recipe", version = 1 : i64, ancestry = ["score.recipe"], sym_name = "score.recipe", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
        ^bb0(%16: !lang.payload<"record">):
          %17 = "lang.call"(%16) <{origin = "score", version = 1 : i64, ancestry = ["score"], binding = @score, binding_version = "1"}> : (!lang.payload<"record">) -> !lang.payload<"scalar">
          "lang.yield"(%17) <{origin = "score.recipe.yield", version = 1 : i64, ancestry = ["score.recipe.yield"]}> : (!lang.payload<"scalar">) -> ()
        }) : (!lang.payload<"record">) -> !lang.payload<"scalar">
        %16 = "lang.access"(%10, %15) <{origin = "feedback.access", version = 1 : i64, ancestry = ["feedback.access"], sym_name = "feedback.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
        ^bb0(%17: !lang.capability<!lang.resource<"policy">, "owner_update">, %18: !lang.payload<"scalar">):
          %19 = "lang.call"(%17, %18) <{origin = "feedback", version = 1 : i64, ancestry = ["feedback"], binding = @feedback, binding_version = "1"}> : (!lang.capability<!lang.resource<"policy">, "owner_update">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
          "lang.yield"(%19) <{origin = "feedback.access.yield", version = 1 : i64, ancestry = ["feedback.access.yield"]}> : (!lang.payload<"scalar">) -> ()
        }) : (!lang.resource<"policy">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
        %17 = "lang.access"(%11, %14, %15) <{origin = "publish_score.access", version = 1 : i64, ancestry = ["publish_score.access"], sym_name = "publish_score.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
        ^bb0(%18: !lang.capability<!lang.resource<"products">, "owner_update">, %19: !lang.payload<"record">, %20: !lang.payload<"scalar">):
          %21 = "lang.call"(%18, %19, %20) <{origin = "publish_score", version = 1 : i64, ancestry = ["publish_score"], binding = @publish_score, binding_version = "1"}> : (!lang.capability<!lang.resource<"products">, "owner_update">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
          "lang.yield"(%21) <{origin = "publish_score.body.yield", version = 1 : i64, ancestry = ["publish_score.body.yield"]}> : (!lang.payload<"scalar">) -> ()
        }) : (!lang.resource<"products">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
        "lang.retire"(%13) <{origin = "consume.retire", version = 1 : i64, ancestry = ["consume.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
        "lang.yield"() <{origin = "consume.body.yield", version = 1 : i64, ancestry = ["consume.body.yield"]}> : () -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"policy">, !lang.resource<"products">) -> ()
      "lang.yield"() <{origin = "consumer.branch.yield", version = 1 : i64, ancestry = ["consumer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%9: !lang.endpoint<!lang.payload<"record">>, %10: !lang.resource<"policy">, %11: !lang.resource<"products">):
      "lang.stop_after"() <{origin = "stop", version = 1 : i64, ancestry = ["stop"], source = @consumer, target = @producer}> : () -> ()
      "lang.yield"() <{origin = "control.branch.yield", version = 1 : i64, ancestry = ["control.branch.yield"]}> : () -> ()
    }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"policy">, !lang.resource<"products">) -> ()
  }) : () -> ()
}) : () -> ()
