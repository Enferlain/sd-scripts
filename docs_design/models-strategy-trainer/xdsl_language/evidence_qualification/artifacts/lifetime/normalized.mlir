"builtin.module"() ({
  "lang.program"() <{origin = "run", version = 1 : i64, ancestry = ["run"], stage = "normalized"}> ({
    %0 = "lang.resource_decl"() <{origin = "encoder", version = 1 : i64, ancestry = ["run", "encoder"], sym_name = "encoder"}> : () -> !lang.resource<"scale">
    %1 = "lang.resource_decl"() <{origin = "model", version = 1 : i64, ancestry = ["run", "model"], sym_name = "model"}> : () -> !lang.resource<"scale">
    %2 = "lang.stream"() <{origin = "records", version = 1 : i64, ancestry = ["run", "records"], sym_name = "records", horizon = 3 : i64, ready_bound = 2 : i64, admission = "any_with_provenance", owners = [@encoder], metric = "owner_update_ordinal_within_lineage", comparison_boundary = "item_admission", max_lag = 0 : i64, reproducibility = "unknown", rank_rule = "local", assumptions = []}> : () -> !lang.endpoint<!lang.payload<"record">>
    %3 = "lang.resource_decl"() <{origin = "products", version = 1 : i64, ancestry = ["run", "products"], sym_name = "products"}> : () -> !lang.resource<"products">
    "lang.parallel"(%2, %0, %1, %3) <{origin = "activities", version = 1 : i64, ancestry = ["run", "activities"], sym_name = "activities", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
    ^bb0(%4: !lang.endpoint<!lang.payload<"record">>, %5: !lang.resource<"scale">, %6: !lang.resource<"scale">, %7: !lang.resource<"products">):
      "lang.producer"(%4, %5) <{origin = "producer", version = 1 : i64, ancestry = ["run", "activities", "producer"], sym_name = "producer", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%8: !lang.resource<"scale">, %9: !lang.payload<"index">):
        %10 = "lang.access"(%8, %9) <{origin = "encode.access", version = 1 : i64, ancestry = ["run", "activities", "producer", "encode.access"], sym_name = "encode.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "snapshot", freshness = "published", version_selector = "initial", profile = "ordinary_owner"}> ({
        ^bb0(%11: !lang.capability<!lang.resource<"scale">, "snapshot">, %12: !lang.payload<"index">):
          %13 = "lang.call"(%11, %12) <{origin = "encode", version = 1 : i64, ancestry = ["run", "activities", "producer", "encode.access", "encode"], binding = @encode, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "snapshot">, !lang.payload<"index">) -> !lang.payload<"record">
          "lang.yield"(%13) <{origin = "encode.access.yield", version = 1 : i64, ancestry = ["run", "activities", "producer", "encode.access", "encode.access.yield"]}> : (!lang.payload<"record">) -> ()
        }) : (!lang.resource<"scale">, !lang.payload<"index">) -> !lang.payload<"record">
        "lang.yield"(%10) <{origin = "production.body.yield", version = 1 : i64, ancestry = ["run", "activities", "producer", "production.body.yield"]}> : (!lang.payload<"record">) -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">) -> ()
      "lang.yield"() <{origin = "producer.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "producer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%8: !lang.endpoint<!lang.payload<"record">>, %9: !lang.resource<"scale">, %10: !lang.resource<"scale">, %11: !lang.resource<"products">):
      "lang.parallel"(%8, %10, %11) <{origin = "consumers", version = 1 : i64, ancestry = ["run", "activities", "consumers"], sym_name = "consumers", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%12: !lang.endpoint<!lang.payload<"record">>, %13: !lang.resource<"scale">, %14: !lang.resource<"products">):
        "lang.repeat"(%12, %13) <{origin = "training", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training"], sym_name = "training", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 2 : i64, carry_count = 0 : i64, policy = @bounded_count}> ({
        ^bb0(%15: !lang.endpoint<!lang.payload<"record">>, %16: !lang.resource<"scale">, %17: !lang.payload<"index">):
          %18 = "lang.constant"() <{origin = "zero", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "zero"], literal = "0.0"}> : () -> !lang.payload<"scalar">
          %19 = "lang.repeat"(%15, %16, %18) <{origin = "members", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "members"], sym_name = "members", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 2 : i64, carry_count = 1 : i64, policy = @bounded_count}> ({
          ^bb0(%20: !lang.endpoint<!lang.payload<"record">>, %21: !lang.resource<"scale">, %22: !lang.payload<"index">, %23: !lang.payload<"scalar">):
            %24 = "lang.take"(%20) <{origin = "training.take", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "members", "training.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
            %25 = "lang.borrow"(%24) <{origin = "training.take.borrow", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "members", "training.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
            %26 = "lang.access"(%21, %25, %23) <{origin = "gradient.access", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "members", "gradient.access"], sym_name = "gradient.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "protected_current", freshness = "current_at_start", version_selector = "current", profile = "ordinary_owner"}> ({
            ^bb0(%27: !lang.capability<!lang.resource<"scale">, "protected_current">, %28: !lang.payload<"record">, %29: !lang.payload<"scalar">):
              %30 = "lang.call"(%27, %28, %29) <{origin = "gradient", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "members", "gradient.access", "gradient"], binding = @gradient, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "protected_current">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
              "lang.yield"(%30) <{origin = "gradient.access.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "members", "gradient.access", "gradient.access.yield"]}> : (!lang.payload<"scalar">) -> ()
            }) : (!lang.resource<"scale">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
            "lang.retire"(%24) <{origin = "training.retire", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "members", "training.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
            "lang.yield"(%26) <{origin = "members.body.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "members", "members.body.yield"]}> : (!lang.payload<"scalar">) -> ()
          }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
          %20 = "lang.access"(%16, %19) <{origin = "update.access", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "update.access"], sym_name = "update.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
          ^bb0(%21: !lang.capability<!lang.resource<"scale">, "owner_update">, %22: !lang.payload<"scalar">):
            %23 = "lang.call"(%21, %22) <{origin = "advance", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "update.access", "advance"], binding = @advance, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "owner_update">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
            "lang.yield"(%23) <{origin = "update.access.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "update.access", "update.access.yield"]}> : (!lang.payload<"scalar">) -> ()
          }) : (!lang.resource<"scale">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
          %21 = "lang.constant"() <{origin = "second.window", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "second.window"], literal = "1"}> : () -> !lang.payload<"index">
          %22 = "lang.call"(%17, %21) <{origin = "evaluation.due", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "evaluation.due"], binding = @equal, binding_version = "1"}> : (!lang.payload<"index">, !lang.payload<"index">) -> !lang.payload<"bool">
          "lang.choice"(%22, %17) <{origin = "request.choice", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "request.choice"], sym_name = "request.choice", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
          ^bb0(%23: !lang.payload<"index">):
            "lang.request"(%23) <{origin = "request", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "request.choice", "request"], handler = @evaluation, source = @training, boundary = "settled_success"}> : (!lang.payload<"index">) -> ()
            "lang.yield"() <{origin = "request.true.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "request.choice", "request.true.yield"]}> : () -> ()
          }, {
          ^bb0(%24: !lang.payload<"index">):
            "lang.yield"() <{origin = "request.false.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "request.choice", "request.false.yield"]}> : () -> ()
          }) : (!lang.payload<"bool">, !lang.payload<"index">) -> ()
          "lang.yield"() <{origin = "window.body.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training", "window.body.yield"]}> : () -> ()
        }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">) -> ()
        "lang.yield"() <{origin = "training.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "training.branch.yield"]}> : () -> ()
      }, {
      ^bb0(%15: !lang.endpoint<!lang.payload<"record">>, %16: !lang.resource<"scale">, %17: !lang.resource<"products">):
        "lang.requests"(%15, %16, %17) <{origin = "evaluation", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation"], sym_name = "evaluation", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], sources = [@training], request_schema = !lang.payload<"index">, max_outstanding = 1 : i64, max_active = 1 : i64}> ({
        ^bb0(%18: !lang.endpoint<!lang.payload<"record">>, %19: !lang.resource<"scale">, %20: !lang.resource<"products">, %21: !lang.request<!lang.payload<"index">>):
          %22 = "lang.take"(%18) <{origin = "evaluation.take", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "evaluation.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
          %23 = "lang.borrow"(%22) <{origin = "evaluation.take.borrow", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "evaluation.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
          %24 = "lang.access"(%19, %23) <{origin = "measure.access", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "measure.access"], sym_name = "measure.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "protected_current", freshness = "current_at_start", version_selector = "current", profile = "ordinary_owner"}> ({
          ^bb0(%25: !lang.capability<!lang.resource<"scale">, "protected_current">, %26: !lang.payload<"record">):
            %27 = "lang.call"(%25, %26) <{origin = "measure", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "measure.access", "measure"], binding = @measure, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "protected_current">, !lang.payload<"record">) -> !lang.payload<"measurement">
            "lang.yield"(%27) <{origin = "measure.access.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "measure.access", "measure.access.yield"]}> : (!lang.payload<"measurement">) -> ()
          }) : (!lang.resource<"scale">, !lang.payload<"record">) -> !lang.payload<"measurement">
          %25 = "lang.access"(%20, %24) <{origin = "publish_measurement.access", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "publish_measurement.access"], sym_name = "publish_measurement.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
          ^bb0(%26: !lang.capability<!lang.resource<"products">, "owner_update">, %27: !lang.payload<"measurement">):
            %28 = "lang.call"(%26, %27) <{origin = "publish_measurement", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "publish_measurement.access", "publish_measurement"], binding = @publish_measurement, binding_version = "1"}> : (!lang.capability<!lang.resource<"products">, "owner_update">, !lang.payload<"measurement">) -> !lang.payload<"measurement">
            "lang.yield"(%28) <{origin = "publish_measurement.body.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "publish_measurement.access", "publish_measurement.body.yield"]}> : (!lang.payload<"measurement">) -> ()
          }) : (!lang.resource<"products">, !lang.payload<"measurement">) -> !lang.payload<"measurement">
          "lang.retire"(%22) <{origin = "evaluation.retire", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "evaluation.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
          "lang.yield"() <{origin = "evaluation.body.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation", "evaluation.body.yield"]}> : () -> ()
        }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.resource<"products">) -> ()
        "lang.yield"() <{origin = "evaluation.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "evaluation.branch.yield"]}> : () -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.resource<"products">) -> ()
      "lang.yield"() <{origin = "consumers.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%12: !lang.endpoint<!lang.payload<"record">>, %13: !lang.resource<"scale">, %14: !lang.resource<"scale">, %15: !lang.resource<"products">):
      "lang.stop_after"() <{origin = "stop", version = 1 : i64, ancestry = ["run", "activities", "stop"], source = @consumers, target = @producer}> : () -> ()
      "lang.yield"() <{origin = "control.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "control.branch.yield"]}> : () -> ()
    }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.resource<"scale">, !lang.resource<"products">) -> ()
  }) : () -> ()
}) : () -> ()
