"builtin.module"() ({
  "lang.program"() <{origin = "nested.run", version = 1 : i64, ancestry = ["nested.run"], stage = "accepted"}> ({
    %0 = "lang.resource_decl"() <{origin = "nested.encoder", version = 1 : i64, ancestry = ["nested.encoder"], sym_name = "nested.encoder"}> : () -> !lang.resource<"scale">
    %1 = "lang.resource_decl"() <{origin = "nested.model", version = 1 : i64, ancestry = ["nested.model"], sym_name = "nested.model"}> : () -> !lang.resource<"scale">
    %2 = "lang.stream"() <{origin = "nested.records", version = 1 : i64, ancestry = ["nested.records"], sym_name = "nested.records", horizon = 3 : i64, ready_bound = 2 : i64, admission = "any_with_provenance", owners = [@nested.encoder], metric = "owner_update_ordinal_within_lineage", comparison_boundary = "item_admission", max_lag = 0 : i64, reproducibility = "unknown", rank_rule = "local", assumptions = []}> : () -> !lang.endpoint<!lang.payload<"record">>
    %3 = "lang.resource_decl"() <{origin = "nested.products", version = 1 : i64, ancestry = ["nested.products"], sym_name = "nested.products"}> : () -> !lang.resource<"products">
    "lang.parallel"(%2, %0, %1, %3) <{origin = "nested.activities", version = 1 : i64, ancestry = ["nested.activities"], sym_name = "nested.activities", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
    ^bb0(%4: !lang.endpoint<!lang.payload<"record">>, %5: !lang.resource<"scale">, %6: !lang.resource<"scale">, %7: !lang.resource<"products">):
      "lang.producer"(%4, %5) <{origin = "nested.producer", version = 1 : i64, ancestry = ["nested.producer"], sym_name = "nested.producer", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%8: !lang.resource<"scale">, %9: !lang.payload<"index">):
        %10 = "lang.access"(%8, %9) <{origin = "nested.encode.access", version = 1 : i64, ancestry = ["nested.encode.access"], sym_name = "nested.encode.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "snapshot", freshness = "published", version_selector = "initial", profile = "ordinary_owner"}> ({
        ^bb0(%11: !lang.capability<!lang.resource<"scale">, "snapshot">, %12: !lang.payload<"index">):
          %13 = "lang.call"(%11, %12) <{origin = "nested.encode", version = 1 : i64, ancestry = ["nested.encode"], binding = @encode, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "snapshot">, !lang.payload<"index">) -> !lang.payload<"record">
          "lang.yield"(%13) <{origin = "nested.encode.access.yield", version = 1 : i64, ancestry = ["nested.encode.access.yield"]}> : (!lang.payload<"record">) -> ()
        }) : (!lang.resource<"scale">, !lang.payload<"index">) -> !lang.payload<"record">
        "lang.yield"(%10) <{origin = "nested.production.body.yield", version = 1 : i64, ancestry = ["nested.production.body.yield"]}> : (!lang.payload<"record">) -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">) -> ()
      "lang.yield"() <{origin = "nested.producer.branch.yield", version = 1 : i64, ancestry = ["nested.producer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%8: !lang.endpoint<!lang.payload<"record">>, %9: !lang.resource<"scale">, %10: !lang.resource<"scale">, %11: !lang.resource<"products">):
      "lang.once"(%8, %10, %11) <{origin = "nested.consumer.scope", version = 1 : i64, ancestry = ["nested.consumer.scope"], sym_name = "nested.consumer.scope", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%12: !lang.endpoint<!lang.payload<"record">>, %13: !lang.resource<"scale">, %14: !lang.resource<"products">):
        "lang.parallel"(%12, %13, %14) <{origin = "nested.consumers", version = 1 : i64, ancestry = ["nested.consumers"], sym_name = "nested.consumers", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
        ^bb0(%15: !lang.endpoint<!lang.payload<"record">>, %16: !lang.resource<"scale">, %17: !lang.resource<"products">):
          "lang.repeat"(%15, %16) <{origin = "nested.training", version = 1 : i64, ancestry = ["nested.training"], sym_name = "nested.training", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 2 : i64, carry_count = 0 : i64, policy = @bounded_count}> ({
          ^bb0(%18: !lang.endpoint<!lang.payload<"record">>, %19: !lang.resource<"scale">, %20: !lang.payload<"index">):
            %21 = "lang.constant"() <{origin = "nested.zero", version = 1 : i64, ancestry = ["nested.zero"], literal = "0.0"}> : () -> !lang.payload<"scalar">
            %22 = "lang.repeat"(%18, %19, %21) <{origin = "nested.members", version = 1 : i64, ancestry = ["nested.members"], sym_name = "nested.members", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 2 : i64, carry_count = 1 : i64, policy = @bounded_count}> ({
            ^bb0(%23: !lang.endpoint<!lang.payload<"record">>, %24: !lang.resource<"scale">, %25: !lang.payload<"index">, %26: !lang.payload<"scalar">):
              %27 = "lang.take"(%23) <{origin = "nested.training.take", version = 1 : i64, ancestry = ["nested.training.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
              %28 = "lang.borrow"(%27) <{origin = "nested.training.take.borrow", version = 1 : i64, ancestry = ["nested.training.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
              %29 = "lang.access"(%24, %28, %26) <{origin = "nested.gradient.access", version = 1 : i64, ancestry = ["nested.gradient.access"], sym_name = "nested.gradient.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "protected_current", freshness = "current_at_start", version_selector = "current", profile = "ordinary_owner"}> ({
              ^bb0(%30: !lang.capability<!lang.resource<"scale">, "protected_current">, %31: !lang.payload<"record">, %32: !lang.payload<"scalar">):
                %33 = "lang.call"(%30, %31, %32) <{origin = "nested.gradient", version = 1 : i64, ancestry = ["nested.gradient"], binding = @gradient, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "protected_current">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
                "lang.yield"(%33) <{origin = "nested.gradient.access.yield", version = 1 : i64, ancestry = ["nested.gradient.access.yield"]}> : (!lang.payload<"scalar">) -> ()
              }) : (!lang.resource<"scale">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
              "lang.retire"(%27) <{origin = "nested.training.retire", version = 1 : i64, ancestry = ["nested.training.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
              "lang.yield"(%29) <{origin = "nested.members.body.yield", version = 1 : i64, ancestry = ["nested.members.body.yield"]}> : (!lang.payload<"scalar">) -> ()
            }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
            %23 = "lang.access"(%19, %22) <{origin = "nested.update.access", version = 1 : i64, ancestry = ["nested.update.access"], sym_name = "nested.update.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
            ^bb0(%24: !lang.capability<!lang.resource<"scale">, "owner_update">, %25: !lang.payload<"scalar">):
              %26 = "lang.call"(%24, %25) <{origin = "nested.advance", version = 1 : i64, ancestry = ["nested.advance"], binding = @advance, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "owner_update">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
              "lang.yield"(%26) <{origin = "nested.update.access.yield", version = 1 : i64, ancestry = ["nested.update.access.yield"]}> : (!lang.payload<"scalar">) -> ()
            }) : (!lang.resource<"scale">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
            %24 = "lang.constant"() <{origin = "nested.second.window", version = 1 : i64, ancestry = ["nested.second.window"], literal = "1"}> : () -> !lang.payload<"index">
            %25 = "lang.call"(%20, %24) <{origin = "nested.evaluation.due", version = 1 : i64, ancestry = ["nested.evaluation.due"], binding = @equal, binding_version = "1"}> : (!lang.payload<"index">, !lang.payload<"index">) -> !lang.payload<"bool">
            "lang.choice"(%25, %20) <{origin = "nested.request.choice", version = 1 : i64, ancestry = ["nested.request.choice"], sym_name = "nested.request.choice", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
            ^bb0(%26: !lang.payload<"index">):
              "lang.request"(%26) <{origin = "nested.request", version = 1 : i64, ancestry = ["nested.request"], handler = @nested.evaluation, source = @nested.training, boundary = "settled_success"}> : (!lang.payload<"index">) -> ()
              "lang.yield"() <{origin = "nested.request.true.yield", version = 1 : i64, ancestry = ["nested.request.true.yield"]}> : () -> ()
            }, {
            ^bb0(%27: !lang.payload<"index">):
              "lang.yield"() <{origin = "nested.request.false.yield", version = 1 : i64, ancestry = ["nested.request.false.yield"]}> : () -> ()
            }) : (!lang.payload<"bool">, !lang.payload<"index">) -> ()
            "lang.yield"() <{origin = "nested.window.body.yield", version = 1 : i64, ancestry = ["nested.window.body.yield"]}> : () -> ()
          }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">) -> ()
          "lang.yield"() <{origin = "nested.training.branch.yield", version = 1 : i64, ancestry = ["nested.training.branch.yield"]}> : () -> ()
        }, {
        ^bb0(%18: !lang.endpoint<!lang.payload<"record">>, %19: !lang.resource<"scale">, %20: !lang.resource<"products">):
          "lang.requests"(%18, %19, %20) <{origin = "nested.evaluation", version = 1 : i64, ancestry = ["nested.evaluation"], sym_name = "nested.evaluation", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], sources = [@nested.training], request_schema = !lang.payload<"index">, max_outstanding = 1 : i64, max_active = 1 : i64}> ({
          ^bb0(%21: !lang.endpoint<!lang.payload<"record">>, %22: !lang.resource<"scale">, %23: !lang.resource<"products">, %24: !lang.request<!lang.payload<"index">>):
            %25 = "lang.take"(%21) <{origin = "nested.evaluation.take", version = 1 : i64, ancestry = ["nested.evaluation.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
            %26 = "lang.borrow"(%25) <{origin = "nested.evaluation.take.borrow", version = 1 : i64, ancestry = ["nested.evaluation.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
            %27 = "lang.access"(%22, %26) <{origin = "nested.measure.access", version = 1 : i64, ancestry = ["nested.measure.access"], sym_name = "nested.measure.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "protected_current", freshness = "current_at_start", version_selector = "current", profile = "ordinary_owner"}> ({
            ^bb0(%28: !lang.capability<!lang.resource<"scale">, "protected_current">, %29: !lang.payload<"record">):
              %30 = "lang.call"(%28, %29) <{origin = "nested.measure", version = 1 : i64, ancestry = ["nested.measure"], binding = @measure, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "protected_current">, !lang.payload<"record">) -> !lang.payload<"measurement">
              "lang.yield"(%30) <{origin = "nested.measure.access.yield", version = 1 : i64, ancestry = ["nested.measure.access.yield"]}> : (!lang.payload<"measurement">) -> ()
            }) : (!lang.resource<"scale">, !lang.payload<"record">) -> !lang.payload<"measurement">
            %28 = "lang.access"(%23, %27) <{origin = "nested.publish_measurement.access", version = 1 : i64, ancestry = ["nested.publish_measurement.access"], sym_name = "nested.publish_measurement.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
            ^bb0(%29: !lang.capability<!lang.resource<"products">, "owner_update">, %30: !lang.payload<"measurement">):
              %31 = "lang.call"(%29, %30) <{origin = "nested.publish_measurement", version = 1 : i64, ancestry = ["nested.publish_measurement"], binding = @publish_measurement, binding_version = "1"}> : (!lang.capability<!lang.resource<"products">, "owner_update">, !lang.payload<"measurement">) -> !lang.payload<"measurement">
              "lang.yield"(%31) <{origin = "nested.publish_measurement.body.yield", version = 1 : i64, ancestry = ["nested.publish_measurement.body.yield"]}> : (!lang.payload<"measurement">) -> ()
            }) : (!lang.resource<"products">, !lang.payload<"measurement">) -> !lang.payload<"measurement">
            "lang.retire"(%25) <{origin = "nested.evaluation.retire", version = 1 : i64, ancestry = ["nested.evaluation.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
            "lang.yield"() <{origin = "nested.evaluation.body.yield", version = 1 : i64, ancestry = ["nested.evaluation.body.yield"]}> : () -> ()
          }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.resource<"products">) -> ()
          "lang.yield"() <{origin = "nested.evaluation.branch.yield", version = 1 : i64, ancestry = ["nested.evaluation.branch.yield"]}> : () -> ()
        }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.resource<"products">) -> ()
        "lang.yield"() <{origin = "nested.enclosing.body.yield", version = 1 : i64, ancestry = ["nested.enclosing.body.yield"]}> : () -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.resource<"products">) -> ()
      "lang.yield"() <{origin = "nested.consumers.branch.yield", version = 1 : i64, ancestry = ["nested.consumers.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%12: !lang.endpoint<!lang.payload<"record">>, %13: !lang.resource<"scale">, %14: !lang.resource<"scale">, %15: !lang.resource<"products">):
      "lang.stop_after"() <{origin = "nested.stop", version = 1 : i64, ancestry = ["nested.stop"], source = @nested.consumer.scope, target = @nested.producer}> : () -> ()
      "lang.yield"() <{origin = "nested.control.branch.yield", version = 1 : i64, ancestry = ["nested.control.branch.yield"]}> : () -> ()
    }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.resource<"scale">, !lang.resource<"products">) -> ()
  }) : () -> ()
}) : () -> ()
