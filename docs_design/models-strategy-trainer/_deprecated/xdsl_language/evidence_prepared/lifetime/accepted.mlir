"builtin.module"() ({
  "lang.program"() <{origin = "run", version = 1 : i64, ancestry = ["run"], stage = "accepted"}> ({
    %0 = "lang.resource_decl"() <{origin = "encoder", version = 1 : i64, ancestry = ["encoder"], sym_name = "encoder"}> : () -> !lang.resource<"scale">
    %1 = "lang.resource_decl"() <{origin = "model", version = 1 : i64, ancestry = ["model"], sym_name = "model"}> : () -> !lang.resource<"scale">
    %2 = "lang.stream"() <{origin = "records", version = 1 : i64, ancestry = ["records"], sym_name = "records", horizon = 3 : i64, ready_bound = 2 : i64, admission = "any_with_provenance", owners = [@encoder], metric = "owner_update_ordinal_within_lineage", comparison_boundary = "item_admission", max_lag = 0 : i64, reproducibility = "unknown", rank_rule = "local", assumptions = []}> : () -> !lang.endpoint<!lang.payload<"record">>
    "lang.parallel"(%2, %0, %1) <{origin = "activities", version = 1 : i64, ancestry = ["activities"], sym_name = "activities", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
    ^bb0(%3: !lang.endpoint<!lang.payload<"record">>, %4: !lang.resource<"scale">, %5: !lang.resource<"scale">):
      "lang.producer"(%3, %4) <{origin = "producer", version = 1 : i64, ancestry = ["producer"], sym_name = "producer", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%6: !lang.resource<"scale">, %7: !lang.payload<"index">):
        %8 = "lang.access"(%6, %7) <{origin = "encode.access", version = 1 : i64, ancestry = ["encode.access"], sym_name = "encode.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "snapshot", freshness = "published", version_selector = "initial", profile = "ordinary_owner"}> ({
        ^bb0(%9: !lang.capability<!lang.resource<"scale">, "snapshot">, %10: !lang.payload<"index">):
          %11 = "lang.call"(%9, %10) <{origin = "encode", version = 1 : i64, ancestry = ["encode"], binding = @encode, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "snapshot">, !lang.payload<"index">) -> !lang.payload<"record">
          "lang.yield"(%11) <{origin = "encode.access.yield", version = 1 : i64, ancestry = ["encode.access.yield"]}> : (!lang.payload<"record">) -> ()
        }) : (!lang.resource<"scale">, !lang.payload<"index">) -> !lang.payload<"record">
        "lang.yield"(%8) <{origin = "production.body.yield", version = 1 : i64, ancestry = ["production.body.yield"]}> : (!lang.payload<"record">) -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">) -> ()
      "lang.yield"() <{origin = "producer.branch.yield", version = 1 : i64, ancestry = ["producer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%6: !lang.endpoint<!lang.payload<"record">>, %7: !lang.resource<"scale">, %8: !lang.resource<"scale">):
      "lang.parallel"(%6, %8) <{origin = "consumers", version = 1 : i64, ancestry = ["consumers"], sym_name = "consumers", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%9: !lang.endpoint<!lang.payload<"record">>, %10: !lang.resource<"scale">):
        "lang.repeat"(%9, %10) <{origin = "training", version = 1 : i64, ancestry = ["training"], sym_name = "training", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 2 : i64, carry_count = 0 : i64, policy = @bounded_count}> ({
        ^bb0(%11: !lang.endpoint<!lang.payload<"record">>, %12: !lang.resource<"scale">, %13: !lang.payload<"index">):
          %14 = "lang.constant"() <{origin = "zero", version = 1 : i64, ancestry = ["zero"], literal = "0.0"}> : () -> !lang.payload<"scalar">
          %15 = "lang.repeat"(%11, %12, %14) <{origin = "members", version = 1 : i64, ancestry = ["members"], sym_name = "members", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 2 : i64, carry_count = 1 : i64, policy = @bounded_count}> ({
          ^bb0(%16: !lang.endpoint<!lang.payload<"record">>, %17: !lang.resource<"scale">, %18: !lang.payload<"index">, %19: !lang.payload<"scalar">):
            %20 = "lang.take"(%16) <{origin = "training.take", version = 1 : i64, ancestry = ["training.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
            %21 = "lang.borrow"(%20) <{origin = "training.take.borrow", version = 1 : i64, ancestry = ["training.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
            %22 = "lang.access"(%17, %21, %19) <{origin = "gradient.access", version = 1 : i64, ancestry = ["gradient.access"], sym_name = "gradient.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "protected_current", freshness = "current_at_start", version_selector = "current", profile = "ordinary_owner"}> ({
            ^bb0(%23: !lang.capability<!lang.resource<"scale">, "protected_current">, %24: !lang.payload<"record">, %25: !lang.payload<"scalar">):
              %26 = "lang.call"(%23, %24, %25) <{origin = "gradient", version = 1 : i64, ancestry = ["gradient"], binding = @gradient, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "protected_current">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
              "lang.yield"(%26) <{origin = "gradient.access.yield", version = 1 : i64, ancestry = ["gradient.access.yield"]}> : (!lang.payload<"scalar">) -> ()
            }) : (!lang.resource<"scale">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
            "lang.retire"(%20) <{origin = "training.retire", version = 1 : i64, ancestry = ["training.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
            "lang.yield"(%22) <{origin = "members.body.yield", version = 1 : i64, ancestry = ["members.body.yield"]}> : (!lang.payload<"scalar">) -> ()
          }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
          %16 = "lang.access"(%12, %15) <{origin = "update.access", version = 1 : i64, ancestry = ["update.access"], sym_name = "update.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
          ^bb0(%17: !lang.capability<!lang.resource<"scale">, "owner_update">, %18: !lang.payload<"scalar">):
            %19 = "lang.call"(%17, %18) <{origin = "advance", version = 1 : i64, ancestry = ["advance"], binding = @advance, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "owner_update">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
            "lang.yield"(%19) <{origin = "update.access.yield", version = 1 : i64, ancestry = ["update.access.yield"]}> : (!lang.payload<"scalar">) -> ()
          }) : (!lang.resource<"scale">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
          %17 = "lang.constant"() <{origin = "second.window", version = 1 : i64, ancestry = ["second.window"], literal = "1"}> : () -> !lang.payload<"index">
          %18 = "lang.call"(%13, %17) <{origin = "evaluation.due", version = 1 : i64, ancestry = ["evaluation.due"], binding = @equal, binding_version = "1"}> : (!lang.payload<"index">, !lang.payload<"index">) -> !lang.payload<"bool">
          "lang.choice"(%18, %13) <{origin = "request.choice", version = 1 : i64, ancestry = ["request.choice"], sym_name = "request.choice", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
          ^bb0(%19: !lang.payload<"index">):
            "lang.request"(%19) <{origin = "request", version = 1 : i64, ancestry = ["request"], handler = @evaluation, source = @training, boundary = "settled_success"}> : (!lang.payload<"index">) -> ()
            "lang.yield"() <{origin = "request.true.yield", version = 1 : i64, ancestry = ["request.true.yield"]}> : () -> ()
          }, {
          ^bb0(%20: !lang.payload<"index">):
            "lang.yield"() <{origin = "request.false.yield", version = 1 : i64, ancestry = ["request.false.yield"]}> : () -> ()
          }) : (!lang.payload<"bool">, !lang.payload<"index">) -> ()
          "lang.yield"() <{origin = "window.body.yield", version = 1 : i64, ancestry = ["window.body.yield"]}> : () -> ()
        }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">) -> ()
        "lang.yield"() <{origin = "training.branch.yield", version = 1 : i64, ancestry = ["training.branch.yield"]}> : () -> ()
      }, {
      ^bb0(%11: !lang.endpoint<!lang.payload<"record">>, %12: !lang.resource<"scale">):
        "lang.requests"(%11, %12) <{origin = "evaluation", version = 1 : i64, ancestry = ["evaluation"], sym_name = "evaluation", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], sources = [@training], request_schema = !lang.payload<"index">, max_outstanding = 1 : i64, max_active = 1 : i64}> ({
        ^bb0(%13: !lang.endpoint<!lang.payload<"record">>, %14: !lang.resource<"scale">, %15: !lang.request<!lang.payload<"index">>):
          %16 = "lang.take"(%13) <{origin = "evaluation.take", version = 1 : i64, ancestry = ["evaluation.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
          %17 = "lang.borrow"(%16) <{origin = "evaluation.take.borrow", version = 1 : i64, ancestry = ["evaluation.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
          %18 = "lang.access"(%14, %17) <{origin = "measure.access", version = 1 : i64, ancestry = ["measure.access"], sym_name = "measure.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "protected_current", freshness = "current_at_start", version_selector = "current", profile = "ordinary_owner"}> ({
          ^bb0(%19: !lang.capability<!lang.resource<"scale">, "protected_current">, %20: !lang.payload<"record">):
            %21 = "lang.call"(%19, %20) <{origin = "measure", version = 1 : i64, ancestry = ["measure"], binding = @measure, binding_version = "1"}> : (!lang.capability<!lang.resource<"scale">, "protected_current">, !lang.payload<"record">) -> !lang.payload<"measurement">
            "lang.yield"(%21) <{origin = "measure.access.yield", version = 1 : i64, ancestry = ["measure.access.yield"]}> : (!lang.payload<"measurement">) -> ()
          }) : (!lang.resource<"scale">, !lang.payload<"record">) -> !lang.payload<"measurement">
          "lang.retire"(%16) <{origin = "evaluation.retire", version = 1 : i64, ancestry = ["evaluation.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
          "lang.yield"() <{origin = "evaluation.body.yield", version = 1 : i64, ancestry = ["evaluation.body.yield"]}> : () -> ()
        }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">) -> ()
        "lang.yield"() <{origin = "evaluation.branch.yield", version = 1 : i64, ancestry = ["evaluation.branch.yield"]}> : () -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">) -> ()
      "lang.yield"() <{origin = "consumers.branch.yield", version = 1 : i64, ancestry = ["consumers.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%9: !lang.endpoint<!lang.payload<"record">>, %10: !lang.resource<"scale">, %11: !lang.resource<"scale">):
      "lang.stop_after"() <{origin = "stop", version = 1 : i64, ancestry = ["stop"], source = @consumers, target = @producer}> : () -> ()
      "lang.yield"() <{origin = "control.branch.yield", version = 1 : i64, ancestry = ["control.branch.yield"]}> : () -> ()
    }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"scale">, !lang.resource<"scale">) -> ()
  }) : () -> ()
}) : () -> ()
