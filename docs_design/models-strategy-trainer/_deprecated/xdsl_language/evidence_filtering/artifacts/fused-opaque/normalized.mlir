"builtin.module"() ({
  "lang.program"() <{origin = "run", version = 1 : i64, ancestry = ["run"], stage = "normalized"}> ({
    %0 = "lang.resource_decl"() <{origin = "config", version = 1 : i64, ancestry = ["run", "config"], sym_name = "config"}> : () -> !lang.resource<"filter_config">
    %1 = "lang.resource_decl"() <{origin = "products", version = 1 : i64, ancestry = ["run", "products"], sym_name = "products"}> : () -> !lang.resource<"filter_products">
    %2 = "lang.stream"() <{origin = "raw", version = 1 : i64, ancestry = ["run", "raw"], sym_name = "raw", horizon = 3 : i64, ready_bound = 2 : i64, admission = "any_with_provenance", owners = [@config], metric = "owner_update_ordinal_within_lineage", comparison_boundary = "item_admission", max_lag = 0 : i64, reproducibility = "unknown", rank_rule = "local", assumptions = []}> : () -> !lang.endpoint<!lang.payload<"record">>
    %3 = "lang.stream"() <{origin = "kept", version = 1 : i64, ancestry = ["run", "kept"], sym_name = "kept", horizon = 3 : i64, ready_bound = 2 : i64, admission = "any_with_provenance", owners = [@config], metric = "owner_update_ordinal_within_lineage", comparison_boundary = "item_admission", max_lag = 0 : i64, reproducibility = "unknown", rank_rule = "local", assumptions = []}> : () -> !lang.endpoint<!lang.payload<"record">>
    "lang.parallel"(%2, %3, %0, %1) <{origin = "activities", version = 1 : i64, ancestry = ["run", "activities"], sym_name = "activities", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
    ^bb0(%4: !lang.endpoint<!lang.payload<"record">>, %5: !lang.endpoint<!lang.payload<"record">>, %6: !lang.resource<"filter_config">, %7: !lang.resource<"filter_products">):
      "lang.producer"(%4, %6) <{origin = "source", version = 1 : i64, ancestry = ["run", "activities", "source"], sym_name = "source", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%8: !lang.resource<"filter_config">, %9: !lang.payload<"index">):
        %10 = "lang.access"(%8, %9) <{origin = "produce.access", version = 1 : i64, ancestry = ["run", "activities", "source", "produce.access"], sym_name = "produce.access", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "snapshot", freshness = "published", version_selector = "initial", profile = "ordinary_owner"}> ({
        ^bb0(%11: !lang.capability<!lang.resource<"filter_config">, "snapshot">, %12: !lang.payload<"index">):
          %13 = "lang.call"(%11, %12) <{origin = "produce", version = 1 : i64, ancestry = ["run", "activities", "source", "produce.access", "produce"], binding = @filter_produce, binding_version = "1"}> : (!lang.capability<!lang.resource<"filter_config">, "snapshot">, !lang.payload<"index">) -> !lang.payload<"record">
          "lang.yield"(%13) <{origin = "produce.access.yield", version = 1 : i64, ancestry = ["run", "activities", "source", "produce.access", "produce.access.yield"]}> : (!lang.payload<"record">) -> ()
        }) : (!lang.resource<"filter_config">, !lang.payload<"index">) -> !lang.payload<"record">
        "lang.yield"(%10) <{origin = "source.body.yield", version = 1 : i64, ancestry = ["run", "activities", "source", "source.body.yield"]}> : (!lang.payload<"record">) -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"filter_config">) -> ()
      "lang.yield"() <{origin = "source.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "source.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%8: !lang.endpoint<!lang.payload<"record">>, %9: !lang.endpoint<!lang.payload<"record">>, %10: !lang.resource<"filter_config">, %11: !lang.resource<"filter_products">):
      "lang.producer"(%9, %8) <{origin = "filtering", version = 1 : i64, ancestry = ["run", "activities", "filtering"], sym_name = "filtering", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%12: !lang.endpoint<!lang.payload<"record">>, %13: !lang.payload<"index">):
        %14 = "lang.take"(%12) <{origin = "filter.take", version = 1 : i64, ancestry = ["run", "activities", "filtering", "filter.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
        %15 = "lang.filter"(%14) <{origin = "filter", version = 1 : i64, ancestry = ["run", "activities", "filtering", "filter"], sym_name = "filter", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
        ^bb0(%16: !lang.payload<"record">):
          %17 = "lang.call"(%16) <{origin = "predicate", version = 1 : i64, ancestry = ["run", "activities", "filtering", "filter", "predicate"], binding = @filter_even, binding_version = "1"}> : (!lang.payload<"record">) -> !lang.payload<"bool">
          "lang.yield"(%17) <{origin = "filter.predicate.yield", version = 1 : i64, ancestry = ["run", "activities", "filtering", "filter", "filter.predicate.yield"]}> : (!lang.payload<"bool">) -> ()
        }) : (!lang.claim<!lang.payload<"record">>) -> !lang.optional<!lang.payload<"record">>
        "lang.yield"(%15) <{origin = "filter.body.yield", version = 1 : i64, ancestry = ["run", "activities", "filtering", "filter.body.yield"]}> : (!lang.optional<!lang.payload<"record">>) -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.endpoint<!lang.payload<"record">>) -> ()
      "lang.yield"() <{origin = "filtering.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "filtering.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%12: !lang.endpoint<!lang.payload<"record">>, %13: !lang.endpoint<!lang.payload<"record">>, %14: !lang.resource<"filter_config">, %15: !lang.resource<"filter_products">):
      "lang.parallel"(%13, %15) <{origin = "consumers", version = 1 : i64, ancestry = ["run", "activities", "consumers"], sym_name = "consumers", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
      ^bb0(%16: !lang.endpoint<!lang.payload<"record">>, %17: !lang.resource<"filter_products">):
        "lang.once"(%16, %17) <{origin = "initial", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial"], sym_name = "initial", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
        ^bb0(%18: !lang.endpoint<!lang.payload<"record">>, %19: !lang.resource<"filter_products">):
          %20 = "lang.take"(%18) <{origin = "first.take", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
          %21 = "lang.borrow"(%20) <{origin = "first.take.borrow", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
          %22 = "lang.once"(%21) <{origin = "first.local", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.local"], sym_name = "first.local", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
          ^bb0(%23: !lang.payload<"record">):
            %24 = "lang.fused_local"(%23) <{origin = "$fused/512ef058f096e52d1c1519e90f1655301bbf9e794dc2469ae5a253944a147c1a", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.local", "$fused/512ef058f096e52d1c1519e90f1655301bbf9e794dc2469ae5a253944a147c1a"], sym_name = "$fused/512ef058f096e52d1c1519e90f1655301bbf9e794dc2469ae5a253944a147c1a", members = ["first.extract", "first.offset"], quiescence = []}> ({
            ^bb0(%25: !lang.payload<"record">):
              %26 = "lang.call"(%25) <{origin = "first.extract", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.local", "$fused/512ef058f096e52d1c1519e90f1655301bbf9e794dc2469ae5a253944a147c1a", "first.extract"], binding = @filter_extract, binding_version = "1"}> : (!lang.payload<"record">) -> !lang.payload<"scalar">
              %27 = "lang.call"(%26) <{origin = "first.offset", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.local", "$fused/512ef058f096e52d1c1519e90f1655301bbf9e794dc2469ae5a253944a147c1a", "first.offset"], binding = @filter_offset, binding_version = "1"}> : (!lang.payload<"scalar">) -> !lang.payload<"scalar">
              "lang.yield"(%27) <{origin = "$fused/512ef058f096e52d1c1519e90f1655301bbf9e794dc2469ae5a253944a147c1a/yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.local", "$fused/512ef058f096e52d1c1519e90f1655301bbf9e794dc2469ae5a253944a147c1a", "$fused/512ef058f096e52d1c1519e90f1655301bbf9e794dc2469ae5a253944a147c1a/yield"]}> : (!lang.payload<"scalar">) -> ()
            }) : (!lang.payload<"record">) -> !lang.payload<"scalar">
            %25 = "lang.call"(%24) <{origin = "first.opaque", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.local", "first.opaque"], binding = @filter_opaque, binding_version = "1"}> : (!lang.payload<"scalar">) -> !lang.payload<"scalar">
            %26 = "lang.call"(%25) <{origin = "first.double", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.local", "first.double"], binding = @filter_double, binding_version = "1"}> : (!lang.payload<"scalar">) -> !lang.payload<"scalar">
            "lang.yield"(%26) <{origin = "first.local.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.local", "first.local.yield"]}> : (!lang.payload<"scalar">) -> ()
          }) : (!lang.payload<"record">) -> !lang.payload<"scalar">
          %23 = "lang.access"(%19, %21, %22) <{origin = "first.write", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.write"], sym_name = "first.write", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
          ^bb0(%24: !lang.capability<!lang.resource<"filter_products">, "owner_update">, %25: !lang.payload<"record">, %26: !lang.payload<"scalar">):
            %27 = "lang.call"(%24, %25, %26) <{origin = "first.publish", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.write", "first.publish"], binding = @filter_publish, binding_version = "1"}> : (!lang.capability<!lang.resource<"filter_products">, "owner_update">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
            "lang.yield"(%27) <{origin = "first.write.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.write", "first.write.yield"]}> : (!lang.payload<"scalar">) -> ()
          }) : (!lang.resource<"filter_products">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
          "lang.retire"(%20) <{origin = "first.retire", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "first.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
          %24 = "lang.constant"() <{origin = "two", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "two"], literal = "2"}> : () -> !lang.payload<"scalar">
          "lang.request"(%24) <{origin = "request", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "request"], handler = @more, source = @initial, boundary = "settled_success"}> : (!lang.payload<"scalar">) -> ()
          "lang.yield"() <{origin = "initial.body.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial", "initial.body.yield"]}> : () -> ()
        }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"filter_products">) -> ()
        "lang.yield"() <{origin = "initial.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "initial.branch.yield"]}> : () -> ()
      }, {
      ^bb0(%18: !lang.endpoint<!lang.payload<"record">>, %19: !lang.resource<"filter_products">):
        "lang.requests"(%18, %19) <{origin = "more", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more"], sym_name = "more", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], sources = [@initial], request_schema = !lang.payload<"scalar">, max_outstanding = 1 : i64, max_active = 1 : i64}> ({
        ^bb0(%20: !lang.endpoint<!lang.payload<"record">>, %21: !lang.resource<"filter_products">, %22: !lang.request<!lang.payload<"scalar">>):
          "lang.repeat"(%20, %21) <{origin = "requested", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested"], sym_name = "requested", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], limit = 2 : i64, carry_count = 0 : i64, policy = @bounded_count}> ({
          ^bb0(%23: !lang.endpoint<!lang.payload<"record">>, %24: !lang.resource<"filter_products">, %25: !lang.payload<"index">):
            %26 = "lang.take"(%23) <{origin = "next.take", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.take"]}> : (!lang.endpoint<!lang.payload<"record">>) -> !lang.claim<!lang.payload<"record">>
            %27 = "lang.borrow"(%26) <{origin = "next.take.borrow", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.take.borrow"]}> : (!lang.claim<!lang.payload<"record">>) -> !lang.payload<"record">
            %28 = "lang.once"(%27) <{origin = "next.local", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.local"], sym_name = "next.local", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = []}> ({
            ^bb0(%29: !lang.payload<"record">):
              %30 = "lang.fused_local"(%29) <{origin = "$fused/997a796d7f1e20e619d6134145ad918b129112be659df4a2211e9a4398ca61df", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.local", "$fused/997a796d7f1e20e619d6134145ad918b129112be659df4a2211e9a4398ca61df"], sym_name = "$fused/997a796d7f1e20e619d6134145ad918b129112be659df4a2211e9a4398ca61df", members = ["next.extract", "next.offset"], quiescence = []}> ({
              ^bb0(%31: !lang.payload<"record">):
                %32 = "lang.call"(%31) <{origin = "next.extract", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.local", "$fused/997a796d7f1e20e619d6134145ad918b129112be659df4a2211e9a4398ca61df", "next.extract"], binding = @filter_extract, binding_version = "1"}> : (!lang.payload<"record">) -> !lang.payload<"scalar">
                %33 = "lang.call"(%32) <{origin = "next.offset", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.local", "$fused/997a796d7f1e20e619d6134145ad918b129112be659df4a2211e9a4398ca61df", "next.offset"], binding = @filter_offset, binding_version = "1"}> : (!lang.payload<"scalar">) -> !lang.payload<"scalar">
                "lang.yield"(%33) <{origin = "$fused/997a796d7f1e20e619d6134145ad918b129112be659df4a2211e9a4398ca61df/yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.local", "$fused/997a796d7f1e20e619d6134145ad918b129112be659df4a2211e9a4398ca61df", "$fused/997a796d7f1e20e619d6134145ad918b129112be659df4a2211e9a4398ca61df/yield"]}> : (!lang.payload<"scalar">) -> ()
              }) : (!lang.payload<"record">) -> !lang.payload<"scalar">
              %31 = "lang.call"(%30) <{origin = "next.opaque", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.local", "next.opaque"], binding = @filter_opaque, binding_version = "1"}> : (!lang.payload<"scalar">) -> !lang.payload<"scalar">
              %32 = "lang.call"(%31) <{origin = "next.double", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.local", "next.double"], binding = @filter_double, binding_version = "1"}> : (!lang.payload<"scalar">) -> !lang.payload<"scalar">
              "lang.yield"(%32) <{origin = "next.local.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.local", "next.local.yield"]}> : (!lang.payload<"scalar">) -> ()
            }) : (!lang.payload<"record">) -> !lang.payload<"scalar">
            %29 = "lang.access"(%24, %27, %28) <{origin = "next.write", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.write"], sym_name = "next.write", reproducibility = "unknown", rank_rule = "local", assumptions = [], quiescence = [], mode = "owner_update", freshness = "owner_protocol", version_selector = "current", profile = "ordinary_owner"}> ({
            ^bb0(%30: !lang.capability<!lang.resource<"filter_products">, "owner_update">, %31: !lang.payload<"record">, %32: !lang.payload<"scalar">):
              %33 = "lang.call"(%30, %31, %32) <{origin = "next.publish", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.write", "next.publish"], binding = @filter_publish, binding_version = "1"}> : (!lang.capability<!lang.resource<"filter_products">, "owner_update">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
              "lang.yield"(%33) <{origin = "next.write.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.write", "next.write.yield"]}> : (!lang.payload<"scalar">) -> ()
            }) : (!lang.resource<"filter_products">, !lang.payload<"record">, !lang.payload<"scalar">) -> !lang.payload<"scalar">
            "lang.retire"(%26) <{origin = "next.retire", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "next.retire"], disposition = "consumed"}> : (!lang.claim<!lang.payload<"record">>) -> ()
            "lang.yield"() <{origin = "requested.iteration.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "requested", "requested.iteration.yield"]}> : () -> ()
          }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"filter_products">) -> ()
          "lang.yield"() <{origin = "more.body.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more", "more.body.yield"]}> : () -> ()
        }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"filter_products">) -> ()
        "lang.yield"() <{origin = "more.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "consumers", "more.branch.yield"]}> : () -> ()
      }) : (!lang.endpoint<!lang.payload<"record">>, !lang.resource<"filter_products">) -> ()
      "lang.yield"() <{origin = "consumer.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "consumer.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%16: !lang.endpoint<!lang.payload<"record">>, %17: !lang.endpoint<!lang.payload<"record">>, %18: !lang.resource<"filter_config">, %19: !lang.resource<"filter_products">):
      "lang.stop_after"() <{origin = "stop.filter", version = 1 : i64, ancestry = ["run", "activities", "stop.filter"], source = @consumers, target = @filtering}> : () -> ()
      "lang.yield"() <{origin = "stop.filter.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "stop.filter.branch.yield"]}> : () -> ()
    }, {
    ^bb0(%20: !lang.endpoint<!lang.payload<"record">>, %21: !lang.endpoint<!lang.payload<"record">>, %22: !lang.resource<"filter_config">, %23: !lang.resource<"filter_products">):
      "lang.stop_after"() <{origin = "stop.source", version = 1 : i64, ancestry = ["run", "activities", "stop.source"], source = @filtering, target = @source}> : () -> ()
      "lang.yield"() <{origin = "stop.source.branch.yield", version = 1 : i64, ancestry = ["run", "activities", "stop.source.branch.yield"]}> : () -> ()
    }) : (!lang.endpoint<!lang.payload<"record">>, !lang.endpoint<!lang.payload<"record">>, !lang.resource<"filter_config">, !lang.resource<"filter_products">) -> ()
  }) : () -> ()
}) : () -> ()
