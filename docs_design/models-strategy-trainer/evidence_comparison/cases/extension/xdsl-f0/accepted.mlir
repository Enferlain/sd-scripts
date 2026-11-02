"builtin.module"() ({
  "wf.session"() <{source_id = "data-session", contract = "scalar-feedback-v1", version = 1 : i64, horizon = 2 : i64, stop_rule = "settle_selected"}> ({
    "wf.entity"() <{sym_name = "cursor", source_id = "owner.cursor", kind = "cursor"}> : () -> ()
    "wf.entity"() <{sym_name = "policy", source_id = "owner.policy", kind = "policy"}> : () -> ()
    "wf.entity"() <{sym_name = "publication", source_id = "owner.publication", kind = "publication"}> : () -> ()
    %0 = "wf.endpoint"() <{source_id = "selected"}> : () -> !wf.flow<"record">
    %1 = "wf.endpoint"() <{source_id = "mapped"}> : () -> !wf.flow<"record">
    %2 = "wf.endpoint"() <{source_id = "scored"}> : () -> !wf.flow<"score">
    %3 = "wf.endpoint"() <{source_id = "feedback"}> : () -> !wf.flow<"feedback">
    "data.select"(%3, %0) <{source_id = "selector", cursor = @cursor, policy = @policy, application = "ack_before_next_selection"}> : (!wf.flow<"feedback">, !wf.flow<"record">) -> ()
    "wf.compose"(%0, %1) <{source_id = "data.composed"}> ({
    ^bb0(%4: !wf.flow<"record">):
      %5 = "data.map"(%4) <{source_id = "data.double", implementation = @double_v1, effect = "pure"}> : (!wf.flow<"record">) -> !wf.flow<"record">
      %6 = "wf.local"(%5) <{source_id = "data.nested"}> ({
      ^bb0(%7: !wf.flow<"record">):
        %8 = "data.map"(%7) <{source_id = "data.offset", implementation = @offset_v1, effect = "pure"}> : (!wf.flow<"record">) -> !wf.flow<"record">
        %9 = "pairing.pack"(%8) <{source_id = "pair.pack", implementation = @pack_pair_v1, effect = "read_frozen", resource = @pair_coefficients, resource_version = "1", resource_digest = "49a64717d5d4cb19952e6eac2946415cf6879adacf9908e7d872332d32c6e684"}> : (!wf.flow<"record">) -> !wf.flow<!pairing.vector<2 : i64>>
        %10 = "data.map"(%9) <{source_id = "pair.first", implementation = @first_lane_v1, effect = "pure"}> : (!wf.flow<!pairing.vector<2 : i64>>) -> !wf.flow<"record">
        "wf.yield"(%10) : (!wf.flow<"record">) -> ()
      }) : (!wf.flow<"record">) -> !wf.flow<"record">
      "wf.yield"(%6) : (!wf.flow<"record">) -> ()
    }) : (!wf.flow<"record">, !wf.flow<"record">) -> ()
    "data.score"(%1, %2) <{source_id = "scorer", implementation = @square_v1}> : (!wf.flow<"record">, !wf.flow<"score">) -> ()
    "data.publish"(%2, %3) <{source_id = "publisher", implementation = @receipt_v1, owner = @publication, acknowledgement = "returned_receipt"}> : (!wf.flow<"score">, !wf.flow<"feedback">) -> ()
  }) : () -> ()
}) : () -> ()
