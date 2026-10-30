"builtin.module"() ({
  "exec.plan"() <{accepted_digest = "7c35e625216db67200999e1b44c411ebdf9c04a5654cf20f4b9855cd46998bc8", target = "scalar-async-cpu-v1", contract = "scalar-feedback-v1", horizon = 2 : i64, capacity = 1 : i64, stop_rule = "settle_selected", feedback = "dedicated_control_task; receipt -> policy_apply -> ticket_retire -> slot_release"}> ({
    "wf.entity"() <{sym_name = "cursor", source_id = "owner.cursor", kind = "cursor"}> : () -> ()
    "wf.entity"() <{sym_name = "policy", source_id = "owner.policy", kind = "policy"}> : () -> ()
    "wf.entity"() <{sym_name = "publication", source_id = "owner.publication", kind = "publication"}> : () -> ()
    "exec.local_signature"() <{origin = "data.double", input_type = !wf.flow<"record">, output_type = !wf.flow<"record">, input_codec = "exact-integer-v1", output_codec = "exact-integer-v1", resources = [], outcomes = ["returned", "invalid_output", "raised"]}> : () -> ()
    "exec.local_signature"() <{origin = "data.offset", input_type = !wf.flow<"record">, output_type = !wf.flow<"record">, input_codec = "exact-integer-v1", output_codec = "exact-integer-v1", resources = [], outcomes = ["returned", "invalid_output", "raised"]}> : () -> ()
    "exec.local_signature"() <{origin = "pair.pack", input_type = !wf.flow<"record">, output_type = !wf.flow<!pairing.vector<2 : i64>>, input_codec = "exact-integer-v1", output_codec = "pair-int2-v1", resources = [{symbol = "pair_coefficients", access = "read_frozen", version = "1", digest = "49a64717d5d4cb19952e6eac2946415cf6879adacf9908e7d872332d32c6e684"}], outcomes = ["returned", "invalid_output", "raised", "negative_input"]}> : () -> ()
    "exec.local_signature"() <{origin = "pair.first", input_type = !wf.flow<!pairing.vector<2 : i64>>, output_type = !wf.flow<"record">, input_codec = "pair-int2-v1", output_codec = "exact-integer-v1", resources = [], outcomes = ["returned", "invalid_output", "raised"]}> : () -> ()
    "exec.select"() <{origin = "selector", cursor = @cursor, policy = @policy, selection = "scalar-feedback-v1: cursor + current_policy.bias; source_revision=0", application = "ack_before_next_selection"}> : () -> ()
    "exec.handoff"() <{origin = "data.composed", endpoint = "mapped", reader = "scorer", release = "after_feedback_ack; no_protection_held_while_waiting"}> : () -> ()
    "exec.feedback"() <{origin = "publisher", publication = @publication, policy = @policy, update = "scalar-feedback-v1: policy.bias=score%5; revision+=1", admission = "live_ticket_with_validated_publication_receipt; once_per_key"}> : () -> ()
    "exec.chain"() <{role = "producer", transport = "direct_local_calls_then_bounded_mailbox"}> ({
      "exec.fused"() <{profile = "ordered-validated-local-v1", origins = ["data.double", "data.offset"]}> ({
        "exec.call"() <{origin = "data.double", ancestors = ["data-session", "data.composed"], implementation = @double_v1, binding_version = "1", effect = "pure", role = "map", completion = "await_physical_then_validate"}> : () -> ()
        "exec.call"() <{origin = "data.offset", ancestors = ["data-session", "data.composed", "data.nested"], implementation = @offset_v1, binding_version = "1", effect = "pure", role = "map", completion = "await_physical_then_validate"}> : () -> ()
      }) : () -> ()
      "exec.call"() <{origin = "pair.pack", ancestors = ["data-session", "data.composed", "data.nested"], implementation = @pack_pair_v1, binding_version = "1", effect = "read_frozen", role = "map", completion = "await_physical_then_validate"}> : () -> ()
      "exec.call"() <{origin = "pair.first", ancestors = ["data-session", "data.composed", "data.nested"], implementation = @first_lane_v1, binding_version = "1", effect = "pure", role = "map", completion = "await_physical_then_validate"}> : () -> ()
    }) : () -> ()
    "exec.chain"() <{role = "consumer", transport = "retained_mailbox_slot_through_feedback_ack"}> ({
      "exec.call"() <{origin = "scorer", ancestors = ["data-session"], implementation = @square_v1, binding_version = "1", effect = "pure", role = "score", completion = "await_physical_then_validate"}> : () -> ()
      "exec.call"() <{origin = "publisher", ancestors = ["data-session"], implementation = @receipt_v1, binding_version = "1", effect = "publication", role = "publish", completion = "await_physical_then_validate"}> : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()
