"builtin.module"() ({
  "exec.plan"() <{accepted_digest = "8c69febe4beefdea6b35b79d7a1ad6221fca5a3cf0e3502554f8b18c3255ff3b", target = "scalar-async-cpu-v1", contract = "scalar-feedback-v1", horizon = 1 : i64, capacity = 1 : i64, stop_rule = "settle_selected", feedback = "dedicated_control_task; receipt -> policy_apply -> ticket_retire -> slot_release"}> ({
    "wf.entity"() <{sym_name = "cursor", source_id = "owner.cursor", kind = "cursor"}> : () -> ()
    "wf.entity"() <{sym_name = "policy", source_id = "owner.policy", kind = "policy"}> : () -> ()
    "wf.entity"() <{sym_name = "publication", source_id = "owner.publication", kind = "publication"}> : () -> ()
    "exec.select"() <{origin = "selector", cursor = @cursor, policy = @policy, selection = "scalar-feedback-v1: cursor + current_policy.bias; source_revision=0", application = "ack_before_next_selection"}> : () -> ()
    "exec.handoff"() <{origin = "data.composed", endpoint = "mapped", reader = "scorer", release = "after_feedback_ack; no_protection_held_while_waiting"}> : () -> ()
    "exec.feedback"() <{origin = "publisher", publication = @publication, policy = @policy, update = "scalar-feedback-v1: policy.bias=score%5; revision+=1", admission = "live_ticket_with_validated_publication_receipt; once_per_key"}> : () -> ()
    "exec.chain"() <{role = "producer", transport = "direct_local_calls_then_bounded_mailbox"}> ({
      "exec.call"() <{origin = "data.double", ancestors = ["data-session", "data.composed"], implementation = @double_v1, binding_version = "1", effect = "pure", role = "map", completion = "await_physical_then_validate"}> : () -> ()
      "exec.call"() <{origin = "data.offset", ancestors = ["data-session", "data.composed", "data.nested"], implementation = @offset_v1, binding_version = "1", effect = "pure", role = "map", completion = "await_physical_then_validate"}> : () -> ()
    }) : () -> ()
    "exec.chain"() <{role = "consumer", transport = "retained_mailbox_slot_through_feedback_ack"}> ({
      "exec.call"() <{origin = "scorer", ancestors = ["data-session"], implementation = @square_v1, binding_version = "1", effect = "pure", role = "score", completion = "await_physical_then_validate"}> : () -> ()
      "exec.call"() <{origin = "publisher", ancestors = ["data-session"], implementation = @receipt_v1, binding_version = "1", effect = "publication", role = "publish", completion = "await_physical_then_validate"}> : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()
