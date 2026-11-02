//! Mixed producer/learning feedback contracts and cold correspondence.
mod dialect;
use dialect::shared::{CallOp, CorrelationContract, DeclarationOp, FusedOp, LocalPlanOp, PlanOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::attributes::StringAttr,
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        cloning::IrMapping,
        equivalence::{EqResult, IGNORE_LOC_NAMES, operation_eq},
        inserter::{Inserter, OpInsertionPoint},
        match_rewrite::{MatchRewrite, MatchRewriter, RewriterOrder, apply_match_rewrite},
        rewriter::Rewriter,
    },
    linked_list::ContainsLinkedList,
    op::{Op, op_cast},
    operation::{Operation, verify_operation},
    printable::Printable,
    result::Result as IrResult,
    r#type::Typed,
    value::Value,
};
use rmst_pliron_b::{contract, prepare::parse};
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
type Checked<T> = Result<T, String>;
pub const TARGET: &str = "continuous-owner-cpu-v1";
const COMPLETION: &str = "protected_physical_return_then_validate";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub version: String,
    pub role: String,
    pub effect: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    pub target: String,
    pub registry: String,
    pub lowering: String,
    pub fuse: bool,
    pub bindings: BTreeMap<String, Binding>,
}
struct Info {
    op: Ptr<Operation>,
    parent: Option<Ptr<Operation>>,
    kind: &'static str,
    attributes: BTreeMap<String, String>,
}
impl Info {
    fn id(&self) -> &str {
        &self.attributes["source_id"]
    }
    fn attr(&self, key: &str) -> &str {
        &self.attributes[key]
    }
    fn expect(&self, key: &str, wanted: &str) -> Checked<()> {
        ensure(
            self.attr(key) == wanted,
            format!(
                "{}: {key}: expected {wanted}, provided {}",
                self.id(),
                self.attr(key)
            ),
        )
    }
    fn float(&self, key: &str) -> Checked<f64> {
        self.attr(key)
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("{}: {key}: finite value required", self.id()))
    }
}
fn ensure(ok: bool, message: impl Into<String>) -> Checked<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn operand(ctx: &Context, info: &Info, index: usize) -> Value {
    info.op.deref(ctx).get_operand(index)
}
fn result(ctx: &Context, info: &Info) -> Value {
    info.op.deref(ctx).get_result(0)
}
fn type_name(ctx: &Context, value: Value) -> String {
    value.get_type(ctx).disp(ctx).to_string().trim().into()
}
fn children(ctx: &Context, op: Ptr<Operation>, index: usize) -> Checked<Vec<Ptr<Operation>>> {
    let region = op.deref(ctx).get_region(index);
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    ensure(blocks.len() == 1, "one block required per mixed region")?;
    Ok(blocks[0].deref(ctx).iter(ctx).collect())
}
fn collect(ctx: &Context, root: Ptr<Operation>) -> Checked<Vec<Info>> {
    fn visit(
        ctx: &Context,
        op: Ptr<Operation>,
        parent: Option<Ptr<Operation>>,
        infos: &mut Vec<Info>,
    ) -> Checked<()> {
        let object = Operation::get_op_dyn(op, ctx);
        let shape = op_cast::<dyn CorrelationContract>(&*object)
            .ok_or_else(|| {
                format!(
                    "unsupported mixed operation {}",
                    Operation::get_opid(op, ctx)
                )
            })?
            .shape();
        let attributes = contract::strings(ctx, op)?;
        ensure(
            attributes
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == shape.attributes.iter().copied().collect(),
            format!(
                "{}: missing or unknown {} attributes",
                attributes
                    .get("source_id")
                    .map(String::as_str)
                    .unwrap_or("operation"),
                shape.kind
            ),
        )?;
        let id = &attributes["source_id"];
        ensure(
            !id.is_empty() && infos.iter().all(|n| n.id() != id),
            format!("{id}: empty or duplicate source identity"),
        )?;
        let operation = op.deref(ctx);
        let inputs: Vec<_> = (0..operation.get_num_operands())
            .map(|i| type_name(ctx, operation.get_operand(i)))
            .collect();
        let outputs: Vec<_> = (0..operation.get_num_results())
            .map(|i| type_name(ctx, operation.get_result(i)))
            .collect();
        ensure(
            if shape.variadic {
                !inputs.is_empty() && inputs.iter().all(|ty| ty == shape.inputs[0])
            } else {
                inputs.iter().map(String::as_str).collect::<Vec<_>>() == shape.inputs
            },
            format!("{id}: typed input contract differs"),
        )?;
        ensure(
            outputs.iter().map(String::as_str).collect::<Vec<_>>() == shape.outputs,
            format!("{id}: typed output contract differs"),
        )?;
        ensure(
            operation.regions().count() == usize::from(shape.arguments.is_some()),
            format!("{id}: region count differs"),
        )?;
        if let Some(arguments) = shape.arguments {
            let region = operation.get_region(0);
            let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
            ensure(
                blocks.len() == 1,
                format!("{id}: one region block required"),
            )?;
            let block = blocks[0].deref(ctx);
            let provided: Vec<_> = (0..block.get_num_arguments())
                .map(|i| type_name(ctx, block.get_argument(i)))
                .collect();
            ensure(
                provided.iter().map(String::as_str).collect::<Vec<_>>() == arguments,
                format!("{id}: mixed region arguments differ"),
            )?;
        }
        drop(operation);
        infos.push(Info {
            op,
            parent,
            kind: shape.kind,
            attributes,
        });
        if shape.arguments.is_some() {
            for child in children(ctx, op, 0)? {
                visit(ctx, child, Some(op), infos)?;
            }
        }
        Ok(())
    }
    let mut infos = vec![];
    visit(ctx, root, None, &mut infos)?;
    Ok(infos)
}
fn nested(infos: &[Info], parent: Ptr<Operation>) -> Vec<&Info> {
    infos.iter().filter(|n| n.parent == Some(parent)).collect()
}

fn at(ctx: &Context, node: &Info, index: usize) -> Value {
    node.op.deref(ctx).get_result(index)
}
fn number(node: &Info, key: &str) -> Checked<usize> {
    node.attr(key)
        .parse::<usize>()
        .map_err(|_| format!("{}: {key}: nonnegative integer required", node.id()))
}
fn one<'a>(infos: &'a [Info], kind: &str) -> Checked<&'a Info> {
    let found: Vec<_> = infos.iter().filter(|n| n.kind == kind).collect();
    ensure(found.len() == 1, format!("one {kind} required"))?;
    Ok(found[0])
}
fn local(ctx: &Context, infos: &[Info], container: &Info) -> Checked<()> {
    let region = container.op.deref(ctx).get_region(0);
    let block = region.deref(ctx).iter(ctx).next().expect("checked block");
    let mut previous = block.deref(ctx).get_argument(0);
    let body = nested(infos, container.op);
    ensure(
        body.len() >= 2 && body.last().is_some_and(|n| n.kind == "yield"),
        format!(
            "{}: nonempty ordered local chain and yield required",
            container.id()
        ),
    )?;
    for (index, node) in body.iter().enumerate() {
        ensure(
            operand(ctx, node, 0) == previous,
            format!("{}: ordered local SSA chain differs", node.id()),
        )?;
        if index == body.len() - 1 {
            continue;
        }
        ensure(
            ["map", "local"].contains(&node.kind),
            format!("{}: unsupported local stage", node.id()),
        )?;
        if node.kind == "local" {
            local(ctx, infos, node)?;
        }
        previous = result(ctx, node);
    }
    Ok(())
}
fn argument(ctx: &Context, node: &Info, index: usize) -> Value {
    let region = node.op.deref(ctx).get_region(0);
    let block = region.deref(ctx).iter(ctx).next().expect("checked block");
    block.deref(ctx).get_argument(index)
}

const STOP: &str = "external_at_checkpoint; close_selection; discard_unclaimed; drain_entered";
const POLICY: &str = "post_advance_weight; bias=int(abs(weight)*10)%5; revision+=1; retire_once";
const PROVENANCE: &str = "same_binding_protected_at_production";
fn verify(ctx: &Context, infos: &[Info]) -> Checked<()> {
    let allowed = [
        "mixed_session",
        "producer",
        "cursor",
        "policy_state",
        "select",
        "dynamic_filter",
        "policy_update",
        "publish",
        "repeat",
        "learned",
        "mixed_window_end",
        "checkpoint",
        "mixed_finish",
        "model",
        "unit",
        "fanout",
        "compose",
        "local",
        "map",
        "yield",
        "encode",
        "correlate",
        "contribute",
        "advance",
    ];
    ensure(
        infos.iter().all(|n| allowed.contains(&n.kind)),
        "unsupported mixed semantic operation",
    )?;
    let root = one(infos, "mixed_session")?;
    ensure(infos[0].op == root.op, "root must be mixed.session")?;
    root.expect("version", "1")?;
    root.expect("stop_rule", STOP)?;
    ensure(
        number(root, "horizon")? > 0,
        "positive aggregate horizon required",
    )?;
    let model = one(infos, "model")?;
    let unit = one(infos, "unit")?;
    let producer = one(infos, "producer")?;
    let repeat = one(infos, "repeat")?;
    let cursor = one(infos, "cursor")?;
    let policy = one(infos, "policy_state")?;
    let select = one(infos, "select")?;
    let filter = one(infos, "dynamic_filter")?;
    let update = one(infos, "policy_update")?;
    let publish = one(infos, "publish")?;
    let fanout = one(infos, "fanout")?;
    let compose = one(infos, "compose")?;
    let encode = one(infos, "encode")?;
    let join = one(infos, "correlate")?;
    let contribution = one(infos, "contribute")?;
    let advance = one(infos, "advance")?;
    let learned = one(infos, "learned")?;
    let end = one(infos, "mixed_window_end")?;
    let checkpoint = one(infos, "checkpoint")?;
    let finish = one(infos, "mixed_finish")?;
    let identities = [
        model.attr("participant"),
        unit.attr("unit"),
        cursor.attr("owner"),
        policy.attr("owner"),
        producer.attr("activity"),
        repeat.attr("activity"),
    ];
    ensure(
        identities.iter().all(|s| !s.is_empty())
            && identities.iter().copied().collect::<HashSet<_>>().len() == 6,
        "distinct model/unit/cursor/policy/activity identities required",
    )?;
    model.expect("treatment", "trainable")?;
    model.float("initial")?;
    ensure(
        !unit.attr("parameter").is_empty(),
        "unit parameter required",
    )?;
    ensure(
        operand(ctx, unit, 0) == result(ctx, model),
        "unit must own shared learner",
    )?;
    cursor.expect("start", "0")?;
    policy.expect("bias", "0")?;
    policy.expect("revision", "0")?;
    ensure(
        number(producer, "capacity")? > 0,
        "positive per-port capacity required",
    )?;
    producer.expect("publication", "ordered_bounded_ports_after_model_release")?;
    producer.expect(
        "settlement",
        "close_selection_discard_unclaimed_join_private_physical_work",
    )?;
    ensure(
        operand(ctx, producer, 0) == result(ctx, model)
            && operand(ctx, producer, 1) == at(ctx, repeat, 0)
            && operand(ctx, producer, 2) == at(ctx, checkpoint, 0),
        "producer shared model/feedback/stop dependencies differ",
    )?;
    ensure(
        operand(ctx, select, 0) == result(ctx, cursor)
            && operand(ctx, select, 1) == result(ctx, policy)
            && operand(ctx, select, 2) == result(ctx, update)
            && operand(ctx, select, 3) == argument(ctx, producer, 2),
        "selection cursor/applied policy/acknowledgement/stop differs",
    )?;
    select.expect("application", "ack_before_next_selection")?;
    ensure(
        operand(ctx, filter, 0) == result(ctx, select)
            && operand(ctx, filter, 1) == result(ctx, policy),
        "dynamic filter policy/input differs",
    )?;
    ensure(
        operand(ctx, update, 0) == argument(ctx, producer, 1)
            && operand(ctx, update, 1) == result(ctx, policy),
        "feedback receiver must consume learning evidence for its owned policy",
    )?;
    update.expect("protocol", POLICY)?;
    ensure(
        operand(ctx, fanout, 0) == result(ctx, filter)
            && operand(ctx, compose, 0) == at(ctx, fanout, 0)
            && operand(ctx, encode, 0) == at(ctx, fanout, 1),
        "filtered fanout branch wiring differs",
    )?;
    ensure(
        operand(ctx, encode, 1) == argument(ctx, producer, 0),
        "preprocessing must read shared learner",
    )?;
    encode.expect(
        "protection",
        "same_binding_read_through_physical_completion",
    )?;
    ensure(
        operand(ctx, join, 0) == result(ctx, compose)
            && operand(ctx, join, 1) == result(ctx, encode)
            && operand(ctx, join, 2) == argument(ctx, producer, 0),
        "correlation branch/shared binding differs",
    )?;
    join.expect("membership", "same_ticket_and_ordinal")?;
    join.expect("provenance", PROVENANCE)?;
    ensure(
        operand(ctx, publish, 0) == result(ctx, join),
        "producer must expose correlated output",
    )?;
    ensure(
        operand(ctx, repeat, 0) == at(ctx, producer, 0)
            && operand(ctx, repeat, 1) == result(ctx, model)
            && operand(ctx, repeat, 2) == result(ctx, unit),
        "repeat input/shared subject/unit differs",
    )?;
    let count = number(repeat, "count")?;
    ensure(count > 0, "positive contribution count required")?;
    repeat.expect("accumulation", "sum")?;
    ensure(
        number(contribution, "normalization")? == count,
        "loss normalization must match contribution count",
    )?;
    ensure(
        nested(infos, repeat.op)
            .iter()
            .map(|n| n.kind)
            .collect::<Vec<_>>()
            == ["contribute", "advance", "learned", "mixed_window_end"],
        "repeat must contribute, advance, observe admitted input, then end",
    )?;
    ensure(
        (0..3).all(|i| operand(ctx, contribution, i) == argument(ctx, repeat, i)),
        "contribution input/subject/unit differs",
    )?;
    ensure(
        operand(ctx, advance, 0) == result(ctx, contribution)
            && operand(ctx, advance, 1) == argument(ctx, repeat, 2),
        "advancement gradient/unit differs",
    )?;
    advance.expect("authority", "retained_owner_only_after_all_contributions")?;
    ensure(
        advance.float("learning_rate")? > 0.0,
        "positive learning rate required",
    )?;
    ensure(
        operand(ctx, learned, 0) == result(ctx, advance)
            && operand(ctx, learned, 1) == argument(ctx, repeat, 0)
            && operand(ctx, learned, 2) == argument(ctx, repeat, 1)
            && operand(ctx, end, 0) == result(ctx, learned),
        "post-step feedback must follow advancement and actual admitted input",
    )?;
    learned.expect(
        "observation",
        "returned_advance_current_weight_and_input_associations",
    )?;
    ensure(
        operand(ctx, checkpoint, 0) == at(ctx, repeat, 1)
            && number(checkpoint, "after_windows")? > 0,
        "reachable learning checkpoint required",
    )?;
    checkpoint.expect("protocol", STOP)?;
    ensure(
        operand(ctx, finish, 0) == at(ctx, checkpoint, 1)
            && operand(ctx, finish, 1) == at(ctx, producer, 1),
        "run completion must await checkpoint and independent producer draining",
    )?;
    for node in infos {
        if [
            "model",
            "map",
            "encode",
            "contribute",
            "dynamic_filter",
            "policy_update",
        ]
        .contains(&node.kind)
        {
            ensure(
                !node.attr("implementation").is_empty(),
                format!("{}: implementation required", node.id()),
            )?;
        }
        if node.kind == "map" {
            ensure(
                ["pure", "may_effect"].contains(&node.attr("effect")),
                format!("{}: unsupported scalar local effect", node.id()),
            )?;
        }
        let parent = node
            .parent
            .and_then(|p| infos.iter().find(|n| n.op == p))
            .map(|n| n.kind);
        let expected = match node.kind {
            "mixed_session" => None,
            "model" | "unit" | "producer" | "repeat" | "checkpoint" | "mixed_finish" => {
                Some("mixed_session")
            }
            "contribute" | "advance" | "learned" | "mixed_window_end" => Some("repeat"),
            "map" | "local" | "yield" => {
                ensure(
                    matches!(parent, Some("compose" | "local")),
                    format!("{}: stage outside ordered preprocessing", node.id()),
                )?;
                continue;
            }
            _ => Some("producer"),
        };
        ensure(
            parent == expected,
            format!("{}: invalid mixed containment", node.id()),
        )?;
    }
    local(ctx, infos, compose)
}

fn equivalent(ctx: &Context, left: Ptr<Operation>, right: Ptr<Operation>) -> bool {
    operation_eq(
        ctx,
        &mut IrMapping::default(),
        left,
        right,
        &IGNORE_LOC_NAMES,
    ) == EqResult::Eq
}
pub fn accept(text: &str) -> Checked<String> {
    let ctx = &mut Context::new();
    let root = parse(ctx, text)?;
    verify(ctx, &collect(ctx, root)?)?;
    let encoded = root.deref(ctx).disp(ctx).to_string();
    let decoded = parse(ctx, &encoded)?;
    ensure(
        equivalent(ctx, root, decoded),
        "semantic structural round-trip differs",
    )?;
    let fresh = &mut Context::new();
    let root = parse(fresh, &encoded)?;
    verify(fresh, &collect(fresh, root)?)?;
    Ok(encoded)
}

fn role(node: &Info) -> Option<(&str, &str)> {
    match node.kind {
        "model" => Some(("model", "load")),
        "map" => Some(("map", node.attr("effect"))),
        "encode" => Some(("encode", "read_source")),
        "contribute" => Some(("objective", "build_loss")),
        "dynamic_filter" => Some(("filter", "read_policy")),
        "policy_update" => Some(("policy", "derive_update")),
        _ => None,
    }
}
fn profile(_ctx: &Context, infos: &[Info], manifest: &Manifest) -> Checked<Json> {
    ensure(
        manifest.schema == 1 && manifest.target == TARGET && !manifest.registry.is_empty(),
        "unsupported mixed target/manifest",
    )?;
    ensure(
        ["calls", "inline"].contains(&manifest.lowering.as_str()),
        "unsupported owner lowering",
    )?;
    for node in infos {
        if let Some((role, effect)) = role(node) {
            let symbol = node.attr("implementation");
            let binding = manifest
                .bindings
                .get(symbol)
                .ok_or_else(|| format!("{}: missing binding @{symbol}", node.id()))?;
            ensure(
                !binding.version.is_empty() && binding.role == role && binding.effect == effect,
                format!(
                    "{}: incompatible @{symbol}: expected {role}/{effect}, provided {binding:?}",
                    node.id()
                ),
            )?;
        }
    }
    let root = one(infos, "mixed_session")?;
    let producer = one(infos, "producer")?;
    let repeat = one(infos, "repeat")?;
    let unit = one(infos, "unit")?;
    let checkpoint = one(infos, "checkpoint")?;
    let model = one(infos, "model")?;
    let contribution = one(infos, "contribute")?;
    let advance = one(infos, "advance")?;
    ensure(
        number(root, "horizon")? == 3
            && number(producer, "capacity")? == 1
            && number(repeat, "count")? == 1
            && number(checkpoint, "after_windows")? == 3
            && unit.attr("parameter") == "weight",
        "accepted but unsupported mixed CPU target: horizon3,ports1,one contribution,checkpoint3,weight parameter required",
    )?;
    let bindings: BTreeMap<_, _> = infos
        .iter()
        .filter_map(|n| role(n).map(|_| (n.id(), n.attr("implementation"))))
        .collect();
    Ok(
        json!({"learner":model.attr("participant"),"unit":unit.attr("unit"),"parameter":unit.attr("parameter"),"initial":model.float("initial")?,"learning_rate":advance.float("learning_rate")?,"horizon":number(root,"horizon")?,"after_windows":number(checkpoint,"after_windows")?,"normalization":number(contribution,"normalization")?,"window":repeat.attr("activity"),"model_origin":model.id(),"encode_origin":one(infos,"encode")?.id(),"objective_origin":contribution.id(),"filter_origin":one(infos,"dynamic_filter")?.id(),"policy_origin":one(infos,"policy_update")?.id(),"bindings":bindings,"lowering":manifest.lowering}),
    )
}

fn values(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(k, v)| ((*k).into(), (*v).into()))
        .collect()
}
fn new<T: Op>(
    ctx: &mut Context,
    regions: usize,
    attributes: BTreeMap<String, String>,
) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        T::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        regions,
    );
    for (key, value) in attributes {
        op.deref_mut(ctx).attributes.set(
            Identifier::try_from(key).expect("static field"),
            StringAttr::new(value),
        );
    }
    for index in 0..regions {
        let block = BasicBlock::new(ctx, None, vec![]);
        let region = op.deref(ctx).get_region(index);
        block.insert_at_back(region, ctx);
    }
    op
}
fn block(ctx: &Context, op: Ptr<Operation>, index: usize) -> Ptr<BasicBlock> {
    op.deref(ctx)
        .get_region(index)
        .deref(ctx)
        .iter(ctx)
        .next()
        .expect("fresh block")
}
struct Fuse;
impl MatchRewrite for Fuse {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        Operation::get_op::<LocalPlanOp>(op, ctx).is_some()
    }
    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> IrResult<()> {
        let mut run = vec![];
        let mut runs = vec![];
        for call in children(ctx, op, 0)
            .expect("checked local plan")
            .into_iter()
            .map(Some)
            .chain([None])
        {
            if let Some(call) = call
                && contract::strings(ctx, call)
                    .is_ok_and(|attrs| attrs.get("effect").is_some_and(|e| e == "pure"))
            {
                run.push(call);
                continue;
            }
            if run.len() > 1 {
                runs.push(std::mem::take(&mut run));
            } else {
                run.clear();
            }
        }
        for run in runs {
            let group =
                new::<FusedOp>(ctx, 1, values(&[("profile", "ordered-validated-local-v1")]));
            rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(run[0]));
            rewriter.insert_operation(ctx, group);
            let destination = block(ctx, group, 0);
            for call in run {
                rewriter.move_operation(ctx, call, OpInsertionPoint::AtBlockEnd(destination));
            }
        }
        Ok(())
    }
}

fn recipe(
    ctx: &mut Context,
    semantic: Ptr<Operation>,
    infos: &[Info],
    manifest: &Manifest,
    _profile: &Json,
) -> Ptr<Operation> {
    let plan = new::<PlanOp>(
        ctx,
        2,
        values(&[
            ("target", TARGET),
            ("schema", "1"),
            ("completion", "settle_checkpoint_and_producer_physical_work"),
            ("manifest", &serde_json::to_string(manifest).unwrap()),
        ]),
    );
    semantic.insert_at_back(block(ctx, plan, 0), ctx);
    let destination = block(ctx, plan, 1);
    for node in infos {
        if node.kind == "map" {
            continue;
        }
        let mut attrs = node.attributes.clone();
        attrs.insert("origin".into(), node.id().into());
        attrs.insert("kind".into(), node.kind.into());
        if let Some((role, effect)) = role(node) {
            let binding = &manifest.bindings[node.attr("implementation")];
            attrs.extend(values(&[
                ("role", role),
                ("effect", effect),
                ("binding_version", &binding.version),
                ("completion", COMPLETION),
            ]));
        }
        let protocol = match node.kind {
            "producer" => "aggregate3_ports1_ordered_publish_after_read_release_stop_then_drain",
            "correlate" => {
                "same_ticket_both_members_and_actual_production_binding_before_any_claim"
            }
            "repeat" => "one_retained_contribution_then_advance_reset_then_feedback",
            "learned" => "actual_admitted_input_value_link_not_diagnostic_archive",
            "policy_update" => "producer_owned_policy_validate_ack_retire_once",
            "checkpoint" => "three_returned_updates_then_explicit_external_stop",
            "mixed_finish" => "checkpoint_settlement_and_independent_producer_physical_drain",
            _ => "accepted_semantic_declaration",
        };
        if let Some(parent) = node.parent {
            attrs.insert(
                "parent_origin".into(),
                infos.iter().find(|n| n.op == parent).unwrap().id().into(),
            );
        }
        attrs.insert("protocol".into(), protocol.into());
        new::<DeclarationOp>(ctx, 0, attrs).insert_at_back(destination, ctx);
    }
    let local = new::<LocalPlanOp>(
        ctx,
        1,
        values(&[("role", "producer"), ("completion", COMPLETION)]),
    );
    local.insert_at_back(destination, ctx);
    for node in infos.iter().filter(|n| n.kind == "map") {
        let binding = &manifest.bindings[node.attr("implementation")];
        new::<CallOp>(
            ctx,
            0,
            values(&[
                ("origin", node.id()),
                ("implementation", node.attr("implementation")),
                ("role", "map"),
                ("effect", node.attr("effect")),
                ("binding_version", &binding.version),
                ("completion", COMPLETION),
            ]),
        )
        .insert_at_back(block(ctx, local, 0), ctx);
    }
    if manifest.fuse {
        apply_match_rewrite(ctx, &mut Fuse, RewriterOrder::default(), plan)
            .expect("checked scalar local rewrite");
    }
    plan
}

pub fn prepare(text: &str, saved: Option<&str>, manifest: Manifest) -> Checked<Json> {
    let ctx = &mut Context::new();
    let semantic = parse(ctx, text)?;
    let infos = collect(ctx, semantic)?;
    verify(ctx, &infos)?;
    let mut profile = profile(ctx, &infos, &manifest)?;
    let expected = recipe(ctx, semantic, &infos, &manifest, &profile);
    verify_operation(expected, ctx).map_err(|e| e.disp(ctx).to_string())?;
    let printed = expected.deref(ctx).disp(ctx).to_string();
    let decoded = parse(ctx, &printed)?;
    ensure(
        equivalent(ctx, expected, decoded),
        "prepared structural round-trip differs",
    )?;
    if let Some(saved) = saved {
        let decoded = parse(ctx, saved)?;
        ensure(
            equivalent(ctx, expected, decoded),
            "cold mixed semantic/source/binding/policy/access/lifetime correspondence differs",
        )?;
    }
    let local = children(ctx, expected, 1)?
        .into_iter()
        .find(|op| Operation::get_op::<LocalPlanOp>(*op, ctx).is_some())
        .expect("one local plan");
    let groups: Vec<Vec<String>> = children(ctx, local, 0)?
        .into_iter()
        .map(|op| {
            if Operation::get_op::<FusedOp>(op, ctx).is_some() {
                children(ctx, op, 0)
                    .unwrap()
                    .iter()
                    .map(|c| contract::strings(ctx, *c).unwrap()["origin"].clone())
                    .collect()
            } else {
                vec![contract::strings(ctx, op).unwrap()["origin"].clone()]
            }
        })
        .collect();
    profile["local_groups"] = json!(groups);
    Ok(
        json!({"schema":1,"pliron":"0.20.0","target":TARGET,"accepted":text,"prepared":saved.unwrap_or(&printed),
        "accepted_sha256":format!("{:x}",Sha256::digest(text.as_bytes())),"manifest":manifest,"profile":profile}),
    )
}
