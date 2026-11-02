//! Independent A correlation/accumulation representation and cold correspondence.
mod dialect;
use dialect::{CallOp, CorrelationContract, DeclarationOp, FusedOp, LocalPlanOp, PlanOp};
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
pub const TARGET: &str = "candidate-owner-cpu-v1";
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
    pub workers: bool,
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
    ensure(
        blocks.len() == 1,
        "one block required per correlation region",
    )?;
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
                    "unsupported correlation operation {}",
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
                format!("{id}: correlation region arguments differ"),
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
fn integers(node: &Info, key: &str) -> Checked<Vec<i64>> {
    serde_json::from_str(node.attr(key))
        .map_err(|_| format!("{}: {key}: integer array required", node.id()))
}
fn one<'a>(infos: &'a [Info], kind: &str) -> Checked<&'a Info> {
    let found: Vec<_> = infos.iter().filter(|n| n.kind == kind).collect();
    ensure(found.len() == 1, format!("one {kind} required"))?;
    Ok(found[0])
}
fn model<'a>(infos: &'a [Info], treatment: &str) -> Checked<&'a Info> {
    let found: Vec<_> = infos
        .iter()
        .filter(|n| n.kind == "model" && n.attr("treatment") == treatment)
        .collect();
    ensure(found.len() == 1, format!("one {treatment} model required"))?;
    Ok(found[0])
}
fn selected(infos: &[Info]) -> Checked<Vec<i64>> {
    let excluded = integers(one(infos, "filter")?, "excluded")?;
    Ok(integers(one(infos, "source")?, "keys")?
        .into_iter()
        .filter(|k| !excluded.contains(k))
        .collect())
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
fn verify(ctx: &Context, infos: &[Info]) -> Checked<()> {
    let root = one(infos, "session")?;
    ensure(infos[0].op == root.op, "root must be ac.session")?;
    root.expect("version", "1")?;
    root.expect("stop_rule", "drain_entered_close_unadmitted")?;
    ensure(
        number(root, "horizon")? > 0,
        "positive source horizon required",
    )?;
    let encoder = model(infos, "frozen")?;
    let learner = model(infos, "trainable")?;
    ensure(
        infos.iter().filter(|n| n.kind == "model").count() == 2,
        "two model participants required",
    )?;
    let unit = one(infos, "unit")?;
    let source = one(infos, "source")?;
    let filter = one(infos, "filter")?;
    let fanout = one(infos, "fanout")?;
    let compose = one(infos, "compose")?;
    let encode = one(infos, "encode")?;
    let join = one(infos, "correlate")?;
    let window = one(infos, "window")?;
    let contribution = one(infos, "contribute")?;
    let advance = one(infos, "advance")?;
    let end = one(infos, "window_end")?;
    let evaluate = one(infos, "evaluate")?;
    let finish = one(infos, "finish")?;
    let identities = [
        encoder.attr("participant"),
        learner.attr("participant"),
        unit.attr("unit"),
        evaluate.attr("owner"),
    ];
    ensure(
        identities.iter().all(|s| !s.is_empty())
            && identities.iter().copied().collect::<HashSet<_>>().len() == 4,
        "distinct participant/unit/observation identities required",
    )?;
    ensure(
        !unit.attr("parameter").is_empty(),
        "unit parameter required",
    )?;
    ensure(
        !window.attr("activity").is_empty() && !identities.contains(&window.attr("activity")),
        "distinct repeat activity required",
    )?;
    ensure(
        operand(ctx, unit, 0) == result(ctx, learner),
        "optimization unit must own the training subject",
    )?;
    for node in [encoder, learner] {
        node.float("initial")?;
    }
    let keys = integers(source, "keys")?;
    let excluded = integers(filter, "excluded")?;
    let active = selected(infos)?;
    ensure(
        !keys.is_empty() && keys.iter().collect::<HashSet<_>>().len() == keys.len(),
        "unique nonempty source samples required",
    )?;
    ensure(
        excluded.iter().collect::<HashSet<_>>().len() == excluded.len()
            && excluded.iter().all(|k| keys.contains(k)),
        "authoritative exclusions must be unique source samples",
    )?;
    ensure(!active.is_empty(), "cannot exclude every source sample")?;
    ensure(
        operand(ctx, filter, 0) == result(ctx, source)
            && operand(ctx, fanout, 0) == result(ctx, filter),
        "filter/fanout source wiring differs",
    )?;
    ensure(
        operand(ctx, compose, 0) == at(ctx, fanout, 0)
            && operand(ctx, encode, 0) == at(ctx, fanout, 1),
        "fanout branch wiring differs",
    )?;
    ensure(
        operand(ctx, encode, 1) == result(ctx, encoder),
        "encoder must use the frozen participant",
    )?;
    encode.expect("protection", "frozen_source_through_physical_completion")?;
    ensure(
        operand(ctx, join, 0) == result(ctx, compose)
            && operand(ctx, join, 1) == result(ctx, encode)
            && operand(ctx, join, 2) == result(ctx, encoder),
        "correlated member/source wiring differs",
    )?;
    join.expect("membership", "same_sample_and_slot_before_any_claim")?;
    join.expect("provenance", "current_source_at_admission")?;
    ensure(
        operand(ctx, window, 0) == result(ctx, join)
            && operand(ctx, window, 1) == result(ctx, learner)
            && operand(ctx, window, 2) == result(ctx, unit),
        "window input/subject/unit wiring differs",
    )?;
    let count = number(window, "count")?;
    ensure(
        count > 0 && active.len() % count == 0,
        "complete positive contribution windows required",
    )?;
    window.expect("accumulation", "sum")?;
    ensure(
        number(contribution, "normalization")? == count,
        "declared loss normalization must match contribution count",
    )?;
    let region = window.op.deref(ctx).get_region(0);
    let block = region.deref(ctx).iter(ctx).next().expect("checked block");
    ensure(
        nested(infos, window.op)
            .iter()
            .map(|n| n.kind)
            .collect::<Vec<_>>()
            == ["contribute", "advance", "window_end"],
        "window must contribute, retain advancement and settle",
    )?;
    ensure(
        (0..3).all(|i| operand(ctx, contribution, i) == block.deref(ctx).get_argument(i)),
        "contribution must use its correlated input and bound subject/unit",
    )?;
    ensure(
        operand(ctx, advance, 0) == result(ctx, contribution)
            && operand(ctx, advance, 1) == block.deref(ctx).get_argument(2)
            && operand(ctx, end, 0) == result(ctx, advance),
        "gradient/advancement/completion chain differs",
    )?;
    advance.expect("authority", "retained_owner_only_after_all_contributions")?;
    ensure(
        advance.float("learning_rate")? > 0.0,
        "positive retained learning rate required",
    )?;
    ensure(
        operand(ctx, evaluate, 0) == at(ctx, window, 0)
            && operand(ctx, evaluate, 1) == result(ctx, learner),
        "evaluation request source/subject differs",
    )?;
    ensure(
        number(evaluate, "after_window")? > 0
            && number(evaluate, "after_window")? <= active.len() / count,
        "reachable evaluation request required",
    )?;
    evaluate.expect("consistency", "current_at_start")?;
    evaluate.expect(
        "protection",
        "current_read_through_physical_return_validation_and_mode_restore",
    )?;
    let finish_op = finish.op.deref(ctx);
    let values: HashSet<_> = (0..finish_op.get_num_operands())
        .map(|i| finish_op.get_operand(i))
        .collect();
    ensure(
        finish_op.get_num_operands() == 2
            && values == HashSet::from([at(ctx, window, 1), result(ctx, evaluate)]),
        "run completion must settle training and requested evaluation separately",
    )?;
    drop(finish_op);
    for node in infos {
        if ["model", "map", "encode", "contribute", "evaluate"].contains(&node.kind) {
            ensure(
                !node.attr("implementation").is_empty(),
                format!("{}: implementation required", node.id()),
            )?;
        }
        if node.kind == "map" {
            ensure(
                ["pure", "may_effect"].contains(&node.attr("effect")),
                "unsupported scalar local effect",
            )?;
        }
        let parent = node
            .parent
            .and_then(|p| infos.iter().find(|n| n.op == p))
            .map(|n| n.kind);
        let expected = match node.kind {
            "session" => None,
            "map" | "local" | "yield" => {
                ensure(
                    matches!(parent, Some("compose" | "local")),
                    format!("{}: local stage outside ordered preprocessing", node.id()),
                )?;
                continue;
            }
            "contribute" | "advance" | "window_end" => Some("window"),
            _ => Some("session"),
        };
        ensure(
            parent == expected,
            format!("{}: invalid semantic containment", node.id()),
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
        "evaluate" => Some(("measure", "current_read")),
        _ => None,
    }
}
fn profile(ctx: &Context, infos: &[Info], manifest: &Manifest) -> Checked<Json> {
    ensure(
        manifest.schema == 1 && manifest.target == TARGET && !manifest.registry.is_empty(),
        "unsupported A target/manifest",
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
                    "{}: incompatible binding @{symbol}: expected {role}/{effect}, provided {binding:?}",
                    node.id()
                ),
            )?;
        }
    }
    let root = one(infos, "session")?;
    let window = one(infos, "window")?;
    let unit = one(infos, "unit")?;
    let contribution = one(infos, "contribute")?;
    let advance = one(infos, "advance")?;
    let evaluate = one(infos, "evaluate")?;
    let encoder = model(infos, "frozen")?;
    let learner = model(infos, "trainable")?;
    let active = selected(infos)?;
    ensure(
        number(window, "count")? == 2
            && active.len() <= 4
            && number(root, "horizon")? >= active.len(),
        "accepted but unsupported A target: two-contribution windows, at most four selected samples within horizon required",
    )?;
    unit.expect("parameter", "weight")?;
    let addresses:Vec<_>=active.iter().enumerate().map(|(index,sample)|json!({"window":index/2,"slot":index%2,"sample":sample,"left":format!("left.{}",index%2),"right":format!("right.{}",index%2)})).collect();
    let models:Vec<_>=infos.iter().filter(|n|n.kind=="model").map(|n|Ok(json!({"name":n.attr("participant"),"origin":n.id(),"initial":n.float("initial")?,"trainable":n.attr("treatment")=="trainable"}))).collect::<Checked<_>>()?;
    let bindings: BTreeMap<_, _> = infos
        .iter()
        .filter_map(|n| role(n).map(|_| (n.id(), n.attr("implementation"))))
        .collect();
    let _ = ctx;
    Ok(
        json!({"models":models,"window":window.attr("activity"),"encoder":encoder.attr("participant"),"learner":learner.attr("participant"),
        "unit":unit.attr("unit"),"parameter":unit.attr("parameter"),"learning_rate":advance.float("learning_rate")?,"excluded":integers(one(infos,"filter")?,"excluded")?,
        "request_after":number(evaluate,"after_window")?,"normalization":number(contribution,"normalization")?,"encode_origin":one(infos,"encode")?.id(),
        "objective_origin":contribution.id(),"evaluate_origin":evaluate.id(),"addresses":addresses,"bindings":bindings,"lowering":manifest.lowering,"workers":manifest.workers}),
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
    profile: &Json,
) -> Ptr<Operation> {
    let plan = new::<PlanOp>(
        ctx,
        2,
        values(&[
            ("target", TARGET),
            ("schema", "1"),
            (
                "completion",
                "settle_source_training_and_requested_physical_work",
            ),
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
            "correlate" => "validate_all_members_and_current_source_before_any_claim",
            "window" => "two_separate_atomic_member_joins_then_sum_retained_contributions",
            "advance" => "consume_complete_window_gradient_then_owner_step_and_reset",
            "evaluate" => "request_cause_retained_separately_from_current_at_start_state",
            "finish" => "settle_all_source_and_consumer_physical_work",
            _ => "accepted_semantic_declaration",
        };
        attrs.insert("protocol".into(), protocol.into());
        new::<DeclarationOp>(ctx, 0, attrs).insert_at_back(destination, ctx);
    }
    for address in profile["addresses"].as_array().unwrap() {
        let mut attrs = values(&[
            ("kind", "address"),
            ("origin", one(infos, "correlate").unwrap().id()),
        ]);
        for key in ["window", "slot", "sample"] {
            attrs.insert(key.into(), address[key].to_string());
        }
        for key in ["left", "right"] {
            attrs.insert(key.into(), address[key].as_str().unwrap().into());
        }
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
            "cold A semantic/source/binding/address/owner-protocol correspondence differs",
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
