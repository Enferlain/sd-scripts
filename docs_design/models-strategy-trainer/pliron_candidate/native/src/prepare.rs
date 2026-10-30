use std::collections::{BTreeMap, BTreeSet};

use pliron::{
    basic_block::BasicBlock,
    builtin::attributes::{GivenNamesAttr, StringAttr},
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        cloning::IrMapping,
        equivalence::{EqResult, IGNORE_LOC_NAMES, operation_eq, operation_hash},
        inserter::{Inserter, OpInsertionPoint},
        match_rewrite::{MatchRewrite, MatchRewriter, RewriterOrder, apply_match_rewrite},
        rewriter::Rewriter,
    },
    linked_list::ContainsLinkedList,
    op::{Op, op_cast},
    operation::{Operation, verify_operation},
    parsable::parse_from_str,
    pass::{Analysis, AnalysisManager, Pass, PassResult, Passes},
    printable::Printable,
    result::Result as IrResult,
    r#type::Typed,
    value::Value,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::dialect::*;

pub type Checked<T> = Result<T, String>;
pub const TARGET: &str = "scalar-async-cpu-v1";
const CONTRACT: &str = "scalar-feedback-v1";
const OUTCOMES: [&str; 3] = ["returned", "invalid_output", "raised"];
const COMPLETION: &str = "await_physical_then_validate";
const CODEC: &str = "exact-integer-v1";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub version: String,
    pub role: String,
    pub effect: String,
    pub outcomes: Vec<String>,
    pub codec: String,
    pub resources: Vec<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    pub registry: String,
    pub bindings: BTreeMap<String, Binding>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Call {
    pub origin: String,
    pub symbol: String,
    pub role: String,
    pub effect: String,
    pub ancestors: Vec<String>,
}
#[derive(Clone, Debug)]
struct Meaning {
    horizon: usize,
    producer: Vec<Call>,
    consumer: Vec<Call>,
}
#[derive(Debug, Serialize)]
pub struct Artifact {
    pub schema: u32,
    pub pliron: &'static str,
    pub accepted: String,
    pub prepared: String,
    pub accepted_sha256: String,
    pub target: &'static str,
    pub horizon: usize,
    pub capacity: usize,
    pub producer: Vec<Vec<Call>>,
    pub consumer: Vec<Call>,
    pub manifest: Manifest,
    pub analysis: Vec<(usize, usize, usize)>,
}

pub fn parse(ctx: &mut Context, text: &str) -> Checked<Ptr<Operation>> {
    let op = parse_from_str(Operation::top_level_parser(), ctx, text)
        .map_err(|e| format!("parse: {e}"))?;
    verify_operation(op, ctx).map_err(|e| format!("structural verification: {}", e.disp(ctx)))?;
    no_successors(ctx, op)?;
    Ok(op)
}
fn no_successors(ctx: &Context, op: Ptr<Operation>) -> Checked<()> {
    ensure(
        op.deref(ctx).get_num_successors() == 0,
        "this workflow target has no CFG successor edges",
    )?;
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            for child in block.deref(ctx).iter(ctx) {
                no_successors(ctx, child)?;
            }
        }
    }
    Ok(())
}
fn is<T: Op>(ctx: &Context, op: Ptr<Operation>) -> bool {
    Operation::get_op::<T>(op, ctx).is_some()
}
fn ensure(ok: bool, message: impl Into<String>) -> Checked<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn attrs(ctx: &Context, op: Ptr<Operation>) -> Checked<BTreeMap<String, String>> {
    op.deref(ctx)
        .attributes
        .0
        .iter()
        .filter(|(_, a)| !a.is::<GivenNamesAttr>())
        .map(|(key, value)| {
            value
                .downcast_ref::<StringAttr>()
                .map(|v| (key.to_string(), v.as_str().to_string()))
                .ok_or_else(|| format!("{key}: expected string attribute"))
        })
        .collect()
}
fn attr(ctx: &Context, op: Ptr<Operation>, key: &str) -> Checked<String> {
    attrs(ctx, op)?
        .get(key)
        .filter(|s| !s.is_empty())
        .cloned()
        .ok_or_else(|| format!("{}: missing {key}", Operation::get_opid(op, ctx)))
}
fn keys(ctx: &Context, op: Ptr<Operation>, names: &[&str]) -> Checked<()> {
    ensure(
        attrs(ctx, op)?
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == names.iter().copied().collect(),
        format!(
            "{}: missing or unknown semantic attributes",
            Operation::get_opid(op, ctx)
        ),
    )
}
fn eq_attr(ctx: &Context, op: Ptr<Operation>, key: &str, expected: &str) -> Checked<()> {
    ensure(
        attr(ctx, op, key)? == expected,
        format!("{}: incompatible {key}", Operation::get_opid(op, ctx)),
    )
}
fn body(ctx: &Context, op: Ptr<Operation>, index: usize) -> Checked<Ptr<BasicBlock>> {
    let region = op.deref(ctx).get_region(index);
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    ensure(blocks.len() == 1, "one region block required")?;
    Ok(blocks[0])
}
fn children(ctx: &Context, op: Ptr<Operation>, index: usize) -> Checked<Vec<Ptr<Operation>>> {
    Ok(body(ctx, op, index)?.deref(ctx).iter(ctx).collect())
}
fn operands(ctx: &Context, op: Ptr<Operation>) -> Vec<Value> {
    (0..op.deref(ctx).get_num_operands())
        .map(|i| op.deref(ctx).get_operand(i))
        .collect()
}
fn result(ctx: &Context, op: Ptr<Operation>) -> Value {
    op.deref(ctx).get_result(0)
}
fn source(ctx: &Context, op: Ptr<Operation>, seen: &mut BTreeSet<String>) -> Checked<String> {
    let id = attr(ctx, op, "source_id")?;
    ensure(
        seen.insert(id.clone()),
        format!("{id}: duplicate source identity"),
    )?;
    Ok(id)
}
fn closed(ctx: &Context, op: Ptr<Operation>, index: usize) -> Checked<()> {
    ensure(
        body(ctx, op, index)?.deref(ctx).get_num_arguments() == 0,
        "closed graph region required",
    )
}
fn stage(
    ctx: &Context,
    op: Ptr<Operation>,
    role: &str,
    ancestors: &[String],
    seen: &mut BTreeSet<String>,
) -> Checked<Call> {
    let origin = source(ctx, op, seen)?;
    let effect = if role == "publish" {
        "publication"
    } else {
        "pure"
    };
    eq_attr(ctx, op, "effect", effect)?;
    Ok(Call {
        origin,
        symbol: attr(ctx, op, "implementation")?,
        role: role.into(),
        effect: effect.into(),
        ancestors: ancestors.into(),
    })
}
fn local(
    ctx: &Context,
    op: Ptr<Operation>,
    ancestors: &[String],
    seen: &mut BTreeSet<String>,
) -> Checked<Vec<Call>> {
    let block = body(ctx, op, 0)?;
    let record = RecordType::get(ctx).into();
    ensure(
        block.deref(ctx).get_num_arguments() == 1
            && block.deref(ctx).get_argument(0).get_type(ctx) == record,
        "local region needs one record argument",
    )?;
    let ops = children(ctx, op, 0)?;
    ensure(
        ops.len() >= 2 && is::<YieldOp>(ctx, *ops.last().unwrap()),
        "local region needs a nonempty chain and final yield",
    )?;
    let mut previous = block.deref(ctx).get_argument(0);
    let mut calls = Vec::new();
    for child in &ops[..ops.len() - 1] {
        ensure(
            operands(ctx, *child) == [previous]
                && child.deref(ctx).get_num_results() == 1
                && result(ctx, *child).get_type(ctx) == record,
            "broken ordered local record chain",
        )?;
        if is::<LocalOp>(ctx, *child) {
            keys(ctx, *child, &["source_id"])?;
            let mut path = ancestors.to_vec();
            path.push(source(ctx, *child, seen)?);
            calls.extend(local(ctx, *child, &path, seen)?);
        } else {
            let object = Operation::get_op_dyn(*child, ctx);
            let interface = op_cast::<dyn LocalStageInterface>(&*object)
                .ok_or("unsupported local operation/interface")?;
            ensure(
                interface.cardinality() == "one-to-one",
                "unsupported cardinality",
            )?;
            keys(ctx, *child, &["source_id", "implementation", "effect"])?;
            calls.push(stage(ctx, *child, "map", ancestors, seen)?);
        }
        previous = result(ctx, *child);
    }
    let yield_op = *ops.last().unwrap();
    keys(ctx, yield_op, &[])?;
    ensure(
        operands(ctx, yield_op) == [previous],
        "yield must expose the final local endpoint",
    )?;
    Ok(calls)
}
fn meaning(ctx: &Context, session: Ptr<Operation>) -> Checked<Meaning> {
    ensure(
        is::<SessionOp>(ctx, session),
        "accepted root must be wf.session",
    )?;
    keys(
        ctx,
        session,
        &["source_id", "contract", "version", "horizon", "stop_rule"],
    )?;
    eq_attr(ctx, session, "contract", CONTRACT)?;
    eq_attr(ctx, session, "version", "1")?;
    eq_attr(ctx, session, "stop_rule", "settle_selected")?;
    let horizon = attr(ctx, session, "horizon")?
        .parse::<usize>()
        .map_err(|_| "invalid admission horizon")?;
    ensure(horizon > 0, "positive admission horizon required")?;
    closed(ctx, session, 0)?;
    let mut seen = BTreeSet::new();
    let root = source(ctx, session, &mut seen)?;
    let mut entities = BTreeMap::new();
    let mut flows = BTreeMap::new();
    let (mut select, mut compose, mut score, mut publish) = (None, None, None, None);
    for op in children(ctx, session, 0)? {
        if is::<EntityOp>(ctx, op) {
            keys(ctx, op, &["source_id", "kind"])?;
            source(ctx, op, &mut seen)?;
            let kind = attr(ctx, op, "kind")?;
            ensure(
                ["cursor", "policy", "publication"].contains(&kind.as_str())
                    && result(ctx, op).get_type(ctx) == EntityType::get(ctx).into()
                    && entities.insert(kind, result(ctx, op)).is_none(),
                "invalid/duplicate entity",
            )?;
        } else if is::<FlowOp>(ctx, op) {
            keys(ctx, op, &["source_id"])?;
            let id = source(ctx, op, &mut seen)?;
            ensure(
                flows.insert(id, result(ctx, op)).is_none(),
                "duplicate flow",
            )?;
        } else {
            let slot = if is::<SelectOp>(ctx, op) {
                &mut select
            } else if is::<ComposeOp>(ctx, op) {
                &mut compose
            } else if is::<ScoreOp>(ctx, op) {
                &mut score
            } else if is::<PublishOp>(ctx, op) {
                &mut publish
            } else {
                return Err("unsupported session operation".into());
            };
            ensure(slot.replace(op).is_none(), "duplicate workflow role")?;
        }
    }
    ensure(
        entities.len() == 3 && flows.len() == 4,
        "three entities and four flow endpoints required",
    )?;
    for (id, ty) in [
        ("selected", RecordType::get(ctx).into()),
        ("mapped", RecordType::get(ctx).into()),
        ("scored", ScoreType::get(ctx).into()),
        ("feedback", FeedbackType::get(ctx).into()),
    ] {
        ensure(
            flows.get(id).is_some_and(|v| v.get_type(ctx) == ty),
            format!("{id}: missing/incompatible endpoint"),
        )?;
    }
    let (select, compose, score, publish) = (
        select.ok_or("missing selector")?,
        compose.ok_or("missing composition")?,
        score.ok_or("missing scorer")?,
        publish.ok_or("missing publisher")?,
    );
    keys(ctx, select, &["source_id", "application"])?;
    source(ctx, select, &mut seen)?;
    eq_attr(ctx, select, "application", "ack_before_next_selection")?;
    ensure(
        operands(ctx, select)
            == [
                entities["cursor"],
                entities["policy"],
                flows["feedback"],
                flows["selected"],
            ],
        "selector entity/feedback/output wiring differs",
    )?;
    keys(ctx, compose, &["source_id"])?;
    let path = vec![root.clone(), source(ctx, compose, &mut seen)?];
    ensure(
        operands(ctx, compose) == [flows["selected"], flows["mapped"]],
        "composition endpoint wiring differs",
    )?;
    let producer = local(ctx, compose, &path, &mut seen)?;
    keys(ctx, score, &["source_id", "implementation", "effect"])?;
    keys(
        ctx,
        publish,
        &["source_id", "implementation", "effect", "acknowledgement"],
    )?;
    eq_attr(ctx, publish, "acknowledgement", "returned_receipt")?;
    ensure(
        operands(ctx, score) == [flows["mapped"], flows["scored"]]
            && operands(ctx, publish)
                == [flows["scored"], flows["feedback"], entities["publication"]],
        "scorer/publication/feedback wiring differs",
    )?;
    let consumer = vec![
        stage(ctx, score, "score", std::slice::from_ref(&root), &mut seen)?,
        stage(ctx, publish, "publish", &[root], &mut seen)?,
    ];
    Ok(Meaning {
        horizon,
        producer,
        consumer,
    })
}

pub fn accept(text: &str) -> Checked<String> {
    let ctx = &mut Context::new();
    let op = parse(ctx, text)?;
    meaning(ctx, op)?;
    let canonical = op.deref(ctx).disp(ctx).to_string();
    let restored = parse(ctx, &canonical)?;
    ensure(
        operation_eq(
            ctx,
            &mut IrMapping::default(),
            op,
            restored,
            &IGNORE_LOC_NAMES,
        ) == EqResult::Eq,
        "framework structural round-trip differs",
    )?;
    // A fresh context cannot retain the original parser/authoring object.
    let fresh = &mut Context::new();
    let restored = parse(fresh, &canonical)?;
    meaning(fresh, restored)?;
    Ok(canonical)
}
fn check_manifest(manifest: &Manifest, semantic: &Meaning) -> Checked<()> {
    ensure(
        manifest.schema == 1 && !manifest.registry.is_empty(),
        "unsupported binding manifest",
    )?;
    for call in semantic.producer.iter().chain(&semantic.consumer) {
        let binding = manifest
            .bindings
            .get(&call.symbol)
            .ok_or_else(|| format!("{}: missing binding {}", call.origin, call.symbol))?;
        ensure(
            !binding.version.is_empty()
                && binding.role == call.role
                && binding.effect == call.effect
                && binding.outcomes == OUTCOMES
                && binding.codec == CODEC
                && binding.resources.is_empty(),
            format!(
                "{}: incompatible binding/schema/resources/outcomes",
                call.origin
            ),
        )?;
    }
    Ok(())
}
fn call_attrs(call: &Call, manifest: &Manifest) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("origin".into(), call.origin.clone()),
        ("implementation".into(), call.symbol.clone()),
        ("role".into(), call.role.clone()),
        ("effect".into(), call.effect.clone()),
        (
            "ancestors".into(),
            serde_json::to_string(&call.ancestors).unwrap(),
        ),
        (
            "binding_version".into(),
            manifest.bindings[&call.symbol].version.clone(),
        ),
        ("completion".into(), COMPLETION.into()),
        ("codec".into(), CODEC.into()),
        ("outcomes".into(), serde_json::to_string(&OUTCOMES).unwrap()),
        ("resources".into(), "none".into()),
    ])
}
fn set_attrs(ctx: &Context, op: Ptr<Operation>, attributes: BTreeMap<String, String>) {
    for (key, value) in attributes {
        op.deref_mut(ctx).attributes.set(
            Identifier::try_from(key).expect("static schema key"),
            StringAttr::new(value),
        );
    }
}
fn pairs(items: &[(&str, &str)]) -> BTreeMap<String, String> {
    items
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}
fn new<T: Op>(
    ctx: &mut Context,
    nregions: usize,
    attributes: BTreeMap<String, String>,
) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        T::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        nregions,
    );
    set_attrs(ctx, op, attributes);
    for i in 0..nregions {
        let block = BasicBlock::new(ctx, None, vec![]);
        let region = op.deref(ctx).get_region(i);
        block.insert_at_back(region, ctx);
    }
    op
}
fn recipes() -> [BTreeMap<String, String>; 3] {
    [
        pairs(&[
            (
                "selection",
                "cursor_plus_current_policy_bias;source_revision=0",
            ),
            ("admission", "count_until_feedback_ack"),
        ]),
        pairs(&[
            ("transport", "bounded_mailbox"),
            ("release", "after_feedback_ack;no_protection_while_waiting"),
        ]),
        pairs(&[
            (
                "control",
                "dedicated_task;receipt_then_apply_then_retire_then_release",
            ),
            ("update", "bias=score%5;revision+=1"),
            ("admission", "live_validated_publication;once_per_key"),
        ]),
    ]
}
fn plan_attrs(
    ctx: &Context,
    accepted: Ptr<Operation>,
    horizon: usize,
    capacity: usize,
) -> BTreeMap<String, String> {
    let mut attrs = pairs(&[
        ("schema", "1"),
        ("target", TARGET),
        ("stop_rule", "settle_selected"),
        ("completion", "drain_physical_before_release"),
    ]);
    attrs.insert(
        "accepted_hash".into(),
        operation_hash(ctx, accepted, &IGNORE_LOC_NAMES).to_string(),
    );
    attrs.insert("horizon".into(), horizon.to_string());
    attrs.insert("capacity".into(), capacity.to_string());
    attrs
}
fn target(meaning: &Meaning, capacity: usize) -> Checked<()> {
    ensure(
        meaning.horizon <= 2 && capacity == 1,
        "accepted but unsupported target horizon/capacity",
    )
}

struct Fuse;
impl MatchRewrite for Fuse {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        is::<ChainOp>(ctx, op)
            && attr(ctx, op, "role").is_ok_and(|r| r == "producer")
            && children(ctx, op, 0).is_ok_and(|ops| {
                ops.len() > 1
                    && ops.iter().all(|p| {
                        is::<CallOp>(ctx, *p)
                            && attr(ctx, *p, "effect").is_ok_and(|e| e == "pure")
                            && attr(ctx, *p, "resources").is_ok_and(|r| r == "none")
                    })
            })
    }
    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> IrResult<()> {
        let calls = children(ctx, op, 0).expect("verified chain");
        let fused = new::<FusedOp>(
            ctx,
            1,
            pairs(&[("protocol", "preserve_each_stage_guard_and_completion")]),
        );
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(calls[0]));
        rewriter.insert_operation(ctx, fused);
        let block = body(ctx, fused, 0).expect("fresh fused block");
        for call in calls {
            rewriter.move_operation(ctx, call, OpInsertionPoint::AtBlockEnd(block));
        }
        Ok(())
    }
}

// Two actual framework-managed analyses: dispatch count changes under fusion;
// accepted source structure is preserved. Counters live only in PM evidence.
pliron::dict_key!(COMPUTATIONS, "dispatch_computations");
pliron::dict_key!(SOURCE_COMPUTATIONS, "source_computations");
pliron::dict_key!(OBSERVATIONS, "observations");
fn bump(analyses: &mut AnalysisManager, key: pliron::identifier::Identifier) {
    let count = analyses
        .pm_data_mut()
        .state_mut()
        .custom_state
        .entry(key)
        .or_insert_with(|| Box::new(0usize));
    *count.downcast_mut::<usize>().unwrap() += 1;
}
fn count(analyses: &AnalysisManager, key: &pliron::identifier::Identifier) -> usize {
    analyses
        .pm_data()
        .state()
        .custom_state
        .get(key)
        .map_or(0, |n| *n.downcast_ref::<usize>().unwrap())
}
struct DispatchAnalysis(usize);
impl Analysis for DispatchAnalysis {
    fn name(&self) -> &str {
        "producer_dispatch_count"
    }
    fn compute(
        op: Ptr<Operation>,
        ctx: &Context,
        analyses: &mut AnalysisManager,
    ) -> IrResult<Self> {
        bump(analyses, COMPUTATIONS.clone());
        let producer = children(ctx, op, 1)
            .unwrap()
            .into_iter()
            .find(|p| is::<ChainOp>(ctx, *p) && attr(ctx, *p, "role").unwrap() == "producer")
            .unwrap();
        Ok(Self(children(ctx, producer, 0).unwrap().len()))
    }
}
struct SourceAnalysis(u64);
impl Analysis for SourceAnalysis {
    fn name(&self) -> &str {
        "accepted_structure_hash"
    }
    fn compute(
        op: Ptr<Operation>,
        ctx: &Context,
        analyses: &mut AnalysisManager,
    ) -> IrResult<Self> {
        bump(analyses, SOURCE_COMPUTATIONS.clone());
        Ok(Self(operation_hash(
            ctx,
            children(ctx, op, 0).unwrap()[0],
            &IGNORE_LOC_NAMES,
        )))
    }
}
struct Observe;
impl Pass for Observe {
    fn name(&self) -> &str {
        "observe_cached_analyses"
    }
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> IrResult<PassResult> {
        let dispatch = analyses.get_analysis::<DispatchAnalysis>(op, ctx)?.0;
        assert_eq!(
            dispatch,
            analyses.get_analysis::<DispatchAnalysis>(op, ctx)?.0
        );
        let source_hash = analyses.get_analysis::<SourceAnalysis>(op, ctx)?.0;
        assert_eq!(
            source_hash,
            operation_hash(ctx, children(ctx, op, 0).unwrap()[0], &IGNORE_LOC_NAMES)
        );
        let row = (
            dispatch,
            count(analyses, &COMPUTATIONS),
            count(analyses, &SOURCE_COMPUTATIONS),
        );
        let entries = analyses
            .pm_data_mut()
            .state_mut()
            .custom_state
            .entry(OBSERVATIONS.clone())
            .or_insert_with(|| Box::new(Vec::<(usize, usize, usize)>::new()));
        entries
            .downcast_mut::<Vec<(usize, usize, usize)>>()
            .unwrap()
            .push(row);
        Ok(PassResult::default())
    }
}
struct FusionPass;
impl Pass for FusionPass {
    fn name(&self) -> &str {
        "fuse_guarded_local_calls"
    }
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> IrResult<PassResult> {
        let ir_changed = apply_match_rewrite(ctx, &mut Fuse, RewriterOrder::default(), op)?;
        let mut result = PassResult::default();
        result.ir_changed = ir_changed;
        result.set_preserved::<SourceAnalysis>();
        Ok(result)
    }
}

pub fn prepare(
    accepted: &str,
    manifest: Manifest,
    capacity: usize,
    fuse: bool,
) -> Checked<Artifact> {
    let ctx = &mut Context::new();
    let semantic_op = parse(ctx, accepted)?;
    let semantic = meaning(ctx, semantic_op)?;
    target(&semantic, capacity)?;
    check_manifest(&manifest, &semantic)?;
    let plan = new::<PlanOp>(
        ctx,
        2,
        plan_attrs(ctx, semantic_op, semantic.horizon, capacity),
    );
    semantic_op.insert_at_back(body(ctx, plan, 0)?, ctx);
    let recipes_block = body(ctx, plan, 1)?;
    let [selection, handoff, feedback] = recipes();
    for recipe in [
        new::<SelectRecipeOp>(ctx, 0, selection),
        new::<HandoffOp>(ctx, 0, handoff),
        new::<FeedbackOp>(ctx, 0, feedback),
    ] {
        recipe.insert_at_back(recipes_block, ctx);
    }
    for (role, calls) in [
        ("producer", &semantic.producer),
        ("consumer", &semantic.consumer),
    ] {
        let chain = new::<ChainOp>(ctx, 1, pairs(&[("role", role)]));
        chain.insert_at_back(recipes_block, ctx);
        for call in calls {
            let call = new::<CallOp>(ctx, 0, call_attrs(call, &manifest));
            call.insert_at_back(body(ctx, chain, 0)?, ctx);
        }
    }
    verify_operation(plan, ctx).map_err(|e| e.disp(ctx).to_string())?;
    let mut pipeline = Passes::default();
    pipeline.add_pass(Observe);
    if fuse {
        pipeline.add_pass(FusionPass);
    }
    pipeline.add_pass(Observe);
    let mut analyses = AnalysisManager::default();
    pipeline
        .run(plan, ctx, &mut analyses)
        .map_err(|e| e.disp(ctx).to_string())?;
    verify_operation(plan, ctx).map_err(|e| e.disp(ctx).to_string())?;
    let evidence = analyses.pm_data().state().custom_state[&OBSERVATIONS]
        .downcast_ref::<Vec<(usize, usize, usize)>>()
        .unwrap()
        .clone();
    let text = plan.deref(ctx).disp(ctx).to_string();
    let mut artifact = load(accepted, &text, manifest)?;
    artifact.analysis = evidence;
    Ok(artifact)
}

pub fn load(accepted: &str, prepared: &str, manifest: Manifest) -> Checked<Artifact> {
    let ctx = &mut Context::new();
    let original = parse(ctx, accepted)?;
    let semantic = meaning(ctx, original)?;
    check_manifest(&manifest, &semantic)?;
    let plan = parse(ctx, prepared)?;
    ensure(is::<PlanOp>(ctx, plan), "prepared root must be exec.plan")?;
    let snapshot = children(ctx, plan, 0)?;
    ensure(
        snapshot.len() == 1,
        "one complete semantic snapshot required",
    )?;
    closed(ctx, plan, 0)?;
    closed(ctx, plan, 1)?;
    ensure(
        operation_eq(
            ctx,
            &mut IrMapping::default(),
            original,
            snapshot[0],
            &IGNORE_LOC_NAMES,
        ) == EqResult::Eq,
        "accepted/prepared semantic snapshot correspondence differs",
    )?;
    meaning(ctx, snapshot[0])?;
    let capacity = attr(ctx, plan, "capacity")?
        .parse::<usize>()
        .map_err(|_| "invalid prepared capacity")?;
    target(&semantic, capacity)?;
    ensure(
        attrs(ctx, plan)? == plan_attrs(ctx, original, semantic.horizon, capacity),
        "prepared target/bounds/stop/completion/hash correspondence differs",
    )?;
    let ops = children(ctx, plan, 1)?;
    ensure(
        ops.len() == 5
            && is::<SelectRecipeOp>(ctx, ops[0])
            && is::<HandoffOp>(ctx, ops[1])
            && is::<FeedbackOp>(ctx, ops[2])
            && is::<ChainOp>(ctx, ops[3])
            && is::<ChainOp>(ctx, ops[4]),
        "missing/duplicate/unsupported prepared recipes",
    )?;
    for (op, expected) in ops[..3].iter().zip(recipes()) {
        ensure(
            attrs(ctx, *op)? == expected,
            "prepared selection/handoff/feedback protocol differs",
        )?;
    }
    let mut producer = Vec::new();
    for (chain, role, expected) in [
        (ops[3], "producer", &semantic.producer),
        (ops[4], "consumer", &semantic.consumer),
    ] {
        ensure(
            attrs(ctx, chain)? == pairs(&[("role", role)]),
            "prepared chain role differs",
        )?;
        closed(ctx, chain, 0)?;
        let mut index = 0;
        for op in children(ctx, chain, 0)? {
            let fused = is::<FusedOp>(ctx, op);
            let calls = if fused {
                ensure(
                    role == "producer"
                        && attrs(ctx, op)?
                            == pairs(&[("protocol", "preserve_each_stage_guard_and_completion")]),
                    "invalid fusion protocol/role",
                )?;
                closed(ctx, op, 0)?;
                let calls = children(ctx, op, 0)?;
                ensure(calls.len() >= 2, "fused group needs multiple stages")?;
                calls
            } else {
                vec![op]
            };
            let mut group = Vec::new();
            for call in calls {
                ensure(is::<CallOp>(ctx, call), "unsupported prepared call/group")?;
                let source = expected.get(index).ok_or("extra prepared stage")?;
                ensure(
                    attrs(ctx, call)? == call_attrs(source, &manifest),
                    format!(
                        "{}: prepared binding/source/ancestry/schema/outcome/completion differs",
                        source.origin
                    ),
                )?;
                ensure(
                    !fused || (source.role == "map" && source.effect == "pure"),
                    "ineligible fused stage",
                )?;
                group.push(source.clone());
                index += 1;
            }
            if role == "producer" {
                producer.push(group);
            }
        }
        ensure(
            index == expected.len(),
            format!("{role}: missing prepared stage"),
        )?;
    }
    Ok(Artifact {
        schema: 1,
        pliron: "0.20.0",
        accepted: accepted.into(),
        prepared: prepared.into(),
        accepted_sha256: format!("{:x}", Sha256::digest(accepted.as_bytes())),
        target: TARGET,
        horizon: semantic.horizon,
        capacity,
        producer,
        consumer: semantic.consumer,
        manifest,
        analysis: vec![],
    })
}
