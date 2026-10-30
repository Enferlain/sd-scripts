use std::collections::{BTreeMap, BTreeSet};

use pliron::{
    basic_block::BasicBlock,
    builtin::attributes::StringAttr,
    combine::{Parser, eof, parser::char::spaces},
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
    r#type::{TypeHandle, Typed},
    value::Value,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::contract::{self, BASE_OUTCOMES, RESERVED_OUTCOMES, ResourceUse};
use crate::dialect::*;

pub type Checked<T> = Result<T, String>;
pub const TARGET: &str = "scalar-async-cpu-v1";
const CONTRACT: &str = "scalar-feedback-v1";
const OUTCOMES: [&str; 3] = BASE_OUTCOMES;
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
    pub input: String,
    pub output: String,
    pub resources: Option<Vec<ResourceUse>>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenResource {
    pub version: String,
    pub digest: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    pub registry: String,
    pub bindings: BTreeMap<String, Binding>,
    pub codecs: BTreeMap<String, String>,
    pub resources: BTreeMap<String, FrozenResource>,
    pub runtime_outcomes: Vec<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Call {
    pub origin: String,
    pub symbol: String,
    pub role: String,
    pub effect: String,
    pub ancestors: Vec<String>,
    pub input: String,
    pub output: String,
    pub input_codec: String,
    pub output_codec: String,
    pub resources: Option<Vec<ResourceUse>>,
    pub outcomes: Vec<String>,
}
#[derive(Clone, Debug)]
struct Meaning {
    origin: String,
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
    pub resource_analysis: ResourceReport,
}

pub fn parse(ctx: &mut Context, text: &str) -> Checked<Ptr<Operation>> {
    // parse_from_str discards the remainder; this boundary verifies a document.
    let op = parse_from_str(
        Operation::top_level_parser().skip(spaces()).skip(eof()),
        ctx,
        text,
    )
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
    contract::strings(ctx, op)
}
fn attr(ctx: &Context, op: Ptr<Operation>, key: &str) -> Checked<String> {
    attrs(ctx, op)?
        .get(key)
        .filter(|s| !s.is_empty())
        .cloned()
        .ok_or_else(|| format!("{}: missing {key}", Operation::get_opid(op, ctx)))
}
fn keys(ctx: &Context, op: Ptr<Operation>, names: &[&str]) -> Checked<()> {
    let attributes = attrs(ctx, op)?;
    let provided = attributes
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected = names.iter().copied().collect::<BTreeSet<_>>();
    ensure(
        provided == expected,
        format!(
            "{}: {} missing or unknown semantic attributes; expected {:?}, provided {:?}",
            attributes
                .get("source_id")
                .map(String::as_str)
                .unwrap_or("operation"),
            Operation::get_opid(op, ctx),
            expected,
            provided
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
        input: "record".into(),
        output: "record".into(),
        input_codec: CODEC.into(),
        output_codec: CODEC.into(),
        resources: Some(vec![]),
        outcomes: OUTCOMES.iter().map(|s| s.to_string()).collect(),
    })
}
fn local(
    ctx: &Context,
    op: Ptr<Operation>,
    ancestors: &[String],
    seen: &mut BTreeSet<String>,
    input_type: TypeHandle,
    output_type: TypeHandle,
) -> Checked<Vec<Call>> {
    let block = body(ctx, op, 0)?;
    ensure(
        block.deref(ctx).get_num_arguments() == 1
            && block.deref(ctx).get_argument(0).get_type(ctx) == input_type,
        "local region needs one argument with declared input schema",
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
            operands(ctx, *child) == [previous] && child.deref(ctx).get_num_results() == 1,
            "broken ordered local value chain",
        )?;
        if is::<LocalOp>(ctx, *child) {
            keys(ctx, *child, &["source_id"])?;
            let mut path = ancestors.to_vec();
            path.push(source(ctx, *child, seen)?);
            calls.extend(local(
                ctx,
                *child,
                &path,
                seen,
                previous.get_type(ctx),
                result(ctx, *child).get_type(ctx),
            )?);
        } else {
            let object = Operation::get_op_dyn(*child, ctx);
            let interface = op_cast::<dyn LocalStageInterface>(&*object)
                .ok_or("unsupported local operation/interface")?;
            let description = interface.describe(ctx)?;
            let origin = source(ctx, *child, seen)?;
            ensure(
                description.cardinality == "one-to-one"
                    && description.input == previous.get_type(ctx)
                    && description.output == result(ctx, *child).get_type(ctx),
                format!("{origin}: local schema/cardinality interface differs from SSA types"),
            )?;
            keys(ctx, *child, &description.attributes)?;
            let outcomes = &description.outcomes;
            ensure(
                OUTCOMES.iter().all(|s| outcomes.iter().any(|o| o == s))
                    && outcomes.iter().all(|s| !s.is_empty())
                    && outcomes.iter().collect::<BTreeSet<_>>().len() == outcomes.len()
                    && outcomes.iter().all(|s| {
                        !RESERVED_OUTCOMES.contains(&s.as_str())
                            || BASE_OUTCOMES.contains(&s.as_str())
                    }),
                format!("{origin}: missing baseline or invalid declared outcomes"),
            )?;
            let (input, input_codec) =
                contract::payload(ctx, description.input).map_err(|e| format!("{origin}: {e}"))?;
            let (output, output_codec) =
                contract::payload(ctx, description.output).map_err(|e| format!("{origin}: {e}"))?;
            calls.push(Call {
                origin,
                symbol: attr(ctx, *child, "implementation")?,
                role: "map".into(),
                effect: attr(ctx, *child, "effect")?,
                ancestors: ancestors.into(),
                input,
                output,
                input_codec,
                output_codec,
                resources: description.resources,
                outcomes: description.outcomes,
            });
        }
        previous = result(ctx, *child);
    }
    let yield_op = *ops.last().unwrap();
    keys(ctx, yield_op, &[])?;
    ensure(
        operands(ctx, yield_op) == [previous] && previous.get_type(ctx) == output_type,
        "yield must expose the final local endpoint with declared output schema",
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
    let producer = local(
        ctx,
        compose,
        &path,
        &mut seen,
        flows["selected"].get_type(ctx),
        flows["mapped"].get_type(ctx),
    )?;
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
        origin: attr(ctx, session, "source_id")?,
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
    let reserved = RESERVED_OUTCOMES
        .iter()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    ensure(
        manifest.runtime_outcomes == reserved,
        format!(
            "target {} runtime outcome ABI mismatch; expected {:?}, provided {:?}",
            TARGET, reserved, manifest.runtime_outcomes
        ),
    )?;
    ensure(
        manifest.schema == 2 && !manifest.registry.is_empty(),
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
                && binding.outcomes == call.outcomes
                && binding.input == call.input
                && binding.output == call.output
                && binding.codec == call.output_codec
                && binding.resources == call.resources
                && manifest.codecs.get(&call.input) == Some(&call.input_codec)
                && manifest.codecs.get(&call.output) == Some(&call.output_codec),
            format!(
                "{}: target {} incompatible binding/schema/resources/outcomes for {}: expected input={} output={} codecs={}/{} resources={:?} outcomes={:?}; provided {:?}, installed codecs={:?}",
                call.origin,
                TARGET,
                call.symbol,
                call.input,
                call.output,
                call.input_codec,
                call.output_codec,
                call.resources,
                call.outcomes,
                binding,
                manifest.codecs
            ),
        )?;
        let mut seen = BTreeSet::new();
        for resource in call.resources.iter().flatten() {
            ensure(
                resource.access == "read_frozen"
                    && !resource.symbol.is_empty()
                    && !resource.version.is_empty()
                    && !resource.digest.is_empty()
                    && seen.insert(&resource.symbol),
                format!(
                    "{}: target {} requires unique read_frozen resource contracts, provided {:?}",
                    call.origin, TARGET, resource
                ),
            )?;
            let actual = manifest.resources.get(&resource.symbol);
            ensure(
                actual
                    .is_some_and(|r| r.version == resource.version && r.digest == resource.digest),
                format!(
                    "{}: target {} resource @{} requires frozen version={} digest={}, provided {:?}",
                    call.origin, TARGET, resource.symbol, resource.version, resource.digest, actual
                ),
            )?;
        }
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
        ("input".into(), call.input.clone()),
        ("output".into(), call.output.clone()),
        ("input_codec".into(), call.input_codec.clone()),
        ("codec".into(), call.output_codec.clone()),
        (
            "outcomes".into(),
            serde_json::to_string(&call.outcomes).unwrap(),
        ),
        (
            "resources".into(),
            serde_json::to_string(&call.resources).unwrap(),
        ),
    ])
}
fn differences(expected: &BTreeMap<String, String>, provided: &BTreeMap<String, String>) -> String {
    expected
        .keys()
        .chain(provided.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key| expected.get(*key) != provided.get(*key))
        .map(|key| {
            format!(
                "{key}: expected {:?}, provided {:?}",
                expected.get(key),
                provided.get(key)
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
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
        ("schema", "2"),
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
        format!(
            "{}: accepted but unsupported target {}: provided horizon={}, capacity={}; expected horizon 1..=2, capacity 1",
            meaning.origin, TARGET, meaning.horizon, capacity
        ),
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StageDecision {
    pub origin: String,
    pub groupable: bool,
    pub reason: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResourceReport {
    pub reads: Vec<ResourceUse>,
    pub unknown_sources: Vec<String>,
    pub decisions: Vec<StageDecision>,
    pub computations: usize,
}
impl ResourceReport {
    fn from_plan(ctx: &Context, op: Ptr<Operation>) -> Self {
        let semantic = meaning(
            ctx,
            children(ctx, op, 0).expect("verified snapshot region")[0],
        )
        .expect("verified semantic contracts");
        Self::from_calls(semantic.producer)
    }
    fn from_calls(calls: Vec<Call>) -> Self {
        let mut reads = BTreeSet::new();
        let mut unknown_sources = vec![];
        let mut decisions = vec![];
        for call in calls {
            let known =
                call.resources.is_some() && ["pure", "read_frozen"].contains(&call.effect.as_str());
            if let Some(uses) = &call.resources {
                reads.extend(uses.iter().cloned());
            }
            if !known {
                unknown_sources.push(call.origin.clone());
            }
            decisions.push(StageDecision{origin:call.origin,groupable:known,reason:if known {"ordered unary schemas; frozen snapshots pinned; preserve every guard/outcome/completion".into()} else {format!("target {} declines grouping: unknown resource access/effect {}",TARGET,call.effect)}});
        }
        Self {
            reads: reads.into_iter().collect(),
            unknown_sources,
            decisions,
            computations: 0,
        }
    }
    fn permits(&self, origin: &str) -> bool {
        self.decisions
            .iter()
            .any(|d| d.origin == origin && d.groupable)
    }
}
pliron::dict_key!(RESOURCE_COMPUTATIONS, "resource_computations");
struct ResourceAnalysis(ResourceReport);
impl Analysis for ResourceAnalysis {
    fn name(&self) -> &str {
        "local_resource_use_and_stability"
    }
    fn compute(
        op: Ptr<Operation>,
        ctx: &Context,
        analyses: &mut AnalysisManager,
    ) -> IrResult<Self> {
        bump(analyses, RESOURCE_COMPUTATIONS.clone());
        Ok(Self(ResourceReport::from_plan(ctx, op)))
    }
}

struct Fuse {
    resources: ResourceReport,
}
impl Fuse {
    fn runs(&self, ctx: &Context, op: Ptr<Operation>) -> Vec<Vec<Ptr<Operation>>> {
        let mut runs = vec![];
        let mut run = vec![];
        for child in children(ctx, op, 0).expect("verified chain") {
            if is::<CallOp>(ctx, child)
                && attr(ctx, child, "origin").is_ok_and(|o| self.resources.permits(&o))
            {
                run.push(child);
            } else {
                if run.len() > 1 {
                    runs.push(std::mem::take(&mut run));
                } else {
                    run.clear();
                }
            }
        }
        if run.len() > 1 {
            runs.push(run);
        }
        runs
    }
}
impl MatchRewrite for Fuse {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        is::<ChainOp>(ctx, op)
            && attr(ctx, op, "role").is_ok_and(|r| r == "producer")
            && !self.runs(ctx, op).is_empty()
    }
    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> IrResult<()> {
        for calls in self.runs(ctx, op) {
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
        let resources = analyses
            .get_analysis::<ResourceAnalysis>(op, ctx)?
            .0
            .clone();
        assert_eq!(resources, ResourceReport::from_plan(ctx, op));
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
        analyses: &mut AnalysisManager,
    ) -> IrResult<PassResult> {
        let resources = analyses
            .get_analysis::<ResourceAnalysis>(op, ctx)?
            .0
            .clone();
        let ir_changed =
            apply_match_rewrite(ctx, &mut Fuse { resources }, RewriterOrder::default(), op)?;
        let mut result = PassResult::default();
        result.ir_changed = ir_changed;
        result.set_preserved::<SourceAnalysis>();
        result.set_preserved::<ResourceAnalysis>();
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
    artifact.resource_analysis.computations = count(&analyses, &RESOURCE_COMPUTATIONS);
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
    let mut analyses = AnalysisManager::default();
    let mut resource_report = analyses
        .get_analysis::<ResourceAnalysis>(plan, ctx)
        .expect("verified source analysis")
        .0
        .clone();
    resource_report.computations = count(&analyses, &RESOURCE_COMPUTATIONS);
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
                let claim = attrs(ctx, call)?;
                let required = call_attrs(source, &manifest);
                ensure(
                    claim == required,
                    format!(
                        "{}: target {} prepared binding/source/ancestry/schema/outcome/completion differs: {}",
                        source.origin,
                        TARGET,
                        differences(&required, &claim)
                    ),
                )?;
                ensure(
                    !fused || (source.role == "map" && resource_report.permits(&source.origin)),
                    format!(
                        "{}: ineligible fused stage; resource analysis requires known frozen stability, provided effect={} resources={:?}",
                        source.origin, source.effect, source.resources
                    ),
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
        schema: 2,
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
        resource_analysis: resource_report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_known_reads_do_not_hide_unknown_effects() {
        let read = ResourceUse {
            symbol: "coefficients".into(),
            access: "read_frozen".into(),
            version: "1".into(),
            digest: "snapshot".into(),
        };
        let report = ResourceReport::from_calls(vec![Call {
            origin: "partial".into(),
            symbol: "selected".into(),
            role: "map".into(),
            effect: "may_effect".into(),
            ancestors: vec![],
            input: "record".into(),
            output: "record".into(),
            input_codec: CODEC.into(),
            output_codec: CODEC.into(),
            resources: Some(vec![read.clone()]),
            outcomes: OUTCOMES.iter().map(|s| s.to_string()).collect(),
        }]);
        assert_eq!(report.reads, vec![read]);
        assert_eq!(report.unknown_sources, vec!["partial"]);
        assert!(!report.permits("partial"));
    }
}
