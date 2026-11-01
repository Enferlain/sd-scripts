//! C's semantic authority exchange, independently verified in Pliron 0.20.0.
mod dialect;

use dialect::{AuthorityContract, DeclarationOp, PlanOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::attributes::StringAttr,
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        cloning::IrMapping,
        equivalence::{EqResult, IGNORE_LOC_NAMES, operation_eq},
    },
    linked_list::ContainsLinkedList,
    op::{Op, op_cast},
    operation::{Operation, verify_operation},
    printable::Printable,
    r#type::Typed,
    value::Value,
};
use rmst_pliron_b::{contract, prepare::parse};
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};

type Checked<T> = Result<T, String>;
pub const TARGET: &str = "candidate-grant-cpu-v1";
const PROFILE: &str = "restored-two-pass-v1";
const COMPLETION: &str = "physical_return_then_owner_handback_verification";
const DELEGATED: &str = "backward,temporary_edit,intermediate_reset,restore";
const RETAINED: &str = "clipping,final_advancement,final_reset";
const PHASES: [&str; 5] = [
    "backward",
    "temporary_edit",
    "intermediate_reset",
    "backward",
    "restore",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub version: String,
    pub role: String,
    pub effect: String,
    pub completion: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    pub target: String,
    pub offered: Vec<String>,
    pub lowering: String,
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
    ensure(blocks.len() == 1, "one block required per authority region")?;
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
        let shape = op_cast::<dyn AuthorityContract>(&*object)
            .ok_or_else(|| {
                format!(
                    "unsupported authority operation {}",
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
                format!("{id}: authority region arguments differ"),
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
fn definition<'a>(ctx: &Context, infos: &'a [Info], value: Value, kind: &str) -> Checked<&'a Info> {
    infos
        .iter()
        .find(|n| {
            n.kind == kind && n.op.deref(ctx).get_num_results() == 1 && result(ctx, n) == value
        })
        .ok_or_else(|| format!("expected accepted {kind} endpoint"))
}
fn nested(infos: &[Info], parent: Ptr<Operation>) -> Vec<&Info> {
    infos.iter().filter(|n| n.parent == Some(parent)).collect()
}
fn clip(advance: &Info) -> Checked<Option<f64>> {
    if advance.attr("clip_norm") == "none" {
        Ok(None)
    } else {
        let value = advance.float("clip_norm")?;
        ensure(
            value > 0.0,
            format!("{}: positive retained clip norm required", advance.id()),
        )?;
        Ok(Some(value))
    }
}
fn verify(ctx: &Context, infos: &[Info]) -> Checked<()> {
    let root = &infos[0];
    ensure(
        root.kind == "session" && root.parent.is_none(),
        "expected one ga.session document",
    )?;
    root.expect("version", "1")?;
    root.expect("failure", "drain_entered_preserve_uncertain")?;
    let direct = nested(infos, root.op);
    ensure(
        direct.last().is_some_and(|n| n.kind == "finish")
            && direct.iter().filter(|n| n.kind == "finish").count() == 1,
        "session must finish exactly once after retained outcomes",
    )?;
    let mut participants = HashSet::new();
    let mut units = HashSet::new();
    for node in infos {
        match node.kind {
            "session" => ensure(
                node.op == root.op,
                "nested authority sessions are unsupported",
            )?,
            "model" => {
                ensure(
                    node.parent == Some(root.op),
                    format!("{}: model must be session-owned", node.id()),
                )?;
                ensure(
                    !node.attr("participant").is_empty()
                        && participants.insert(node.attr("participant")),
                    format!("{}: empty/duplicate participant identity", node.id()),
                )?;
                node.expect("treatment", "trainable")?;
                node.float("initial")?;
                ensure(
                    !node.attr("implementation").is_empty(),
                    format!("{}: model implementation required", node.id()),
                )?;
            }
            "unit" => {
                ensure(
                    node.parent == Some(root.op),
                    format!("{}: unit must be session-owned", node.id()),
                )?;
                definition(ctx, infos, operand(ctx, node, 0), "model")?;
                ensure(
                    !node.attr("unit").is_empty() && units.insert(node.attr("unit")),
                    format!("{}: empty/duplicate unit identity", node.id()),
                )?;
                ensure(
                    !node.attr("parameter").is_empty(),
                    format!("{}: parameter required", node.id()),
                )?;
            }
            "offer" => {
                ensure(
                    node.parent == Some(root.op),
                    format!("{}: profile must be session-owned", node.id()),
                )?;
                node.expect("profile", PROFILE)?;
                node.expect("delegated", DELEGATED)?;
                node.expect("retained", RETAINED)?;
            }
            "execute" => {
                ensure(
                    node.parent == Some(root.op),
                    format!("{}: grant must be session-owned", node.id()),
                )?;
                definition(ctx, infos, operand(ctx, node, 0), "model")?;
                let unit = definition(ctx, infos, operand(ctx, node, 1), "unit")?;
                definition(ctx, infos, operand(ctx, node, 2), "offer")?;
                ensure(
                    operand(ctx, node, 0) == operand(ctx, unit, 0),
                    format!("{}: grant model/unit association differs", node.id()),
                )?;
                node.expect("completion", COMPLETION)?;
                node.expect(
                    "handback",
                    "restored_current_final_gradient_then_owner_verifies",
                )?;
                ensure(
                    !node.attr("implementation").is_empty(),
                    format!("{}: selected grant implementation required", node.id()),
                )?;
                let body = nested(infos, node.op);
                ensure(
                    body.len() == 6 && body[5].kind == "yield",
                    format!(
                        "{}: complete five-phase protocol and handback yield required",
                        node.id()
                    ),
                )?;
                let region = node.op.deref(ctx).get_region(0);
                let block = region
                    .deref(ctx)
                    .iter(ctx)
                    .next()
                    .expect("verified single block");
                let mut previous = block.deref(ctx).get_argument(0);
                for (index, phase) in PHASES.iter().enumerate() {
                    ensure(
                        body[index].kind == "phase",
                        format!("{}: ordered phase operation required", node.id()),
                    )?;
                    body[index].expect("phase", phase)?;
                    ensure(
                        operand(ctx, body[index], 0) == previous,
                        format!("{}: authority phase token chain differs", body[index].id()),
                    )?;
                    previous = result(ctx, body[index]);
                }
                ensure(
                    operand(ctx, body[5], 0) == previous,
                    format!("{}: handback does not follow final restoration", node.id()),
                )?;
                let handback = result(ctx, node);
                let uses = infos
                    .iter()
                    .map(|n| {
                        let o = n.op.deref(ctx);
                        (0..o.get_num_operands())
                            .filter(|i| o.get_operand(*i) == handback)
                            .count()
                    })
                    .sum::<usize>();
                ensure(
                    uses == 1,
                    format!(
                        "{}: handback must have exactly one retained consumer",
                        node.id()
                    ),
                )?;
            }
            "phase" | "yield" => ensure(
                node.parent
                    .is_some_and(|p| infos.iter().any(|n| n.op == p && n.kind == "execute")),
                format!("{}: phase outside delegated scope", node.id()),
            )?,
            "advance" => {
                ensure(
                    node.parent == Some(root.op),
                    format!("{}: retained advancement must be session-owned", node.id()),
                )?;
                let grant = definition(ctx, infos, operand(ctx, node, 0), "execute")?;
                definition(ctx, infos, operand(ctx, node, 1), "unit")?;
                ensure(
                    operand(ctx, node, 1) == operand(ctx, grant, 1),
                    format!(
                        "{}: handback/advancement unit association differs",
                        node.id()
                    ),
                )?;
                node.expect("authority", "retained_owner_only")?;
                ensure(
                    node.float("learning_rate")? > 0.0,
                    format!("{}: positive retained learning rate required", node.id()),
                )?;
                clip(node)?;
            }
            "finish" => {
                ensure(
                    node.parent == Some(root.op),
                    format!("{}: finish outside session", node.id()),
                )?;
                let expected: HashSet<_> = direct
                    .iter()
                    .filter(|n| n.kind == "advance")
                    .map(|n| result(ctx, n))
                    .collect();
                let operation = node.op.deref(ctx);
                let actual: HashSet<_> = (0..operation.get_num_operands())
                    .map(|i| operation.get_operand(i))
                    .collect();
                ensure(
                    !expected.is_empty()
                        && actual == expected
                        && operation.get_num_operands() == expected.len(),
                    format!(
                        "{}: completion must cover all and only retained advancement outcomes",
                        node.id()
                    ),
                )?;
            }
            _ => return Err("unsupported authority contract".into()),
        }
    }
    Ok(())
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
    let canonical = root.deref(ctx).disp(ctx).to_string();
    let restored = parse(ctx, &canonical)?;
    ensure(
        equivalent(ctx, root, restored),
        "authority structural round-trip differs",
    )?;
    let fresh = &mut Context::new();
    let root = parse(fresh, &canonical)?;
    verify(fresh, &collect(fresh, root)?)?;
    Ok(canonical)
}
fn one<'a>(infos: &'a [Info], kind: &str) -> Checked<&'a Info> {
    let found: Vec<_> = infos.iter().filter(|n| n.kind == kind).collect();
    ensure(
        found.len() == 1,
        format!("{TARGET}: unsupported target profile: one {kind} required"),
    )?;
    Ok(found[0])
}
fn check_binding(node: &Info, manifest: &Manifest) -> Checked<()> {
    let (role, effect, completion) = match node.kind {
        "model" => ("model", "load", "before_install"),
        "execute" => ("grant", "scoped_write", COMPLETION),
        _ => return Ok(()),
    };
    let symbol = node.attr("implementation");
    let binding = manifest
        .bindings
        .get(symbol)
        .ok_or_else(|| format!("{}: missing binding @{symbol}", node.id()))?;
    ensure(
        !binding.version.is_empty()
            && binding.role == role
            && binding.effect == effect
            && binding.completion == completion,
        format!(
            "{}: @{symbol}: expected role={role}, effect={effect}, completion={completion}, nonempty version; provided {binding:?}",
            node.id()
        ),
    )
}
fn profile(ctx: &Context, infos: &[Info], manifest: &Manifest) -> Checked<Json> {
    ensure(
        manifest.schema == 1 && manifest.target == TARGET,
        "unsupported authority manifest/target",
    )?;
    ensure(
        ["calls", "inline"].contains(&manifest.lowering.as_str()),
        "unsupported owner lowering",
    )?;
    for node in infos {
        check_binding(node, manifest)?;
    }
    let session = one(infos, "session")?;
    let model = one(infos, "model")?;
    let unit = one(infos, "unit")?;
    let offered = one(infos, "offer")?;
    let grant = one(infos, "execute")?;
    let advance = one(infos, "advance")?;
    ensure(
        manifest
            .offered
            .iter()
            .any(|p| p == offered.attr("profile")),
        format!(
            "{}: target does not offer requested profile {}",
            grant.id(),
            offered.attr("profile")
        ),
    )?;
    unit.expect("parameter", "weight")?;
    let phases: Vec<_> = nested(infos, grant.op)
        .iter()
        .filter(|n| n.kind == "phase")
        .map(|n| json!([n.id(), n.attr("phase")]))
        .collect();
    ensure(
        operand(ctx, unit, 0) == result(ctx, model),
        "selected target model/unit differs",
    )?;
    Ok(
        json!({"session":session.id(),"model":model.attr("participant"),"unit":unit.attr("unit"),"parameter":unit.attr("parameter"),
        "initial":model.float("initial")?,"learning_rate":advance.float("learning_rate")?,"clip_norm":clip(advance)?,
        "grant":grant.id(),"advance":advance.id(),"phases":phases,"profile":offered.attr("profile"),
        "model_implementation":model.attr("implementation"),"grant_implementation":grant.attr("implementation"),"lowering":manifest.lowering}),
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
fn recipe(
    ctx: &mut Context,
    semantic: Ptr<Operation>,
    infos: &[Info],
    manifest: &Manifest,
) -> Ptr<Operation> {
    let encoded = serde_json::to_string(manifest).expect("serializable manifest");
    let plan = new::<PlanOp>(
        ctx,
        2,
        values(&[
            ("schema", "1"),
            ("target", TARGET),
            ("manifest", &encoded),
            ("owner_lowering", &manifest.lowering),
            (
                "protocol",
                "actual_return_then_verified_owner_handback_then_retained_step",
            ),
        ]),
    );
    let region = plan.deref(ctx).get_region(0);
    let block = region.deref(ctx).iter(ctx).next().expect("fresh block");
    semantic.insert_at_back(block, ctx);
    let mut declarations = vec![];
    for node in infos {
        let mut attributes = node.attributes.clone();
        attributes.insert("origin".into(), node.id().into());
        attributes.insert("kind".into(), node.kind.into());
        match node.kind {
            "session" => {
                attributes.insert("role".into(), "scope".into());
            }
            "model" | "execute" => {
                let binding = &manifest.bindings[node.attr("implementation")];
                attributes.insert("role".into(), "call".into());
                attributes.insert("version".into(), binding.version.clone());
                attributes.insert("effect".into(), binding.effect.clone());
                attributes.insert("binding_role".into(), binding.role.clone());
                attributes.insert("call_completion".into(), binding.completion.clone());
                if node.kind == "execute" {
                    attributes.insert(
                        "participant".into(),
                        definition(ctx, infos, operand(ctx, node, 0), "model")
                            .expect("verified model")
                            .attr("participant")
                            .into(),
                    );
                    attributes.insert(
                        "unit".into(),
                        definition(ctx, infos, operand(ctx, node, 1), "unit")
                            .expect("verified unit")
                            .attr("unit")
                            .into(),
                    );
                    attributes.insert(
                        "protection".into(),
                        "current_write_through_physical_return_and_handback".into(),
                    );
                }
            }
            "unit" => {
                attributes.insert("role".into(), "optimization_unit".into());
                attributes.insert(
                    "participant".into(),
                    definition(ctx, infos, operand(ctx, node, 0), "model")
                        .expect("verified model")
                        .attr("participant")
                        .into(),
                );
            }
            "offer" => {
                attributes.insert("role".into(), "offered_profile_requirement".into());
            }
            "phase" => {
                attributes.insert("role".into(), "delegated_phase_contract".into());
            }
            "yield" => {
                attributes.insert("role".into(), "handback_claim".into());
                attributes.insert("authority".into(), "not_an_owner_verified_token".into());
            }
            "advance" => {
                attributes.insert("role".into(), "retained_advance".into());
                attributes.insert(
                    "requires".into(),
                    "owner_verified_handback_not_restoration".into(),
                );
                attributes.insert(
                    "source".into(),
                    definition(ctx, infos, operand(ctx, node, 0), "execute")
                        .expect("verified grant")
                        .id()
                        .into(),
                );
                attributes.insert(
                    "unit".into(),
                    definition(ctx, infos, operand(ctx, node, 1), "unit")
                        .expect("verified unit")
                        .attr("unit")
                        .into(),
                );
            }
            "finish" => {
                attributes.insert("role".into(), "settle_retained_outcomes".into());
            }
            _ => unreachable!("verified contract"),
        }
        declarations.push(new::<DeclarationOp>(ctx, 0, attributes));
    }
    let region = plan.deref(ctx).get_region(1);
    let block = region.deref(ctx).iter(ctx).next().expect("fresh block");
    for declaration in declarations {
        declaration.insert_at_back(block, ctx);
    }
    plan
}
pub fn prepare(text: &str, saved: Option<&str>, manifest: Manifest) -> Checked<Json> {
    let ctx = &mut Context::new();
    let semantic = parse(ctx, text)?;
    let infos = collect(ctx, semantic)?;
    verify(ctx, &infos)?;
    let target = profile(ctx, &infos, &manifest)?;
    let expected = recipe(ctx, semantic, &infos, &manifest);
    verify_operation(expected, ctx).map_err(|e| e.disp(ctx).to_string())?;
    let expected_text = expected.deref(ctx).disp(ctx).to_string();
    let restored = parse(ctx, &expected_text)?;
    ensure(
        equivalent(ctx, expected, restored),
        "prepared authority structural round-trip differs",
    )?;
    if let Some(text) = saved {
        let restored = parse(ctx, text)?;
        ensure(
            equivalent(ctx, expected, restored),
            "cold authority plan: semantic/source/authority/binding correspondence differs",
        )?;
    }
    Ok(
        json!({"schema":1,"pliron":"0.20.0","target":TARGET,"accepted":text,"prepared":saved.unwrap_or(&expected_text),
        "accepted_sha256":format!("{:x}",Sha256::digest(text.as_bytes())),"manifest":manifest,"profile":target}),
    )
}
