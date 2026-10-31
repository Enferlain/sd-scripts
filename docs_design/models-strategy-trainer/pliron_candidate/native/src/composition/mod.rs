//! Composition acceptance and the bounded existing-owner target.
mod dialect;

use dialect::CompositionContract;
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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

type Checked<T> = Result<T, String>;
const TARGET: &str = "existing-owner-composition-cpu-v1";

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
    pub bindings: BTreeMap<String, Binding>,
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct Endpoint {
    source: String,
    result: usize,
    r#type: String,
}
#[derive(Clone, Debug, Serialize)]
struct Node {
    id: String,
    parent: String,
    kind: String,
    attributes: BTreeMap<String, String>,
    inputs: Vec<Endpoint>,
    outputs: Vec<String>,
}
impl Node {
    fn attr(&self, key: &str) -> &str {
        &self.attributes[key]
    }
    fn number(&self, key: &str) -> Checked<usize> {
        self.attr(key)
            .parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| format!("{}: {key} must be a positive integer", self.id))
    }
    fn floating(&self, key: &str) -> Checked<f64> {
        self.attr(key)
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("{}: {key} must be finite", self.id))
    }
    fn value(&self, key: &str, expected: &str) -> Checked<()> {
        ensure(
            self.attr(key) == expected,
            format!(
                "{}: {key}: expected {expected}, provided {}",
                self.id,
                self.attr(key)
            ),
        )
    }
    fn end(&self, result: usize) -> Endpoint {
        Endpoint {
            source: self.id.clone(),
            result,
            r#type: self.outputs[result].clone(),
        }
    }
}
fn ensure(ok: bool, message: impl Into<String>) -> Checked<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn nested(ctx: &Context, op: Ptr<Operation>) -> Checked<Vec<Ptr<Operation>>> {
    let mut children = vec![];
    for region in op.deref(ctx).regions() {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        ensure(
            blocks.len() == 1,
            "composition requires one block per region",
        )?;
        children.extend(blocks[0].deref(ctx).iter(ctx));
    }
    Ok(children)
}
fn type_name(ctx: &Context, value: Value) -> String {
    value.get_type(ctx).disp(ctx).to_string().trim().into()
}

/// Inspect registered interfaces and actual SSA values, including captured
/// region arguments. Names printed for SSA values have no authority.
fn collect(ctx: &Context, root: Ptr<Operation>) -> Checked<Vec<Node>> {
    let mut nodes = vec![];
    let mut operations = vec![];
    let mut definitions = HashMap::<Value, Endpoint>::new();
    let mut aliases = HashMap::<Value, Value>::new();
    fn visit(
        ctx: &Context,
        op: Ptr<Operation>,
        parent: &str,
        nodes: &mut Vec<Node>,
        operations: &mut Vec<Ptr<Operation>>,
        definitions: &mut HashMap<Value, Endpoint>,
        aliases: &mut HashMap<Value, Value>,
    ) -> Checked<()> {
        let object = Operation::get_op_dyn(op, ctx);
        let interface = op_cast::<dyn CompositionContract>(&*object).ok_or_else(|| {
            format!(
                "unsupported semantic operation {}",
                Operation::get_opid(op, ctx)
            )
        })?;
        let description = interface.contract();
        let attributes = contract::strings(ctx, op)?;
        let expected: BTreeSet<_> = description.attributes.iter().copied().collect();
        ensure(
            attributes
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == expected,
            format!(
                "{}: missing or unknown {} attributes",
                attributes
                    .get("source_id")
                    .map(String::as_str)
                    .unwrap_or("operation"),
                description.kind
            ),
        )?;
        let id = attributes["source_id"].clone();
        ensure(
            !id.is_empty() && nodes.iter().all(|node| node.id != id),
            format!("{id}: empty or duplicate source identity"),
        )?;
        let operation = op.deref(ctx);
        let inputs: Vec<_> = (0..operation.get_num_operands())
            .map(|i| type_name(ctx, operation.get_operand(i)))
            .collect();
        ensure(
            inputs.len() >= description.inputs.len()
                && inputs.len() - description.inputs.len() <= description.max_tail,
            format!("{id}: operand count differs from its interface"),
        )?;
        for (i, ty) in inputs.iter().enumerate() {
            let expected = description
                .inputs
                .get(i)
                .copied()
                .or(description.tail)
                .ok_or_else(|| format!("{id}: unexpected operand"))?;
            ensure(
                ty == expected,
                format!("{id}: operand {i}: expected {expected}, provided {ty}"),
            )?;
        }
        let outputs: Vec<_> = (0..operation.get_num_results())
            .map(|i| type_name(ctx, operation.get_result(i)))
            .collect();
        ensure(
            outputs.iter().map(String::as_str).collect::<Vec<_>>() == description.outputs,
            format!(
                "{id}: result types differ: expected {:?}, provided {outputs:?}",
                description.outputs
            ),
        )?;
        ensure(
            operation.regions().count() == usize::from(description.region.is_some()),
            format!("{id}: region count differs"),
        )?;
        for (i, ty) in outputs.iter().enumerate() {
            definitions.insert(
                operation.get_result(i),
                Endpoint {
                    source: id.clone(),
                    result: i,
                    r#type: ty.clone(),
                },
            );
        }
        if description.region.is_some() {
            let region = operation.get_region(0);
            let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
            ensure(
                blocks.len() == 1,
                format!("{id}: one region block required"),
            )?;
            let block = blocks[0].deref(ctx);
            let argument_count = if description.arguments {
                inputs.len()
            } else {
                0
            };
            ensure(
                block.get_num_arguments() == argument_count,
                format!("{id}: region arguments differ"),
            )?;
            for (i, input) in inputs.iter().enumerate().take(argument_count) {
                let argument = block.get_argument(i);
                ensure(
                    type_name(ctx, argument) == *input,
                    format!("{id}: region argument type differs"),
                )?;
                aliases.insert(argument, operation.get_operand(i));
            }
        }
        nodes.push(Node {
            id: id.clone(),
            parent: parent.into(),
            kind: description.kind.into(),
            attributes,
            inputs: vec![],
            outputs,
        });
        operations.push(op);
        drop(operation);
        for child in nested(ctx, op)? {
            visit(ctx, child, &id, nodes, operations, definitions, aliases)?;
        }
        Ok(())
    }
    visit(
        ctx,
        root,
        "",
        &mut nodes,
        &mut operations,
        &mut definitions,
        &mut aliases,
    )?;
    for (node, op) in nodes.iter_mut().zip(operations) {
        let operation = op.deref(ctx);
        for i in 0..operation.get_num_operands() {
            let mut value = operation.get_operand(i);
            let mut seen = HashSet::new();
            while let Some(alias) = aliases.get(&value) {
                ensure(seen.insert(value), "cyclic region argument aliases")?;
                value = *alias;
            }
            node.inputs.push(
                definitions
                    .get(&value)
                    .cloned()
                    .ok_or_else(|| format!("{}: value outside accepted source tree", node.id))?,
            );
        }
    }
    Ok(nodes)
}
fn children<'a>(nodes: &'a [Node], parent: &str) -> Vec<&'a Node> {
    nodes.iter().filter(|node| node.parent == parent).collect()
}
fn lookup<'a>(nodes: &'a [Node], endpoint: &Endpoint, kind: &str) -> Checked<&'a Node> {
    nodes
        .iter()
        .find(|node| node.id == endpoint.source && node.kind == kind)
        .ok_or_else(|| format!("{}: expected {kind} endpoint", endpoint.source))
}
fn enclosed(nodes: &[Node], child: &str, ancestor: &str) -> bool {
    let mut current = child;
    while let Some(node) = nodes.iter().find(|node| node.id == current) {
        if node.id == ancestor {
            return true;
        }
        current = &node.parent;
    }
    false
}
fn verify_region(nodes: &[Node], owner: &Node, terminator: &str, allowed: &[&str]) -> Checked<()> {
    let nested = children(nodes, &owner.id);
    ensure(
        nested.last().is_some_and(|n| n.kind == terminator),
        format!("{}: missing final {terminator}", owner.id),
    )?;
    ensure(
        nested.iter().all(|n| allowed.contains(&n.kind.as_str())),
        format!("{}: unsupported nested activity", owner.id),
    )
}
fn verify(nodes: &[Node]) -> Checked<()> {
    let root = &nodes[0];
    ensure(root.kind == "run", "expected one lc.run document")?;
    root.value("version", "1")?;
    root.value("failure", "cancel_siblings_and_drain")?;
    verify_region(
        nodes,
        root,
        "finish",
        &["model", "producer", "group", "finish"],
    )?;
    for node in nodes {
        match node.kind.as_str() {
            "run" => ensure(node.parent.is_empty(), "nested runs are unsupported")?,
            "model" => {
                ensure(
                    node.parent == root.id,
                    format!("{}: model must be run-owned", node.id),
                )?;
                ensure(
                    ["frozen", "trainable"].contains(&node.attr("treatment")),
                    format!("{}: unknown treatment", node.id),
                )?;
                node.floating("initial")?;
            }
            "producer" => {
                ensure(
                    node.parent == root.id,
                    format!("{}: producer must be run-owned", node.id),
                )?;
                lookup(nodes, &node.inputs[0], "model")?;
                let stop = lookup(nodes, &node.inputs[1], "group")?;
                ensure(
                    stop.parent == root.id,
                    format!(
                        "{}: stop boundary must be a sibling consumer group",
                        node.id
                    ),
                )?;
                for consumer in nodes
                    .iter()
                    .filter(|n| ["repeat", "evaluator"].contains(&n.kind.as_str()))
                {
                    if consumer.inputs.iter().any(|e| *e == node.end(0)) {
                        ensure(
                            enclosed(nodes, &consumer.id, &stop.id),
                            format!(
                                "{}: stop boundary leaves consumer {} active",
                                node.id, consumer.id
                            ),
                        )?;
                    }
                }
                node.number("horizon")?;
                node.number("ready")?;
                node.number("workers")?;
                node.value("remaining", "discard_unclaimed")?;
                node.value("failure", "close_then_drain")?;
            }
            "group" => {
                node.value("completion", "children_physically_settled")?;
                node.value("pending", "include_requested")?;
                node.value("failure", "cancel_siblings_and_drain")?;
                verify_region(
                    nodes,
                    node,
                    "settle",
                    &["repeat", "request", "evaluator", "group", "settle"],
                )?;
            }
            "repeat" => {
                ensure(
                    nodes
                        .iter()
                        .any(|n| n.id == node.parent && n.kind == "group"),
                    format!("{}: repeat requires consumer group", node.id),
                )?;
                lookup(nodes, &node.inputs[0], "producer")?;
                let model = lookup(nodes, &node.inputs[1], "model")?;
                model.value("treatment", "trainable")?;
                node.number("max_windows")?;
                node.value("admission", "policy_choice_before_first_claim")?;
                node.value("stop", "finish_selected_window")?;
                let body = children(nodes, &node.id);
                verify_region(
                    nodes,
                    node,
                    "window_end",
                    &["take", "contribute", "advance", "window_end"],
                )?;
                let count = body.iter().filter(|n| n.kind == "take").count();
                ensure(
                    count > 0 && body.len() == 2 * count + 2,
                    format!("{}: incomplete contribution window", node.id),
                )?;
                let mut previous = None;
                for slot in 0..count {
                    let take = body[2 * slot];
                    let contribute = body[2 * slot + 1];
                    ensure(
                        take.kind == "take" && contribute.kind == "contribute",
                        format!("{}: ordered take/contribution pair required", node.id),
                    )?;
                    take.value("selection", &format!("slot:{slot}"))?;
                    take.value("provenance", "actual_frozen_source")?;
                    ensure(
                        take.inputs == [node.inputs[0].clone()],
                        format!("{}: stream differs", take.id),
                    )?;
                    let mut expected = vec![take.end(0), node.inputs[1].clone()];
                    if let Some(gradient) = previous {
                        expected.push(gradient);
                    }
                    ensure(
                        contribute.inputs == expected,
                        format!("{}: gradient/input chain differs", contribute.id),
                    )?;
                    ensure(
                        contribute.number("normalization")? == count,
                        format!(
                            "{}: normalization differs from window cardinality",
                            contribute.id
                        ),
                    )?;
                    previous = Some(contribute.end(0));
                }
                let advance = body[2 * count];
                let end = body[2 * count + 1];
                ensure(
                    advance.kind == "advance"
                        && advance.inputs == [previous.unwrap(), node.inputs[1].clone()],
                    format!("{}: advance does not consume the complete window", node.id),
                )?;
                advance.value("authority", "ordinary_owner")?;
                ensure(
                    advance.floating("learning_rate")? > 0.0,
                    "positive learning rate required",
                )?;
                ensure(
                    end.inputs == [advance.end(0)],
                    format!("{}: completion does not follow advancement", node.id),
                )?;
            }
            "request" => {
                let source = lookup(nodes, &node.inputs[0], "repeat")?;
                ensure(
                    source.parent == node.parent,
                    format!("{}: request source outside its consumer group", node.id),
                )?;
                ensure(
                    node.number("after_window")? <= source.number("max_windows")?,
                    format!("{}: request position outside repeat bounds", node.id),
                )?;
                node.value("multiplicity", "once_per_matching_completion")?;
                node.value("pending", "keeps_group_alive")?;
                node.value("max_pending", "1")?;
                node.value("cancellation", "cancel_or_unfulfilled")?;
                ensure(
                    nodes
                        .iter()
                        .filter(|n| n.kind == "evaluator" && n.inputs[0] == node.end(0))
                        .count()
                        == 1,
                    format!(
                        "{}: request must have exactly one selected recipient",
                        node.id
                    ),
                )?;
            }
            "evaluator" => {
                let request = lookup(nodes, &node.inputs[0], "request")?;
                let source = lookup(nodes, &request.inputs[0], "repeat")?;
                ensure(
                    request.parent == node.parent && node.inputs[1..] == source.inputs[..],
                    format!("{}: request/stream/model association differs", node.id),
                )?;
                node.value("completion", "physical_and_cleanup")?;
                let body = children(nodes, &node.id);
                verify_region(
                    nodes,
                    node,
                    "evaluation_end",
                    &["take", "read", "evaluation_end"],
                )?;
                ensure(
                    body.len() == 3 && body[0].kind == "take" && body[1].kind == "read",
                    format!("{}: one take/read evaluation required", node.id),
                )?;
                body[0].value("selection", "after_trigger_members")?;
                body[0].value("provenance", "actual_frozen_source")?;
                ensure(
                    body[0].inputs == [node.inputs[1].clone()]
                        && body[1].inputs == [body[0].end(0), node.inputs[2].clone()]
                        && body[2].inputs == [body[1].end(0)],
                    format!("{}: evaluation value chain differs", node.id),
                )?;
                body[1].value("state", "current_at_start")?;
                body[1].value("mode", "evaluation")?;
                body[1].value("completion", "physical_validate_restore")?;
            }
            "finish" | "settle" => {
                let parent = nodes
                    .iter()
                    .find(|n| n.id == node.parent)
                    .ok_or("missing settlement parent")?;
                let expected: BTreeSet<_> = children(nodes, &parent.id)
                    .iter()
                    .filter_map(|n| match n.kind.as_str() {
                        "producer" | "repeat" => Some(n.end(1)),
                        "group" | "evaluator" => Some(n.end(0)),
                        _ => None,
                    })
                    .collect();
                ensure(
                    !expected.is_empty()
                        && node.inputs.len() == expected.len()
                        && node.inputs.iter().cloned().collect::<BTreeSet<_>>() == expected,
                    format!(
                        "{}: settlement must include all and only direct child obligations",
                        node.id
                    ),
                )?;
            }
            "take" | "contribute" | "advance" | "read" | "window_end" | "evaluation_end" => {
                ensure(
                    nodes.iter().any(|n| {
                        n.id == node.parent && ["repeat", "evaluator"].contains(&n.kind.as_str())
                    }),
                    format!("{}: local operation outside an activity", node.id),
                )?;
            }
            _ => return Err(format!("{}: unknown composition interface", node.id)),
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
    let op = parse(ctx, text)?;
    verify(&collect(ctx, op)?)?;
    let canonical = op.deref(ctx).disp(ctx).to_string();
    let restored = parse(ctx, &canonical)?;
    ensure(
        equivalent(ctx, op, restored),
        "composition structural round-trip differs",
    )?;
    let fresh = &mut Context::new();
    let restored = parse(fresh, &canonical)?;
    verify(&collect(fresh, restored)?)?;
    Ok(canonical)
}
fn binding_kind(kind: &str) -> Option<(&str, &str, &str)> {
    match kind {
        "model" => Some(("model", "load", "before_install")),
        "producer" => Some(("produce", "read_source", "physical_return_before_release")),
        "contribute" => Some(("objective", "build_loss", "return_before_backward")),
        "read" => Some(("evaluate", "protected_read", "physical_validate_restore")),
        _ => None,
    }
}
fn check_manifest(nodes: &[Node], manifest: &Manifest) -> Checked<()> {
    ensure(
        manifest.schema == 1 && manifest.target == TARGET,
        "unsupported composition manifest/target",
    )?;
    for node in nodes {
        if let Some((role, effect, completion)) = binding_kind(&node.kind) {
            let symbol = node.attr("implementation");
            let binding = manifest
                .bindings
                .get(symbol)
                .ok_or_else(|| format!("{}: missing binding @{symbol}", node.id))?;
            ensure(
                binding.version == "1"
                    && binding.role == role
                    && binding.effect == effect
                    && binding.completion == completion,
                format!(
                    "{}: @{symbol}: expected role={role}, effect={effect}, completion={completion}, version=1; provided {binding:?}",
                    node.id
                ),
            )?;
        }
    }
    Ok(())
}
fn one<'a>(nodes: &'a [Node], kind: &str) -> Checked<&'a Node> {
    let found: Vec<_> = nodes.iter().filter(|n| n.kind == kind).collect();
    ensure(
        found.len() == 1,
        format!("{TARGET}: unsupported target profile: one {kind} required"),
    )?;
    Ok(found[0])
}
fn profile(nodes: &[Node]) -> Checked<Json> {
    let run = one(nodes, "run")?;
    let producer = one(nodes, "producer")?;
    let group = one(nodes, "group")?;
    let training = one(nodes, "repeat")?;
    let evaluator = one(nodes, "evaluator")?;
    let request = one(nodes, "request")?;
    let advance = one(nodes, "advance")?;
    let read = one(nodes, "read")?;
    let contributions: Vec<_> = children(nodes, &training.id)
        .into_iter()
        .filter(|n| n.kind == "contribute")
        .collect();
    ensure(
        training.number("max_windows")? == 2
            && contributions.len() == 2
            && request.number("after_window")? == 2,
        format!("{TARGET}: unsupported target window/request bounds"),
    )?;
    ensure(
        producer.number("horizon")? == 3
            && producer.number("ready")? == 2
            && producer.number("workers")? == 1,
        format!("{TARGET}: unsupported aggregate/ready/physical bounds"),
    )?;
    ensure(
        advance.attr("parameter") == "weight",
        format!("{}: target supports scalar weight", advance.id),
    )?;
    let encoder = lookup(nodes, &producer.inputs[0], "model")?;
    encoder.value("treatment", "frozen")?;
    ensure(
        nodes.iter().filter(|n| n.kind == "model").count() == 2,
        format!("{TARGET}: unsupported model count"),
    )?;
    ensure(
        contributions
            .iter()
            .all(|n| n.attr("implementation") == contributions[0].attr("implementation")),
        format!("{TARGET}: unsupported heterogeneous objective bindings"),
    )?;
    Ok(
        json!({"run":run.id,"producer":producer.id,"group":group.id,"training":training.id,"evaluation":evaluator.id,
        "encoder":encoder.id,"learner":training.inputs[1].source,"horizon":3,"ready":2,"workers":1,"members":2,"windows":2,"after_window":2,
        "learning_rate":advance.floating("learning_rate")?,
        "ports":(0..2).map(|slot|format!("{}.slot.{slot}",producer.id)).collect::<Vec<_>>(),
        "evaluation_port":format!("{}.evaluation",producer.id),"unused_port":format!("{}.unused",producer.id),
        "models":nodes.iter().filter(|n| n.kind=="model").map(|n| Ok(json!({"id":n.id,"initial":n.floating("initial")?,
            "trainable":n.attr("treatment")=="trainable","implementation":n.attr("implementation")}))).collect::<Checked<Vec<_>>>()?,
        "calls":{"encode":producer.attr("implementation"),"objective":contributions[0].attr("implementation"),"measure":read.attr("implementation")}}),
    )
}
fn recipe(name: &str, values: BTreeMap<String, String>) -> (String, BTreeMap<String, String>) {
    (name.into(), values)
}
fn map(values: &[(&str, &str)]) -> BTreeMap<String, String> {
    values
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
            Identifier::try_from(key).expect("static schema key"),
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
fn prepared_plan(
    ctx: &mut Context,
    semantic: Ptr<Operation>,
    nodes: &[Node],
    manifest: &Manifest,
) -> Ptr<Operation> {
    let mut recipes = vec![];
    for node in nodes {
        if ["run", "producer", "group", "repeat", "evaluator"].contains(&node.kind.as_str()) {
            let mut values = node.attributes.clone();
            values.insert("origin".into(), node.id.clone());
            values.insert("parent".into(), node.parent.clone());
            values.insert("kind".into(), node.kind.clone());
            recipes.push(recipe("scope", values));
        }
        if let Some((role, effect, completion)) = binding_kind(&node.kind) {
            let symbol = node.attr("implementation");
            recipes.push(recipe(
                "call",
                map(&[
                    ("origin", &node.id),
                    ("symbol", symbol),
                    ("version", &manifest.bindings[symbol].version),
                    ("role", role),
                    ("effect", effect),
                    ("completion", completion),
                ]),
            ));
        }
        let access = match node.kind.as_str() {
            "producer" => Some((0, "read_source", "physical_return")),
            "contribute" => Some((1, "read_gradient", "backward")),
            "advance" => Some((1, "write", "update_outcome")),
            "read" => Some((1, "read_evaluation", "physical_and_restore")),
            _ => None,
        };
        if let Some((operand, mode, lifetime)) = access {
            recipes.push(recipe(
                "access",
                map(&[
                    ("origin", &node.id),
                    ("participant", &node.inputs[operand].source),
                    ("mode", mode),
                    ("through", lifetime),
                ]),
            ));
        }
        if node.kind == "request" {
            let mut values = node.attributes.clone();
            values.insert("origin".into(), node.id.clone());
            values.insert("source".into(), node.inputs[0].source.clone());
            recipes.push(recipe("request", values));
        }
        if node.kind == "producer" {
            recipes.push(recipe(
                "link",
                map(&[
                    ("source", &node.inputs[1].source),
                    ("target", &node.id),
                    ("relation", "normal_group_end_requests_stop"),
                    ("failure", "enclosing_cancel_then_physical_drain"),
                    ("run_completion", "producer_and_group_settled"),
                ]),
            ));
            for role in ["slot:0", "slot:1", "evaluation", "unused"] {
                let channel = if let Some(slot) = role.strip_prefix("slot:") {
                    format!("{}.slot.{slot}", node.id)
                } else {
                    format!("{}.{role}", node.id)
                };
                recipes.push(recipe(
                    "port",
                    map(&[
                        ("producer", &node.id),
                        ("role", role),
                        ("identity", "actual_record_and_source"),
                        ("channel", &channel),
                        ("ready_budget", "aggregate_provider_budget"),
                        ("settlement", "actual_use_then_notification"),
                    ]),
                ));
            }
        }
    }
    let plan = new::<dialect::PlanOp>(
        ctx,
        2,
        map(&[
            ("schema", "1"),
            ("target", TARGET),
            ("completion", "group_then_producer_then_run"),
            (
                "binding_manifest",
                &serde_json::to_string(manifest).unwrap(),
            ),
        ]),
    );
    let semantic_region = plan.deref(ctx).get_region(0);
    let semantic_block = semantic_region.deref(ctx).iter(ctx).next().unwrap();
    semantic.insert_at_back(semantic_block, ctx);
    let recipe_region = plan.deref(ctx).get_region(1);
    let recipe_block = recipe_region.deref(ctx).iter(ctx).next().unwrap();
    for (kind, attributes) in recipes {
        let op = match kind.as_str() {
            "scope" => new::<dialect::ScopeRecipe>(ctx, 0, attributes),
            "call" => new::<dialect::CallRecipe>(ctx, 0, attributes),
            "access" => new::<dialect::AccessRecipe>(ctx, 0, attributes),
            "request" => new::<dialect::RequestRecipe>(ctx, 0, attributes),
            "link" => new::<dialect::LinkRecipe>(ctx, 0, attributes),
            "port" => new::<dialect::PortRecipe>(ctx, 0, attributes),
            _ => unreachable!("closed target recipe constructors"),
        };
        op.insert_at_back(recipe_block, ctx);
    }
    plan
}
pub fn prepare(text: &str, saved: Option<&str>, manifest: Manifest) -> Checked<Json> {
    let ctx = &mut Context::new();
    let semantic = parse(ctx, text)?;
    let nodes = collect(ctx, semantic)?;
    verify(&nodes)?;
    check_manifest(&nodes, &manifest)?;
    let target = profile(&nodes)?;
    // Preserve the verified document. Printer-generated block names need not
    // be idempotent; structural equivalence, not text equality, authorizes it.
    let accepted = text.to_string();
    let expected = prepared_plan(ctx, semantic, &nodes, &manifest);
    verify_operation(expected, ctx).map_err(|e| e.disp(ctx).to_string())?;
    let expected_text = expected.deref(ctx).disp(ctx).to_string();
    let restored = parse(ctx, &expected_text)?;
    ensure(
        equivalent(ctx, expected, restored),
        "prepared composition structural round-trip differs",
    )?;
    if let Some(saved) = saved {
        let decoded = parse(ctx, saved)?;
        ensure(
            equivalent(ctx, expected, decoded),
            "cold composition plan: semantic/source/lifetime/binding correspondence differs",
        )?;
    }
    let prepared_text = saved.unwrap_or(&expected_text);
    Ok(
        json!({"schema":1,"pliron":"0.20.0","target":TARGET,"accepted":accepted,"prepared":prepared_text,
        "accepted_sha256":format!("{:x}",Sha256::digest(accepted.as_bytes())),"manifest":manifest,"nodes":nodes,"profile":target}),
    )
}
