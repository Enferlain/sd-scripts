//! Independently registered schema and operations; no core operation-name dispatch.
use crate::{
    contract::{self, BASE_OUTCOMES, LocalDescription, PayloadTypeInterface, ResourceUse},
    dialect::{LocalStageInterface, RecordType},
};
use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NRegionsInterface, OneResultInterface},
    context::Context,
    derive::{op_interface_impl, pliron_op, pliron_type, type_interface_impl},
    op::Op,
    printable::Printable,
    r#type::{TypeHandle, Typed},
};

#[pliron_type(name = "pairing.pair", format, generate_get = true, verifier = "succ")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PairType;
#[type_interface_impl]
impl PayloadTypeInterface for PairType {
    fn schema(&self) -> &'static str {
        "pair-int2"
    }
    fn codec(&self) -> &'static str {
        "pair-int2-v1"
    }
}
#[pliron_op(name="pairing.pack",format,verifier="succ",interfaces=[NOpdsInterface<1>,OneResultInterface,NRegionsInterface<0>])]
pub struct PackOp;
#[pliron_op(name="pairing.first",format,verifier="succ",interfaces=[NOpdsInterface<1>,OneResultInterface,NRegionsInterface<0>])]
pub struct FirstOp;

fn typed(ctx: &Context, op: &dyn Op, input: TypeHandle, output: TypeHandle) -> Result<(), String> {
    let operation = op.get_operation();
    if operation.deref(ctx).get_operand(0).get_type(ctx) != input
        || operation.deref(ctx).get_result(0).get_type(ctx) != output
    {
        return Err(format!(
            "{}: incompatible extension input/output schema; expected {} -> {}, provided {} -> {}",
            contract::required(ctx, op, "source_id")?,
            input.disp(ctx),
            output.disp(ctx),
            operation.deref(ctx).get_operand(0).get_type(ctx).disp(ctx),
            operation.deref(ctx).get_result(0).get_type(ctx).disp(ctx)
        ));
    }
    Ok(())
}
#[op_interface_impl]
impl LocalStageInterface for PackOp {
    fn describe(&self, ctx: &Context) -> Result<LocalDescription, String> {
        let input = RecordType::get(ctx).into();
        let output = PairType::get(ctx).into();
        typed(ctx, self, input, output)?;
        let effect = contract::required(ctx, self, "effect")?;
        if effect != "read_frozen" {
            return Err(format!(
                "{}: resource @{} pairing.pack requires read_frozen access, provided {}",
                contract::required(ctx, self, "source_id")?,
                contract::required(ctx, self, "resource")?,
                effect
            ));
        }
        let resource = ResourceUse {
            symbol: contract::required(ctx, self, "resource")?,
            access: "read_frozen".into(),
            version: contract::required(ctx, self, "resource_version")?,
            digest: contract::required(ctx, self, "resource_digest")?,
        };
        Ok(LocalDescription {
            input,
            output,
            cardinality: "one-to-one",
            resources: Some(vec![resource]),
            outcomes: BASE_OUTCOMES
                .iter()
                .map(|s| s.to_string())
                .chain(["negative_input".into()])
                .collect(),
            attributes: vec![
                "source_id",
                "implementation",
                "effect",
                "resource",
                "resource_version",
                "resource_digest",
            ],
        })
    }
}
#[op_interface_impl]
impl LocalStageInterface for FirstOp {
    fn describe(&self, ctx: &Context) -> Result<LocalDescription, String> {
        let input = PairType::get(ctx).into();
        let output = RecordType::get(ctx).into();
        typed(ctx, self, input, output)?;
        if contract::required(ctx, self, "effect")? != "pure" {
            return Err(format!(
                "{}: pairing.first requires pure access",
                contract::required(ctx, self, "source_id")?
            ));
        }
        Ok(LocalDescription {
            input,
            output,
            cardinality: "one-to-one",
            resources: Some(vec![]),
            outcomes: BASE_OUTCOMES.iter().map(|s| s.to_string()).collect(),
            attributes: vec!["source_id", "implementation", "effect"],
        })
    }
}
