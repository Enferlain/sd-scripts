//! Experimental shared contracts, independent of installed extension identities.
use pliron::{
    builtin::attributes::{GivenNamesAttr, StringAttr},
    context::{Context, Ptr},
    derive::type_interface,
    op::Op,
    operation::Operation,
    result::Result as IrResult,
    r#type::{Type, TypeHandle, type_cast},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const BASE_OUTCOMES: [&str; 3] = ["returned", "invalid_output", "raised"];
pub const RESERVED_OUTCOMES: [&str; 7] = [
    "returned",
    "invalid_output",
    "raised",
    "not_entered",
    "entered_failed",
    "invalid_input",
    "undeclared_outcome",
];
pub const INTEGER_CODEC: &str = "exact-integer-v1";

#[type_interface]
pub trait PayloadTypeInterface {
    fn schema(&self) -> &'static str;
    fn codec(&self) -> &'static str;
    fn verify(_ty: &dyn Type, _ctx: &Context) -> IrResult<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
pub fn payload(ctx: &Context, ty: TypeHandle) -> Result<(String, String), String> {
    let borrowed = ty.deref(ctx);
    let interface = type_cast::<dyn PayloadTypeInterface>(&*borrowed)
        .ok_or("missing local payload type interface")?;
    Ok((interface.schema().into(), interface.codec().into()))
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ResourceUse {
    pub symbol: String,
    pub access: String,
    pub version: String,
    pub digest: String,
}

pub struct LocalDescription {
    pub input: TypeHandle,
    pub output: TypeHandle,
    pub cardinality: &'static str,
    pub resources: Option<Vec<ResourceUse>>,
    pub outcomes: Vec<String>,
    pub attributes: Vec<&'static str>,
}
pub fn strings(ctx: &Context, op: Ptr<Operation>) -> Result<BTreeMap<String, String>, String> {
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
pub fn required(ctx: &Context, op: &dyn Op, key: &str) -> Result<String, String> {
    let attributes = strings(ctx, op.get_operation())?;
    attributes
        .get(key)
        .filter(|s| !s.is_empty())
        .cloned()
        .ok_or_else(|| {
            format!(
                "{}: missing {key}",
                attributes
                    .get("source_id")
                    .map(String::as_str)
                    .unwrap_or("local stage")
            )
        })
}
