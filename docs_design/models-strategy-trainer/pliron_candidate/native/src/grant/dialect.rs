//! Typed C authority claims. An IR handback is not a verified runtime token.
use pliron::{
    builtin::op_interfaces::{
        IsTerminatorInterface, IsolatedFromAboveInterface, NOpdsInterface, NRegionsInterface,
        NResultsInterface, NoTerminatorInterface, OneRegionInterface, RegionKind,
        RegionKindInterface, SingleBlockRegionInterface,
    },
    context::Context,
    derive::{op_interface, op_interface_impl, pliron_op, pliron_type},
    op::Op,
    result::Result,
};

macro_rules! ty {
    ($name:ident, $id:literal) => {
        #[pliron_type(name = $id, format, generate_get = true, verifier = "succ")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name;
    };
}
ty!(ModelType, "ga.model");
ty!(UnitType, "ga.unit");
ty!(ProfileType, "ga.profile");
ty!(AccessType, "ga.access");
ty!(HandbackType, "ga.handback");
ty!(DoneType, "ga.done");

pub struct Shape {
    pub kind: &'static str,
    pub attributes: &'static [&'static str],
    pub inputs: &'static [&'static str],
    pub outputs: &'static [&'static str],
    pub arguments: Option<&'static [&'static str]>,
    pub variadic: bool,
}
#[op_interface]
pub trait AuthorityContract {
    fn shape(&self) -> Shape;
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
macro_rules! leaf {
    ($name:ident, $id:literal) => {
        #[pliron_op(name = $id, format, verifier = "succ", interfaces = [NRegionsInterface<0>])]
        pub struct $name;
    };
}
#[pliron_op(name = "ga.session", format, verifier = "succ", interfaces = [OneRegionInterface,
    SingleBlockRegionInterface, IsolatedFromAboveInterface])]
pub struct SessionOp;
#[pliron_op(name = "ga.execute", format, verifier = "succ", interfaces = [OneRegionInterface,
    SingleBlockRegionInterface, IsolatedFromAboveInterface])]
pub struct ExecuteOp;
#[op_interface_impl]
impl RegionKindInterface for SessionOp {
    fn get_region_kind(&self, _idx: usize) -> RegionKind {
        RegionKind::SSACFG
    }
}
#[op_interface_impl]
impl RegionKindInterface for ExecuteOp {
    fn get_region_kind(&self, _idx: usize) -> RegionKind {
        RegionKind::SSACFG
    }
}
leaf!(ModelOp, "ga.model");
leaf!(UnitOp, "ga.unit");
leaf!(OfferOp, "ga.offer");
leaf!(PhaseOp, "ga.phase");
leaf!(AdvanceOp, "ga.advance");
#[pliron_op(name = "ga.yield", format, verifier = "succ", interfaces = [NRegionsInterface<0>, IsTerminatorInterface])]
pub struct YieldOp;
#[pliron_op(name = "ga.finish", format, verifier = "succ", interfaces = [NRegionsInterface<0>, IsTerminatorInterface])]
pub struct FinishOp;

macro_rules! shape {
    ($name:ident, $kind:literal, $attrs:expr, $inputs:expr, $outputs:expr, $args:expr, $variadic:expr) => {
        #[op_interface_impl]
        impl AuthorityContract for $name {
            fn shape(&self) -> Shape {
                Shape {
                    kind: $kind,
                    attributes: $attrs,
                    inputs: $inputs,
                    outputs: $outputs,
                    arguments: $args,
                    variadic: $variadic,
                }
            }
        }
    };
}
shape!(
    SessionOp,
    "session",
    &["source_id", "version", "failure"],
    &[],
    &[],
    Some(&[]),
    false
);
shape!(
    ModelOp,
    "model",
    &[
        "source_id",
        "participant",
        "initial",
        "treatment",
        "implementation"
    ],
    &[],
    &["ga.model"],
    None,
    false
);
shape!(
    UnitOp,
    "unit",
    &["source_id", "unit", "parameter"],
    &["ga.model"],
    &["ga.unit"],
    None,
    false
);
shape!(
    OfferOp,
    "offer",
    &["source_id", "profile", "delegated", "retained"],
    &[],
    &["ga.profile"],
    None,
    false
);
shape!(
    ExecuteOp,
    "execute",
    &["source_id", "implementation", "completion", "handback"],
    &["ga.model", "ga.unit", "ga.profile"],
    &["ga.handback"],
    Some(&["ga.access"]),
    false
);
shape!(
    PhaseOp,
    "phase",
    &["source_id", "phase"],
    &["ga.access"],
    &["ga.access"],
    None,
    false
);
shape!(
    YieldOp,
    "yield",
    &["source_id"],
    &["ga.access"],
    &[],
    None,
    false
);
shape!(
    AdvanceOp,
    "advance",
    &["source_id", "learning_rate", "clip_norm", "authority"],
    &["ga.handback", "ga.unit"],
    &["ga.done"],
    None,
    false
);
shape!(
    FinishOp,
    "finish",
    &["source_id"],
    &["ga.done"],
    &[],
    None,
    true
);

#[pliron_op(name = "ga_exec.plan", format, verifier = "succ", interfaces = [NRegionsInterface<2>,
    SingleBlockRegionInterface, IsolatedFromAboveInterface, NoTerminatorInterface,
    NOpdsInterface<0>, NResultsInterface<0>])]
pub struct PlanOp;
#[op_interface_impl]
impl RegionKindInterface for PlanOp {
    fn get_region_kind(&self, _idx: usize) -> RegionKind {
        RegionKind::Graph
    }
}
#[pliron_op(name = "ga_exec.declaration", format, verifier = "succ", interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>])]
pub struct DeclarationOp;
