//! Research-only vocabulary. Region kinds are framework interfaces; workflow
//! lifecycle and correspondence are checked separately in `prepare`.
use pliron::{
    builtin::op_interfaces::{
        IsTerminatorInterface, IsolatedFromAboveInterface, NOpdsInterface, NRegionsInterface,
        NResultsInterface, NoTerminatorInterface, OneRegionInterface, OneResultInterface,
        RegionKind, RegionKindInterface, SingleBlockRegionInterface,
    },
    context::Context,
    derive::{op_interface, op_interface_impl, pliron_op, pliron_type},
    op::Op,
    result::Result,
};

macro_rules! payload {
    ($name:ident, $id:literal) => {
        #[pliron_type(name = $id, format, generate_get = true, verifier = "succ")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name;
    };
}
payload!(RecordType, "wf.record");
payload!(ScoreType, "wf.score");
payload!(FeedbackType, "wf.feedback");
payload!(EntityType, "wf.entity");

#[pliron_op(name = "wf.session", format, verifier = "succ", interfaces = [
    OneRegionInterface, SingleBlockRegionInterface, IsolatedFromAboveInterface,
    NoTerminatorInterface, NOpdsInterface<0>, NResultsInterface<0>
])]
pub struct SessionOp;

#[op_interface_impl]
impl RegionKindInterface for SessionOp {
    fn get_region_kind(&self, _idx: usize) -> RegionKind {
        RegionKind::Graph
    }
}

#[pliron_op(name = "wf.entity", format, verifier = "succ", interfaces = [NOpdsInterface<0>, OneResultInterface, NRegionsInterface<0>])]
pub struct EntityOp;
#[pliron_op(name = "wf.flow", format, verifier = "succ", interfaces = [NOpdsInterface<0>, OneResultInterface, NRegionsInterface<0>])]
pub struct FlowOp;
#[pliron_op(name = "wf.select", format, verifier = "succ", interfaces = [NOpdsInterface<4>, NResultsInterface<0>, NRegionsInterface<0>])]
pub struct SelectOp;

#[pliron_op(name = "wf.compose", format, verifier = "succ", interfaces = [OneRegionInterface, SingleBlockRegionInterface, NOpdsInterface<2>, NResultsInterface<0>])]
pub struct ComposeOp;
#[pliron_op(name = "wf.local", format, verifier = "succ", interfaces = [OneRegionInterface, SingleBlockRegionInterface, NOpdsInterface<1>, OneResultInterface])]
pub struct LocalOp;
#[op_interface_impl]
impl RegionKindInterface for ComposeOp {
    fn get_region_kind(&self, _idx: usize) -> RegionKind {
        RegionKind::SSACFG
    }
}
#[op_interface_impl]
impl RegionKindInterface for LocalOp {
    fn get_region_kind(&self, _idx: usize) -> RegionKind {
        RegionKind::SSACFG
    }
}

/// The small unary interface being screened, not an extension ABI adoption.
#[op_interface]
pub trait LocalStageInterface {
    fn cardinality(&self) -> &'static str;
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
#[pliron_op(name = "wf.map", format, verifier = "succ", interfaces = [NOpdsInterface<1>, OneResultInterface, NRegionsInterface<0>])]
pub struct MapOp;
#[op_interface_impl]
impl LocalStageInterface for MapOp {
    fn cardinality(&self) -> &'static str {
        "one-to-one"
    }
}
#[pliron_op(name = "wf.yield", format, verifier = "succ", interfaces = [NOpdsInterface<1>, NResultsInterface<0>, NRegionsInterface<0>, IsTerminatorInterface])]
pub struct YieldOp;
#[pliron_op(name = "wf.score", format, verifier = "succ", interfaces = [NOpdsInterface<2>, NResultsInterface<0>, NRegionsInterface<0>])]
pub struct ScoreOp;
#[pliron_op(name = "wf.publish", format, verifier = "succ", interfaces = [NOpdsInterface<3>, NResultsInterface<0>, NRegionsInterface<0>])]
pub struct PublishOp;

#[pliron_op(name = "exec.plan", format, verifier = "succ", interfaces = [NRegionsInterface<2>, SingleBlockRegionInterface, IsolatedFromAboveInterface, NoTerminatorInterface, NOpdsInterface<0>, NResultsInterface<0>])]
pub struct PlanOp;
#[op_interface_impl]
impl RegionKindInterface for PlanOp {
    fn get_region_kind(&self, _idx: usize) -> RegionKind {
        RegionKind::Graph
    }
}
#[pliron_op(name = "exec.chain", format, verifier = "succ", interfaces = [OneRegionInterface, SingleBlockRegionInterface, NoTerminatorInterface, NOpdsInterface<0>, NResultsInterface<0>])]
pub struct ChainOp;
#[op_interface_impl]
impl RegionKindInterface for ChainOp {
    fn get_region_kind(&self, _idx: usize) -> RegionKind {
        RegionKind::Graph
    }
}
#[pliron_op(name = "exec.fused", format, verifier = "succ", interfaces = [OneRegionInterface, SingleBlockRegionInterface, NoTerminatorInterface, NOpdsInterface<0>, NResultsInterface<0>])]
pub struct FusedOp;
#[op_interface_impl]
impl RegionKindInterface for FusedOp {
    fn get_region_kind(&self, _idx: usize) -> RegionKind {
        RegionKind::Graph
    }
}
macro_rules! recipe {
    ($name:ident, $id:literal) => {
        #[pliron_op(name = $id, format, verifier = "succ", interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>])]
        pub struct $name;
    };
}
recipe!(CallOp, "exec.call");
recipe!(SelectRecipeOp, "exec.select");
recipe!(HandoffOp, "exec.handoff");
recipe!(FeedbackOp, "exec.feedback");
