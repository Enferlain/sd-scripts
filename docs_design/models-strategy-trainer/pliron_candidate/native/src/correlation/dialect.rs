//! A's typed streams, correlated membership, retained gradients and requests.
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
// Reuse the existing native scalar/local vocabulary, not Python IR machinery.
use rmst_pliron_b::dialect::{LocalOp, MapOp, YieldOp};

macro_rules! ty {
    ($name:ident, $id:literal) => {
        #[pliron_type(name = $id, format, generate_get = true, verifier = "succ")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name;
    };
}
ty!(ModelType, "ac.model");
ty!(UnitType, "ac.unit");
ty!(StreamType, "ac.stream");
ty!(CorrelatedType, "ac.correlated");
ty!(GradientType, "ac.gradient");
ty!(EventsType, "ac.events");
ty!(DoneType, "ac.done");

pub struct Shape {
    pub kind: &'static str,
    pub attributes: &'static [&'static str],
    pub inputs: &'static [&'static str],
    pub outputs: &'static [&'static str],
    pub arguments: Option<&'static [&'static str]>,
    pub variadic: bool,
}
#[op_interface]
pub trait CorrelationContract {
    fn shape(&self) -> Shape;
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
macro_rules! leaf {
    ($name:ident,$id:literal) => {
        #[pliron_op(name=$id,format,verifier="succ",interfaces=[NRegionsInterface<0>])]
        pub struct $name;
    };
}
#[pliron_op(name="ac.session",format,verifier="succ",interfaces=[OneRegionInterface,
    SingleBlockRegionInterface,IsolatedFromAboveInterface,NoTerminatorInterface])]
pub struct SessionOp;
#[pliron_op(name="ac.compose",format,verifier="succ",interfaces=[OneRegionInterface,
    SingleBlockRegionInterface,IsolatedFromAboveInterface])]
pub struct ComposeOp;
#[pliron_op(name="ac.window",format,verifier="succ",interfaces=[OneRegionInterface,
    SingleBlockRegionInterface,IsolatedFromAboveInterface])]
pub struct WindowOp;
macro_rules! region {
    ($name:ident,$kind:expr) => {
        #[op_interface_impl]
        impl RegionKindInterface for $name {
            fn get_region_kind(&self, _idx: usize) -> RegionKind {
                $kind
            }
        }
    };
}
region!(SessionOp, RegionKind::Graph);
region!(ComposeOp, RegionKind::SSACFG);
region!(WindowOp, RegionKind::SSACFG);
leaf!(ModelOp, "ac.model");
leaf!(UnitOp, "ac.unit");
leaf!(SourceOp, "ac.source");
leaf!(FilterOp, "ac.filter");
leaf!(FanOutOp, "ac.fanout");
leaf!(EncodeOp, "ac.encode");
leaf!(CorrelateOp, "ac.correlate");
leaf!(ContributeOp, "ac.contribute");
leaf!(AdvanceOp, "ac.advance");
leaf!(EvaluateOp, "ac.evaluate");
leaf!(FinishOp, "ac.finish");
#[pliron_op(name="ac.window_end",format,verifier="succ",interfaces=[NRegionsInterface<0>,IsTerminatorInterface])]
pub struct WindowEndOp;

macro_rules! shape {
    ($name:ident,$kind:literal,$attrs:expr,$inputs:expr,$outputs:expr,$args:expr,$variadic:expr) => {
        #[op_interface_impl]
        impl CorrelationContract for $name {
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
    &["source_id", "version", "horizon", "stop_rule"],
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
        "implementation",
        "initial",
        "treatment"
    ],
    &[],
    &["ac.model"],
    None,
    false
);
shape!(
    UnitOp,
    "unit",
    &["source_id", "unit", "parameter"],
    &["ac.model"],
    &["ac.unit"],
    None,
    false
);
shape!(
    SourceOp,
    "source",
    &["source_id", "keys"],
    &[],
    &["ac.stream"],
    None,
    false
);
shape!(
    FilterOp,
    "filter",
    &["source_id", "excluded"],
    &["ac.stream"],
    &["ac.stream"],
    None,
    false
);
shape!(
    FanOutOp,
    "fanout",
    &["source_id"],
    &["ac.stream"],
    &["ac.stream", "ac.stream"],
    None,
    false
);
shape!(
    ComposeOp,
    "compose",
    &["source_id"],
    &["ac.stream"],
    &["ac.stream"],
    Some(&["wf.record"]),
    false
);
shape!(
    LocalOp,
    "local",
    &["source_id"],
    &["wf.record"],
    &["wf.record"],
    Some(&["wf.record"]),
    false
);
shape!(
    MapOp,
    "map",
    &["source_id", "implementation", "effect"],
    &["wf.record"],
    &["wf.record"],
    None,
    false
);
shape!(
    YieldOp,
    "yield",
    &["source_id"],
    &["wf.record"],
    &[],
    None,
    false
);
shape!(
    EncodeOp,
    "encode",
    &["source_id", "implementation", "protection"],
    &["ac.stream", "ac.model"],
    &["ac.stream"],
    None,
    false
);
shape!(
    CorrelateOp,
    "correlate",
    &["source_id", "membership", "provenance"],
    &["ac.stream", "ac.stream", "ac.model"],
    &["ac.correlated"],
    None,
    false
);
shape!(
    WindowOp,
    "window",
    &["source_id", "activity", "count", "accumulation"],
    &["ac.correlated", "ac.model", "ac.unit"],
    &["ac.events", "ac.done"],
    Some(&["ac.correlated", "ac.model", "ac.unit"]),
    false
);
shape!(
    ContributeOp,
    "contribute",
    &["source_id", "implementation", "normalization"],
    &["ac.correlated", "ac.model", "ac.unit"],
    &["ac.gradient"],
    None,
    false
);
shape!(
    AdvanceOp,
    "advance",
    &["source_id", "learning_rate", "authority"],
    &["ac.gradient", "ac.unit"],
    &["ac.done"],
    None,
    false
);
shape!(
    WindowEndOp,
    "window_end",
    &["source_id"],
    &["ac.done"],
    &[],
    None,
    false
);
shape!(
    EvaluateOp,
    "evaluate",
    &[
        "source_id",
        "owner",
        "implementation",
        "after_window",
        "consistency",
        "protection"
    ],
    &["ac.events", "ac.model"],
    &["ac.done"],
    None,
    false
);
shape!(
    FinishOp,
    "finish",
    &["source_id"],
    &["ac.done"],
    &[],
    None,
    true
);

#[pliron_op(name="ac_exec.plan",format,verifier="succ",interfaces=[NRegionsInterface<2>,SingleBlockRegionInterface,
    IsolatedFromAboveInterface,NoTerminatorInterface,NOpdsInterface<0>,NResultsInterface<0>])]
pub struct PlanOp;
#[pliron_op(name="ac_exec.local",format,verifier="succ",interfaces=[OneRegionInterface,SingleBlockRegionInterface,
    IsolatedFromAboveInterface,NoTerminatorInterface,NOpdsInterface<0>,NResultsInterface<0>])]
pub struct LocalPlanOp;
#[pliron_op(name="ac_exec.fused",format,verifier="succ",interfaces=[OneRegionInterface,SingleBlockRegionInterface,
    IsolatedFromAboveInterface,NoTerminatorInterface,NOpdsInterface<0>,NResultsInterface<0>])]
pub struct FusedOp;
region!(PlanOp, RegionKind::Graph);
region!(LocalPlanOp, RegionKind::Graph);
region!(FusedOp, RegionKind::Graph);
#[pliron_op(name="ac_exec.declaration",format,verifier="succ",interfaces=[NRegionsInterface<0>,NOpdsInterface<0>,NResultsInterface<0>])]
pub struct DeclarationOp;
#[pliron_op(name="ac_exec.call",format,verifier="succ",interfaces=[NRegionsInterface<0>,NOpdsInterface<0>,NResultsInterface<0>])]
pub struct CallOp;
