//! Small compositional vocabulary. Source identities are data, never dispatch.
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
ty!(ModelType, "lc.model");
ty!(StreamType, "lc.stream");
ty!(DoneType, "lc.done");
ty!(EventsType, "lc.events");
ty!(RequestType, "lc.request");
ty!(SampleType, "lc.sample");
ty!(GradientType, "lc.gradient");
ty!(CompletionType, "lc.completion");
ty!(MeasurementType, "lc.measurement");

pub struct Contract {
    pub kind: &'static str,
    pub attributes: &'static [&'static str],
    pub inputs: &'static [&'static str],
    pub tail: Option<&'static str>,
    pub max_tail: usize,
    pub outputs: &'static [&'static str],
    pub region: Option<RegionKind>,
    pub arguments: bool,
}
#[op_interface]
pub trait CompositionContract {
    fn contract(&self) -> Contract;
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
macro_rules! describe {
    ($name:ident, $kind:literal, $attrs:expr, $inputs:expr, $tail:expr, $max:expr, $outputs:expr, $region:expr, $args:expr) => {
        #[op_interface_impl]
        impl CompositionContract for $name {
            fn contract(&self) -> Contract {
                Contract {
                    kind: $kind,
                    attributes: $attrs,
                    inputs: $inputs,
                    tail: $tail,
                    max_tail: $max,
                    outputs: $outputs,
                    region: $region,
                    arguments: $args,
                }
            }
        }
    };
}
macro_rules! leaf {
    ($name:ident, $id:literal) => {
        #[pliron_op(name = $id, format, verifier = "succ", interfaces = [NRegionsInterface<0>])]
        pub struct $name;
    };
}
#[pliron_op(name = "lc.run", format, verifier = "succ", interfaces = [OneRegionInterface,
    SingleBlockRegionInterface, IsolatedFromAboveInterface, NoTerminatorInterface])]
pub struct RunOp;
#[pliron_op(name = "lc.group", format, verifier = "succ", interfaces = [OneRegionInterface,
    SingleBlockRegionInterface, NoTerminatorInterface])]
pub struct GroupOp;
#[pliron_op(name = "lc.repeat", format, verifier = "succ", interfaces = [OneRegionInterface, SingleBlockRegionInterface])]
pub struct RepeatOp;
#[pliron_op(name = "lc.evaluator", format, verifier = "succ", interfaces = [OneRegionInterface, SingleBlockRegionInterface])]
pub struct EvaluatorOp;
macro_rules! region {
    ($name:ident, $kind:expr) => {
        #[op_interface_impl]
        impl RegionKindInterface for $name {
            fn get_region_kind(&self, _idx: usize) -> RegionKind {
                $kind
            }
        }
    };
}
region!(RunOp, RegionKind::Graph);
region!(GroupOp, RegionKind::Graph);
region!(RepeatOp, RegionKind::SSACFG);
region!(EvaluatorOp, RegionKind::SSACFG);
leaf!(ModelOp, "lc.model");
leaf!(ProducerOp, "lc.producer");
leaf!(RequestOp, "lc.request");
leaf!(TakeOp, "lc.take");
leaf!(ContributeOp, "lc.contribute");
leaf!(AdvanceOp, "lc.advance");
leaf!(ReadOp, "lc.read");
leaf!(SettleOp, "lc.settle");
leaf!(FinishOp, "lc.finish");
#[pliron_op(name = "lc.window_end", format, verifier = "succ", interfaces = [NRegionsInterface<0>, IsTerminatorInterface])]
pub struct WindowEndOp;
#[pliron_op(name = "lc.evaluation_end", format, verifier = "succ", interfaces = [NRegionsInterface<0>, IsTerminatorInterface])]
pub struct EvaluationEndOp;

describe!(
    RunOp,
    "run",
    &["source_id", "version", "failure"],
    &[],
    None,
    0,
    &[],
    Some(RegionKind::Graph),
    false
);
describe!(
    ModelOp,
    "model",
    &["source_id", "implementation", "initial", "treatment"],
    &[],
    None,
    0,
    &["lc.model"],
    None,
    false
);
describe!(
    ProducerOp,
    "producer",
    &[
        "source_id",
        "implementation",
        "horizon",
        "ready",
        "workers",
        "remaining",
        "failure"
    ],
    &["lc.model", "lc.done"],
    None,
    0,
    &["lc.stream", "lc.done"],
    None,
    false
);
describe!(
    GroupOp,
    "group",
    &["source_id", "completion", "pending", "failure"],
    &[],
    None,
    0,
    &["lc.done"],
    Some(RegionKind::Graph),
    false
);
describe!(
    RepeatOp,
    "repeat",
    &["source_id", "max_windows", "admission", "stop"],
    &["lc.stream", "lc.model"],
    None,
    0,
    &["lc.events", "lc.done"],
    Some(RegionKind::SSACFG),
    true
);
describe!(
    RequestOp,
    "request",
    &[
        "source_id",
        "after_window",
        "multiplicity",
        "pending",
        "max_pending",
        "cancellation"
    ],
    &["lc.events"],
    None,
    0,
    &["lc.request"],
    None,
    false
);
describe!(
    EvaluatorOp,
    "evaluator",
    &["source_id", "completion"],
    &["lc.request", "lc.stream", "lc.model"],
    None,
    0,
    &["lc.done"],
    Some(RegionKind::SSACFG),
    true
);
describe!(
    TakeOp,
    "take",
    &["source_id", "selection", "provenance"],
    &["lc.stream"],
    None,
    0,
    &["lc.sample"],
    None,
    false
);
describe!(
    ContributeOp,
    "contribute",
    &["source_id", "implementation", "normalization"],
    &["lc.sample", "lc.model"],
    Some("lc.gradient"),
    1,
    &["lc.gradient"],
    None,
    false
);
describe!(
    AdvanceOp,
    "advance",
    &["source_id", "parameter", "learning_rate", "authority"],
    &["lc.gradient", "lc.model"],
    None,
    0,
    &["lc.completion"],
    None,
    false
);
describe!(
    ReadOp,
    "read",
    &["source_id", "implementation", "state", "mode", "completion"],
    &["lc.sample", "lc.model"],
    None,
    0,
    &["lc.measurement"],
    None,
    false
);
describe!(
    SettleOp,
    "settle",
    &["source_id"],
    &[],
    Some("lc.done"),
    usize::MAX,
    &[],
    None,
    false
);
describe!(
    FinishOp,
    "finish",
    &["source_id"],
    &[],
    Some("lc.done"),
    usize::MAX,
    &[],
    None,
    false
);
describe!(
    WindowEndOp,
    "window_end",
    &["source_id"],
    &["lc.completion"],
    None,
    0,
    &[],
    None,
    false
);
describe!(
    EvaluationEndOp,
    "evaluation_end",
    &["source_id"],
    &["lc.measurement"],
    None,
    0,
    &[],
    None,
    false
);

#[pliron_op(name = "lc_exec.plan", format, verifier = "succ", interfaces = [NRegionsInterface<2>,
    SingleBlockRegionInterface, IsolatedFromAboveInterface, NoTerminatorInterface, NOpdsInterface<0>, NResultsInterface<0>])]
pub struct PlanOp;
region!(PlanOp, RegionKind::Graph);
macro_rules! recipe {
    ($name:ident, $id:literal) => {
        #[pliron_op(name = $id, format, verifier = "succ", interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>])]
        pub struct $name;
    };
}
recipe!(ScopeRecipe, "lc_exec.scope");
recipe!(CallRecipe, "lc_exec.call");
recipe!(AccessRecipe, "lc_exec.access");
recipe!(RequestRecipe, "lc_exec.request");
recipe!(LinkRecipe, "lc_exec.link");
recipe!(PortRecipe, "lc_exec.port");
