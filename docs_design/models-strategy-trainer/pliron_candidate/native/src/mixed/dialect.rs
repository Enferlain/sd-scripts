//! Mixed lifetime/control operations reuse A's typed model/local vocabulary.
#[path = "../correlation/dialect.rs"]
pub mod shared;

use pliron::{
    builtin::op_interfaces::{
        IsTerminatorInterface, IsolatedFromAboveInterface, NRegionsInterface,
        NoTerminatorInterface, OneRegionInterface, RegionKind, RegionKindInterface,
        SingleBlockRegionInterface,
    },
    derive::{op_interface_impl, pliron_op, pliron_type},
};
use shared::{CorrelationContract, Shape};

macro_rules! ty {
    ($name:ident,$id:literal) => {
        #[pliron_type(name=$id,format,generate_get=true,verifier="succ")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name;
    };
}
ty!(CursorType, "mixed.cursor");
ty!(PolicyType, "mixed.policy");
ty!(FeedbackType, "mixed.feedback");
ty!(AcknowledgementType, "mixed.ack");
ty!(StopType, "mixed.stop");

macro_rules! leaf {
    ($name:ident,$id:literal) => {
        #[pliron_op(name=$id,format,verifier="succ",interfaces=[NRegionsInterface<0>])]
        pub struct $name;
    };
}
#[pliron_op(name="mixed.session",format,verifier="succ",interfaces=[OneRegionInterface,
    SingleBlockRegionInterface,IsolatedFromAboveInterface,NoTerminatorInterface])]
pub struct SessionOp;
#[pliron_op(name="mixed.producer",format,verifier="succ",interfaces=[OneRegionInterface,
    SingleBlockRegionInterface,IsolatedFromAboveInterface,NoTerminatorInterface])]
pub struct ProducerOp;
#[pliron_op(name="mixed.repeat",format,verifier="succ",interfaces=[OneRegionInterface,
    SingleBlockRegionInterface,IsolatedFromAboveInterface])]
pub struct RepeatOp;
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
region!(ProducerOp, RegionKind::Graph);
region!(RepeatOp, RegionKind::SSACFG);
leaf!(CursorOp, "mixed.cursor");
leaf!(PolicyOp, "mixed.policy");
leaf!(SelectOp, "mixed.select");
leaf!(FilterOp, "mixed.filter");
leaf!(UpdateOp, "mixed.update");
leaf!(PublishOp, "mixed.publish");
leaf!(LearnedOp, "mixed.learned");
leaf!(CheckpointOp, "mixed.checkpoint");
leaf!(FinishOp, "mixed.finish");
#[pliron_op(name="mixed.window_end",format,verifier="succ",interfaces=[NRegionsInterface<0>,IsTerminatorInterface])]
pub struct WindowEndOp;

macro_rules! shape {
    ($name:ident,$kind:literal,$attrs:expr,$inputs:expr,$outputs:expr,$args:expr) => {
        #[op_interface_impl]
        impl CorrelationContract for $name {
            fn shape(&self) -> Shape {
                Shape {
                    kind: $kind,
                    attributes: $attrs,
                    inputs: $inputs,
                    outputs: $outputs,
                    arguments: $args,
                    variadic: false,
                }
            }
        }
    };
}
shape!(
    SessionOp,
    "mixed_session",
    &["source_id", "version", "horizon", "stop_rule"],
    &[],
    &[],
    Some(&[])
);
shape!(
    ProducerOp,
    "producer",
    &[
        "source_id",
        "activity",
        "capacity",
        "publication",
        "settlement"
    ],
    &["ac.model", "mixed.feedback", "mixed.stop"],
    &["ac.correlated", "ac.done"],
    Some(&["ac.model", "mixed.feedback", "mixed.stop"])
);
shape!(
    CursorOp,
    "cursor",
    &["source_id", "owner", "start"],
    &[],
    &["mixed.cursor"],
    None
);
shape!(
    PolicyOp,
    "policy_state",
    &["source_id", "owner", "bias", "revision"],
    &[],
    &["mixed.policy"],
    None
);
shape!(
    SelectOp,
    "select",
    &["source_id", "application"],
    &["mixed.cursor", "mixed.policy", "mixed.ack", "mixed.stop"],
    &["ac.stream"],
    None
);
shape!(
    FilterOp,
    "dynamic_filter",
    &["source_id", "implementation"],
    &["ac.stream", "mixed.policy"],
    &["ac.stream"],
    None
);
shape!(
    UpdateOp,
    "policy_update",
    &["source_id", "implementation", "protocol"],
    &["mixed.feedback", "mixed.policy"],
    &["mixed.ack"],
    None
);
shape!(
    PublishOp,
    "publish",
    &["source_id"],
    &["ac.correlated"],
    &[],
    None
);
shape!(
    RepeatOp,
    "repeat",
    &["source_id", "activity", "count", "accumulation"],
    &["ac.correlated", "ac.model", "ac.unit"],
    &["mixed.feedback", "ac.done"],
    Some(&["ac.correlated", "ac.model", "ac.unit"])
);
shape!(
    LearnedOp,
    "learned",
    &["source_id", "observation"],
    &["ac.done", "ac.correlated", "ac.model"],
    &["mixed.feedback"],
    None
);
shape!(
    WindowEndOp,
    "mixed_window_end",
    &["source_id"],
    &["mixed.feedback"],
    &[],
    None
);
shape!(
    CheckpointOp,
    "checkpoint",
    &["source_id", "after_windows", "protocol"],
    &["ac.done"],
    &["mixed.stop", "ac.done"],
    None
);
shape!(
    FinishOp,
    "mixed_finish",
    &["source_id"],
    &["ac.done", "ac.done"],
    &[],
    None
);
