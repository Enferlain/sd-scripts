"""Typed lifetime operations and xDSL serialization; no execution services."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, DictionaryAttr, ModuleOp, StringAttr
from xdsl.ir import Dialect, Operation, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import IsolatedFromAbove, IsTerminator, NoTerminator, OpTrait
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class ModelType(ParametrizedAttribute, TypeAttribute):
    name = "life.model"


@irdl_attr_definition
class StreamType(ParametrizedAttribute, TypeAttribute):
    name = "life.stream"


@irdl_attr_definition
class DoneType(ParametrizedAttribute, TypeAttribute):
    name = "life.done"


@irdl_attr_definition
class EventsType(ParametrizedAttribute, TypeAttribute):
    name = "life.events"


@irdl_attr_definition
class RequestType(ParametrizedAttribute, TypeAttribute):
    name = "life.request"


@irdl_attr_definition
class SampleType(ParametrizedAttribute, TypeAttribute):
    name = "life.sample"


@irdl_attr_definition
class GradientType(ParametrizedAttribute, TypeAttribute):
    name = "life.gradient"


@irdl_attr_definition
class CompletionType(ParametrizedAttribute, TypeAttribute):
    name = "life.completion"


@irdl_attr_definition
class MeasurementType(ParametrizedAttribute, TypeAttribute):
    name = "life.measurement"


@dataclass(frozen=True)
class ActivityContract(OpTrait):
    """Domain attributes and region execution meaning, independent of IDs."""

    keys: tuple[str, ...]
    region: str = "none"

    def verify(self, op: Operation) -> None:
        if op.attributes or set(op.properties) != {"source_id", "policy"}:
            raise VerifyException(f"{op.name}: unsupported attributes/properties")
        source = op.properties["source_id"]
        policy = op.properties["policy"]
        if not isinstance(source, StringAttr) or not source.data:
            raise VerifyException(f"{op.name}: nonempty source identity required")
        if not isinstance(policy, DictionaryAttr) or set(policy.data) != set(self.keys):
            raise VerifyException(f"{source.data}: required policy keys are {self.keys}")
        if any(not isinstance(item, StringAttr) for item in policy.data.values()):
            raise VerifyException(f"{source.data}: string policy declarations required")


class ActivityOp(IRDLOperation):
    source_id = prop_def(StringAttr)
    policy = prop_def(DictionaryAttr)


@irdl_op_definition
class RunOp(ActivityOp):
    name = "life.run"
    body = region_def("single_block")
    traits = traits_def(ActivityContract(("version", "failure"), "graph"), IsolatedFromAbove())


@irdl_op_definition
class ModelOp(ActivityOp):
    name = "life.model"
    model = result_def(ModelType)
    traits = traits_def(ActivityContract(("implementation", "initial", "treatment")))


@irdl_op_definition
class ProducerOp(ActivityOp):
    name = "life.producer"
    model = operand_def(ModelType)
    stop = operand_def(DoneType)
    records = result_def(StreamType)
    done = result_def(DoneType)
    traits = traits_def(ActivityContract(("implementation", "horizon", "ready", "workers", "remaining", "failure")))


@irdl_op_definition
class GroupOp(ActivityOp):
    name = "life.group"
    done = result_def(DoneType)
    body = region_def("single_block")
    traits = traits_def(ActivityContract(("completion", "pending", "failure"), "graph"))


@irdl_op_definition
class RepeatOp(ActivityOp):
    name = "life.repeat"
    records = operand_def(StreamType)
    model = operand_def(ModelType)
    events = result_def(EventsType)
    done = result_def(DoneType)
    body = region_def("single_block")
    traits = traits_def(ActivityContract(("max_windows", "admission", "stop"), "ordered"), IsolatedFromAbove())


@irdl_op_definition
class RequestOp(ActivityOp):
    name = "life.request"
    events = operand_def(EventsType)
    due = result_def(RequestType)
    traits = traits_def(ActivityContract(("after_window", "multiplicity", "pending", "max_pending", "cancellation")))


@irdl_op_definition
class EvaluatorOp(ActivityOp):
    name = "life.evaluator"
    due = operand_def(RequestType)
    records = operand_def(StreamType)
    model = operand_def(ModelType)
    done = result_def(DoneType)
    body = region_def("single_block")
    traits = traits_def(ActivityContract(("completion",), "ordered"), IsolatedFromAbove())


@irdl_op_definition
class TakeOp(ActivityOp):
    name = "life.take"
    records = operand_def(StreamType)
    sample = result_def(SampleType)
    traits = traits_def(ActivityContract(("selection", "provenance")))


@irdl_op_definition
class ContributeOp(ActivityOp):
    name = "life.contribute"
    sample = operand_def(SampleType)
    model = operand_def(ModelType)
    previous = opt_operand_def(GradientType)
    gradient = result_def(GradientType)
    traits = traits_def(ActivityContract(("implementation", "normalization")))


@irdl_op_definition
class AdvanceOp(ActivityOp):
    name = "life.advance"
    gradient = operand_def(GradientType)
    model = operand_def(ModelType)
    completion = result_def(CompletionType)
    traits = traits_def(ActivityContract(("parameter", "learning_rate", "authority")))


@irdl_op_definition
class ReadOp(ActivityOp):
    name = "life.read"
    sample = operand_def(SampleType)
    model = operand_def(ModelType)
    measurement = result_def(MeasurementType)
    traits = traits_def(ActivityContract(("implementation", "state", "mode", "completion")))


@irdl_op_definition
class WindowEndOp(ActivityOp):
    name = "life.window_end"
    completion = operand_def(CompletionType)
    traits = traits_def(ActivityContract(()), IsTerminator())


@irdl_op_definition
class EvaluationEndOp(ActivityOp):
    name = "life.evaluation_end"
    measurement = operand_def(MeasurementType)
    traits = traits_def(ActivityContract(()), IsTerminator())


@irdl_op_definition
class SettleOp(ActivityOp):
    name = "life.settle"
    done = var_operand_def(DoneType)
    traits = traits_def(ActivityContract(()), IsTerminator())


@irdl_op_definition
class FinishOp(ActivityOp):
    name = "life.finish"
    done = var_operand_def(DoneType)
    traits = traits_def(ActivityContract(()), IsTerminator())


@irdl_op_definition
class PlanOp(IRDLOperation):
    name = "life_exec.plan"
    schema = prop_def(StringAttr)
    target = prop_def(StringAttr)
    manifest = prop_def(StringAttr)
    semantics = region_def("single_block")
    recipe = region_def("single_block")
    traits = traits_def(NoTerminator(), IsolatedFromAbove())


@irdl_op_definition
class RecipeOp(IRDLOperation):
    name = "life_exec.declaration"
    role = prop_def(StringAttr)
    values = prop_def(DictionaryAttr)


SEMANTIC = (
    RunOp,
    ModelOp,
    ProducerOp,
    GroupOp,
    RepeatOp,
    RequestOp,
    EvaluatorOp,
    TakeOp,
    ContributeOp,
    AdvanceOp,
    ReadOp,
    WindowEndOp,
    EvaluationEndOp,
    SettleOp,
    FinishOp,
)
Life = Dialect(
    "life",
    list(SEMANTIC),
    [ModelType, StreamType, DoneType, EventsType, RequestType, SampleType, GradientType, CompletionType, MeasurementType],
)
LifeExec = Dialect("life_exec", [PlanOp, RecipeOp], [])


def parse(text: str) -> ModuleOp:
    context = Context()
    for dialect in (Builtin, Life, LifeExec):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    return module


def printed(op: Operation) -> str:
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(op)
    return stream.getvalue()


def equivalent(left: Operation, right: Operation) -> bool:
    """Pre-map corresponding definitions so framework comparison handles cycles.

    xDSL's comparator discovers results in walk order; graph forward references
    need the positional result map first. Operand relationships are still checked
    by the framework, never predeclared equivalent by their source names.
    Its property, operand and block-argument comparisons are part of the cold
    verifier's pinned dependency contract; direct regression checks guard them.
    """
    a, b = tuple(left.walk()), tuple(right.walk())
    if len(a) != len(b) or any(x.name != y.name or len(x.results) != len(y.results) for x, y in zip(a, b, strict=True)):
        return False
    mapping = {}
    for x, y in zip(a, b, strict=True):
        for result, other in zip(x.results, y.results, strict=True):
            if result.type != other.type:
                return False
            mapping[result] = other
    return left.is_structurally_equivalent(right, mapping)
