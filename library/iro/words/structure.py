"""Structure words: strategy, sequence, parallel, repeat, choice and derived (D16)."""

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr, UnitAttr
from xdsl.ir import Operation, Region, SSAValue
from xdsl.irdl import irdl_op_definition, opt_prop_def, prop_def, region_def, traits_def, var_region_def
from xdsl.traits import IsolatedFromAbove, NoTerminator
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import DerivedWord, LifecycleProtocol, OrderedBody, RegionSemantics, path
from library.iro.words.base import Activity, BodyShape, DeclaredFields, Named, Word


def _activity(protocol: LifecycleProtocol, *extra):
    return traits_def(RegionSemantics(protocol), BodyShape(), IsolatedFromAbove(), DeclaredFields(), *extra)


@irdl_op_definition
class StrategyOp(Word):
    """The accepted strategy: the root operation, which one run executes.

    Its ``uses`` header pins each implementation's version, once for the
    whole strategy (D16). A ``normalized`` strategy has its derived words
    expanded and its releases placed; an accepted one has neither.
    """

    name = "iro.strategy"

    sym_name = prop_def(StringAttr)
    normalized = opt_prop_def(UnitAttr)
    uses = opt_prop_def(ArrayAttr[SymbolRefAttr])
    use_versions = opt_prop_def(ArrayAttr[StringAttr])
    body = region_def("single_block")

    traits = traits_def(OrderedBody(), IsolatedFromAbove(), NoTerminator(), DeclaredFields())

    def verify_(self) -> None:
        names = [ref.root_reference.data for ref in self.uses or ()]
        versions = list(self.use_versions or ())
        if len(names) != len(versions):
            raise VerifyException(f"@{self.sym_name.data}: each implementation in uses needs one version")
        if len(set(names)) != len(names):
            raise VerifyException(f"@{self.sym_name.data}: an implementation is pinned twice in uses")


@irdl_op_definition
class SequenceOp(Activity):
    """A named activity that runs its steps in order, once."""

    name = "iro.sequence"

    body = region_def("single_block")
    traits = _activity(LifecycleProtocol.SEQUENCE, Named())


@irdl_op_definition
class ParallelOp(Activity):
    """Runs its branches concurrently until all settle; each statement is one branch."""

    name = "iro.parallel"

    branches = var_region_def("single_block")
    traits = _activity(LifecycleProtocol.PARALLEL, Named())


@irdl_op_definition
class RepeatOp(Activity):
    """Repeats its body a number of times or until a policy ends it.

    Its last ``carry_count`` operands are carried values; its results are
    their final values.
    """

    name = "iro.repeat"

    limit = opt_prop_def(IntegerAttr)
    policy = opt_prop_def(SymbolRefAttr)
    carry_count = prop_def(IntegerAttr)
    body = region_def("single_block")
    traits = _activity(LifecycleProtocol.REPEAT, Named())

    def verify_(self) -> None:
        if (self.limit is None) == (self.policy is None):
            raise VerifyException(f"{path(self)}: a repeat runs a number of times or until a policy, exactly one")
        if self.limit is not None and self.limit.value.data < 0:
            raise VerifyException(f"{path(self)}: a repeat cannot run a negative number of times")


@irdl_op_definition
class ChoiceOp(Activity):
    """Runs exactly one of two alternatives, chosen by a bool or by whether an optional holds a value."""

    name = "iro.choice"

    alternatives = var_region_def("single_block")
    traits = _activity(LifecycleProtocol.CHOICE)


def _as_sequence(op: Operation) -> tuple[list[Operation], list[SSAValue]]:
    """A derived body becomes the named sequence it stands for, with the same captures and results."""
    body = Region()
    op.regions[0].move_blocks(body)
    properties = {key: value for key, value in op.properties.items() if key != "ancestry"}
    sequence = SequenceOp.create(
        operands=list(op.operands), result_types=[r.type for r in op.results], properties=properties, regions=[body]
    )
    return [sequence], list(sequence.results)


@irdl_op_definition
class DerivedOp(Activity):
    """A derived word: one defined by expanding into other words during normalization."""

    name = "iro.derived"

    body = region_def("single_block")
    traits = _activity(LifecycleProtocol.SEQUENCE, Named(), DerivedWord(_as_sequence))


OPERATIONS = (StrategyOp, SequenceOp, ParallelOp, RepeatOp, ChoiceOp, DerivedOp)

__all__ = ["OPERATIONS", "ChoiceOp", "DerivedOp", "ParallelOp", "RepeatOp", "SequenceOp", "StrategyOp"]
