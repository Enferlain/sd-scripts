"""Actual IR, capture boundaries, source/symbol separation and reconstruction."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Dialect, Operation, Region
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.traits import IsolatedFromAbove, OpTrait, SymbolOpInterface
from xdsl.utils.exceptions import ParseError, VerifyException

from xdsl.irdl import (
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
    var_operand_def,
)

from .contracts import (
    AccessMode,
    ExpandSequence,
    FlowSemantics,
    Freshness,
    origin,
    Protocol,
    ReferenceSemantics,
    RegionSemantics,
    ResourceSemantics,
)
from .dialects import (
    AccessOp,
    CallOp,
    CaptureShape,
    ConstantOp,
    dump,
    integer,
    ParallelOp,
    parse,
    PayloadType,
    ProgramOp,
    RecipeOp,
    RepeatOp,
    RequestOp,
    RequestsOp,
    ResourceOp,
    SequenceOp,
    SourcedOp,
    SourceShape,
    StreamOp,
    StructuredOp,
)
from .fixtures import data_only, lifetime, props
from .normalize import inspect_structure, normalize, source_index
from . import normalize as normalization
from . import fixtures as f


def find[T: Operation](module: Operation, kind: type[T], name: str | None = None) -> T:
    return next(op for op in module.walk() if isinstance(op, kind) and (name is None or origin(op) == name))


def trait[T: OpTrait](op: Operation, kind: type[T]) -> T:
    result = op.get_trait(kind)
    assert result is not None
    return result


@pytest.mark.parametrize("build", [lifetime, data_only, lambda: lifetime(extra_scope=True)])
def test_real_ir_roundtrip_and_normalization_are_detached(build: Callable[[], ModuleOp]) -> None:
    author = build()
    accepted = dump(author)
    restored = parse(accepted)
    assert author.is_structurally_equivalent(restored)
    normalized = normalize(restored)
    cold = normalize(parse(normalized.text))
    assert normalized.module.is_structurally_equivalent(cold.module)
    assert normalized.structure.origins == inspect_structure(author).origins
    assert normalized.structure.links == inspect_structure(author).links
    find(author, CallOp).properties["binding"] = SymbolRefAttr("author_changed")
    assert "author_changed" not in normalized.text


def test_normalization_replaces_a_value_producing_recipe_and_remaps_uses() -> None:
    author = data_only()
    recipe = find(author, RecipeOp)
    before = tuple(recipe.results[0].uses)
    assert before
    normalized = normalize(author)
    expanded = source_index(normalized.module)[origin(recipe)]
    assert expanded.name == "lang.sequence"
    assert expanded.results[0].uses
    assert all(use.operation.parent_op() is not None for use in expanded.results[0].uses)
    assert find(author, RecipeOp) is recipe
    assert normalized.module.is_structurally_equivalent(normalize(normalized.module).module)


@irdl_op_definition
class IndependentRecipe(StructuredOp):
    name = "example.preprocess"
    traits = traits_def(
        SourceShape(), CaptureShape(), RegionSemantics(Protocol.SEQUENCE), ExpandSequence(), SymbolOpInterface(), IsolatedFromAbove()
    )


def ordered_values_program(shape: str, *, normalized: bool = False, invalid: bool = True) -> ModuleOp:
    late = f.node(ConstantOp, "later.definition", outputs=[f.SCALAR], literal=StringAttr("3"))
    if shape == "direct":
        use = f.node(
            CallOp, "earlier.use", inputs=late.results, outputs=[f.SCALAR], binding=SymbolRefAttr("echo"), binding_version=StringAttr("1")
        )
        operations = [late, use]
    else:
        recipe = shape in ("recipe_capture", "recipe_body", "independent_recipe")
        capture = shape != "recipe_body"
        inner = Block(arg_types=[f.SCALAR] if capture else [])
        if not capture:
            inner.add_op(late)
        use = f.call(inner, "earlier.use", "echo", inner.args if capture else late.results, [f.SCALAR])
        f.end(inner, "capture.body", *use.results)
        kind = IndependentRecipe if shape == "independent_recipe" else RecipeOp if recipe else SequenceOp
        scope = f.structured(kind, "capture.scope", late.results if capture else [], [inner], outputs=[f.SCALAR])
        operations = [late, scope] if capture else [scope]
    root = ProgramOp.create(properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block(operations))])
    module = ModuleOp([root])
    if normalized:
        module = normalize(module).module
    if invalid:
        index = source_index(module)
        late = index["later.definition"]
        anchor = index["earlier.use" if shape in ("direct", "recipe_body") else "capture.scope"]
        block = anchor.parent_block()
        assert block is not None
        block.detach_op(late)
        block.insert_op_after(late, anchor)
    return module


@pytest.mark.parametrize("shape", ["direct", "nested_capture", "recipe_capture", "recipe_body", "independent_recipe"])
@pytest.mark.parametrize("normalized", [False, True])
def test_ordered_definition_availability_rejects_forward_uses_and_captures(shape, normalized):
    module = ordered_values_program(shape, normalized=normalized)
    with pytest.raises(VerifyException) as failed:
        normalize(module)
    diagnostic = str(failed.value)
    assert "later.definition" in diagnostic
    assert ("earlier.use" if shape in ("direct", "recipe_body") else "capture.scope") in diagnostic
    assert "operand 0" in diagnostic and "unavailable in ordered body" in diagnostic


def test_an_ordered_operation_cannot_consume_its_own_result():
    module = ordered_values_program("direct", invalid=False)
    call = source_index(module)["earlier.use"]
    call.operands = call.results
    with pytest.raises(VerifyException, match=r"earlier.use operand 0.*earlier.use result 0.*unavailable"):
        normalize(module)


def test_parallel_role_cannot_borrow_a_sibling_entry_argument_as_a_local_value():
    value = f.node(ConstantOp, "input", outputs=[f.SCALAR], literal=StringAttr("3"))
    first, second = f.body(value.results), f.body(value.results)
    f.end(first, "first")
    f.call(second, "cross_role.use", "echo", first.args, [f.SCALAR])
    f.end(second, "second")
    branches = f.structured(ParallelOp, "branches", value.results, [first, second])
    root = ProgramOp.create(properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([value, branches]))])
    with pytest.raises(VerifyException, match=r"cross_role.use operand 0.*branches region 0 argument 0.*unavailable"):
        normalize(ModuleOp([root]))


def test_a_separate_registered_operation_uses_existing_contract_and_expansion() -> None:
    author = data_only()
    recipe = find(author, RecipeOp)
    region = Region()
    recipe.body.clone_into(region)
    replacement = IndependentRecipe.create(
        operands=recipe.operands, result_types=[x.type for x in recipe.results], properties=dict(recipe.properties), regions=[region]
    )
    PatternRewriter(recipe).replace(recipe, replacement)
    extension = Dialect("example", [IndependentRecipe], [])
    restored = parse(dump(author), extensions=(extension,))
    normalized = normalize(restored)
    assert all(op.name != "example.preprocess" for op in normalized.module.walk())
    assert source_index(normalized.module)[origin(recipe)].name == "lang.sequence"


@pytest.mark.parametrize("symbol", ["training", "evaluation", "producer", "model"])
def test_source_origins_do_not_resolve_semantic_references(symbol: str) -> None:
    author = lifetime()
    operation = source_index(author)[symbol]
    operation.properties["origin"] = StringAttr("diagnostic." + symbol)
    normalized = normalize(author)
    assert "diagnostic." + symbol in normalized.structure.origins
    assert any(link.target == symbol for link in normalized.structure.links) or symbol == "model"


def test_contract_queries_retain_actual_types_bounds_and_changing_policy() -> None:
    author = lifetime(windows=7)
    loop = find(author, RepeatOp, "training")
    contract = trait(loop, RegionSemantics).describe(loop)
    assert contract.protocol == Protocol.REPEAT and contract.limit == 7
    assert contract.policy == "bounded_count"
    assert contract.roles[0].entries[-1] == PayloadType(StringAttr("index"))
    assert contract.roles[0].captures[0].type == loop.operands[0].type
    handler = find(author, RequestsOp)
    requests = trait(handler, RegionSemantics).describe(handler)
    assert requests.request_sources == ("training",)
    assert requests.max_active == requests.max_outstanding == 1
    stream = find(author, StreamOp)
    flow = trait(stream, FlowSemantics).describe(stream)
    assert flow.payload == stream.value.type.payload
    assert flow.versions.owners == ("encoder",)
    assert flow.versions.boundary == "item_admission"


def test_access_realization_and_freshness_are_separate_typed_contract_fields() -> None:
    author = lifetime()
    op = find(author, AccessOp, "measure.access")
    contract = trait(op, ResourceSemantics).describe(op)
    assert contract.mode == AccessMode.PROTECTED_CURRENT
    assert contract.freshness == Freshness.CURRENT_AT_START
    assert contract.subject is op.operands[0]
    snapshot = trait(find(author, AccessOp, "encode.access"), ResourceSemantics).describe(find(author, AccessOp, "encode.access"))
    assert snapshot.mode == AccessMode.SNAPSHOT and snapshot.freshness == Freshness.PUBLISHED
    assert contract.extent == "body_and_admitted_uses_settled"


@pytest.mark.parametrize(
    "case",
    [
        "capture_type",
        "hidden_capture",
        "request_target",
        "source_reference",
        "duplicate_origin",
        "duplicate_symbol",
        "version",
        "unknown_fields",
        "ancestry",
    ],
)
def test_local_structure_corruptions_are_rejected(case: str) -> None:
    author = lifetime()
    if case == "capture_type":
        op = find(author, AccessOp, "gradient.access")
        op.operands = [op.operands[0], op.operands[2], op.operands[1]]
    elif case == "hidden_capture":
        call = find(author, CallOp, "gradient")
        call.operands = [call.operands[0], find(author, ResourceOp, "model").value, call.operands[2]]
    elif case == "request_target":
        find(author, RequestOp).properties["handler"] = SymbolRefAttr("producer")
    elif case == "source_reference":
        find(author, RequestsOp).properties["sources"] = ArrayAttr([SymbolRefAttr("missing")])
    elif case == "duplicate_origin":
        find(author, AccessOp).properties["origin"] = StringAttr("producer")
    elif case == "duplicate_symbol":
        find(author, AccessOp).properties["sym_name"] = StringAttr("producer")
    elif case == "version":
        find(author, AccessOp).properties["version"] = integer(2)
    elif case == "unknown_fields":
        find(author, AccessOp).properties["hidden_scheduler"] = StringAttr("custom")
    else:
        author = normalize(author).module
        find(author, AccessOp).properties["ancestry"] = ArrayAttr([StringAttr("forged")])
    with pytest.raises(VerifyException):
        normalize(author)


@pytest.mark.parametrize("suffix", ["garbage", '"lang.program"() : () -> ()'])
def test_parser_requires_the_complete_document(suffix: str) -> None:
    with pytest.raises(ParseError):
        parse(dump(data_only()) + suffix)


def test_declaration_fields_survive_cold_roundtrip_without_becoming_verified_claims() -> None:
    author = lifetime()
    op = find(author, RepeatOp, "training")
    op.properties["quiescence"] = ArrayAttr([StringAttr("between_settled_iterations")])
    op.properties["rank_rule"] = StringAttr("decided_once_and_distributed")
    op.properties["reproducibility"] = StringAttr("may_depend_on_schedule")
    op.properties["assumptions"] = ArrayAttr([StringAttr("declared_policy_v1")])
    decoded = normalize(parse(normalize(author).text))
    restored = source_index(decoded.module)["training"]
    contract = trait(restored, RegionSemantics).describe(restored)
    assert contract.quiescence == ("between_settled_iterations",)
    assert contract.decision.rank == "decided_once_and_distributed"
    assert contract.decision.assumptions == ("declared_policy_v1",)


def test_request_sources_must_belong_to_the_declared_handler_source_set() -> None:
    author = lifetime()
    find(author, RequestsOp).properties["sources"] = ArrayAttr([SymbolRefAttr("gradient.access")])
    with pytest.raises(VerifyException, match="undeclared request source"):
        normalize(author)


def test_flow_and_resource_queries_retain_the_shared_decision_descriptor() -> None:
    author = lifetime()
    for op in [find(author, StreamOp), find(author, AccessOp, "measure.access")]:
        op.properties["reproducibility"] = StringAttr("may_depend_on_schedule")
        op.properties["rank_rule"] = StringAttr("decided_once_and_distributed")
        op.properties["assumptions"] = ArrayAttr([StringAttr("admission_policy_v1")])
    decoded = normalize(parse(normalize(author).text)).module
    flow = trait(find(decoded, StreamOp), FlowSemantics).describe(find(decoded, StreamOp))
    access = find(decoded, AccessOp, "measure.access")
    use = trait(access, ResourceSemantics).describe(access)
    assert flow.decision == use.decision == trait(access, RegionSemantics).describe(access).decision
    assert flow.decision.assumptions == ("admission_policy_v1",)


@pytest.mark.parametrize("case", ["region", "flow", "access", "binding", "literal", "wiring"])
def test_normalization_rejects_a_rewrite_that_changes_contracts_or_wiring(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    original = normalization.ExpandRecipe.match_and_rewrite
    changed = False
    model = case in ("access", "literal", "wiring")

    def corrupt(self: normalization.ExpandRecipe, op: Operation, rewriter: PatternRewriter) -> None:
        nonlocal changed
        root = op
        while (parent := root.parent_op()) is not None:
            root = parent
        matched = op.get_trait(ExpandSequence) is not None
        original(self, op, rewriter)
        if changed or not ((model and isinstance(op, ProgramOp)) or (not model and matched)):
            return
        changed = True
        if case == "region":
            find(root, RepeatOp).properties["limit"] = integer(5)
        elif case == "flow":
            find(root, StreamOp).properties["horizon"] = integer(4)
        elif case == "access":
            find(root, AccessOp, "encode.access").properties["freshness"] = StringAttr("current_at_start")
        elif case == "binding":
            find(root, CallOp).properties["binding_version"] = StringAttr("2")
        elif case == "literal":
            find(root, ConstantOp).properties["literal"] = StringAttr("9.0")
        else:
            parallel = find(root, ParallelOp, "activities")
            parallel.operands = [parallel.operands[0], parallel.operands[2], parallel.operands[1]]

    monkeypatch.setattr(normalization.ExpandRecipe, "match_and_rewrite", corrupt)
    with pytest.raises(VerifyException, match="normalization changed"):
        normalize(lifetime() if model else data_only())


@pytest.mark.parametrize("field", ["admission", "mode", "freshness"])
def test_invalid_contract_enums_have_typed_source_diagnostics(field: str) -> None:
    author = lifetime()
    op = find(author, StreamOp) if field == "admission" else find(author, AccessOp)
    op.properties[field] = StringAttr("unsupported_enum_value")
    with pytest.raises(VerifyException, match="unsupported"):
        normalize(author)


@irdl_op_definition
class MalformedReferenceOp(SourcedOp):
    name = "example.malformed_reference"
    inputs = var_operand_def()
    target = prop_def(SymbolRefAttr)
    source = opt_prop_def(SymbolRefAttr)
    traits = traits_def(SourceShape(), ReferenceSemantics((("target", "requests"),)))


@pytest.mark.parametrize("case", ["missing_field", "missing_payload", "missing_source"])
def test_incomplete_extension_reference_contract_has_a_typed_error(case: str) -> None:
    properties = {**props("extension"), "target": SymbolRefAttr("evaluation")}
    operands = []
    if case == "missing_payload":
        properties["source"] = SymbolRefAttr("training")
    elif case == "missing_source":
        operands = [find(lifetime(), ConstantOp).results[0]]
    op = MalformedReferenceOp.create(properties=properties, operands=operands)
    interface = ReferenceSemantics((("missing", "region"),)) if case == "missing_field" else trait(op, ReferenceSemantics)
    with pytest.raises(VerifyException):
        interface.describe(op)
