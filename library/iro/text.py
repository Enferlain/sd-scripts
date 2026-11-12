"""The written form: printer, parser and cold loading (D16).

A saved strategy is parsed back on load, so the readable form is the saved
form. The printer writes one canonical layout, and printing what the parser
read gives the same text back. Comments are for people and are dropped.

Layout rules the printer follows, all from D16:

- Nesting reads as lifetime; relationships read as statements.
- Access modes appear on each use; bounds and admission on each stream.
- Defaults are omitted. Source locations print only on request.
- Activities list their captures. Local bodies (access, choice, derived) use
  enclosing values directly; parsing turns those uses into explicit captures.
- An entry list shows only what the protocol supplies, and only when used.
- A local body whose value is its last statement's results needs no yield.
- Literals print inline where they are used, with Python's type names.

The core words' forms live here. A domain vocabulary passes its words' forms
through the composition (``Definition.form``).
"""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, IntegerType, StringAttr, SymbolRefAttr, f64, i1, i64
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from xdsl.utils.exceptions import VerifyException

from library.iro.composition import Composition, core
from library.iro.diagnostics import Problems, Rejection, Stage
from library.iro.operation_contracts import (
    DEFAULT_SELECTION,
    AccessMode,
    Admission,
    ResourceUseAttr,
    VersionSelection,
    path,
    resource_uses,
    sym_name,
)
from library.iro.words import types as t
from library.iro.words.access import AccessOp, ReleaseOp
from library.iro.words.calls import CallOp, ConstantOp, YieldOp, literal_type, retained_positions
from library.iro.words.data import DEFAULT_COMPARISON, DEFAULT_METRIC, BorrowOp, DisposeOp, FilterOp, ForkOp, StreamOp, TakeOp
from library.iro.words.lifetimes import HandlerOp, ProducerOp, RequestOp, StopOp
from library.iro.words.model import ResourceOp, RelationshipOp
from library.iro.words.structure import ChoiceOp, DerivedOp, ParallelOp, RepeatOp, SequenceOp, StrategyOp

WIDTH = 80
INDENT = "  "
MODES = {mode.value for mode in AccessMode}
DECLARATIONS = ("reproducibility", "rank_rule", "assumes", "quiescence")
LOCAL_BODIES = (AccessOp, ChoiceOp, DerivedOp)
_PLACEHOLDER = t.value("?")


# ----------------------------------------------------------------------------
# Lexing.

_TOKENS = re.compile(
    r"""
    (?P<space>[ \t\r\n]+)
  | (?P<comment>//[^\n]*)
  | (?P<arrow>->)
  | (?P<value>%[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<symbol>@[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<number>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
  | (?P<string>"(?:[^"\\\n]|\\.)*")
  | (?P<word>[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<punct>[{}()\[\],:=^])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    column: int


def tokenize(text: str, source: str) -> list[Token]:
    tokens: list[Token] = []
    line, line_start, position = 1, 0, 0
    while position < len(text):
        match = _TOKENS.match(text, position)
        if match is None:
            raise _syntax(source, text, line, position - line_start, f"unexpected character {text[position]!r}")
        kind = match.lastgroup or ""
        if kind not in ("space", "comment"):
            tokens.append(Token(kind, match.group(), line, position - line_start))
        newlines = match.group().count("\n")
        if newlines:
            line += newlines
            line_start = match.start() + match.group().rfind("\n") + 1
        position = match.end()
    tokens.append(Token("end", "", line, position - line_start))
    return tokens


def _syntax(source: str, text: str, line: int, column: int, message: str, code: str = "iro-text-001") -> Rejection:
    lines = text.split("\n")
    shown = lines[line - 1] if 0 < line <= len(lines) else ""
    problems = Problems(Stage.PARSE)
    problems.add(code, message, source=f"{source}:{line}", fragment=f"{shown}\n{' ' * column}^")
    return Rejection(problems.found)


# ----------------------------------------------------------------------------
# Reading.


class Scope:
    """Names visible in a body.

    A local body's scope captures an enclosing value on its first use,
    appending an entry argument for it. An activity's scope sees only its
    captures and entries; ``outside`` is kept for diagnostics.
    """

    def __init__(self, parent: Scope | None = None, *, block: Block | None = None, outside: Scope | None = None):
        self.names: dict[str, SSAValue] = {}
        self.parent = parent
        self.block = block
        self.outside = outside
        self.captured: list[SSAValue] = []

    def visible(self, name: str) -> bool:
        return name in self.names or (self.parent is not None and self.parent.visible(name))

    def lookup(self, name: str) -> SSAValue | None:
        if name in self.names:
            return self.names[name]
        if self.parent is None or self.block is None:
            return None
        outer = self.parent.lookup(name)
        if outer is None:
            return None
        argument = self.block.insert_arg(outer.type, len(self.block.args))
        argument.name_hint = name[1:]
        self.names[name] = argument
        self.captured.append(outer)
        return argument

    def known_outside(self, name: str) -> bool:
        scope: Scope | None = self.outside
        while scope is not None:
            if scope.visible(name):
                return True
            scope = scope.outside
        return False


@dataclass
class Statement:
    """What the reader knows about the statement it is reading."""

    results: list[str]
    first: Token
    declarations: dict[str, Attribute] = field(default_factory=dict)


ReadForm = Callable[["Reader", Scope, Block, Statement], Operation]


class Reader:
    def __init__(self, text: str, composition: Composition, source: str):
        self.text = text
        self.source = source
        self.composition = composition
        self.tokens = tokenize(text, source)
        self.position = 0
        self.deferred_handlers: list[tuple[HandlerOp, Token]] = []
        self.lines: dict[Operation, Token] = {}
        self.current = self.peek()

    # Tokens.

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.position + offset, len(self.tokens) - 1)]

    def next(self) -> Token:
        token = self.peek()
        self.position += 1
        return token

    def at(self, text: str, offset: int = 0) -> bool:
        token = self.peek(offset)
        return token.kind in ("word", "punct", "arrow") and token.text == text

    def accept(self, text: str) -> bool:
        if self.at(text):
            self.position += 1
            return True
        return False

    def expect(self, text: str) -> Token:
        if not self.at(text):
            raise self.error(f"expected {text!r}, found {self.peek().text or 'the end'!r}")
        return self.next()

    def expect_kind(self, kind: str, what: str) -> Token:
        if self.peek().kind != kind:
            raise self.error(f"expected {what}, found {self.peek().text or 'the end'!r}")
        return self.next()

    def word(self, what: str = "a word") -> str:
        return self.expect_kind("word", what).text

    def symbol(self) -> str:
        return self.expect_kind("symbol", "a name such as @producer").text[1:]

    def integer(self) -> int:
        token = self.expect_kind("number", "a number")
        if "." in token.text:
            raise self.error("expected a whole number", token)
        return int(token.text)

    def error(self, message: str, token: Token | None = None, code: str = "iro-text-001", fix: str | None = None) -> Rejection:
        at = token or self.peek()
        rejection = _syntax(self.source, self.text, at.line, at.column, message, code)
        if fix is None:
            return rejection
        found = rejection.diagnostics[0]
        problems = Problems(Stage.PARSE)
        problems.add(found.code, found.message, source=found.source, fragment=found.fragment, fix=fix)
        return Rejection(problems.found)

    def string(self, token: Token) -> str:
        try:
            return json.loads(token.text)
        except ValueError as failed:
            raise self.error('a string uses JSON escapes, such as \\n or \\"', token) from failed

    # Values and types.

    def value(self, scope: Scope, block: Block, expected: Attribute | None = None) -> SSAValue:
        """A value: a name in scope, or an inline literal, which becomes a constant."""
        token = self.peek()
        if token.kind == "value":
            self.position += 1
            found = scope.lookup(token.text)
            if found is not None:
                return found
            if scope.known_outside(token.text):
                raise self.error(
                    f"{token.text} is not captured here",
                    token,
                    "iro-text-003",
                    fix=f"add {token.text} to the activity's capture(...)",
                )
            raise self.error(f"unknown value {token.text}", token, "iro-text-002")
        literal, type_ = self.literal()
        constant = ConstantOp.create(properties={"value": literal}, result_types=[type_])
        self.lines[constant] = self.current
        block.add_op(constant)
        return constant.results[0]

    def literal(self) -> tuple[Attribute, t.ValueType]:
        if self.accept("("):
            literal, _ = self.literal()
            self.expect(":")
            type_ = self.type_()
            self.expect(")")
            if not isinstance(type_, t.ValueType):
                raise self.error("a literal has a value type")
            return literal, type_
        token = self.next()
        if token.kind == "number":
            if "." in token.text or "e" in token.text.lower():
                return FloatAttr(float(token.text), f64), t.value("float")
            return IntegerAttr(int(token.text), i64), t.value("int")
        if token.kind == "string":
            return StringAttr(self.string(token)), t.value("str")
        if token.kind == "word" and token.text in ("true", "false"):
            return IntegerAttr(1 if token.text == "true" else 0, i1), t.value("bool")
        raise self.error(f"expected a value or a literal, found {token.text or 'the end'!r}", token)

    def type_(self) -> Attribute:
        name = self.word("a type")
        if name == "optional":
            return t.optional(self._value_type())
        if name == "retained":
            return t.retained(self._value_type())
        return t.value(name)

    def _value_type(self) -> t.ValueType:
        found = self.type_()
        if not isinstance(found, t.ValueType):
            raise self.error("expected a value type")
        return found

    def define(self, scope: Scope, name: str, value: SSAValue, token: Token) -> None:
        if scope.visible(name):
            raise self.error(f"{name} is already defined here", token, "iro-text-004", fix="give it another name")
        value.name_hint = name[1:]
        scope.names[name] = value

    # Clauses.

    def captures(self, scope: Scope, block: Block) -> list[tuple[str, SSAValue]]:
        """``capture(%a, %b as %c)``: the outer values and the names they take inside."""
        found: list[tuple[str, SSAValue]] = []
        if not self.accept("capture"):
            return found
        self.expect("(")
        while not self.at(")"):
            token = self.peek()
            outer = self.value(scope, block)
            inner = token.text
            if self.accept("as"):
                token = self.expect_kind("value", "a name")
                inner = token.text
            if any(name == inner for name, _ in found):
                raise self.error(f"{inner} is captured twice", token, "iro-text-004", fix="give each capture its own name")
            found.append((inner, outer))
            if not self.accept(","):
                break
        self.expect(")")
        return found

    def declarations(self, statement: Statement) -> None:
        while self.peek().kind == "word" and self.peek().text in DECLARATIONS:
            keyword = self.next().text
            self.expect("(")
            items: list[str] = []
            while not self.at(")"):
                token = self.next()
                if token.kind == "string":
                    items.append(self.string(token))
                elif token.kind == "word":
                    items.append(token.text)
                else:
                    raise self.error("expected a declared value", token)
                if not self.accept(","):
                    break
            self.expect(")")
            if keyword in ("reproducibility", "rank_rule"):
                if len(items) != 1:
                    raise self.error(f"{keyword} declares one value")
                statement.declarations[keyword] = StringAttr(items[0])
            else:
                key = "assumptions" if keyword == "assumes" else keyword
                statement.declarations[key] = ArrayAttr([StringAttr(x) for x in items])

    def location(self) -> str | None:
        if not self.at("loc") or not self.at("(", 1):
            return None
        self.position += 2
        path = self.string(self.expect_kind("string", "a file name"))
        if self.accept(":"):
            path = f"{path}:{self.integer()}"
        self.expect(")")
        return path

    def entries(self, expected: int) -> list[tuple[str, Attribute | None, Token]] | None:
        """``^(%a: int, %lease):`` when present; None when the entry list is omitted."""
        if not self.at("^"):
            return None
        self.next()
        self.expect("(")
        found: list[tuple[str, Attribute | None, Token]] = []
        while not self.at(")"):
            token = self.expect_kind("value", "an entry name")
            type_ = self.type_() if self.accept(":") else None
            found.append((token.text, type_, token))
            if not self.accept(","):
                break
        self.expect(")")
        self.expect(":")
        if len(found) != expected:
            raise self.error(f"this body takes {expected} entries, the list names {len(found)}")
        return found

    # Statements and bodies.

    def statement(self, scope: Scope, block: Block) -> Operation:
        first = self.peek()
        enclosing, self.current = self.current, first
        results: list[str] = []
        if first.kind == "value" and (self.at("=", 1) or self.at(",", 1)):
            while True:
                results.append(self.expect_kind("value", "a result name").text)
                if not self.accept(","):
                    break
            self.expect("=")
        start = self.peek()
        keyword = self.word("a word such as iro.call")
        read = self.form(keyword, start)
        statement = Statement(results, start)
        op = read(self, scope, block, statement)
        self.declarations(statement)
        origin = self.location()
        for key, declared in statement.declarations.items():
            op.properties[key] = declared
        if origin is not None:
            op.properties["origin"] = StringAttr(origin)
        if results and len(results) != len(op.results):
            raise self.error(f"{keyword} has {len(op.results)} results, {len(results)} are named", first)
        for name, result in zip(results, op.results, strict=False):
            self.define(scope, name, result, first)
        block.add_op(op)
        self.lines[op] = first
        self.current = enclosing
        return op

    def form(self, keyword: str, token: Token) -> ReadForm:
        if keyword not in self.composition:
            raise self.error(f"unknown word {keyword}", token, "iro-text-005")
        read = getattr(self.composition[keyword].form, "read", None)
        if callable(read):
            return read
        if keyword in READ_FORMS:
            return READ_FORMS[keyword]
        raise self.error(f"{keyword} has no written form", token, "iro-text-005")

    def body(self, scope: Scope, block: Block, *, local: bool) -> None:
        """Statements up to the closing brace, then the body's yield.

        A local body without a yield yields its last statement's results; any
        other body without one yields nothing.
        """
        last: Operation | None = None
        while not self.at("}"):
            if self.peek().kind == "end":
                raise self.error("this body is not closed", code="iro-text-006", fix="add the closing }")
            last = self.statement(scope, block)
        self.expect("}")
        if isinstance(last, YieldOp):
            return
        yielded = list(last.results) if local and last is not None else []
        block.add_op(YieldOp.create(operands=yielded))

    def isolated(self, outer: Scope, captured: list[tuple[str, SSAValue]], block: Block, offset: int) -> Scope:
        scope = Scope(outside=outer)
        for index, (name, _) in enumerate(captured):
            argument = block.args[offset + index]
            argument.name_hint = name[1:]
            scope.names[name] = argument
        return scope


# ----------------------------------------------------------------------------
# The core words' read forms.


def _read_resource(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    name = r.symbol()
    kind = r.word("a resource kind") if r.accept(":") else t.DEFAULT_RESOURCE_KIND
    properties: dict[str, Attribute] = {"sym_name": StringAttr(name)}
    reads: list[SSAValue] = []
    views: list[StringAttr] = []
    if r.accept("materialize"):
        r.expect("(")
        properties["materialize"] = SymbolRefAttr(r.symbol())
        while r.accept(","):
            reads.append(r.value(scope, block))
            r.expect("view")
            views.append(StringAttr(r.word("a view kind")))
        r.expect(")")
    if views:
        properties["read_views"] = ArrayAttr(views)
    return ResourceOp.create(operands=reads, result_types=[t.resource(kind)], properties=properties)


def _read_relationship(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    name = r.symbol()
    r.expect(":")
    kind = r.word("a relationship kind")
    r.expect("from")
    source = r.value(scope, block)
    r.expect("to")
    target = r.value(scope, block)
    return RelationshipOp.create(operands=[source, target], result_types=[t.relationship(kind)], properties={"sym_name": StringAttr(name)})


def _read_stream(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    name = r.symbol()
    r.expect(":")
    payload = r._value_type()
    r.expect("bounds")
    r.expect("(")
    r.expect("outstanding")
    outstanding = r.integer()
    r.expect(",")
    r.expect("ready")
    ready = r.integer()
    r.expect(")")
    admission, owners, lag = Admission.ANY_WITH_PROVENANCE.value, [], 0
    if r.accept("admit"):
        admission = r.word("an admission policy")
        r.expect("(")
        while not r.at(")"):
            if r.accept("max"):
                lag = r.integer()
            else:
                owners.append(SymbolRefAttr(r.symbol()))
            if not r.accept(","):
                break
        r.expect(")")
    return StreamOp.create(
        result_types=[t.stream(payload)],
        properties={
            "sym_name": StringAttr(name),
            "outstanding": IntegerAttr(outstanding, i64),
            "ready": IntegerAttr(ready, i64),
            "admission": StringAttr(admission),
            "owners": ArrayAttr(owners),
            "metric": StringAttr(DEFAULT_METRIC),
            "boundary": StringAttr(DEFAULT_COMPARISON),
            "max_lag": IntegerAttr(lag, i64),
        },
    )


def _read_take(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    source = r.value(scope, block)
    if not isinstance(source.type, t.StreamType):
        raise r.error("a take takes from a stream", s.first)
    return TakeOp.create(operands=[source], result_types=[t.claim(source.type.payload)])


def _read_borrow(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    item = r.value(scope, block)
    if not isinstance(item.type, t.ClaimType):
        raise r.error("a borrow reads a claim", s.first)
    return BorrowOp.create(operands=[item], result_types=[item.type.payload])


def _read_dispose(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    item = r.value(scope, block)
    return DisposeOp.create(operands=[item], properties={"disposition": StringAttr(r.word("a disposition"))})


def _read_fork(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    item = r.value(scope, block)
    return ForkOp.create(operands=[item], result_types=[item.type] * len(s.results))


def _read_release(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    return ReleaseOp.create(operands=[r.value(scope, block)])


def _read_call(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    implementation = r.symbol()
    r.expect("(")
    inputs: list[SSAValue] = []
    while not r.at(")"):
        inputs.append(r.value(scope, block))
        if not r.accept(","):
            break
    r.expect(")")
    outputs: list[Attribute] = []
    if r.accept("->"):
        outputs.append(r.type_())
        while r.accept(","):
            outputs.append(r.type_())
    call = CallOp.create(operands=inputs, result_types=outputs, properties={"implementation": SymbolRefAttr(implementation)})
    if any(isinstance(o, t.RetainedType) for o in outputs):
        call.properties["retains"] = ArrayAttr([IntegerAttr(i, i64) for i in retained_positions(call)])
    return call


def _read_yield(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    values: list[SSAValue] = []
    if _assignment_follows(r):
        return YieldOp.create(operands=values)
    if r.peek().kind in ("value", "number", "string") or r.at("true") or r.at("false") or r.at("("):
        values.append(r.value(scope, block))
        while r.accept(","):
            values.append(r.value(scope, block))
    return YieldOp.create(operands=values)


def _assignment_follows(r: Reader) -> bool:
    offset = 0
    while r.peek(offset).kind == "value":
        if r.at("=", offset + 1):
            return True
        if not r.at(",", offset + 1):
            return False
        offset += 2
    return False


def _read_request(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    handler = r.symbol()
    r.expect("(")
    payload = r.value(scope, block)
    r.expect(")")
    r.expect("from")
    source = r.symbol()
    r.expect("at")
    boundary = r.word("a boundary such as settled_success")
    return RequestOp.create(
        operands=[payload],
        properties={"handler": SymbolRefAttr(handler), "source": SymbolRefAttr(source), "boundary": StringAttr(boundary)},
    )


def _read_stop(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    target = r.symbol()
    r.expect("after")
    source = r.symbol()
    return StopOp.create(properties={"target": SymbolRefAttr(target), "source": SymbolRefAttr(source)})


def _read_selection(r: Reader) -> str | None:
    if r.accept("published"):
        r.expect("initial")
        return VersionSelection.PUBLISHED_INITIAL.value
    for word in (VersionSelection.CURRENT_AT_START, VersionSelection.AT_REQUEST):
        if r.accept(word.value):
            return word.value
    return None


def _read_access(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    subjects: list[SSAValue] = []
    uses: list[ResourceUseAttr] = []
    while True:
        subject = r.value(scope, block)
        route = view = None
        if r.accept("route"):
            route = r.symbol()
        elif r.accept("view"):
            view = r.word("a view kind")
        mode = r.word("an access mode")
        if mode not in MODES:
            raise r.error(f"unknown access mode {mode}", code="iro-text-007")
        selection = _read_selection(r)
        uses.append(ResourceUseAttr.of(AccessMode(mode), VersionSelection(selection) if selection else None, route=route, view=view))
        subjects.append(subject)
        if not r.accept(","):
            break
    r.declarations(s)
    r.expect("{")
    leases = [
        t.lease(subject.type, use.mode.data)
        for subject, use in zip(subjects, uses, strict=True)
        if isinstance(subject.type, t.ResourceType)
    ]
    if len(leases) != len(subjects):
        raise r.error("an access uses resources", s.first)
    inner = Block(arg_types=leases)
    local = Scope(scope, block=inner)
    _name_entries(r, local, inner, r.entries(len(leases)), 0)
    r.body(local, inner, local=True)
    yields = [v.type for v in _yield(inner).operands]
    return AccessOp.create(
        operands=[*subjects, *local.captured],
        result_types=yields,
        properties={"uses": ArrayAttr(uses)},
        regions=[Region(inner)],
    )


def _read_choice(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    condition = r.value(scope, block)
    r.declarations(s)
    present = (condition.type.payload,) if isinstance(condition.type, t.OptionalType) else ()
    first = Block(arg_types=list(present))
    first_scope = Scope(scope, block=first)
    r.expect("{")
    _name_entries(r, first_scope, first, r.entries(len(present)) if present else None, 0)
    r.body(first_scope, first, local=True)
    second = Block()
    second_scope = Scope(scope, block=second)
    if r.accept("else"):
        r.expect("{")
        r.body(second_scope, second, local=True)
    else:
        second.add_op(YieldOp.create())
    # Both alternatives receive every value either one captures, in one order.
    captured = list(dict.fromkeys([*first_scope.captured, *second_scope.captured]))
    _align_captures(first, first_scope, captured, len(present))
    _align_captures(second, second_scope, captured, 0)
    yields = [v.type for v in _yield(first).operands]
    return ChoiceOp.create(operands=[condition, *captured], result_types=yields, regions=[Region(first), Region(second)])


def _align_captures(block: Block, scope: Scope, captured: list[SSAValue], offset: int) -> None:
    """Gives an alternative an entry for every captured value, in the choice's order."""
    own = dict(zip(scope.captured, block.args[offset:], strict=True))
    for position, outer in enumerate(captured):
        index = offset + position
        if outer in own:
            argument = own[outer]
            if block.args.index(argument) != index:
                replacement = block.insert_arg(argument.type, index)
                replacement.name_hint = argument.name_hint
                argument.replace_all_uses_with(replacement)
                block.erase_arg(argument)
        else:
            block.insert_arg(outer.type, index)


def _read_derived(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    name = r.symbol()
    r.declarations(s)
    r.expect("{")
    inner = Block()
    local = Scope(scope, block=inner)
    r.body(local, inner, local=True)
    yields = [v.type for v in _yield(inner).operands]
    return DerivedOp.create(
        operands=local.captured, result_types=yields, properties={"sym_name": StringAttr(name)}, regions=[Region(inner)]
    )


def _read_sequence(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    name = r.symbol()
    captured = r.captures(scope, block)
    r.declarations(s)
    r.expect("{")
    inner = Block(arg_types=[v.type for _, v in captured])
    r.body(r.isolated(scope, captured, inner, 0), inner, local=False)
    return SequenceOp.create(operands=[v for _, v in captured], properties={"sym_name": StringAttr(name)}, regions=[Region(inner)])


def _read_parallel(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    name = r.symbol()
    captured = r.captures(scope, block)
    r.declarations(s)
    r.expect("{")
    branches: list[Region] = []
    while not r.at("}"):
        if r.peek().kind == "end":
            raise r.error("this body is not closed", code="iro-text-006", fix="add the closing }")
        branch = Block(arg_types=[v.type for _, v in captured])
        r.statement(r.isolated(scope, captured, branch, 0), branch)
        branch.add_op(YieldOp.create())
        branches.append(Region(branch))
    r.expect("}")
    return ParallelOp.create(operands=[v for _, v in captured], properties={"sym_name": StringAttr(name)}, regions=branches)


def _read_repeat(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    name = r.symbol()
    properties: dict[str, Attribute] = {"sym_name": StringAttr(name)}
    if r.accept("until"):
        properties["policy"] = SymbolRefAttr(r.symbol())
    else:
        properties["limit"] = IntegerAttr(r.integer(), i64)
        r.expect("times")
    captured = r.captures(scope, block)
    carried: list[tuple[str, SSAValue]] = []
    if r.accept("carry"):
        r.expect("(")
        while not r.at(")"):
            inner_name = r.expect_kind("value", "a carried value's name").text
            r.expect("=")
            carried.append((inner_name, r.value(scope, block)))
            if not r.accept(","):
                break
        r.expect(")")
    properties["carry_count"] = IntegerAttr(len(carried), i64)
    r.declarations(s)
    r.expect("{")
    inner = Block(arg_types=[*(v.type for _, v in captured), t.int_type(), *(v.type for _, v in carried)])
    local = r.isolated(scope, captured, inner, 0)
    _name_entries(r, local, inner, r.entries(1), len(captured))
    for index, (inner_name, _) in enumerate(carried):
        r.define(local, inner_name, inner.args[len(captured) + 1 + index], s.first)
    r.body(local, inner, local=False)
    return RepeatOp.create(
        operands=[*(v for _, v in captured), *(v for _, v in carried)],
        result_types=[v.type for _, v in carried],
        properties=properties,
        regions=[Region(inner)],
    )


def _read_producer(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    name = r.symbol()
    r.expect("into")
    endpoint = r.value(scope, block)
    captured = r.captures(scope, block)
    r.declarations(s)
    r.expect("{")
    inner = Block(arg_types=[*(v.type for _, v in captured), t.int_type()])
    local = r.isolated(scope, captured, inner, 0)
    _name_entries(r, local, inner, r.entries(1), len(captured))
    r.body(local, inner, local=False)
    return ProducerOp.create(
        operands=[endpoint, *(v for _, v in captured)], properties={"sym_name": StringAttr(name)}, regions=[Region(inner)]
    )


def _read_handler(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    name = r.symbol()
    r.expect("sources")
    r.expect("[")
    sources: list[SymbolRefAttr] = []
    while not r.at("]"):
        sources.append(SymbolRefAttr(r.symbol()))
        if not r.accept(","):
            break
    r.expect("]")
    r.expect("max")
    outstanding = r.integer()
    captured = r.captures(scope, block)
    r.declarations(s)
    r.expect("{")
    entry = r.entries(1)
    request = entry[0][1] if entry and entry[0][1] is not None else _PLACEHOLDER
    if entry and entry[0][1] is None:
        raise r.error("a handler's entry names its type", entry[0][2])
    inner = Block(arg_types=[*(v.type for _, v in captured), request])
    local = r.isolated(scope, captured, inner, 0)
    _name_entries(r, local, inner, entry, len(captured))
    r.body(local, inner, local=False)
    handler = HandlerOp.create(
        operands=[v for _, v in captured],
        properties={
            "sym_name": StringAttr(name),
            "sources": ArrayAttr(sources),
            "request": request,
            "max_outstanding": IntegerAttr(outstanding, i64),
        },
        regions=[Region(inner)],
    )
    if request == _PLACEHOLDER:
        r.deferred_handlers.append((handler, s.first))
    return handler


def _read_filter(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
    item = r.value(scope, block)
    if not isinstance(item.type, t.ClaimType):
        raise r.error("a filter takes a claim", s.first)
    captured = r.captures(scope, block)
    r.declarations(s)
    r.expect("{")
    inner = Block(arg_types=[item.type.payload, *(v.type for _, v in captured)])
    local = r.isolated(scope, captured, inner, 1)
    _name_entries(r, local, inner, r.entries(1), 0)
    r.body(local, inner, local=False)
    return FilterOp.create(
        operands=[item, *(v for _, v in captured)], result_types=[t.optional(item.type.payload)], regions=[Region(inner)]
    )


def _name_entries(r: Reader, scope: Scope, block: Block, entries: list[tuple[str, Attribute | None, Token]] | None, offset: int) -> None:
    if entries is None:
        return
    for index, (name, written, token) in enumerate(entries):
        argument = block.args[offset + index]
        if written is not None and written != argument.type and not isinstance(argument.type, t.LeaseType):
            raise r.error(f"{name} is a {_type_text(argument.type)} here", token, "iro-text-008")
        r.define(scope, name, argument, token)


def _yield(block: Block) -> YieldOp:
    last = block.last_op
    assert isinstance(last, YieldOp)
    return last


READ_FORMS: dict[str, ReadForm] = {
    "iro.resource": _read_resource,
    "iro.relationship": _read_relationship,
    "iro.stream": _read_stream,
    "iro.take": _read_take,
    "iro.borrow": _read_borrow,
    "iro.dispose": _read_dispose,
    "iro.fork": _read_fork,
    "iro.release": _read_release,
    "iro.call": _read_call,
    "iro.yield": _read_yield,
    "iro.request": _read_request,
    "iro.stop": _read_stop,
    "iro.access": _read_access,
    "iro.choice": _read_choice,
    "iro.derived": _read_derived,
    "iro.sequence": _read_sequence,
    "iro.parallel": _read_parallel,
    "iro.repeat": _read_repeat,
    "iro.producer": _read_producer,
    "iro.handler": _read_handler,
    "iro.filter": _read_filter,
}


def parse(text: str, composition: Composition | None = None, *, source: str = "<written form>") -> StrategyOp:
    """Reads one strategy from its written form and verifies its structure.

    Problems are reported as a rejection with source-attributed diagnostics.
    """
    reader = Reader(text, composition or core(), source)
    reader.expect("iro.strategy")
    name = reader.symbol()
    names: list[SymbolRefAttr] = []
    versions: list[StringAttr] = []
    if reader.accept("uses"):
        reader.expect("(")
        while not reader.at(")"):
            names.append(SymbolRefAttr(reader.symbol()))
            version = reader.word("a version such as v1")
            if not re.fullmatch(r"v\d+", version):
                raise reader.error(f"expected a version such as v1, found {version}")
            versions.append(StringAttr(version[1:]))
            if not reader.accept(","):
                break
        reader.expect(")")
    statement = Statement([], reader.peek())
    reader.declarations(statement)
    reader.expect("{")
    body = Block()
    scope = Scope()
    while not reader.at("}"):
        if reader.peek().kind == "end":
            raise reader.error("the strategy is not closed", code="iro-text-006", fix="add the closing }")
        reader.statement(scope, body)
    reader.expect("}")
    origin = reader.location()
    if reader.peek().kind != "end":
        raise reader.error("a written form holds one strategy")
    properties: dict[str, Attribute] = {"sym_name": StringAttr(name), **statement.declarations}
    if names:
        properties |= {"uses": ArrayAttr(names), "use_versions": ArrayAttr(versions)}
    if origin is not None:
        properties["origin"] = StringAttr(origin)
    strategy = StrategyOp.create(properties=properties, regions=[Region(body)])
    reader.lines[strategy] = reader.tokens[0]
    _resolve_handlers(reader, strategy)
    _check(reader, strategy)
    return strategy


def _check(reader: Reader, strategy: StrategyOp) -> None:
    """Verifies every operation on its own, so each problem names its statement's line."""
    problems = Problems(Stage.PARSE)

    def line(op: Operation) -> str:
        current: Operation | None = op
        while current is not None and current not in reader.lines:
            current = current.parent_op()
        token = reader.lines.get(current) if current is not None else None
        return f"{reader.source}:{token.line}" if token is not None else reader.source

    named: dict[str, Operation] = {}
    for op in strategy.walk():
        name = sym_name(op)
        if name is None or op is strategy:
            continue
        if name in named:
            problems.add(
                "iro-text-011", f"@{name} names two statements", names=(f"@{name}",), source=line(op), fix="give each its own name"
            )
        else:
            named[name] = op
    for op in strategy.walk():
        try:
            op.verify(verify_nested_ops=False)
        except VerifyException as failed:
            problems.add("iro-text-009", _explain(failed), source=line(op), path=path(op))
    problems.reject_if_any()


_WRITTEN_TYPES = (
    (re.compile(r'!iro\.value<"([^"]+)">'), r"\1"),
    (re.compile(r'!iro\.(?:resource_type|relationship_type)<"([^"]+)">'), r"\1"),
    (re.compile(r'!iro\.lease<"[^"]+", "([^"]+)">'), r"\1 lease"),
    (re.compile(r"!iro\.(optional|retained|claim)<([^<>]+)>"), r"\1 \2"),
    (re.compile(r"!iro\.stream_type<([^<>]+)>"), r"stream \1"),
    (re.compile(r"!iro\.request_type<([^<>]+)>"), r"request \1"),
)


def _explain(failed: VerifyException) -> str:
    """xDSL's message without the operation dump, in the written form's type names."""
    cause = failed.__cause__ if isinstance(failed.__cause__, VerifyException) else failed
    message = " ".join(line.strip() for line in str(cause).split("\n") if line.strip())
    message = message.removeprefix("Operation does not verify: ")
    for pattern, replacement in _WRITTEN_TYPES:
        message = pattern.sub(replacement, message)
    return message


def _resolve_handlers(reader: Reader, strategy: StrategyOp) -> None:
    """Gives a handler whose entry was omitted the value type its requests carry."""
    carried: dict[str, set[Attribute]] = {}
    for op in strategy.walk():
        if isinstance(op, RequestOp):
            carried.setdefault(op.handler.root_reference.data, set()).add(op.payload.type)
    for handler, token in reader.deferred_handlers:
        name = handler.sym_name.data if handler.sym_name is not None else "?"
        types = carried.get(name, set())
        if len(types) != 1:
            raise reader.error(
                f"@{name}'s request type cannot be read from its requests",
                token,
                "iro-text-010",
                fix="write the handler's entry, such as ^(%step: int):",
            )
        (request,) = types
        block = handler.regions[0].block
        placeholder = block.args[-1]
        replacement = block.insert_arg(request, len(block.args))
        placeholder.replace_all_uses_with(replacement)
        block.erase_arg(placeholder)
        handler.properties["request"] = request


# ----------------------------------------------------------------------------
# Writing.


def _type_text(type_: Attribute) -> str:
    """As written in a statement: a stream's, claim's or request's type is its value's."""
    if isinstance(type_, t.StreamType | t.ClaimType | t.RequestType):
        return type_.payload.text
    return t.written(type_)


def _literal_text(op: ConstantOp) -> str:
    value = op.value
    if isinstance(value, IntegerAttr) and isinstance(value.type, IntegerType) and value.type.width.data == 1:
        text = "true" if value.value.data else "false"
    elif isinstance(value, IntegerAttr):
        text = str(value.value.data)
    elif isinstance(value, FloatAttr):
        text = repr(float(value.value.data))
    elif isinstance(value, StringAttr):
        text = json.dumps(value.data, ensure_ascii=False)
    else:
        raise ValueError(f"no literal form for {value}")
    natural = literal_type(value)
    written = op.result.type
    if isinstance(written, t.ValueType) and written.text == natural:
        return text
    return f"({text} : {_type_text(written)})"


class Names:
    """Value names while printing, scoped like the reader's."""

    def __init__(self, parent: Names | None = None, *, isolated: bool = False):
        self.parent = None if isolated else parent
        self.of: dict[SSAValue, str] = {}
        self.taken: set[str] = set()
        self.counter = parent.counter if parent is not None else [0]

    def visible(self, name: str) -> bool:
        return name in self.taken or (self.parent is not None and self.parent.visible(name))

    def get(self, value: SSAValue) -> str | None:
        if value in self.of:
            return self.of[value]
        return self.parent.get(value) if self.parent is not None else None

    def bind(self, value: SSAValue, preferred: str | None = None) -> str:
        base = preferred or (f"%{value.name_hint}" if value.name_hint else None)
        if base is None:
            base = f"%v{self.counter[0]}"
            self.counter[0] += 1
        name, suffix = base, 2
        while self.visible(name):
            name, suffix = f"{base}_{suffix}", suffix + 1
        self.of[value] = name
        self.taken.add(name)
        return name


@dataclass
class Rendered:
    """A statement as printed: its result names, the text after ``=`` and any following lines."""

    results: str
    head: str
    rest: list[str]
    has_body: bool
    continuation: list[str] = field(default_factory=list)
    clauses: str = ""
    suffix: str = ""


class Writer:
    def __init__(self, strategy: StrategyOp, *, locations: bool):
        self.locations = locations
        self.requested: dict[str, set[Attribute]] = {}
        for op in strategy.walk():
            if isinstance(op, RequestOp):
                self.requested.setdefault(op.handler.root_reference.data, set()).add(op.payload.type)

    def operand(self, names: Names, value: SSAValue) -> str:
        owner = value.owner if isinstance(value, OpResult) else None
        if isinstance(owner, ConstantOp):
            return _literal_text(owner)
        found = names.get(value)
        if found is None:
            raise ValueError(f"value {value} is used before it is printed")
        return found

    def operands(self, names: Names, values: Sequence[SSAValue]) -> str:
        return ", ".join(self.operand(names, v) for v in values)

    def tail(self, op: Operation) -> str:
        parts: list[str] = []
        for key, keyword in (("reproducibility", "reproducibility"), ("rank_rule", "rank_rule")):
            if key in op.properties:
                declared = op.properties[key]
                assert isinstance(declared, StringAttr)
                parts.append(f"{keyword}({declared.data})")
        for key, keyword in (("assumptions", "assumes"), ("quiescence", "quiescence")):
            declared = op.properties.get(key)
            if isinstance(declared, ArrayAttr) and declared.data:
                items = [json.dumps(x.data) if key == "assumptions" else x.data for x in declared.data if isinstance(x, StringAttr)]
                parts.append(f"{keyword}({', '.join(items)})")
        return "".join(f" {p}" for p in parts)

    def location(self, op: Operation) -> str:
        origin = op.properties.get("origin")
        if not self.locations or not isinstance(origin, StringAttr):
            return ""
        path, _, line = origin.data.rpartition(":")
        if path and line.isdigit():
            return f" loc({json.dumps(path)}:{line})"
        return f" loc({json.dumps(origin.data)})"

    # Blocks.

    def block(self, block: Block, names: Names, indent: str, *, local: bool, spaced: bool = False) -> list[str]:
        ops = [op for op in block.ops if not isinstance(op, ConstantOp)]
        last = ops[-1] if ops else None
        if isinstance(last, YieldOp):
            before = ops[-2] if len(ops) > 1 else None
            if local:
                drop = tuple(last.operands) == (tuple(before.results) if before is not None else ())
            else:
                drop = not last.operands
            if drop:
                ops = ops[:-1]
        rendered = [self.statement(op, names, indent) for op in ops]
        lines: list[str] = []
        run: list[Rendered] = []

        def flush() -> None:
            if not run:
                return
            width = max(len(item.results) for item in run)
            for item in run:
                lines.extend(self.join(item, indent, width))
            run.clear()

        previous: Rendered | None = None
        for item in rendered:
            if spaced and previous is not None and (item.has_body or previous.has_body):
                flush()
                lines.append("")
            if item.results and not item.has_body:
                run.append(item)
            else:
                flush()
                lines.extend(self.join(item, indent, len(item.results)))
            previous = item
        flush()
        return lines

    def join(self, item: Rendered, indent: str, width: int) -> list[str]:
        prefix = f"{item.results.ljust(width)} = " if item.results else ""
        whole = f"{indent}{prefix}{item.head}{item.clauses}{item.suffix}"
        moved = (item.clauses + item.suffix).strip()
        if len(whole) > WIDTH and moved not in ("", "{"):
            first = [f"{indent}{prefix}{item.head}", f"{indent}{INDENT * 2}{moved}"]
        else:
            first = [whole]
        column = " " * (len(indent) + len(prefix) + 2)
        return [*first, *(column + line for line in item.continuation), *item.rest]

    def statement(self, op: Operation, names: Names, indent: str) -> Rendered:
        write = WRITE_FORMS.get(op.name)
        if write is None:
            raise ValueError(f"{op.name} has no written form")
        rendered = write(self, op, names, indent)
        visible = [r for r in op.results if r.uses and not all(self._implicitly_yielded(r, use.operation) for use in r.uses)]
        if visible or (op.results and isinstance(op, ForkOp)):
            rendered.results = ", ".join(names.bind(r) for r in op.results)
        else:
            for r in op.results:
                names.bind(r)
        return rendered

    def _implicitly_yielded(self, value: SSAValue, user: Operation) -> bool:
        if not isinstance(user, YieldOp):
            return False
        parent = user.parent_op()
        block = user.parent_block()
        if not isinstance(parent, LOCAL_BODIES) or block is None:
            return False
        ops = [o for o in block.ops if not isinstance(o, ConstantOp)]
        before = ops[-2] if len(ops) > 1 else None
        return before is not None and tuple(user.operands) == tuple(before.results)

    def body(self, op: Operation, region: Region, names: Names, indent: str, entries: list[tuple[int, bool]], *, local: bool) -> list[str]:
        """``{``, the entry list when any entry is used, the statements and ``}``.

        ``entries`` lists the protocol-supplied argument positions and whether
        each prints its type.
        """
        block = region.block
        lines: list[str] = []
        supplied = [block.args[i] for i, _ in entries]
        if any(arg.uses for arg in supplied):
            written = []
            for (_, typed), argument in zip(entries, supplied, strict=True):
                name = names.bind(argument)
                written.append(f"{name}: {_type_text(argument.type)}" if typed else name)
            lines.append(f"{indent}^({', '.join(written)}):")
        else:
            for argument in supplied:
                names.bind(argument)
        spaced = isinstance(op, StrategyOp | ParallelOp)
        lines.extend(self.block(block, names, indent + INDENT, local=local, spaced=spaced))
        return lines

    def captures(self, op: Operation, outer: Names, inner: Names, block: Block, offset: int, count: int) -> str:
        if count == 0:
            return ""
        written = []
        for index in range(count):
            value = op.operands[offset + index]
            outside = self.operand(outer, value)
            argument = block.args[self._capture_argument(op, index)]
            preferred = f"%{argument.name_hint}" if argument.name_hint else outside
            name = inner.bind(argument, preferred)
            written.append(outside if name == outside else f"{outside} as {name}")
        return f" capture({', '.join(written)})"

    @staticmethod
    def _capture_argument(op: Operation, index: int) -> int:
        if isinstance(op, FilterOp):
            return index + 1
        return index


# ----------------------------------------------------------------------------
# The core words' write forms. Each returns the statement after its ``=``.

WriteForm = Callable[[Writer, Operation, Names, str], Rendered]


def _write_leaf(head: str) -> Rendered:
    return Rendered("", head, [], False)


def _w_resource(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, ResourceOp)
    kind = op.handle.type.kind.data if isinstance(op.handle.type, t.ResourceType) else t.DEFAULT_RESOURCE_KIND
    head = f"iro.resource @{op.sym_name.data}"
    if kind != t.DEFAULT_RESOURCE_KIND:
        head += f" : {kind}"
    if op.materialize is not None:
        reads = op.read_views.data if op.read_views is not None else ()
        extra = "".join(f", {w.operand(names, v)} view {view.data}" for v, view in zip(op.reads, reads, strict=True))
        head += f" materialize(@{op.materialize.root_reference.data}{extra})"
    return _write_leaf(head + w.tail(op) + w.location(op))


def _w_relationship(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, RelationshipOp)
    kind = _type_text(op.handle.type)
    head = f"iro.relationship @{op.sym_name.data} : {kind} from {w.operand(names, op.source)} to {w.operand(names, op.target)}"
    return _write_leaf(head + w.tail(op) + w.location(op))


def _w_stream(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, StreamOp)
    if op.metric.data != DEFAULT_METRIC or op.boundary.data != DEFAULT_COMPARISON:
        raise ValueError(f"@{op.sym_name.data}: only the default admission metric and comparison have a written form")
    rendered = _write_leaf(f"iro.stream @{op.sym_name.data} : {_type_text(op.endpoint.type)}")
    rendered.continuation.append(f"bounds(outstanding {op.outstanding.value.data}, ready {op.ready.value.data})")
    owners = [f"@{ref.root_reference.data}" for ref in op.owners.data]
    lag = op.max_lag.value.data
    if op.admission.data != Admission.ANY_WITH_PROVENANCE or owners:
        arguments = owners + ([f"max {lag}"] if lag else [])
        rendered.continuation.append(f"admit {op.admission.data}({', '.join(arguments)})")
    tail = w.tail(op) + w.location(op)
    if tail:
        rendered.continuation[-1] += tail
    return rendered


def _w_take(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, TakeOp)
    return _write_leaf(f"iro.take {w.operand(names, op.source)}" + w.tail(op) + w.location(op))


def _w_borrow(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, BorrowOp)
    return _write_leaf(f"iro.borrow {w.operand(names, op.item)}" + w.tail(op) + w.location(op))


def _w_dispose(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, DisposeOp)
    return _write_leaf(f"iro.dispose {w.operand(names, op.item)} {op.disposition.data}" + w.tail(op) + w.location(op))


def _w_fork(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, ForkOp)
    return _write_leaf(f"iro.fork {w.operand(names, op.item)}" + w.tail(op) + w.location(op))


def _w_release(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, ReleaseOp)
    return _write_leaf(f"iro.release {w.operand(names, op.product)}" + w.tail(op) + w.location(op))


def _w_call(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, CallOp)
    head = f"iro.call @{op.implementation.root_reference.data}({w.operands(names, op.inputs)})"
    if op.outputs:
        head += " -> " + ", ".join(_type_text(r.type) for r in op.outputs)
    return _write_leaf(head + w.tail(op) + w.location(op))


def _w_yield(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, YieldOp)
    head = "iro.yield" + (f" {w.operands(names, op.values)}" if op.values else "")
    return _write_leaf(head + w.tail(op) + w.location(op))


def _w_request(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, RequestOp)
    head = (
        f"iro.request @{op.handler.root_reference.data}({w.operand(names, op.payload)})"
        f" from @{op.source.root_reference.data} at {op.boundary.data}"
    )
    return _write_leaf(head + w.tail(op) + w.location(op))


def _w_stop(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, StopOp)
    head = f"iro.stop @{op.target.root_reference.data} after @{op.source.root_reference.data}"
    return _write_leaf(head + w.tail(op) + w.location(op))


def _with_body(head: str, op: Operation, w: Writer, lines: list[str], indent: str, clauses: str = "") -> Rendered:
    return Rendered("", head, [*lines, f"{indent}}}{w.location(op)}"], True, clauses=clauses, suffix=w.tail(op) + " {")


def _w_access(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, AccessOp)
    uses = resource_uses(op)
    written = []
    for subject, use in zip(op.operands, uses, strict=False):
        part = w.operand(names, subject)
        if use.route.data:
            part += f" route @{use.route.data}"
        elif use.view.data:
            part += f" view {use.view.data}"
        part += f" {use.mode.data}"
        if use.selection.data != DEFAULT_SELECTION[AccessMode(use.mode.data)]:
            part += f" {use.selection.data}"
        written.append(part)
    inner = Names(names)
    block = op.body.block
    for index, outer in enumerate(op.operands[len(uses) :]):
        argument = block.args[len(uses) + index]
        inner.of[argument] = w.operand(names, outer)
    lines = w.body(op, op.body, inner, indent, [(i, False) for i in range(len(uses))], local=True)
    return _with_body(f"iro.access {', '.join(written)}", op, w, lines, indent)


def _local_inner(w: Writer, op: Operation, names: Names, block: Block, offset: int) -> Names:
    inner = Names(names)
    for index, outer in enumerate(op.operands[offset:]):
        inner.of[block.args[index + (len(block.args) - (len(op.operands) - offset))]] = w.operand(names, outer)
    return inner


def _w_choice(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, ChoiceOp)
    first, second = op.regions
    present = isinstance(op.operands[0].type, t.OptionalType)
    first_names = _local_inner(w, op, names, first.block, 1)
    lines = w.body(op, first, first_names, indent, [(0, True)] if present else [], local=True)
    second_names = _local_inner(w, op, names, second.block, 1)
    alternative = w.body(op, second, second_names, indent, [], local=True)
    head = f"iro.choice {w.operand(names, op.operands[0])}" + w.tail(op) + " {"
    if len(alternative) > 0:
        rest = [*lines, f"{indent}}} else {{", *alternative, f"{indent}}}{w.location(op)}"]
    else:
        rest = [*lines, f"{indent}}}{w.location(op)}"]
    return Rendered("", head, rest, True)


def _w_derived(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, DerivedOp)
    inner = _local_inner(w, op, names, op.body.block, 0)
    lines = w.body(op, op.body, inner, indent, [], local=True)
    name = op.sym_name.data if op.sym_name is not None else "derived"
    return _with_body(f"iro.derived @{name}", op, w, lines, indent)


def _named(op: Operation) -> str:
    return sym_name(op) or "unnamed"


def _w_sequence(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, SequenceOp)
    inner = Names(names, isolated=True)
    captures = w.captures(op, names, inner, op.body.block, 0, len(op.operands))
    lines = w.body(op, op.body, inner, indent, [], local=False)
    return _with_body(f"iro.sequence @{_named(op)}", op, w, lines, indent, captures)


def _w_parallel(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, ParallelOp)
    count = len(op.operands)
    outer_captures = ""
    lines: list[str] = []
    shared: list[str] = []
    for index, region in enumerate(op.regions):
        inner = Names(names, isolated=True)
        if index == 0:
            outer_captures = w.captures(op, names, inner, region.block, 0, count)
            shared = [inner.of[argument] for argument in region.block.args[:count]]
        else:
            for argument, name in zip(region.block.args[:count], shared, strict=True):
                inner.of[argument] = name
                inner.taken.add(name)
        branch = w.block(region.block, inner, indent + INDENT, local=False)
        if index > 0:
            lines.append("")
        lines.extend(branch)
    return _with_body(f"iro.parallel @{_named(op)}", op, w, lines, indent, outer_captures)


def _w_repeat(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, RepeatOp)
    carried = op.carry_count.value.data
    captured = len(op.operands) - carried
    inner = Names(names, isolated=True)
    block = op.body.block
    captures = w.captures(op, names, inner, block, 0, captured)
    count = f"until @{op.policy.root_reference.data}" if op.policy is not None else f"{op.limit.value.data if op.limit else 0} times"
    words = f"iro.repeat @{_named(op)} {count}"
    carry = ""
    if carried:
        parts = []
        for index in range(carried):
            argument = block.args[captured + 1 + index]
            outside = w.operand(names, op.operands[captured + index])
            parts.append(f"{inner.bind(argument)} = {outside}")
        carry = f" carry({', '.join(parts)})"
    lines = w.body(op, op.body, inner, indent, [(captured, True)], local=False)
    return _with_body(words, op, w, lines, indent, captures + carry)


def _w_producer(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, ProducerOp)
    inner = Names(names, isolated=True)
    block = op.body.block
    captures = w.captures(op, names, inner, block, 1, len(op.operands) - 1)
    words = f"iro.producer @{_named(op)} into {w.operand(names, op.operands[0])}"
    lines = w.body(op, op.body, inner, indent, [(len(op.operands) - 1, True)], local=False)
    return _with_body(words, op, w, lines, indent, captures)


def _w_handler(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, HandlerOp)
    inner = Names(names, isolated=True)
    block = op.body.block
    captures = w.captures(op, names, inner, block, 0, len(op.operands))
    sources = ", ".join(f"@{ref.root_reference.data}" for ref in op.sources.data)
    words = f"iro.handler @{_named(op)} sources [{sources}] max {op.max_outstanding.value.data}"
    request = block.args[-1]
    inferable = w.requested.get(_named(op)) == {request.type}
    if not request.uses and not inferable:
        # The type cannot be read back from requests, so the entry is written.
        name = inner.bind(request)
        lines = [f"{indent}^({name}: {_type_text(request.type)}):", *w.block(block, inner, indent + INDENT, local=False)]
    else:
        lines = w.body(op, op.body, inner, indent, [(len(block.args) - 1, True)], local=False)
    return _with_body(words, op, w, lines, indent, captures)


def _w_filter(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
    assert isinstance(op, FilterOp)
    inner = Names(names, isolated=True)
    block = op.body.block
    captures = w.captures(op, names, inner, block, 1, len(op.operands) - 1)
    lines = w.body(op, op.body, inner, indent, [(0, True)], local=False)
    return _with_body(f"iro.filter {w.operand(names, op.operands[0])}", op, w, lines, indent, captures)


WRITE_FORMS: dict[str, WriteForm] = {
    "iro.resource": _w_resource,
    "iro.relationship": _w_relationship,
    "iro.stream": _w_stream,
    "iro.take": _w_take,
    "iro.borrow": _w_borrow,
    "iro.dispose": _w_dispose,
    "iro.fork": _w_fork,
    "iro.release": _w_release,
    "iro.call": _w_call,
    "iro.yield": _w_yield,
    "iro.request": _w_request,
    "iro.stop": _w_stop,
    "iro.access": _w_access,
    "iro.choice": _w_choice,
    "iro.derived": _w_derived,
    "iro.sequence": _w_sequence,
    "iro.parallel": _w_parallel,
    "iro.repeat": _w_repeat,
    "iro.producer": _w_producer,
    "iro.handler": _w_handler,
    "iro.filter": _w_filter,
}


def print_strategy(strategy: StrategyOp, *, locations: bool = False) -> str:
    """The strategy's written form. ``locations`` adds where each statement was authored."""
    writer = Writer(strategy, locations=locations)
    names = Names()
    head = f"iro.strategy @{strategy.sym_name.data}"
    uses = [
        f"@{ref.root_reference.data} v{version.data}" for ref, version in zip(strategy.uses or (), strategy.use_versions or (), strict=True)
    ]
    clause = f" uses({', '.join(uses)})" if uses else ""
    tail = writer.tail(strategy) + " {"
    if len(head + clause + tail) <= WIDTH or not clause:
        header = [head + clause + tail]
    else:
        header = [head, *_wrap_list("    uses(", uses, ")" + tail)]
    lines = [*header, *writer.block(strategy.body.block, names, INDENT, local=False, spaced=True), "}" + writer.location(strategy)]
    return "\n".join(lines) + "\n"


def _wrap_list(opening: str, items: list[str], closing: str) -> list[str]:
    """Items after ``opening``, filling lines up to the width and aligning continuations."""
    lines: list[str] = []
    current = opening
    for index, item in enumerate(items):
        piece = item + ("," if index < len(items) - 1 else closing)
        candidate = piece if current in (opening, " " * len(opening)) else " " + piece
        if len(current) + len(candidate) > WIDTH and current.strip() not in ("", opening.strip()):
            lines.append(current)
            current = " " * len(opening) + piece
        else:
            current += candidate
    lines.append(current)
    return lines


def round_trips(text: str, composition: Composition | None = None) -> bool:
    """Whether printing what was read gives the same text."""
    return print_strategy(parse(text, composition)) == text


__all__ = ["Reader", "Token", "Writer", "parse", "print_strategy", "round_trips", "tokenize"]
