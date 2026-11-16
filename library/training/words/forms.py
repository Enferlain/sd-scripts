"""The written forms of training's words, which mirror their Python face (D18).

A unit prints as ``iro.train.unit @denoiser of %unet optimizer @adamw8bit(lr 1e-05) max_grad_norm 1.0``,
a window as ``iro.train.window %unit members 4 { ... }`` and a contribution as
``iro.train.contribute %loss to %unit``. The profile prints only when it is
not the standard one.
"""

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr, SymbolRefAttr, f64, i64
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue

from library.iro.text import Names, Reader, Rendered, Scope, Statement, Writer, literal_text, local_inner, with_body, write_leaf
from library.iro.words import types as t
from library.training.words.operations import STANDARD, UNIT_KIND, ContributeOp, ProfileAttr, UnitOp, WindowOp


class UnitForm:
    @staticmethod
    def read(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
        name = r.symbol()
        r.expect("of")
        members: list[SSAValue] = [r.value(scope, block)]
        while r.accept(","):
            members.append(r.value(scope, block))
        r.expect("optimizer")
        properties: dict[str, Attribute] = {"sym_name": StringAttr(name), "optimizer": SymbolRefAttr(r.symbol())}
        names: list[StringAttr] = []
        values: list[Attribute] = []
        if r.accept("("):
            while not r.at(")"):
                names.append(StringAttr(r.word("a setting's name")))
                if r.at("("):
                    raise r.error("a setting's type is its literal's; write the literal without a type", r.peek())
                value, _ = r.literal()
                values.append(value)
                if not r.accept(","):
                    break
            r.expect(")")
        if names:
            properties["setting_names"] = ArrayAttr(names)
            properties["setting_values"] = ArrayAttr(values)
        if r.accept("max_grad_norm"):
            if r.at("("):
                raise r.error("a max_grad_norm is a plain number", r.peek())
            limit, _ = r.literal()
            number = limit.value.data if isinstance(limit, FloatAttr | IntegerAttr) else None
            if number is None or isinstance(number, bool):
                raise r.error("a unit's max_grad_norm is a number")
            properties["max_grad_norm"] = FloatAttr(float(number), f64)
        profile = r.symbol() if r.accept("profile") else STANDARD
        properties["profile"] = ProfileAttr(StringAttr(profile))
        r.declarations(s)
        return UnitOp.create(operands=members, result_types=[t.resource(UNIT_KIND)], properties=properties)

    @staticmethod
    def write(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
        assert isinstance(op, UnitOp)
        members = ", ".join(w.operand(names, member) for member in op.members)
        head = f"iro.train.unit @{op.sym_name.data} of {members} optimizer @{op.optimizer.root_reference.data}"
        if op.setting_names is not None and op.setting_values is not None:
            settings = ", ".join(f"{n.data} {literal_text(v)}" for n, v in zip(op.setting_names.data, op.setting_values.data, strict=True))
            head += f"({settings})"
        if op.max_grad_norm is not None:
            head += f" max_grad_norm {literal_text(op.max_grad_norm)}"
        if op.profile.profile.data != STANDARD:
            head += f" profile @{op.profile.profile.data}"
        return write_leaf(head + w.tail(op) + w.location(op))


class WindowForm:
    @staticmethod
    def read(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
        unit = r.value(scope, block)
        r.expect("members")
        members = r.integer()
        r.declarations(s)
        r.expect("{")
        inner = Block()
        local = Scope(scope, block=inner)
        r.body(local, inner, local=False)
        return WindowOp.create(operands=[unit, *local.captured], properties={"members": IntegerAttr(members, i64)}, regions=[Region(inner)])

    @staticmethod
    def write(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
        assert isinstance(op, WindowOp)
        inner = local_inner(w, op, names, op.body.block, 1)
        lines = w.body(op, op.body, inner, indent, [], local=False)
        return with_body(f"iro.train.window {w.operand(names, op.unit)} members {op.members.value.data}", op, w, lines, indent)


class ContributeForm:
    @staticmethod
    def read(r: Reader, scope: Scope, block: Block, s: Statement) -> Operation:
        loss = r.value(scope, block)
        r.expect("to")
        unit = r.value(scope, block)
        r.declarations(s)
        return ContributeOp.create(operands=[loss, unit])

    @staticmethod
    def write(w: Writer, op: Operation, names: Names, indent: str) -> Rendered:
        assert isinstance(op, ContributeOp)
        return write_leaf(f"iro.train.contribute {w.operand(names, op.loss)} to {w.operand(names, op.unit)}" + w.tail(op) + w.location(op))


__all__ = ["ContributeForm", "UnitForm", "WindowForm"]
