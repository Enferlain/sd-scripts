"""Two domains using the same scoped-product and owner contracts."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region

from .contracts import AccessMode
from .dialects import CapabilityType, PayloadType, ProgramOp, ReleaseProductOp, RetainedType, integer
from .fixtures import SCALAR, access, call, end, node, props, resource
from .product_contracts import initial as initial, manifest as manifest


def program(kind: str = "buffer", *, derived: bool = False, update: str = "none") -> ModuleOp:
    storage, output = resource("storage", kind), resource("result", "product_result")
    mode = AccessMode.OWNER_UPDATE if update != "none" else AccessMode.PROTECTED_CURRENT
    block = Block(arg_types=[CapabilityType(storage.value.type, StringAttr(mode))])
    payload = RetainedType(PayloadType(StringAttr("view" if kind == "buffer" else "autograd")))
    made = call(block, "make", "make_" + kind, block.args, [payload])
    made.properties["retains"] = ArrayAttr([integer(0)])
    product = made.results[0]
    if derived:
        child = call(block, "derive", "derive_" + kind, [product], [payload])
        child.properties["retains"] = ArrayAttr([integer(0)])
        product = child.results[0]
    consumed = call(block, "consume", "consume_" + kind, [product, block.args[0]], [SCALAR])
    if update == "before":
        call(block, "replace", "replace", block.args, [])
    if derived:
        block.add_op(node(ReleaseProductOp, "release.child", inputs=[product]))
    block.add_op(node(ReleaseProductOp, "release", inputs=made.results))
    if update == "after":
        call(block, "replace", "replace", block.args, [])
    end(block, "use", consumed.results[0])
    use = access("use", [storage.value], block, mode, [SCALAR])
    write = Block(arg_types=[CapabilityType(output.value.type, StringAttr(AccessMode.OWNER_UPDATE)), SCALAR])
    published = call(write, "publish", "publish", write.args, [SCALAR])
    end(write, "publish.use", published.results[0])
    publish = access("publish.use", [output.value, use.results[0]], write, AccessMode.OWNER_UPDATE, [SCALAR])
    root = ProgramOp.create(
        properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([storage, output, use, publish]))]
    )
    return ModuleOp([root])
