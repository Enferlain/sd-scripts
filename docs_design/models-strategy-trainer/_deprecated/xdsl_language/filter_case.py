"""Small composition fixture: drops do not discharge requested consumption."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Region

from .contracts import AccessMode
from .dialects import (
    bool_type,
    CapabilityType,
    ClaimType,
    ConstantOp,
    FilterOp,
    index_type,
    integer,
    OnceOp,
    OptionalPayloadType,
    ParallelOp,
    ProducerOp,
    ProgramOp,
    RequestOp,
    RequestsOp,
    RequestType,
    StopAfterOp,
    TakeOp,
)
from .fixtures import RECORD, SCALAR, access, body, call, end, node, props, repeat, resource, retire, stream, structured, take


def program(*, opaque: bool = False, horizon: int = 3) -> ModuleOp:
    config, products = resource("config", "filter_config"), resource("products", "filter_products")
    raw, kept = stream("raw", ["config"], horizon), stream("kept", ["config"], horizon)
    captures = [raw.value, kept.value, config.value, products.value]
    source, filtering, consumers, stop_filter, stop_source = (body(captures) for _ in range(5))

    production = body([source.args[2]], index_type())
    read = Block(arg_types=[CapabilityType(config.value.type, StringAttr(AccessMode.SNAPSHOT)), index_type()])
    record = call(read, "produce", "filter_produce", read.args, [RECORD])
    end(read, "produce.access", record.results[0])
    snapshot = access("produce.access", production.args, read, AccessMode.SNAPSHOT, [RECORD])
    production.add_op(snapshot)
    end(production, "source.body", snapshot.results[0])
    source.add_op(structured(ProducerOp, "source", [source.args[0], source.args[2]], [production]))
    end(source, "source.branch")

    selected = body([filtering.args[0]], index_type())
    claim = node(TakeOp, "filter.take", inputs=[selected.args[0]], outputs=[ClaimType(RECORD)])
    selected.add_op(claim)
    predicate = Block(arg_types=[RECORD])
    keep = call(predicate, "predicate", "filter_even", predicate.args, [bool_type()])
    end(predicate, "filter.predicate", keep.results[0])
    selection = structured(FilterOp, "filter", claim.results, [predicate], outputs=[OptionalPayloadType(RECORD)])
    selected.add_op(selection)
    end(selected, "filter.body", selection.results[0])
    filtering.add_op(structured(ProducerOp, "filtering", [filtering.args[1], filtering.args[0]], [selected]))
    end(filtering, "filtering.branch")

    initial, handler = (body([consumers.args[1], consumers.args[3]]) for _ in range(2))

    def consume(block: Block, name: str) -> None:
        owned, record = take(block, name + ".take", block.args[0])
        local = body([record])
        value = call(local, name + ".extract", "filter_extract", local.args, [SCALAR]).results[0]
        value = call(local, name + ".offset", "filter_offset", [value], [SCALAR]).results[0]
        if opaque:
            value = call(local, name + ".opaque", "filter_opaque", [value], [SCALAR]).results[0]
        value = call(local, name + ".double", "filter_double", [value], [SCALAR]).results[0]
        end(local, name + ".local", value)
        recipe = structured(OnceOp, name + ".local", [record], [local], outputs=[SCALAR])
        block.add_op(recipe)
        write = Block(arg_types=[CapabilityType(products.value.type, StringAttr(AccessMode.OWNER_UPDATE)), RECORD, SCALAR])
        result = call(write, name + ".publish", "filter_publish", write.args, [SCALAR])
        end(write, name + ".write", result.results[0])
        block.add_op(access(name + ".write", [block.args[1], record, recipe.results[0]], write, AccessMode.OWNER_UPDATE, [SCALAR]))
        retire(block, name + ".retire", owned)

    first = body(initial.args)
    consume(first, "first")
    two = node(ConstantOp, "two", outputs=[SCALAR], literal=StringAttr("2"))
    first.add_op(two)
    first.add_op(
        node(
            RequestOp,
            "request",
            inputs=two.results,
            handler=SymbolRefAttr("more"),
            source=SymbolRefAttr("initial"),
            boundary=StringAttr("settled_success"),
        )
    )
    end(first, "initial.body")
    initial.add_op(structured(OnceOp, "initial", initial.args, [first]))
    end(initial, "initial.branch")

    requested = body(handler.args, RequestType(SCALAR))
    iteration = body(requested.args[:2], index_type())
    consume(iteration, "next")
    end(iteration, "requested.iteration")
    requested.add_op(repeat("requested", requested.args[:2], iteration, 2))
    end(requested, "more.body")
    handler.add_op(
        structured(
            RequestsOp,
            "more",
            handler.args,
            [requested],
            sources=ArrayAttr([SymbolRefAttr("initial")]),
            request_schema=SCALAR,
            max_outstanding=integer(1),
            max_active=integer(1),
        )
    )
    end(handler, "more.branch")
    consumers.add_op(structured(ParallelOp, "consumers", [consumers.args[1], consumers.args[3]], [initial, handler]))
    end(consumers, "consumer.branch")

    stop_filter.add_op(node(StopAfterOp, "stop.filter", source=SymbolRefAttr("consumers"), target=SymbolRefAttr("filtering")))
    end(stop_filter, "stop.filter.branch")
    stop_source.add_op(node(StopAfterOp, "stop.source", source=SymbolRefAttr("filtering"), target=SymbolRefAttr("source")))
    end(stop_source, "stop.source.branch")
    activities = structured(ParallelOp, "activities", captures, [source, filtering, consumers, stop_filter, stop_source])
    root = ProgramOp.create(
        properties={**props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([config, products, raw, kept, activities]))]
    )
    return ModuleOp([root])
