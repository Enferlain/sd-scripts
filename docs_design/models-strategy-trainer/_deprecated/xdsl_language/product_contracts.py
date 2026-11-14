"""Installed product fixture contracts and inputs, usable without authoring."""

from __future__ import annotations

from .prepare import BindingSpec, Manifest, ResourceSpec


def manifest(kind: str = "buffer", *, derived: bool = False) -> Manifest:
    payload = "view" if kind == "buffer" else "autograd"
    retained = "retained:" + payload
    bindings = [
        BindingSpec("make_" + kind, ("capability",), (retained,), reads=(0,), unknown_effects=False, retains=(0,)),
        BindingSpec("consume_" + kind, (retained, "capability"), ("scalar",), reads=(1,), unknown_effects=False),
        BindingSpec("publish", ("capability", "scalar"), ("scalar",), updates=(0,), reads=(), unknown_effects=False),
        BindingSpec("replace", ("capability",), (), updates=(0,), reads=(), unknown_effects=False),
    ]
    if derived:
        bindings.append(BindingSpec("derive_" + kind, (retained,), (retained,), reads=(), unknown_effects=False, retains=(0,)))
    return Manifest(tuple(bindings), (ResourceSpec(kind, "state"), ResourceSpec("product_result", "state")), (payload,))


def initial(kind: str = "buffer") -> dict[str, object]:
    return {"storage": bytearray(b"abcdef") if kind == "buffer" else (2.0, 3.0, 4.0), "result": 0.0}
