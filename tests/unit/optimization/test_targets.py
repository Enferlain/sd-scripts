import torch

from library.optimization.targets import (
    build_component_target_ref,
    build_module_target_ref,
    build_parameter_target_ref,
    resolve_parameter_owner_modules,
)


def test_build_component_target_ref_uses_public_component_selector():
    component = object()

    target_ref = build_component_target_ref(
        component="unet",
        component_key="denoiser",
        obj=component,
    )

    assert target_ref.kind == "component"
    assert target_ref.path == ""
    assert target_ref.selector == "unet"
    assert target_ref.obj is component


def test_build_module_target_ref_records_module_type_and_selector():
    module = torch.nn.Linear(4, 4)

    target_ref = build_module_target_ref(
        component="clip_l",
        component_key="text_encoder1",
        path="proj",
        module=module,
    )

    assert target_ref.kind == "module"
    assert target_ref.path == "proj"
    assert target_ref.selector == "clip_l.proj"
    assert target_ref.module_type == "Linear"
    assert target_ref.obj is module


def test_resolve_parameter_owner_modules_prefers_direct_module_owner():
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))
            self.block = torch.nn.Linear(4, 4)

    module = DummyModule()
    owners = resolve_parameter_owner_modules(module)

    assert owners[id(module.scale)] == ("", "DummyModule")
    assert owners[id(module.block.weight)] == ("block", "Linear")
    assert owners[id(module.block.bias)] == ("block", "Linear")


def test_build_parameter_target_ref_records_owner_module_provenance():
    module = torch.nn.Linear(4, 4)

    target_ref = build_parameter_target_ref(
        component="unet",
        component_key="denoiser",
        path="to_q.weight",
        parameter=module.weight,
        owner_module_path="to_q",
        owner_module_type="Linear",
    )

    assert target_ref.kind == "parameter"
    assert target_ref.selector == "unet.to_q.weight"
    assert target_ref.owner_module_path == "to_q"
    assert target_ref.owner_module_type == "Linear"
    assert target_ref.obj is module.weight
