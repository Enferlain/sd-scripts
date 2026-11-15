"""Target references: component-qualified selectors, standalone participants, and no invented ancestry."""

import pytest

from library.runstate.authority import ParticipantRef
from library.runstate.targets import TargetKind, TargetRef, component_target, module_target, parameter_target

UNET = ParticipantRef("run1", 1)
WEIGHTS = ParticipantRef("run1", 2)


def test_targets_inside_a_component_use_the_component_qualified_selector() -> None:
    parameter = parameter_target(UNET, "input_blocks.0.0.weight", component="unet", component_key="denoiser")
    module = module_target(UNET, "input_blocks.0.0", component="unet", component_key="denoiser", module_type="Conv2d")
    assert parameter.selector == "unet.input_blocks.0.0.weight"
    assert module.selector == "unet.input_blocks.0.0"
    assert (parameter.component, parameter.component_key) == ("unet", "denoiser"), "the internal key stays apart from the public label"


def test_a_component_target_names_its_label_and_key() -> None:
    target = component_target(UNET, "clip_l", "text_encoder_1")
    assert (target.kind, target.selector, target.path) == (TargetKind.COMPONENT, "clip_l", "")


def test_a_standalone_participant_is_selected_through_its_address_without_a_component() -> None:
    target = parameter_target(WEIGHTS, "vector", address="loss_weights")
    assert target.selector == "@loss_weights.vector"
    assert target.component is None and target.component_key is None


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: TargetRef(WEIGHTS, TargetKind.PARAMETER, "unet.vector", "vector"), "through its participant's address"),
        (lambda: TargetRef(UNET, TargetKind.PARAMETER, "unet.x", "x", component="unet"), "both the public label and the internal key"),
        (
            lambda: TargetRef(UNET, TargetKind.PARAMETER, "clip_l.x", "x", component="unet", component_key="denoiser"),
            "selected as 'unet.x'",
        ),
        (lambda: parameter_target(WEIGHTS, "vector", component="unet", component_key="d", address="loss_weights"), "not its participant"),
        (lambda: TargetRef(UNET, TargetKind.MODULE, "unet.a", "a", "unet", "d", owner_module="a"), "only a parameter"),
    ],
)
def test_a_target_cannot_claim_ancestry_it_does_not_have(build, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()
