"""Target references: references into a participant's structure (D19, D12).

A target reference names a participant and a selected component, module or
parameter within it, with its public selector and any declared component
provenance. It holds no live object: modules and parameters are reached only
through revision-pinned projections, and resolving a target against loaded
structure, with its source revisions, belongs to the consumer that resolves
it. Optimization, adapter targeting and persistence share these references,
and sharing them makes none of them the owner of targeting
(``optimization-target-refs``).

Selectors (``component-qualified-selectors``):

- With genuine component provenance, a selector is the public component
  label followed by the component-local path, such as
  ``unet.input_blocks.0.0.weight``; the internal component key stays a
  separate field.
- A participant's own structure without component provenance, such as a
  standalone learned loss-weight vector, is selected through its authored
  address, ``@loss_weights.vector``, so it never impersonates a component.
"""

from dataclasses import dataclass
from enum import StrEnum

from library.runstate.authority.records import ParticipantRef


class TargetKind(StrEnum):
    COMPONENT = "component"
    MODULE = "module"
    PARAMETER = "parameter"


@dataclass(frozen=True)
class TargetRef:
    """A reference into one participant's structure, holding no live object.

    ``path`` is component-local with component provenance, and
    participant-local without it; it is empty for a whole component.
    ``component`` is the public component label and ``component_key`` the
    internal key, both present exactly when the target has genuine component
    provenance. ``catalog_path`` and ``structural_path`` are optional
    associations from the model metadata identity work and never replace the
    semantic identity.
    """

    participant: ParticipantRef
    kind: TargetKind
    selector: str
    path: str = ""
    component: str | None = None
    component_key: str | None = None
    module_type: str | None = None
    owner_module: str | None = None
    owner_module_type: str | None = None
    catalog_path: str | None = None
    structural_path: str | None = None

    def __post_init__(self) -> None:
        if (self.component is None) != (self.component_key is None):
            raise ValueError("component provenance names both the public label and the internal key, or neither")
        if self.kind == TargetKind.COMPONENT:
            if self.component is None or self.path:
                raise ValueError("a component target names its component and no path within it")
        elif not self.path:
            raise ValueError(f"a {self.kind} target names its path")
        if self.kind != TargetKind.PARAMETER and (self.owner_module or self.owner_module_type):
            raise ValueError("only a parameter target names its owning module")
        if self.component is not None:
            expected = f"{self.component}.{self.path}" if self.path else self.component
            if self.selector != expected:
                raise ValueError(f"a target in component {self.component} is selected as {expected!r}, not {self.selector!r}")
        elif not self.selector.startswith("@") or not self.selector.endswith(f".{self.path}"):
            raise ValueError(f"a target without component provenance is selected through its participant's address, not {self.selector!r}")


def component_target(participant: ParticipantRef, component: str, component_key: str) -> TargetRef:
    """A top-level model component, from the family's declaration and the participant it belongs to."""
    return TargetRef(participant, TargetKind.COMPONENT, component, component=component, component_key=component_key)


def module_target(
    participant: ParticipantRef,
    path: str,
    *,
    component: str | None = None,
    component_key: str | None = None,
    address: str | None = None,
    module_type: str | None = None,
) -> TargetRef:
    """A module beneath a declared component, or, with ``address`` and no component, in a participant's own structure."""
    return TargetRef(
        participant,
        TargetKind.MODULE,
        _selector(path, component, address),
        path,
        component,
        component_key,
        module_type=module_type,
    )


def parameter_target(
    participant: ParticipantRef,
    path: str,
    *,
    component: str | None = None,
    component_key: str | None = None,
    address: str | None = None,
    owner_module: str | None = None,
    owner_module_type: str | None = None,
) -> TargetRef:
    """A parameter beneath a declared component, or, with ``address`` and no component, in a participant's own structure."""
    return TargetRef(
        participant,
        TargetKind.PARAMETER,
        _selector(path, component, address),
        path,
        component,
        component_key,
        owner_module=owner_module,
        owner_module_type=owner_module_type,
    )


def _selector(path: str, component: str | None, address: str | None) -> str:
    if component is not None:
        if address is not None:
            raise ValueError("a target with component provenance is selected through its component, not its participant's address")
        return f"{component}.{path}"
    if address is None:
        raise ValueError("a target without component provenance is selected through its participant's address")
    return f"@{address}.{path}"


__all__ = ["TargetKind", "TargetRef", "component_target", "module_target", "parameter_target"]
