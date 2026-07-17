"""Central builders for model-realization metadata facts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from library.metadata.dataclasses.model import ModelRealizationFacts, RealizedModelComponentFacts
from library.metadata.validation import validate_metadata_items


class _LoadedModelComponentView(Protocol):
    """Narrow loaded-component input required to describe a realization."""

    @property
    def key(self) -> str: ...

    @property
    def public_name(self) -> str: ...

    @property
    def module(self) -> Any: ...

    @property
    def roles(self) -> tuple[str, ...]: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class ModelRealizationState:
    """Validated realization facts retained for later identity references."""

    realization: ModelRealizationFacts
    components: tuple[RealizedModelComponentFacts, ...]

    @property
    def realization_identifier(self) -> str:
        """Return the qualified model-realization identity."""
        return self.realization.realization_identifier

    def component_identifier(self, component_key: str) -> str | None:
        """Return the qualified identity for one family-local component key."""
        component = next((item for item in self.components if item.component_key == component_key), None)
        return None if component is None else component.component_identifier

    def as_metadata_items(self) -> tuple[ModelRealizationFacts | RealizedModelComponentFacts, ...]:
        """Return realization then components in family-declared filing order."""
        return (self.realization, *self.components)


def build_model_realization_state(
    *,
    run_identifier: str,
    realization_key: str,
    family_identifier: str,
    model_version: str | None,
    loaded_components: Sequence[_LoadedModelComponentView],
) -> ModelRealizationState:
    """Build validated, module-free facts from an explicit loaded-component view."""
    realization = ModelRealizationFacts.for_run(
        run_identifier=run_identifier,
        realization_key=realization_key,
        family_identifier=family_identifier,
        model_version=model_version,
    )
    components = tuple(
        RealizedModelComponentFacts.for_realization(
            run_identifier=run_identifier,
            realization_identifier=realization.realization_identifier,
            component_key=component.key,
            public_name=component.public_name,
            declaration_order=declaration_order,
            roles=component.roles,
            capabilities=component.capabilities,
            present=component.module is not None,
        )
        for declaration_order, component in enumerate(loaded_components)
    )
    state = ModelRealizationState(realization=realization, components=components)
    validate_metadata_items(state.as_metadata_items())
    return state


__all__ = ["ModelRealizationState", "build_model_realization_state"]
