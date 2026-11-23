"""Training's preparation coordination: one attempt that prepares the routes and realizes the units (D8, D19, D20).

The attempt requests every route the strategy's lease obligations need,
contributes Trainer optimization's next section, and publishes both
together or nothing. Inside it, off to the side:

- the units are resolved against the participants the attempt pinned;
- trainability is set from unit membership, in place: a parameter a unit
  resolved to requires gradients and every other parameter of the
  participants does not, which stays fixed for the run (D20);
- each route is prepared from its participant's bound object, by the
  route's preparation where one is given, such as model code turning on
  gradient checkpointing, and otherwise as the bound object itself; and
  each carries a held setup that sets its participant's module mode for
  every call through it, from the units that train it (D16, D20);
- Trainer optimization builds each unit's optimizer on its final
  parameters.

The first pass prepares before any route is published, where in-place work
has nothing to withdraw (D19). Placement, casting and wrapping, backend
groups and the backend coordination they belong to come with the second
pass of section 16 and with section 20.
"""

from collections.abc import Callable, Mapping, Sequence

import torch

from library.runstate.authority import NORMAL, ParticipantRef, Published, Rejected, RunAuthority
from library.runstate.authority.records import SetupPin
from library.runstate.owners import Mutation
from library.runstate.preparation import PreparationAttempt, PreparationRequest, PreparationResult, PreparedRoute
from library.training.units import TRAINER, ModuleModes, TrainerOptimization, resolve

RoutePreparation = Callable[[object], object]
"""How one route is prepared from its participant's bound object: it may change the object in place and returns the route's callable."""


def prepare_training(
    authority: RunAuthority,
    optimization: TrainerOptimization,
    routes: Sequence[tuple[str, str]],
    preparations: Mapping[tuple[str, str], RoutePreparation] | None = None,
    pinned: tuple[SetupPin, ...] = (),
) -> Published | Rejected:
    """Prepares the routes, given as ``(participant, route)``, and every unit of the plan, in one in-place attempt.

    ``pinned`` are setups the run already pins, such as the prepared form's; a
    route setup claiming one of their names and versions with other code is
    refused (D16).
    """
    preparations = preparations or {}
    unprepared = [f"@{address}'s {route} route" for address, route in routes if route != NORMAL and (address, route) not in preparations]
    if unprepared:
        return Rejected(tuple(f"{name} has no preparation; only a normal route is prepared as its bound object" for name in unprepared))
    requested = tuple((authority.incarnation(address), route) for address, route in routes)
    trained = {target.participant for definition in optimization.plan.units.values() for target in definition.members}
    request = PreparationRequest(
        routes=requested,
        participants=tuple(sorted(trained - {reference for reference, _ in requested}, key=str)),
        mutation=Mutation.IN_PLACE,
        contributors=(TRAINER,),
        pinned=pinned,
    )
    attempt = PreparationAttempt.derive(authority, request)
    if isinstance(attempt, Rejected):
        return attempt
    try:
        return _prepare(attempt, optimization, requested, preparations)
    except BaseException:
        authority.end_in_place(attempt.attempt.reference)  # the attempt will never publish
        raise


def _prepare(
    attempt: PreparationAttempt,
    optimization: TrainerOptimization,
    requested: tuple[tuple[ParticipantRef, str], ...],
    preparations: Mapping[tuple[str, str], RoutePreparation],
) -> Published | Rejected:
    request = attempt.request
    reference = attempt.attempt.reference
    candidates = resolve(optimization.plan, attempt.projection)
    if candidates and isinstance(candidates[0], str):
        return attempt.publish(PreparationResult(reference, failures=tuple(str(reason) for reason in candidates)))
    resolved = [candidate for candidate in candidates if not isinstance(candidate, str)]
    learning = {id(parameter) for candidate in resolved for parameter in candidate.parameters}
    for participant in request.required:
        binding = attempt.projection[participant].binding
        assert binding is not None, "the attempt is derived only over bound participants"
        if isinstance(binding.value, torch.nn.Module):
            for parameter in binding.value.parameters():
                parameter.requires_grad_(id(parameter) in learning)
    prepared = []
    failures = []
    for participant, route in requested:
        record = attempt.projection[participant]
        assert record.binding is not None
        preparation = preparations.get((record.address, route))
        # Every unit the plan says trains the participant, as training's derivation declares them, with the modules it resolved to.
        by_unit = {candidate.definition.name: candidate for candidate in resolved}
        modes = ModuleModes(
            {
                name: by_unit[name].modules.get(record.address, ()) if name in by_unit else ()
                for name, definition in optimization.plan.units.items()
                if any(target.participant == participant for target in definition.members)
            }
        )
        try:
            callable_ = preparation(record.binding.value) if preparation else record.binding.value
            prepared.append(PreparedRoute(participant, route, callable_, modes))
        except Exception as failed:
            failures.append(f"@{record.address}'s {route} route could not be prepared: {failed}")
    if failures:
        return attempt.publish(PreparationResult(reference, failures=tuple(failures)))
    section = optimization.prepare(resolved, reference)
    if isinstance(section, tuple):
        return attempt.publish(PreparationResult(reference, failures=section))
    return attempt.publish(PreparationResult(reference, tuple(prepared), (section,)))


__all__ = ["RoutePreparation", "prepare_training"]
