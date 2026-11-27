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
  every call through it, from the units that train it (D16, D20), and a
  call setup for each realization its participant declares, such as mixed
  precision, built by the realization installed under that name with the
  declared settings for the device the calls run on (D15). The setups are
  built once per participant, its routes sharing them, for every
  participant that declares one, before anything changes in place, so a
  realization that cannot be built refuses an attempt that prepared nothing;
- Trainer optimization builds each unit's optimizer on its final
  parameters.

The first pass prepares before any route is published, where in-place work
has nothing to withdraw (D19). Placement, casting and wrapping, backend
groups and the backend coordination they belong to come with the second
pass of section 16 and with section 20.
"""

from collections.abc import Callable, Mapping, Sequence

import torch

from library.iro.evidence import CodeEvidence
from library.iro.seams import CallSetup
from library.runstate.authority import NORMAL, ParticipantRef, Published, Rejected, RunAuthority
from library.runstate.owners import Mutation
from library.runstate.preparation import PreparationAttempt, PreparationRequest, PreparationResult, PreparedRoute
from library.training.units import TRAINER, ModuleModes, TrainerOptimization, resolve

RoutePreparation = Callable[[object], object]
"""How one route is prepared from its participant's bound object: it may change the object in place and returns the route's callable."""

Realization = Callable[..., CallSetup]
"""Builds a realization's call setup for the run's device-work adapter, where the participants are materialized, from the
settings the participant declares it with: ``realize(@mixed_precision, "bfloat16")`` builds ``mixed_precision(device, "bfloat16")``.
An activity the target places on another device is not built for yet (bead sd-scripts-syv.83)."""


def prepare_training(
    authority: RunAuthority,
    optimization: TrainerOptimization,
    routes: Sequence[tuple[str, str]],
    preparations: Mapping[tuple[str, str], RoutePreparation] | None = None,
    recorded: tuple[CodeEvidence, ...] = (),
    realizations: Mapping[str, Realization] | None = None,
    device: object = None,
) -> Published | Rejected:
    """Prepares the routes, given as ``(participant, route)``, and every unit of the plan, in one in-place attempt.

    ``recorded`` are setups the run already records, such as the prepared
    form's; a route setup claiming one of their names with other code is
    refused (D16). ``device`` is the device adapter the participants'
    calls run on, which each realization is built for.
    """
    preparations = preparations or {}
    setups = _realized(authority, realizations or {}, device)  # before anything changes in place, for every participant that declares one
    if isinstance(setups, Rejected):
        return setups
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
        recorded=recorded,
    )
    attempt = PreparationAttempt.derive(authority, request)
    if isinstance(attempt, Rejected):
        return attempt
    try:
        return _prepare(attempt, optimization, requested, preparations, setups)
    except BaseException:
        authority.end_in_place(attempt.attempt.reference)  # the attempt will never publish
        raise


def _realized(
    authority: RunAuthority, realizations: Mapping[str, Realization], device: object
) -> dict[str, tuple[CallSetup, ...]] | Rejected:
    """Each participant's call setups, built once from the realizations it declares and shared by its routes, or why one cannot be."""
    setups: dict[str, tuple[CallSetup, ...]] = {}
    failures = []
    for participant in sorted(authority.section.participants.values(), key=lambda p: p.address):
        built = []
        for name, settings in participant.declaration.realizations:
            realization = realizations.get(name)
            if realization is None:
                failures.append(f"@{participant.address} is realized by @{name}, and no realization is installed under that name")
                continue
            try:
                built.append(realization(device, *settings))
            except Exception as failed:
                failures.append(f"@{participant.address}'s realization @{name} could not be built from {settings}: {failed}")
        if built:
            setups[participant.address] = tuple(built)
    return Rejected(tuple(failures)) if failures else setups


def _prepare(
    attempt: PreparationAttempt,
    optimization: TrainerOptimization,
    requested: tuple[tuple[ParticipantRef, str], ...],
    preparations: Mapping[tuple[str, str], RoutePreparation],
    setups: Mapping[str, tuple[CallSetup, ...]],
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
            prepared.append(PreparedRoute(participant, route, callable_, modes, setups.get(record.address, ())))
        except Exception as failed:
            failures.append(f"@{record.address}'s {route} route could not be prepared: {failed}")
    if failures:
        return attempt.publish(PreparationResult(reference, failures=tuple(failures)))
    section = optimization.prepare(resolved, reference)
    if isinstance(section, tuple):
        return attempt.publish(PreparationResult(reference, failures=section))
    return attempt.publish(PreparationResult(reference, tuple(prepared), (section,)))


__all__ = ["Realization", "RoutePreparation", "prepare_training"]
