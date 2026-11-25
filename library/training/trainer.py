"""The Trainer: the host that carries an accepted run to execution (D17).

It consumes exactly an accepted run, the installed implementations, the
target it runs on with its device adapters, and the composition the run was
accepted under, which launch composes (D17, D18). Its host sequence:

1. **check** that every implementation the run uses is installed with the
   code evidence the run recorded (D16), and cold-load the run;
2. **seed** the run authority from the strategy's declarations;
3. **materialize** every declared participant as one governed job;
4. **plan**: Trainer optimization derives its units from the strategy;
5. **prepare**: the IR prepares the strategy for the target, with training's
   derivation of grad mode and the units each call serves (D16), and one
   training preparation attempt prepares the routes the run's leases need
   and realizes the units, publishing both together;
6. **check readiness**: the run authority and Trainer optimization each
   judge the obligations on what they hold at core execution;
7. **execute** the executable run, once.

The standard mechanics are Trainer optimization's, realized through its
owner protocol (:mod:`library.training.units`). Launch settings beyond the
target, products, observation and restoration come with later sections.
"""

from collections.abc import Mapping

from library.iro.accepted import AcceptedRun, check_implementations, cold_load
from library.iro.composition import Composition
from library.iro.prepare import HOST, Target, prepare
from library.iro.runtime import Observer, Outcome
from library.iro.seams import Device, Implementation, Owner
from library.iro.verify import ObligationKind
from library.runstate.authority import AUTHORITY, NORMAL, MaterializationAttempt, Rejected, RunAuthority
from library.runstate.generation import ExecutableRun
from library.runstate.owners import Refused
from library.runstate.projections import Execution
from library.training.preparation import RoutePreparation, prepare_training
from library.training.units import TRAINER, OptimizationPlan, TrainerOptimization
from library.training.units.derivation import TrainingDerivation


class Trainer:
    """Realizes one accepted run on one target and executes it."""

    def __init__(
        self,
        run: AcceptedRun,
        composition: Composition,
        installed: Mapping[str, Implementation],
        target: Target,
        devices: Mapping[str, Device],
        *,
        preparations: Mapping[tuple[str, str], RoutePreparation] | None = None,
        supplied: Mapping[str, object] | None = None,
        observe: Observer | None = None,
    ):
        """``preparations`` prepare named routes, keyed by ``(participant, route)``; ``supplied`` binds participants declared without a materializer."""
        self.run = run
        self.composition = composition
        self.installed = dict(installed)
        self.target = target
        self.devices = dict(devices)
        self.preparations = dict(preparations or {})
        self.supplied = dict(supplied or {})
        self.observe = observe
        self.authority: RunAuthority | None = None
        self.optimization: TrainerOptimization | None = None

    def realize(self) -> ExecutableRun:
        """Carries the run through the host sequence to an executable run; raises :class:`Refused` with the reasons at the first step that fails."""
        run = self.run
        check_implementations(run, self.installed)
        loaded = cold_load(run, self.composition, installed=self.installed)
        authority = self.authority = RunAuthority.seed(run)
        participants = [participant.name for participant in run.seed.participants]
        device = self.devices[self.target.device_work or HOST.name]
        bound = MaterializationAttempt(
            authority, participants, installed=self.installed, device=device, supplied=self.supplied, supplier="the launcher"
        ).run()
        if isinstance(bound, Rejected):
            raise Refused(_reasons("materialization", bound))
        plan = OptimizationPlan.derive(loaded.strategy, authority.incarnation)
        optimization = self.optimization = TrainerOptimization(authority.generations, plan, run.obligations)
        addresses = {authority.incarnation(name): name for name in participants}
        trains = {name: frozenset(addresses[m.participant] for m in unit.members) for name, unit in plan.units.items()}
        prepared = prepare(loaded, self.installed, self.target, [TrainingDerivation(trains, frozenset(participants))])
        routes = sorted(
            {
                (o.use.resource, o.use.route or NORMAL)
                for o in run.obligations
                if o.kind == ObligationKind.LEASE and o.use is not None and o.use.resource in participants and o.use.view is None
            }
        )
        published = prepare_training(authority, optimization, routes, self.preparations, prepared.derivers)
        if isinstance(published, Rejected):
            raise Refused(_reasons("preparation", published))
        execution = Execution(authority)
        owners: dict[str, Owner] = {**dict.fromkeys(participants, execution), **dict.fromkeys(plan.units, optimization)}
        section = optimization.section
        attempts = {route.evidence.attempt for p in authority.section.participants.values() for route in p.routes.values()}
        attempts |= {unit.attempt for unit in section.units.values()} if section is not None else set()
        return ExecutableRun(
            authority.generations,
            prepared,
            owners,
            self.installed,
            self.devices,
            {AUTHORITY: authority, TRAINER: optimization},
            attempts=tuple(sorted(attempts, key=str)),
            observe=self.observe,
        )

    def train(self) -> Outcome:
        """Realizes the run and executes it."""
        return self.realize().execute()


def _reasons(step: str, rejected: Rejected) -> str:
    return f"{step} was refused:\n" + "\n".join(f"  {reason}" for reason in rejected.reasons)


__all__ = ["Trainer"]
