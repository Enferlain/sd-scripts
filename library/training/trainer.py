"""The Trainer: the host that carries an accepted run to execution (D17).

It consumes exactly an accepted run, the installed implementations, the
target it runs on with its device adapters, and the composition the run was
accepted under, which launch composes (D17, D18). Its host sequence:

1. **check** that the GIL is off and keep it off: modules known to turn it
   on stay unavailable and CPython's warning is an error (D15); then that
   every implementation the run uses is installed with the code evidence
   the run recorded (D16), and cold-load the run;
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

After materializing and after preparing, the run's loads, the host reads the
GIL state, and the runtime reads it periodically while the run works; a GIL
that turned on fails the run (D15).

The standard mechanics are Trainer optimization's, realized through its
owner protocol (:mod:`library.training.units`). Launch settings beyond the
target, products, observation and restoration come with later sections.
"""

from collections.abc import Callable, Mapping

from library.iro import gil
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
from library.training.preparation import Realization, RoutePreparation, prepare_training
from library.training.units import TRAINER, OptimizationPlan, TrainerOptimization
from library.training.units.derivation import TrainingDerivation


WATCH_GIL = 1.0
"""How often, in seconds, the runtime reads the GIL state while a run works (D15)."""


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
        keep_gil_off: bool = True,
        realizations: Mapping[str, Realization] | None = None,
    ):
        """``preparations`` prepare named routes, keyed by ``(participant, route)``; ``supplied`` binds participants declared without a materializer.

        ``realizations`` are the realizations launch installs by name, which
        the participants' ``realize`` clauses select (D15).

        ``keep_gil_off`` applies D15's rule to the run. Only a process that
        already runs with the GIL on for other reasons, such as a test
        session that loaded today's code, turns it off.
        """
        self.run = run
        self.composition = composition
        self.installed = dict(installed)
        self.target = target
        self.devices = dict(devices)
        self.preparations = dict(preparations or {})
        self.supplied = dict(supplied or {})
        self.observe = observe
        self.keep_gil_off = keep_gil_off
        self.realizations = dict(realizations or {})
        self.authority: RunAuthority | None = None
        self.optimization: TrainerOptimization | None = None

    def realize(self) -> ExecutableRun:
        """Carries the run through the host sequence to an executable run; raises :class:`Refused` with the reasons at the first step that fails."""
        run = self.run
        self._gil(lambda: gil.check("as the run starts"))  # a read only: a run refused next leaves the process as it was
        check_implementations(run, self.installed)
        loaded = cold_load(run, self.composition, installed=self.installed)
        self._gil(lambda: gil.keep_off("after checking the implementations and loading the run"))
        authority = self.authority = RunAuthority.seed(run)
        participants = [participant.name for participant in run.seed.participants]
        device = self.devices[self.target.device_work or HOST.name]
        bound = MaterializationAttempt(
            authority, participants, installed=self.installed, device=device, supplied=self.supplied, supplier="the launcher"
        ).run()
        if isinstance(bound, Rejected):
            raise Refused(_reasons("materialization", bound))
        self._gil(lambda: gil.check("after materializing the participants"))
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
        published = prepare_training(authority, optimization, routes, self.preparations, prepared.derivers, self.realizations, device)
        if isinstance(published, Rejected):
            raise Refused(_reasons("preparation", published))
        self._gil(lambda: gil.check("after preparing the routes and units"))
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
            watch_gil=WATCH_GIL if self.keep_gil_off else None,
        )

    def train(self) -> Outcome:
        """Realizes the run and executes it."""
        return self.realize().execute()

    def _gil(self, step: Callable[[], None]) -> None:
        """Applies one step of the GIL rule where the run keeps the GIL off; the GIL found on refuses the run (D15)."""
        if not self.keep_gil_off:
            return
        try:
            step()
        except gil.GilTurnedOn as on:
            raise Refused(f"the GIL rule was refused:\n  {on}") from on


def _reasons(step: str, rejected: Rejected) -> str:
    return f"{step} was refused:\n" + "\n".join(f"  {reason}" for reason in rejected.reasons)


__all__ = ["Trainer"]
