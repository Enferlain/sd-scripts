"""Lower a checked whole-run description into directly callable machinery.

Only preparation dispatches on IR constructs. Engine.run invokes the derived
root. Blocks become direct Python call chains; repeated and independent
scopes become selected-policy loops and TaskGroup children. Their request
connections are resolved once. This is one concrete CPU/asyncio target.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, NoReturn, cast

import torch

from .access import block_access, check_waits, mode_participants, prefix_joins, retained_exclusions, sequence_uses
from .graph import (
    Accepted,
    Advance,
    Block,
    Call,
    Choice,
    Completion,
    Differentiate,
    Join,
    NotReady,
    OutOfBounds,
    Producer,
    Repeat,
    Retain,
    Release,
    Replace,
    Requested,
    Scope,
    Sequence,
    Together,
    TwoPass,
    Use,
    Work,
    inputs,
    ordered,
    outputs,
    sequence_order,
)
from .state import (
    Image,
    PreparedView,
    Product,
    Projection,
    Replacement,
    RetainedState,
    RunState,
    Stamp,
    UnitOutcome,
    join,
    wait_for_products,
)
from .workers import NotEntered, PhysicalFailed, Workers


@dataclass(frozen=True)
class Context:
    """Accepted runtime seam, not a strategy or unrestricted Trainer."""

    models: Mapping[str, torch.nn.Module]
    states: Mapping[str, object]
    choice: Choice
    invocation: int
    stamps: Mapping[str, Stamp]
    views: Mapping[str, PreparedView] = field(default_factory=dict)
    stop_requested: Callable[[], bool] | None = None


@dataclass
class Frame:
    state: RunState
    scope: str
    choice: Choice
    projection: Projection
    invocation: int
    implementation_returned: bool = False
    region: object | None = None

    def current(self) -> Projection:
        # The lease keeps bindings stable, not numerical state frozen across
        # this block's own accepted writes/optimizer calls.
        return Projection(
            self.projection.models,
            MappingProxyType({name: self.state.stamp(name) for name in self.projection.models}),
        )

    def context(self, call: Call, retained: RetainedState | None = None) -> Context:
        if retained is not None:
            self.state.check_retained(retained, self.region)
            projection, views = retained.projection, retained.views
        else:
            assert self.state.image is not None
            projection, views = self.current(), self.state.image.views
        return Context(
            MappingProxyType({use.participant: projection.models[use.participant] for use in call.uses if use.view is None}),
            MappingProxyType({name: self.state.owners[name] for name in call.states}),
            self.choice,
            self.invocation,
            MappingProxyType({use.participant: projection.stamps[use.participant] for use in call.uses}),
            MappingProxyType({use.view: views[use.view] for use in call.uses if use.view is not None}),
        )


class WorkFailed(RuntimeError):
    """A reached invocation, including invalid output after possible effects."""

    def __init__(self, source: str, frame: Frame, cause: BaseException, cleanup_failures: tuple[BaseException, ...] = ()):
        super().__init__(f"{source}: post-invocation failure ({cause})")
        self.source, self.scope, self.invocation, self.cause = source, frame.scope, frame.invocation, cause
        self.cleanup_failures = cleanup_failures
        self.cancellation: asyncio.CancelledError | None = None
        self.implementation_returned = frame.implementation_returned
        assert frame.state.image is not None
        self.unit_outcomes: tuple[UnitOutcome, ...] = tuple(
            outcome for runtime in frame.state.image.units.values() for outcome in runtime.evidence(frame.invocation, frame.region)
        )


class CallFailed(RuntimeError):
    """Internal separation of selected execution from mode restoration."""

    def __init__(
        self,
        primary: BaseException | None,
        cleanup: tuple[BaseException, ...],
        cancellation: asyncio.CancelledError | None = None,
    ):
        super().__init__("Selected execution and/or evaluation cleanup failed")
        self.primary, self.cleanup = primary, cleanup
        self.cancellation = cancellation


@dataclass(frozen=True)
class CallReturn:
    """Private physical-call evidence; never a universal run result."""

    value: object
    returned: bool
    primary: BaseException | None
    cleanup: tuple[BaseException, ...]


class ProducerFailed(RuntimeError):
    """Owner-specific evidence, not a universal completion/result object."""

    def __init__(self, source: str, primary: BaseException | None, cleanup: tuple[BaseException, ...], returned: bool):
        super().__init__(f"{source}: producer execution/cleanup failed")
        self.source, self.cause, self.cleanup_failures, self.implementation_returned = source, primary, cleanup, returned


class GrantedAccess:
    """This CPU two-pass grant has no final-advancement method.

    It owns backward, temporary parameter editing and intermediate resets.
    The standard owner retains clipping, final optimizer call and final reset.
    This interface limits available services; it is not a Python sandbox.
    """

    def __init__(self, state: RunState, unit: str, invocation: int):
        assert state.image is not None
        state.begin_contribution((unit,), invocation, granted=True)
        self.invocation = invocation
        self.parameters = state.image.units[unit].parameters
        self.original = tuple(parameter.detach().clone() for parameter in self.parameters)
        self.phases: list[str] = []
        self.restored = False

    def backward(self, loss: torch.Tensor) -> None:
        if self.phases not in ([], ["backward", "temporary_edit", "intermediate_reset"]):
            raise NotReady("Unsupported granted backward phase")
        gradients = torch.autograd.grad(loss, self.parameters)
        for parameter, gradient in zip(self.parameters, gradients, strict=True):
            parameter.grad = gradient
        self.phases.append("backward")

    def perturb(self, amount: float) -> None:
        if self.phases != ["backward"]:
            raise NotReady("Unsupported two-pass phase order")
        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is None:
                    raise NotReady("No first-pass gradient")
                parameter.add_(amount * parameter.grad)
        self.phases.append("temporary_edit")

    def reset(self) -> None:
        if self.phases != ["backward", "temporary_edit"]:
            raise NotReady("Unsupported intermediate reset")
        for parameter in self.parameters:
            parameter.grad = None
        self.phases.append("intermediate_reset")

    def restore(self) -> None:
        if self.phases != ["backward", "temporary_edit", "intermediate_reset", "backward"]:
            raise NotReady("Unsupported restoration phase")
        with torch.no_grad():
            for parameter, original in zip(self.parameters, self.original, strict=True):
                parameter.copy_(original)
        self.restored = True
        self.phases.append("restore")

    def _check_handback(self) -> None:
        if self.phases != ["backward", "temporary_edit", "intermediate_reset", "backward", "restore"]:
            raise NotReady("Incomplete two-pass handback")
        if not self.restored or any(
            not torch.equal(parameter.detach(), original) for parameter, original in zip(self.parameters, self.original, strict=True)
        ):
            raise NotReady("Current view was not restored")
        if any(parameter.grad is None for parameter in self.parameters):
            raise NotReady("No final gradient to hand back")


class ProviderContext:
    """A provider owns private workers; coordination owns its handoff ports."""

    def __init__(self, source: Producer, state: RunState, owners: Mapping[str, object]):
        self.source = source
        self._state = state
        self.states = owners
        self.ports = MappingProxyType({name: state.feeds[name] for name in source.channels})

    @asynccontextmanager
    async def use_source(self, uses: tuple[Use, ...] | None = None):
        async with self._state.use_source(self.source.uses, uses) as projection:
            yield projection

    async def publish(self, channel: str, product: Product) -> None:
        if channel not in self.ports:
            raise OutOfBounds("Provider published outside its accepted handoff ports")
        await self.ports[channel].publish(product)
        self._state.trace.append(("produced", (self.source.name, product.key, product.provenance)))


@dataclass(frozen=True)
class Ready:
    state: RunState
    root: Callable[[], Awaitable[None]]
    source: Mapping[str, str]
    access: Literal["general", "prepared"] = "general"


@dataclass(frozen=True)
class Preparation:
    state: RunState
    expected: int
    image: Image
    root: Callable[[], Awaitable[None]]
    source: Mapping[str, str]
    access: Literal["general", "prepared"] = "general"

    def publish(self) -> Ready:
        self.state.install(self.image, self.expected)
        return Ready(self.state, self.root, self.source, access=self.access)


class Engine:
    """The same entrypoint for every arrangement, with no algorithm branches."""

    def __init__(self, ready: Ready):
        self.ready = ready
        self.failure: BaseException | None = None

    async def run(self) -> None:
        state = self.ready.state
        if state.image is None or state.running:
            raise NotReady("No current image, or this execution session is already active")
        state.running = True
        state.completed = False
        try:
            await self.ready.root()
            state.completed = True
        except BaseException as error:
            self.failure = error
            raise
        finally:
            state.running = False


def _choice(value: object, alternatives, description: str) -> Choice | None:
    if value is None:
        return None
    if not isinstance(value, Choice) or value.block not in alternatives:
        raise OutOfBounds(description)
    try:
        hash(value.key)
    except TypeError as error:
        raise OutOfBounds("Runtime input identity must be hashable") from error
    return value


def prepare(
    accepted: Accepted,
    state: RunState | None = None,
    *,
    lowering: Literal["calls", "inline"] = "calls",
    access: Literal["general", "prepared"] = "general",
    worker_calls: tuple[str, ...] = (),
    worker_limit: int = 1,
) -> Preparation:
    """One unpublished attempt: load, verify, prepare units, lower whole run."""
    if lowering not in ("calls", "inline"):
        raise NotReady("Unknown candidate CPU lowering")
    if access not in ("general", "prepared"):
        raise NotReady("Unknown candidate access realization")
    workers: Workers | None = Workers(worker_limit)
    if not worker_calls:
        workers = None
    if workers is not None:
        _check_workers(accepted, frozenset(worker_calls))
    physical_modes = mode_participants(accepted, selected_workers=workers is not None)
    check_waits(accepted, physical_modes=physical_modes)
    state = state or RunState(accepted)
    if state.accepted is not accepted:
        raise NotReady("Preparation belongs to another accepted authority")
    if state.image is not None:
        raise NotReady("Use the explicit replacement path for an already installed run")
    expected = state.generation
    models = {name: declaration.load() for name, declaration in accepted.participants.items()}
    image = state.candidate(models)
    lowerer = Lowering(
        accepted,
        state,
        inline_blocks=lowering == "inline",
        prepared_access=access == "prepared",
        workers=workers,
        worker_calls=frozenset(worker_calls),
        physical_modes=physical_modes,
    )
    root = lowerer.scope(accepted.root)
    if workers is not None:
        body = root

        async def placed():
            with workers.lifetime():
                await body()

        root = placed
    return Preparation(state, expected, image, root, MappingProxyType(lowerer.source), access=access)


def _check_workers(accepted: Accepted, names: frozenset[str]) -> None:
    found: set[str] = set()

    def visit(scope: Scope):
        if isinstance(scope, Sequence):
            for block in scope.blocks:
                visit(block)
        elif isinstance(scope, Together):
            for child in scope.children:
                visit(child)
        elif isinstance(scope, Repeat):
            for block in scope.blocks:
                visit(block)
        elif isinstance(scope, Requested) and scope.enabled:
            visit(scope.block)
        elif isinstance(scope, Block):
            for item in scope.work:
                if item.name not in names:
                    continue
                if not isinstance(item, Call):
                    raise NotReady("Worker placement is established only for synchronous calls")
                function = item.implementation
                implementation_call = type(function).__call__
                if any(
                    inspect.iscoroutinefunction(value) or inspect.isasyncgenfunction(value) for value in (function, implementation_call)
                ):
                    raise NotReady("Worker placement requires a synchronous implementation")
                found.add(item.name)

    visit(accepted.root)
    if names - found:
        raise NotReady(f"Unknown/ inactive worker call: {sorted(names - found)}")


class Lowering:
    """A candidate target, not the language's permanent implementation model."""

    def __init__(
        self,
        accepted: Accepted,
        state: RunState,
        *,
        inline_blocks: bool = False,
        prepared_access: bool = False,
        workers: Workers | None = None,
        worker_calls: frozenset[str] = frozenset(),
        physical_modes: frozenset[str] | None = None,
    ):
        self.accepted, self.state = accepted, state
        self.inline_blocks = inline_blocks
        self.prepared_access = prepared_access
        self.workers, self.worker_calls = workers, worker_calls
        self.physical_modes = (
            mode_participants(accepted, selected_workers=workers is not None) if physical_modes is None else physical_modes
        )
        self.source: dict[str, str] = {}
        self.requests: dict[str, list[asyncio.Queue]] = {}
        self.request_queues: dict[str, asyncio.Queue] = {}
        self.finished: dict[str, asyncio.Event] = {}
        self._connect(accepted.root)

    def _connect(self, scope: Scope, activity: str | None = None) -> None:
        if isinstance(scope, Together):
            for child in scope.children:
                self._connect(child)
        elif isinstance(scope, (Sequence, Repeat)):
            for block in scope.blocks:
                self._connect(block, activity or scope.name)
        elif isinstance(scope, Block):
            for work in scope.work:
                if isinstance(work, Join):
                    for channel in work.channels:
                        self.state.progress.consumers.setdefault(channel, set()).add(activity or scope.name)
        elif isinstance(scope, Requested) and scope.enabled:
            queue = asyncio.Queue(maxsize=1)
            self.request_queues[scope.name] = queue
            self.requests.setdefault(scope.source, []).append(queue)
            self._connect(scope.block, scope.name)
        elif isinstance(scope, Producer) and scope.stop_when is not None:
            self.finished.setdefault(scope.stop_when, asyncio.Event())

    def block(self, block: Block) -> Callable[[Choice, str], Awaitable[int]]:
        work = ordered(block)
        uses, states = block_access(block, self.accepted, physical_modes=self.physical_modes)
        prefix = prefix_joins(work)
        first_joins = {item.name for item in prefix}
        waits = tuple(tuple(self.state.feeds[channel] for channel in item.channels) for item in prefix)
        channels = frozenset(channel for item in work if isinstance(item, Join) for channel in item.channels)
        # Ordinary blocks have no retained region. Both paths use the same lease.
        direct_access = self.prepared_access
        track_demand = bool(channels) or not self.prepared_access
        lines = ["async def body(frame):", "    choice = frame.choice.payload"]
        variables = {"choice": "choice"}
        namespace: dict[str, object] = {}
        for index, item in enumerate(work):
            args = ", ".join(["frame", *(variables[value] for value in inputs(item))])
            produced = outputs(item)
            names = [f"v_{index}_{slot}" for slot in range(len(produced))]
            assignment = ", ".join(names) + " = " if names else ""
            lines.append(f"    # accepted source {item.name!r}")
            if self.inline_blocks:
                # Specialize the dispatch/guard shell, not selected algorithms.
                # The raw invocation, output rule and failure rule are shared
                # with bind(); Call's evaluation/cleanup still surrounds its
                # output validation. There is no second authored description.
                invoke, check_output, failed = self.bind_parts(item, wait_input=item.name not in first_joins)
                namespace[f"invoke_{index}"] = invoke
                namespace[f"failed_{index}"] = failed
                result = names[0] if len(names) == 1 else f"result_{index}"
                lines.extend(
                    (
                        f"    frame.state.trace.append(('entered', ({item.name!r}, frame.invocation)))",
                        "    frame.implementation_returned = False",
                        "    try:",
                        f"        {result} = await invoke_{index}({args})",
                    )
                )
                if check_output is not None:
                    namespace[f"output_{index}"] = check_output
                    lines.append(f"        output_{index}({result})")
                lines.extend(
                    (
                        f"        frame.state.trace.append(('returned', ({item.name!r}, frame.invocation)))",
                        "    except BaseException as error:",
                        f"        failed_{index}(frame, error)",
                    )
                )
                if len(names) != 1:
                    if names:
                        lines.append(f"    {assignment}{result}")
                    lines.append(f"    del {result}")
            else:
                namespace[f"work_{index}"] = self.bind(item, wait_input=item.name not in first_joins)
                lines.append(f"    {assignment}await work_{index}({args})")
            variables.update({output.name: name for output, name in zip(produced, names, strict=True)})
        if not work:
            lines.append("    pass")
        source = "\n".join(lines) + "\n"
        self.source[block.name] = source
        exec(compile(source, f"<candidate:{block.name}>", "exec"), namespace)
        body = cast(Callable[[Frame], Awaitable[None]], namespace["body"])

        async def invoke(choice: Choice, parent: str) -> int:
            with self.state.progress.demand(parent, channels, choice.key) if track_demand else nullcontext():
                for feeds in waits:
                    await wait_for_products(feeds, choice.key)
                async with self.state.acquire(uses, states) if direct_access else self.state.use(uses, states) as projection:
                    self.state.invocation += 1
                    frame = Frame(self.state, parent, choice, projection, self.state.invocation)
                    self.state.trace.append(("invocation", (parent, block.name, choice.key, frame.invocation)))
                    await body(frame)
                    return frame.invocation

        return invoke

    def bind_parts(self, work: Work, *, wait_input: bool = True, keep_graph: bool = False, retained_exclusive: tuple[str, ...] = ()):
        """One set of runtime rules used by both CPU emission paths."""
        declared = outputs(work)

        def check_output(result):
            if not declared:
                if result is not None:
                    raise TypeError("Declared no output")
            else:
                values = (result,) if len(declared) == 1 else result
                if not isinstance(values, tuple) or len(values) != len(declared):
                    raise TypeError("Returned output structure disagrees with accepted structure")
                if any(not isinstance(value, cast(type, output.type)) for value, output in zip(values, declared, strict=True)):
                    raise TypeError("Returned output type disagrees with accepted schema")

        if isinstance(work, Call):
            # Resolve participating addresses once. Context.models deliberately
            # omits named-view-only access, but evaluation still covers it.
            evaluation_names = tuple(dict.fromkeys(use.participant for use in work.uses)) if work.evaluation else ()
            workers = self.workers if work.name in self.worker_calls else None

            def synchronous(context, evaluation_models, args) -> CallReturn:
                modes = tuple((module, module.training) for model in evaluation_models for module in model.modules())
                result, primary, returned = None, None, False
                try:
                    for model in evaluation_models:
                        model.eval()
                    result = work.implementation(context, *args)
                    if inspect.isawaitable(result) or inspect.isasyncgen(result):
                        if inspect.iscoroutine(result):
                            result.close()
                        raise TypeError("Worker implementation returned asynchronous work")
                    returned = True
                    check_output(result)
                except BaseException as error:
                    primary = error
                cleanup = []
                for module, training in modes:
                    try:
                        module.training = training
                    except BaseException as error:
                        cleanup.append(error)
                return CallReturn(result, returned, primary, tuple(cleanup))

            async def invoke(frame, *args):
                retained = args[-1] if work.retained is not None else None
                args = args[:-1] if work.retained is not None else args
                context = frame.context(work, retained)
                evaluation_models = tuple(frame.projection.models[name] for name in evaluation_names) if evaluation_names else ()
                if workers is not None:
                    stop = threading.Event()
                    context = replace(context, stop_requested=stop.is_set)
                    # Preserve selected-return evidence even if the outer CPU/
                    # Python context fails on exit. Observe only after the
                    # physical future is quiescent; no run state enters threads.
                    selected_return = None

                    def execute():
                        nonlocal selected_return
                        selected_return = synchronous(context, evaluation_models, args)
                        return selected_return

                    try:
                        finished = await workers.execute(execute, stop)
                    except PhysicalFailed as error:
                        if selected_return is not None:
                            frame.implementation_returned = selected_return.returned
                            if selected_return.returned:
                                self.state.trace.append(("implementation_returned", (work.name, frame.invocation)))
                                self.state.returned_writes(work.uses)
                            if selected_return.primary is not None:
                                primary, cleanup = selected_return.primary, (*selected_return.cleanup, error.cause)
                            else:
                                primary, cleanup = error.cause, selected_return.cleanup
                        else:
                            primary, cleanup = error.cause, ()
                        raise CallFailed(primary, cleanup, error.cancellation) from error
                    report = finished.value
                    frame.implementation_returned = report.returned
                    self.state.trace.append(("worker_completed", (work.name, frame.invocation, finished.cancellation is not None)))
                    if report.returned:
                        self.state.trace.append(("implementation_returned", (work.name, frame.invocation)))
                        self.state.returned_writes(work.uses)
                    if report.primary is not None or report.cleanup or finished.cancellation is not None:
                        primary = report.primary if report.primary is not None else finished.cancellation
                        raise CallFailed(primary, report.cleanup, finished.cancellation)
                    return report.value
                modes = (
                    tuple((module, module.training) for model in evaluation_models for module in model.modules())
                    if evaluation_models
                    else ()
                )
                primary = None
                result = None
                try:
                    for model in evaluation_models:
                        model.eval()
                    result = work.implementation(context, *args)
                    result = await result if inspect.isawaitable(result) else result
                    frame.implementation_returned = True
                    self.state.trace.append(("implementation_returned", (work.name, frame.invocation)))
                    self.state.returned_writes(work.uses)
                    check_output(result)  # An invalid return must survive cleanup failure too.
                except BaseException as error:
                    primary = error
                cleanup = []
                for module, training in modes:
                    try:
                        module.training = training
                    except BaseException as error:
                        cleanup.append(error)
                if primary is not None or cleanup:
                    raise CallFailed(primary, tuple(cleanup))
                return result
        elif isinstance(work, Join):
            feeds = tuple(self.state.feeds[name] for name in work.channels)

            async def invoke(frame):
                values = await join(feeds, frame.choice.key, frame.invocation, work.admit, frame.current(), wait=wait_input)
                self.state.trace.append(("admitted", (frame.invocation, frame.choice.key, work.channels)))
                return values[0] if len(values) == 1 else values
        elif isinstance(work, Differentiate):

            async def invoke(frame, loss, *links):
                retained = links[0] if work.retained is not None else None
                previous = links[-1] if work.accumulate is not None else None
                return self.state.contribute(
                    loss,
                    work.units,
                    frame.invocation,
                    retained=retained,
                    region=frame.region,
                    keep_graph=keep_graph,
                    previous=previous,
                )
        elif isinstance(work, Advance):

            async def invoke(frame, contribution):
                self.state.advance(contribution, work.units, frame.invocation, region=frame.region)
        elif isinstance(work, Retain):

            async def invoke(frame):
                if frame.region is None:
                    raise NotReady("No accepted cross-block lifetime")
                return await self.state.retain(work, frame.region, exclusive=retained_exclusive)
        elif isinstance(work, Release):

            async def invoke(frame, retained):
                await self.state.release(retained, frame.region)
        elif isinstance(work, TwoPass):

            async def invoke(frame, *args):
                access = GrantedAccess(self.state, work.unit, frame.invocation)
                models = MappingProxyType({use.participant: frame.projection.models[use.participant] for use in work.uses})
                result = work.implementation(access, models, *args)
                if inspect.isawaitable(result):
                    await result
                frame.implementation_returned = True
                self.state.trace.append(("implementation_returned", (work.name, frame.invocation)))
                access._check_handback()
                self.state.returned_writes(work.uses)
                return self.state.complete_contribution((work.unit,), frame.invocation, granted=True)
        elif isinstance(work, Replace):

            async def invoke(frame, proposal):
                if not isinstance(proposal, Replacement):
                    raise NotReady("Expected a typed replacement proposal")
                await self.state.replace(proposal, work.participant)
        else:
            raise NotReady(f"No target rule for {type(work).__name__}")

        def failed(frame: Frame, error: BaseException) -> NoReturn:
            if isinstance(error, NotEntered):
                self.state.trace.append(("worker_not_started", (work.name, frame.invocation)))
                raise error.cause from error
            if isinstance(work, Join) and isinstance(error, NotReady):
                # Rejected before claiming inputs/computation: don't
                # turn inadmissibility into uncertain numerical effects.
                raise error
            # Withholding survives lease release. No rollback or safe retry.
            self.state.unusable.update(frame.projection.models)
            if isinstance(work, Call):
                self.state.unusable_states.update(work.states)
            if isinstance(error, CallFailed):
                primary = error.primary
                if isinstance(primary, asyncio.CancelledError) and not error.cleanup:
                    raise primary from error
                cause = primary if primary is not None else error.cleanup[0]
                failure = WorkFailed(work.name, frame, cause, error.cleanup)
                failure.cancellation = error.cancellation
                raise failure from error
            if isinstance(error, asyncio.CancelledError):
                raise error
            raise WorkFailed(work.name, frame, error) from error

        return invoke, None if isinstance(work, Call) else check_output, failed

    def bind(self, work: Work, *, wait_input: bool = True, keep_graph: bool = False, retained_exclusive: tuple[str, ...] = ()):
        invoke, check_output, failed = self.bind_parts(
            work, wait_input=wait_input, keep_graph=keep_graph, retained_exclusive=retained_exclusive
        )

        async def checked(frame: Frame, *args):
            self.state.trace.append(("entered", (work.name, frame.invocation)))
            frame.implementation_returned = False
            try:
                result = await invoke(frame, *args)
                if check_output is not None:
                    check_output(result)
                self.state.trace.append(("returned", (work.name, frame.invocation)))
                return result
            except BaseException as error:
                failed(frame, error)

        return checked

    def sequence(self, scope: Sequence) -> Callable[[Choice | None, str], Awaitable[int]]:
        """Lower cross-block links to locals, with work-sized current-use protection.

        Retained sources have their OWN lifetime. Acquiring a whole block's
        future write permissions here would deadlock its earlier release.
        """
        groups = sequence_order(scope)
        work_items = tuple(item for _, work in groups for item in work)
        channels = frozenset(channel for item in work_items if isinstance(item, Join) for channel in item.channels)
        exclusions = retained_exclusions(work_items, self.accepted, physical_modes=self.physical_modes)
        bound = []
        for index, work in enumerate(work_items):
            keep_graph = (
                isinstance(work, Differentiate)
                and work.retained is not None
                and any(isinstance(later, Differentiate) and later.retained == work.retained for later in work_items[index + 1 :])
            )
            exclusive = exclusions.get(work.output, ()) if isinstance(work, Retain) else ()
            function = self.bind(work, keep_graph=keep_graph, retained_exclusive=exclusive)

            def protect(item, selected):
                uses = sequence_uses(item, self.accepted, physical_modes=self.physical_modes)
                states = item.states if isinstance(item, Call) else ()

                async def protected(frame, *args):
                    async with self.state.use(uses, states, region=frame.region, publication=isinstance(item, Replace)) as projection:
                        return await selected(replace(frame, projection=projection), *args)

                return protected

            bound.append(protect(work, function))

        async def enter(name: str, region: object, selected: Choice | None):
            # The immediate owner is this Sequence. Its Repeat, when present,
            # receives a separate whole-region Completion using the last
            # invocation, not a fabricated flat dispatch-scope identity.
            self.state.invocation += 1
            choice = selected if selected is not None else Choice(name)
            frame = Frame(self.state, scope.name, choice, Projection({}, {}), self.state.invocation, region=region)
            self.state.trace.append(("invocation", (scope.name, name, choice.key, frame.invocation)))
            return frame

        def completed(name: str):
            counts = self.state.coordinates.setdefault(scope.name, {})
            counts[name] = counts.get(name, 0) + 1

        namespace: dict[str, object] = {"enter": enter, "completed": completed}
        lines = ["async def body(region, selected):", "    choice = selected.payload if selected is not None else None"]
        variables = {"choice": "choice"}
        index = 0
        for block, work in groups:
            lines.append(f"    frame = await enter({block.name!r}, region, selected)")
            for item in work:
                namespace[f"work_{index}"] = bound[index]
                args = ", ".join(["frame", *(variables[value] for value in inputs(item))])
                produced = outputs(item)
                names = [f"v_{index}_{slot}" for slot in range(len(produced))]
                assignment = ", ".join(names) + " = " if names else ""
                lines.append(f"    # accepted source {item.name!r}")
                lines.append(f"    {assignment}await work_{index}({args})")
                variables.update({output.name: name for output, name in zip(produced, names, strict=True)})
                index += 1
            lines.append(f"    completed({block.name!r})")
        lines.append("    return frame.invocation")
        source = "\n".join(lines) + "\n"
        self.source[scope.name] = source
        exec(compile(source, f"<candidate:{scope.name}>", "exec"), namespace)
        body = cast(Callable[[object, Choice | None], Awaitable[int]], namespace["body"])
        participants = {use.participant for item in work_items for use in getattr(item, "uses", ())} | {
            name for item in work_items if isinstance(item, Retain) for name in item.participants
        }
        owner_states = {name for item in work_items if isinstance(item, Call) for name in item.states}

        async def sequence(choice: Choice | None, parent: str) -> int:
            region = object()  # Execution correspondence, not a semantic identity.
            primary = None
            invocation = 0
            try:
                with self.state.progress.demand(parent, channels, choice.key if choice is not None else ""):
                    invocation = await body(region, choice)
            except BaseException as error:
                primary = error
                self.state.unusable.update(participants)
                self.state.unusable_states.update(owner_states)

            # Drain only this region's remaining lifetimes, even on repeated
            # cancellation. This releases protection, NOT failed gradients or
            # evidence of uncertainty; it does not make replay safe.
            task = asyncio.create_task(self.state.release_region(region))
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    if primary is None:
                        primary = error
                    self.state.unusable.update(participants)
                    self.state.unusable_states.update(owner_states)
                except BaseException:
                    break
            cleanup_failure = task.exception()
            if primary is not None and cleanup_failure is not None:
                raise BaseExceptionGroup("Sequence execution and lifetime cleanup failed", (primary, cleanup_failure))
            if primary is not None:
                raise primary
            if cleanup_failure is not None:
                raise cleanup_failure
            return invocation

        return sequence

    def scope(self, scope: Scope) -> Callable[[], Coroutine[Any, Any, None]]:
        body = self._scope(scope)
        finished = self.finished.get(scope.name)

        async def tracked():
            with self.state.progress.activity(scope.name):
                await body()
                if finished is not None:
                    finished.set()  # Whole lifetime normal completion, not one block invocation.
                    self.state.trace.append(("demand_ended", scope.name))

        return tracked

    def _scope(self, scope: Scope) -> Callable[[], Coroutine[Any, Any, None]]:
        if isinstance(scope, Sequence):
            sequence = self.sequence(scope)

            async def once():
                await sequence(None, scope.name)

            return once
        if isinstance(scope, Block):
            body = self.block(scope)

            async def once():
                await body(Choice(scope.name), scope.name)

            return once

        if isinstance(scope, Together):
            children = tuple(self.scope(child) for child in scope.children)
            self.source[scope.name] = "TaskGroup(" + ", ".join(child.name for child in scope.children) + ")"

            async def together():
                async with asyncio.TaskGroup() as tasks:
                    for child in children:
                        tasks.create_task(child())

            return together

        if isinstance(scope, Repeat):
            blocks = {block.name: self.sequence(block) if isinstance(block, Sequence) else self.block(block) for block in scope.blocks}
            subscribers = tuple(self.requests.get(scope.name, ()))
            counts = self.state.coordinates.setdefault(scope.name, {})
            self.source[scope.name] = f"selected policy -> bounded alternatives {tuple(blocks)}; {len(subscribers)} request connections"

            async def repeat():
                position = sum(counts.values())
                while True:
                    async with self.state.acquire((), (scope.state,)):
                        try:
                            choice = _choice(
                                scope.policy(MappingProxyType(counts), self.state.owners[scope.state]),
                                blocks,
                                "Runtime policy selected an unaccepted alternative",
                            )
                        except BaseException:
                            # An invoked policy can mutate its owned state.
                            self.state.unusable_states.add(scope.state)
                            raise
                    if choice is None:
                        break
                    invocation = await blocks[choice.block](choice, scope.name)
                    counts[choice.block] = counts.get(choice.block, 0) + 1
                    position += 1
                    completion = Completion(scope.name, choice.block, choice.key, invocation, position)
                    for queue in subscribers:
                        await queue.put(completion)
                # Only normal completion closes streams. Failures propagate to
                # the owning TaskGroup; they must not block while closing one.
                for queue in subscribers:
                    await queue.put(None)

            return repeat

        if isinstance(scope, Producer):
            return self.producer(scope)

        if isinstance(scope, Requested):
            if not scope.enabled:

                async def dormant():
                    return None

                return dormant
            body = self.block(scope.block)
            queue = self.request_queues[scope.name]
            self.source[scope.name] = f"requests from {scope.source} -> accepted request policy -> protected {scope.block.name}"

            async def requested():
                while (completion := await queue.get()) is not None:
                    choice = _choice(scope.request(completion), (scope.block.name,), "Request selected an unprovided capability")
                    if choice is not None:
                        await body(choice, scope.name)

            return requested
        raise NotReady(f"Unsupported scope {type(scope).__name__}")

    def producer(self, scope: Producer) -> Callable[[], Coroutine[Any, Any, None]]:
        self.source[scope.name] = f"independent activity -> bounded ports {scope.channels}; stop={scope.stop_when!r}"
        finished = self.finished.get(scope.stop_when) if scope.stop_when is not None else None
        feeds = tuple(self.state.feeds[name] for name in scope.channels)

        async def producer():
            stopping = False
            phase = "starting"
            if finished is not None and finished.is_set():
                for feed in feeds:
                    await feed.end_demand()
                self.state.trace.append(("producer_not_started", scope.name))
                return

            async def activity():
                nonlocal phase
                if finished is not None and finished.is_set():
                    self.state.trace.append(("producer_not_started", scope.name))
                    return
                # Lookup at actual startup; pin this state until implementation
                # and explicit cleanup have joined all private work.
                async with self.state.producer_owner(scope.name, scope.states) as owners:
                    phase = "execution"
                    context = ProviderContext(scope, self.state, owners)
                    primary = None
                    returned = False
                    try:
                        await scope.implementation(context)
                        returned = True
                        self.state.trace.append(("production_returned", scope.name))
                    except BaseException as error:
                        primary = error
                    phase = "cleanup"
                    planned_stop = stopping and isinstance(primary, asyncio.CancelledError)
                    if primary is not None and not planned_stop:
                        # Known uncertainty gates other owners immediately,
                        # even while private cleanup is still awaiting work.
                        self.state.unusable.update(use.participant for use in scope.uses)
                        self.state.unusable_states.update(scope.states)
                        for feed in feeds:
                            await feed.close(primary)
                    cleanup = []
                    if scope.cleanup is not None:
                        try:
                            await scope.cleanup(context)
                        except BaseException as error:
                            cleanup.append(error)
                    if cleanup or (primary is not None and not planned_stop):
                        self.state.unusable.update(use.participant for use in scope.uses)
                        self.state.unusable_states.update(scope.states)
                        if cleanup:
                            raise ProducerFailed(scope.name, primary, tuple(cleanup), returned)
                        assert primary is not None
                        raise primary
                    self.state.trace.append(("producer_quiescent", (scope.name, returned, planned_stop)))

            task = asyncio.create_task(activity())
            stop = asyncio.create_task(finished.wait()) if finished is not None else None
            failure = None
            try:
                if stop is not None:
                    await asyncio.wait((task, stop), return_when=asyncio.FIRST_COMPLETED)
                    if task.done() and not stop.done():
                        await asyncio.shield(task)  # Report real failure immediately.
                        for feed in feeds:
                            await feed.close()
                        await stop  # Retain port coordination, not the finished private owner.
                    stopping = True
                    if phase in ("starting", "execution") and not task.done():
                        task.cancel()
                    for feed in feeds:
                        await feed.end_demand()
                    self.state.trace.append(("producer_stop_requested", scope.name))
                await asyncio.shield(task)
            except asyncio.CancelledError as cancelled:
                if phase in ("starting", "execution") and not task.done():
                    task.cancel()
                # Additional parent cancellation must not detach still-running
                # cleanup or release its continuation registration prematurely.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                secondary = None if task.cancelled() else task.exception()
                if secondary is not None:
                    failure = BaseExceptionGroup("Producer interruption and cleanup failure", (cancelled, secondary))
                    raise failure from cancelled
                if phase == "starting" and task.cancelled():
                    # No owner registration or selected work was reached: no
                    # private cleanup is owed and no effects are inferred. A
                    # planned pre-start stop is normal, unlike parent failure.
                    self.state.trace.append(("producer_not_started", scope.name))
                    current = asyncio.current_task()
                    if stopping and current is not None and not current.cancelling():
                        return
                failure = cancelled
                raise
            except BaseException as error:
                failure = error
                raise
            finally:
                if stop is not None:
                    stop.cancel()
                    await asyncio.gather(stop, return_exceptions=True)
                for feed in feeds:
                    await feed.close(failure)

        return producer
