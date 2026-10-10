"""Lower a checked whole-run description into directly callable machinery.

Only preparation dispatches on IR constructs. Engine.run invokes the derived
root. Blocks become direct Python call chains; repeated and independent
scopes become selected-policy loops and TaskGroup children. Their request
connections are resolved once. This is one concrete CPU/asyncio target.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import torch

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
    block_uses,
    inputs,
    ordered,
    outputs,
    sequence_order,
)
from .state import (
    Contribution,
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


@dataclass(frozen=True)
class Context:
    """Accepted runtime seam, not a strategy or unrestricted Trainer."""

    models: Mapping[str, torch.nn.Module]
    states: Mapping[str, object]
    choice: Choice
    invocation: int
    stamps: Mapping[str, Stamp]
    views: Mapping[str, PreparedView] = field(default_factory=dict)


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
        self.implementation_returned = frame.implementation_returned
        assert frame.state.image is not None
        self.unit_outcomes: tuple[UnitOutcome, ...] = tuple(
            copy.copy(outcome)
            for runtime in frame.state.image.units.values()
            for outcome in runtime.outcomes
            if outcome.invocation == frame.invocation or (frame.region is not None and outcome.region is frame.region)
        )


class CallFailed(RuntimeError):
    """Internal separation of selected execution from mode restoration."""

    def __init__(self, primary: BaseException | None, cleanup: tuple[BaseException, ...]):
        super().__init__("Selected execution and/or evaluation cleanup failed")
        self.primary, self.cleanup = primary, cleanup


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
        self._runtime = state.image.units[unit]
        if self._runtime.pending is not None:
            raise NotReady("Two-pass entry requires a quiescent window")
        self.invocation = invocation
        self.parameters = self._runtime.parameters
        self.original = tuple(parameter.detach().clone() for parameter in self.parameters)
        self.phases: list[str] = []
        self.restored = False
        self._runtime.outcome(invocation).contribution = "unverified_grant"

    def backward(self, loss: torch.Tensor) -> None:
        if self.phases not in ([], ["backward", "temporary_edit", "intermediate_reset"]):
            raise NotReady("Unsupported granted backward phase")
        gradients = torch.autograd.grad(loss, self.parameters)
        for parameter, gradient in zip(self.parameters, gradients):
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
            for parameter, original in zip(self.parameters, self.original):
                parameter.copy_(original)
        self.restored = True
        self.phases.append("restore")

    def _handback(self, unit: str) -> Contribution:
        if self.phases != ["backward", "temporary_edit", "intermediate_reset", "backward", "restore"]:
            raise NotReady("Incomplete two-pass handback")
        if not self.restored or any(
            not torch.equal(parameter.detach(), original) for parameter, original in zip(self.parameters, self.original)
        ):
            raise NotReady("Current view was not restored")
        if any(parameter.grad is None for parameter in self.parameters):
            raise NotReady("No final gradient to hand back")
        self._runtime.pending = self.invocation
        self._runtime.outcome(self.invocation).contribution = "granted_ready"
        return Contribution(self.invocation, (unit,), (self._runtime.prepared_key,))


class ProviderContext:
    """A provider owns private workers; coordination owns its handoff ports."""

    def __init__(self, source: Producer, state: RunState, owners: Mapping[str, object]):
        self.source = source
        self._state = state
        self.states = owners
        self.ports = MappingProxyType({name: state.feeds[name] for name in source.channels})

    @asynccontextmanager
    async def use_source(self, uses: tuple[Use, ...] | None = None):
        uses = self.source.uses if uses is None else uses
        allowed = {use.participant: "read" for use in self.source.uses}
        allowed.update({use.participant: "write" for use in self.source.uses if use.access == "write"})
        if any(
            use.participant not in allowed
            or use.access not in ("read", "write")
            or (use.access == "write" and allowed[use.participant] != "write")
            for use in uses
        ):
            raise OutOfBounds("Provider source access exceeds accepted authority")
        async with self._state.acquire(uses) as projection:
            try:
                yield projection
            except BaseException:
                self._state.unusable.update(use.participant for use in uses if use.access == "write")
                raise
            else:
                # Complete the write fact before releasing protection. Entry
                # stamps do NOT describe intermediate/final writes inside it.
                self._state.returned_writes(uses)

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


@dataclass(frozen=True)
class Preparation:
    state: RunState
    expected: int
    image: Image
    root: Callable[[], Awaitable[None]]
    source: Mapping[str, str]

    def publish(self) -> Ready:
        self.state.install(self.image, self.expected)
        return Ready(self.state, self.root, self.source)


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


def prepare(accepted: Accepted, state: RunState | None = None) -> Preparation:
    """One unpublished attempt: load, verify, prepare units, lower whole run."""
    state = state or RunState(accepted)
    if state.accepted is not accepted:
        raise NotReady("Preparation belongs to another accepted authority")
    if state.image is not None:
        raise NotReady("Use the explicit replacement path for an already installed run")
    expected = state.generation
    models = {name: declaration.load() for name, declaration in accepted.participants.items()}
    image = state.candidate(models)
    lowerer = Lowering(accepted, state)
    root = lowerer.scope(accepted.root)
    return Preparation(state, expected, image, root, MappingProxyType(lowerer.source))


class Lowering:
    """A candidate target, not the language's permanent implementation model."""

    def __init__(self, accepted: Accepted, state: RunState):
        self.accepted, self.state = accepted, state
        self.source: dict[str, str] = {}
        self.requests: dict[str, list[asyncio.Queue]] = {}
        self.request_queues: dict[str, asyncio.Queue] = {}
        self.finished: dict[str, asyncio.Event] = {}
        self._connect(accepted.root)

    def _connect(self, scope: Scope) -> None:
        if isinstance(scope, Together):
            for child in scope.children:
                self._connect(child)
        elif isinstance(scope, Sequence):
            for block in scope.blocks:
                self._connect(block)
        elif isinstance(scope, Requested) and scope.enabled:
            queue = asyncio.Queue(maxsize=1)
            self.request_queues[scope.name] = queue
            self.requests.setdefault(scope.source, []).append(queue)
        elif isinstance(scope, Producer) and scope.stop_when is not None:
            self.finished.setdefault(scope.stop_when, asyncio.Event())

    def block(self, block: Block) -> Callable[[Choice, str], Awaitable[int]]:
        work = ordered(block)
        uses = block_uses(block, self.accepted.units)
        states = tuple(dict.fromkeys(name for item in work if isinstance(item, Call) for name in item.states))
        prefix: list[Join] = []
        for item in work:
            if not isinstance(item, Join):
                break
            prefix.append(item)
        first_joins = {item.name for item in prefix}
        waits = tuple(tuple(self.state.feeds[channel] for channel in item.channels) for item in prefix)
        bound = tuple(self.bind(item, wait_input=item.name not in first_joins) for item in work)
        lines = ["async def body(frame):", "    choice = frame.choice.payload"]
        variables = {"choice": "choice"}
        namespace: dict[str, object] = {}
        for index, (item, function) in enumerate(zip(work, bound)):
            namespace[f"work_{index}"] = function
            args = ", ".join(["frame", *(variables[value] for value in inputs(item))])
            produced = outputs(item)
            names = [f"v_{index}_{slot}" for slot in range(len(produced))]
            assignment = ", ".join(names) + " = " if names else ""
            lines.append(f"    # accepted source {item.name!r}")
            lines.append(f"    {assignment}await work_{index}({args})")
            variables.update({output.name: name for output, name in zip(produced, names)})
        if not work:
            lines.append("    pass")
        source = "\n".join(lines) + "\n"
        self.source[block.name] = source
        exec(compile(source, f"<candidate:{block.name}>", "exec"), namespace)
        body = cast(Callable[[Frame], Awaitable[None]], namespace["body"])

        async def invoke(choice: Choice, parent: str) -> int:
            for feeds in waits:
                await wait_for_products(feeds, choice.key)
            async with self.state.acquire(uses, states) as projection:
                self.state.invocation += 1
                frame = Frame(self.state, parent, choice, projection, self.state.invocation)
                self.state.trace.append(("invocation", (parent, block.name, choice.key, frame.invocation)))
                await body(frame)
                return frame.invocation

        return invoke

    def bind(self, work: Work, *, wait_input: bool = True, keep_graph: bool = False):
        declared = outputs(work)

        def check_output(result):
            if not declared:
                if result is not None:
                    raise TypeError("Declared no output")
            else:
                values = (result,) if len(declared) == 1 else result
                if not isinstance(values, tuple) or len(values) != len(declared):
                    raise TypeError("Returned output structure disagrees with accepted structure")
                if any(not isinstance(value, output.type) for value, output in zip(values, declared)):
                    raise TypeError("Returned output type disagrees with accepted schema")

        if isinstance(work, Call):
            # Resolve participating addresses once. Context.models deliberately
            # omits named-view-only access, but evaluation still covers it.
            evaluation_names = tuple(dict.fromkeys(use.participant for use in work.uses)) if work.evaluation else ()

            async def invoke(frame, *args):
                retained = args[-1] if work.retained is not None else None
                args = args[:-1] if work.retained is not None else args
                context = frame.context(work, retained)
                evaluation_models = tuple(frame.projection.models[name] for name in evaluation_names) if evaluation_names else ()
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

            async def invoke(frame, loss, retained=None):
                return self.state.contribute(
                    loss,
                    work.units,
                    frame.invocation,
                    retained=retained,
                    region=frame.region,
                    keep_graph=keep_graph,
                )
        elif isinstance(work, Advance):

            async def invoke(frame, contribution):
                self.state.advance(contribution, work.units, frame.invocation, region=frame.region)
        elif isinstance(work, Retain):

            async def invoke(frame):
                if frame.region is None:
                    raise NotReady("No accepted cross-block lifetime")
                return await self.state.retain(work, frame.region)
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
                contribution = access._handback(work.unit)
                self.state.returned_writes(work.uses)
                return contribution
        elif isinstance(work, Replace):

            async def invoke(frame, proposal):
                if not isinstance(proposal, Replacement):
                    raise NotReady("Expected a typed replacement proposal")
                await self.state.replace(proposal, work.participant)
        else:
            raise NotReady(f"No target rule for {type(work).__name__}")

        async def checked(frame: Frame, *args):
            self.state.trace.append(("entered", (work.name, frame.invocation)))
            frame.implementation_returned = False
            try:
                result = await invoke(frame, *args)
                if not isinstance(work, Call):
                    check_output(result)
                self.state.trace.append(("returned", (work.name, frame.invocation)))
                return result
            except BaseException as error:
                if isinstance(work, Join) and isinstance(error, NotReady):
                    # Rejected before claiming inputs/computation: don't
                    # turn inadmissibility into uncertain numerical effects.
                    raise
                # Withholding survives lease release. No rollback or safe retry.
                self.state.unusable.update(frame.projection.models)
                if isinstance(work, Call):
                    self.state.unusable_states.update(work.states)
                if isinstance(error, CallFailed):
                    primary = error.primary
                    if isinstance(primary, asyncio.CancelledError) and not error.cleanup:
                        raise primary from error
                    cause = primary if primary is not None else error.cleanup[0]
                    raise WorkFailed(work.name, frame, cause, error.cleanup) from error
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise WorkFailed(work.name, frame, error) from error

        return checked

    def sequence(self, scope: Sequence) -> Callable[[], Coroutine[Any, Any, None]]:
        """Lower cross-block links to locals, with work-sized current-use protection.

        Retained sources have their OWN lifetime. Acquiring a whole block's
        future write permissions here would deadlock its earlier release.
        """
        groups = sequence_order(scope)
        work_items = tuple(item for _, work in groups for item in work)
        bound = []
        for index, work in enumerate(work_items):
            keep_graph = (
                isinstance(work, Differentiate)
                and work.retained is not None
                and any(isinstance(later, Differentiate) and later.retained == work.retained for later in work_items[index + 1 :])
            )
            function = self.bind(work, keep_graph=keep_graph)

            def protect(item, selected):
                uses = getattr(item, "uses", ())
                if isinstance(item, Call) and item.retained is not None:
                    uses = ()  # The retained source already owns its protected/copy lifetime.
                if isinstance(item, (Differentiate, Advance)):
                    access = "write" if isinstance(item, Advance) else "read"
                    uses = tuple(Use(member[0], access) for unit in item.units for member in self.accepted.units[unit].members)
                if isinstance(item, TwoPass):
                    uses = (*uses, *(Use(member[0], "write") for member in self.accepted.units[item.unit].members))
                if isinstance(item, Call) and item.evaluation:
                    uses = (*uses, *(Use(use.participant, "write") for use in item.uses))
                states = item.states if isinstance(item, Call) else ()

                async def protected(frame, *args):
                    writes = {use.participant for use in uses if use.access == "write"}
                    own_live = {
                        name
                        for retained in self.state.active_retentions.values()
                        if retained.region is frame.region and not retained.isolated
                        for name in retained.projection.models
                    }
                    if writes & own_live or (isinstance(item, Replace) and own_live):
                        raise NotReady("A sequence cannot wait on its own protected old-state last use")
                    async with self.state.acquire(uses, states) as projection:
                        return await selected(replace(frame, projection=projection), *args)

                return protected

            bound.append(protect(work, function))

        async def enter(name: str, region: object):
            self.state.invocation += 1
            frame = Frame(self.state, scope.name, Choice(name), Projection({}, {}), self.state.invocation, region=region)
            self.state.trace.append(("invocation", (scope.name, name, "", frame.invocation)))
            return frame

        def completed(name: str):
            counts = self.state.coordinates.setdefault(scope.name, {})
            counts[name] = counts.get(name, 0) + 1

        namespace: dict[str, object] = {"enter": enter, "completed": completed}
        lines = ["async def body(region):", "    choice = None"]
        variables = {"choice": "choice"}
        index = 0
        for block, work in groups:
            lines.append(f"    frame = await enter({block.name!r}, region)")
            for item in work:
                namespace[f"work_{index}"] = bound[index]
                args = ", ".join(["frame", *(variables[value] for value in inputs(item))])
                produced = outputs(item)
                names = [f"v_{index}_{slot}" for slot in range(len(produced))]
                assignment = ", ".join(names) + " = " if names else ""
                lines.append(f"    # accepted source {item.name!r}")
                lines.append(f"    {assignment}await work_{index}({args})")
                variables.update({output.name: name for output, name in zip(produced, names)})
                index += 1
            lines.append(f"    completed({block.name!r})")
        source = "\n".join(lines) + "\n"
        self.source[scope.name] = source
        exec(compile(source, f"<candidate:{scope.name}>", "exec"), namespace)
        body = cast(Callable[[object], Awaitable[None]], namespace["body"])
        participants = {use.participant for item in work_items for use in getattr(item, "uses", ())} | {
            name for item in work_items if isinstance(item, Retain) for name in item.participants
        }
        owner_states = {name for item in work_items if isinstance(item, Call) for name in item.states}

        async def sequence():
            region = object()  # Execution correspondence, not a semantic identity.
            primary = None
            try:
                await body(region)
            except BaseException as error:
                primary = error
                self.state.unusable.update(participants)
                self.state.unusable_states.update(owner_states)

            # Drain only this region's remaining lifetimes, even on repeated
            # cancellation. This releases protection, NOT failed gradients or
            # evidence of uncertainty; it does not make replay safe.
            async def cleanup():
                for retained in tuple(self.state.active_retentions.values()):
                    if retained.region is region:
                        await self.state.release(retained, region, aborted=True)

            task = asyncio.create_task(cleanup())
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

        return sequence

    def scope(self, scope: Scope) -> Callable[[], Coroutine[Any, Any, None]]:
        body = self._scope(scope)
        finished = self.finished.get(scope.name)
        if finished is None:
            return body

        async def tracked():
            await body()
            finished.set()  # Whole lifetime normal completion, not one block invocation.
            self.state.trace.append(("demand_ended", scope.name))

        return tracked

    def _scope(self, scope: Scope) -> Callable[[], Coroutine[Any, Any, None]]:
        if isinstance(scope, Sequence):
            return self.sequence(scope)
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
            blocks = {block.name: self.block(block) for block in scope.blocks}
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
