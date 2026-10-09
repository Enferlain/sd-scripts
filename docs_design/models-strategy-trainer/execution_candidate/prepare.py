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
from dataclasses import dataclass
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
    Replace,
    Requested,
    Scope,
    Together,
    TwoPass,
    Use,
    Work,
    inputs,
    ordered,
    outputs,
)
from .state import Contribution, Image, Product, Projection, Replacement, RunState, Stamp, UnitOutcome, join, wait_for_products


@dataclass(frozen=True)
class Context:
    """Accepted runtime seam, not a strategy or unrestricted Trainer."""

    models: Mapping[str, torch.nn.Module]
    states: Mapping[str, object]
    choice: Choice
    invocation: int
    stamps: Mapping[str, Stamp]


@dataclass
class Frame:
    state: RunState
    scope: str
    choice: Choice
    projection: Projection
    invocation: int

    def current(self) -> Projection:
        # The lease keeps bindings stable, not numerical state frozen across
        # this block's own accepted writes/optimizer calls.
        return Projection(
            self.projection.models,
            MappingProxyType({name: self.state.stamp(name) for name in self.projection.models}),
        )

    def context(self, call: Call) -> Context:
        return Context(
            MappingProxyType({use.participant: self.projection.models[use.participant] for use in call.uses}),
            MappingProxyType({name: self.state.owners[name] for name in call.states}),
            self.choice,
            self.invocation,
            MappingProxyType({use.participant: self.state.stamp(use.participant) for use in call.uses}),
        )


class WorkFailed(RuntimeError):
    """A reached invocation, including invalid output after possible effects."""

    def __init__(self, source: str, frame: Frame, cause: BaseException):
        super().__init__(f"{source}: post-invocation failure ({cause})")
        self.source, self.scope, self.invocation, self.cause = source, frame.scope, frame.invocation, cause
        assert frame.state.image is not None
        self.unit_outcomes: tuple[UnitOutcome, ...] = tuple(
            copy.copy(outcome)
            for runtime in frame.state.image.units.values()
            for outcome in runtime.outcomes
            if outcome.invocation == frame.invocation
        )


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

    def __init__(self, source: Producer, state: RunState):
        self.source = source
        self._state = state
        self.states = MappingProxyType({name: state.owners[name] for name in source.states})
        self.ports = MappingProxyType({name: state.feeds[name] for name in source.channels})

    def use_source(self):
        return self._state.acquire(self.source.uses)

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
        self.producers: dict[str, Producer] = {}
        self.blocks: list[Block] = []
        self._connect(accepted.root)
        self._check_wait_cycles()

    def _connect(self, scope: Scope) -> None:
        if isinstance(scope, Together):
            for child in scope.children:
                self._connect(child)
        elif isinstance(scope, Requested) and scope.enabled:
            queue = asyncio.Queue(maxsize=1)
            self.request_queues[scope.name] = queue
            self.requests.setdefault(scope.source, []).append(queue)
            self.blocks.append(scope.block)
        elif isinstance(scope, Producer):
            self.producers.update(dict.fromkeys(scope.channels, scope))
        elif isinstance(scope, Repeat):
            self.blocks.extend(scope.blocks)
        elif isinstance(scope, Block):
            self.blocks.append(scope)

    def _check_wait_cycles(self) -> None:
        """Cold, conservative analysis of declared lease/handoff relationships.

        Prefix readiness holds no lease. Later joins do: they wait for a
        producer that may need a participant held by another waiting block.
        Include indirect edges, not just a block's own producer conflict.
        This can reject feasible schedules; it is not a global liveness proof
        for arbitrary Python, private provider logic or future regions.
        """
        accesses = {}
        edges: dict[str, set[str]] = {}
        for block in self.blocks:
            uses = self._block_uses(block)
            access = dict.fromkeys((use.participant for use in uses), "read")
            access.update(dict.fromkeys((use.participant for use in uses if use.access == "write"), "write"))
            accesses[block.name] = access
            edges[block.name] = set()
            prefix = True
            for item in ordered(block):
                if not isinstance(item, Join):
                    prefix = False
                elif not prefix:
                    edges[block.name].update(self.producers[channel].name for channel in item.channels)
        for producer in self.producers.values():
            edges[producer.name] = {
                block
                for block, access in accesses.items()
                if any(use.participant in access and (use.access == "write" or access[use.participant] == "write") for use in producer.uses)
            }
        active: set[str] = set()
        done: set[str] = set()

        def visit(node: str) -> None:
            if node in active:
                raise NotReady(f"Potential lease/source wait cycle in this target: {node}")
            if node in done:
                return
            active.add(node)
            for dependency in edges[node]:
                visit(dependency)
            active.remove(node)
            done.add(node)

        for node in edges:
            visit(node)

    def _block_uses(self, block: Block) -> tuple[Use, ...]:
        uses: list[Use] = []
        for work in block.work:
            uses.extend(getattr(work, "uses", ()))
            units = work.units if isinstance(work, (Differentiate, Advance)) else (work.unit,) if isinstance(work, TwoPass) else ()
            for unit in units:
                uses.extend(Use(member[0], "write") for member in self.accepted.units[unit].members)
            if isinstance(work, Call) and work.evaluation:
                uses.extend(Use(use.participant, "write") for use in work.uses)
        return tuple(uses)

    def block(self, block: Block) -> Callable[[Choice, str], Awaitable[int]]:
        work = ordered(block)
        uses = self._block_uses(block)
        states = tuple(dict.fromkeys(name for item in work if isinstance(item, Call) for name in item.states))
        if any(isinstance(item, Replace) for item in work) and (uses or states):
            raise NotReady("This target publishes replacement only outside protected numerical/state work")
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

    def bind(self, work: Work, *, wait_input: bool = True):
        if isinstance(work, Call):

            async def invoke(frame, *args):
                context = frame.context(work)
                modes = tuple((module, module.training) for model in context.models.values() for module in model.modules())
                try:
                    if work.evaluation:
                        for model in context.models.values():
                            model.eval()
                    result = work.implementation(context, *args)
                    return await result if inspect.isawaitable(result) else result
                finally:
                    if work.evaluation:
                        for module, training in modes:
                            module.training = training
        elif isinstance(work, Join):
            feeds = tuple(self.state.feeds[name] for name in work.channels)

            async def invoke(frame):
                values = await join(feeds, frame.choice.key, frame.invocation, work.admit, frame.current(), wait=wait_input)
                self.state.trace.append(("admitted", (frame.invocation, frame.choice.key, work.channels)))
                return values[0] if len(values) == 1 else values
        elif isinstance(work, Differentiate):

            async def invoke(frame, loss):
                return self.state.contribute(loss, work.units, frame.invocation)
        elif isinstance(work, Advance):

            async def invoke(frame, contribution):
                self.state.advance(contribution, work.units, frame.invocation)
        elif isinstance(work, TwoPass):

            async def invoke(frame, *args):
                access = GrantedAccess(self.state, work.unit, frame.invocation)
                models = MappingProxyType({use.participant: frame.projection.models[use.participant] for use in work.uses})
                result = work.implementation(access, models, *args)
                if inspect.isawaitable(result):
                    await result
                return access._handback(work.unit)
        elif isinstance(work, Replace):

            async def invoke(frame, proposal):
                if not isinstance(proposal, Replacement):
                    raise NotReady("Expected a typed replacement proposal")
                await self.state.replace(proposal, work.participant)
        else:
            raise NotReady(f"No target rule for {type(work).__name__}")

        declared = outputs(work)

        async def checked(frame: Frame, *args):
            self.state.trace.append(("entered", (work.name, frame.invocation)))
            try:
                result = await invoke(frame, *args)
                if isinstance(work, Call):
                    # A declared write is conservative dependency evidence,
                    # not numerical comparison of before/after tensors.
                    for name in {use.participant for use in work.uses if use.access == "write"}:
                        self.state.numerical_epochs[name] += 1
                if not declared:
                    if result is not None:
                        raise TypeError("Declared no output")
                else:
                    values = (result,) if len(declared) == 1 else result
                    if not isinstance(values, tuple) or len(values) != len(declared):
                        raise TypeError("Returned output structure disagrees with accepted structure")
                    if any(not isinstance(value, output.type) for value, output in zip(values, declared)):
                        raise TypeError("Returned output type disagrees with accepted schema")
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
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise WorkFailed(work.name, frame, error) from error

        return checked

    def scope(self, scope: Scope) -> Callable[[], Coroutine[Any, Any, None]]:
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
            provider = ProviderContext(scope, self.state)
            self.source[scope.name] = f"independent selected activity -> bounded ports {scope.channels}"

            async def producer():
                failure = None
                try:
                    await scope.implementation(provider)
                except BaseException as error:
                    failure = error
                    # Source use failures must not certify still-usable state.
                    self.state.unusable.update(use.participant for use in scope.uses)
                    self.state.unusable_states.update(scope.states)
                    raise
                finally:
                    for feed in provider.ports.values():
                        await feed.close(failure)

            return producer

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
