"""Preparation: the accepted run matched to installed implementations and a target (D14, D16).

:func:`prepare` takes a cold-loaded accepted run, the installed
implementations and a target, and derives the prepared run:

- **matching**: every call's implementation is installed at its pinned
  implementation version and declaration, and the declaration agrees with the
  call's input and output types;
- **placement**: a call that declares device use goes to the device placed
  for its activity, or the target's device for device work; any other call
  runs on the host;
- **fusion**: consecutive calls in one body, on one device, whose completion
  the device covers and that have no external effects and issue no
  collectives, form one group, submitted as one piece of straight-line work.
  Every call belongs to exactly one group. Groups are numbered body by
  body in the order the strategy is walked, a body before the bodies nested
  in it;
- **device admissions** (D15): one per access body that holds device work, or
  per group of device work outside an access. It begins after the body's
  inputs are taken and ends when the last retained result it produced is
  released, where normalization placed the release;
- **leaving the device** (D15): each participant permitted to leave the
  device between uses, with the device its storage leaves from. Only frozen
  storage leaves, from one device, whose adapter offers storage copies;
- the **implementation manifest**: each pinned implementation with its
  implementation version, declaration digest and code identity.

The prepared run is inspectable and never an authority. Its printout is saved
for inspection; :func:`load_prepared` derives it again and refuses a saved one
that differs. A call that disagrees with its implementation is invalid
meaning; work the target cannot realize is a target limit, reported apart.
"""

import functools
import hashlib
import inspect
import marshal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from xdsl.ir import BlockArgument, Operation, SSAValue

from library.iro.accepted import Loaded, check_implementations
from library.iro.diagnostics import Kind, Problems, Stage
from library.iro.operation_contracts import (
    AccessMode,
    LifecycleProtocol,
    RegionSemantics,
    ResourceSemantics,
    path,
    provider,
    sym_name,
    symbol,
)
from library.iro.prepare.target import HOST, Target
from library.iro.seams import CallContext, Effect, Implementation
from library.iro.words.access import AccessOp, ReleaseOp
from library.iro.words.calls import CallOp, ConstantOp, YieldOp
from library.iro.words.data import TakeOp
from library.iro.words.lifetimes import RequestOp
from library.iro.words.model import ResourceOp
from library.iro.words.structure import StrategyOp
from library.iro.words.types import LeaseType, RetainedType, written

# Activities with lifetimes of their own: an access body's admission never reaches past them.
_OWN_LIFETIME = frozenset(
    {LifecycleProtocol.PARALLEL, LifecycleProtocol.PRODUCER, LifecycleProtocol.HANDLER, LifecycleProtocol.REPEAT, LifecycleProtocol.FILTER}
)


@dataclass(frozen=True)
class ManifestEntry:
    """A pinned implementation as the prepared run uses it.

    ``code`` identifies the code itself, which caches of prepared and compiled
    forms are keyed on; the implementation version is a promise about
    behavior and never stands in for it (D16).
    """

    name: str
    version: str
    digest: str
    code: str


@dataclass(frozen=True)
class Group:
    """Consecutive calls submitted as one piece of work on one device.

    ``context`` is what the calls establish on entry (D15). ``sync`` says a
    result of device work is used by something other than device work on
    the same device, so the work is waited for once before the result is
    handed on; otherwise later work follows it in the device's own order.
    """

    index: int
    device: str
    activity: str
    calls: tuple[str, ...]
    implementations: tuple[str, ...]
    context: CallContext = field(default_factory=CallContext)
    sync: bool = False


@dataclass(frozen=True)
class Admission:
    """One device admission: where it begins, the groups it covers and the releases that end it."""

    device: str
    begins: str
    groups: tuple[int, ...]
    ends: tuple[str, ...]


@dataclass(frozen=True)
class Prepared:
    """The prepared run: what the runtime is lowered from, and what its printout shows."""

    strategy: str
    run: str
    target: Target
    manifest: tuple[ManifestEntry, ...]
    groups: tuple[Group, ...]
    admissions: tuple[Admission, ...]
    leaving: tuple[tuple[str, str], ...]  # (participant, the device its storage leaves from)
    loaded: Loaded = field(compare=False, repr=False)

    def group_of(self, call: str) -> Group:
        return next(g for g in self.groups if call in g.calls)

    def text(self) -> str:
        """The readable prepared form, saved with the run for inspection."""
        lines = [f"prepared @{self.strategy}", f"  run {self.run}", f"  target {self.target.describe()}", "implementations"]
        lines += [f"  @{m.name} v{m.version} digest {m.digest} code {m.code}" for m in self.manifest]
        lines.append("groups")
        for g in self.groups:
            names = ", ".join(f"@{name}" for name in g.implementations)
            settings = [f"grad {g.context.grad}"]
            settings += [f"mode {g.context.module_mode}"] if g.context.module_mode else []
            settings += [f"precision {g.context.precision}"] if g.context.precision else []
            settings += [f"random {g.context.random}"] if g.context.random else []
            settings += ["sync"] if g.sync else []
            lines.append(f"  {g.index} on {g.device} in {g.activity}: {names} ({', '.join(settings)})")
            lines += [f"      {call}" for call in g.calls]
        lines.append("admissions")
        for a in self.admissions:
            covered = ", ".join(str(g) for g in a.groups)
            lines.append(f"  {a.device} from {a.begins}: groups {covered}; ends {', '.join(a.ends)}")
        if self.leaving:
            lines.append("leaving the device")
            lines += [f"  @{participant} from {device}" for participant, device in self.leaving]
        return "\n".join(lines) + "\n"


def code_identity(implementation: object) -> str:
    """The identity of an implementation's code.

    It is the source of the module that defines the function, or of the
    object's class, which changes with any edit there; else the compiled
    code. Code it cannot identify gets an identity of this one object, so a
    cache keyed on it is never reused in another process.
    """
    function = getattr(implementation, "function", implementation)
    while True:
        if isinstance(function, functools.partial):
            function = function.func
        elif inspect.ismethod(function):
            function = function.__func__
        else:
            break
    data: bytes | None = None
    for candidate in (function, type(function)):
        try:
            source = inspect.getsourcefile(candidate)  # ty: ignore[invalid-argument-type]
            data = Path(source).read_bytes() if source else None
        except TypeError, OSError:
            data = None
        if data is not None:
            break
    if data is None:
        code = getattr(function, "__code__", None)
        data = marshal.dumps(code) if code is not None else f"unknown {id(function)}".encode()
    return hashlib.blake2s(data, digest_size=8).hexdigest()


def prepare(loaded: Loaded, installed: Mapping[str, Implementation], target: Target) -> Prepared:
    """The prepared run, or a rejection naming every problem."""
    check_implementations(loaded.run, installed)
    return _Preparation(loaded, installed, target).derive()


def load_prepared(saved: str, loaded: Loaded, installed: Mapping[str, Implementation], target: Target) -> Prepared:
    """A saved prepared form, accepted only when preparing the run again gives exactly the same form."""
    prepared = prepare(loaded, installed, target)
    if prepared.text() != saved:
        problems = Problems(Stage.PREPARE)
        problems.add(
            "iro-prep-004",
            "the saved prepared form does not follow from the run, its implementations and the target",
            fix="prepare the run again; a prepared form is derived, never edited",
        )
        problems.reject_if_any()
    return prepared


class _Preparation:
    def __init__(self, loaded: Loaded, installed: Mapping[str, Implementation], target: Target):
        self.loaded, self.installed, self.target = loaded, installed, target
        self.strategy: StrategyOp = loaded.normalized.strategy
        self.problems = Problems(Stage.PREPARE)
        self.placement = dict(target.placement)
        activities = {sym_name(op) for op in self.strategy.walk() if _is_activity(op)}
        for activity, device in target.placement:
            if activity not in activities:
                self.problems.add(
                    "iro-prep-007",
                    f"the target places @{activity} on {device}, and the strategy has no activity @{activity}",
                    names=(f"@{activity}",),
                    fix="place an activity the strategy names",
                )

    def derive(self) -> Prepared:
        groups = self._groups()
        admissions = self._admissions(groups)
        leaving = self._leaving()
        self.problems.reject_if_any()
        run = self.loaded.run
        manifest = tuple(
            ManifestEntry(name, version, digest, code_identity(self.installed[name])) for name, version, digest in run.implementations
        )
        return Prepared(self.strategy.sym_name.data, run.identity, self.target, manifest, groups, admissions, leaving, self.loaded)

    # Matching and placement.

    def _device(self, call: CallOp) -> str:
        """Where a call runs: the host, unless it declares device use."""
        declared = self.installed[symbol(call.implementation)].declaration
        if Effect.DEVICE not in declared.effects:
            return HOST.name
        current = call.parent_op()
        while current is not None:
            if _is_activity(current) and sym_name(current) in self.placement:
                return self.placement[sym_name(current) or ""]
            current = current.parent_op()
        device = self.target.device_work
        if device is None:
            self.problems.add(
                "iro-prep-002",
                f"@{declared.name} uses a device, and the target has none",
                kind=Kind.TARGET_LIMIT,
                path=path(call),
                fix="prepare for a target with a device",
            )
            return HOST.name
        return device

    def _match(self, call: CallOp) -> None:
        declared = self.installed[symbol(call.implementation)].declaration
        inputs = tuple(written(v.type) for v in call.operands)
        outputs = tuple(written(r.type) for r in call.results)
        if (inputs, outputs) != (declared.inputs, declared.outputs):
            self.problems.add(
                "iro-prep-001",
                f"the call passes ({', '.join(inputs)}) -> ({', '.join(outputs)}), and @{declared.name} "
                f"v{declared.version} declares ({', '.join(declared.inputs)}) -> ({', '.join(declared.outputs)})",
                names=(f"@{declared.name}",),
                path=path(call),
                fix="call the implementation as it is declared, or pin a version that declares this call",
            )
        if Effect.COLLECTIVES in declared.effects and self.loaded.run.replicated:
            self.problems.add(
                "iro-prep-003",
                f"@{declared.name} issues collectives, and this target does not yet order collectives across ranks",
                kind=Kind.TARGET_LIMIT,
                names=(f"@{declared.name}",),
                path=path(call),
            )

    def _fusable(self, call: CallOp) -> bool:
        """Whether a call may share a submission.

        On the host a call's return is its completion. A call that updates
        owner state stands alone, so each update is observed after the call
        that made it.
        """
        declared = self.installed[symbol(call.implementation)].declaration
        return (
            self.target.fuse
            and declared.completion == "device"
            and not declared.updates
            and not declared.effects & {Effect.EXTERNAL, Effect.COLLECTIVES}
        )

    # Fusion.

    def _groups(self) -> tuple[Group, ...]:
        groups: list[Group] = []
        for op in self.strategy.walk():
            for region in op.regions:
                for block in region.blocks:
                    current: list[CallOp] = []
                    device = ""
                    for item in block.ops:
                        if isinstance(item, ConstantOp):
                            continue
                        if not isinstance(item, CallOp):
                            self._close(groups, current, device)
                            current = []
                            continue
                        self._match(item)
                        placed = self._device(item)
                        if current and (placed != device or not self._fusable(item) or not self._fusable(current[-1])):
                            self._close(groups, current, device)
                            current = []
                        current.append(item)
                        device = placed
                    self._close(groups, current, device)
        return tuple(groups)

    def _close(self, groups: list[Group], calls: list[CallOp], device: str) -> None:
        if not calls:
            return
        effects = set().union(*(self.installed[symbol(c.implementation)].declaration.effects for c in calls))
        context = CallContext(random=_activity(calls[0]) if Effect.RANDOMNESS in effects else None)
        sync = device != HOST.name and not all(self._stays(r, device) for c in calls for r in c.results)
        groups.append(
            Group(
                len(groups) + 1,
                device,
                _activity(calls[0]),
                tuple(path(c) for c in calls),
                tuple(symbol(c.implementation) for c in calls),
                context,
                sync,
            )
        )

    def _stays(self, value: SSAValue, device: str, seen: set[SSAValue] | None = None) -> bool:
        """Whether a result is only used by device work on the same device, or released.

        A value a body yields goes on as its operation's result and, from a
        repeat's body, also as the carried entry of the next iteration.
        """
        seen = set() if seen is None else seen
        if value in seen:
            return True
        seen.add(value)
        for use in value.uses:
            user = use.operation
            if isinstance(user, ReleaseOp):
                continue
            if isinstance(user, CallOp):
                if self._device(user) != device:
                    return False
                continue
            if isinstance(user, YieldOp):
                parent = user.parent_op()
                if parent is None or use.index >= len(parent.results) or not self._stays(parent.results[use.index], device, seen):
                    return False
                carried = _carried_entry(parent, user, use.index)
                if carried is not None and not self._stays(carried, device, seen):
                    return False
                continue
            return False
        return True

    # Leaving the device (D15).

    def _leaving(self) -> tuple[tuple[str, str], ...]:
        """Each participant permitted to leave the device, with the device its storage leaves from.

        The first realization keeps one exact host copy, so the storage must
        never change: a participant the strategy updates cannot leave. Its
        storage lives on one device, and that device's adapter must offer
        storage copies.
        """
        permitted = [op for op in self.strategy.walk() if isinstance(op, ResourceOp) and op.may_leave_device is not None]
        if not permitted:
            return ()
        updated: set[str] = set()
        for access in (op for op in self.strategy.walk() if isinstance(op, AccessOp)):
            for use in ResourceSemantics().describe(access).uses:
                if use.mode in (AccessMode.OWNER_UPDATE, AccessMode.GRANT) and (name := _resource_of(use.subject)) is not None:
                    updated.add(name)
        devices: dict[str, set[str]] = {}
        for call in (op for op in self.strategy.walk() if isinstance(op, CallOp)):
            device = self._device(call)
            if device == HOST.name:
                continue
            for operand in call.operands:
                if isinstance(operand.type, LeaseType) and (name := _leased(operand)) is not None:
                    devices.setdefault(name, set()).add(device)
        facts = {d.name: d for d in self.target.devices}
        leaving = []
        for op in permitted:
            name = op.sym_name.data
            if name in updated:
                self.problems.add(
                    "iro-prep-011",
                    f"@{name} may leave the device, and the strategy updates it, so the exact copy it leaves would go stale",
                    kind=Kind.TARGET_LIMIT,
                    path=path(op),
                    fix="permit only frozen participants to leave the device",
                )
                continue
            used = sorted(devices.get(name, ()))
            if not used:
                continue  # no device work uses it, so it never occupies a device
            if len(used) > 1:
                self.problems.add(
                    "iro-prep-012",
                    f"@{name} may leave the device, and device work on {' and '.join(used)} uses it; leaving is realized from one device",
                    kind=Kind.TARGET_LIMIT,
                    path=path(op),
                    fix="place the device work that uses it on one device",
                )
                continue
            if "copies" not in facts[used[0]].offers:
                self.problems.add(
                    "iro-prep-013",
                    f"@{name} may leave the device, and {used[0]}'s adapter does not offer storage copies",
                    kind=Kind.TARGET_LIMIT,
                    path=path(op),
                    fix="prepare for a device whose adapter copies storage, or keep the participant on the device",
                )
                continue
            leaving.append((name, used[0]))
        return tuple(sorted(leaving))

    # Device admissions.

    def _admissions(self, groups: tuple[Group, ...]) -> tuple[Admission, ...]:
        calls = {path(op): op for op in self.strategy.walk() if isinstance(op, CallOp)}
        bodies: dict[str, tuple[str, Operation, list[int]]] = {}
        order: list[str] = []
        for group in groups:
            if group.device == HOST.name:
                continue
            first = calls[group.calls[0]]
            access = _enclosing_access(first)
            key = path(access) if access is not None else f"group {group.index}"
            if key not in bodies:
                bodies[key] = (group.device, access if access is not None else first, [])
                order.append(key)
            device, _, covered = bodies[key]
            if device != group.device:
                self.problems.add(
                    "iro-prep-005",
                    f"one access body holds device work for {device} and {group.device}",
                    kind=Kind.TARGET_LIMIT,
                    path=key,
                    fix="place the activity's device work on one device",
                )
            covered.append(group.index)
        admissions = []
        for key in order:
            device, scope, covered = bodies[key]
            if isinstance(scope, AccessOp):
                self._takes_after_device_work(scope)
                results = [r for r in scope.results if isinstance(r.type, RetainedType)]
            else:
                in_group = [calls[c] for c in groups[covered[0] - 1].calls]
                results = [r for c in in_group for r in c.results if isinstance(r.type, RetainedType)]
            ends = _releases(results)
            if results:
                self._takes_while_retained(scope, ends)
            self._no_requests_in_admission(scope, ends)
            if results and not ends:
                self.problems.add(
                    "iro-prep-008",
                    "this device work keeps a retained result whose release preparation cannot find",
                    path=key,
                    fix="normalize the strategy again; normalization places every release",
                )
            admissions.append(Admission(device, path(scope), tuple(covered), ends or ("at the end of its work",)))
        return tuple(admissions)

    def _held_after(self, scope: Operation, ends: tuple[str, ...]) -> list[Operation]:
        """The statements after the admission's own work during which its retained results stay live."""
        block = scope.parent_block()
        if block is None:
            return []
        ops = list(block.ops)
        start = ops.index(scope)
        positions = [start]
        for op in self.strategy.walk():
            if path(op) in ends:
                current: Operation | None = op
                while current is not None and current.parent_block() is not block:
                    current = current.parent_op()
                if current is not None:
                    positions.append(ops.index(current))
        return ops[start + 1 : max(positions) + 1]

    def _no_requests_in_admission(self, scope: Operation, ends: tuple[str, ...]) -> None:
        """A request may wait for handler capacity, so it is never made while an admission holds the device (D14)."""
        inside = [op for op in scope.walk() if op is not scope] if isinstance(scope, AccessOp) else []
        for statement in [*inside, *self._held_after(scope, ends)]:
            for inner in statement.walk():
                if isinstance(inner, RequestOp):
                    self.problems.add(
                        "iro-prep-010",
                        "this request is made while a device admission is held, and it may wait for its handler's capacity",
                        kind=Kind.TARGET_LIMIT,
                        path=path(inner),
                        fix="request after the device work and the release of its retained results",
                    )

    def _takes_while_retained(self, scope: Operation, ends: tuple[str, ...]) -> None:
        """An admission lasts until its retained results are released, so no take may come before that (D15)."""
        for statement in self._held_after(scope, ends):
            for inner in statement.walk():
                if isinstance(inner, TakeOp):
                    self.problems.add(
                        "iro-prep-009",
                        "this take waits while a device-holding result of an earlier access is still live",
                        kind=Kind.TARGET_LIMIT,
                        path=path(inner),
                        fix="take the input before the access, or end the retained result's use first",
                    )

    def _takes_after_device_work(self, access: AccessOp) -> None:
        """Inputs come before the device (D14): no take after device work in one admission."""
        device_seen = False
        for op in access.walk():
            if op is access or _enclosing_access(op) is not access:
                continue  # an activity with a lifetime of its own takes and admits for itself
            if isinstance(op, CallOp) and self._device(op) != HOST.name:
                device_seen = True
            elif isinstance(op, TakeOp) and device_seen:
                self.problems.add(
                    "iro-prep-006",
                    "this access body takes an input after its device work began, so it would wait while holding the device",
                    kind=Kind.TARGET_LIMIT,
                    path=path(op),
                    fix="take the input before the access",
                )


def _is_activity(op: Operation) -> bool:
    """A named operation with a lifecycle protocol: what a target can place."""
    return sym_name(op) is not None and op.get_trait(RegionSemantics) is not None


def _activity(op: Operation) -> str:
    """The path of the nearest enclosing named activity, or of the strategy."""
    current = op.parent_op()
    while current is not None:
        if _is_activity(current) or isinstance(current, StrategyOp):
            return path(current)
        current = current.parent_op()
    return ""


def _carried_entry(parent: Operation, yielded: YieldOp, index: int) -> SSAValue | None:
    """The entry of a repeat's next iteration that a yielded value becomes, if the body is a repeat's."""
    semantics = parent.get_trait(RegionSemantics)
    block = yielded.parent_block()
    if semantics is None or semantics.protocol != LifecycleProtocol.REPEAT or block is None:
        return None
    role = semantics.describe(parent).roles[0]
    captured = {capture.argument for capture in role.captures}
    supplied = [i for i in range(len(role.entries)) if i not in captured]
    # The protocol supplies the iteration index, then the carried values in yield order.
    position = 1 + index
    return block.args[supplied[position]] if position < len(supplied) else None


def _releases(results: Sequence[SSAValue]) -> tuple[str, ...]:
    """Where retained results are released, following results a body yields out to its operation.

    Normalization releases dependents first, so a result's own release is its
    last point; a retained result passed into another call is released before it.
    """
    found: set[str] = set()
    pending = list(results)
    while pending:
        value = pending.pop()
        for use in value.uses:
            user = use.operation
            if isinstance(user, ReleaseOp):
                found.add(path(user))
            elif isinstance(user, YieldOp) and (parent := user.parent_op()) is not None and use.index < len(parent.results):
                pending.append(parent.results[use.index])
    return tuple(sorted(found))


def _resource_of(subject: SSAValue) -> str | None:
    """The resource an access names, following captures outward."""
    owner = provider(subject).owner
    return owner.sym_name.data if isinstance(owner, ResourceOp) else None


def _leased(value: SSAValue) -> str | None:
    """The resource a lease reaches: its access's use at the lease's position."""
    value = provider(value)
    if not isinstance(value, BlockArgument):
        return None
    access = value.owner.parent_op()
    if not isinstance(access, AccessOp):
        return None
    uses = ResourceSemantics().describe(access).uses
    return _resource_of(uses[value.index].subject) if value.index < len(uses) else None


def _enclosing_access(op: Operation) -> AccessOp | None:
    """The access whose body holds the op, unless an activity with its own lifetime lies between them."""
    current = op.parent_op()
    while current is not None:
        if isinstance(current, AccessOp):
            return current
        semantics = current.get_trait(RegionSemantics)
        if semantics is not None and semantics.protocol in _OWN_LIFETIME:
            return None
        current = current.parent_op()
    return None


__all__ = ["Admission", "Group", "ManifestEntry", "Prepared", "code_identity", "load_prepared", "prepare"]
