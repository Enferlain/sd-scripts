"""Independent xDSL lifetime verification, preparation and cold correspondence.

The shared imports describe selected Python bindings and their existing target;
neither the Pliron parser nor its semantic/prepared verifier participates here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from typing import Any, cast

from xdsl.dialects.builtin import DictionaryAttr, ModuleOp, StringAttr
from xdsl.ir import Block, BlockArgument, OpResult, Operation, Region, SSAValue

from . import lifetime_ir as ir

_binding_api = importlib.import_module("docs_design.models-strategy-trainer.pliron_candidate.composition")
TARGET = _binding_api.TARGET
Registry = _binding_api.Registry
contracts = _binding_api.contracts
registry = _binding_api.registry
ARTIFACT_FIELDS = frozenset({"schema", "framework", "target", "accepted", "prepared", "manifest", "profile", "accepted_sha256", "nodes"})


class Rejected(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rejected(message)


def identity(op: ir.ActivityOp) -> str:
    return op.source_id.data


def policy(op: ir.ActivityOp, key: str) -> str:
    value = op.policy.data[key]
    if not isinstance(value, StringAttr):
        raise Rejected(f"{identity(op)}: {key} requires a string")
    return value.data


def expect(op: ir.ActivityOp, key: str, value: str) -> None:
    actual = policy(op, key)
    require(actual == value, f"{identity(op)}: {key}: expected {value}, provided {actual}")


def number(op: ir.ActivityOp, key: str) -> int:
    try:
        value = int(policy(op, key))
    except ValueError as error:
        raise Rejected(f"{identity(op)}: {key} requires a positive integer") from error
    require(value > 0, f"{identity(op)}: {key} requires a positive integer")
    return value


def floating(op: ir.ActivityOp, key: str) -> float:
    try:
        value = float(policy(op, key))
    except ValueError as error:
        raise Rejected(f"{identity(op)}: {key} requires a finite number") from error
    require(math.isfinite(value), f"{identity(op)}: {key} requires a finite number")
    return value


def resolved(value: SSAValue) -> SSAValue:
    """Map explicit ordered-region arguments to their declared activity inputs."""
    seen = set()
    while isinstance(value, BlockArgument):
        require(value not in seen, "cyclic activity argument aliases")
        seen.add(value)
        parent = value.owner.parent_op()
        require(isinstance(parent, ir.RepeatOp | ir.EvaluatorOp), "argument outside an accepted ordered activity")
        if not isinstance(parent, ir.RepeatOp | ir.EvaluatorOp) or value.index >= len(parent.operands):
            raise Rejected("activity argument has no corresponding input")
        value = parent.operands[value.index]
    return value


def defining[T: ir.ActivityOp](value: SSAValue, cls: type[T]) -> T:
    value = resolved(value)
    if not isinstance(value, OpResult) or not isinstance(value.owner, cls):
        raise Rejected(f"expected {cls.name} endpoint")
    return value.owner


def same(actual: SSAValue, expected: SSAValue, source: ir.ActivityOp, meaning: str) -> None:
    require(resolved(actual) is resolved(expected), f"{identity(source)}: {meaning} association differs")


def body(op: ir.ActivityOp, allowed: tuple[type[ir.ActivityOp], ...], end: type[ir.ActivityOp]) -> tuple[ir.ActivityOp, ...]:
    operations = tuple(op.regions[0].block.ops)
    require(bool(operations) and isinstance(operations[-1], end), f"{identity(op)}: missing final {end.name}")
    require(all(isinstance(child, allowed) for child in operations), f"{identity(op)}: unsupported nested activity")
    return cast(tuple[ir.ActivityOp, ...], operations)


def verify(module: ModuleOp) -> tuple[ir.RunOp, tuple[ir.ActivityOp, ...]]:
    module.verify()
    roots = tuple(module.body.block.ops)
    require(not module.attributes and not module.properties, "lifetime module has unsupported metadata")
    require(len(roots) == 1 and isinstance(roots[0], ir.RunOp), "expected one life.run document")
    root = roots[0]
    if not isinstance(root, ir.RunOp):
        raise Rejected("expected life.run")
    operations = tuple(root.walk())
    require(all(type(op) in ir.SEMANTIC for op in operations), "unsupported operation in semantic lifetime tree")
    sources: set[str] = set()
    owned = set(operations)
    for op in operations:
        if not isinstance(op, ir.ActivityOp):
            raise Rejected("missing lifetime activity interface")
        require(identity(op) not in sources, f"{identity(op)}: duplicate source identity")
        sources.add(identity(op))
        contract = op.get_trait(ir.ActivityContract)
        if contract is None:
            raise Rejected(f"{identity(op)}: missing activity contract")
        if contract.region != "none":
            block = op.regions[0].block
            expected = tuple(v.type for v in op.operands) if contract.region == "ordered" else ()
            require(tuple(v.type for v in block.args) == expected, f"{identity(op)}: region arguments differ")
        for value in op.operands:
            value = resolved(value)
            require(isinstance(value, OpResult) and value.owner in owned, f"{identity(op)}: endpoint outside accepted tree")
    expect(root, "version", "1")
    expect(root, "failure", "cancel_siblings_and_drain")
    body(root, (ir.ModelOp, ir.ProducerOp, ir.GroupOp, ir.FinishOp), ir.FinishOp)
    for op in operations:
        if not isinstance(op, ir.ActivityOp):
            raise Rejected("unrecognized lifetime operation")
        parent = op.parent_op()
        if isinstance(op, ir.RunOp):
            require(op is root, "nested runs are unsupported")
        elif isinstance(op, ir.ModelOp):
            require(parent is root, f"{identity(op)}: model must be run-owned")
            require(policy(op, "treatment") in ("frozen", "trainable"), f"{identity(op)}: unsupported model treatment")
            floating(op, "initial")
        elif isinstance(op, ir.ProducerOp):
            require(parent is root, f"{identity(op)}: producer must be run-owned")
            defining(op.model, ir.ModelOp)
            stop = defining(op.stop, ir.GroupOp)
            require(stop.parent_op() is root, f"{identity(op)}: stop must address sibling consumer group")
            for consumer in operations:
                if isinstance(consumer, ir.RepeatOp | ir.EvaluatorOp) and resolved(consumer.records) is op.records:
                    ancestor = consumer.parent_op()
                    while ancestor is not None and ancestor is not stop:
                        ancestor = ancestor.parent_op()
                    require(ancestor is stop, f"{identity(op)}: stop boundary leaves {identity(consumer)} active")
            for key in ("horizon", "ready", "workers"):
                number(op, key)
            expect(op, "remaining", "discard_unclaimed")
            expect(op, "failure", "close_then_drain")
        elif isinstance(op, ir.GroupOp):
            require(isinstance(parent, ir.RunOp | ir.GroupOp), f"{identity(op)}: group must be enclosed by run/group")
            expect(op, "completion", "children_physically_settled")
            expect(op, "pending", "include_requested")
            expect(op, "failure", "cancel_siblings_and_drain")
            body(op, (ir.RepeatOp, ir.RequestOp, ir.EvaluatorOp, ir.GroupOp, ir.SettleOp), ir.SettleOp)
        elif isinstance(op, ir.RepeatOp):
            require(isinstance(parent, ir.GroupOp), f"{identity(op)}: repeat requires consumer group")
            defining(op.records, ir.ProducerOp)
            expect(defining(op.model, ir.ModelOp), "treatment", "trainable")
            number(op, "max_windows")
            expect(op, "admission", "policy_choice_before_first_claim")
            expect(op, "stop", "finish_selected_window")
            nested = body(op, (ir.TakeOp, ir.ContributeOp, ir.AdvanceOp, ir.WindowEndOp), ir.WindowEndOp)
            count = sum(isinstance(child, ir.TakeOp) for child in nested)
            require(count > 0 and len(nested) == 2 * count + 2, f"{identity(op)}: incomplete contribution window")
            previous = None
            for slot in range(count):
                take, contribution = nested[2 * slot : 2 * slot + 2]
                require(
                    isinstance(take, ir.TakeOp) and isinstance(contribution, ir.ContributeOp),
                    f"{identity(op)}: ordered take/contribution required",
                )
                if not isinstance(take, ir.TakeOp) or not isinstance(contribution, ir.ContributeOp):
                    raise Rejected("invalid ordered contribution pair")
                expect(take, "selection", f"slot:{slot}")
                expect(take, "provenance", "actual_frozen_source")
                same(take.records, op.records, take, "stream")
                same(contribution.sample, take.sample, contribution, "sample")
                same(contribution.model, op.model, contribution, "model")
                require(contribution.previous is previous, f"{identity(contribution)}: complete gradient chain differs")
                require(number(contribution, "normalization") == count, f"{identity(contribution)}: normalization differs from cardinality")
                previous = contribution.gradient
            advance, end = nested[-2:]
            require(
                isinstance(advance, ir.AdvanceOp) and isinstance(end, ir.WindowEndOp), f"{identity(op)}: advance/window completion required"
            )
            if not isinstance(advance, ir.AdvanceOp) or not isinstance(end, ir.WindowEndOp):
                raise Rejected("invalid window completion")
            require(advance.gradient is previous, f"{identity(advance)}: advancement omits contribution")
            same(advance.model, op.model, advance, "model")
            same(end.completion, advance.completion, end, "completion")
            expect(advance, "authority", "ordinary_owner")
            require(floating(advance, "learning_rate") > 0, f"{identity(advance)}: positive learning rate required")
        elif isinstance(op, ir.RequestOp):
            source = defining(op.events, ir.RepeatOp)
            require(
                isinstance(parent, ir.GroupOp) and source.parent_op() is parent, f"{identity(op)}: request source outside consumer group"
            )
            require(number(op, "after_window") <= number(source, "max_windows"), f"{identity(op)}: request outside repeat bounds")
            for key, value in (
                ("multiplicity", "once_per_matching_completion"),
                ("pending", "keeps_group_alive"),
                ("max_pending", "1"),
                ("cancellation", "cancel_or_unfulfilled"),
            ):
                expect(op, key, value)
            require(
                sum(isinstance(n, ir.EvaluatorOp) and n.due is op.due for n in operations) == 1,
                f"{identity(op)}: request requires exactly one recipient",
            )
        elif isinstance(op, ir.EvaluatorOp):
            request = defining(op.due, ir.RequestOp)
            training = defining(request.events, ir.RepeatOp)
            require(request.parent_op() is parent, f"{identity(op)}: request recipient outside group")
            same(op.records, training.records, op, "training/evaluation stream")
            same(op.model, training.model, op, "training/evaluation model")
            expect(op, "completion", "physical_and_cleanup")
            nested = body(op, (ir.TakeOp, ir.ReadOp, ir.EvaluationEndOp), ir.EvaluationEndOp)
            require(
                len(nested) == 3 and isinstance(nested[0], ir.TakeOp) and isinstance(nested[1], ir.ReadOp),
                f"{identity(op)}: one take/read evaluation required",
            )
            take, read, end = nested
            if not isinstance(take, ir.TakeOp) or not isinstance(read, ir.ReadOp) or not isinstance(end, ir.EvaluationEndOp):
                raise Rejected("invalid evaluation value chain")
            expect(take, "selection", "after_trigger_members")
            expect(take, "provenance", "actual_frozen_source")
            same(take.records, op.records, take, "stream")
            same(read.sample, take.sample, read, "sample")
            same(read.model, op.model, read, "model")
            same(end.measurement, read.measurement, end, "measurement")
            for key, value in (("state", "current_at_start"), ("mode", "evaluation"), ("completion", "physical_validate_restore")):
                expect(read, key, value)
        elif isinstance(op, ir.SettleOp | ir.FinishOp):
            require(
                isinstance(parent, ir.RunOp if isinstance(op, ir.FinishOp) else ir.GroupOp),
                f"{identity(op)}: settlement belongs to wrong boundary",
            )
            if parent is None:
                raise Rejected("missing settlement parent")
            expected = {
                child.done
                for child in parent.regions[0].block.ops
                if isinstance(child, ir.ProducerOp | ir.RepeatOp | ir.GroupOp | ir.EvaluatorOp)
            }
            actual = tuple(resolved(value) for value in op.operands)
            require(
                bool(expected) and len(actual) == len(expected) and set(actual) == expected,
                f"{identity(op)}: settlement must cover all and only direct child obligations",
            )
        else:
            require(isinstance(parent, ir.RepeatOp | ir.EvaluatorOp), f"{identity(op)}: local operation outside activity")
    return root, cast(tuple[ir.ActivityOp, ...], operations)


def binding_requirement(op: ir.ActivityOp) -> tuple[str, str, str] | None:
    return {
        ir.ModelOp: ("model", "load", "before_install"),
        ir.ProducerOp: ("produce", "read_source", "physical_return_before_release"),
        ir.ContributeOp: ("objective", "build_loss", "return_before_backward"),
        ir.ReadOp: ("evaluate", "protected_read", "physical_validate_restore"),
    }.get(type(op))


def check_manifest(operations: tuple[ir.ActivityOp, ...], manifest: Mapping[str, Any]) -> dict[str, Any]:
    require(set(manifest) == {"schema", "target", "bindings"}, "unsupported composition manifest fields")
    require(
        type(manifest.get("schema")) is int and manifest["schema"] == 1 and manifest.get("target") == TARGET,
        "unsupported composition manifest/target",
    )
    bindings = manifest.get("bindings")
    require(isinstance(bindings, Mapping), "binding manifest requires a mapping")
    if not isinstance(bindings, Mapping):
        raise Rejected("missing installed binding mapping")
    for symbol, descriptor in bindings.items():
        require(isinstance(symbol, str) and bool(symbol) and isinstance(descriptor, Mapping), "invalid binding declaration")
        require(
            set(descriptor) == {"version", "role", "effect", "completion"} and all(isinstance(value, str) for value in descriptor.values()),
            f"@{symbol}: malformed binding contract",
        )
    for op in operations:
        wanted = binding_requirement(op)
        if wanted is None:
            continue
        symbol = policy(op, "implementation")
        role, effect, completion = wanted
        expected = {"version": "1", "role": role, "effect": effect, "completion": completion}
        require(bindings.get(symbol) == expected, f"{identity(op)}: @{symbol}: expected {expected}, provided {bindings.get(symbol)}")
    return json.loads(json.dumps(manifest))


def target_profile(root: ir.RunOp, operations: tuple[ir.ActivityOp, ...]) -> dict[str, Any]:
    def one[T: ir.ActivityOp](cls: type[T]) -> T:
        found = [op for op in operations if isinstance(op, cls)]
        require(len(found) == 1, f"{TARGET}: unsupported target profile: one {cls.name} required")
        return found[0]

    producer, group, training = one(ir.ProducerOp), one(ir.GroupOp), one(ir.RepeatOp)
    evaluator, request, advance, read = one(ir.EvaluatorOp), one(ir.RequestOp), one(ir.AdvanceOp), one(ir.ReadOp)
    contributions = [op for op in training.body.block.ops if isinstance(op, ir.ContributeOp)]
    require(
        number(training, "max_windows") == 2 and len(contributions) == 2 and number(request, "after_window") == 2,
        f"{TARGET}: unsupported target window/request bounds",
    )
    require(
        (number(producer, "horizon"), number(producer, "ready"), number(producer, "workers")) == (3, 2, 1),
        f"{TARGET}: unsupported aggregate/ready/physical bounds",
    )
    expect(advance, "parameter", "weight")
    encoder, learner = defining(producer.model, ir.ModelOp), defining(training.model, ir.ModelOp)
    expect(encoder, "treatment", "frozen")
    models = [op for op in operations if isinstance(op, ir.ModelOp)]
    require(len(models) == 2, f"{TARGET}: unsupported model count")
    require(len({policy(op, "implementation") for op in contributions}) == 1, f"{TARGET}: unsupported heterogeneous objectives")
    name = identity(producer)
    return {
        "run": identity(root),
        "producer": name,
        "group": identity(group),
        "training": identity(training),
        "evaluation": identity(evaluator),
        "encoder": identity(encoder),
        "learner": identity(learner),
        "horizon": number(producer, "horizon"),
        "ready": number(producer, "ready"),
        "workers": number(producer, "workers"),
        "members": len(contributions),
        "windows": number(training, "max_windows"),
        "after_window": number(request, "after_window"),
        "learning_rate": floating(advance, "learning_rate"),
        "ports": [f"{name}.slot.{slot}" for slot in range(len(contributions))],
        "evaluation_port": f"{name}.evaluation",
        "unused_port": f"{name}.unused",
        "models": [
            {
                "id": identity(op),
                "initial": floating(op, "initial"),
                "trainable": policy(op, "treatment") == "trainable",
                "implementation": policy(op, "implementation"),
            }
            for op in models
        ],
        "calls": {
            "encode": policy(producer, "implementation"),
            "objective": policy(contributions[0], "implementation"),
            "measure": policy(read, "implementation"),
        },
    }


def plan(root: ir.RunOp, operations: tuple[ir.ActivityOp, ...], manifest: Mapping[str, Any], profile: Mapping[str, Any]) -> ModuleOp:
    declarations: list[Operation] = []

    def add(role: str, **values: str) -> None:
        declarations.append(
            ir.RecipeOp(properties={"role": StringAttr(role), "values": DictionaryAttr({k: StringAttr(v) for k, v in values.items()})})
        )

    for op in operations:
        name = identity(op)
        if isinstance(op, ir.RunOp | ir.ProducerOp | ir.GroupOp | ir.RepeatOp | ir.EvaluatorOp):
            parent = op.parent_op()
            add(
                "scope",
                origin=name,
                parent=identity(parent) if isinstance(parent, ir.ActivityOp) else "",
                kind=op.name,
                **{key: value.data for key, value in op.policy.data.items() if isinstance(value, StringAttr)},
            )
        required = binding_requirement(op)
        if required is not None:
            role, effect, completion = required
            symbol = policy(op, "implementation")
            add(
                "call",
                origin=name,
                symbol=symbol,
                version=manifest["bindings"][symbol]["version"],
                binding_role=role,
                effect=effect,
                completion=completion,
            )
        access = {
            ir.ProducerOp: (0, "read_source", "physical_return"),
            ir.ContributeOp: (1, "read_gradient", "backward"),
            ir.AdvanceOp: (1, "write", "update_outcome"),
            ir.ReadOp: (1, "read_evaluation", "physical_and_restore"),
        }.get(type(op))
        if access is not None:
            operand, mode, through = access
            add("access", origin=name, participant=identity(defining(op.operands[operand], ir.ModelOp)), mode=mode, through=through)
        if isinstance(op, ir.RequestOp):
            add(
                "request",
                origin=name,
                source=identity(defining(op.events, ir.RepeatOp)),
                **{key: value.data for key, value in op.policy.data.items() if isinstance(value, StringAttr)},
            )
        if isinstance(op, ir.ProducerOp):
            add(
                "stop_link",
                source=identity(defining(op.stop, ir.GroupOp)),
                target=name,
                relation="normal_group_end_requests_stop",
                failure="enclosing_cancel_then_physical_drain",
                run_completion="producer_and_group_settled",
            )
            # This target pins two training slots before recipe construction.
            for role, channel in zip(
                ("slot:0", "slot:1", "evaluation", "unused"),
                (*profile["ports"], profile["evaluation_port"], profile["unused_port"]),
                strict=True,
            ):
                add(
                    "port",
                    producer=name,
                    port_role=role,
                    channel=channel,
                    identity="actual_record_and_source",
                    ready_budget="aggregate_provider_budget",
                    settlement="actual_use_then_notification",
                )
    prepared = ir.PlanOp(
        properties={
            "schema": StringAttr("1"),
            "target": StringAttr(TARGET),
            "manifest": StringAttr(json.dumps(manifest, sort_keys=True, separators=(",", ":"))),
        },
        regions=[Region(Block((root.clone(),))), Region(Block(declarations))],
    )
    result = ModuleOp((prepared,))
    result.verify()
    return result


class LifetimeCandidate:
    def accept(self, text: str) -> str:
        try:
            module = ir.parse(text)
            verify(module)
            canonical = ir.printed(module)
            restored = ir.parse(canonical)
            verify(restored)
            require(ir.equivalent(module, restored), "semantic structural round-trip differs")
            return canonical
        except Rejected:
            raise
        except Exception as error:
            raise Rejected(f"lifetime document: {error}") from error

    def prepare(self, accepted: str, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            module = ir.parse(accepted)
            root, operations = verify(module)
            installed = check_manifest(operations, contracts() if manifest is None else manifest)
            profile = target_profile(root, operations)
            expected = plan(root, operations, installed, profile)
            text = ir.printed(expected)
            require(ir.equivalent(expected, ir.parse(text)), "prepared structural round-trip differs")
            return {
                "schema": 1,
                "framework": "xdsl",
                "target": TARGET,
                "accepted": accepted,
                "prepared": text,
                "accepted_sha256": hashlib.sha256(accepted.encode()).hexdigest(),
                "manifest": installed,
                "profile": profile,
            }
        except Rejected:
            raise
        except Exception as error:
            raise Rejected(f"lifetime preparation: {error}") from error

    def verify_saved(self, artifact: Mapping[str, Any], installed: Mapping[str, Any]) -> dict[str, Any]:
        require(isinstance(artifact, Mapping), "saved lifetime artifact: expected a mapping")
        unknown = set(artifact) - ARTIFACT_FIELDS
        require(not unknown, f"saved lifetime artifact: unknown fields {unknown}")
        require(
            type(artifact.get("schema")) is int
            and artifact["schema"] == 1
            and artifact.get("framework") == "xdsl"
            and artifact.get("target") == TARGET,
            "unsupported saved lifetime artifact/target",
        )
        for key in ("accepted", "prepared"):
            require(
                isinstance(artifact.get(key), str) and bool(artifact[key]),
                f"saved lifetime artifact {key}: expected nonempty document text",
            )
        require(isinstance(artifact.get("manifest"), Mapping), "saved lifetime artifact manifest: expected binding mapping")
        verified = self.prepare(artifact["accepted"], installed)
        require(artifact["manifest"] == verified["manifest"], "saved binding manifest differs from installed contracts")
        try:
            decoded = ir.parse(artifact["prepared"])
            require(
                ir.equivalent(ir.parse(verified["prepared"]), decoded),
                "cold lifetime plan: semantic/source/lifetime/binding correspondence differs",
            )
        except Rejected:
            raise
        except Exception as error:
            raise Rejected(f"cold lifetime prepared document: {error}") from error
        verified["prepared"] = artifact["prepared"]
        return verified

    def materialize(self, artifact: Mapping[str, Any], selected: Registry | None = None, controls: Any = None):
        selected = registry() if selected is None else selected
        verified = self.verify_saved(artifact, selected.describe())
        runtime = importlib.import_module("docs_design.models-strategy-trainer.pliron_candidate.composition_runtime")
        return runtime.lower(verified["profile"], {name: binding.invoke for name, binding in selected.bindings.items()}, controls)
