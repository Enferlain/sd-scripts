"""Whether a runtime trace is a trace of the reference semantics (D16).

Given the facts a run of the runtime reported, it searches for a schedule of
the reference semantics under which every owner observes exactly the same
facts in the same order: the same invocations, values and versions. The
reference cannot be copied, so the search backtracks by replaying a prefix
of choices. It prefers the task whose next command belongs to an invocation
some owner expects next, which keeps backtracking rare.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from library.iro.reference import Reference
from tests.unit.iro.runtime.harness import RUNTIME_ONLY, comparable


@dataclass
class Verdict:
    explained: bool
    replays: int
    steps: int
    detail: str = ""


def explains(facts, make: Callable[[], Reference], *, max_replays: int = 5000) -> Verdict:
    """Searches for a reference schedule that reproduces every owner's facts in order."""
    target: dict[str, list[tuple]] = defaultdict(list)
    position: dict[tuple, int] = {}
    for index, fact in enumerate(f for f in facts if f.kind not in RUNTIME_ONLY):
        item = comparable(fact)
        position[(item[0], len(target[item[0]]))] = index
        target[item[0]].append(item)

    def replay(prefix: list[int]) -> tuple[Reference, dict[str, int]]:
        reference = make()
        matched: dict[str, int] = defaultdict(int)
        assert _consume(reference.facts, target, matched), "the reference's opening facts differ"
        seen = len(reference.facts)
        for choice in prefix:
            reference.step(choice)
            assert _consume(reference.facts[seen:], target, matched), "a replayed prefix diverged"
            seen = len(reference.facts)
        return reference, matched

    choices: list[int] = []
    tried: dict[int, set[int]] = defaultdict(set)
    reference, matched = replay(choices)
    replays = steps = 0
    detail = ""
    while True:
        depth = len(choices)
        enabled = reference.enabled()
        if not enabled and all(matched[owner] == len(items) for owner, items in target.items()):
            return Verdict(True, replays, steps)
        candidates = [t for t in _ranked(reference, enabled, target, matched, position) if t not in tried[depth]]
        if candidates:
            choice = candidates[0]
            tried[depth].add(choice)
            seen = len(reference.facts)
            reference.step(choice)
            steps += 1
            if _consume(reference.facts[seen:], target, matched):
                choices.append(choice)
                tried.pop(depth + 1, None)
                continue
            detail = f"step {steps}: a step reported facts no owner expected next"
        elif not enabled:
            detail = "the reference settled with facts left unmatched"
        else:
            detail = "every enabled task was tried here"
            if depth == 0:
                return Verdict(False, replays, steps, "no reference schedule gives this trace")
            tried.pop(depth, None)
            choices.pop()
        if replays >= max_replays:
            return Verdict(False, replays, steps, f"search budget spent, inconclusive; last: {detail}")
        replays += 1
        reference, matched = replay(choices)


def _consume(new, target: dict[str, list[tuple]], matched: dict[str, int]) -> bool:
    for fact in new:
        item = comparable(fact)
        owner = item[0]
        index = matched[owner]
        if index >= len(target[owner]) or target[owner][index] != item:
            return False
        matched[owner] = index + 1
    return True


def _ranked(reference: Reference, enabled, target, matched, position) -> list[int]:
    expected: dict[tuple, int] = {}
    for owner, items in target.items():
        index = matched[owner]
        if index < len(items):
            expected.setdefault(items[index][1], position[(owner, index)])

    def score(task: int) -> tuple[int, int]:
        command = reference.tasks[task].command
        invocation = command.invocation if command is not None else ()
        best = min(
            (p for inv, p in expected.items() if inv[: len(invocation)] == invocation or invocation[: len(inv)] == inv), default=10**9
        )
        return (best, task)

    return sorted(enabled, key=score)


__all__ = ["Verdict", "explains"]
