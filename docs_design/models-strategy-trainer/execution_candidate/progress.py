"""Coordinator-only evidence for waits across protection and input capacity.

This neither schedules work nor infers private waits. Unknown wake paths stay
possible. A closed set of observed waits is rejected without releasing state,
claiming inputs or changing the accepted order.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator, Hashable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from .graph import NotReady

type Resource = tuple[str, str]
type Access = Mapping[Resource, str]
type Actor = asyncio.Task


def actor() -> Actor:
    current = asyncio.current_task()
    if current is None:
        raise NotReady("Runtime coordination requires an activity task")
    return current


@dataclass(frozen=True)
class Lease:
    owner: Actor
    access: Access


class Progress:
    def __init__(self):
        self.leases: dict[object, Lease] = {}
        self.consumers: dict[str, set[str]] = {}
        self.ended: set[str] = set()
        self.demands: dict[object, tuple[Actor, str, frozenset[str], Hashable]] = {}
        self.waits: dict[Actor, tuple[str, Literal["all", "any"], Callable[[], set[Actor] | None]]] = {}

    def blockers(self, access: Access) -> set[Actor]:
        return {
            lease.owner
            for lease in self.leases.values()
            if any(key in lease.access and (mode == "write" or lease.access[key] == "write") for key, mode in access.items())
        }

    def claim(self, access: Access) -> object:
        token = object()
        self.leases[token] = Lease(actor(), dict(access))
        return token

    @contextmanager
    def activity(self, name: str) -> Generator[None]:
        self.ended.discard(name)
        returned = False
        try:
            yield
            returned = True
        finally:
            self.ended.add(name)
            if returned:
                # An alternative future claimant may finish without making a
                # demand. Its end can close a previously unknown wake path.
                # Preserve an existing primary failure during cleanup instead.
                self.check()

    @contextmanager
    def demand(self, activity: str, channels: frozenset[str], key: Hashable) -> Generator[None]:
        token = object()
        self.demands[token] = (actor(), activity, channels, key)
        try:
            yield
        finally:
            del self.demands[token]

    def claimants(self, channel: str, ready: Mapping[Hashable, object]) -> set[Actor] | None:
        possible: set[Actor] = set()
        for consumer in self.consumers.get(channel, set()) - self.ended:
            matching = {
                task for task, name, channels, key in self.demands.values() if name == consumer and channel in channels and key in ready
            }
            if not matching:
                # A not-yet-entered/unknown demand may still free capacity.
                return None
            possible.update(matching)
        return possible or None

    @contextmanager
    def waiting(self, description: str, mode: Literal["all", "any"], dependencies: Callable[[], set[Actor] | None]) -> Generator[None]:
        task = actor()
        if task in self.waits:
            raise NotReady("One activity cannot enter two shared waits simultaneously")
        self.waits[task] = (description, mode, dependencies)
        try:
            self.check()
            yield
        finally:
            del self.waits[task]

    def check(self) -> None:
        # Actors outside observed waits may make progress. An acquisition needs
        # ALL holders; capacity needs ANY claimant. Prune possible progress to
        # avoid rejecting a cycle which another runnable claimant can break.
        remaining = set(self.waits)
        while remaining:
            possible = set()
            for task in remaining:
                _, mode, resolve = self.waits[task]
                dependencies = resolve()
                if (
                    dependencies is None
                    or not dependencies
                    or (mode == "all" and not dependencies & remaining)
                    or (mode == "any" and dependencies - remaining)
                ):
                    possible.add(task)
            if not possible:
                names = sorted(self.waits[task][0] for task in remaining)
                raise NotReady("Runtime resource wait cycle: " + ", ".join(names))
            remaining.difference_update(possible)
