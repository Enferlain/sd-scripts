"""Scope records: which statements started, and which are still live, in each scope (D14).

A wait for a source to settle in a scope it shares with the waiter reads
them. Lowering keeps records only in the scopes some wait asks about; checked
mode keeps every invocation's records beside them to compare.
"""

from collections import defaultdict

from library.iro.runtime.state import Identity


class Records:
    """Which statements started, and which are still live, in each scope: what a wait for a source to settle reads."""

    def __init__(self) -> None:
        self.started: dict[tuple[str, Identity], int] = {}
        self.live: dict[tuple[str, Identity], int] = defaultdict(int)
        self.by_scope: dict[Identity, set[tuple[str, Identity]]] = defaultdict(set)

    def begin(self, owner: str, scopes: list[Identity]) -> None:
        for scope in scopes:
            key = (owner, scope)
            self.started[key] = 1
            self.by_scope[scope].add(key)
            self.live[key] += 1

    def retire(self, owner: str, identity: Identity, scopes: list[Identity]) -> bool:
        """Drops a finished invocation; true when a scope enclosing it has nothing of its statement left live."""
        settled = False
        for scope in scopes:
            key = (owner, scope)
            self.live[key] -= 1
            if not self.live[key]:
                del self.live[key]
                # A waiter shares a scope that encloses the invocation; its own scope ending (every
                # iteration of a repeat) settles nothing anyone waits for.
                settled = settled or scope != identity
        return settled

    def close(self, scope: Identity) -> None:
        """The scope's invocation ended: the records kept in it go."""
        if self.by_scope:
            for key in self.by_scope.pop(scope, ()):
                self.started.pop(key, None)

    def empty(self) -> bool:
        return not (self.started or self.live or self.by_scope)

    def size(self) -> int:
        return len(self.started) + len(self.live) + len(self.by_scope)

    def settled(self, chain: tuple[str, ...], scope: Identity) -> bool:
        started = False
        for owner in chain:
            if self.live.get((owner, scope)):
                return False
            started = started or (owner, scope) in self.started
        return started


def scopes(identity: Identity, commons: frozenset[str] | None = None) -> list[Identity]:
    """The invocations enclosing an identity, itself included: the scopes a source can settle in.

    A parallel branch is not an invocation, so it is never a scope. With
    ``commons``, only the scopes a wait can ask about: the outermost, and
    those of the statements a waiter shares with this one.
    """
    return [
        identity[:length]
        for length in range(1, len(identity) + 1)
        if "#" not in identity[length - 1][0] and (commons is None or length == 1 or identity[length - 1][0] in commons)
    ]


def scope(common: str, identity: Identity) -> Identity:
    """The invocation of ``common`` that encloses ``identity``: the innermost, else the outermost scope."""
    for length in range(len(identity), 0, -1):
        if identity[length - 1][0] == common:
            return identity[:length]
    return identity[:1]


__all__ = ["Records", "scope", "scopes"]
