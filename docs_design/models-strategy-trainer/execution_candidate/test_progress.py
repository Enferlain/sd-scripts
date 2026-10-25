"""Shared wait semantics: all holders versus any alternative claimant."""

from __future__ import annotations

import asyncio

import pytest

from .graph import NotReady
from .progress import Progress
from .test_candidate import run


@pytest.mark.parametrize("alternative", (False, True))
@pytest.mark.parametrize("transitive", (False, True))
def test_shared_waits_distinguish_closed_cycles_from_runnable_alternatives(alternative, transitive):
    async def scenario():
        progress = Progress()
        tasks = []
        entered = [asyncio.Event() for _ in range(3)]
        release = asyncio.Event()

        async def wait(index, mode, targets):
            # Entered records the attempt even when registration rejects it.
            entered[index].set()
            with progress.waiting(str(index), mode, lambda: {tasks[target] for target in targets}):
                await release.wait()

        # Create references before any task runs. The third task is initially
        # a possible outside waker; registration later makes its true wait known.
        tasks.append(asyncio.create_task(wait(0, "all", (1,))))
        if transitive:
            tasks.append(asyncio.create_task(wait(1, "all", (2,))))
            tasks.append(asyncio.create_task(wait(2, "any", (0, 3) if alternative else (0,))))
        else:
            tasks.append(asyncio.create_task(wait(1, "any", (0, 2) if alternative else (0,))))
            tasks.append(asyncio.create_task(release.wait()))
        if alternative and transitive:
            tasks.append(asyncio.create_task(release.wait()))
        try:
            await entered[1].wait()
            if transitive:
                await entered[2].wait()
            closing = tasks[2 if transitive else 1]
            if alternative:
                assert not closing.done()
                release.set()
                await asyncio.gather(*tasks)
            else:
                with pytest.raises(NotReady, match="resource wait cycle"):
                    await closing
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert not progress.waits

    run(scenario())


def test_unknown_or_future_input_claimant_is_not_proof_of_deadlock():
    async def scenario():
        progress = Progress()
        progress.consumers["port"] = {"known", "future"}
        with progress.demand("known", frozenset(("port",)), 1):
            assert progress.claimants("port", {1: object()}) is None
            with progress.activity("future"):
                pass
            assert progress.claimants("port", {1: object()}) == {asyncio.current_task()}
            assert progress.claimants("port", {2: object()}) is None
        assert not progress.demands

    run(scenario())


@pytest.mark.parametrize("mode", ("all", "any"))
def test_one_runnable_owner_is_enough_for_capacity_but_not_for_all_holder_release(mode):
    async def scenario():
        progress = Progress()
        release, started = asyncio.Event(), asyncio.Event()
        tasks = []

        async def first():
            with progress.waiting("first", mode, lambda: {tasks[1], tasks[2]}):
                started.set()
                await release.wait()

        async def second():
            with progress.waiting("second", "all", lambda: {tasks[0]}):
                await release.wait()

        tasks.extend((asyncio.create_task(first()), asyncio.create_task(second()), asyncio.create_task(release.wait())))
        try:
            await started.wait()
            if mode == "all":
                with pytest.raises(NotReady, match="resource wait cycle"):
                    await tasks[1]
            else:
                assert not tasks[1].done()
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert not progress.waits

    run(scenario())


def test_cancellation_unregisters_wait_and_demand_without_releasing_another_owner():
    async def scenario():
        progress = Progress()
        retained = progress.claim({("participant", "model"): "read"})
        started = asyncio.Event()

        async def waiting():
            own = progress.claim({("state", "private"): "write"})
            try:
                with (
                    progress.demand("consumer", frozenset(("port",)), 1),
                    progress.waiting("model", "all", lambda: progress.blockers({("participant", "model"): "write"})),
                ):
                    started.set()
                    await asyncio.Event().wait()
            finally:
                del progress.leases[own]

        task = asyncio.create_task(waiting())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not progress.waits and not progress.demands
        assert set(progress.leases) == {retained}
        del progress.leases[retained]

    run(scenario())


def test_last_unknown_claimant_ending_rechecks_existing_waits():
    async def scenario():
        progress = Progress()
        progress.consumers["port"] = {"consumer", "future"}
        started, demanded = asyncio.Event(), asyncio.Event()
        tasks = []

        async def source():
            with progress.waiting("publish", "any", lambda: progress.claimants("port", {1: object()})):
                started.set()
                await asyncio.Event().wait()

        async def consumer():
            with (
                progress.demand("consumer", frozenset(("port",)), 1),
                progress.waiting("access", "all", lambda: {tasks[0]}),
            ):
                demanded.set()
                await asyncio.Event().wait()

        tasks.extend((asyncio.create_task(source()), asyncio.create_task(consumer())))
        try:
            await started.wait()
            await demanded.wait()
            # The future consumer was a legitimate unknown wake path, not a
            # reason to reject early. Once its lifetime ends the cycle is known.
            with pytest.raises(NotReady, match="resource wait cycle"), progress.activity("future"):
                pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert not progress.waits and not progress.demands

    run(scenario())
