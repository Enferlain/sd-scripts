"""Direct local dispatch artifacts generated only after prepared verification.

Unfused stages use one guarded coroutine per binding. Fused groups inline that
shell in one generated coroutine, retaining the same observer/validation ABI.
Only trusted template text and integer indexes enter code generation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, Self


class Frame(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None: ...
    def physical_end(self) -> None: ...
    def validate(self, result: Any, /) -> Any: ...


type FrameFactory = Callable[[str, Any], Frame]


@dataclass(frozen=True)
class Stage:
    origin: str
    invoke: Callable[[Any], Awaitable[Any]]


@dataclass
class LocalMetrics:
    dispatches: int = 0


async def invoke_stage(stage: Stage, value: Any, frame_factory: FrameFactory) -> Any:
    with frame_factory(stage.origin, value) as frame:
        try:
            result = await stage.invoke(value)
        finally:
            frame.physical_end()
        return frame.validate(result)


def fused(stages: tuple[Stage, ...]) -> tuple[Callable[[Any, FrameFactory], Awaitable[Any]], str]:
    body = ["async def run(value, frame_factory):"]
    for index in range(len(stages)):
        body.extend(
            [
                f"    with frame_factory(stages[{index}].origin, value) as frame:",
                "        try:",
                f"            result = await stages[{index}].invoke(value)",
                "        finally:",
                "            frame.physical_end()",
                "        value = frame.validate(result)",
            ]
        )
    body.append("    return value")
    text = "\n".join(body) + "\n"
    namespace: dict[str, Any] = {"stages": stages, "__builtins__": {}}
    exec(compile(text, "<verified-local-fusion>", "exec"), namespace)
    return namespace["run"], text


@dataclass(frozen=True)
class LocalProgram:
    dispatch: tuple[Callable[[Any, FrameFactory], Awaitable[Any]], ...]
    source_groups: tuple[tuple[str, ...], ...]
    generated: tuple[str, ...]

    async def run(self, value: Any, frame_factory: FrameFactory, metrics: LocalMetrics) -> Any:
        for call in self.dispatch:
            metrics.dispatches += 1  # Observation only; never admission/readiness.
            value = await call(value, frame_factory)
        return value


def compile_local(groups: Sequence[Sequence[Stage]]) -> LocalProgram:
    calls, sources, code = [], [], []
    for group in groups:
        stages = tuple(group)
        sources.append(tuple(stage.origin for stage in stages))
        if len(stages) == 1:

            async def single(value: Any, factory: FrameFactory, stage: Stage = stages[0]) -> Any:
                return await invoke_stage(stage, value, factory)

            calls.append(single)
        else:
            call, text = fused(stages)
            calls.append(call)
            code.append(text)
    return LocalProgram(tuple(calls), tuple(sources), tuple(code))
