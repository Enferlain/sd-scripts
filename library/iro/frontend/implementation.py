"""Declaring an implementation (D18).

``@implementation(...)`` records a function's declaration and returns an
implementation the runtime calls through the implementation seam. It has no
registration side effect: the launcher installs implementations explicitly
(D2).

Until the activity reader exists (``tasks.md`` section 17), inputs and outputs
are written as written-form types, such as ``protected_current lease`` or
``retained loss``. The reader will take them from the signature's lease and
value types instead, so they are never stated twice.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from library.iro.seams import Completion, Declaration, Device, Effect


@dataclass(frozen=True)
class Declared:
    """A function with its declaration, callable as an implementation.

    With ``completion="own"`` the function returns ``(results, completion)``,
    where ``results`` follows the same rule as any other return: one value
    for one output, a tuple for several, ``None`` for none.
    """

    declaration: Declaration
    function: Callable[..., object]

    def __call__(self, device: Device, *arguments: object) -> tuple[tuple[object, ...], Completion]:
        """Runs the function and returns its results with their completion.

        The runtime calls it inside the device's context for the call, so
        the completion covers the work the function submitted there.
        """
        declared = self.declaration
        if len(arguments) != len(declared.inputs):
            raise TypeError(f"@{declared.name} declares {len(declared.inputs)} inputs and received {len(arguments)}")
        returned = self.function(*arguments)
        if declared.completion == "own":
            if not (isinstance(returned, tuple) and len(returned) == 2 and isinstance(returned[1], Completion)):
                raise TypeError(f"@{declared.name} declares its own completion and returns (results, completion)")
            returned, completion = returned
        else:
            completion = device.completion()
        return _results(declared, returned), completion


def _results(declared: Declaration, returned: object) -> tuple[object, ...]:
    count = len(declared.outputs)
    if count == 0:
        if returned is not None:
            raise TypeError(f"@{declared.name} declares no results and returned {type(returned).__name__}")
        return ()
    if count == 1:
        return (returned,)
    if not isinstance(returned, tuple) or len(returned) != count:
        raise TypeError(f"@{declared.name} declares {count} results and returns them as a tuple")
    return returned


def implementation(
    *,
    version: int | str,
    inputs: Iterable[str] = (),
    outputs: Iterable[str] = (),
    effects: Iterable[Effect | str] = (),
    concurrency_safe: bool = False,
    completion: str = "device",
    name: str | None = None,
) -> Callable[[Callable[..., object]], Declared]:
    """Declares the decorated function as an implementation at one implementation version.

    D16 says what a version covers: change it whenever the declaration or the
    results change.
    """

    def declare(function: Callable[..., object]) -> Declared:
        declaration = Declaration(
            name or getattr(function, "__name__", ""),
            str(version),
            tuple(inputs),
            tuple(outputs),
            frozenset(Effect(e) for e in effects),
            concurrency_safe,
            completion,
        )
        return Declared(declaration, function)

    return declare


__all__ = ["Declared", "implementation"]
