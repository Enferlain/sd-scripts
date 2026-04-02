import ast
from typing import Any


def parse_key_value_args(args: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` optimizer/scheduler-style argument lists.

    Values are parsed with ``ast.literal_eval`` when possible so numeric, tuple,
    list, and boolean-like literals retain their runtime types.
    """
    parsed: dict[str, Any] = {}
    if not args:
        return parsed

    for arg in args:
        key, value = arg.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        parsed[key] = value

    return parsed
