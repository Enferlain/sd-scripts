"""Independent pair-extension observations; no IR, registry or runtime imports."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .conformance import check_b


def check_extension(report: Mapping[str, Any]) -> None:
    projected = copy.deepcopy(report)
    expected_origins = ("data.double", "data.offset", "pair.pack", "pair.first", "scorer", "publisher")
    for ticket in projected["tickets"]:
        calls = ticket["calls"]
        assert len(calls) == 6 and tuple(item["origin"] for item in calls) == expected_origins, "extension source mapping/entry differs"
        value = 2 * ticket["value"] + 1
        pair = [value, value * 2]  # Independent coefficient/payload expectation.
        pack, first = calls[2:4]
        assert (pack["input"], pack["output"], first["input"], first["output"]) == (value, pair, pair, value), (
            "independent pair resource/schema arithmetic differs"
        )
        assert pack["role"] == first["role"] == "map" and pack["outcome"] == first["outcome"] == "returned", "false extension success"
        previous = ticket["selected"]
        for call in calls:
            assert previous < call["entered"] < call["physical_end"] < call["released"] < ticket["retired"], (
                "extension stage lifetime differs"
            )
            previous = call["released"]
        # This projection is only an oracle adapter. The separate pair checks
        # above prevent hidden payload corruption even when final scalars agree.
        ticket["calls"] = [item for item in calls if item["origin"] not in ("pair.pack", "pair.first")]
    check_b(projected)
