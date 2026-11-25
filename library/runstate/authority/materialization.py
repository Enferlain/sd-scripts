"""Materialization attempts: declared materializers run as one governed job (D17, D19).

Participants that name the same materializer are materialized by one call,
as when one released checkpoint supplies several participants: the call
returns one value per participant, in the order the strategy declares them.
A participant declared without a materializer is bound to what its owner
supplies. The attempt proposes materialization transitions with their
evidence, and the authority judges and publishes them together; a
materializer that fails stops realization with nothing changed.

A materializer receives its settings, such as the checkpoint it loads, as
its inputs: values in the strategy, from its declared parameters (D16).
Participants materialized by one call give the same settings. A materializer
that reads another participant's view comes with views in the second pass of
section 15.
"""

from collections.abc import Iterable, Mapping

from library.iro.accepted import implementation_evidence
from library.iro.seams import CallContext, Device, Implementation
from library.runstate.authority.authority import RunAuthority
from library.runstate.authority.records import Lifecycle, ParticipantRef
from library.runstate.authority.transitions import Materialization, Materialized, Published, Rejected, Supplied


class MaterializationAttempt:
    """One governed job that binds the named participants.

    ``installed`` holds the implementations the run uses, and ``device``
    is where the materializers run. ``supplied`` holds the values for
    participants declared without a materializer, and ``supplier`` names
    who supplied them.
    """

    def __init__(
        self,
        authority: RunAuthority,
        addresses: Iterable[str],
        *,
        installed: Mapping[str, Implementation] | None = None,
        device: Device | None = None,
        supplied: Mapping[str, object] | None = None,
        supplier: str = "its owner",
    ):
        self.authority = authority
        self.addresses = tuple(addresses)
        self.installed = installed or {}
        self.device = device
        self.supplied = supplied or {}
        self.supplier = supplier
        self.attempt = authority.attempt()

    def run(self) -> Published | Rejected:
        """Runs the materializers, then proposes every binding together."""
        section = self.authority.section
        reasons: list[str] = []
        groups: dict[str, list[ParticipantRef]] = {}
        proposals: list[Materialization] = []
        for address in self.addresses:
            try:
                participant = section.at(address)
            except KeyError as missing:
                reasons.append(str(missing.args[0]))
                continue
            if participant.lifecycle != Lifecycle.DECLARED:
                reasons.append(f"@{address} is {participant.lifecycle}, not declared")
            elif participant.declaration.reads:
                reasons.append(f"@{address}'s materializer reads another participant's view; views come with the second pass of section 15")
            elif participant.declaration.materializer is None:
                if address not in self.supplied:
                    reasons.append(f"@{address} is declared without a materializer, and nothing was supplied for it")
                else:
                    evidence = Supplied(self.attempt.reference, self.attempt.basis, self.supplier)
                    proposals.append(Materialization(participant.reference, self.supplied[address], evidence))
            else:
                groups.setdefault(participant.declaration.materializer, []).append(participant.reference)
        if reasons:
            return Rejected(tuple(reasons))
        declared_order = {participant.name: index for index, participant in enumerate(self.authority.accepted.seed.participants)}
        for materializer, members in groups.items():
            members.sort(key=lambda reference: declared_order[section.participants[reference].address])
            made = self._materialize(materializer, members, reasons)
            proposals += made
        if reasons:
            return Rejected(tuple(reasons))
        return self.authority.submit(proposals, self.attempt)

    def _materialize(self, materializer: str, members: list[ParticipantRef], reasons: list[str]) -> list[Materialization]:
        names = ", ".join(f"@{self.authority.section.participants[member].address}" for member in members)
        implementation = self.installed.get(materializer)
        if implementation is None:
            reasons.append(f"@{materializer} materializes {names} and is not installed")
            return []
        declared = implementation.declaration
        written = {_typed(self.authority.section.participants[member].declaration.settings) for member in members}
        if len(written) != 1:
            reasons.append(f"{names} name @{materializer} with different settings, and one call materializes them")
            return []
        given = self.authority.section.participants[members[0]].declaration.settings
        if len(declared.inputs) != len(given):
            reasons.append(f"@{materializer} declares {len(declared.inputs)} inputs, and the strategy gives it {len(given)} settings")
            return []
        mismatched = [
            f"setting {index + 1} is {type(value).__name__}, and @{materializer} declares {expected}"
            for index, (value, expected) in enumerate(zip(given, declared.inputs, strict=True))
            if type(value).__name__ != expected
        ]
        if mismatched:
            reasons.extend(mismatched)
            return []
        if len(declared.outputs) != len(members):
            reasons.append(f"@{materializer} declares {len(declared.outputs)} outputs for {len(members)} participants ({names})")
            return []
        if self.device is None:
            reasons.append(f"@{materializer} has no device to run on")
            return []
        try:
            with self.device.context(CallContext(), ()):
                results, completion = implementation(self.device, *given)
            completion.wait()
        except Exception as failed:
            reasons.append(f"@{materializer} failed to materialize {names}: {failed}")
            return []
        results = tuple(results)
        if len(results) != len(members):
            reasons.append(f"@{materializer} returned {len(results)} values for {len(members)} participants ({names})")
            return []
        evidence = Materialized(
            self.attempt.reference, self.attempt.basis, materializer, implementation_evidence(materializer, implementation)
        )
        return [Materialization(member, value, evidence) for member, value in zip(members, results, strict=True)]


def _typed(settings: tuple[str | int | float | bool, ...]) -> tuple[tuple[str, str | int | float | bool], ...]:
    """Settings compared with their types, so 1, 1.0 and true stay different."""
    return tuple((type(value).__name__, value) for value in settings)


__all__ = ["MaterializationAttempt"]
