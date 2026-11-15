"""Materialization attempts: declared materializers run as one governed job (D17, D19).

Participants that name the same materializer are materialized by one call,
as when one released checkpoint supplies several participants: the call
returns one value per participant, in the order the strategy declares them.
A participant declared without a materializer is bound to what its owner
supplies. The attempt proposes materialization transitions with their
evidence, and the authority judges and publishes them together; a
materializer that fails stops realization with nothing changed.

Materializers take no inputs yet: their settings arrive with the model
definitions (section 18), and one that reads another participant's view
comes with views in the second pass of section 15.
"""

from collections.abc import Iterable, Mapping

from library.iro.seams import CallContext, Device, Implementation
from library.runstate.authority.authority import RunAuthority
from library.runstate.authority.records import Lifecycle, ParticipantRef
from library.runstate.authority.transitions import Materialization, Materialized, Published, Rejected, Supplied


class MaterializationAttempt:
    """One governed job that binds the named participants.

    ``installed`` holds the implementations the run pins, and ``device``
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
        if declared.inputs:
            reasons.append(f"@{materializer} takes inputs; materializer settings come with the model definitions (section 18)")
            return []
        if len(declared.outputs) != len(members):
            reasons.append(f"@{materializer} declares {len(declared.outputs)} outputs for {len(members)} participants ({names})")
            return []
        if self.device is None:
            reasons.append(f"@{materializer} has no device to run on")
            return []
        try:
            with self.device.context(CallContext(), (), ()):
                results, completion = implementation(self.device)
            completion.wait()
        except Exception as failed:
            reasons.append(f"@{materializer} failed to materialize {names}: {failed}")
            return []
        results = tuple(results)
        if len(results) != len(members):
            reasons.append(f"@{materializer} returned {len(results)} values for {len(members)} participants ({names})")
            return []
        evidence = Materialized(self.attempt.reference, self.attempt.basis, materializer, declared.version, declared.digest)
        return [Materialization(member, value, evidence) for member, value in zip(members, results, strict=True)]


__all__ = ["MaterializationAttempt"]
