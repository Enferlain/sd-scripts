"""Leases and their protection, retained results, owner updates and request reservations (D14, D16).

An access takes its leases from the owners together; protection excludes
incompatible uses while a lease is held, and a retained result keeps the
leases it depends on until its release. A request holds the version at the
request for each ``at_request`` use: a snapshot is taken at once, a live
read reserved, and updates wait while a reservation stands.
"""

from collections.abc import Mapping, Sequence

from library.iro.operation_contracts import AccessMode, ResourceUse, VersionSelection
from library.iro.prepare.lowered import Call, HandlerInfo, Invoke
from library.iro.runtime.kernel import Kernel
from library.iro.runtime.state import Held, Identity, Product, Protection, Request, RunFailure
from library.iro.runtime.storage import DeviceStorage
from library.iro.runtime.turns import Turns
from library.iro.seams import Capture, Owner, Use, Version


class Protections:
    """The run's leases, each resource's protection, and the retained results that keep leases."""

    def __init__(
        self,
        kernel: Kernel,
        owners: Mapping[str, Owner],
        turns: Turns,
        storage: DeviceStorage,
        protection: Mapping[str, Protection],
        updated: frozenset[str],
    ):
        self.kernel = kernel
        self.changing = updated  # the resources some access updates; the others keep one version for the run
        self.owners, self.turns, self.storage = owners, turns, storage
        self.by_resource = protection
        self.leases: dict[int, Held] = {}
        self.products: dict[tuple[str, int], Product] = {}
        self.productions: dict[Identity, list[Version]] = {}  # versions observed by leases taken within each production
        self._next_token = 0

    # Taking leases.

    def can_lease(self, name: str, use: ResourceUse) -> bool:
        protection = self.by_resource[name]
        if use.mode == AccessMode.SNAPSHOT and use.selection in (VersionSelection.PUBLISHED_INITIAL, VersionSelection.AT_REQUEST):
            return True
        if use.mode in (AccessMode.SNAPSHOT, AccessMode.PROTECTED_CURRENT):
            return protection.writer is None
        return protection.writer is None and not protection.readers and not protection.reservations

    def acquire(self, step: Invoke, identity: Identity, request: Request | None) -> tuple[Held, ...]:
        """The access's leases, taken together; if one cannot be taken, those already taken are released."""
        held: list[Held] = []
        try:
            return self._acquire_each(step, identity, request, held)
        except BaseException:
            self.ended(held)
            raise

    def ended(self, leases: Sequence[Held]) -> None:
        """An access's body ended: its leases go once nothing retained keeps them; a request's snapshots stay with it."""
        for lease in leases:
            if not lease.requested:
                lease.body_ended = True
        for lease in leases:
            self.release_lease_if_done(lease)

    def _acquire_each(self, step: Invoke, identity: Identity, request: Request | None, held: list[Held]) -> tuple[Held, ...]:
        for name, use in step.uses:
            if use.mode == AccessMode.GRANT:
                raise RunFailure(f"{step.path}: grants come with training's words")
            if use.selection == VersionSelection.AT_REQUEST and request is None:
                raise RunFailure(f"{step.path}: at_request outside requested work")
            if use.mode == AccessMode.SNAPSHOT and use.selection == VersionSelection.AT_REQUEST:
                assert request is not None
                lease = request.snapshots[name]  # every use in the handler's work reads the request's one snapshot
            else:
                reserved = {name: request.reserved[name]} if request is not None and name in request.reserved else None
                (granted,) = self.owners[name].lease([Use(name, use.mode, use.selection, use.route, use.view)], reserved)
                if reserved is not None and granted.version != reserved[name]:
                    granted.release()
                    raise RunFailure(f"{step.path}: @{name} moved past the version its request reserved")
                lease = self.hold(name, use.mode, granted)
            protection = self.by_resource[name]
            if use.mode == AccessMode.PROTECTED_CURRENT:
                protection.readers.add(lease.token)
            elif use.mode == AccessMode.OWNER_UPDATE:
                protection.writer = lease.token
            for production, observed in self.productions.items():
                if identity[: len(production)] == production:
                    observed.append(lease.version)
            self.kernel.fact(step.path, identity, "leased", name, str(use.mode), lease.version)
            held.append(lease)
        return tuple(held)

    def hold(self, name: str, mode: AccessMode, granted) -> Held:
        """Tracks an owner's lease; a snapshot of device storage asks for its copy (D15).

        A participant the run never updates needs no copy: its live storage
        already is the snapshot's version, so the handle is built over it.
        """
        capture = granted.handle if isinstance(granted.handle, Capture) else None
        adapter = None
        built = None
        if capture is not None and mode == AccessMode.SNAPSHOT and name not in self.changing:
            try:
                built, capture = capture.over(capture.storage), None
            except BaseException:
                granted.release()
                raise
        elif capture is not None:
            try:
                adapter = self.storage.capturer(name, mode, granted.version)
            except RunFailure:
                granted.release()
                raise
        lease = Held(self._next_token, name, mode, granted, granted.version)
        if built is not None:
            lease.built, lease.over_storage = built, True
        self._next_token += 1
        self.leases[lease.token] = lease
        self.storage.held(lease, capture, adapter)
        return lease

    # Releasing.

    def release_product(self, product: Product, how: str) -> None:
        if self.products.get(product.key) is not product or not product.active:
            raise RunFailure(f"{product.key[0]}: a retained result is released twice")
        if product.borrowers or any(any(p is product for p in other.parents) for other in self.products.values() if other.active):
            raise RunFailure(f"{product.key[0]}: released while a dependent use is still running")
        product.active = False
        del self.products[product.key]
        self.kernel.fact(product.key[0], (), "product_released", product.key[1], how)
        for lease in product.leases:
            self.release_lease_if_done(lease)
        self.turns.released(product)

    def release_lease_if_done(self, lease: Held) -> None:
        if not lease.active or not lease.body_ended:
            return
        if any(lease in product.leases for product in self.products.values()):
            return
        protection = self.by_resource[lease.resource]
        protection.readers.discard(lease.token)
        if protection.writer == lease.token:
            protection.writer = None
        lease.active = False
        lease.lease.release()
        del self.leases[lease.token]
        self.kernel.fact(lease.resource, (), "lease_released", lease.version)
        self.kernel.changed(("resource", lease.resource))
        self.storage.released(lease.resource)

    # What calls do with leases.

    def updated(self, call: str, held: object, identity: Identity) -> None:
        """Records an owner update, which only the current writer's lease can make.

        A call that updates owner state is never fused with another, so each
        update is observed after the one call that made it. The fact carries
        the version; the value stays with its owner.
        """
        if not isinstance(held, Held) or held.mode not in (AccessMode.OWNER_UPDATE, AccessMode.GRANT):
            raise RunFailure(f"{call}: updates state it holds no owner_update lease for")
        if self.by_resource[held.resource].writer != held.token:
            raise RunFailure(f"{call}: a stale owner update of @{held.resource}")
        now = self.owners[held.resource].current(held.resource)
        if now == held.version:
            return
        if (now.incarnation, now.lineage) != (held.version.incarnation, held.version.lineage) or now.ordinal <= held.version.ordinal:
            raise RunFailure(f"{call}: @{held.resource} moved to an incomparable or older version")
        held.version = now
        self.kernel.fact(held.resource, identity, "updated", now)

    def retain(self, call: Call, ordinal: int, value: object, args: list[object], identity: Identity, device: str) -> Product:
        """A retained result, keeping the leases it depends on, its parents' included, until its release (D16)."""
        retained = [args[i] for i in call.retains]
        parents = tuple(v for v in retained if isinstance(v, Product))
        leases = {id(v): v for v in retained if isinstance(v, Held)}
        for parent in parents:
            leases.update((id(lease), lease) for lease in parent.leases)
        product = Product((call.path, ordinal), value, tuple(leases.values()), parents)
        self.products[product.key] = product
        self.turns.retained(product, identity, device)
        self.kernel.fact(call.path, identity, "product_retained", ordinal, tuple(lease.version for lease in product.leases))
        return product

    # Productions.

    def begin_production(self, identity: Identity) -> None:
        self.productions[identity] = []

    def end_production(self, identity: Identity) -> list[Version]:
        return self.productions.pop(identity, [])

    # Request reservations.

    def reserve(self, request: Request, handler: HandlerInfo) -> None:
        """Holds the version at the request for each at_request use: a snapshot is taken now, a live read reserved.

        Once per resource and kind: every use of it in the handler's work reads
        the same snapshot, or the same reserved version. The request holds its
        snapshots until it settles. If one cannot be held, those already held
        are released: the request is never made.
        """
        try:
            for name, use in handler.at_request:
                owner = self.owners[name]
                if name in (request.snapshots if use.mode == AccessMode.SNAPSHOT else request.reserved):
                    continue
                if use.mode == AccessMode.SNAPSHOT:
                    (granted,) = owner.lease([Use(name, use.mode, VersionSelection.CURRENT_AT_START, use.route, use.view)])
                    lease = self.hold(name, use.mode, granted)
                    lease.requested = True
                    request.snapshots[name] = lease
                    version = lease.version
                else:
                    version = owner.current(name)
                    request.reserved[name] = version
                    self.by_resource[name].reservations.add(id(request))
                self.kernel.fact(name, request.cause, "reserved", version, str(use.mode))
        except BaseException:
            self.end_reservations(request)
            raise

    def end_reservations(self, request: Request) -> None:
        for name in request.reserved:
            self.by_resource[name].reservations.discard(id(request))
        names = list(request.reserved)
        request.reserved.clear()
        for lease in request.snapshots.values():
            lease.requested, lease.body_ended = False, True
            self.release_lease_if_done(lease)
        request.snapshots.clear()
        self.kernel.changed(*(("resource", name) for name in names))


__all__ = ["Protections"]
