"""Epoch-fenced topology changes for a strict-neighbor KADENCE lane.

The :class:`TopologyAuthority` owns membership integrity, not scheduling.  It
never accepts phases, pressure, workload demand, or pair scores and therefore
cannot optimize the lane.  Its global ``order`` property is an operator/test
view.  A job can obtain only a :class:`NeighborLease` and, while participating
in a transaction, a :class:`TopologyProposal` containing its own old and new
neighbors.

Join, leave, failed-member repair, and adjacent swap use a small deterministic
two-phase protocol:

1. ``prepare_*`` validates an expected membership epoch and records a proposed
   order without changing the live ring.
2. Every required participant acknowledges the proposal.
3. ``commit`` atomically installs the order at ``epoch + 1``.  ``abort`` or a
   deadline expiry discards the proposal and leaves the old order intact.

Transaction ids are idempotency keys.  Replayed prepare, acknowledge, commit,
and abort requests return the existing receipt.  An id reused for a different
operation is rejected.  Neighbor data messages are accepted only from one of
the receiver's two neighbors at the current epoch, and duplicate sequences
are harmless.

This is a single-authority reference model.  A replicated implementation
would need a consensus-backed transaction log, but its externally visible
epoch and idempotence rules should remain the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class TopologyError(RuntimeError):
    """Base class for membership protocol failures."""


class InvalidTopology(TopologyError):
    """A requested change would not form a valid strict-neighbor ring."""


class EpochMismatch(TopologyError):
    """A request is fenced by a membership epoch other than the live epoch."""


class StaleEpoch(EpochMismatch):
    """A request or message belongs to an older membership epoch."""


class FutureEpoch(EpochMismatch):
    """A request or message claims an epoch the authority has not installed."""


class TransactionConflict(TopologyError):
    """A transaction conflicts with another prepared or terminal operation."""


class TransactionStateError(TopologyError):
    """An operation is invalid in the transaction's current state."""


class MissingAcknowledgement(TransactionStateError):
    """Commit was attempted before every required participant prepared."""


class UnknownMember(TopologyError):
    """A job is not a member of this lane."""


class NonNeighbor(TopologyError):
    """A job attempted to send a peer frame beyond an immediate neighbor."""


class LeaseNotExpired(TopologyError):
    """Failure repair was requested while the suspected job may still be live."""


class LeaseExpired(TopologyError):
    """A fenced member tried to renew or participate after lease expiry."""


class StaleSequence(TopologyError):
    """A neighbor frame predates the latest accepted frame on that link."""


class ChangeKind(str, Enum):
    JOIN = "join"
    LEAVE = "leave"
    FAILED_LEAVE = "failed_leave"
    SWAP = "swap"


class TransactionStatus(str, Enum):
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"


class MessageDisposition(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class NeighborLease:
    """The complete topology state visible to one admitted job."""

    jid: str
    predecessor: str
    successor: str
    epoch: int
    lease_deadline: float


@dataclass(frozen=True)
class TopologyProposal:
    """One participant's local projection of a prepared topology change."""

    transaction_id: str
    kind: ChangeKind
    base_epoch: int
    proposed_epoch: int
    deadline: float
    current_predecessor: str | None
    current_successor: str | None
    proposed_predecessor: str | None
    proposed_successor: str | None


@dataclass(frozen=True)
class TransactionReceipt:
    """Authority receipt suitable for retries and durable control-plane logging."""

    transaction_id: str
    kind: ChangeKind
    base_epoch: int
    status: TransactionStatus
    required: tuple[str, ...]
    acknowledged: tuple[str, ...]
    deadline: float
    terminal_epoch: int | None
    reason: str | None


@dataclass(frozen=True)
class ParticipantAcknowledgement:
    """Local acknowledgement result that does not reveal other participants."""

    transaction_id: str
    jid: str
    base_epoch: int
    status: TransactionStatus
    acknowledged: bool


@dataclass
class _Transaction:
    transaction_id: str
    kind: ChangeKind
    base_epoch: int
    prepared_at: float
    deadline: float
    before: tuple[str, ...]
    after: tuple[str, ...]
    required: tuple[str, ...]
    fingerprint: tuple[object, ...]
    acknowledged: dict[str, float]
    status: TransactionStatus = TransactionStatus.PREPARED
    terminal_epoch: int | None = None
    reason: str | None = None


def _validate_time(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_jid(jid: str, name: str = "jid") -> str:
    if not isinstance(jid, str) or not jid:
        raise ValueError(f"{name} must be a non-empty string")
    return jid


def _neighbors(order: tuple[str, ...], jid: str) -> tuple[str, str]:
    try:
        index = order.index(jid)
    except ValueError as exc:
        raise UnknownMember(jid) from exc
    return order[index - 1], order[(index + 1) % len(order)]


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class TopologyAuthority:
    """Serialize safe lane membership changes without controlling job phase."""

    def __init__(
        self,
        order: Iterable[str],
        *,
        initial_epoch: int = 0,
        lease_duration: float = 30.0,
        start_time: float = 0.0,
    ) -> None:
        members = tuple(order)
        if len(members) < 3:
            raise InvalidTopology("a strict-neighbor ring requires at least 3 jobs")
        for jid in members:
            _validate_jid(jid)
        if len(set(members)) != len(members):
            raise InvalidTopology("job ids must be unique")
        if not isinstance(initial_epoch, int) or initial_epoch < 0:
            raise ValueError("initial_epoch must be a non-negative integer")
        self._lease_duration = float(lease_duration)
        if not math.isfinite(self._lease_duration) or self._lease_duration <= 0.0:
            raise ValueError("lease_duration must be finite and positive")
        start_time = _validate_time(start_time, "start_time")

        self._order = members
        self._epoch = initial_epoch
        self._member_expiry = {
            jid: start_time + self._lease_duration for jid in members
        }
        self._transactions: dict[str, _Transaction] = {}
        self._active_transaction: str | None = None
        self._last_sequence: dict[tuple[int, str, str], int] = {}

    @property
    def order(self) -> tuple[str, ...]:
        """Return the authority/operator membership view, never a job view."""

        return self._order

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def active_transaction(self) -> str | None:
        return self._active_transaction

    def _check_epoch(self, expected_epoch: int) -> None:
        if not isinstance(expected_epoch, int) or expected_epoch < 0:
            raise ValueError("expected_epoch must be a non-negative integer")
        if expected_epoch < self._epoch:
            raise StaleEpoch(
                f"epoch {expected_epoch} is stale; live epoch is {self._epoch}"
            )
        if expected_epoch > self._epoch:
            raise FutureEpoch(
                f"epoch {expected_epoch} is in the future; live epoch is {self._epoch}"
            )

    def _receipt(self, transaction: _Transaction) -> TransactionReceipt:
        return TransactionReceipt(
            transaction_id=transaction.transaction_id,
            kind=transaction.kind,
            base_epoch=transaction.base_epoch,
            status=transaction.status,
            required=transaction.required,
            acknowledged=tuple(
                jid for jid in transaction.required
                if jid in transaction.acknowledged
            ),
            deadline=transaction.deadline,
            terminal_epoch=transaction.terminal_epoch,
            reason=transaction.reason,
        )

    def transaction(self, transaction_id: str) -> TransactionReceipt:
        """Read an authority receipt without exposing either global order."""

        try:
            transaction = self._transactions[transaction_id]
        except KeyError as exc:
            raise TransactionStateError(
                f"unknown transaction {transaction_id!r}"
            ) from exc
        return self._receipt(transaction)

    @staticmethod
    def _participant_receipt(
        transaction: _Transaction, jid: str,
    ) -> ParticipantAcknowledgement:
        return ParticipantAcknowledgement(
            transaction_id=transaction.transaction_id,
            jid=jid,
            base_epoch=transaction.base_epoch,
            status=transaction.status,
            acknowledged=jid in transaction.acknowledged,
        )

    def neighbor_view(self, jid: str) -> NeighborLease:
        """Return only ``jid`` and its two immediate neighbors."""

        predecessor, successor = _neighbors(self._order, jid)
        return NeighborLease(
            jid=jid,
            predecessor=predecessor,
            successor=successor,
            epoch=self._epoch,
            lease_deadline=self._member_expiry[jid],
        )

    def renew_member(
        self, jid: str, *, expected_epoch: int, now: float,
    ) -> NeighborLease:
        """Renew one live member; an already-expired identity stays fenced."""

        now = _validate_time(now, "now")
        self._check_epoch(expected_epoch)
        if jid not in self._member_expiry:
            raise UnknownMember(jid)
        if now >= self._member_expiry[jid]:
            raise LeaseExpired(f"membership lease for {jid!r} has expired")
        self._member_expiry[jid] = now + self._lease_duration
        return self.neighbor_view(jid)

    def _abort_transaction(self, transaction: _Transaction, reason: str) -> None:
        transaction.status = TransactionStatus.ABORTED
        transaction.reason = reason
        transaction.terminal_epoch = self._epoch
        if self._active_transaction == transaction.transaction_id:
            self._active_transaction = None

    def expire(self, *, now: float) -> tuple[TransactionReceipt, ...]:
        """Abort every prepared transaction whose deadline has elapsed."""

        now = _validate_time(now, "now")
        expired: list[TransactionReceipt] = []
        for transaction in self._transactions.values():
            if (
                transaction.status is TransactionStatus.PREPARED
                and now >= transaction.deadline
            ):
                self._abort_transaction(transaction, "timeout")
                expired.append(self._receipt(transaction))
        return tuple(expired)

    def _prepare(
        self,
        *,
        transaction_id: str,
        kind: ChangeKind,
        expected_epoch: int,
        now: float,
        timeout: float,
        after: tuple[str, ...],
        required: tuple[str, ...],
        fingerprint: tuple[object, ...],
    ) -> TransactionReceipt:
        transaction_id = _validate_jid(transaction_id, "transaction_id")
        now = _validate_time(now, "now")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be finite and positive")
        self.expire(now=now)

        prior = self._transactions.get(transaction_id)
        if prior is not None:
            if prior.fingerprint != fingerprint:
                raise TransactionConflict(
                    f"transaction id {transaction_id!r} was reused"
                )
            return self._receipt(prior)

        self._check_epoch(expected_epoch)
        if self._active_transaction is not None:
            raise TransactionConflict(
                f"transaction {self._active_transaction!r} is still prepared"
            )
        if len(after) < 3 or len(set(after)) != len(after):
            raise InvalidTopology("a committed ring needs at least 3 unique jobs")

        transaction = _Transaction(
            transaction_id=transaction_id,
            kind=kind,
            base_epoch=expected_epoch,
            prepared_at=now,
            deadline=now + timeout,
            before=self._order,
            after=after,
            required=_unique_in_order(required),
            fingerprint=fingerprint,
            acknowledged={},
        )
        self._transactions[transaction_id] = transaction
        self._active_transaction = transaction_id
        return self._receipt(transaction)

    def prepare_join(
        self,
        transaction_id: str,
        *,
        new_jid: str,
        left_jid: str,
        right_jid: str,
        expected_epoch: int,
        now: float,
        timeout: float,
    ) -> TransactionReceipt:
        """Prepare ``left -> new -> right`` over one currently adjacent edge."""

        new_jid = _validate_jid(new_jid, "new_jid")
        left_jid = _validate_jid(left_jid, "left_jid")
        right_jid = _validate_jid(right_jid, "right_jid")
        fingerprint = (
            ChangeKind.JOIN, expected_epoch, new_jid, left_jid, right_jid,
        )
        prior = self._transactions.get(transaction_id)
        if prior is not None:
            return self._prepare(
                transaction_id=transaction_id,
                kind=ChangeKind.JOIN,
                expected_epoch=expected_epoch,
                now=now,
                timeout=timeout,
                after=prior.after,
                required=prior.required,
                fingerprint=fingerprint,
            )
        self._check_epoch(expected_epoch)
        if new_jid in self._order:
            raise InvalidTopology(f"job {new_jid!r} is already a member")
        if _neighbors(self._order, left_jid)[1] != right_jid:
            raise InvalidTopology(
                f"{left_jid!r} -> {right_jid!r} is not a live edge"
            )
        insertion = self._order.index(left_jid) + 1
        after = self._order[:insertion] + (new_jid,) + self._order[insertion:]
        return self._prepare(
            transaction_id=transaction_id,
            kind=ChangeKind.JOIN,
            expected_epoch=expected_epoch,
            now=now,
            timeout=timeout,
            after=after,
            required=(left_jid, new_jid, right_jid),
            fingerprint=fingerprint,
        )

    def prepare_leave(
        self,
        transaction_id: str,
        *,
        jid: str,
        expected_epoch: int,
        now: float,
        timeout: float,
    ) -> TransactionReceipt:
        """Prepare a graceful inverse splice; the departing job must ack."""

        jid = _validate_jid(jid)
        fingerprint = (ChangeKind.LEAVE, expected_epoch, jid)
        prior = self._transactions.get(transaction_id)
        if prior is not None:
            return self._prepare(
                transaction_id=transaction_id,
                kind=ChangeKind.LEAVE,
                expected_epoch=expected_epoch,
                now=now,
                timeout=timeout,
                after=prior.after,
                required=prior.required,
                fingerprint=fingerprint,
            )
        self._check_epoch(expected_epoch)
        if len(self._order) <= 3:
            raise InvalidTopology("leave would break the strict-neighbor ring")
        left_jid, right_jid = _neighbors(self._order, jid)
        after = tuple(member for member in self._order if member != jid)
        return self._prepare(
            transaction_id=transaction_id,
            kind=ChangeKind.LEAVE,
            expected_epoch=expected_epoch,
            now=now,
            timeout=timeout,
            after=after,
            required=(left_jid, jid, right_jid),
            fingerprint=fingerprint,
        )

    def prepare_failed_leave(
        self,
        transaction_id: str,
        *,
        jid: str,
        expected_epoch: int,
        now: float,
        timeout: float,
    ) -> TransactionReceipt:
        """Prepare repair after ``jid``'s lease expired; the failed job cannot ack."""

        jid = _validate_jid(jid)
        fingerprint = (ChangeKind.FAILED_LEAVE, expected_epoch, jid)
        prior = self._transactions.get(transaction_id)
        if prior is not None:
            return self._prepare(
                transaction_id=transaction_id,
                kind=ChangeKind.FAILED_LEAVE,
                expected_epoch=expected_epoch,
                now=now,
                timeout=timeout,
                after=prior.after,
                required=prior.required,
                fingerprint=fingerprint,
            )
        self._check_epoch(expected_epoch)
        now = _validate_time(now, "now")
        if len(self._order) <= 3:
            raise InvalidTopology("failed leave would break the strict-neighbor ring")
        if jid not in self._member_expiry:
            raise UnknownMember(jid)
        if now < self._member_expiry[jid]:
            raise LeaseNotExpired(
                f"membership lease for {jid!r} is live until "
                f"{self._member_expiry[jid]:.12g}"
            )
        left_jid, right_jid = _neighbors(self._order, jid)
        after = tuple(member for member in self._order if member != jid)
        return self._prepare(
            transaction_id=transaction_id,
            kind=ChangeKind.FAILED_LEAVE,
            expected_epoch=expected_epoch,
            now=now,
            timeout=timeout,
            after=after,
            required=(left_jid, right_jid),
            fingerprint=fingerprint,
        )

    def prepare_swap(
        self,
        transaction_id: str,
        *,
        left_jid: str,
        right_jid: str,
        expected_epoch: int,
        now: float,
        timeout: float,
    ) -> TransactionReceipt:
        """Prepare a clockwise adjacent swap with only local link participants."""

        left_jid = _validate_jid(left_jid, "left_jid")
        right_jid = _validate_jid(right_jid, "right_jid")
        fingerprint = (
            ChangeKind.SWAP, expected_epoch, left_jid, right_jid,
        )
        prior = self._transactions.get(transaction_id)
        if prior is not None:
            return self._prepare(
                transaction_id=transaction_id,
                kind=ChangeKind.SWAP,
                expected_epoch=expected_epoch,
                now=now,
                timeout=timeout,
                after=prior.after,
                required=prior.required,
                fingerprint=fingerprint,
            )
        self._check_epoch(expected_epoch)
        if len(self._order) < 4:
            raise InvalidTopology("an adjacent swap needs four link participants")
        outside_left, actual_right = _neighbors(self._order, left_jid)
        if actual_right != right_jid:
            raise InvalidTopology(
                f"{left_jid!r} -> {right_jid!r} is not a live edge"
            )
        _, outside_right = _neighbors(self._order, right_jid)
        left_index = self._order.index(left_jid)
        right_index = self._order.index(right_jid)
        after_list = list(self._order)
        after_list[left_index], after_list[right_index] = (
            after_list[right_index], after_list[left_index]
        )
        return self._prepare(
            transaction_id=transaction_id,
            kind=ChangeKind.SWAP,
            expected_epoch=expected_epoch,
            now=now,
            timeout=timeout,
            after=tuple(after_list),
            required=(outside_left, left_jid, right_jid, outside_right),
            fingerprint=fingerprint,
        )

    def proposal(self, transaction_id: str, jid: str) -> TopologyProposal:
        """Return one participant's local old/new links, never either ring order."""

        try:
            transaction = self._transactions[transaction_id]
        except KeyError as exc:
            raise TransactionStateError(
                f"unknown transaction {transaction_id!r}"
            ) from exc
        if transaction.status is not TransactionStatus.PREPARED:
            raise TransactionStateError(
                f"transaction {transaction_id!r} is {transaction.status.value}"
            )
        if jid not in transaction.required:
            raise TransactionConflict(
                f"job {jid!r} is not a participant in {transaction_id!r}"
            )
        if jid in transaction.before:
            current_left, current_right = _neighbors(transaction.before, jid)
        else:
            current_left = current_right = None
        if jid in transaction.after:
            proposed_left, proposed_right = _neighbors(transaction.after, jid)
        else:
            proposed_left = proposed_right = None
        return TopologyProposal(
            transaction_id=transaction.transaction_id,
            kind=transaction.kind,
            base_epoch=transaction.base_epoch,
            proposed_epoch=transaction.base_epoch + 1,
            deadline=transaction.deadline,
            current_predecessor=current_left,
            current_successor=current_right,
            proposed_predecessor=proposed_left,
            proposed_successor=proposed_right,
        )

    def acknowledge(
        self,
        transaction_id: str,
        jid: str,
        *,
        expected_epoch: int,
        now: float,
    ) -> ParticipantAcknowledgement:
        """Idempotently acknowledge one participant's local proposal."""

        now = _validate_time(now, "now")
        self.expire(now=now)
        try:
            transaction = self._transactions[transaction_id]
        except KeyError as exc:
            raise TransactionStateError(
                f"unknown transaction {transaction_id!r}"
            ) from exc
        if jid not in transaction.required:
            raise TransactionConflict(
                f"job {jid!r} is not a participant in {transaction_id!r}"
            )
        if transaction.status is not TransactionStatus.PREPARED:
            return self._participant_receipt(transaction, jid)
        self._check_epoch(expected_epoch)
        if expected_epoch != transaction.base_epoch:
            raise EpochMismatch("acknowledgement does not match transaction epoch")
        if jid in transaction.acknowledged:
            return self._participant_receipt(transaction, jid)
        if jid in self._member_expiry:
            if now >= self._member_expiry[jid]:
                raise LeaseExpired(f"membership lease for {jid!r} has expired")
            # An authenticated acknowledgement is also proof of liveness.
            self._member_expiry[jid] = now + self._lease_duration
        transaction.acknowledged[jid] = now
        return self._participant_receipt(transaction, jid)

    def commit(
        self,
        transaction_id: str,
        *,
        expected_epoch: int,
        now: float,
    ) -> TransactionReceipt:
        """Atomically install a fully acknowledged proposal at ``epoch + 1``."""

        now = _validate_time(now, "now")
        self.expire(now=now)
        try:
            transaction = self._transactions[transaction_id]
        except KeyError as exc:
            raise TransactionStateError(
                f"unknown transaction {transaction_id!r}"
            ) from exc
        if transaction.status is TransactionStatus.COMMITTED:
            return self._receipt(transaction)
        if transaction.status is TransactionStatus.ABORTED:
            raise TransactionStateError(
                f"transaction {transaction_id!r} was aborted: {transaction.reason}"
            )
        self._check_epoch(expected_epoch)
        if expected_epoch != transaction.base_epoch:
            raise EpochMismatch("commit does not match transaction epoch")
        missing = tuple(
            jid for jid in transaction.required
            if jid not in transaction.acknowledged
        )
        if missing:
            raise MissingAcknowledgement(
                "missing prepare acknowledgements from " + ", ".join(missing)
            )
        for jid in transaction.required:
            if jid in self._member_expiry:
                if now >= self._member_expiry[jid]:
                    raise LeaseExpired(f"membership lease for {jid!r} has expired")
            elif now >= transaction.acknowledged[jid] + self._lease_duration:
                raise LeaseExpired(
                    f"candidate lease for {jid!r} has expired before commit"
                )

        old_members = set(self._order)
        new_members = set(transaction.after)
        for removed in old_members - new_members:
            del self._member_expiry[removed]
        for added in new_members - old_members:
            self._member_expiry[added] = now + self._lease_duration

        self._order = transaction.after
        self._epoch += 1
        transaction.status = TransactionStatus.COMMITTED
        transaction.terminal_epoch = self._epoch
        self._active_transaction = None
        # Frames from previous epochs are fenced by the epoch check; old
        # sequence bookkeeping is no longer needed.
        self._last_sequence.clear()
        return self._receipt(transaction)

    def abort(
        self,
        transaction_id: str,
        *,
        reason: str = "explicit abort",
        now: float,
    ) -> TransactionReceipt:
        """Idempotently roll back a prepared operation (the live order is unchanged)."""

        now = _validate_time(now, "now")
        # Advancing logical time past the deadline deterministically records a
        # timeout, even when the next command happens to be an explicit abort.
        self.expire(now=now)
        try:
            transaction = self._transactions[transaction_id]
        except KeyError as exc:
            raise TransactionStateError(
                f"unknown transaction {transaction_id!r}"
            ) from exc
        if transaction.status is TransactionStatus.ABORTED:
            return self._receipt(transaction)
        if transaction.status is TransactionStatus.COMMITTED:
            raise TransactionStateError("a committed topology change cannot be aborted")
        self._abort_transaction(transaction, reason)
        return self._receipt(transaction)

    def accept_neighbor_message(
        self,
        *,
        receiver: str,
        sender: str,
        epoch: int,
        sequence: int,
    ) -> MessageDisposition:
        """Fence a data-plane frame by epoch, adjacency, and monotonic sequence."""

        self._check_epoch(epoch)
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        predecessor, successor = _neighbors(self._order, receiver)
        if sender not in (predecessor, successor):
            if sender not in self._order:
                raise UnknownMember(sender)
            raise NonNeighbor(
                f"{sender!r} is not an immediate neighbor of {receiver!r}"
            )
        key = (epoch, sender, receiver)
        previous = self._last_sequence.get(key)
        if previous is not None:
            if sequence == previous:
                return MessageDisposition.DUPLICATE
            if sequence < previous:
                raise StaleSequence(
                    f"sequence {sequence} predates accepted sequence {previous}"
                )
        self._last_sequence[key] = sequence
        return MessageDisposition.ACCEPTED
