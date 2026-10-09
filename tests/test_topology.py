import unittest

from kadence.topology import (
    ChangeKind,
    FutureEpoch,
    InvalidTopology,
    LeaseExpired,
    LeaseNotExpired,
    MessageDisposition,
    MissingAcknowledgement,
    NonNeighbor,
    StaleEpoch,
    StaleSequence,
    TopologyAuthority,
    TransactionConflict,
    TransactionStateError,
    TransactionStatus,
    UnknownMember,
)


class TopologyProtocolTests(unittest.TestCase):
    def make_authority(self) -> TopologyAuthority:
        return TopologyAuthority(
            ("a", "b", "c", "d", "e"),
            initial_epoch=7,
            lease_duration=10.0,
        )

    def acknowledge_all(
        self,
        authority: TopologyAuthority,
        transaction_id: str,
        *,
        now: float,
    ) -> None:
        receipt = authority.transaction(transaction_id)
        for jid in receipt.required:
            authority.acknowledge(
                transaction_id, jid, expected_epoch=receipt.base_epoch, now=now,
            )

    def test_job_view_is_strictly_local(self) -> None:
        authority = self.make_authority()
        view = authority.neighbor_view("c")

        self.assertEqual(view.jid, "c")
        self.assertEqual(view.predecessor, "b")
        self.assertEqual(view.successor, "d")
        self.assertEqual(view.epoch, 7)
        self.assertFalse(hasattr(view, "order"))

    def test_join_is_prepared_then_atomically_epoch_committed(self) -> None:
        authority = self.make_authority()
        receipt = authority.prepare_join(
            "join-x",
            new_jid="x",
            left_jid="b",
            right_jid="c",
            expected_epoch=7,
            now=1.0,
            timeout=5.0,
        )

        self.assertEqual(receipt.status, TransactionStatus.PREPARED)
        self.assertEqual(receipt.required, ("b", "x", "c"))
        self.assertEqual(authority.order, ("a", "b", "c", "d", "e"))
        self.assertEqual(authority.epoch, 7)
        self.assertEqual(
            authority.proposal("join-x", "x").proposed_predecessor, "b",
        )
        self.assertEqual(
            authority.proposal("join-x", "x").proposed_successor, "c",
        )
        self.assertIsNone(
            authority.proposal("join-x", "x").current_predecessor,
        )
        with self.assertRaises(MissingAcknowledgement):
            authority.commit("join-x", expected_epoch=7, now=1.5)

        self.acknowledge_all(authority, "join-x", now=2.0)
        committed = authority.commit("join-x", expected_epoch=7, now=2.5)

        self.assertEqual(committed.status, TransactionStatus.COMMITTED)
        self.assertEqual(committed.terminal_epoch, 8)
        self.assertEqual(authority.epoch, 8)
        self.assertEqual(authority.order, ("a", "b", "x", "c", "d", "e"))
        self.assertEqual(authority.neighbor_view("x").predecessor, "b")
        self.assertEqual(authority.neighbor_view("x").successor, "c")

        # Both command replay and prepare replay are idempotent.
        self.assertEqual(
            authority.commit("join-x", expected_epoch=7, now=3.0), committed,
        )
        self.assertEqual(
            authority.prepare_join(
                "join-x",
                new_jid="x",
                left_jid="b",
                right_jid="c",
                expected_epoch=7,
                now=3.0,
                timeout=99.0,
            ),
            committed,
        )
        with self.assertRaises(TransactionConflict):
            authority.prepare_join(
                "join-x",
                new_jid="y",
                left_jid="b",
                right_jid="x",
                expected_epoch=8,
                now=3.0,
                timeout=5.0,
            )

    def test_abort_and_timeout_roll_back_without_an_epoch_change(self) -> None:
        authority = self.make_authority()
        original = authority.order
        authority.prepare_join(
            "abort-me",
            new_jid="x",
            left_jid="d",
            right_jid="e",
            expected_epoch=7,
            now=1.0,
            timeout=5.0,
        )
        aborted = authority.abort("abort-me", reason="participant refused", now=2.0)
        self.assertEqual(aborted.status, TransactionStatus.ABORTED)
        self.assertEqual(authority.order, original)
        self.assertEqual(authority.epoch, 7)
        self.assertEqual(authority.abort("abort-me", now=2.5), aborted)
        with self.assertRaises(TransactionStateError):
            authority.commit("abort-me", expected_epoch=7, now=3.0)

        authority.prepare_join(
            "time-out",
            new_jid="y",
            left_jid="e",
            right_jid="a",
            expected_epoch=7,
            now=3.0,
            timeout=2.0,
        )
        expired = authority.expire(now=5.0)
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].status, TransactionStatus.ABORTED)
        self.assertEqual(expired[0].reason, "timeout")
        self.assertEqual(authority.order, original)
        self.assertEqual(authority.epoch, 7)

    def test_duplicate_acknowledgement_has_no_lease_side_effect(self) -> None:
        authority = self.make_authority()
        authority.prepare_join(
            "join-x",
            new_jid="x",
            left_jid="b",
            right_jid="c",
            expected_epoch=7,
            now=1.0,
            timeout=5.0,
        )
        first = authority.acknowledge(
            "join-x", "b", expected_epoch=7, now=2.0,
        )
        first_deadline = authority.neighbor_view("b").lease_deadline
        replay = authority.acknowledge(
            "join-x", "b", expected_epoch=7, now=4.0,
        )

        self.assertEqual(first, replay)
        self.assertEqual(authority.neighbor_view("b").lease_deadline, first_deadline)

    def test_join_candidate_ack_is_lease_bounded(self) -> None:
        authority = TopologyAuthority(
            ("a", "b", "c", "d"), lease_duration=2.0,
        )
        authority.prepare_join(
            "join-x",
            new_jid="x",
            left_jid="b",
            right_jid="c",
            expected_epoch=0,
            now=0.0,
            timeout=5.0,
        )
        authority.acknowledge("join-x", "x", expected_epoch=0, now=0.1)
        authority.acknowledge("join-x", "b", expected_epoch=0, now=1.9)
        authority.acknowledge("join-x", "c", expected_epoch=0, now=1.9)

        with self.assertRaises(LeaseExpired):
            authority.commit("join-x", expected_epoch=0, now=2.1)
        self.assertEqual(authority.order, ("a", "b", "c", "d"))
        self.assertEqual(authority.epoch, 0)

    def test_graceful_leave_requires_the_departing_job(self) -> None:
        authority = self.make_authority()
        receipt = authority.prepare_leave(
            "leave-c", jid="c", expected_epoch=7, now=1.0, timeout=4.0,
        )
        self.assertEqual(receipt.kind, ChangeKind.LEAVE)
        self.assertEqual(receipt.required, ("b", "c", "d"))
        self.assertIsNone(
            authority.proposal("leave-c", "c").proposed_predecessor,
        )
        for jid in ("b", "d"):
            authority.acknowledge(
                "leave-c", jid, expected_epoch=7, now=1.5,
            )
        with self.assertRaises(MissingAcknowledgement):
            authority.commit("leave-c", expected_epoch=7, now=2.0)
        authority.acknowledge("leave-c", "c", expected_epoch=7, now=2.0)
        authority.commit("leave-c", expected_epoch=7, now=2.1)

        self.assertEqual(authority.order, ("a", "b", "d", "e"))
        self.assertEqual(authority.neighbor_view("b").successor, "d")
        self.assertEqual(authority.neighbor_view("d").predecessor, "b")

    def test_failed_leave_waits_for_expiry_and_never_needs_failed_ack(self) -> None:
        authority = TopologyAuthority(
            ("a", "b", "c", "d", "e"),
            initial_epoch=3,
            lease_duration=5.0,
        )
        with self.assertRaises(LeaseNotExpired):
            authority.prepare_failed_leave(
                "repair-c", jid="c", expected_epoch=3, now=4.9, timeout=3.0,
            )

        # Keep the two surviving endpoints live while c's lease expires.
        authority.renew_member("b", expected_epoch=3, now=4.0)
        authority.renew_member("d", expected_epoch=3, now=4.0)
        receipt = authority.prepare_failed_leave(
            "repair-c", jid="c", expected_epoch=3, now=5.0, timeout=3.0,
        )
        self.assertEqual(receipt.kind, ChangeKind.FAILED_LEAVE)
        self.assertEqual(receipt.required, ("b", "d"))
        self.acknowledge_all(authority, "repair-c", now=5.1)
        authority.commit("repair-c", expected_epoch=3, now=5.2)

        self.assertEqual(authority.order, ("a", "b", "d", "e"))
        self.assertEqual(authority.neighbor_view("b").successor, "d")
        with self.assertRaises(UnknownMember):
            # A fenced zombie cannot renew the removed identity.
            authority.renew_member("c", expected_epoch=4, now=5.2)

    def test_adjacent_swap_changes_only_local_links(self) -> None:
        authority = self.make_authority()
        receipt = authority.prepare_swap(
            "swap-bc",
            left_jid="b",
            right_jid="c",
            expected_epoch=7,
            now=1.0,
            timeout=4.0,
        )
        self.assertEqual(receipt.required, ("a", "b", "c", "d"))
        self.assertEqual(
            (
                authority.proposal("swap-bc", "a").current_successor,
                authority.proposal("swap-bc", "a").proposed_successor,
            ),
            ("b", "c"),
        )
        self.assertEqual(
            (
                authority.proposal("swap-bc", "d").current_predecessor,
                authority.proposal("swap-bc", "d").proposed_predecessor,
            ),
            ("c", "b"),
        )
        self.acknowledge_all(authority, "swap-bc", now=1.5)
        authority.commit("swap-bc", expected_epoch=7, now=2.0)
        self.assertEqual(authority.order, ("a", "c", "b", "d", "e"))

        with self.assertRaises(InvalidTopology):
            authority.prepare_swap(
                "not-adjacent",
                left_jid="a",
                right_jid="b",
                expected_epoch=8,
                now=2.1,
                timeout=2.0,
            )

    def test_epoch_adjacency_and_sequence_fence_neighbor_messages(self) -> None:
        authority = self.make_authority()
        self.assertEqual(
            authority.accept_neighbor_message(
                receiver="b", sender="c", epoch=7, sequence=4,
            ),
            MessageDisposition.ACCEPTED,
        )
        self.assertEqual(
            authority.accept_neighbor_message(
                receiver="b", sender="c", epoch=7, sequence=4,
            ),
            MessageDisposition.DUPLICATE,
        )
        with self.assertRaises(StaleSequence):
            authority.accept_neighbor_message(
                receiver="b", sender="c", epoch=7, sequence=3,
            )
        with self.assertRaises(NonNeighbor):
            authority.accept_neighbor_message(
                receiver="b", sender="d", epoch=7, sequence=5,
            )

        authority.prepare_join(
            "join-x",
            new_jid="x",
            left_jid="b",
            right_jid="c",
            expected_epoch=7,
            now=1.0,
            timeout=3.0,
        )
        self.acknowledge_all(authority, "join-x", now=1.2)
        authority.commit("join-x", expected_epoch=7, now=1.3)
        with self.assertRaises(StaleEpoch):
            authority.accept_neighbor_message(
                receiver="b", sender="c", epoch=7, sequence=5,
            )
        with self.assertRaises(NonNeighbor):
            authority.accept_neighbor_message(
                receiver="b", sender="c", epoch=8, sequence=5,
            )
        with self.assertRaises(FutureEpoch):
            authority.accept_neighbor_message(
                receiver="b", sender="x", epoch=9, sequence=1,
            )

    def test_stale_operations_and_concurrent_prepare_are_fenced(self) -> None:
        authority = self.make_authority()
        authority.prepare_join(
            "first",
            new_jid="x",
            left_jid="a",
            right_jid="b",
            expected_epoch=7,
            now=1.0,
            timeout=4.0,
        )
        with self.assertRaises(TransactionConflict):
            authority.prepare_leave(
                "second", jid="d", expected_epoch=7, now=1.1, timeout=3.0,
            )
        self.acknowledge_all(authority, "first", now=1.2)
        authority.commit("first", expected_epoch=7, now=1.3)

        with self.assertRaises(StaleEpoch):
            authority.prepare_leave(
                "stale", jid="d", expected_epoch=7, now=1.4, timeout=3.0,
            )
        with self.assertRaises(FutureEpoch):
            authority.prepare_leave(
                "future", jid="d", expected_epoch=9, now=1.4, timeout=3.0,
            )


if __name__ == "__main__":
    unittest.main()
