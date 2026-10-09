"""The neighbor view an agent starts with must not license a crossing.

Every agent initialises its belief about its two neighbors before any message
has arrived. If that belief is a perfectly even ring (``phase0 -/+ target``),
the displacement limiter clips the first step against a believed gap that, on
random initial phases, is often several times the true gap, so the first step
can carry the job straight past a neighbor. Neighbors are bound to ports by
index, so the crossing is permanent and the rule then corrects the wrong way.

The damage is done on the first tick at which exactly one neighbor has been
heard from, which is the common case under jitter and loss: one side of the
correction is then a real gap and the other is the fabricated ``target``, so
the correction is non-zero and is clipped against the fabricated side.

Invariant tested: on a ring of random initial phases, no agent's first
displacement may exceed the true gap to the neighbor it moves toward. The
shipped-before behaviour violates it; believing the gap to an unheard
neighbor is zero satisfies it, because the step is then zero until both
neighbors have been heard.

The source guard keeps every copy of the agent honest: the artifact carries
four independent agent implementations (two Python simulation/testbed agents,
the Rust agent, and the Rust-equivalence Python agent) and the defect was in
all of them at once.
"""
import math
import os
import random
import re
import unittest

from experiments.simulation.desync_distributed import local_displacement
from kadence.neighbor_gossip import forward_gap

TWO_PI = 2 * math.pi
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every file that starts an agent's neighbor view. Each one must initialise to
# "not heard from yet", never to an assumed even ring.
AGENT_SOURCES = (
    os.path.join("experiments", "simulation", "desync_distributed.py"),
    os.path.join("experiments", "testbed", "cloudlab", "kadence_agent_counted.py"),
    os.path.join("src", "kadence-rs", "equiv", "py_agent.py"),
    os.path.join("src", "kadence-rs", "src", "main.rs"),
)

# (phase0 - target), (phase0 + cfg.target), ... in Python or Rust.
_EVEN_RING_INIT = re.compile(r"phase0\s*[-+]\s*(?:cfg\.)?target")


def _ring(rng, n):
    return sorted(rng.uniform(0.0, TWO_PI) for _ in range(n))


def _even_ring_view(phase, target, side):
    """The pre-fix belief when only ``side`` has actually been heard from."""
    if side == "left":
        return phase, (phase + target) % TWO_PI       # right still fabricated
    return (phase - target) % TWO_PI, phase           # left still fabricated


def _pessimistic_view(phase, target, side):
    """The fixed belief: an unheard neighbor is ``None``, whatever the side."""
    return (phase, None) if side == "left" else (None, phase)


def _overshoot(phases, view, target, alpha):
    """Largest amount by which a first step exceeds its true neighbor gap.

    Each agent is examined in both partial-knowledge states: left neighbor
    heard and right not, then the mirror image. ``view`` maps the heard
    neighbor's true phase to the pair of believed phases the agent would use.
    """
    n = len(phases)
    worst = 0.0
    for i, phase in enumerate(phases):
        left_true = phases[(i - 1) % n]
        right_true = phases[(i + 1) % n]
        for side, heard in (("left", left_true), ("right", right_true)):
            believed_left, believed_right = view(heard, target, side)
            step = local_displacement(believed_left, phase, believed_right,
                                      target, alpha)
            if step >= 0.0:
                worst = max(worst, step - forward_gap(phase, right_true, TWO_PI))
            else:
                worst = max(worst, -step - forward_gap(left_true, phase, TWO_PI))
    return worst


class FirstStepSafetyTests(unittest.TestCase):
    def test_unheard_neighbor_means_no_motion(self):
        """With no snapshot of a neighbor the believed gap is zero."""
        for phase in (0.0, 1.0, 3.3, 6.2):
            self.assertEqual(
                local_displacement(None, phase, None, TWO_PI / 16, 1.0), 0.0)
            self.assertEqual(
                local_displacement(phase - 0.1, phase, None, TWO_PI / 16, 1.0), 0.0)
            self.assertEqual(
                local_displacement(None, phase, phase + 0.1, TWO_PI / 16, 1.0), 0.0)

    def test_first_displacement_never_exceeds_true_gap(self):
        """The invariant, over the whole CloudLab ladder of ring sizes."""
        rng = random.Random(20261009)
        for n in (8, 16, 48):
            target = TWO_PI / n
            for _ in range(200):
                phases = _ring(rng, n)
                self.assertEqual(
                    _overshoot(phases, _pessimistic_view, target, 1.0), 0.0,
                    f"n={n}: a first step moved toward an unheard neighbor")

    def test_even_ring_init_violates_the_invariant(self):
        """The pre-fix initialisation fails the same invariant, so the test
        above discriminates rather than passing trivially."""
        rng = random.Random(20261009)
        for n in (8, 16, 48):
            target = TWO_PI / n
            violating = sum(
                1 for _ in range(200)
                if _overshoot(_ring(rng, n), _even_ring_view, target, 1.0) > 0.0)
            self.assertGreater(violating, 0,
                               f"n={n}: expected the even-ring init to overshoot")


class AgentSourceTests(unittest.TestCase):
    def test_no_agent_assumes_an_even_ring_before_its_first_message(self):
        for rel in AGENT_SOURCES:
            path = os.path.join(_ROOT, rel)
            with self.subTest(source=rel):
                self.assertTrue(os.path.exists(path), path)
                with open(path) as handle:
                    hit = _EVEN_RING_INIT.search(handle.read())
                self.assertIsNone(
                    hit, f"{rel} seeds its neighbor view with an assumed even "
                         f"ring ({hit.group(0) if hit else ''})")


if __name__ == "__main__":
    unittest.main()
