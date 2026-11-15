"""Tests for FormCheck: angle math and the rep counter (pure python)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from formcheck.angles import angle_at
from formcheck.exercises import EXERCISES, PUSHUP, SQUAT, RepCounter


class TestAngles(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(angle_at((1, 0), (0, 0), (0, 1)), 90.0)

    def test_straight_line_is_180(self):
        self.assertAlmostEqual(angle_at((-1, 0), (0, 0), (1, 0)), 180.0)

    def test_folded_is_small(self):
        a = angle_at((2, 0), (0, 0), (1, 0.1))
        self.assertLess(a, 20)

    def test_degenerate_raises(self):
        with self.assertRaises(ValueError):
            angle_at((0, 0), (0, 0), (1, 1))


class FakeRep:
    """Drive a counter through a scripted angle sequence, collect events."""

    def __init__(self, exercise):
        self.counter = RepCounter(exercise)
        self.events = []

    def feed(self, angles):
        for a in angles:
            ev = self.counter.update(a)
            if ev:
                self.events.append(ev)
        return self


class TestRepCounter(unittest.TestCase):
    def test_clean_squat_counts(self):
        # stand (175) -> descend -> deep bottom (95) -> back up
        r = FakeRep(SQUAT).feed([175, 150, 130, 110, 95, 120, 145, 165, 175])
        self.assertEqual(r.counter.reps, 1)
        self.assertEqual(r.counter.clean_reps, 1)
        self.assertIn("rep", r.events)

    def test_shallow_squat_flags_cue(self):
        # dips below down_threshold but never reaches min_depth (100)
        r = FakeRep(SQUAT).feed([175, 145, 125, 120, 135, 150, 170])
        self.assertEqual(r.counter.reps, 1)
        self.assertEqual(r.counter.clean_reps, 0)
        self.assertIn("shallow", r.events)
        self.assertEqual(r.counter.last_cue, SQUAT.cue)

    def test_no_full_cycle_no_rep(self):
        r = FakeRep(SQUAT).feed([175, 170, 172, 168, 175])
        self.assertEqual(r.counter.reps, 0)

    def test_partial_descent_then_back_up_counts_nothing_clean(self):
        # crosses down threshold but never returns above up threshold
        r = FakeRep(PUSHUP).feed([160, 120, 95, 110, 120])
        self.assertEqual(r.counter.reps, 0)
        self.assertEqual(r.counter.state, "down")

    def test_multiple_reps(self):
        one_rep = [170, 130, 88, 140, 168]
        r = FakeRep(PUSHUP).feed(one_rep * 3)
        self.assertEqual(r.counter.reps, 3)
        self.assertEqual(r.counter.clean_reps, 3)

    def test_deepest_point_tracked_across_frames(self):
        # bottom drifts: 105 -> 98 -> 103; deepest is 98, below min_depth 100
        r = FakeRep(SQUAT).feed([175, 130, 105, 98, 103, 130, 170])
        self.assertEqual(r.counter.clean_reps, 1)

    def test_exercises_registered(self):
        self.assertIn("squat", EXERCISES)
        self.assertIn("push-up", EXERCISES)


if __name__ == "__main__":
    unittest.main()
