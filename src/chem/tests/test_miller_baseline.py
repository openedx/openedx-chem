"""
Check miller against the Miller indices chem 2.0.1 computed for the same answers.

chem 2.0.1 was the last release to do miller's vector arithmetic with numpy,
and crystallography problems written while it was in use depend on how it
graded them. Its results depended on the CPU, though. See
generate_miller_baseline.py for why, for how miller_baseline.json was made, and
before changing it.

Answers whose results depended on the CPU are checked separately, in
TestMillerBaselineFused and TestMillerBaselineRetry.
"""
# pylint: disable=missing-function-docstring
import json
import unittest

from chem import miller
from chem.tests.generate_miller_baseline import BASELINE_PATH, computed_indices

with open(BASELINE_PATH, encoding="utf-8") as baseline_file:
    BASELINE = json.load(baseline_file)

ZERO_DIVISION = {"raises": "ZeroDivisionError"}


class TestMillerBaseline(unittest.TestCase):
    """
    Answers that chem 2.0.1 computed the same result for on every CPU.

    miller must compute that result too.
    """
    advice = "chem 2.0.1 computed these on every CPU, so this is a regression."

    def cases(self):
        """
        Return the points and the expected result of each answer to check.
        """
        return [(case["points"], case["indices"])
                for case in BASELINE["answers"] if "indices" in case]

    def test_indices(self):
        # Report every mismatch, not just the first.
        cases = self.cases()
        self.assertTrue(cases)
        mismatches = []
        for points, expected in cases:
            actual = computed_indices(miller, points)
            if actual != expected:
                mismatches.append(f"{points}: expected {expected!r}, got {actual!r}")
        if mismatches:
            self.fail(
                f"{len(mismatches)} of {len(cases)} answers differ from chem 2.0.1. "
                f"{self.advice}\n" + "\n".join(mismatches[:20])
            )


class TestMillerBaselineFused(TestMillerBaseline):
    """
    Answers that chem 2.0.1 computed a result of its own for on CPUs with
    AVX-512, where np.dot did fused arithmetic.

    miller must compute that result on every CPU, not the one other CPUs
    computed. These are the first to fail if miller._dot_fused changes.
    """
    advice = ("chem 2.0.1 computed these with fused arithmetic. Has "
              "miller._dot_fused changed, or does miller() no longer try it first?")

    def cases(self):
        return [(case["points"], case["fused"])
                for case in BASELINE["answers"]
                if "fused" in case and case["fused"] != ZERO_DIVISION]


class TestMillerBaselineRetry(TestMillerBaseline):
    """
    Answers that chem 2.0.1 raised ZeroDivisionError for on CPUs with AVX-512,
    and computed Miller indices for on other CPUs, where np.dot did plain
    arithmetic.

    miller must compute those indices on every CPU. These are the first to
    fail if miller._dot_plain changes.
    """
    advice = ("chem 2.0.1 computed these with plain arithmetic. Has "
              "miller._dot_plain changed, or does miller() no longer retry with it?")

    def cases(self):
        return [(case["points"], case["plain"])
                for case in BASELINE["answers"]
                if "fused" in case and case["fused"] == ZERO_DIVISION]


class TestMillerBaselineFile(unittest.TestCase):
    """
    Every answer in the baseline is checked by one of the classes above.
    """

    def test_each_answer_has_one_result_or_two_different_ones(self):
        for case in BASELINE["answers"]:
            with self.subTest(points=case["points"]):
                if "indices" in case:
                    self.assertEqual(set(case), {"points", "indices"})
                else:
                    self.assertEqual(set(case), {"points", "fused", "plain"})
                    self.assertNotEqual(case["fused"], case["plain"])
