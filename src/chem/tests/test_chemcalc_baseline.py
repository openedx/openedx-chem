"""
Check chemcalc against the results chem 2.0.1 gave for the same inputs.

chem 2.0.1 was the last release to parse formulas with nltk, and course content
written while it was in use depends on how it rendered and compared them. See
generate_chemcalc_baseline.py for how chemcalc_baseline.json was made, and
before changing it.

Entries whose results depend on a compatibility quirk (see
test_chemcalc_quirks.py) are checked separately, in TestChemcalcBaselineQuirks.
"""
# pylint: disable=missing-function-docstring
import json
import unittest

from chem import chemcalc
from chem.tests import test_chemcalc_quirks
from chem.tests.generate_chemcalc_baseline import BASELINE_PATH, SECTIONS, results

with open(BASELINE_PATH, encoding="utf-8") as baseline_file:
    BASELINE = json.load(baseline_file)


class TestChemcalcBaseline(unittest.TestCase):
    """
    Baseline entries that don't depend on a compatibility quirk.

    chemcalc is meant to match chem 2.0.1 on all of these, so any mismatch is a
    regression.
    """
    quirk_entries = False
    advice = "These don't depend on a known quirk, so this is a regression."

    def assert_matches_baseline(self, section):
        """
        Check each entry in section, and report every mismatch, not just the first.
        """
        cases = [case for case in BASELINE[section]
                 if bool(case.get("quirks")) == self.quirk_entries]
        self.assertTrue(cases)
        mismatches = []
        for case in cases:
            for field, actual in results(chemcalc, section, case).items():
                expected = case[field]
                # Compare types too, so that a 0 doesn't pass for False.
                if type(actual) is not type(expected) or actual != expected:
                    tags = "".join(f"[{quirk}] " for quirk in case.get("quirks", []))
                    inputs = ", ".join(repr(case[key]) for key in SECTIONS[section])
                    mismatches.append(
                        f"{tags}{field}({inputs}): expected {expected!r}, got {actual!r}"
                    )
        if mismatches:
            self.fail(
                f"{len(mismatches)} results in {section} differ from chem 2.0.1. {self.advice}\n"
                + "\n".join(mismatches[:20])
            )

    def test_render_to_html(self):
        self.assert_matches_baseline("expressions")

    def test_divide_chemical_expression(self):
        self.assert_matches_baseline("comparisons")

    def test_chemical_equations_equal(self):
        self.assert_matches_baseline("equations")


class TestChemcalcBaselineQuirks(TestChemcalcBaseline):
    """
    Baseline entries whose results depend on a compatibility quirk: behavior
    that's probably wrong but kept on purpose. Each entry's "quirks" field names
    the quirks, and test_chemcalc_quirks.py describes them.

    A mismatch here is expected only while fixing one of those quirks on
    purpose, and only in entries tagged with it. Update those entries and remove
    the tag. Any other mismatch is a regression.
    """
    quirk_entries = True
    advice = ("They depend on the quirks in [brackets]. If you're fixing one on purpose, "
              "update the entries tagged with it; otherwise this is a regression.")

    def test_each_quirk_has_a_test(self):
        # Keeps tags in step with test_chemcalc_quirks: when a quirk is fixed
        # and its test removed, this fails until its tags are removed too.
        tags = {quirk for section in SECTIONS for case in BASELINE[section]
                for quirk in case.get("quirks", [])}
        tested = {name[len("test_"):].replace("_", "-")
                  for name in dir(test_chemcalc_quirks.TestCompatibilityQuirks)
                  if name.startswith("test_")}
        self.assertEqual(tags, tested)
