"""
Behavior that is probably wrong, but that chemcalc keeps for compatibility.

Course content was written against chemcalc for years, so changing any of these
could change how existing answers are rendered or graded. Each test below pins
one quirk and is named after it. The same name appears:

* in a comment in chemcalc.py where the behavior comes from, and
* in the "quirks" field of chemcalc_baseline.json entries whose results depend
  on it. Those entries are checked by TestChemcalcBaselineQuirks, separately
  from the rest of the baseline.

To fix a quirk on purpose:

1. Change chemcalc, and remove the comment that names the quirk.
2. Replace its test here with one for the new behavior, in test_all.py.
3. Update the baseline entries tagged with it, and remove the tag from them.
   TestChemcalcBaselineQuirks lists the ones that changed.

Everything else should keep passing unchanged. If something else fails, it's a
regression, or a baseline entry that should have been tagged.
"""
# pylint: disable=missing-function-docstring
import unittest
from fractions import Fraction

from chem.chemcalc import chemical_equations_equal, divide_chemical_expression, render_to_html


class TestCompatibilityQuirks(unittest.TestCase):
    """
    One test per quirk. Each test's name, with dashes for underscores, is the
    quirk's name, which test_chemcalc_baseline checks against the baseline's tags.
    """

    def test_charge_starting_with_1(self):
        """
        Quirk "charge-starting-with-1": a charge whose number starts with 1
        loses its number, so ^12+ is read as ^+.

        Only an explicit charge of exactly 1 should be dropped.
        """
        self.assertEqual(render_to_html("Fe^12+"), '<span class="math">Fe<sup>+</sup></span>')
        self.assertTrue(chemical_equations_equal("Fe^12+ -> Fe", "Fe^+ -> Fe"))

    def test_space_inside_number(self):
        """
        Quirk "space-inside-number": spaces between digits are ignored, so
        "1 0H2" is read as 10H2 and "H1 2O" as H12O.

        Spaces between other tokens, as in "2 H2" or "H 2O", are fine.
        """
        self.assertEqual(divide_chemical_expression("1 0H2", "10H2"), 1)
        self.assertEqual(render_to_html("H1 2O"), '<span class="math">H<sub>12</sub>O</span>')

    def test_float_factors(self):
        """
        Quirk "float-factors": divide_chemical_expression works out ratios
        with floats, not exact fractions.

        So H2O / 3H2O returns the value of the float nearest to 1/3, rather
        than Fraction(1, 3) as its docstring says. Rounding can also make
        proportional fractional counts look not proportional.
        """
        self.assertEqual(divide_chemical_expression("H2O", "3H2O"), Fraction(1 / 3))
        self.assertNotEqual(divide_chemical_expression("H2O", "3H2O"), Fraction(1, 3))
        # These are proportional, by a factor of 1/10.
        self.assertFalse(divide_chemical_expression("7/10H2 + 1/10O2", "7H2 + O2"))
        self.assertFalse(chemical_equations_equal("7/10H2 + 1/10O2 -> 7/10H2O2",
                                                  "7H2 + O2 -> 7H2O2"))

    def test_zero_count_raises(self):
        """
        Quirk "zero-count-raises": a count of 0 in the second expression raises
        ZeroDivisionError.

        That includes chemical_equations_equal, which otherwise returns False
        for input it can't compare.
        """
        with self.assertRaises(ZeroDivisionError):
            divide_chemical_expression("H2O", "0H2O")
        with self.assertRaises(ZeroDivisionError):
            chemical_equations_equal("H2 -> H2", "0H2 -> 0H2")
