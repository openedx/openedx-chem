# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods

import importlib.util
import threading
import unittest
from fractions import Fraction

import chem.miller
from chem.chemcalc import (MAX_NESTING, Tree, chemical_equations_equal,
                           compare_chemical_expression,
                           divide_chemical_expression, render_to_html)
from chem.chemtools import (vsepr_build_correct_answer, vsepr_grade,
                            vsepr_parse_user_answer)


class TestCompareEquations(unittest.TestCase):
    def test_simple_equation(self):
        self.assertTrue(chemical_equations_equal('H2 + O2 -> H2O2',
                                                 'O2 + H2 -> H2O2'))
        # left sides don't match
        self.assertFalse(chemical_equations_equal('H2 + O2 -> H2O2',
                                                  'O2 + 2H2 -> H2O2'))
        # right sides don't match
        self.assertFalse(chemical_equations_equal('H2 + O2 -> H2O2',
                                                  'O2 + H2 -> H2O'))

        # factors don't match
        self.assertFalse(chemical_equations_equal('H2 + O2 -> H2O2',
                                                  'O2 + H2 -> 2H2O2'))

    def test_different_factor(self):
        self.assertTrue(chemical_equations_equal('H2 + O2 -> H2O2',
                                                 '2O2 + 2H2 -> 2H2O2'))

        self.assertFalse(
            chemical_equations_equal(
                '2H2 + O2 -> H2O2',
                '2O2 + 2H2 -> 2H2O2',
            )
        )

    def test_different_arrows(self):
        self.assertTrue(chemical_equations_equal('H2 + O2 -> H2O2',
                                                 '2O2 + 2H2 -> 2H2O2'))

        self.assertFalse(chemical_equations_equal('H2 + O2 -> H2O2',
                                                  'O2 + H2 <-> 2H2O2'))

    def test_exact_match(self):
        self.assertTrue(chemical_equations_equal('H2 + O2 -> H2O2',
                                                 '2O2 + 2H2 -> 2H2O2'))

        self.assertFalse(
            chemical_equations_equal(
                'H2 + O2 -> H2O2',
                '2O2 + 2H2 -> 2H2O2',
                exact=True,
            )
        )

        # order still doesn't matter
        self.assertTrue(chemical_equations_equal('H2 + O2 -> H2O2',
                                                 'O2 + H2 -> H2O2', exact=True))

    def test_syntax_errors(self):
        self.assertFalse(chemical_equations_equal('H2 + O2 a-> H2O2',
                                                  '2O2 + 2H2 -> 2H2O2'))

        self.assertFalse(chemical_equations_equal('H2O( -> H2O2',
                                                  'H2O -> H2O2'))

        self.assertFalse(chemical_equations_equal('H2 + O2 ==> H2O2',   # strange arrow
                                                  '2O2 + 2H2 -> 2H2O2'))

        # Braces used to raise ValueError instead of counting as a syntax error.
        self.assertFalse(chemical_equations_equal('H2 + O2 -> {H2O2}',
                                                  'H2 + O2 -> H2O2'))


class TestCompareExpressions(unittest.TestCase):

    def test_compare_incorrect_order_of_atoms_in_molecule(self):
        self.assertFalse(compare_chemical_expression("H2O + CO2", "O2C + OH2"))

    def test_compare_same_order_no_phases_no_factors_no_ions(self):
        self.assertTrue(compare_chemical_expression("H2O + CO2", "CO2+H2O"))

    def test_compare_different_order_no_phases_no_factors_no_ions(self):
        self.assertTrue(compare_chemical_expression("H2O + CO2", "CO2 + H2O"))

    def test_compare_different_order_three_multimolecule(self):
        self.assertTrue(compare_chemical_expression("H2O + Fe(OH)3 +  CO2", "CO2 + H2O + Fe(OH)3"))

    def test_compare_same_factors(self):
        self.assertTrue(compare_chemical_expression("3H2O +  2CO2", "2CO2 + 3H2O "))

    def test_compare_different_factors(self):
        self.assertFalse(compare_chemical_expression("2H2O +  3CO2", "2CO2 + 3H2O "))

    def test_compare_correct_ions(self):
        self.assertTrue(compare_chemical_expression("H^+ + OH^-", " OH^- + H^+ "))

    def test_compare_wrong_ions(self):
        self.assertFalse(compare_chemical_expression("H^+ + OH^-", " OH^- + H^- "))

    def test_compare_parent_groups_ions(self):
        self.assertTrue(compare_chemical_expression("Fe(OH)^2- + (OH)^-", " (OH)^- + Fe(OH)^2- "))

    def test_compare_correct_factors_ions_and_one(self):
        self.assertTrue(compare_chemical_expression("3H^+ + 2OH^-", " 2OH^- + 3H^+ "))

    def test_compare_wrong_factors_ions(self):
        self.assertFalse(compare_chemical_expression("2H^+ + 3OH^-", " 2OH^- + 3H^+ "))

    def test_compare_float_factors(self):
        self.assertTrue(compare_chemical_expression("7/2H^+ + 3/5OH^-", " 3/5OH^- + 7/2H^+ "))

    # Phases tests
    def test_compare_phases_ignored(self):
        self.assertTrue(compare_chemical_expression(
            "H2O(s) + CO2", "H2O+CO2", ignore_state=True))

    def test_compare_phases_not_ignored_explicitly(self):
        self.assertFalse(compare_chemical_expression(
            "H2O(s) + CO2", "H2O+CO2", ignore_state=False))

    def test_compare_phases_not_ignored(self):  # same as previous
        self.assertFalse(compare_chemical_expression(
            "H2O(s) + CO2", "H2O+CO2"))

    # all in one cases
    def test_complex_additivity(self):
        self.assertTrue(compare_chemical_expression(
            "5(H1H212)^70010- + 2H20 + 7/2HCl + H2O",
            "7/2HCl + 2H20 + H2O + 5(H1H212)^70010-"))

    def test_complex_additivity_wrong(self):
        self.assertFalse(compare_chemical_expression(
            "5(H1H212)^70010- + 2H20 + 7/2HCl + H2O",
            "2H20 + 7/2HCl + H2O + 5(H1H212)^70011-"))

    def test_complex_all_grammar(self):
        self.assertTrue(compare_chemical_expression(
            "5[Ni(NH3)4]^2+ + 5/2SO4^2-",
            "5/2SO4^2- + 5[Ni(NH3)4]^2+"))

    # special cases

    def test_compare_one_superscript_explicitly_set(self):
        self.assertTrue(compare_chemical_expression("H^+ + OH^1-", " OH^- + H^+ "))

    def test_compare_equal_factors_differently_set(self):
        self.assertTrue(compare_chemical_expression("6/2H^+ + OH^-", " OH^- + 3H^+ "))

    def test_compare_one_subscript_explicitly_set(self):
        self.assertFalse(compare_chemical_expression("H2 + CO2", "H2 + C102"))

    def test_compare_bracketed_molecule(self):
        # A molecule that is just a bracketed group still compares by its contents.
        self.assertTrue(compare_chemical_expression("(OH) + [Ar]", "[Ar] + (OH)"))
        self.assertFalse(compare_chemical_expression("(OH)", "(H2)"))
        self.assertFalse(compare_chemical_expression("[Ar]", "[Kr]"))


class TestDivideExpressions(unittest.TestCase):
    ''' as compare_ use divide_,
    tests here must consider different
    division (not equality) cases '''

    def test_divide_by_zero(self):
        self.assertFalse(divide_chemical_expression(
            "0H2O", "H2O"))

    def test_divide_wrong_factors(self):
        self.assertFalse(divide_chemical_expression(
            "5(H1H212)^70010- + 10H2O", "5H2O + 10(H1H212)^70010-"))

    def test_divide_right(self):
        self.assertEqual(divide_chemical_expression(
            "5(H1H212)^70010- + 10H2O", "10H2O + 5(H1H212)^70010-"), 1)

    def test_divide_wrong_reagents(self):
        self.assertFalse(divide_chemical_expression(
            "H2O + CO2", "CO2"))

    def test_divide_right_simple(self):
        self.assertEqual(divide_chemical_expression(
            "H2O + CO2", "H2O+CO2"), 1)

    def test_divide_right_phases(self):
        self.assertEqual(divide_chemical_expression(
            "H2O(s) + CO2", "2H2O(s)+2CO2"), Fraction(1, 2))

    def test_divide_right_phases_other_order(self):
        self.assertEqual(divide_chemical_expression(
            "2H2O(s) + 2CO2", "H2O(s)+CO2"), 2)

    def test_divide_wrong_phases(self):
        self.assertFalse(divide_chemical_expression(
            "H2O(s) + CO2", "2H2O+2CO2(s)"))

    def test_divide_wrong_phases_but_phases_ignored(self):
        self.assertEqual(divide_chemical_expression(
            "H2O(s) + CO2", "2H2O+2CO2(s)", ignore_state=True), Fraction(1, 2))

    def test_divide_order(self):
        self.assertEqual(divide_chemical_expression(
            "2CO2 + H2O", "2H2O+4CO2"), Fraction(1, 2))

    def test_divide_fract_to_int(self):
        self.assertEqual(divide_chemical_expression(
            "3/2CO2 + H2O", "2H2O+3CO2"), Fraction(1, 2))

    def test_divide_fract_to_frac(self):
        self.assertEqual(divide_chemical_expression(
            "3/4CO2 + H2O", "2H2O+9/6CO2"), Fraction(1, 2))

    def test_divide_fract_to_frac_wrog(self):
        self.assertFalse(divide_chemical_expression(
            "6/2CO2 + H2O", "2H2O+9/6CO2"), 2)


class TestRenderEquations(unittest.TestCase):
    """
    Tests to validate the HTML rendering of plaintext (input) equations
    """
    # pylint: disable=line-too-long
    def test_render1(self):
        test_string = "H2O + CO2"
        out = render_to_html(test_string)
        correct = '<span class="math">H<sub>2</sub>O+CO<sub>2</sub></span>'
        self.assertEqual(out, correct)

    def test_render_uncorrect_reaction(self):
        test_string = "O2C + OH2"
        out = render_to_html(test_string)
        correct = '<span class="math">O<sub>2</sub>C+OH<sub>2</sub></span>'
        self.assertEqual(out, correct)

    def test_render2(self):
        test_string = "CO2 + H2O + Fe(OH)3"
        out = render_to_html(test_string)
        correct = '<span class="math">CO<sub>2</sub>+H<sub>2</sub>O+Fe(OH)<sub>3</sub></span>'
        self.assertEqual(out, correct)

    def test_render3(self):
        test_string = "3H2O + 2CO2"
        out = render_to_html(test_string)
        correct = '<span class="math">3H<sub>2</sub>O+2CO<sub>2</sub></span>'
        self.assertEqual(out, correct)

    def test_render4(self):
        test_string = "H^+ + OH^-"
        out = render_to_html(test_string)
        correct = '<span class="math">H<sup>+</sup>+OH<sup>-</sup></span>'
        self.assertEqual(out, correct)

    def test_render5(self):
        test_string = "Fe(OH)^2- + (OH)^-"
        out = render_to_html(test_string)
        correct = '<span class="math">Fe(OH)<sup>2-</sup>+(OH)<sup>-</sup></span>'
        self.assertEqual(out, correct)

    def test_render6(self):
        test_string = "7/2H^+ + 3/5OH^-"
        out = render_to_html(test_string)
        correct = '<span class="math"><sup>7</sup>&frasl;<sub>2</sub>H<sup>+</sup>+<sup>3</sup>&frasl;<sub>5</sub>OH<sup>-</sup></span>'
        self.assertEqual(out, correct)

    def test_render7(self):
        test_string = "5(H1H212)^70010- + 2H2O + 7/2HCl + H2O"
        out = render_to_html(test_string)
        correct = '<span class="math">5(H<sub>1</sub>H<sub>212</sub>)<sup>70010-</sup>+2H<sub>2</sub>O+<sup>7</sup>&frasl;<sub>2</sub>HCl+H<sub>2</sub>O</span>'
        self.assertEqual(out, correct)

    def test_render8(self):
        test_string = "H2O(s) + CO2"
        out = render_to_html(test_string)
        correct = '<span class="math">H<sub>2</sub>O(s)+CO<sub>2</sub></span>'
        self.assertEqual(out, correct)

    def test_render9(self):
        test_string = "5[Ni(NH3)4]^2+ + 5/2SO4^2-"
        out = render_to_html(test_string)
        correct = '<span class="math">5[Ni(NH<sub>3</sub>)<sub>4</sub>]<sup>2+</sup>+<sup>5</sup>&frasl;<sub>2</sub>SO<sub>4</sub><sup>2-</sup></span>'
        self.assertEqual(out, correct)

    def test_render_error(self):
        test_string = "5.2H20"
        out = render_to_html(test_string)
        correct = '<span class="math"><span class="inline-error inline">5.2H20</span></span>'
        self.assertEqual(out, correct)

    def test_render_simple_round_brackets(self):
        test_string = "(Ar)"
        out = render_to_html(test_string)
        correct = '<span class="math">(Ar)</span>'
        self.assertEqual(out, correct)

    def test_render_simple_square_brackets(self):
        test_string = "[Ar]"
        out = render_to_html(test_string)
        correct = '<span class="math">[Ar]</span>'
        self.assertEqual(out, correct)

    def test_render_eq1(self):
        test_string = "H^+ + OH^- -> H2O"
        out = render_to_html(test_string)
        correct = '<span class="math">H<sup>+</sup>+OH<sup>-</sup>\u2192H<sub>2</sub>O</span>'
        self.assertEqual(out, correct)

    def test_render_eq2(self):
        test_string = "H^+ + OH^- <-> H2O"
        out = render_to_html(test_string)
        correct = '<span class="math">H<sup>+</sup>+OH<sup>-</sup>\u2194H<sub>2</sub>O</span>'
        self.assertEqual(out, correct)

    def test_render_eq3(self):
        test_string = "H^+ + OH^- <= H2O"   # unsupported arrow
        out = render_to_html(test_string)
        correct = '<span class="math"><span class="inline-error inline">H^+ + OH^- &lt;= H2O</span></span>'
        self.assertEqual(out, correct)

    def test_render_eq4(self):
        test_string = "[H^+] + OH^- <-> (H2O)"  # with brackets
        out = render_to_html(test_string)
        correct = '<span class="math">[H<sup>+</sup>]+OH<sup>-</sup>\u2194(H<sub>2</sub>O)</span>'
        self.assertEqual(out, correct)

    def test_render_braces(self):
        test_string = "{H2O}"
        out = render_to_html(test_string)
        correct = '<span class="math"><span class="inline-error inline">{H2O}</span></span>'
        self.assertEqual(out, correct)

    def test_render_deeply_nested(self):
        test_string = "(" * 500 + "H" + ")" * 500
        out = render_to_html(test_string)
        self.assertIn('<span class="inline-error inline">', out)
        self.assertFalse(chemical_equations_equal(test_string + " -> H", test_string + " -> H"))

    def test_render_nesting_limit(self):
        error = '<span class="inline-error inline">'
        at_limit = "(" * MAX_NESTING + "H" + ")" * MAX_NESTING
        self.assertNotIn(error, render_to_html(at_limit))
        self.assertTrue(compare_chemical_expression(at_limit, at_limit))
        self.assertIn(error, render_to_html("(" + at_limit + ")"))
        # Round and square brackets count towards the same limit.
        self.assertIn(error, render_to_html("[" + at_limit + "]"))
        # Only brackets that are open at the same time count.
        self.assertNotIn(error, render_to_html("(OH)" * (MAX_NESTING + 1)))

    def test_escaping(self):
        """
        Tests that invalid input is escaped.
        """
        test_string = "<script>f()</script>"
        out = render_to_html(test_string)
        correct = '<span class="math"><span class="inline-error inline">&lt;script&gt;f()&lt;/script&gt;</span></span>'
        self.assertEqual(out, correct)


class TestTree(unittest.TestCase):
    """
    Tree replaced nltk.tree.Tree, and should compare and sort the same way.
    """
    def test_equality_includes_label(self):
        self.assertEqual(Tree('element', ['H']), Tree('element', ['H']))
        self.assertNotEqual(Tree('element', ['H']), Tree('phase', ['H']))
        self.assertNotEqual(Tree('element', ['H']), ['H'])

    def test_ordering(self):
        h, o = Tree('element', ['H']), Tree('element', ['O'])
        self.assertLess(h, o)
        self.assertLessEqual(h, h)
        self.assertGreater(o, h)
        self.assertGreaterEqual(o, o)
        # Label first, then children
        self.assertLess(Tree('element', ['Z']), Tree('phase', ['A']))
        # Trees sort before strings
        self.assertLess(h, 'H')
        self.assertGreater('H', h)
        self.assertEqual(sorted(['H', o, h]), [h, o, 'H'])

    def test_str(self):
        tree = Tree('group', [Tree('element', ['H']), Tree('element', ['O'])])
        self.assertEqual(str(tree), '(group (element H) (element O))')


class TestThreadSafety(unittest.TestCase):
    """
    The first parse in a process can come from several threads at once. For
    example, a problem with two chemical equation inputs asks for both previews
    as the page loads.
    """
    def first_parse_errors(self, threads):
        """
        Parse in several threads at once, and return the exceptions raised.
        """
        # Load a new copy of chemcalc, so that its grammar hasn't parsed anything yet.
        spec = importlib.util.find_spec("chem.chemcalc")
        chemcalc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chemcalc)

        barrier = threading.Barrier(threads)
        errors = []

        def parse():
            barrier.wait()
            try:
                chemcalc.render_to_html("2H2O(l) + [Cu(NH3)4]^2+")
            except Exception as err:  # pylint: disable=broad-exception-caught
                errors.append(err)

        workers = [threading.Thread(target=parse) for _ in range(threads)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        return errors

    def test_first_parse_in_several_threads(self):
        # When the grammar isn't thread-safe, one attempt fails about four
        # times in five, so make several.
        for _ in range(10):
            self.assertEqual(self.first_parse_errors(threads=8), [])


class TestCrystallographyMiller(unittest.TestCase):
    """Tests  for crystallography grade function."""
    # pylint: disable=line-too-long
    def test_empty_points(self):
        user_input = '{"lattice": "bcc", "points": []}'
        self.assertFalse(chem.miller.grade(user_input, {'miller': '(2,2,2)', 'lattice': 'bcc'}))

    def test_only_one_point(self):
        user_input = '{"lattice": "bcc", "points": [["0.50", "0.00", "0.00"]]}'
        self.assertFalse(chem.miller.grade(user_input, {'miller': '(2,2,2)', 'lattice': 'bcc'}))

    def test_only_two_points(self):
        user_input = '{"lattice": "bcc", "points": [["0.50", "0.00", "0.00"], ["0.00", "0.50", "0.00"]]}'
        self.assertFalse(chem.miller.grade(user_input, {'miller': '(2,2,2)', 'lattice': 'bcc'}))

    def test_1(self):
        user_input = '{"lattice": "bcc", "points": [["0.50", "0.00", "0.00"], ["0.00", "0.50", "0.00"], ["0.00", "0.00", "0.50"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(2,2,2)', 'lattice': 'bcc'}))

    def test_2(self):
        user_input = '{"lattice": "bcc", "points": [["1.00", "0.00", "0.00"], ["0.00", "1.00", "0.00"], ["0.00", "0.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(1,1,1)', 'lattice': 'bcc'}))

    def test_3(self):
        user_input = '{"lattice": "bcc", "points": [["1.00", "0.50", "1.00"], ["1.00", "1.00", "0.50"], ["0.50", "1.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(2,2,2)', 'lattice': 'bcc'}))

    def test_4(self):
        user_input = '{"lattice": "bcc", "points": [["0.33", "1.00", "0.00"], ["0.00", "0.664", "0.00"], ["0.00", "1.00", "0.33"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(-3, 3, -3)', 'lattice': 'bcc'}))

    def test_5(self):
        """ return true only in case points coordinates are exact.
        But if they transform to closest 0.05 value it is not true"""
        user_input = '{"lattice": "bcc", "points": [["0.33", "1.00", "0.00"], ["0.00", "0.33", "0.00"], ["0.00", "1.00", "0.33"]]}'
        self.assertFalse(chem.miller.grade(user_input, {'miller': '(-6,3,-6)', 'lattice': 'bcc'}))

    def test_6(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "0.25", "0.00"], ["0.25", "0.00", "0.00"], ["0.00", "0.00", "0.25"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(4,4,4)', 'lattice': 'bcc'}))

    def test_7(self):  # goes throug origin
        user_input = '{"lattice": "bcc", "points": [["0.00", "1.00", "0.00"], ["1.00", "0.00", "0.00"], ["0.50", "1.00", "0.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(0,0,-1)', 'lattice': 'bcc'}))

    def test_8(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "1.00", "0.50"], ["1.00", "0.00", "0.50"], ["0.50", "1.00", "0.50"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(0,0,2)', 'lattice': 'bcc'}))

    def test_9(self):
        user_input = '{"lattice": "bcc", "points": [["1.00", "0.00", "1.00"], ["0.00", "1.00", "1.00"], ["1.00", "0.00", "0.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(1,1,0)', 'lattice': 'bcc'}))

    def test_10(self):
        user_input = '{"lattice": "bcc", "points": [["1.00", "0.00", "1.00"], ["0.00", "0.00", "0.00"], ["0.00", "1.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(1,1,-1)', 'lattice': 'bcc'}))

    def test_11(self):
        user_input = '{"lattice": "bcc", "points": [["1.00", "0.00", "0.50"], ["1.00", "1.00", "0.00"], ["0.00", "1.00", "0.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(0,1,2)', 'lattice': 'bcc'}))

    def test_12(self):
        user_input = '{"lattice": "bcc", "points": [["1.00", "0.00", "0.50"], ["0.00", "0.00", "0.50"], ["1.00", "1.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(0,1,-2)', 'lattice': 'bcc'}))

    def test_13(self):
        user_input = '{"lattice": "bcc", "points": [["0.50", "0.00", "0.00"], ["0.50", "1.00", "0.00"], ["0.00", "0.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(2,0,1)', 'lattice': 'bcc'}))

    def test_14(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "0.00", "0.00"], ["0.00", "0.00", "1.00"], ["0.50", "1.00", "0.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(2,-1,0)', 'lattice': 'bcc'}))

    def test_15(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "0.00", "0.00"], ["1.00", "1.00", "0.00"], ["0.00", "1.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(1,-1,1)', 'lattice': 'bcc'}))

    def test_16(self):
        user_input = '{"lattice": "bcc", "points": [["1.00", "0.00", "0.00"], ["0.00", "1.00", "0.00"], ["1.00", "1.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(1,1,-1)', 'lattice': 'bcc'}))

    def test_17(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "0.00", "0.00"], ["1.00", "0.00", "1.00"], ["1.00", "1.00", "0.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(-1,1,1)', 'lattice': 'bcc'}))

    def test_18(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "0.00", "0.00"], ["1.00", "1.00", "0.00"], ["0.00", "1.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(1,-1,1)', 'lattice': 'bcc'}))

    def test_19(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "0.00", "0.00"], ["1.00", "1.00", "0.00"], ["0.00", "0.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(-1,1,0)', 'lattice': 'bcc'}))

    def test_20(self):
        user_input = '{"lattice": "bcc", "points": [["1.00", "0.00", "0.00"], ["1.00", "1.00", "0.00"], ["0.00", "0.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(1,0,1)', 'lattice': 'bcc'}))

    def test_21(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "0.00", "0.00"], ["0.00", "1.00", "0.00"], ["1.00", "0.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(-1,0,1)', 'lattice': 'bcc'}))

    def test_22(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "1.00", "0.00"], ["1.00", "1.00", "0.00"], ["0.00", "0.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(0,1,1)', 'lattice': 'bcc'}))

    def test_23(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "0.00", "0.00"], ["1.00", "0.00", "0.00"], ["1.00", "1.00", "1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(0,-1,1)', 'lattice': 'bcc'}))

    def test_24(self):
        user_input = '{"lattice": "bcc", "points": [["0.66", "0.00", "0.00"], ["0.00", "0.66", "0.00"], ["0.00", "0.00", "0.66"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(3,3,3)', 'lattice': 'bcc'}))

    def test_25(self):
        user_input = '{"lattice":"","points":[["0.00","0.00","0.01"],["1.00","1.00","0.01"],["0.00","1.00","1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(1,-1,1)', 'lattice': ''}))

    def test_26(self):
        user_input = '{"lattice":"","points":[["0.00","0.01","0.00"],["1.00","0.00","0.00"],["0.00","0.00","1.00"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(0,-1,0)', 'lattice': ''}))

    def test_27(self):
        """ rounding to 0.35"""
        user_input = '{"lattice":"","points":[["0.33","0.00","0.00"],["0.00","0.33","0.00"],["0.00","0.00","0.33"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(3,3,3)', 'lattice': ''}))

    def test_28(self):
        """ rounding to 0.30"""
        user_input = '{"lattice":"","points":[["0.30","0.00","0.00"],["0.00","0.30","0.00"],["0.00","0.00","0.30"]]}'
        self.assertTrue(chem.miller.grade(user_input, {'miller': '(10,10,10)', 'lattice': ''}))

    def test_wrong_lattice(self):
        user_input = '{"lattice": "bcc", "points": [["0.00", "0.00", "0.00"], ["1.00", "0.00", "0.00"], ["1.00", "1.00", "1.00"]]}'
        self.assertFalse(chem.miller.grade(user_input, {'miller': '(3,3,3)', 'lattice': 'fcc'}))


class TestGrade(unittest.TestCase):
    """
    Test grade functions
    """

    def test_incorrect_geometry(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX4E0',
            atoms={
                'c0': 'N',
                'p0': 'H',
                'p1': '(ep)',
                'p2': 'H',
                'p3': 'H',
            },
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX3E0","atoms":'
            '{"c0": "B","p0": "F","p1": "B","p2": "F"}}'
        )
        self.assertFalse(vsepr_grade(user_answer, correct_answer))

    def test_correct_answer_p(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX4E0',
            atoms={
                'c0': 'N',
                'p0': 'H',
                'p1': '(ep)',
                'p2': 'H',
                'p3': 'H',
            }
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX4E0","atoms":'
            '{"c0": "N","p0": "H","p1": "(ep)","p2": "H", "p3": "H"}}'
        )
        self.assertTrue(vsepr_grade(user_answer, correct_answer))

    def test_correct_answer_ae(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX6E0',
            atoms={
                'c0': 'Br',
                'a0': 'test',
                'a1': '(ep)',
                'e0': 'H',
                'e1': 'H',
                'e2': '(ep)',
                'e3': '(ep)',
            }
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX6E0","atoms":'
            '{"c0": "Br","a0": "test","a1": "(ep)","e10": "H",'
            '"e11": "H","e20": "(ep)","e21": "(ep)"}}'
        )
        self.assertTrue(vsepr_grade(user_answer, correct_answer))

    def test_correct_answer_ae_convert_to_p_but_input_not_in_p(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX6E0',
            atoms={
                'c0': 'Br',
                'a0': '(ep)',
                'a1': 'test',
                'e0': 'H',
                'e1': 'H',
                'e2': '(ep)',
                'e3': '(ep)',
            }
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX6E0","atoms":'
            '{"c0": "Br","a0": "test","a1": "(ep)","e10": "H",'
            '"e11": "(ep)","e20": "H","e21": "(ep)"}}'
        )
        self.assertFalse(
            vsepr_grade(
                user_answer,
                correct_answer,
                convert_to_peripheral=True,
            )
        )

    def test_correct_answer_ae_convert_to_p(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX6E0',
            atoms={
                'c0': 'Br',
                'p0': '(ep)',
                'p1': 'test',
                'p2': 'H',
                'p3': 'H',
                'p4': '(ep)',
                'p6': '(ep)',
            }
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX6E0","atoms":'
            '{"c0": "Br","a0": "test","a1": "(ep)","e10": "H",'
            '"e11": "(ep)","e20": "H","e21": "(ep)"}}'
        )
        self.assertTrue(
            vsepr_grade(
                user_answer,
                correct_answer,
                convert_to_peripheral=True,
            )
        )

    def test_correct_answer_e1e2_in_a(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX6E0',
            atoms={
                'c0': 'Br',
                'a0': '(ep)',
                'a1': '(ep)',
                'e10': 'H',
                'e11': 'H',
                'e20': 'H',
                'e21': 'H',
            }
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX6E0","atoms":'
            '{"c0": "Br","a0": "(ep)","a1": "(ep)","e10": "H",'
            '"e11": "H","e20": "H","e21": "H"}}'
        )
        self.assertTrue(vsepr_grade(user_answer, correct_answer))

    def test_correct_answer_e1e2_in_e1(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX6E0',
            atoms={
                'c0': 'Br',
                'a0': '(ep)',
                'a1': '(ep)',
                'e10': 'H',
                'e11': 'H',
                'e20': 'H',
                'e21': 'H',
            }
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX6E0","atoms":'
            '{"c0": "Br","a0": "H","a1": "H","e10": "(ep)",'
            '"e11": "(ep)","e20": "H","e21": "H"}}'
        )
        self.assertTrue(vsepr_grade(user_answer, correct_answer))

    def test_correct_answer_e1e2_in_e2(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX6E0',
            atoms={
                'c0': 'Br',
                'a0': '(ep)',
                'a1': '(ep)',
                'e10': 'H',
                'e11': 'H',
                'e20': 'H',
                'e21': 'H',
            }
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX6E0","atoms":'
            '{"c0": "Br","a0": "H","a1": "H","e10": "H","e11": "H",'
            '"e20": "(ep)","e21": "(ep)"}}'
        )
        self.assertTrue(vsepr_grade(user_answer, correct_answer))

    def test_incorrect_answer_e1e2(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX6E0',
            atoms={
                'c0': 'Br',
                'a0': '(ep)',
                'a1': '(ep)',
                'e10': 'H',
                'e11': 'H',
                'e20': 'H',
                'e21': 'H',
            }
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX6E0","atoms":'
            '{"c0": "Br","a0": "H","a1": "H","e10": "(ep)",'
            '"e11": "H","e20": "H","e21": "(ep)"}}'
        )
        self.assertFalse(vsepr_grade(user_answer, correct_answer))

    def test_incorrect_c0(self):
        correct_answer = vsepr_build_correct_answer(
            geometry='AX6E0',
            atoms={
                'c0': 'Br',
                'a0': '(ep)',
                'a1': 'test',
                'e0': 'H',
                'e1': 'H',
                'e2': 'H',
                'e3': '(ep)',
            }
        )
        user_answer = vsepr_parse_user_answer(
            '{"geometry": "AX6E0","atoms":'
            '{"c0": "H","a0": "test","a1": "(ep)","e0": "H",'
            '"e1": "H","e2": "(ep)","e3": "H"}}'
        )
        self.assertFalse(vsepr_grade(user_answer, correct_answer))
