from fractions import Fraction

import markupsafe
from pyparsing import (Char, Forward, Literal, OneOrMore, Opt, ParseException,
                       Regex, StringEnd, Suppress, ZeroOrMore, nums, one_of)

ARROWS = ('<->', '->')

elements = ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be',
            'Bh', 'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm',
            'Cn', 'Co', 'Cr', 'Cs', 'Cu', 'Db', 'Ds', 'Dy', 'Er', 'Es', 'Eu',
            'F', 'Fe', 'Fl', 'Fm', 'Fr', 'Ga', 'Gd', 'Ge', 'H', 'He', 'Hf',
            'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K', 'Kr', 'La', 'Li', 'Lr',
            'Lu', 'Lv', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na', 'Nb', 'Nd',
            'Ne', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd', 'Pm',
            'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh', 'Rn',
            'Ru', 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta',
            'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'Uuo', 'Uup',
            'Uus', 'Uut', 'V', 'W', 'Xe', 'Y', 'Yb', 'Zn', 'Zr']
phases = ["(s)", "(l)", "(g)", "(aq)"]

# The most brackets that can be open at once. No real formula comes close. The
# limit keeps the parser, and the recursive functions that walk its trees, well
# inside Python's recursion limit.
MAX_NESTING = 20

# HTML, Text are temporarily copied from openedx.core.djangolib.markup
# These libraries need to be moved out of edx-platform to be used by
# other applications.
# See LEARNER-5853 for more details.
Text = markupsafe.escape                        # pylint: disable=invalid-name


def HTML(html):                                 # pylint: disable=invalid-name
    return markupsafe.Markup(html)


class Tree(list):
    """
    A parse tree node: a label and a list of children (Trees or strings).

    This covers the parts of nltk.tree.Tree that this module used to rely on.
    It also orders the same way (by label, then children, with Trees before
    strings), since divide_chemical_expression sorts molecules to compare them.
    """

    def __init__(self, label, children=()):
        super().__init__(children)
        self._label = label

    def label(self):
        return self._label

    def subtrees(self, filter=None):  # pylint: disable=redefined-builtin
        """
        Yield this tree and every Tree below it that passes `filter`, depth first.
        """
        if filter is None or filter(self):
            yield self
        for child in self:
            if isinstance(child, Tree):
                yield from child.subtrees(filter)

    def _key(self):
        return (self._label, list(self))

    def __eq__(self, other):
        return isinstance(other, Tree) and self._key() == other._key()

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if isinstance(other, Tree):
            return self._key() < other._key()
        return type(self).__name__ < type(other).__name__

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __str__(self):
        return "({} {})".format(self._label, " ".join(str(child) for child in self))


# The parse actions below all take pyparsing's full (string, location, tokens)
# arguments, though they only use the tokens. If an action takes fewer,
# pyparsing finds out how many by trial and error the first time it's called.
# That isn't thread-safe: threads that parse at the same time in a new process
# can leave the action raising TypeError on every later call.


def _tree(label):
    """
    Make a parse action that wraps all of an expression's tokens in one Tree.
    """
    # pyparsing reads a returned list (which a Tree is) as a list of tokens,
    # so return the Tree inside a list.
    return lambda _string, _location, tokens: [Tree(label, tokens.as_list())]


def _bracketed(label):
    """
    Make a parse action for a bracketed group: its children are the group's.
    """
    return lambda _string, _location, tokens: [Tree(label, tokens[0])]


def _suffixed(_string, _location, tokens):
    """
    Parse action for an element or bracketed group with an optional suffix.

    Only wrap it in a "suffixed" Tree if there is a suffix.
    """
    if len(tokens) > 1:
        return [Tree("suffixed", tokens.as_list())]
    return tokens


def _ion_suffix(_string, _location, tokens):
    """
    Parse action for a charge like ^2+ or ^-.
    """
    children = tokens.as_list()
    if len(children) == 2 and children[0][0].startswith("1"):
        # Drop an explicit charge of 1, so that ^1+ is the same as ^+.
        #
        # Compatibility quirk "charge-starting-with-1" (see
        # test_chemcalc_quirks.py): this only checks the first digit, so ^12+
        # is also read as ^+. That's probably a bug, but fixing it would
        # change how existing answers are rendered and graded.
        children = children[1:]
    return [Tree("ion_suffix", children)]


def _digits():
    """
    One or more digits, read as a single string.

    Compatibility quirk "space-inside-number" (see test_chemcalc_quirks.py):
    each digit is a separate token, so spaces between digits are skipped and
    "1 0" reads as 10.
    """
    digits = OneOrMore(Char(nums))
    return digits.add_parse_action(lambda _string, _location, tokens: "".join(tokens))


# The grammar for a chemical expression: molecules, each with an optional
# count and phase, separated by "+". For example, "2H2O(l) + [Cu(NH3)4]^2+".
# Each parse action turns its tokens into a Tree for _render_to_html and
# divide_chemical_expression to work from.
_number = _digits().add_parse_action(_tree("number"))
_plus_minus = one_of("+ -").set_parse_action(_tree("plus_minus"))
_ion = (Suppress("^") + Opt(_number) + _plus_minus).set_parse_action(_ion_suffix)
_number_suffix = _digits().add_parse_action(_tree("number_suffix"))
_suffix = (_number_suffix + Opt(_ion) | _ion).set_parse_action(_tree("suffix"))

# Try longer symbols first, so that "Co" is cobalt rather than "C" + "o".
_element = Regex("|".join(sorted(elements, key=len, reverse=True)))
_element.set_parse_action(_tree("element"))

_group = Forward()
_round = (Suppress("(") + _group + Suppress(")")).set_parse_action(_bracketed("paren_group_round"))
_square = (Suppress("[") + _group + Suppress("]")).set_parse_action(_bracketed("paren_group_square"))
_group <<= OneOrMore(
    ((_element | _round | _square) + Opt(_suffix)).set_parse_action(_suffixed)
).set_parse_action(_tree("group"))

_count = (_number + Opt(Literal("/") + _number)).set_parse_action(_tree("count"))
_phase = one_of(phases).set_parse_action(_tree("phase"))
_molecule = (_group + Opt(_phase)).set_parse_action(_tree("molecule"))
_multimolecule = (Opt(_count) + _molecule).set_parse_action(_tree("multimolecule"))
_expression = (
    (_multimolecule + ZeroOrMore(Literal("+") + _multimolecule)).set_parse_action(_tree("S"))
    + StringEnd()
)


def _render_to_html(tree):
    """
    Renders a cleaned tree to HTML
    """
    def molecule_count(tree, children):
        # If an integer, return that integer
        if len(tree) == 1:
            return tree[0][0]
        # If a fraction, return the fraction
        if len(tree) == 3:
            return HTML(" <sup>{num}</sup>&frasl;<sub>{den}</sub> ").format(num=tree[0][0], den=tree[2][0])
        return "Error"

    def subscript(tree, children):
        return HTML("<sub>{sub}</sub>").format(sub=children)

    def superscript(tree, children):
        return HTML("<sup>{sup}</sup>").format(sup=children)

    def round_brackets(tree, children):
        return HTML("({insider})").format(insider=children)

    def square_brackets(tree, children):
        return HTML("[{insider}]").format(insider=children)

    dispatch = {'count': molecule_count,
                'number_suffix': subscript,
                'ion_suffix': superscript,
                'paren_group_round': round_brackets,
                'paren_group_square': square_brackets}

    if isinstance(tree, str):
        return tree
    else:
        children = HTML("").join(map(_render_to_html, tree))
        if tree.label() in dispatch:
            return dispatch[tree.label()](tree, children)
        else:
            return children.replace(' ', '')


def render_to_html(eq):
    """
    Render a chemical equation string to html.

    Renders each molecule separately, and returns invalid input wrapped in a <span>.
    """
    def err(s):
        """
        Render as an error span
        """
        return HTML('<span class="inline-error inline">{0}</span>').format(s)

    def render_arrow(arrow):
        """
        Turn text arrows into pretty ones
        """
        if arrow == '->':
            return HTML('\u2192')
        if arrow == '<->':
            return HTML('\u2194')

        # this won't be reached unless we add more arrow types, but keep it to avoid explosions when
        # that happens. HTML-escape this unknown arrow just in case.
        return Text(arrow)

    def render_expression(ex):
        """
        Render a chemical expression--no arrows.
        """
        try:
            return _render_to_html(_get_final_tree(ex))
        except ParseException:
            return err(ex)

    def spanify(s):
        return HTML('<span class="math">{0}</span>').format(s)

    left, arrow, right = split_on_arrow(eq)
    if arrow == '':
        # only one side
        return spanify(render_expression(left))

    return spanify(render_expression(left) + render_arrow(arrow) + render_expression(right))


def _get_final_tree(s):
    """
    Parse a chemical expression (one side of an equation) into a Tree.

    Raises pyparsing.ParseException if s is invalid.
    """
    depth = 0
    for char in s:
        if char in "([":
            depth += 1
            if depth > MAX_NESTING:
                raise ParseException(s, msg="Brackets are nested too deeply")
        elif char in ")]":
            depth -= 1
    return _expression.parse_string(s)[0]


def _check_equality(tuple1, tuple2):
    """
    return True if tuples of multimolecules are equal
    """
    list1 = list(tuple1)
    list2 = list(tuple2)

    # Hypo: trees where are levels count+molecule vs just molecule
    # cannot be sorted properly (tested on test_complex_additivity)
    # But without factors and phases sorting seems to work.

    # Also for lists of multimolecules without factors and phases
    # sorting seems to work fine.
    list1.sort()
    list2.sort()
    return list1 == list2


def compare_chemical_expression(s1, s2, ignore_state=False):
    """
    It does comparison between two expressions.
    It uses divide_chemical_expression and check if division is 1
    """
    return divide_chemical_expression(s1, s2, ignore_state) == 1


def divide_chemical_expression(s1, s2, ignore_state=False):
    """
    Compare two chemical expressions for equivalence up to a multiplicative factor:

    - If they are not the same chemicals, returns False.
    - If they are the same, "divide" s1 by s2 to returns a factor x such that s1 / s2 == x as a Fraction object.
    - if ignore_state is True, ignores phases when doing the comparison.

    Examples:
    divide_chemical_expression("H2O", "3H2O") -> Fraction(1,3)
    divide_chemical_expression("3H2O", "H2O") -> 3  # actually Fraction(3, 1), but compares == to 3.
    divide_chemical_expression("2H2O(s) + 2CO2", "H2O(s)+CO2") -> 2
    divide_chemical_expression("H2O(s) + CO2", "3H2O(s)+2CO2") -> False

    Implementation sketch:
        - extract factors and phases to standalone lists,
        - compare expressions without factors and phases,
        - divide lists of factors for each other and check
             for equality of every element in list,
        - return result of factor division

    """

    # parsed final trees
    treedic = {
        '1': _get_final_tree(s1),
        '2': _get_final_tree(s2)
    }

    # strip phases and factors
    # collect factors in list
    for i in ('1', '2'):
        treedic[i + ' cleaned_mm_list'] = []
        treedic[i + ' factors'] = []
        treedic[i + ' phases'] = []
        for el in treedic[i].subtrees(filter=lambda t: t.label() == 'multimolecule'):
            count_subtree = [t for t in el.subtrees() if t.label() == 'count']
            group_subtree = [t for t in el.subtrees() if t.label() == 'group']
            phase_subtree = [t for t in el.subtrees() if t.label() == 'phase']
            # Compatibility quirk "float-factors" (see test_chemcalc_quirks.py):
            # counts become floats here. So the ratio returned below is a
            # float's value rather than an exact fraction, and rounding can make
            # proportional fractional counts look not proportional.
            if count_subtree:
                if len(count_subtree[0]) > 1:
                    treedic[i + ' factors'].append(
                        int(count_subtree[0][0][0]) /
                        int(count_subtree[0][2][0]))
                else:
                    treedic[i + ' factors'].append(int(count_subtree[0][0][0]))
            else:
                treedic[i + ' factors'].append(1.0)
            if phase_subtree:
                treedic[i + ' phases'].append(phase_subtree[0][0])
            else:
                treedic[i + ' phases'].append(' ')
            treedic[i + ' cleaned_mm_list'].append(
                Tree('multimolecule', [Tree('molecule', group_subtree)]))

    # order of factors and phases must mirror the order of multimolecules,
    # use 'decorate, sort, undecorate' pattern
    treedic['1 cleaned_mm_list'], treedic['1 factors'], treedic['1 phases'] = list(zip(
        *sorted(zip(treedic['1 cleaned_mm_list'], treedic['1 factors'], treedic['1 phases']))))

    treedic['2 cleaned_mm_list'], treedic['2 factors'], treedic['2 phases'] = list(zip(
        *sorted(zip(treedic['2 cleaned_mm_list'], treedic['2 factors'], treedic['2 phases']))))

    # check if expressions are correct without factors
    if not _check_equality(treedic['1 cleaned_mm_list'], treedic['2 cleaned_mm_list']):
        return False

    # phases are ruled by ingore_state flag
    if not ignore_state:  # phases matters
        if treedic['1 phases'] != treedic['2 phases']:
            return False

    # Compatibility quirk "zero-count-raises" (see test_chemcalc_quirks.py):
    # a count of 0 in s2 makes this raise ZeroDivisionError.
    if any(
        [
            x / y - treedic['1 factors'][0] / treedic['2 factors'][0]
            for (x, y) in zip(treedic['1 factors'], treedic['2 factors'])
        ]
    ):
        # factors are not proportional
        return False
    else:
        # return ratio
        return Fraction(treedic['1 factors'][0] / treedic['2 factors'][0])


def split_on_arrow(eq):
    """
    Split a string on an arrow.  Returns left, arrow, right.  If there is no arrow, returns the
    entire eq in left, and '' in arrow and right.

    Return left, arrow, right.
    """
    # order matters -- need to try <-> first
    for arrow in ARROWS:
        left, a, right = eq.partition(arrow)
        if a != '':
            return left, a, right

    return eq, '', ''


def chemical_equations_equal(eq1, eq2, exact=False):
    """
    Check whether two chemical equations are the same.  (equations have arrows)

    If exact is False, then they are considered equal if they differ by a
    constant factor.

    arrows matter: -> and <-> are different.

    e.g.
    chemical_equations_equal('H2 + O2 -> H2O2', 'O2 + H2 -> H2O2') -> True
    chemical_equations_equal('H2 + O2 -> H2O2', 'O2 + 2H2 -> H2O2') -> False

    chemical_equations_equal('H2 + O2 -> H2O2', 'O2 + H2 <-> H2O2') -> False

    chemical_equations_equal('H2 + O2 -> H2O2', '2 H2 + 2 O2 -> 2 H2O2') -> True
    chemical_equations_equal('H2 + O2 -> H2O2', '2 H2 + 2 O2 -> 2 H2O2', exact=True) -> False


    If there's a syntax error, we return False.
    """

    left1, arrow1, right1 = split_on_arrow(eq1)
    left2, arrow2, right2 = split_on_arrow(eq2)

    if arrow1 == '' or arrow2 == '':
        return False

    # TODO: may want to be able to give student helpful feedback about why things didn't work.
    if arrow1 != arrow2:
        # arrows don't match
        return False

    try:
        factor_left = divide_chemical_expression(left1, left2)
        if not factor_left:
            # left sides don't match
            return False

        factor_right = divide_chemical_expression(right1, right2)
        if not factor_right:
            # right sides don't match
            return False

        if factor_left != factor_right:
            # factors don't match (molecule counts to add up)
            return False

        if exact and factor_left != 1:
            # want an exact match.
            return False

        return True
    except ParseException:
        # Don't want external users to have to deal with parsing exceptions.  Just return False.
        return False
