"""
Generate chemcalc_baseline.json, which test_chemcalc_baseline checks against.

The baseline records what chem 2.0.1 returned for a set of inputs. That was the
last release to parse formulas with nltk, and course content written over the
years it was in use depends on how it rendered and compared them.

Some of that behavior is probably wrong but kept on purpose: the compatibility
quirks described in test_chemcalc_quirks.py. Baseline entries whose results
depend on a quirk are tagged with its name in a "quirks" field, so that they can
be checked, and later updated, separately from the rest.

Run this against chem 2.0.1, not against the code in this repo:

    uv run --isolated --no-project --with chem==2.0.1 \
        python src/chem/tests/generate_chemcalc_baseline.py

Without --isolated, uv may reuse this repo's virtualenv, and the checked-out
code will be imported instead of 2.0.1. main() checks for that.

Don't regenerate the baseline from the current code; it would then only check
the code against itself. If a result should change on purpose, edit that entry
by hand and explain why in the commit message.
"""
# pylint: disable=missing-function-docstring
import inspect
import json
import random
import re
import sys
import types
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

from chem import chemcalc

BASELINE_PATH = Path(__file__).with_name("chemcalc_baseline.json")
REFERENCE_VERSION = "2.0.1"

# The baseline's sections, and the input fields of each entry in them.
SECTIONS = {"expressions": ("input",), "comparisons": ("a", "b"), "equations": ("a", "b")}


def record(func, *args):
    """
    Call func and return its result the way the baseline stores it.
    """
    try:
        value = func(*args)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return {"raises": type(err).__name__}
    if isinstance(value, (Fraction, str)):
        # divide_chemical_expression returns Fractions, and render_to_html returns Markup.
        return str(value)
    return value


def results(module, section, case):
    """
    Return the results the baseline stores for case, using module as chemcalc.
    """
    if section == "expressions":
        return {"render_to_html": record(module.render_to_html, case["input"])}
    if section == "comparisons":
        divide = module.divide_chemical_expression
        return {"divide": record(divide, case["a"], case["b"]),
                "divide_ignore_state": record(divide, case["a"], case["b"], True)}
    equal = module.chemical_equations_equal
    return {"equal": record(equal, case["a"], case["b"]),
            "equal_exact": record(equal, case["a"], case["b"], True)}


# Inputs chosen by hand to cover each part of the grammar. Inputs that show a
# compatibility quirk are in the QUIRK_ lists below instead.
HAND_PICKED_EXPRESSIONS = [
    # Grammar features
    "H2O", "C6H12O6", "H2O2", "NaCl", "2H2O", "10H2O", "1/2O2", "7/2HCl", "0H2O",
    "H2O(s)", "H2O(l)", "CO2(g)", "NaCl(aq)", "2H2O(l)",
    "(OH)", "(OH)2", "Ca(OH)2", "[Ar]", "[Ar]2", "[Cu(NH3)4]", "[Cu(NH3)4]^2+",
    "5[Ni(NH3)4]^2+", "((CH3)3C)2O", "[Fe(CN)6]^3-", "(NH4)2SO4(aq)", "(OH)(aq)",
    "H^+", "OH^-", "Fe^3+", "SO4^2-", "H2^+", "NH4^+", "PO4^3-",
    "H2 + O2", "H2+O2", "2H2 + O2", "H^+ + OH^-", "H^+ + HSO4^-", "H2O + CO2 + Fe(OH)3",
    # Element symbols match longest first, and case matters
    "Co", "CO", "Cs", "CS", "Hs", "HS", "No", "NO", "Sn", "SN", "Uuo", "Uus3", "U",
    # An explicit charge of 1 is dropped, so ^1+ is the same as ^+
    "Fe^1+", "SO4^1-", "Fe^01+",
    # Whitespace between tokens is ignored
    "H 2O", "2 H2", "C 6H 12O 6", "  H2O  ", "H2O\t", "\nH2O",
    # ^ takes the next + as a charge, even with spaces between them
    "H^ + O",
    "H22", "H02", "H1H212", "5(H1H212)^70010-", "2H20",
    # Invalid input
    "", "   ", "h2o", "Q2", "Ab", "H2O(", "H2O)", "()", "[]", "(H2O", "(H2O]", "Xe+",
    "+H2", "H2 +", "H2++O2", "H-", "2", "1/H2", "1//2H2", "1/2/3H2", "H2O(aq", "(aq)",
    "H2O(l)(g)", "(l)H2O", "H2O(l)2", "H2O(S)", "5.2H20", "H^", "H^2", "H^+-", "H^-+",
    "H2O2^", "<script>f()</script>", "H2O & CO2", "H₂O",
    # Whole equations, which render_to_html also accepts
    "H^+ + OH^- -> H2O", "H^+ + OH^- <-> H2O", "2H2 + O2 -> 2H2O(l)",
    "[H^+] + OH^- <-> (H2O)", "H2 -> O2 -> H2O", "H2 <=> H2", "H2 <= H2", "-> H2O",
    "H2O ->", "H2 - > H2O", "H2 <- > H2O",
]

HAND_PICKED_PAIRS = [
    ("H2O + CO2", "CO2 + H2O"), ("H2O + CO2", "O2C + OH2"),
    ("3H2O + 2CO2", "2CO2 + 3H2O"), ("2H2O + 3CO2", "2CO2 + 3H2O"),
    ("H^+ + OH^-", "OH^- + H^+"), ("H^+ + OH^-", "OH^- + H^-"), ("H^+ + OH^1-", "OH^- + H^+"),
    ("(OH)", "(OH)"), ("(OH)", "(H2)"), ("[Ar]", "[Kr]"), ("(OH)", "OH"), ("(OH)", "[OH]"),
    ("(OH)2", "O2H2"), ("(OH)(aq)", "OH(aq)"),
    ("H2O(s) + CO2", "H2O + CO2"), ("H2O(s) + CO2", "2H2O(s) + 2CO2"),
    ("H2O(s) + CO2", "2H2O + 2CO2(s)"),
    ("3H2O", "H2O"), ("1/2O2", "O2"), ("3/4CO2 + H2O", "2H2O + 9/6CO2"),
    ("0H2O", "H2O"), ("H2O + H2O", "2H2O"), ("H2O + H2O", "H2O + H2O"),
    ("H2 + CO2", "H2 + C102"), ("H2O", "H2O("), ("H2O", ""),
    ("5[Ni(NH3)4]^2+ + 5/2SO4^2-", "5/2SO4^2- + 5[Ni(NH3)4]^2+"),
    ("Co", "CO"), ("NaCl", "ClNa"),
]

HAND_PICKED_EQUATIONS = [
    ("H2 + O2 -> H2O2", "O2 + H2 -> H2O2"), ("H2 + O2 -> H2O2", "O2 + 2H2 -> H2O2"),
    ("H2 + O2 -> H2O2", "2O2 + 2H2 -> 2H2O2"), ("2H2 + O2 -> H2O2", "2O2 + 2H2 -> 2H2O2"),
    ("H2 + O2 -> H2O2", "O2 + H2 <-> H2O2"), ("H2 + O2 <-> H2O2", "O2 + H2 <-> H2O2"),
    ("H2 + O2 a-> H2O2", "2O2 + 2H2 -> 2H2O2"), ("H2O( -> H2O2", "H2O -> H2O2"),
    ("H2 + O2 ==> H2O2", "2O2 + 2H2 -> 2H2O2"), ("H2O", "H2O"), ("H2O -> H2O", "H2O"),
    ("0H2 -> 0H2", "H2 -> H2"), ("H^+ + OH^- -> H2O", "OH^- + H^+ -> H2O"),
    ("2H2 + O2 -> 2H2O", "4H2 + 2O2 -> 4H2O"), ("2H2 + O2 -> 2H2O", "H2 + 1/2O2 -> H2O"),
    ("H2 -> O2 -> H2O", "H2 -> O2 -> H2O"),
]

# Inputs chosen by hand to show each compatibility quirk: behavior that's
# probably wrong but kept on purpose. See test_chemcalc_quirks.py.
QUIRK_EXPRESSIONS = [
    "Fe^12+", "Fe^10-", "SO4^12-",  # charge-starting-with-1
    "1 0H2", "H1 2O",  # space-inside-number
]

QUIRK_PAIRS = [
    ("Fe^12+", "Fe^+"),  # charge-starting-with-1
    ("10H2", "1 0H2"),  # space-inside-number
    ("H2O", "3H2O"), ("7/10H2 + 1/10O2", "7H2 + O2"),  # float-factors
    ("H2O", "0H2O"),  # zero-count-raises
]

QUIRK_EQUATIONS = [
    ("Fe^12+ -> Fe", "Fe^+ -> Fe"),  # charge-starting-with-1
    ("7/10H2 + 1/10O2 -> 7/10H2O2", "7H2 + O2 -> 7H2O2"),  # float-factors
    ("H2 -> H2", "0H2 -> 0H2"),  # zero-count-raises
]

# For these quirks, the entries that depend on one are found by running chem
# 2.0.1 again with the quirk fixed, and seeing which results change. Each fix is
# a list of (old, new) edits to 2.0.1's chemcalc.py.
QUIRK_FIXES = {
    # Only drop a charge that is exactly 1.
    "charge-starting-with-1": [
        ("n[1:][0][0][0] == '1'", "len(n[1:][0]) == 1 and n[1:][0][0][0] == '1'"),
    ],
    # Keep counts as exact fractions.
    "float-factors": [
        ("int(count_subtree[0][0][0]) /\n                        int(count_subtree[0][2][0])",
         "Fraction(int(count_subtree[0][0][0]), int(count_subtree[0][2][0]))"),
        ("append(int(count_subtree[0][0][0]))", "append(Fraction(int(count_subtree[0][0][0])))"),
        ("append(1.0)", "append(Fraction(1))"),
    ],
}


def with_quirk_fixed(name):
    """
    Return a copy of chem 2.0.1's chemcalc module with the named quirk fixed.
    """
    source = inspect.getsource(chemcalc)
    for old, new in QUIRK_FIXES[name]:
        assert source.count(old) == 1, f"{name}: can't find {old!r}"
        source = source.replace(old, new)
    module = types.ModuleType(f"chemcalc_without_{name}")
    exec(compile(source, f"<chemcalc without {name}>", "exec"), module.__dict__)  # pylint: disable=exec-used
    return module


def find_quirks(section, case, stored, fixed_modules):
    """
    Return the names of the quirks that case's stored results depend on.
    """
    quirks = {name for name, module in fixed_modules.items()
              if results(module, section, case) != stored}
    # Spaces between digits always join them into one number.
    if any(re.search(r"\d\s+\d", case[key]) for key in SECTIONS[section]):
        quirks.add("space-inside-number")
    if {"raises": "ZeroDivisionError"} in stored.values():
        quirks.add("zero-count-raises")
    return sorted(quirks)


COMMON_ELEMENTS = ["H", "O", "C", "N", "S", "Cl", "Na", "Fe", "Cu", "Co", "Ca", "Ni", "P", "K",
                   "Mg", "Al"]
NOISE = list("HONCSPKabclosq0123456789()[]^+-/ .<>=*") + ["(aq)", "(s)", "(l)", "(g)", "->"]
SEPARATORS = [" + ", "+", " +", "+ ", "  +  "]


class InputGenerator:
    """
    Make random but repeatable inputs.

    An expression is built as a list of molecules, each a (count, body, phase)
    tuple, so that it can be shuffled, rescaled, or changed in one place before
    it's joined into a string.
    """

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def element(self):
        return self.rng.choice(COMMON_ELEMENTS if self.rng.random() < 0.8 else chemcalc.elements)

    def item(self, depth):
        """
        An element or bracketed group, maybe with a subscript and a charge.
        """
        roll = self.rng.random()
        if depth < 3 and roll < 0.15:
            text = "(" + self.group(depth + 1) + ")"
        elif depth < 3 and roll < 0.25:
            text = "[" + self.group(depth + 1) + "]"
        else:
            text = self.element()
        if self.rng.random() < 0.35:
            text += str(self.rng.choice([1, 2, 2, 3, 4, 6, 10, 12, self.rng.randint(0, 99)]))
        if self.rng.random() < 0.15:
            charge = self.rng.choice(["", "", "1", "2", "3", "10", "12", "0"])
            text += "^" + charge + self.rng.choice("+-")
        return text

    def group(self, depth=0):
        return "".join(self.item(depth) for _ in range(self.rng.randint(1, 4)))

    def molecule(self):
        roll = self.rng.random()
        if roll < 0.3:
            count = str(self.rng.randint(1, 12))
        elif roll < 0.4:
            count = f"{self.rng.randint(1, 9)}/{self.rng.randint(1, 9)}"
        else:
            count = ""
        phase = self.rng.choice(chemcalc.phases) if self.rng.random() < 0.25 else ""
        return (count, self.group(), phase)

    def molecules(self):
        return [self.molecule() for _ in range(self.rng.randint(1, 4))]

    def join(self, molecules):
        texts = ["".join(m) for m in molecules]
        return texts[0] + "".join(self.rng.choice(SEPARATORS) + t for t in texts[1:])

    def shuffled(self, molecules):
        molecules = molecules[:]
        self.rng.shuffle(molecules)
        return molecules

    @staticmethod
    def scaled(molecules, factor):
        """
        Multiply every molecule's count by factor.
        """
        result = []
        for count, body, phase in molecules:
            if "/" in count:
                numerator, denominator = count.split("/")
                count = f"{int(numerator) * factor}/{denominator}"
            else:
                count = str(int(count or 1) * factor)
            result.append((count, body, phase))
        return result

    def near_miss(self, molecules):
        """
        Copy molecules with one small change, such as a different element or charge.
        """
        molecules = molecules[:]
        index = self.rng.randrange(len(molecules))
        count, body, phase = molecules[index]
        change = self.rng.choice(
            ["element", "subscript", "charge", "brackets", "phase", "count", "drop"]
        )
        if change == "phase":
            phase = self.rng.choice([p for p in chemcalc.phases + [""] if p != phase])
        elif change == "count":
            count = self.rng.choice([c for c in ["", "2", "3", "1/2"] if c != count])
        elif change == "drop":
            if len(molecules) > 1:
                del molecules[index]
            return molecules
        else:
            body = self.changed_body(change, body)
        molecules[index] = (count, body, phase)
        return molecules

    def changed_body(self, change, body):  # pylint: disable=too-many-return-statements
        """
        Make one change of the given kind to a molecule's formula, without its count or phase.
        """
        if change == "element":
            match = self.rng.choice(list(re.finditer(r"[A-Z][a-z]*", body)))
            replacement = self.rng.choice([e for e in COMMON_ELEMENTS if e != match.group()])
            return body[:match.start()] + replacement + body[match.end():]
        if change == "subscript":
            numbers = list(re.finditer(r"(?<![\^\d])\d+", body))
            if not numbers:
                return re.sub(r"[A-Z][a-z]*", lambda m: m.group() + "2", body, count=1)
            match = self.rng.choice(numbers)
            return body[:match.start()] + str(int(match.group()) + 1) + body[match.end():]
        if change == "charge":
            charges = list(re.finditer(r"\^\d*[+-]", body))
            if not charges:
                return body + "^+"
            match = self.rng.choice(charges)
            flipped = match.group()[:-1] + ("-" if match.group().endswith("+") else "+")
            return body[:match.start()] + flipped + body[match.end():]
        # brackets
        if "(" in body or "[" in body:
            return body.translate(str.maketrans("()[]", "[]()"))
        return "(" + body + ")"

    def mangled(self, text):
        """
        Copy text with a few random characters inserted, deleted, or replaced.
        """
        chars = list(text)
        for _ in range(self.rng.randint(1, 3)):
            roll = self.rng.random()
            position = self.rng.randint(0, len(chars))
            if roll < 0.4:
                chars.insert(position, self.rng.choice(NOISE))
            elif chars and roll < 0.7:
                del chars[min(position, len(chars) - 1)]
            elif chars:
                chars[min(position, len(chars) - 1)] = self.rng.choice(NOISE)
        return "".join(chars)


def shape(text):
    """
    Reduce an input to its structure, to avoid keeping many inputs that only
    differ in which elements and numbers they use.
    """
    text = re.sub(r"[A-Z][a-z]*", "E", text)
    return re.sub(r"\d+", lambda m: "1" if m.group().startswith("1") else "9", text)


def unique_by_shape(candidates, limit, seen):
    result = []
    for text in candidates:
        if shape(text) not in seen:
            seen.add(shape(text))
            result.append(text)
            if len(result) == limit:
                break
    return result


def generate_inputs():
    """
    Return the inputs to record for each section, as a list of dicts of input fields.

    Each list starts with the hand-picked inputs, then the quirk ones, then
    generated ones.
    """
    gen = InputGenerator(seed=2026)

    seen = {shape(text) for text in HAND_PICKED_EXPRESSIONS + QUIRK_EXPRESSIONS}
    valid = unique_by_shape((gen.join(gen.molecules()) for _ in range(20000)), 250, seen)
    mangled = unique_by_shape(
        (gen.mangled(gen.join(gen.molecules())) for _ in range(20000)), 100, seen
    )
    equations = [f"{gen.join(gen.molecules())} {gen.rng.choice(['->', '<->'])} "
                 f"{gen.join(gen.molecules())}" for _ in range(40)]
    expressions = HAND_PICKED_EXPRESSIONS + QUIRK_EXPRESSIONS + valid + mangled + equations

    pairs = HAND_PICKED_PAIRS + QUIRK_PAIRS
    for _ in range(120):
        molecules = gen.molecules()
        text = gen.join(molecules)
        pairs += [
            (text, gen.join(gen.shuffled(molecules))),
            (text, gen.join(gen.shuffled(gen.scaled(molecules, gen.rng.randint(2, 4))))),
            (text, gen.join(gen.near_miss(molecules))),
            (text, gen.join(gen.shuffled(gen.near_miss(molecules)))),
            (text, gen.mangled(text)),
        ]

    equation_pairs = HAND_PICKED_EQUATIONS + QUIRK_EQUATIONS
    for _ in range(60):
        left, right = gen.molecules(), gen.molecules()
        arrow = gen.rng.choice(["->", "<->"])
        other_arrow = "<->" if arrow == "->" else "->"
        factor = gen.rng.randint(2, 4)
        text = f"{gen.join(left)} {arrow} {gen.join(right)}"
        equation_pairs += [
            (text, f"{gen.join(gen.shuffled(left))} {arrow} {gen.join(gen.shuffled(right))}"),
            (text, f"{gen.join(gen.scaled(left, factor))} {arrow} "
                   f"{gen.join(gen.scaled(right, factor))}"),
            (text, f"{gen.join(gen.scaled(left, factor))} {arrow} {gen.join(right)}"),
            (text, f"{gen.join(left)} {arrow} {gen.join(gen.near_miss(right))}"),
            (text, f"{gen.join(left)} {other_arrow} {gen.join(right)}"),
        ]

    return {
        "expressions": [{"input": text} for text in expressions],
        "comparisons": [{"a": a, "b": b} for a, b in pairs],
        "equations": [{"a": a, "b": b} for a, b in equation_pairs],
    }


def check_hand_picked(sections):
    """
    Check that hand-picked inputs are in the right list: QUIRK_ ones depend on a
    quirk, and the others don't.
    """
    lists = {
        "expressions": (HAND_PICKED_EXPRESSIONS, QUIRK_EXPRESSIONS),
        "comparisons": (HAND_PICKED_PAIRS, QUIRK_PAIRS),
        "equations": (HAND_PICKED_EQUATIONS, QUIRK_EQUATIONS),
    }
    for section, (hand_picked, quirky) in lists.items():
        entries = sections[section]
        for entry in entries[:len(hand_picked)]:
            assert "quirks" not in entry, f"Move this to a QUIRK_ list: {entry}"
        for entry in entries[len(hand_picked):len(hand_picked) + len(quirky)]:
            assert "quirks" in entry, f"This doesn't depend on a quirk: {entry}"


def write_baseline(path, sections):
    """
    Write the baseline as JSON with one case per line, so diffs stay readable.
    """
    lines = ["{"]
    for i, (name, value) in enumerate(sections.items()):
        comma = "," if i < len(sections) - 1 else ""
        if isinstance(value, list):
            lines.append(f"{json.dumps(name)}: [")
            lines += [json.dumps(case, ensure_ascii=False) + ("," if j < len(value) - 1 else "")
                      for j, case in enumerate(value)]
            lines.append("]" + comma)
        else:
            lines.append(f"{json.dumps(name)}: {json.dumps(value, ensure_ascii=False)}{comma}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if version("chem") != REFERENCE_VERSION or not hasattr(chemcalc, "nltk"):
        sys.exit(f"Run this against chem {REFERENCE_VERSION}. See the module docstring.")

    fixed_modules = {name: with_quirk_fixed(name) for name in QUIRK_FIXES}
    sections = {
        "about": f"Results from chem {REFERENCE_VERSION} (nltk). "
                 "See generate_chemcalc_baseline.py before editing.",
    }
    for section, cases in generate_inputs().items():
        sections[section] = []
        for case in cases:
            stored = results(chemcalc, section, case)
            entry = {**case, **stored}
            quirks = find_quirks(section, case, stored, fixed_modules)
            if quirks:
                entry["quirks"] = quirks
            sections[section].append(entry)

    check_hand_picked(sections)
    # chem 2.0.1 raised ValueError for { and }, which is deliberately not kept.
    # Those inputs are covered in test_all instead, so none should be here.
    assert "ValueError" not in json.dumps(sections), "Leave out inputs with { or }"

    write_baseline(BASELINE_PATH, sections)
    for section in SECTIONS:
        entries = sections[section]
        tagged = sum(1 for entry in entries if "quirks" in entry)
        print(f"{section}: {len(entries)} entries, {tagged} depend on a quirk")


if __name__ == "__main__":
    main()
