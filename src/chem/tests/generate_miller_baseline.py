"""
Generate miller_baseline.json, which test_miller_baseline checks against.

The baseline records the Miller indices chem 2.0.1 computed for a set of student
answers. That was the last release to do miller's vector arithmetic with numpy,
and crystallography problems written while it was in use depend on how it
graded them.

chem 2.0.1 didn't give the same indices everywhere, though. np.dot calls
OpenBLAS, which picks a kernel to suit the CPU, and the kernels work out a
three-term dot product in two ways:

* "fused": add the second and third products with fused multiply-add, which
  skips a rounding step. The SkylakeX and Cooperlake kernels, for CPUs with
  AVX-512, do this.
* "plain": multiply each pair, and add the products one at a time. The Haswell,
  Zen and Sandybridge kernels do this.

The two can differ in the last bit. miller() tests dot products for exact
zeros, so that bit decides how some answers are graded.

Most entries have one result, "indices". chem 2.0.1 computed it with both
kinds of kernel, and the current code must compute it too.

The other entries are answers that a host with AVX-512 graded differently from
other hosts. They have two results: "fused" and "plain". miller() now does the
same thing on every CPU. It gives the fused result, unless that was a
ZeroDivisionError, and then it gives the plain result. The comment in miller()
says why. test_miller_baseline checks the current code against that rule.

Run this against chem 2.0.1, not against the code in this repo, and with the
numpy that openedx-platform pinned while 2.0.1 was current:

    uv run --isolated --no-project --python 3.12 \
        --with chem==2.0.1 --with numpy==1.26.4 \
        python src/chem/tests/generate_miller_baseline.py

Without --isolated, uv may reuse this repo's virtualenv, and the checked-out
code will be imported instead of 2.0.1. main() checks for that. It also needs
an x86-64 CPU with AVX-512, because it runs both kernels, and it checks which
arithmetic each one really does.

Don't regenerate the baseline from the current code; it would then only check
the code against itself. If a result should change on purpose, edit that entry
by hand and explain why in the commit message.
"""
# pylint: disable=missing-function-docstring
import json
import os
import random
import subprocess
import sys
import warnings
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from unittest import mock

from chem import miller

BASELINE_PATH = Path(__file__).with_name("miller_baseline.json")
REFERENCE_VERSION = "2.0.1"

# The OpenBLAS kernel to run for each kind of arithmetic.
KERNELS = {"fused": "SkylakeX", "plain": "Haswell"}

# How many answers that depend on the kernel to keep, of those found.
CPU_DEPENDENT_LIMIT = 150


def computed_indices(module, points):
    """
    Return the Miller indices module.grade() computes for a student's points,
    the way the baseline stores them.

    grade() only returns True or False, so this wraps module.miller() to see
    what it returned.
    """
    calls = []
    original = module.miller

    def spy(*args):
        calls.append(original(*args))
        return calls[-1]

    user_input = json.dumps({"lattice": "", "points": points})
    with mock.patch.object(module, "miller", spy):
        try:
            module.grade(user_input, {"lattice": "", "miller": "(1,1,1)"})
        except Exception as err:  # pylint: disable=broad-exception-caught
            return {"raises": type(err).__name__}
    return calls[0] if calls else None


# Answers chosen by hand. Each is the "points" a student's answer holds.
HAND_PICKED = [
    # Planes that miss the origin
    [["0.5", "0", "0"], ["0", "0.5", "0"], ["0", "0", "0.5"]],
    [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
    [["1", "0.5", "1"], ["1", "1", "0.5"], ["0.5", "1", "1"]],
    [["0", "0.25", "0"], ["0.25", "0", "0"], ["0", "0", "0.25"]],
    [["1", "0", "0.5"], ["1", "1", "0"], ["0", "1", "0"]],
    # Planes parallel to one axis, and to two
    [["0.5", "0", "0"], ["0.5", "1", "0"], ["0", "0", "1"]],
    [["1", "0", "0"], ["1", "1", "0"], ["0", "0", "1"]],
    [["0", "1", "0.5"], ["1", "0", "0.5"], ["0.5", "1", "0.5"]],
    # Planes through the origin
    [["0", "1", "0"], ["1", "0", "0"], ["0.5", "1", "0"]],
    [["1", "0", "1"], ["0", "0", "0"], ["0", "1", "1"]],
    [["0", "0", "0"], ["0", "0", "1"], ["0.5", "1", "0"]],
    [["0", "0", "0"], ["1", "1", "0"], ["0", "1", "1"]],
    [["1", "0", "1"], ["0", "1", "1"], ["1", "0", "0"]],
    # Coordinates are rounded to the nearest 0.05 first
    [["0.33", "0", "0"], ["0", "0.33", "0"], ["0", "0", "0.33"]],
    [["0.66", "0", "0"], ["0", "0.66", "0"], ["0", "0", "0.66"]],
    [["0.30", "0", "0"], ["0", "0.30", "0"], ["0", "0", "0.30"]],
    [["0.33", "1", "0"], ["0", "0.664", "0"], ["0", "1", "0.33"]],
    [["0", "0", "0.01"], ["1", "1", "0.01"], ["0", "1", "1"]],
    [["0", "0.01", "0"], ["1", "0", "0"], ["0", "0", "1"]],
    # Segments too short for a fraction with a denominator of 10 or less
    [["0.05", "0", "0"], ["0", "0.05", "0"], ["0", "0", "0.05"]],
    [["0.01", "0", "0"], ["0", "0.01", "0"], ["0", "0", "0.01"]],
    # Points that don't define a plane
    [["0", "0", "0"], ["0.5", "0.5", "0.5"], ["1", "1", "1"]],
    [["0.5", "0", "0"], ["0.5", "0", "0"], ["0", "0", "0.5"]],
    [["0.5", "0", "0"], ["0.5", "0", "0"], ["0.5", "0", "0"]],
    # Points outside the unit cube
    [["-0.5", "0", "0"], ["0", "-0.5", "0"], ["0", "0", "-0.5"]],
    [["2", "0", "0"], ["0", "2", "0"], ["0", "0", "2"]],
    [["1e200", "0", "0"], ["0", "1e200", "0"], ["0", "0", "1e200"]],
    # ...and far enough outside for a dot product to overflow partway through
    [["5e102", "5e102", "0"], ["0", "5e102", "5e102"], ["5e102", "0", "5e102"]],
    [["1e120", "1e120", "0"], ["0", "0", "0"], ["1e120", "0", "1e120"]],
    # Only the first three points count
    [["0.5", "0", "0"], ["0", "0.5", "0"], ["0", "0", "0.5"], ["1", "1", "1"]],
    [["0.5", "0", "0"], ["0", "0.5", "0"], ["0", "0", "0.5"], ["1", "1"]],
    # Numbers instead of strings
    [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    # Points without three coordinates
    [["0.5", "0"], ["0", "0.5"], ["1", "1"]],
    [["0.5", "0", "0", "0"], ["0", "0.5", "0", "0"], ["0", "0", "0.5", "0"]],
    [["0.5", "0", "0"], ["0", "0.5"], ["0", "0", "0.5"]],
    [["0.5", "0", "0"], ["0", "0.5", "0", "0"], ["0", "0", "0.5"]],
    [["0.5"], ["0"], ["1"]],
    [[], [], []],
    # Coordinates that aren't finite numbers
    [["nan", "0", "0"], ["0", "0.5", "0"], ["0", "0", "0.5"]],
    [["inf", "0", "0"], ["0", "0.5", "0"], ["0", "0", "0.5"]],
    [["a", "0", "0"], ["0", "0.5", "0"], ["0", "0", "0.5"]],
]

# segment_to_fraction's docstring says the UI sends multiples of 0.05.
TICKS = [f"{tick / 20:.2f}" for tick in range(21)]
# grade() rounds any other coordinate to one of those, so cover that as well.
HUNDREDTHS = [f"{value / 100:.2f}" for value in range(101)]


class Sampler:
    """
    Makes random answers. The same seed always makes the same ones.
    """
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def edge_point(self, values):
        """
        Return a point on an edge of the unit cube.
        """
        point = [self.rng.choice(["0.00", "1.00"]) for _ in range(3)]
        point[self.rng.randrange(3)] = self.rng.choice(values)
        return point

    def inside_point(self, values):
        """
        Return a point anywhere in the unit cube.
        """
        return [self.rng.choice(values) for _ in range(3)]

    def answers(self, count, make_point, values, keep=None):
        """
        Return count answers of three points each, made by make_point from values.
        """
        found = []
        while len(found) < count:
            points = [make_point(values) for _ in range(3)]
            if keep is None or keep(points):
                found.append(points)
        return found


def through_origin(points):
    """
    Return True if points define a plane, and it contains the origin.
    """
    a, b, c = [[Fraction(text) for text in point] for point in points]
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    return any(normal) and sum(a[i] * normal[i] for i in range(3)) == 0


def generate_answers():
    """
    Return the answers that always go in the baseline, and a larger pool to
    search for answers whose indices depend on the kernel.
    """
    sampler = Sampler(20261120)
    answers = (
        HAND_PICKED
        + sampler.answers(500, sampler.edge_point, TICKS)
        + sampler.answers(200, sampler.edge_point, HUNDREDTHS)
        + sampler.answers(200, sampler.inside_point, TICKS)
        + sampler.answers(100, sampler.inside_point, HUNDREDTHS)
        + sampler.answers(300, sampler.edge_point, TICKS, keep=through_origin)
    )
    pool = sampler.answers(40000, sampler.edge_point, TICKS)
    return answers, pool


def dot_arithmetic(np):
    """
    Return the arithmetic np.dot does for 3-vectors here: "plain" or "fused".
    """
    # The two products are equal and opposite. Plain arithmetic rounds both the
    # same way, so they cancel. Fused arithmetic rounds only the first, which
    # leaves that product's rounding error.
    result = np.dot(np.array([0.1, 0.7, 0.0]), np.array([0.7, -0.1, 0.0]))
    return "plain" if result == 0 else "fused"


def worker(arithmetic):
    """
    Read answers from stdin, and write the indices chem computes for them to
    stdout. main() runs this in a process of its own for each kernel.
    """
    # chem 2.0.1's miller imported numpy as np.
    found = dot_arithmetic(miller.np)  # pylint: disable=no-member
    if found != arithmetic:
        sys.exit(f"np.dot does {found} arithmetic with OPENBLAS_CORETYPE="
                 f"{os.environ.get('OPENBLAS_CORETYPE')}, not {arithmetic}. "
                 "See the module docstring.")
    warnings.simplefilter("ignore")  # numpy warns when 1e200 overflows
    answers = json.load(sys.stdin)
    json.dump([computed_indices(miller, points) for points in answers], sys.stdout)


def run_kernel(arithmetic, answers):
    """
    Return the indices chem computes for answers when OpenBLAS uses the kernel
    that does this arithmetic.
    """
    done = subprocess.run(
        [sys.executable, __file__, "--worker", arithmetic],
        env={**os.environ, "OPENBLAS_CORETYPE": KERNELS[arithmetic]},
        input=json.dumps(answers), stdout=subprocess.PIPE, text=True, check=False,
    )
    if done.returncode:
        sys.exit(f"The {KERNELS[arithmetic]} kernel didn't run. See the module docstring.")
    return json.loads(done.stdout)


def write_baseline(path, about, entries):
    """
    Write the baseline as JSON with one answer per line, so diffs stay readable.
    """
    lines = ["{", f'"about": {json.dumps(about)},', '"answers": [']
    lines += [json.dumps(entry) + ("," if i < len(entries) - 1 else "")
              for i, entry in enumerate(entries)]
    lines += ["]", "}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if version("chem") != REFERENCE_VERSION or not hasattr(miller, "np"):
        sys.exit(f"Run this against chem {REFERENCE_VERSION}. See the module docstring.")
    if "--worker" in sys.argv:
        worker(sys.argv[-1])
        return

    answers, pool = generate_answers()
    fused = run_kernel("fused", answers + pool)
    plain = run_kernel("plain", answers + pool)

    entries = []
    seen = set()
    cpu_dependent = 0
    for i, points in enumerate(answers + pool):
        key = json.dumps(points)
        differs = fused[i] != plain[i]
        from_pool = i >= len(answers)
        if key in seen or (from_pool and not (differs and cpu_dependent < CPU_DEPENDENT_LIMIT)):
            continue
        seen.add(key)
        if differs:
            entries.append({"points": points, "fused": fused[i], "plain": plain[i]})
            cpu_dependent += 1
        else:
            entries.append({"points": points, "indices": fused[i]})

    about = (f"Miller indices from chem {REFERENCE_VERSION} (numpy). "
             "See generate_miller_baseline.py before editing.")
    write_baseline(Path(sys.argv[1]) if len(sys.argv) > 1 else BASELINE_PATH, about, entries)
    in_pool = sum(1 for i in range(len(answers), len(fused)) if fused[i] != plain[i])
    print(f"{len(entries)} answers, {cpu_dependent} depend on the kernel")
    print(f"{in_pool} of the {len(pool)} answers in the pool depend on the kernel")


if __name__ == "__main__":
    main()
