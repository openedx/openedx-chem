""" Calculation of Miller indices """


import decimal
import fractions as fr
import json
import math
from functools import reduce


def _sub(a, b):
    """
    Returns the 3D vector a - b.
    """
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    """
    Returns the cross product of 3D vectors a and b.
    """
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _fma(x, y, z):
    """
    Returns x * y + z, rounded once at the end: a fused multiply-add.

    This is math.fma, which needs Python 3.13, except that where math.fma
    raises an exception this returns inf or nan, as numpy did.
    """
    if not (math.isfinite(x) and math.isfinite(y)):
        return x * y + z
    if not math.isfinite(z):
        return z
    exact = fr.Fraction(x) * fr.Fraction(y) + fr.Fraction(z)
    if not exact:
        return x * y + z  # Exact here too, and it gives the zero its sign.
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


# _miller() tests dot products for exact zeros, so the last bit of a dot product
# decides how some answers are graded. chem used to get them from np.dot, which
# rounded in one of the two ways below, depending on the CPU. Both are kept,
# term for term and in the same order. math.sumprod rounds differently from
# both of them. See miller() for how they're used.


def _dot_fused(a, b):
    """
    Returns the dot product of 3D vectors a and b, rounded as np.dot rounded
    it on x86-64 CPUs with AVX-512.

    The first product is rounded. Each of the others is added to the sum so
    far without being rounded first.
    """
    return _fma(a[2], b[2], _fma(a[1], b[1], a[0] * b[0]))


def _dot_plain(a, b):
    """
    Returns the dot product of 3D vectors a and b, rounded as np.dot rounded
    it on x86-64 CPUs without AVX-512.

    Every product is rounded before it's added.
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def lcm(a, b):
    """
    Returns least common multiple of a, b

    Args:
        a, b: floats

    Returns:
        int: Because math.gcd will only return int after python 3.9
        Previously it depended on type of args, now it'll only return int
    """
    return a * b / math.gcd(int(a), int(b))


def segment_to_fraction(distance):
    """
    Converts lengths of which the plane cuts the axes to fraction.

    Tries convert distance to closest nicest fraction with denominator less or
    equal than 10. It is
    purely for simplicity and clearance of learning purposes. Jenny: 'In typical
    courses students usually do not encounter indices any higher than 6'.

    If distance is not a number (nan), it means that plane is parallel to
    axis or contains it. Inverted fraction to nan (nan is 1/0) = 0 / 1 is
    returned

    Generally (special cases):

    a) if distance is smaller than some constant, i.g. 0.01011,
    than fraction's denominator usually much greater than 10.

    b) Also, if student will set point on 0.66 -> 1/3, so it is 333 plane,
    But if he will slightly move the mouse and click on 0.65 -> it will be
    (16,15,16) plane. That's why we are doing adjustments for points coordinates,
    to the closest tick, tick + tick / 2 value. And now UI sends to server only
    values multiple to 0.05  (half of tick). Same rounding is implemented for
    unittests.

    But if one will want to calculate miller indices with exact coordinates and
    with nice fractions (which produce small Miller indices), he may want shift
    to new origin if segments are like S = (0.015, > 0.05, >0.05) - close to zero
    in one coordinate. He may update S to (0, >0.05, >0.05) and shift origin.
    In this way he can receive nice small fractions. Also there is can be
    degenerated case when S = (0.015, 0.012, >0.05) - if update S to (0, 0, >0.05) -
    it is a line. This case  should be considered separately. Small nice Miller
    numbers and possibility to create very small segments can not be implemented
    at same time).


    Args:
        distance: float distance that plane cuts on axis, it must not be 0.
        Distance is multiple of 0.05.

    Returns:
         Inverted fraction.
         0 / 1 if distance is nan

    """
    if math.isnan(distance):
        return fr.Fraction(0, 1)
    else:
        fract = fr.Fraction(distance).limit_denominator(10)
        return fr.Fraction(fract.denominator, fract.numerator)


def sub_miller(segments):
    '''
    Calculates Miller indices from segments.

    Algorithm:

    1. Obtain inverted fraction from segments

    2. Find common denominator of inverted fractions

    3. Lead fractions to common denominator and throws denominator away.

    4. Return obtained values.

    Args:
        List of 3 floats, meaning distances that plane cuts on x, y, z axes.
        Any float not equals zero, it means that plane does not intersect origin,
        i. e. shift of origin has already been done.

    Returns:
        String that represents Miller indices, e.g: (-6,3,-6) or (2,2,2)
    '''
    fracts = [segment_to_fraction(segment) for segment in segments]
    common_denominator = reduce(lcm, [fract.denominator for fract in fracts])
    miller_indices = ([
        fract.numerator * math.fabs(common_denominator) / fract.denominator
        for fract in fracts
    ])
    return'(' + ','.join(map(str, list(map(decimal.Decimal, miller_indices)))) + ')'


def _miller(points, dot):
    """
    Calculates Miller indices from points, using dot for the dot products.

    Algorithm:

    1. Calculate normal vector to a plane that goes trough all points.

    2. Set origin.

    3. Create Cartesian coordinate system (Ccs).

    4. Find the lengths of segments of which the plane cuts the axes. Equation
       of a line for axes: Origin + (Coordinate_vector - Origin) * parameter.

    5. If plane goes trough Origin:

       a) Find new random origin: find unit cube vertex, not crossed by a plane.

       b) Repeat 2-4.

       c) Fix signs of segments after Origin shift. This means to consider
          original directions of axes. I.g.: Origin was 0,0,0 and became
          new_origin. If new_origin has same Y coordinate as Origin, then segment
          does not change its sign. But if new_origin has another Y coordinate than
          origin (was 0, became 1), than segment has to change its sign (it now
          lies on negative side of Y axis). New Origin 0 value of X or Y or Z
          coordinate means that segment does not change sign, 1 value -> does
          change. So new sign  is (1 - 2 * new_origin): 0 -> 1, 1 -> -1

    6. Run function that calculates miller indices from segments.

    Args:
        points: see miller().
        dot: function that returns the dot product of two 3D vectors.

    Returns:
        String that represents Miller indices, e.g: (-6,3,-6) or (2,2,2)
    """
    N = _cross(_sub(points[1], points[0]), _sub(points[2], points[0]))
    O = (0.0, 0.0, 0.0)
    P = points[0]  # point of plane
    Ccs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    segments = [
        dot(_sub(P, O), N) / dot(ort, N) if dot(ort, N) != 0
        else math.nan for ort in Ccs
    ]
    if any(x == 0 for x in segments):  # Plane goes through origin.
        vertices = [
            # top:
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            # bottom, except 0,0,0:
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
        ]
        for vertex in vertices:
            if dot(_sub(vertex, O), N) != 0:  # vertex not in plane
                new_origin = vertex
                break
        # obtain new axes with center in new origin
        X = (1 - new_origin[0], new_origin[1], new_origin[2])
        Y = (new_origin[0], 1 - new_origin[1], new_origin[2])
        Z = (new_origin[0], new_origin[1], 1 - new_origin[2])
        new_Ccs = [_sub(X, new_origin), _sub(Y, new_origin), _sub(Z, new_origin)]
        segments = [
            dot(_sub(P, new_origin), N) / dot(ort, N) if dot(ort, N) != 0
            else math.nan for ort in new_Ccs
        ]
        # fix signs of indices: 0 -> 1, 1 -> -1 (
        segments = [
            (1 - 2 * coordinate) * segment
            for coordinate, segment in zip(new_origin, segments)
        ]

    return sub_miller(segments)


def miller(points):
    """
    Calculates Miller indices from points.

    Args:
        List of points. Each point is a sequence of three float coordinates,
        in the order x, y, z. Points are different! Only the first three
        points are used.

    Returns:
        String that represents Miller indices, e.g: (-6,3,-6) or (2,2,2)

    Raises:
        ValueError: if a point doesn't have exactly three coordinates.
    """
    if any(len(point) != 3 for point in points[:3]):
        raise ValueError("Each point must have exactly three coordinates.")

    # Fused rounding goes first, so every answer that got Miller indices on a
    # CPU with AVX-512 still gets the same ones. It's worse than plain rounding
    # at noticing that a plane goes through the origin, though, and when it
    # doesn't notice, sub_miller() raises ZeroDivisionError. So those answers
    # get a second try, with plain rounding.
    try:
        return _miller(points, _dot_fused)
    except ZeroDivisionError:
        pass
    return _miller(points, _dot_plain)


def grade(user_input, correct_answer):
    '''
    Grade crystallography problem.

    Returns true if lattices are the same and Miller indices are same or minus
    same. E.g. (2,2,2) = (2, 2, 2) or (-2, -2, -2). Because sign depends only
    on student's selection of origin.

    Args:
        user_input, correct_answer: json. Format:

        user_input: {"lattice":"sc","points":[["0.77","0.00","1.00"],
        ["0.78","1.00","0.00"],["0.00","1.00","0.72"]]}

        correct_answer: {'miller': '(00-1)', 'lattice': 'bcc'}

        "lattice" is one of: "", "sc", "bcc", "fcc"

    Returns:
        True or false.
    '''
    def negative(m):
        """
        Change sign of Miller indices.

        Args:
            m: string with meaning of Miller indices. E.g.:
            (-6,3,-6) -> (6, -3, 6)

        Returns:
            String with changed signs.
        """
        output = ''
        i = 1
        while i in range(1, len(m) - 1):
            if m[i] in (',', ' '):
                output += m[i]
            elif m[i] not in ('-', '0'):
                output += '-' + m[i]
            elif m[i] == '0':
                output += m[i]
            else:
                i += 1
                output += m[i]
            i += 1
        return '(' + output + ')'

    def round0_25(point):
        """
        Rounds point coordinates to closest 0.5 value.

        Args:
            point: list of float coordinates. Order of coordinates: x, y, z.

        Returns:
            list of coordinates rounded to closes 0.5 value
        """
        rounded_points = []
        for coord in point:
            base = math.floor(coord * 10)
            fractional_part = (coord * 10 - base)
            aliquot0_25 = math.floor(fractional_part / 0.25)
            if aliquot0_25 == 0.0:
                rounded_points.append(base / 10)
            if aliquot0_25 in (1.0, 2.0):
                rounded_points.append(base / 10 + 0.05)
            if aliquot0_25 == 3.0:
                rounded_points.append(base / 10 + 0.1)
        return rounded_points

    user_answer = json.loads(user_input)

    if user_answer['lattice'] != correct_answer['lattice']:
        return False

    points = [list(map(float, p)) for p in user_answer['points']]

    if len(points) < 3:
        return False

    # round point to closes 0.05 value
    points = [round0_25(point) for point in points]

    # print miller(points), (correct_answer['miller'].replace(' ', ''),
    #     negative(correct_answer['miller']).replace(' ', ''))
    if miller(points) in (correct_answer['miller'].replace(' ', ''), negative(correct_answer['miller']).replace(' ', '')):
        return True

    return False
