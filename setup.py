from __future__ import absolute_import
from setuptools import setup


def load_requirements(*requirements_paths):
    """
    Load all requirements from the specified requirements files.
    Returns a list of requirement strings.
    """
    requirements = set()
    for path in requirements_paths:
        with open(path) as reqs:
            requirements.update(
                line.split('#')[0].strip() for line in reqs
                if is_requirement(line.strip())
            )
    return list(requirements)


def is_requirement(line):
    """
    Return True if the requirement line is a package requirement;
    that is, it is not blank, a comment, a URL, or an included file.
    """
    return line and not line.startswith(('-r', '#', '-e', 'git+', '-c'))

setup(
    name="chem",
    version='1.0.1',
    packages=["chem"],
    install_requires=load_requirements('requirements/base.in'),
    test_suite='chem.tests',
    tests_require=load_requirements('requirements/test.in'),
)
