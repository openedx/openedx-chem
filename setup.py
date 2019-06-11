from __future__ import absolute_import
from setuptools import setup


setup(
    name="chem",
    version='1.0.0',
    packages=["chem"],
    install_requires=[
        'markupsafe',  # Should be replaced by other utilities. See LEARNER-5853 for more details.
        'nltk',
        'numpy',
        'pyparsing',
        'scipy',
    ],
    test_suite='chem.tests',
    tests_require=[
        'coverage',
    ],
)
