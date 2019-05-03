from __future__ import absolute_import
from setuptools import setup


setup(
    name="chem",
    version='0.4.0',
    packages=["chem"],
    install_requires=[
        "pyparsing==2.2.0",
        "numpy==1.6.2",
        "scipy==0.14.0",
        "nltk",
        "markupsafe",  # Should be replaced by other utilities. See LEARNER-5853 for more details.
    ],
    test_suite='chem.tests',
    tests_require=[
        'coverage',
    ],
)
