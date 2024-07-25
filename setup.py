import os
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
version = '1.0.0'

# Get the documentation
with open(os.path.join(cwd, 'README.md'), "r") as fh:
    long_description = fh.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3.11",
]

setup(
    name             = "covasim_es",
    author           = "Mercy Mvundura, Lucky G. Ngwira, Kabita Bade Shrestha, Reshma Tuladhar, Jillian Gauld, Cliff Kerr, Kayla Barnes, Bhawana Sharma, Nicholas Feasey",
    author_email     = "info@covasim.org",
    description      = "Malawi ES manuscript",
    url              = 'http://covasim.org',
    keywords         = ["COVID-19", "Covasim", "Malawi", "Nepal", "environmental surveillance"],
    install_requires = [
        "covasim",
    ],

    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
)
