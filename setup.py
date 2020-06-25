# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Package setup"""

import os
from shutil import rmtree
import sys

from setuptools import find_packages, setup, Command

from trellis.__version__ import __version__


with open("README.md", "r") as fh:
    long_description = fh.read()


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            here = os.path.abspath(os.path.dirname(__file__))
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('"{0}" setup.py sdist bdist_wheel'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('"{0}" -m twine upload dist/*'.format(sys.executable))

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(__version__))
        os.system('git push --tags')

        sys.exit()


setup(
    name="beacon-trellis",
    version=__version__,
    author="Benjamin Pryke",
    author_email="ben.pryke@beacon.io",
    maintainer="Beacon Platform",
    description="Trellis is a deep hedging and deep pricing framework for quantitative finance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Beacon-Platform/trellis",
    project_urls={"Maintainer Homepage": "https://www.beacon.io"},
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=[
        "matplotlib>=3.0.0",
        "numpy>=1.16.0",
        "scipy>=1.4.1",
        "seaborn>=0.9.0",
        "tensorflow==2.1.0",
    ],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    cmdclass={'upload': UploadCommand},
)
