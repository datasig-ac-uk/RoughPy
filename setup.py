import io
import os
import fnmatch
import sys
import re

from skbuild import setup
from pathlib import Path
from setuptools import find_packages


PROJECT_ROOT = Path(__file__).parent.absolute()


README_PATH = PROJECT_ROOT / "README.md"
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG"

DESCRIPTION = README_PATH.read_text()
DESCRIPTION += "\n\n\n## Changelog"
DESCRIPTION += CHANGELOG_PATH.read_text()

VERSION = "0.0.0"

CMAKE_SETTINGS = []


def filter_cmake_manifests(items: list[str]) -> list[str]:
    def _filter(item):
        item = str(item)
        if item.endswith(".pc"):
            return False

        if item.endswith(".cmake"):
            return False

        if item.endswith(".cpp"):
            return False

        if item.endswith(".h"):
            return False

        if item.endswith(".a"):
            return False

        # m = re.search(r"[a-zA-Z0-9_]+\.so(?:\.\d+\.\d+\.\d+)?$", item)
        # if m is not None:
        #     return False

        if item.endswith("recombine.so") or item.endswith("recombine.so.2.0.2"):
            return False

        return True


    manifest = list(filter(_filter, items))
    with open("manifests.txt", "wt") as fp:
        print(*manifest, sep='\n', file=fp)
    return manifest


setup(
    name="roughpy",
    version=VERSION,
    author="Terry Lyons, Terry Lyons, DataSig group",
    author_email="info@datasig.ac.uk",
    license="BSD-3-Clause",
    keywords=["data", "streams", "rough paths", "signatures"],

    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",

    include_package_data=True,
    packages=["roughpy"],

    cmake_process_manifest_hook=filter_cmake_manifests,
    cmake_args=CMAKE_SETTINGS,

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: BSD License",

        "Programming Language :: C++"
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",

        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MaxOS"
    ]

)
