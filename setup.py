import io
import os
import fnmatch
import platform
import sys
import re
import sysconfig
import importlib.metadata as ilm


from skbuild import setup
from pathlib import Path
from setuptools import find_packages


PROJECT_ROOT = Path(__file__).parent.absolute()


README_PATH = PROJECT_ROOT / "README.md"
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG"

DESCRIPTION = README_PATH.read_text()
DESCRIPTION += "\n\n\n## Changelog\n"
DESCRIPTION += CHANGELOG_PATH.read_text()

VERSION = "0.0.1"

if "VCPKG_INSTALLATION_ROOT" in os.environ:
    vcpkg = Path(os.environ["VCPKG_INSTALLATION_ROOT"], "scripts", "buildsystems", "vcpkg.cmake").resolve()
else:
    if not Path("vcpkg").exists():
        import subprocess as sp
        sp.run(["git", "clone", "https://github.com/Microsoft/vcpkg.git"])
    bootstrap_end = "bat" if platform.system() == "Windows" else "sh"
    sp.run([f"vcpkg/bootstrap-vcpkg.{bootstrap_end}"], shell=True, check=True)
    vcpkg = Path("vcpkg", "scripts", "buildsystems", "vcpkg.cmake").resolve()

prefix_path = []
if "CMAKE_PREFIX_PATH" in os.environ:
    prefix_path.extend(os.environ["CMAKE_PREFIX_PATH"].split(os.pathsep))





CMAKE_SETTINGS = [
    "-DROUGHPY_BUILD_TESTS:BOOL=OFF",
    "-DROUGHPY_BUILD_LA_CONTEXTS:BOOL=OFF",  # Temporarily
    "-DROUGHPY_GENERATE_DEVICE_CODE:BOOL=OFF",  # Until it's finished
    f"-DCMAKE_TOOLCHAIN_FILE={vcpkg}",
    "-DVCPKG_BUILD_TYPE=release",
    # "--debug-find", "--debug-output"
]

if platform.system() == "MacOs" and "CMAKE_OMP_ROOT" in os.environ:
    CMAKE_SETTINGS.extend([
        f"-DCMAKE_LIBRARY_PATH={os.environ['CMAKE_OMP_ROOT']}/lib",
        f"-DCMAKE_INCLUDE_PATH={os.environ['CMAKE_OMP_ROOT']}/include",
    ])


try:
    mkl = ilm.distribution("mkl-devel")

    # locate the cmake folder
    cmake_files = [f for f in mkl.files if f.name.endswith("cmake")]
    # should be {root}/lib/cmake/mkl/{f}
    cmake = cmake_files[0].locate().resolve().parent.parent
    # append {root} to prefix_path
    prefix_path.append(str(cmake.parent.parent))
    CMAKE_SETTINGS.append(f"-DMKL_DIR={cmake}")

except ilm.PackageNotFoundError:
    # pass
    raise

CMAKE_SETTINGS.append(
    f"-DCMAKE_PREFIX_PATH={os.pathsep.join(prefix_path)}"
)
os.environ["CMAKE_PREFIX_PATH"] = os.pathsep.join(prefix_path)

def filter_cmake_manifests(items):
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
    return manifest


setup(
    name="RoughPy",
    version=VERSION,
    author="The RoughPy Authors",
    author_email="info@datasig.ac.uk",
    license="BSD-3-Clause",
    keywords=["data", "streams", "rough paths", "signatures"],

    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",

    include_package_data=True,
    packages=["roughpy"],
    package_data={
        "roughpy": ["py.typed"]
    },

    cmake_process_manifest_hook=filter_cmake_manifests,
    cmake_args=CMAKE_SETTINGS,

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: BSD License",

        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",

        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS"
    ]

)
