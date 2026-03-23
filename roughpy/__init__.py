import importlib.metadata as _ilm
import os
import platform

from pathlib import Path as _Path

try:
    __version__ = _ilm.version("RoughPy")
except _ilm.PackageNotFoundError:
    __version__ = "0.0.0"


def get_include() -> str:
    """Return the include root for downstream builds using RoughPy headers.

    This is primarily intended for extension modules that need to compile
    against the headers shipped with the RoughPy wheel, including the staged
    ``roughpy_compute`` header tree.
    """
    package_root = _Path(__file__).resolve().parent
    installed_include = package_root / "include"
    if installed_include.is_dir():
        return str(installed_include)

    source_include = package_root.parent
    if (source_include / "roughpy_compute").is_dir():
        return str(source_include)

    return str(installed_include)


def _add_dynload_location(path: _Path):
    if platform.system() == "Windows":
        os.add_dll_directory(str(path))
        return


if platform.system() == "Windows":
    LIBS_DIR = _Path(__file__).parent.parent / "roughpy.libs"
    if LIBS_DIR.exists():
        os.add_dll_directory(str(LIBS_DIR))

try:
    iomp = _ilm.distribution("intel-openmp")
    libs = [f for f in iomp.files if f.name.startswith("libiomp5")]
    if libs:
        _add_dynload_location(libs[0].locate().resolve().parent)
    del iomp
    del libs
except _ilm.PackageNotFoundError:
    pass

import roughpy._roughpy
from roughpy._roughpy import *

from . import tensor_functions
