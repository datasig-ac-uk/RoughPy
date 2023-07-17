import importlib.metadata as _ilm
import os
import platform

from pathlib import Path

try:
    __version__ = _ilm.version("RoughPy")
except _ilm.PackageNotFoundError:
    __version__ = "0.0.0"


def _add_dynload_location(path: Path):
    if platform.system() == "Windows":
        os.add_dll_directory(str(path))
        return


if platform.system() == "Windows":
    LIBS_DIR = Path(__file__).parent.parent / "roughpy.libs"
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
