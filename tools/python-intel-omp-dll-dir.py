
import os
import sys
import importlib.metadata as ilm



try:
    omp = ilm.distribution("intel-openmp")
    dlls = [f for f in omp.files if f.name == "libiomp5md.dll"]
    if not dlls:
        sys.exit(1)
    dll_dir = dlls[0].locate().resolve().parent
    print(dll_dir)
except ilm.PackageNotFoundError:
    sys.exit(1)
