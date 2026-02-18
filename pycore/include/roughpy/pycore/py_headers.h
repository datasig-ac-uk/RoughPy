#ifndef ROUGHPY_PYCORE_PY_HEADERS_H
#define ROUGHPY_PYCORE_PY_HEADERS_H


#define PY_SSIZE_T_CLEAN
// #define Py_LIMITED_API 0x030a0000
#  include <Python.h>



#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#ifndef NPY_IMPORT_THE_APIS_PLEASE
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#endif
#define PY_ARRAY_UNIQUE_SYMBOL RPY_COMPUTE_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL RPY_COMPUTE_UFUNC_API

#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>


#define PYVER_HEX(major, minor) \
(((major) << 24) | \
((minor) << 16))


#include <roughpy/core/macros.h>








#endif// ROUGHPY_PYCORE_PY_HEADERS_H
