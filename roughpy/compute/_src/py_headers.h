#ifndef ROUGHPY_COMPUTE__SRC_PY_HEADERS_H
#define ROUGHPY_COMPUTE__SRC_PY_HEADERS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>



#ifndef NPY_IMPORT_THE_APIS_PLEASE
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL

#include <numpy/arrayobject.h>



#if defined(__GNUC__) || defined(__clang__)
#define RPY_NO_EXPORT __attribute__((visibility("hidden")))
#else
#define RPY_NO_EXPORT
#endif


#endif //ROUGHPY_COMPUTE__SRC_PY_HEADERS_H
