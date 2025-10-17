#ifndef ROUGHPY_COMPUTE__SRC_PY_COMPAT_H
#define ROUGHPY_COMPUTE__SRC_PY_COMPAT_H

#include "py_headers.h"

#include <limits.h>
#include <stdint.h>

#if PY_VERSION_HEX < PYVER_HEX(3, 13)
#ifdef __cplusplus
#define RPC_PY_KWORD_CAST(ARG) const_cast<char**>(ARG)
#else
#define RPC_PY_KWORD_CAST(ARG) (char**) ARG
#endif
#else
#define RPC_PY_KWORD_CAST(ARG) ARG
#endif


// Python 3.10 added Py_Is and Py_IsNone which are both useful
#if !defined(Py_Is)
#define Py_Is(x, y) ((x) == (y))
#endif
#if !defined(Py_IsNone)
#define Py_IsNone(x) Py_Is((x), Py_None)
#endif

#ifdef __cplusplus
extern "C" {
#endif



// Python 3.14 introduced PyLong_AsInt32 which we use in places to convert to a
// 32-bit integer when int and long are not the same size (e.g. most 64-bit Unix
// systems). We implement it here if it is not defined.

RPY_NO_EXPORT
int32_t RPC_PyLongAsInt32(PyObject* pylong);

#if PY_VERSION_HEX < PYVER_HEX(3, 14)
#if LONG_MAX == INT32_MAX
#define PyLong_AsInt32(obj) PyLong_AsLong((obj))
#elif PY_VERSION_HEX >= PYVER_HEX(3, 13) && INT_MAX == INT32_MAX
#define PyLong_AsInt32(obj) PyLong_AsInt((obj))
#else
#define PyLong_AsInt32((obj)) RPC_PyLongAsInt32((obj))
#endif




#ifdef __cplusplus
}
#endif

#endif// ROUGHPY_COMPUTE__SRC_PY_COMPAT_H
