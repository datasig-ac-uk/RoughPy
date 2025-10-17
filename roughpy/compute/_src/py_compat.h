#ifndef ROUGHPY_COMPUTE__SRC_PY_COMPAT_H
#define ROUGHPY_COMPUTE__SRC_PY_COMPAT_H

#include "py_headers.h"

#include <limits.h>
#include <stdint.h>

/*
 * structmember.h was deprecated in Python 3.12 with the functionality folded
 * into the main Python headers. Unfortunately, they also renamed all the
 * constants in this process and didn't include any compatibility. If we need
 * structmember.h, include it here by defining RPC_PYCOMPAT_INCLUDE_STRUCTMEMBER
 * This will make sure the header is included and proper aliases given for the
 * constants prior to 3.12
 */
#if defined(RPC_PYCOMPAT_INCLUDE_STRUCTMEMBER) && PY_VERSION_HEX < PYVER_HEX(3, 12)
#include <structmember.h>

#ifndef Py_T_CHAR
#define Py_T_CHAR T_CHAR
#endif

#ifndef Py_T_BYTE
#define Py_T_BYTE T_BYTE
#endif

#ifndef Py_T_UBYTE
#define Py_T_UBYTE T_UBYTE
#endif

#ifndef Py_T_SHORT
#define Py_T_SHORT T_SHORT
#endif

#ifndef Py_T_USHORT
#define Py_T_USHORT T_USHORT
#endif

#ifndef Py_T_INT
#define Py_T_INT T_INT
#endif

#ifndef Py_T_UINT
#define Py_T_UINT T_UINT
#endif

#ifndef Py_T_LONG
#define Py_T_LONG T_LONG
#endif

#ifndef Py_T_ULONG
#define Py_T_ULONG T_ULONG
#endif

#ifndef Py_T_LONGLONG
#define Py_T_LONGLONG T_LONGLONG
#endif

#ifndef Py_T_ULONGLONG
#define Py_T_ULONGLONG T_ULONGLONG
#endif

#ifndef Py_T_PYSSIZET
#define Py_T_PYSSIZET T_PYSSIZET
#endif

#ifndef Py_T_FLOAT
#define Py_T_FLOAT T_FLOAT
#endif

#ifndef Py_T_DOUBLE
#define Py_T_DOUBLE T_DOUBLE
#endif

#ifndef Py_T_BOOL
#define Py_T_BOOL T_BOOL
#endif

#ifndef Py_T_STRING
#define Py_T_STRING T_STRING
#endif

#ifndef Py_T_STRING_INPLACE
#define Py_T_STRING_INPLACE T_STRING_INPLACE
#endif

#ifndef Py_T_OBJECT_EX
#define Py_T_OBJECT_EX T_OBJECT_EX
#endif

#ifndef Py_T_NONE
#define Py_T_NONE T_NONE
#endif

// Member flags
#ifndef Py_READONLY
#define Py_READONLY READONLY
#endif

#ifndef Py_AUDIT_READ
#define Py_AUDIT_READ AUDIT_READ
#endif

#ifndef Py_RELATIVE_OFFSET
#define Py_RELATIVE_OFFSET RELATIVE_OFFSET
#endif

#endif

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


// Python 3.10 added new functions Py_NewRef and Py_XNewRef which are very useful
// we need those all over the place, so backport
#if PY_VERSION_HEX < PYVER_HEX(3, 10)
static inline PyObject* Py_NewRef(PyObject* obj)
{
  Py_INCREF(obj);
  return obj;
}

static inline PyObject* Py_XNewRef(PyObject* obj)
{
  Py_XINCREF(obj);
  return obj;
}


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
#define PyLong_AsInt32(obj) RPC_PyLongAsInt32((obj))
#endif
#endif




#ifdef __cplusplus
}
#endif

#endif// ROUGHPY_COMPUTE__SRC_PY_COMPAT_H
