#ifndef ROUGHPY_PYCORE_COMPAT_H
#define ROUGHPY_PYCORE_COMPAT_H

#include "py_headers.h"

#include <limits.h>
#include <stdint.h>

/*
 * The pythoncapi_compat header contains backports of a large number of number
 * of functions from the CAPI of convenient functions and macros that have been
 * added in more recent versions of Python that are extremely useful for
 * writing extensions. This header was originally our attempt to recreate this
 * header before it was known to exist.
 */
#include <pythoncapi_compat.h>

#ifndef Py_XSETREF
#define Py_XSETREF(op, op2) \
    do {                    \
        PyObject *_py_tmp = (PyObject *)(op); \
        (op) = (op2);       \
        Py_XDECREF(_py_tmp); \
    } while (0)
#endif

#ifndef Py_SETREF
#define Py_SETREF(op, op2) \
    do {                   \
        PyObject *_py_tmp = (PyObject *)(op); \
        (op) = (op2);      \
        Py_DECREF(_py_tmp); \
    } while (0)
#endif

#ifndef Py_NewRef
static inline PyObject* _RPY_Py_NewRef(PyObject *obj)
{
    Py_INCREF(obj);
    return obj;
}
#define Py_NewRef(obj) _RPY_Py_NewRef((PyObject *)(obj))
#endif

#ifndef Py_XNewRef
static inline PyObject* _RPY_Py_XNewRef(PyObject *obj)
{
    Py_XINCREF(obj);
    return obj;
}
#define Py_XNewRef(obj) _RPY_Py_XNewRef((PyObject *)(obj))
#endif

/*
 * structmember.h was deprecated in Python 3.12 with the functionality folded
 * into the main Python headers. Unfortunately, they also renamed all the
 * constants in this process and didn't include any compatibility. If we need
 * structmember.h, include it here by defining RPC_PYCOMPAT_INCLUDE_STRUCTMEMBER
 * This will make sure the header is included and proper aliases given for the
 * constants prior to 3.12
 */
#if defined(RPY_PYCOMPAT_INCLUDE_STRUCTMEMBER) && PY_VERSION_HEX < PYVER_HEX(3, 12)
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
#define RPY_PY_KWORD_CAST(ARG) const_cast<char**>(ARG)
#else
#define RPY_PY_KWORD_CAST(ARG) (char**) ARG
#endif
#else
#define RPY_PY_KWORD_CAST(ARG) ARG
#endif



#endif// ROUGHPY_PYCORE_COMPAT_H
