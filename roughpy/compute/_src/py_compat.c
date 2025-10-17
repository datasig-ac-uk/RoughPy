#include "py_compat.h"



int32_t RPC_PyLongAsInt32(PyObject* pylong)
{
    const long result = PyLong_AsLong(pylong);
    // we would usually need to test if result is -1 and if error occurred, but
    // actually the following tests will always pass if result == -1 so we can
    // just rely on the caller to do that.

    if (result < INT32_MIN || result > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError,
            "Value is out of range for a 32-bit integer");
        return -1;
    }

    return (int32_t) result;
}
