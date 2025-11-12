#include "py_compat.h"



int RPC_PyLongAsInt32(PyObject* pylong, int32_t* value)
{
    const long result = PyLong_AsLong(pylong);

    if (result == -1 && PyErr_Occurred()) {
        return -1;
    }

    if (result < INT32_MIN || result > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError,
            "Value is out of range for a 32-bit integer");
        return -1;
    }

    *value = result;

    return 0;
}
