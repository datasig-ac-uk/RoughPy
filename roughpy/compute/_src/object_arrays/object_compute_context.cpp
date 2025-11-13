
#include "object_compute_context.hpp"



using namespace rpy::compute;

ObjectComputeContext::ObjectComputeContext()
    : zero_(nullptr), one_(nullptr)
{

}
ObjectComputeContext::ObjectComputeContext(PyObject* type_or_class)
    : zero_(nullptr), one_(nullptr)
{
    if (!PyType_Check(type_or_class)) {
        PyErr_SetString(PyExc_TypeError,
            "Object type must be a type object");
        throw PyErrAlreadySet();
    }

    PyObjHandle pylong_one(PyLong_FromLong(1), false);
    if (!pylong_one) {
        throw PyErrAlreadySet();
    }
    PyObjHandle pylong_zero(PyLong_FromLong(0), false);
    if (!pylong_zero) {
        throw PyErrAlreadySet();
    }

    one_ = PyObject_CallFunctionObjArgs(type_or_class, pylong_one.obj(), nullptr);
    if (!one_) {
        throw PyErrAlreadySet();
    }

    zero_ = PyObject_CallFunctionObjArgs(type_or_class, pylong_zero.obj(), nullptr);
    if (!zero_) {
        Py_DECREF(one_);
        throw PyErrAlreadySet();
    }
}

ObjectComputeContext::~ObjectComputeContext()
{
    Py_DECREF(zero_); zero_ = nullptr;
    Py_DECREF(one_); one_ = nullptr;
}
