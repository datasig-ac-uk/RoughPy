#ifndef ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_COMPUTE_CONTEXT_HPP
#define ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_COMPUTE_CONTEXT_HPP

#include "py_headers.hpp"
#include "py_obj_handle.hpp"

namespace rpy::compute {


class ObjectComputeContext
{
    PyObject* zero_;
    PyObject* one_;

public:
    using Scalar = PyObjHandle;
    using Rational = PyObjHandle;
    using Real = PyObjHandle;

    ObjectComputeContext();
    explicit ObjectComputeContext(PyObject* type);

    ObjectComputeContext(PyObject* zero, PyObject* one)
        : zero_(Py_NewRef(zero)), one_(Py_NewRef(one))
    {
    }

    ~ObjectComputeContext();

    constexpr ObjectRef zero() { return ObjectRef(&zero_); }
    constexpr ObjectRef one() { return ObjectRef(&one_); }



};




}

#endif// ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_COMPUTE_CONTEXT_HPP
