#ifndef ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_COMPUTE_CONTEXT_HPP
#define ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_COMPUTE_CONTEXT_HPP

#include "py_headers.h"

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

    // Python objects are inherently mutable, the wrapper maintains the
    // const invariant, rather than the pointer.
    constexpr ObjectRef zero() const
    {
        return ObjectRef(const_cast<PyObject**>(&zero_));
    }
    constexpr ObjectRef one() const
    {
        return ObjectRef(const_cast<PyObject**>(&one_));
    }




};




}

#endif// ROUGHPY_COMPUTE__SRC_OBJECT_ARRAYS_OBJECT_COMPUTE_CONTEXT_HPP
