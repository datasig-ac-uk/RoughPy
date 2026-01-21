#ifndef ROUGHPY_COMPUTE__SRC_PY_TERNARY_ARRAY_FN_HPP
#define ROUGHPY_COMPUTE__SRC_PY_TERNARY_ARRAY_FN_HPP

#include <roughpy/pycore/py_headers.h>

#include <roughpy_compute/common/cache_array.hpp>

#include "call_config.hpp"
#include "check_dims.hpp"

namespace rpy::compute {
template<typename Fn>
RPY_NO_EXPORT
PyObject *outer_loop_ternary(
    PyArrayObject *out,
    PyArrayObject *lhs,
    PyArrayObject *rhs,
    Fn &&fn
) {
    using Scalar = typename Fn::Scalar;
    npy_intp ndims = PyArray_NDIM(out);

    npy_intp n_elements = 1;
    auto *shape = PyArray_SHAPE(out);
    for (npy_intp i = 0; i < ndims - Fn::CoreDims; ++i) {
        n_elements *= shape[i];
    }

    CacheArray<npy_intp, Fn::CoreDims + 1> index(ndims);
    for (npy_intp i = 0; i < ndims; ++i) {
        index[i] = 0;
    }

    auto advance = [&index, &ndims, &shape] {
        for (npy_intp pos = ndims - 1 - Fn::CoreDims; pos >= 0; --pos) {
            index[pos] += 1;
            if (index[pos] < shape[pos]) { break; } else { index[pos] = 0; }
        }
    };

    auto const out_stride = PyArray_STRIDE(out, ndims - 1) / sizeof(Scalar);
    auto const lhs_stride = PyArray_STRIDE(lhs, ndims - 1) / sizeof(Scalar);
    auto const rhs_stride = PyArray_STRIDE(rhs, ndims - 1) / sizeof(Scalar);

    for (npy_intp i = 0; i < n_elements; ++i, advance()) {
        auto *out_ptr = static_cast<Scalar *>(PyArray_GetPtr(out, index.data()));
        auto const *lhs_ptr = static_cast<Scalar const *>(PyArray_GetPtr(
            lhs,
            index.data()));
        auto const *rhs_ptr = static_cast<Scalar const *>(PyArray_GetPtr(
            rhs,
            index.data()));

        // if the stride is one then pass the raw pointer instead so we can
        // benefit from contiguous iteration.
        if (out_stride == 1 && lhs_stride == 1 && rhs_stride == 1) {
            fn(out_ptr, lhs_ptr, rhs_ptr);
        } else {
            fn(
                StridedDenseIterator<Scalar *>(out_ptr, out_stride),
                StridedDenseIterator<Scalar const *>(lhs_ptr, lhs_stride),
                StridedDenseIterator<Scalar const *>(rhs_ptr, rhs_stride));
        }
    }

    Py_RETURN_NONE;
}


template<template <typename> class Fn>
[[gnu::always_inline]] RPY_NO_EXPORT inline
PyObject *ternary_function_outer(PyObject *out_obj [[maybe_unused]],
                                 PyObject *lhs_obj,
                                 PyObject *rhs_obj,
                                 CallConfig &config
) {
    constexpr auto core_dims = Fn<float>::CoreDims;

    if (!update_algebra_params(config, Fn<float>::n_args, Fn<float>::arg_basis_mapping)) {
        return nullptr;
    }

    if (!PyArray_Check(out_obj)) {
        PyErr_SetString(PyExc_TypeError, "out must be a numpy array");
        return nullptr;
    }

    auto *out_arr = reinterpret_cast<PyArrayObject *>(out_obj);

    auto const n_dims = PyArray_NDIM(out_arr);
    auto const dtype = PyArray_TYPE(out_arr);

    auto const *shape = PyArray_DIMS(out_arr);

    if (n_dims < core_dims) {
        PyErr_SetString(PyExc_ValueError, "invalid shape");
        return nullptr;
    }

    if (!PyArray_Check(lhs_obj)) {
        PyErr_SetString(PyExc_TypeError, "lhs must be a numpy array");
        return nullptr;
    }

    auto *lhs_arr = reinterpret_cast<PyArrayObject *>(lhs_obj);

    if (PyArray_TYPE(lhs_arr) != dtype) {
        PyErr_SetString(PyExc_TypeError, "lhs must have the same dtype as out");
        return nullptr;
    }

    auto const lhs_ndims = PyArray_NDIM(lhs_arr);
    auto const *lhs_shape = PyArray_DIMS(lhs_arr);

    if (!check_dims(lhs_shape, lhs_ndims - core_dims, shape, n_dims - core_dims)) {
        PyErr_SetString(PyExc_ValueError,
                        "lhs and out must have the same shape");
        return nullptr;
    }

    if (!PyArray_Check(rhs_obj)) {
        PyErr_SetString(PyExc_TypeError, "rhs must be a numpy array");
        return nullptr;
    }

    auto *rhs_arr = reinterpret_cast<PyArrayObject *>(rhs_obj);

    if (PyArray_TYPE(rhs_arr) != dtype) {
        PyErr_SetString(PyExc_TypeError, "rhs must have the same dtype as out");
        return nullptr;
    }

    auto const rhs_ndims = PyArray_NDIM(rhs_arr);
    auto const *rhs_shape = PyArray_DIMS(rhs_arr);

    if (!check_dims(rhs_shape,
                    rhs_ndims - core_dims,
                    shape,
                    n_dims - core_dims)) {
        PyErr_SetString(PyExc_ValueError,
                        "rhs and out must have the same shape");
        return nullptr;
    }

    switch (dtype) {
        case NPY_FLOAT64: return outer_loop_ternary(
                out_arr,
                lhs_arr,
                rhs_arr,
                Fn<double>{config}
            );
        case NPY_FLOAT32: return outer_loop_ternary(
                out_arr,
                lhs_arr,
                rhs_arr,
                Fn<float>{config}
            );
        default: PyErr_SetString(PyExc_TypeError, "unsupported dtype");
            return nullptr;
    }
}
} // namespace rpy::compute

#endif //ROUGHPY_COMPUTE__SRC_PY_TERNARY_ARRAY_FN_HPP
