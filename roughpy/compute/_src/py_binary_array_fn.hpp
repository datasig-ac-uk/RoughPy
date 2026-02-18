#ifndef ROUGHPY_COMPUTE__SRC_PY_BINARY_ARARY_FN_HPP
#define ROUGHPY_COMPUTE__SRC_PY_BINARY_ARARY_FN_HPP

#include <roughpy/pycore/py_headers.h>

#include <roughpy_compute/common/cache_array.hpp>

#include "call_config.hpp"
#include "check_dims.hpp"


namespace rpy::compute {
template<typename Fn>
RPY_NO_EXPORT
PyObject *outer_loop_binary(
    PyArrayObject *out,
    PyArrayObject *arg,
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
    auto const arg_stride = PyArray_STRIDE(arg, ndims - 1) / sizeof(Scalar);

    for (npy_intp i = 0; i < n_elements; ++i, advance()) {
        auto *out_ptr = static_cast<Scalar *>(PyArray_GetPtr(out, index.data()));
        auto const *arg_ptr = static_cast<Scalar const *>(PyArray_GetPtr(
            arg,
            index.data()));

        // if the stride is one then pass the raw pointer instead so we can
        // benefit from contiguous iteration.
        if (out_stride == 1 && arg_stride == 1) {
            fn(out_ptr, arg_ptr);
        } else {
            fn(
                StridedDenseIterator<Scalar *>(out_ptr, out_stride),
                StridedDenseIterator<Scalar const *>(arg_ptr, arg_stride));
        }
    }

    Py_RETURN_NONE;
}

template<template <typename> class Fn>
[[gnu::always_inline]] RPY_NO_EXPORT inline
PyObject *binary_function_outer(PyObject *out_obj,
                                PyObject *arg_obj,
                                CallConfig &config) {
    // static constexpr char const* const kwords[] = {
    //         "out", "arg", "basis", "rhs_max_degree", nullptr
    // };
    //
    // PyObject *out_obj, *arg_obj;
    // PyObject *basis_obj=nullptr;
    // CallConfig config;
    //
    // if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|i", kwords,
    //     &out_obj, &arg_obj, &basis_obj, &config.rhs_max_degree)) {
    //     return nullptr;
    // }
    //
    // if (config.rhs_max_degree == -1 || config.rhs_max_degree >= config.depth) {
    //     config.rhs_max_degree = config.depth;
    // }

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

    if (!PyArray_Check(arg_obj)) {
        PyErr_SetString(PyExc_TypeError, "arg must be a numpy array");
        return nullptr;
    }

    auto *arg_arr = reinterpret_cast<PyArrayObject *>(arg_obj);
    if (PyArray_TYPE(arg_arr) != dtype) {
        PyErr_SetString(PyExc_TypeError, "arg must have the same dtype as out");
        return nullptr;
    }

    auto const arg_ndims = PyArray_NDIM(arg_arr);
    auto const *arg_shape = PyArray_DIMS(arg_arr);

    if (!check_dims(arg_shape,
                    arg_ndims - core_dims,
                    shape,
                    n_dims - core_dims)) {
        PyErr_SetString(PyExc_ValueError,
                        "arg and out must have the same shape");
        return nullptr;
    }

    switch (dtype) {
        case NPY_FLOAT64: return outer_loop_binary(
                out_arr,
                arg_arr,
                Fn<double>{config}
            );
        case NPY_FLOAT32: return outer_loop_binary(
                out_arr,
                arg_arr,
                Fn<float>{config}
            );
        default: PyErr_SetString(PyExc_TypeError, "unsupported dtype");
            return nullptr;
    }
}
} // namespace rpy::compute

#endif //ROUGHPY_COMPUTE__SRC_PY_BINARY_ARARY_FN_HPP
