


#include "dense_basic.h"


#include <roughpy_compute/common/cache_array.hpp>
#include <roughpy_compute/dense/views.hpp>

#include <roughpy_compute/dense/basic/free_tensor_antipode.hpp>
#include <roughpy_compute/dense/basic/free_tensor_fma.hpp>
#include <roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp>
#include <roughpy_compute/dense/basic/shuffle_tensor_product.hpp>
#include <roughpy_compute/dense/basic/vector_addition.hpp>
#include <roughpy_compute/dense/basic/vector_inplace_addition.hpp>
#include <roughpy_compute/dense/basic/vector_inpace_scalar_mulitplication.hpp>
#include <roughpy_compute/dense/basic/vector_scalar_multiply.hpp>


enum class Status {
    Success = 0,
    PyError = 1,
};



namespace {


[[gnu::always_inline]] inline
bool check_dims(npy_intp const * shape1, npy_intp ndims1 , npy_intp const * shape2, npy_intp ndims2)
{
	if (ndims1 != ndims2) {
		return false;
	}

	for (npy_intp i = 0; i < ndims1; ++i) {
		if (shape1[i] != shape2[i]) {
			return false;
		}
	}

	return true;
}


template <typename Fn>
[[gnu::always_inline]] inline
PyObject* outer_loop_ternary(
    PyObjectArray* out,
    PyObjectArray* lhs
    PyObjectArray* rhs,
    Fn&& fn
)
{
    using Scalar = typename Fn::Scalar;
    npy_intp ndims = PyArray_NDIM(out);

    npy_intp n_elements = 1;
    auto* shape = PyArray_SHAPE(out);
    for (npy_intp i = 0; i < ndims - CoreDims; ++i) {
        n_elements *= shape[i];
    }

    rpy::compute::CacheArray<npy_intp, CoreDims + 1> index(ndims);

    auto advance = [&index, &ndims] {
        for (npy_intp pos = ndims-1 -CoreDims; pos >= 0; --pos) {
            index[pos] += 1;
            if (index[pos] < shape[pos]) {
                break;
            } else {
                index[pos] = 0;
            }
        }
    };

    for (npy_intp i=0; i < n_elements; ++i, advance()) {

       fn(static_cast<Scalar*>(PyArray_GetPtr(out, index.data())),
          static_cast<Scalar const*>(PyArray_GetPtr(lhs, index.data())),
          static_cast<Scalar const*>(PyArray_GetPtr(rhs, index.data())));
    }

    Py_RETURN_NONE;
}


template <template <typename> class Fn>
[[gnu::always_inline]] inline
PyObject* ternary_function_outer(PyObject* self [[maybe_unused]], PyObject* args, PyObject* kwargs)
    static char const* const kwords = {
        "out", "lhs", "rhs", "width", "depth", "lhs_depth", "rhs_depth", nullptr
    };"

    PyObject* out_obj, lhs_obj, rhs_obj;
    int32_t width, depth, lhs_depth = -1, rhs_depth = -1;

    if (!PyArgs_ParseTupleAndKeywords(args, kwargs, "OOOii|ii", kwords,
        &out_obj, &lhs_obj, &rhs_obj, &width, &depth, &lhs_depth, &rhs_depth)) {
        return nullptr;
    }


    if (lhs_depth == -1 || lhs_depth >= depth) {
        lhs_depth = depth;
    }
    if (rhs_depth == -1 || rhs_depth >= depth) {
        rhs_depth = depth;
    }

    constexpr auto core_dims = Fn<double>::CoreDims;

    if (!PyArray_Check(out_obj)) {
        PyErr_SetString(PyExc_TypeError, "out must be a numpy array");
        return nullptr;
    }

    auto* out_arr = reinterpret_cast<PyArrayObject*>(out_obj);

    auto const n_dims = PyArray_NDIM(out_arr);
    auto const dtype = PyArray_TYPE(out_arr);
    auto const itemsize = PyArray_ITEMSIZE(out_arr);

    auto const* shape = PyArray_DIMS(out_arr);

	if (n_dims < core_dims) {
		PyErr_SetString(PyExc_ValueError, "invalid shape");
		return nullptr;
	}

	if (PyArray_STRIDE(out_arr, n_dims - 1) != itemsize) {
		PyErr_SetString(PyExc_ValueError, "inner-most dimension must be contiguous");
		return nullptr;
	}

	if (PyArray_Check(lhs_obj) {
		PyErr_SetString(PyExc_TypeError, "lhs must be a numpy array");
		return nullptr;
	}

	auto* lhs_arr = reinterpret_cast<PyArrayObject*>(lhs_obj);

	if (PyArray_TYPE(lhs_arr) != dtype) {
		PyErr_SetString(PyExc_TypeError, "lhs must have the same dtype as out");
		return nullptr;
	}

	auto lhs_ndims = PyArray_NDIM(lhs_arr);
	auto const* lhs_shape = PyArray_DIMS(lhs_arr);

	if (!check_dims(lhs_shape, lhs_ndims, shape, n_dims)) {
		PyErr_SetString(PyExc_ValueError, "lhs and out must have the same shape");
		return nullptr;
	}

	if (PyArray_STRIDE(lhs_arr, n_dims - 1) != itemsize) {
		PyErr_SetString(PyExc_ValueError, "inner-most dimension must be contiguous");
		return nullptr;
	}


	if (PyArray_Check(rhs_obj) {
		PyErr_SetString(PyExc_TypeError, "rhs must be a numpy array");
		return nullptr;
	}

	auto* rhs_arr = reinterpret_cast<PyArrayObject*>(rhs_obj);

	if (PyArray_TYPE(rhs_arr) != dtype) {
		PyErr_SetString(PyExc_TypeError, "rhs must have the same dtype as out");
		return nullptr;
	}

	auto rhs_ndims = PyArray_NDIM(rhs_arr);
	auto const* rhs_shape = PyArray_DIMS(rhs_arr);

	if (!check_dims(rhs_shape, rhs_ndims, shape, n_dims)) {
		PyErr_SetString(PyExc_ValueError, "rhs and out must have the same shape");
		return nullptr;
	}

	if (PyArray_STRIDE(rhs_arr, n_dims - 1) != itemsize) {
		PyErr_SetString(PyExc_ValueError, "inner-most dimension must be contiguous");
		return nullptr;
	}


    switch (dtype) {
        case NPY_FLOAT64:
            return outer_loop_ternary(
                out_arr, lhs_arr, rhs_arr,
                Fn<double>{width, depth}
                );
        case NPY_FLOAT32:
            return outer_loop_ternary(
                out_arr, lhs_arr, rhs_arr,
                Fn<float>{width, depth}
            );
        default:
            PyErr_SetString(PyExc_TypeError, "unsupported dtype");
            return nullptr;
    }
}




/*******************************************************************************
 * Free tensor FMA
 ******************************************************************************/
namespace {

template <typename Scalar_>
struct DenseFTFma {
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    rpy::compute::CacheArray<size_t, CoreDims + 1> degree_begin;

    DenseFTFma(int32_t width, int32_t depth)
        : degree_begin(depth + 2)
    {
        auto const s_width = static_cast<size_t>(width);
        degree_begin[0] = 0;
        for (size_t i = 1; i <= depth + 1; ++i) {
            degree_begin[i] = 1 + degree_begin[i - 1] * s_width;
        }
    }

    void operator()(void* out_ptr, void const* lhs_ptr, void const* rhs_ptr, AlgebraData const* fn_data) const
    {

        DenseTensorView<Scalar*> out_view(
			out_ptr,
            {degree_begin.data(), width, depth}, 0, depth
        );

        DenseTensorView<Scalar const*> lhs_view(
			lhs_ptr,
            {degree_begin.data(), width, depth}, 0, rhs_depth
        );

        DenseTensorView<Scalar const*> rhs_view(
			rhs_ptr,
            {degree_begin.data(), width, depth}, 0, rhs_depth
        );

        basic::free_tensor_fma(out, lhs, rhs);
    }
};

} // namespace


PyObject* dense_ft_fma(PyObject* out, PyObject* lhs, PyObject* rhs)
{



}