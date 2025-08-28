#include "dense_basic.h"


#include <roughpy_compute/common/cache_array.hpp>
#include <roughpy_compute/dense/views.hpp>

#include <roughpy_compute/dense/basic/free_tensor_antipode.hpp>
#include <roughpy_compute/dense/basic/free_tensor_fma.hpp>
#include <roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp>
#include <roughpy_compute/dense/basic/shuffle_tensor_product.hpp>
#include <roughpy_compute/dense/basic/apply_sparse_linear_map.hpp>


// The vector operations are of limited use to us here. We
// #include <roughpy_compute/dense/basic/vector_addition.hpp>
// #include <roughpy_compute/dense/basic/vector_inplace_addition.hpp>
// #include <roughpy_compute/dense/basic/vector_inplace_scalar_multiply.hpp>
// #include <roughpy_compute/dense/basic/vector_scalar_multiply.hpp>


#include "call_config.hpp"

#include "py_obj_handle.hpp"
#include "py_binary_array_fn.hpp"
#include "py_ternary_array_fn.hpp"


using namespace rpy::compute;


/*******************************************************************************
 * Free tensor FMA
 ******************************************************************************/
namespace {
template<typename Scalar_>
struct DenseFTFma {
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    CallConfig const *config_;

    explicit DenseFTFma(CallConfig const &config)
        : config_(&config) {
    }

    template<typename OutIter, typename LhsIter, typename RhsIter>
    void operator()(OutIter out_iter,
                    LhsIter lhs_iter,
                    RhsIter rhs_iter) const {
        auto const *basis = static_cast<TensorBasis const *>(config_->basis_data);

        DenseTensorView<OutIter> out_view(
            out_iter,
            *basis,
            config_->out_min_degree,
            config_->out_max_degree
        );

        DenseTensorView<LhsIter> lhs_view(
            lhs_iter,
            *basis,
            config_->lhs_min_degree,
            config_->lhs_max_degree
        );

        DenseTensorView<RhsIter> rhs_view(
            rhs_iter,
            *basis,
            config_->rhs_min_degree,
            config_->rhs_max_degree
        );

        basic::ft_fma(out_view, lhs_view, rhs_view);
    }
};
} // namespace


PyObject *py_dense_ft_fma(PyObject *self [[maybe_unused]], PyObject *args, PyObject *kwargs) {
    static constexpr char const *const kwords[] = {
        "out", "lhs", "rhs", "basis", "out_depth", "lhs_depth", "rhs_depth",
        nullptr
    };

    PyObject *out_obj, *lhs_obj, *rhs_obj;
    PyObject *basis_obj = nullptr;

    CallConfig config;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOOO|iii",
                                     kwords,
                                     &out_obj,
                                     &lhs_obj,
                                     &rhs_obj,
                                     &basis_obj,
                                     &config.out_max_degree,
                                     &config.lhs_max_degree,
                                     &config.rhs_max_degree
    )) {
        return nullptr;
    }

    // if (!update_depth_params(config)) {
    //     PyErr_SetString(PyExc_ValueError, "incompatible depth parameters");
    //     return nullptr;
    // }

    TensorBasis basis;
    auto const degree_begins_handle = to_basis(basis_obj, basis);

    if (!degree_begins_handle) {
        // Error already set
        return nullptr;
    }

    config.basis_data = &basis;

    if (!update_algebra_params(config)) {
        return nullptr;
    }

    return ternary_function_outer<DenseFTFma>(out_obj, lhs_obj, rhs_obj, config);
}


/*******************************************************************************
 * Free tensor Inplace multiply
 ******************************************************************************/
namespace {
template<typename Scalar_>
struct DenseFTInplaceMul {
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    CallConfig const *config_;

    explicit DenseFTInplaceMul(CallConfig const &config)
        : config_(&config) {
    }

    template<typename OutIter, typename RhsIter>
    void operator()(OutIter out_iter,
                    RhsIter rhs_iter) const {
        auto const *basis = static_cast<TensorBasis const *>(config_->basis_data);

        DenseTensorView<OutIter> out_view(
            out_iter,
            *basis,
            config_->out_min_degree,
            config_->out_max_degree
        );

        DenseTensorView<RhsIter> rhs_view(
            rhs_iter,
            *basis,
            config_->rhs_min_degree,
            config_->rhs_max_degree
        );

        basic::ft_inplace_mul(out_view, rhs_view);
    }
};
} // namespace


PyObject *py_dense_ft_inplace_mul(PyObject *self, PyObject *args, PyObject *kwargs) {
    static constexpr char const *const kwords[] = {
        "out", "lhs", "rhs", "basis", "out_depth", "rhs_depth",
        nullptr
    };

    PyObject *out_obj, *rhs_obj;
    PyObject *basis_obj = nullptr;

    CallConfig config;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|ii", kwords,
                                     &out_obj, &rhs_obj, &basis_obj, &config.rhs_max_degree)) {
        return nullptr;
    }

    TensorBasis basis;
    auto handle = to_basis(basis_obj, basis);
    if (!handle) {
        // error already set
        return nullptr;
    }

    config.basis_data = &basis;

    if (!update_algebra_params(config)) {
        // error already set
        return nullptr;
    }

    return binary_function_outer<DenseFTInplaceMul>(out_obj, rhs_obj, config);
}


/*******************************************************************************
 * free tensor antipode
 ******************************************************************************/
namespace {
template<typename S>
struct DenseAntipode {
    using Scalar = S;
    static constexpr npy_intp CoreDims = 1;

    CallConfig const *config_;

    explicit constexpr DenseAntipode(CallConfig const &config)
        : config_(&config) {}

    template<typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const {
        auto const *basis = static_cast<TensorBasis const *>(config_->basis_data);

        DenseTensorView<OutIter> out(out_iter, *basis, config_->out_min_degree, config_->out_max_degree);
        DenseTensorView<ArgIter> arg(arg_iter, *basis, config_->rhs_min_degree, config_->rhs_max_degree);

        basic::ft_antipode(out, arg, basic::BasicAntipodeConfig{}, basic::DefaultSigner{});
    }
};
} //namespace

PyObject *py_dense_antipode(PyObject * self [[maybe_unused]], PyObject *args, PyObject *kwargs) {

    static constexpr char const *const kwords[] = {
        "out", "arg", "basis", "out_depth", "arg_depth", nullptr
    };

    PyObject *out_obj, *arg_obj;
    PyObject *basis_obj = nullptr;

    CallConfig config;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|ii", kwords, &out_obj, &arg_obj, &basis_obj, &config.rhs_max_degree)) {
        return nullptr;
    }

    TensorBasis basis;
    auto handle = to_basis(basis_obj, basis);
    if (!handle) {
        // error already set
        return nullptr;
    }

    config.basis_data = &basis;

    if (!update_algebra_params(config)) {
        // error already set
        return nullptr;
    }

    return binary_function_outer<DenseAntipode>(out_obj, arg_obj, config);

    Py_RETURN_NONE;
}

PyObject *py_dense_st_fma(PyObject *, PyObject *, PyObject *) {
    Py_RETURN_NOTIMPLEMENTED;
}

PyObject *py_dense_st_inplace_mul(PyObject *, PyObject *, PyObject *) {
    Py_RETURN_NOTIMPLEMENTED;
}

PyObject *py_dense_lie_to_tensor(PyObject *, PyObject *, PyObject *) {
    Py_RETURN_NOTIMPLEMENTED;
}

PyObject *py_dense_tensor_to_lie(PyObject *, PyObject *, PyObject *) {
    Py_RETURN_NOTIMPLEMENTED;
}
