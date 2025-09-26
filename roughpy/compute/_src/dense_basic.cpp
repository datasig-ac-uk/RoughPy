#include "dense_basic.h"


#include <roughpy_compute/common/cache_array.hpp>
#include <roughpy_compute/dense/views.hpp>

#include <roughpy_compute/dense/basic/free_tensor_antipode.hpp>
#include <roughpy_compute/dense/basic/free_tensor_adjoint_left_mul.hpp>
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
#include "tensor_basis.h"


using namespace rpy::compute;


/*******************************************************************************
 * Free tensor FMA
 ******************************************************************************/
namespace {
template <typename Scalar_>
struct DenseFTFma
{
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 3;
    static constexpr npy_intp arg_basis_mapping[3] = {0, 0, 0};

    CallConfig const* config_;

    explicit DenseFTFma(CallConfig const& config)
        : config_(&config) {}

    template <typename OutIter, typename LhsIter, typename RhsIter>
    void operator()(OutIter out_iter,
                    LhsIter lhs_iter,
                    RhsIter rhs_iter) const
    {
        auto const* basis = static_cast<TensorBasis const*>(config_->
            basis_data[0]);

        DenseTensorView<OutIter> out_view(
            out_iter,
            *basis,
            config_->degree_bounds[0].min_degree,
            config_->degree_bounds[0].max_degree
        );

        DenseTensorView<LhsIter> lhs_view(
            lhs_iter,
            *basis,
            config_->degree_bounds[1].min_degree,
            config_->degree_bounds[1].max_degree
        );

        DenseTensorView<RhsIter> rhs_view(
            rhs_iter,
            *basis,
            config_->degree_bounds[2].min_degree,
            config_->degree_bounds[2].max_degree
        );

        basic::ft_fma(out_view, lhs_view, rhs_view);
    }
};

}// namespace


PyObject* py_dense_ft_fma(PyObject* self [[maybe_unused]],
                          PyObject* args,
                          PyObject* kwargs)
{

    static constexpr char const* const kwords[] = {
            "out", "lhs", "rhs", "basis", "out_depth", "lhs_depth", "rhs_depth",
            nullptr
    };

    PyObject *out_obj, *lhs_obj, *rhs_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 3> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOOO|iii",
                                     kwords,
                                     &out_obj,
                                     &lhs_obj,
                                     &rhs_obj,
                                     &basis_obj,
                                     &degree_bounds[0].max_degree,
                                     &degree_bounds[1].max_degree,
                                     &degree_bounds[2].max_degree
    )) { return nullptr; }

    // if (!update_depth_params(config)) {
    //     PyErr_SetString(PyExc_ValueError, "incompatible depth parameters");
    //     return nullptr;
    // }

    const BasisBase* basis_data[1];

    TensorBasis basis;
    auto const degree_begins_handle = to_basis(basis_obj, basis);

    if (!degree_begins_handle) {
        // Error already set
        return nullptr;
    }
    basis_data[0] = &basis;

    CallConfig config{
            degree_bounds.data(),
            basis_data,
            nullptr,
    };

    return ternary_function_outer<
        DenseFTFma>(out_obj, lhs_obj, rhs_obj, config);
}


/*******************************************************************************
 * Free tensor Inplace multiply
 ******************************************************************************/
namespace {

template <typename Scalar_>
struct DenseFTInplaceMul
{
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 2;
    static constexpr npy_intp arg_basis_mapping[2] = {0, 0};

    CallConfig const* config_;

    explicit DenseFTInplaceMul(CallConfig const& config)
        : config_(&config) {}

    template <typename OutIter, typename RhsIter>
    void operator()(OutIter out_iter,
                    RhsIter rhs_iter) const
    {
        auto const* basis = static_cast<TensorBasis const*>(config_->
            basis_data[0]);

        DenseTensorView<OutIter> out_view(
            out_iter,
            *basis,
            config_->degree_bounds[0].min_degree,
            config_->degree_bounds[0].max_degree
        );

        DenseTensorView<RhsIter> rhs_view(
            rhs_iter,
            *basis,
            config_->degree_bounds[1].min_degree,
            config_->degree_bounds[1].max_degree
        );

        basic::ft_inplace_mul(out_view, rhs_view);
    }
};

}// namespace


PyObject* py_dense_ft_inplace_mul(PyObject* Py_UNUSED(self),
                                  PyObject* args,
                                  PyObject* kwargs)
{
    static constexpr char const* const kwords[] = {
            "lhs", "rhs", "basis", "out_depth", "rhs_depth",
            nullptr
    };

    PyObject *out_obj, *rhs_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 2> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOO|ii",
                                     kwords,
                                     &out_obj,
                                     &rhs_obj,
                                     &basis_obj,
                                     &degree_bounds[0].max_degree,
                                     &degree_bounds[1].max_degree)) {
        return nullptr;
    }

    const BasisBase* basis_data[1];

    TensorBasis basis;
    auto handle = to_basis(basis_obj, basis);
    if (!handle) {
        // error already set
        return nullptr;
    }
    basis_data[0] = &basis;

    CallConfig config{
            degree_bounds.data(),
            basis_data,
            nullptr
    };

    return binary_function_outer<DenseFTInplaceMul>(out_obj, rhs_obj, config);
}


/*******************************************************************************
 * free tensor antipode
 ******************************************************************************/
namespace {

template <typename S>
struct DenseAntipode
{
    using Scalar = S;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 2;
    static constexpr npy_intp arg_basis_mapping[2] = {0, 0};

    CallConfig const* config_;

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter,
                    ArgIter arg_iter,
                    CallConfig const& config) const {}


    explicit constexpr DenseAntipode(CallConfig const& config)
        : config_(&config) {}

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const
    {
        auto const* basis = static_cast<TensorBasis const*>(config_->
            basis_data[0]);

        DenseTensorView<OutIter> out(out_iter,
                                     *basis,
                                     config_->degree_bounds[0].min_degree,
                                     config_->degree_bounds[0].max_degree);
        DenseTensorView<ArgIter> arg(arg_iter,
                                     *basis,
                                     config_->degree_bounds[2].min_degree,
                                     config_->degree_bounds[2].max_degree);

        basic::ft_antipode(out,
                           arg,
                           basic::BasicAntipodeConfig{},
                           basic::DefaultSigner{});
    }
};
}//namespace


PyObject* py_dense_antipode(PyObject* self [[maybe_unused]],
                            PyObject* args,
                            PyObject* kwargs)
{
    static constexpr char const* const kwords[] = {
            "out", "arg", "basis", "out_depth", "arg_depth", nullptr
    };

    PyObject *out_obj, *arg_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 2> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOO|ii",
                                     kwords,
                                     &out_obj,
                                     &arg_obj,
                                     &basis_obj,
                                     &degree_bounds[1].max_degree)) {
        return nullptr;
    }

    const BasisBase* basis_data[1];
    TensorBasis basis;
    auto handle = to_basis(basis_obj, basis);
    if (!handle) {
        // error already set
        return nullptr;
    }
    basis_data[0] = &basis;

    CallConfig config{
            degree_bounds.data(),
            basis_data,
            nullptr
    };

    return binary_function_outer<DenseAntipode>(out_obj, arg_obj, config);
}

/*******************************************************************************
 * Free tensor left multiplication adjoint
 ******************************************************************************/

namespace {

template <typename S>
struct DenseFTAdjLMul
{
    using Scalar = S;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 3;
    static constexpr npy_intp arg_basis_mapping[3] = {0, 0, 0};

    const CallConfig* config_;

    explicit DenseFTAdjLMul(CallConfig const& config)
        : config_(&config) {}

    template <typename OutIter, typename OpIter, typename ArgIter>
    void operator()(OutIter out_iter, OpIter op_iter, ArgIter arg_iter) const
    {
        auto const* basis = static_cast<TensorBasis const*>(config_->
            basis_data[0]);

        DenseTensorView<OutIter> out(
            out_iter,
            *basis,
            config_->degree_bounds[0].min_degree,
            config_->degree_bounds[0].max_degree
        );

        DenseTensorView<OpIter> op(
            op_iter,
            *basis,
            config_->degree_bounds[1].min_degree,
            config_->degree_bounds[1].max_degree);

        DenseTensorView<ArgIter> arg(
            arg_iter,
            *basis,
            config_->degree_bounds[2].min_degree,
            config_->degree_bounds[2].max_degree);

        basic::ft_adj_lmul(out, op, arg);
    }
};

}// namespace


PyObject* py_dense_ft_adj_lmul(PyObject* self [[maybe_unused]],
                               PyObject* args,
                               PyObject* kwargs)
{
    static constexpr char const* const kwords[] = {
            "out", "operator", "argument", "basis", "out_depth", "lhs_depth",
            "rhs_depth", nullptr
    };

    PyObject *out_obj, *op_obj, *arg_obj;

    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 3> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOOO|iii",
                                     kwords,
                                     &out_obj,
                                     &op_obj,
                                     &arg_obj,
                                     &basis_obj,
                                     &degree_bounds[0].max_degree,
                                     &degree_bounds[1].max_degree,
                                     &degree_bounds[2].max_degree)) {
        return nullptr;
    }

    const BasisBase* basis_data[1];

    TensorBasis basis;
    auto handle = to_basis(basis_obj, basis);
    if (!handle) { return nullptr; }
    basis_data[0] = &basis;

    CallConfig config {
        degree_bounds.data(),
        basis_data,
        nullptr
    };

    return ternary_function_outer<DenseFTAdjLMul>(
        out_obj,
        op_obj,
        arg_obj,
        config);

}

PyObject* py_dense_st_fma(PyObject*, PyObject*, PyObject*)
{
    Py_RETURN_NOTIMPLEMENTED;
}

PyObject* py_dense_st_inplace_mul(PyObject*, PyObject*, PyObject*)
{
    Py_RETURN_NOTIMPLEMENTED;
}


/*******************************************************************************
 * Lie to tensor
 ******************************************************************************/

/*
 * Lie to tensor is a bit different from other operations because it mixes the
 * basis types for the arguments and carries an additional matrix sparse matrix
 * that has to be passed down to the driver routine. This matrix can be either
 * CSC or CSR format (with CSC being the default obtained from the PyLieBasis
 * we defined). We have to support both.
 */

namespace {

template <typename S>
struct DenseLieToTensor
{
    using Scalar = S;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 2;
    static constexpr npy_intp arg_basis_mapping[2] = {1, 0};

    const CallConfig* config_;

    explicit constexpr DenseLieToTensor(const CallConfig& config)
        : config_(&config) {}

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const noexcept
    {
        const auto& lie_basis = *static_cast<const LieBasis*>(config_->
            basis_data[0]);
        const auto& tensor_basis = *static_cast<const TensorBasis*>(config_->
            basis_data[1]);

        DenseTensorView<OutIter> out(
            out_iter,
            tensor_basis,
            config_->degree_bounds[0].min_degree,
            config_->degree_bounds[0].max_degree
        );

        DenseLieView<ArgIter> arg(arg_iter,
                                  lie_basis,
                                  config_->degree_bounds[1].min_degree,
                                  config_->degree_bounds[1].max_degree);

    }
};


}// namespace

PyObject* py_dense_lie_to_tensor(PyObject* Py_UNUSED(self),
                                 PyObject* args,
                                 PyObject* kwargs)
{
    static constexpr char const* const kwords[] = {
            "out", "arg", "lie_basis", "tensor_basis", nullptr
    };

    PyObject *out_obj, *arg_obj, *l2t_matrix;
    PyObject* lie_basis_obj = nullptr;
    PyObject* tensor_basis_obj = nullptr;

    std::array<DegreeBounds, 2> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOOO|O",
                                     kwords,
                                     &out_obj,
                                     &arg_obj,
                                     &l2t_matrix,
                                     &lie_basis_obj,
                                     &tensor_basis_obj)) { return nullptr; }
    const BasisBase* basis_data[2];

    LieBasis lie_basis;
    auto lie_basis_handle = to_basis(lie_basis_obj, lie_basis);
    if (!lie_basis_handle) { return nullptr; }
    basis_data[0] = &lie_basis;

    TensorBasis tensor_basis;
    PyObjHandle tensor_basis_handle;
    if (tensor_basis_obj != nullptr) {
        tensor_basis_handle = to_basis(tensor_basis_obj, tensor_basis);
        if (tensor_basis.width != lie_basis.width) {
            PyErr_SetString(PyExc_ValueError,
                            "mismatched width for Lie and tensor bases");
            return nullptr;
        }
    } else {
        auto* new_tb = PyTensorBasis_get(lie_basis.width, lie_basis.depth);
        if (new_tb == nullptr) { return nullptr; }

        tensor_basis_handle.reset(reinterpret_cast<PyObject*>(new_tb));

        tensor_basis.width = lie_basis.width;
        tensor_basis.depth = lie_basis.depth;

        // the newly constructed Tensor basis is guaranteed to have a contiguous
        // degree_begin array.
        tensor_basis.degree_begin = static_cast<npy_intp*>(PyArray_DATA(
            reinterpret_cast<PyArrayObject*>(new_tb->degree_begin)));
    }

    basis_data[1] = &tensor_basis;

    CallConfig config{
            degree_bounds.data(),
            basis_data,
            nullptr
    };

    return binary_function_outer<DenseLieToTensor>(out_obj, arg_obj, config);
}

PyObject* py_dense_tensor_to_lie(PyObject*, PyObject*, PyObject*)
{
    Py_RETURN_NOTIMPLEMENTED;
}