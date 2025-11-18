#include "dense_intermediate.h"

#include <array>

#include <roughpy_compute/dense/intermediate/free_tensor_exp.hpp>
#include <roughpy_compute/dense/intermediate/free_tensor_fmexp.hpp>
#include <roughpy_compute/dense/intermediate/free_tensor_log.hpp>

#include "py_obj_handle.hpp"
#include "call_config.hpp"
#include "py_binary_array_fn.hpp"
#include "py_compat.h"
#include "py_ternary_array_fn.hpp"
#include "call_utils.hpp"

using namespace rpy::compute;

/*******************************************************************************
 * free tensor exp
 ******************************************************************************/

namespace {

template <typename Scalar_>
struct DenseFTExp : ComputeCallFunctor<1, 0, 0> {
    using Scalar = Scalar_;
    using ComputeCallFunctor::ComputeCallFunctor;

    template <typename Ctx,typename OutIter, typename ArgIter>
    int operator()(const Ctx& ctx, OutIter out_iter, ArgIter arg_iter) const
    {
        auto out = make_tensor_view(0, std::move(out_iter));
        auto arg = make_tensor_view(1, std::move(arg_iter));

        return RPY_CATCH_ERRORS(
                intermediate::ft_exp(ctx, out, arg)
        );
    }
};

}// namespace

PyObject* py_dense_ft_exp(
        PyObject* unused_self [[maybe_unused]],
        PyObject* args,
        PyObject* kwargs
)
{
    static constexpr char const* const kwords[]
            = {"out", "arg", "basis", "out_depth", "arg_depth", nullptr};

    PyObject *out_obj, *arg_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 2> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "OOO|ii",
                RPC_PY_KWORD_CAST(kwords),
                &out_obj,
                &arg_obj,
                &basis_obj,
                &degree_bounds[0].max_degree,
                &degree_bounds[1].max_degree
        )) {
        return nullptr;
    }

    BasisBase const* basis_data[1];
    TensorBasis basis;
    auto handle = to_basis(basis_obj, basis);
    if (!handle) {
        // error already set
        return nullptr;
    }
    basis_data[0] = &basis;

    CallConfig config{degree_bounds.data(), basis_data, nullptr};

    return binary_function_outer<DenseFTExp>(out_obj, arg_obj, config);
}

/*******************************************************************************
 * free tensor fmexp
 ******************************************************************************/

namespace {

template <typename Scalar_>
struct FtFMExp : ComputeCallFunctor<1, 0, 0, 0>{
    using Scalar = Scalar_;
    using ComputeCallFunctor::ComputeCallFunctor;

    template <typename Ctx, typename OutIter, typename AIter, typename XIter>
    int operator()(const Ctx& ctx, OutIter out_iter, AIter a_iter, XIter x_iter) const
    {
        auto out = make_tensor_view(0, std::move(out_iter));
        auto a = make_tensor_view(1, std::move(a_iter));
        auto x = make_tensor_view(2, std::move(x_iter));

        return RPY_CATCH_ERRORS(
                intermediate::ft_fmexp(ctx, out, a, x)
        );
    }
};

}// namespace

PyObject* py_dense_ft_fmexp(
        PyObject* unused_self [[maybe_unused]],
        PyObject* args,
        PyObject* kwargs
)
{
    static constexpr char const* const kwords[]
            = {"out",
               "multiplier",
               "exponent",
               "basis",
               "out_depth",
               "multiplier_depth",
               "exponent_depth",
               nullptr};

    PyObject *out_obj, *multiplier_obj, *exponent_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 3> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "OOOO|iii",
                RPC_PY_KWORD_CAST(kwords),
                &out_obj,
                &multiplier_obj,
                &exponent_obj,
                &basis_obj,
                &degree_bounds[0].max_degree,
                &degree_bounds[1].max_degree,
                &degree_bounds[2].max_degree
        )) {
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

    CallConfig config{degree_bounds.data(), basis_data, nullptr};

    return ternary_function_outer<FtFMExp>(
            out_obj,
            multiplier_obj,
            exponent_obj,
            config
    );
}

/*******************************************************************************
 * free tensor log
 ******************************************************************************/

namespace {

template <typename Scalar_>
struct FTLog : ComputeCallFunctor<1, 0, 0>{
    using Scalar = Scalar_;
    using ComputeCallFunctor::ComputeCallFunctor;

    template <typename Ctx, typename OutIter, typename ArgIter>
    int operator()(const Ctx& ctx, OutIter out_iter, ArgIter arg_iter) const
    {
        auto out = make_tensor_view(0, std::move(out_iter));
        auto arg = make_tensor_view(1, std::move(arg_iter));

        return RPY_CATCH_ERRORS(
                intermediate::ft_log(ctx, out, arg)
        );
    }
};

}// namespace

PyObject* py_dense_ft_log(
        PyObject* unused_self [[maybe_unused]],
        PyObject* args,
        PyObject* kwargs
)
{
    static constexpr char const* const kwords[]
            = {"out", "arg", "basis", "out_depth", "arg_depth", nullptr};

    PyObject *out_obj, *arg_obj;
    PyObject* basis_obj = nullptr;

    std::array<DegreeBounds, 2> degree_bounds;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "OOO|ii",
                RPC_PY_KWORD_CAST(kwords),
                &out_obj,
                &arg_obj,
                &basis_obj,
                &degree_bounds[0].max_degree,
                &degree_bounds[1].max_degree
        )) {
        return nullptr;
    }

    BasisBase const* basis_data[1];

    TensorBasis basis;
    auto handle = to_basis(basis_obj, basis);
    if (!handle) {
        // error already set
        return nullptr;
    }
    basis_data[0] = &basis;

    CallConfig config{degree_bounds.data(), basis_data, nullptr};

    return binary_function_outer<FTLog>(out_obj, arg_obj, config);
}