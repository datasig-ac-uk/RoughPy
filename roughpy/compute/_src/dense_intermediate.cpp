#include "dense_intermediate.h"

#include <array>

#include <roughpy_compute/dense/intermediate/free_tensor_exp.hpp>
#include <roughpy_compute/dense/intermediate/free_tensor_fmexp.hpp>
#include <roughpy_compute/dense/intermediate/free_tensor_log.hpp>

#include <roughpy/pycore/compat.h>
#include <roughpy/pycore/object_handle.hpp>

#include "call_config.hpp"
#include "py_binary_array_fn.hpp"
#include "py_ternary_array_fn.hpp"

using namespace rpy::compute;

/*******************************************************************************
 * free tensor exp
 ******************************************************************************/

namespace {

template <typename Scalar_>
struct DenseFTExp {
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 2;
    static constexpr npy_intp arg_basis_mapping[2] = {0, 0};

    CallConfig const* config_;

    explicit DenseFTExp(CallConfig const& config) : config_(&config) {}

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const
    {
        auto const* basis
                = static_cast<TensorBasis const*>(config_->basis_data[0]);

        DenseTensorView<OutIter> out(
                out_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );

        DenseTensorView<ArgIter> arg(
                arg_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );

        intermediate::ft_exp(out, arg);
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
                RPY_PY_KWORD_CAST(kwords),
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
struct FtFMExp {
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 3;
    static constexpr npy_intp arg_basis_mapping[3] = {0, 0, 0};

    CallConfig const* config_;

    explicit FtFMExp(CallConfig const& config) : config_(&config) {}

    template <typename OutIter, typename AIter, typename XIter>
    void operator()(OutIter out_iter, AIter a_iter, XIter x_iter) const
    {
        auto const* basis
                = static_cast<TensorBasis const*>(config_->basis_data[0]);

        DenseTensorView<OutIter> out(
                out_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );
        DenseTensorView<AIter> a(
                a_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );
        DenseTensorView<XIter> x(
                x_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );

        intermediate::ft_fmexp(out, a, x);
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
                RPY_PY_KWORD_CAST(kwords),
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
struct FTLog {
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    static constexpr npy_intp n_args = 2;
    static constexpr npy_intp arg_basis_mapping[2] = {0, 0};

    CallConfig const* config_;

    explicit FTLog(CallConfig const& config) : config_(&config) {}

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const
    {
        auto const* basis
                = static_cast<TensorBasis const*>(config_->basis_data[0]);

        DenseTensorView<OutIter> out(
                out_iter,
                *basis,
                config_->degree_bounds[0].min_degree,
                config_->degree_bounds[0].max_degree
        );

        DenseTensorView<ArgIter> arg(
                arg_iter,
                *basis,
                config_->degree_bounds[1].min_degree,
                config_->degree_bounds[1].max_degree
        );

        intermediate::ft_log(out, arg);
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
                RPY_PY_KWORD_CAST(kwords),
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