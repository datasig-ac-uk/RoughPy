#include "dense_intermediate.h"

#include <roughpy_compute/dense/intermediate/free_tensor_exp.hpp>
#include <roughpy_compute/dense/intermediate/free_tensor_log.hpp>
#include <roughpy_compute/dense/intermediate/free_tensor_fmexp.hpp>


#include "call_config.hpp"
#include "py_obj_handle.hpp"
#include "py_binary_array_fn.hpp"
#include "py_ternary_array_fn.hpp"


using namespace rpy::compute;

/*******************************************************************************
 * free tensor exp
 ******************************************************************************/

namespace {

template <typename Scalar_>
struct DenseFTExp
{
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    CallConfig const* config_;

    explicit DenseFTExp(CallConfig const& config)
        : config_(&config) {}

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const
    {
        auto const* basis = static_cast<TensorBasis const*>(config_->
            basis_data);

        DenseTensorView<OutIter> out(out_iter,
                                     *basis,
                                     config_->out_min_degree,
                                     config_->out_max_degree);

        DenseTensorView<ArgIter> arg(arg_iter,
                                     *basis,
                                     config_->rhs_min_degree,
                                     config_->rhs_max_degree);

        intermediate::ft_exp(out, arg);
    }
};


}


PyObject* py_dense_ft_exp(PyObject* unused_self [[maybe_unused]],
                          PyObject* args,
                          PyObject* kwargs)
{
    static constexpr char const* const kwords[] = {
            "out", "arg", "basis", "out_depth", "arg_depth",
            nullptr
    };

    PyObject *out_obj, *arg_obj;
    PyObject* basis_obj = nullptr;

    CallConfig config;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOO|ii",
                                     kwords,
                                     &out_obj,
                                     &arg_obj,
                                     &basis_obj,
                                     &config.out_max_degree,
                                     &config.lhs_max_degree,
                                     &config.rhs_max_degree)) {
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

    return binary_function_outer<DenseFTExp>(out_obj, arg_obj, config);
}

/*******************************************************************************
 * free tensor fmexp
 ******************************************************************************/

namespace {

template <typename Scalar_>
struct FtFMExp
{
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    CallConfig const* config_;

    explicit FtFMExp(CallConfig const& config)
        : config_(&config) {}

    template <typename OutIter, typename AIter, typename XIter>
    void operator()(OutIter out_iter, AIter a_iter, XIter x_iter) const
    {
        auto const* basis = static_cast<TensorBasis const*>(config_->
            basis_data);

        DenseTensorView<OutIter> out(out_iter,
                                     *basis,
                                     config_->out_min_degree,
                                     config_->out_max_degree);
        DenseTensorView<AIter> a(a_iter,
                                 *basis,
                                 config_->rhs_min_degree,
                                 config_->rhs_max_degree);
        DenseTensorView<XIter> x(x_iter,
                                 *basis,
                                 config_->rhs_min_degree,
                                 config_->rhs_max_degree);

        intermediate::ft_fmexp(out, a, x);
    }
};

}

PyObject* py_dense_ft_fmexp(PyObject* unused_self [[maybe_unused]],
                            PyObject* args,
                            PyObject* kwargs)
{
    static constexpr char const* const kwords[] = {
            "out", "multiplier", "exponent",
            "basis", "out_depth", "multiplier_depth", "exponent_depth",
            nullptr
    };

    PyObject *out_obj, *multiplier_obj, *exponent_obj;
    PyObject* basis_obj = nullptr;

    CallConfig config;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OOOO|iii",
                                     kwords,
                                     &out_obj,
                                     &multiplier_obj,
                                     &exponent_obj,
                                     &basis_obj,
                                     &config.out_max_degree,
                                     &config.lhs_max_degree,
                                     &config.rhs_max_degree)) {
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

    return ternary_function_outer<FtFMExp>(out_obj,
                                           multiplier_obj,
                                           exponent_obj,
                                           config);
}

/*******************************************************************************
 * free tensor log
 ******************************************************************************/

namespace {

template <typename Scalar_>
struct FTLog
{
    using Scalar = Scalar_;
    static constexpr npy_intp CoreDims = 1;

    CallConfig const* config_;

    explicit FTLog(CallConfig const& config)
        : config_(&config) {}

    template <typename OutIter, typename ArgIter>
    void operator()(OutIter out_iter, ArgIter arg_iter) const
    {
        auto const* basis = static_cast<TensorBasis const*>(config_->
            basis_data);

        DenseTensorView<OutIter> out(out_iter,
                                     *basis,
                                     config_->out_min_degree,
                                     config_->out_max_degree);

        DenseTensorView<ArgIter> arg(arg_iter,
                                     *basis,
                                     config_->rhs_min_degree,
                                     config_->rhs_max_degree);

        intermediate::ft_log(out, arg);
    }
};


}

PyObject* py_dense_ft_log(PyObject* unused_self [[maybe_unused]], PyObject* args, PyObject* kwargs)
{
    static constexpr char const* const kwords[] = {
            "out", "arg", "basis", "out_depth", "arg_depth",
            nullptr
    };

    PyObject *out_obj, *arg_obj;
    PyObject* basis_obj = nullptr;

    CallConfig config;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|ii", kwords,
                                     &out_obj, &arg_obj, &basis_obj, &config.out_max_degree,
                                     &config.rhs_max_degree)) {
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

    return binary_function_outer<FTLog>(out_obj, arg_obj, config);

}