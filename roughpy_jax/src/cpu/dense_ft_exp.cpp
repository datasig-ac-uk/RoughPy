#include "dense_ft_exp.h"

#include "roughpy_compute/dense/intermediate/free_tensor_exp.hpp"

#include "xla_common.hpp"
#include "batching_loop.hpp"


using namespace rpy::compute;


namespace rpy::jax::cpu {

struct DenseFTExpStaticArgs
{
    TensorBasis basis;
    int32_t arg_max_degree;
};

template <ffi::DataType DType>
struct DenseFTExpFunctor : DenseFTExpStaticArgs
{
    using Scalar = ffi::NativeType<DType>;
    static constexpr size_t core_dims = 1;

    using StaticData = DenseFTExpStaticArgs;

    explicit DenseFTExpFunctor(DenseFTExpStaticArgs args)
        : DenseFTExpStaticArgs(std::move(args)) {}

    ffi::Error operator()(Scalar* out_data, const Scalar* arg_data)
    {
        DenseTensorView<Scalar*>
                result_view(out_data, basis, 0, arg_max_degree);
        DenseTensorView<const Scalar*>
                arg_view(arg_data, basis, 0, arg_max_degree);

        intermediate::ft_exp(result_view, arg_view);

        return ffi::Error::Success();
    }
};

ffi::Error cpu_dense_ft_exp_impl(
    ffi::Result<ffi::AnyBuffer> result,
    ffi::AnyBuffer arg,
    int32_t width,
    int32_t depth,
    ffi::Span<const int64_t> degree_begin,
    int32_t arg_max_degree
) {

    if (arg_max_degree == -1 || arg_max_degree > depth) {
        arg_max_degree = depth;
    }

    DenseFTExpStaticArgs static_args {
        TensorBasis {degree_begin.begin(), width, depth},
        arg_max_degree
    };

    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(result, static_args.basis, arg_max_degree));

    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(arg, static_args.basis, arg_max_degree));

    return select_implementation_and_go<DenseFTExpFunctor>(
        std::move(static_args),
        result->element_type(),
        result,
        arg
        );
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_exp,
    rpy::jax::cpu::cpu_dense_ft_exp_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
        .Attr<int32_t>("arg_max_deg")
);
