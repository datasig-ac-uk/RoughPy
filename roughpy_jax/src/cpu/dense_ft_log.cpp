#include "dense_ft_log.h"

#include "roughpy_compute/dense/intermediate/free_tensor_log.hpp"

#include "xla_common.hpp"
#include "batching_loop.hpp"


using namespace rpy::compute;


namespace rpy::jax::cpu {

struct DenseFTLogStaticArgs
{
    TensorBasis basis;
    int32_t arg_max_degree;

};

template <ffi::DataType DType>
struct DenseFTLogFunctor : DenseFTLogStaticArgs
{
    using Scalar = ffi::NativeType<DType>;
    static constexpr size_t core_dims = 1;

    using StaticData = DenseFTLogStaticArgs;

    DenseFTLogFunctor(DenseFTLogStaticArgs args)
        : DenseFTLogStaticArgs(std::move(args)) {}


    ffi::Error operator()(Scalar* out_data, const Scalar* arg_data)
    {
        DenseTensorView<Scalar*> result_view(out_data, basis, 0, arg_max_degree);
        DenseTensorView<const Scalar*> arg_view(arg_data, basis, 0, arg_max_degree);

        std::fill_n(out_data, result_view.size(), Scalar{});

        intermediate::ft_log(result_view, arg_view);

        return ffi::Error::Success();
    }
};

ffi::Error cpu_dense_ft_log_impl(
    ffi::Result<ffi::AnyBuffer> result,
    ffi::AnyBuffer arg,
    int32_t width,
    int32_t depth,
    ffi::Span<const int64_t> degree_begin,
    int32_t arg_max_deg
) {

    DenseFTLogStaticArgs  static_args {
        TensorBasis { degree_begin.begin(), width, depth},
        arg_max_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(result, static_args.basis, arg_max_deg)
    );

    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(arg, static_args.basis, arg_max_deg)
    );

    return select_implementation_and_go<DenseFTLogFunctor>(
        std::move(static_args),
        result->element_type(),
        result,
        arg
        );

}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_log,
    rpy::jax::cpu::cpu_dense_ft_log_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
        .Attr<int32_t>("arg_max_deg")
);
