#include "dense_ft_antipode.h"

#include "roughpy_compute/dense/basic/free_tensor_antipode.hpp"

#include "xla_common.hpp"
#include "batching_loop.hpp"


using namespace rpy::compute;

namespace rpy::jax::cpu {

struct DenseFTAntipodeStaticArgs
{
    TensorBasis basis;
    int32_t max_degree;
    bool no_sign;
};

template <ffi::DataType DType>
struct DenseFTAntipode : DenseFTAntipodeStaticArgs
{
    using Scalar = ffi::NativeType<DType>;
    static constexpr size_t core_dims = 1;
    using StaticData = DenseFTAntipodeStaticArgs;

    explicit DenseFTAntipode(StaticData arg)
        : DenseFTAntipodeStaticArgs(std::move(arg)) {}


    ffi::Error operator()(Scalar* out_data, const Scalar* arg_data) noexcept
    {
        DenseTensorView<Scalar*> result_view(out_data, basis, 0, max_degree);
        DenseTensorView<const Scalar*> arg_view(arg_data, basis, 0, max_degree);

        if (no_sign) {
            basic::ft_antipode(
                result_view,
                arg_view,
                basic::BasicAntipodeConfig{},
                basic::DefaultSigner{}
            );
        } else {
            basic::ft_antipode(
                result_view,
                arg_view,
                basic::BasicAntipodeConfig{},
                basic::DefaultSigner{}
            );
        }

        return ffi::Error::Success();
    }
};



ffi::Error cpu_dense_ft_antipode_impl(
    ffi::Result<ffi::AnyBuffer> result,
    ffi::AnyBuffer arg,
    int32_t width,
    int32_t depth,
    ffi::Span<const int64_t> degree_begin,
    int32_t max_deg,
    bool no_sign
) {


    DenseFTAntipodeStaticArgs static_args {
        TensorBasis{ degree_begin.begin(), width, depth},
        max_deg,
        no_sign
    };

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(result, static_args.basis, max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(arg, static_args.basis, max_deg));

    return select_implementation_and_go<DenseFTAntipode>(
        std::move(static_args),
        result->element_type(),
        result,
        arg
        );
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_antipode,
    rpy::jax::cpu::cpu_dense_ft_antipode_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int>("width")
        .Attr<int>("depth")
        .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
        .Attr<int>("max_degree")
        .Attr<bool>("no_sign")
);
