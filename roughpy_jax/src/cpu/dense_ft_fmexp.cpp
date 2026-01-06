#include "dense_ft_fmexp.h"

#include "roughpy_compute/dense/intermediate/free_tensor_fmexp.hpp"

#include "xla_common.hpp"

#include "batching_loop.hpp"


using namespace rpy::compute;

namespace rpy::jax::cpu {

struct DenseFTFMExpStaticArgs {
    TensorBasis basis;
    int32_t out_max_degree;
    int32_t mul_max_degree;
    int32_t exp_max_degree;
    int32_t mul_min_degree;
    int32_t exp_min_degree;
};

template <ffi::DataType DType>
struct DenseFTFMExpFunctor : DenseFTFMExpStaticArgs {
    using Scalar = ffi::NativeType<DType>;
    static constexpr size_t core_dims = 1;

    using StaticData = DenseFTFMExpStaticArgs;

    explicit DenseFTFMExpFunctor(DenseFTFMExpStaticArgs args)
        : DenseFTFMExpStaticArgs(std::move(args))
    {}

    ffi::Error operator()(
            Scalar* out_data,
            const Scalar* multiplier_data,
            const Scalar* exponent_data
    )
    {
        DenseTensorView<Scalar*> result_view(out_data, basis, 0, out_max_degree);
        DenseTensorView<const Scalar*> multiplier_view(
                multiplier_data,
                basis,
                mul_min_degree,
                mul_max_degree
        );
        DenseTensorView<const Scalar*> exponent_view(
                exponent_data,
                basis,
                exp_min_degree,
                exp_max_degree
        );

        intermediate::ft_fmexp(result_view, multiplier_view, exponent_view);

        return ffi::Error::Success();
    }
};

ffi::Error cpu_dense_ft_fmexp_impl(
        ffi::Result<ffi::AnyBuffer> result,
        ffi::AnyBuffer multiplier,
        ffi::AnyBuffer exponent,
        int32_t width,
        int32_t depth,
        ffi::Span<const int64_t> degree_begin,
        int32_t out_max_deg,
        int32_t mul_max_deg,
        int32_t exp_max_deg,
        int32_t mul_min_deg,
        int32_t exp_min_deg
)
{
    DenseFTFMExpStaticArgs static_args {
        TensorBasis {degree_begin.begin(), width, depth},
        out_max_deg,
        mul_max_deg,
        exp_max_deg,
        mul_min_deg,
        exp_min_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(result, static_args.basis, out_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(multiplier, static_args.basis, mul_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(exponent, static_args.basis, exp_max_deg)
    );

    return select_implementation_and_go<DenseFTFMExpFunctor>(
        std::move(static_args),
        result->element_type(),
        result,
        multiplier,
        exponent
        );
}

}// namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
        cpu_dense_ft_fmexp,
        rpy::jax::cpu::cpu_dense_ft_fmexp_impl,
        xla::ffi::Ffi::Bind()
                .Ret<xla::ffi::AnyBuffer>()
                .Arg<xla::ffi::AnyBuffer>()
                .Arg<xla::ffi::AnyBuffer>()
                .Attr<int32_t>("width")
                .Attr<int32_t>("depth")
                .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
                .Attr<int32_t>("out_max_deg")
                .Attr<int32_t>("mul_max_deg")
                .Attr<int32_t>("exp_max_deg")
                .Attr<int32_t>("mul_min_deg")
                .Attr<int32_t>("exp_min_deg")
);
