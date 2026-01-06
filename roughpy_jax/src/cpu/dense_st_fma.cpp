#include "dense_ft_fma.h"

#include "roughpy_compute/dense/basic/shuffle_tensor_product.hpp"

#include "xla_common.hpp"
#include "batching_loop.hpp"

using namespace rpy::compute;

namespace rpy::jax::cpu {


struct DenseSTFmaStaticArgs
{
    TensorBasis basis;
    int32_t a_max_degree;
    int32_t b_max_degree;
    int32_t c_max_degree;

    int32_t b_min_degree = 0;
    int32_t c_min_degree = 0;
};


template <ffi::DataType DType>
struct DenseSTFmaFunctor : DenseSTFmaStaticArgs
{
    using Scalar = ffi::NativeType<DType>;
    static constexpr size_t core_dims = 1;

    using StaticData = DenseSTFmaStaticArgs;

    explicit DenseSTFmaFunctor(DenseSTFmaStaticArgs args)
        : DenseSTFmaStaticArgs(std::move(args))
    {}


    ffi::Error operator()(
        Scalar* a_data,
        const Scalar* b_data,
        const Scalar* c_data
        )
    {
        DenseTensorView<Scalar*> result_view(a_data, basis, 0, a_max_degree);
        DenseTensorView<const Scalar*>
                b_view(b_data, basis, b_min_degree, b_max_degree);
        DenseTensorView<const Scalar*>
                c_view(c_data, basis, c_min_degree, c_max_degree);

        basic::st_fma(result_view, b_view, c_view);

        return ffi::Error::Success();
    }


    ffi::Error operator()(
        Scalar* out_data,
        const Scalar* a_data,
        const Scalar* b_data,
        const Scalar* c_data)
    {
        std::copy_n(a_data, data_size_to_degree(basis, a_max_degree), out_data);
        return operator()(out_data, b_data, c_data);
    }
};



ffi::Error cpu_dense_st_fma_impl(
    ffi::Result<ffi::AnyBuffer> result,
    ffi::AnyBuffer a,
    ffi::AnyBuffer b,
    ffi::AnyBuffer c,
    int32_t width,
    int32_t depth,
    ffi::Span<const int64_t> degree_begin,
    int32_t a_max_deg,
    int32_t b_max_deg,
    int32_t c_max_deg,
    int32_t b_min_deg,
    int32_t c_min_deg
) noexcept {

    DenseSTFmaStaticArgs static_args {
        TensorBasis { degree_begin.begin(), width, depth },
        a_max_deg,
        b_max_deg,
        c_max_deg,
        b_min_deg,
        c_min_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(result, static_args.basis, a_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(a, static_args.basis, a_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(b, static_args.basis, b_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(c, static_args.basis, c_max_deg)
    );

    return select_implementation_and_go<DenseSTFmaFunctor>(
        std::move(static_args),
        result->element_type(),
        result,
        a,
        b,
        c
        );
}


ffi::Error cpu_dense_st_mul_impl(
    ffi::Result<ffi::AnyBuffer> result,
    ffi::AnyBuffer lhs,
    ffi::AnyBuffer rhs,
    int32_t width,
    int32_t depth,
    ffi::Span<const int64_t> degree_begin,
    int32_t out_max_deg,
    int32_t lhs_max_deg,
    int32_t rhs_max_deg,
    int32_t lhs_min_deg,
    int32_t rhs_min_deg
    ) noexcept
{
    DenseSTFmaStaticArgs static_args {
        TensorBasis { degree_begin.begin(), width, depth },
        out_max_deg,
        lhs_max_deg,
        rhs_max_deg,
        lhs_min_deg,
        rhs_min_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(result, static_args.basis, out_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(lhs, static_args.basis, lhs_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(rhs, static_args.basis, rhs_max_deg)
    );

    return select_implementation_and_go<DenseSTFmaFunctor>(
            std::move(static_args),
            result->element_type(),
            result,
            lhs,
            rhs
    );
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_st_fma,
    rpy::jax::cpu::cpu_dense_st_fma_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int>("width")
        .Attr<int>("depth")
        .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
        .Attr<int32_t>("a_max_deg")
        .Attr<int32_t>("b_max_deg")
        .Attr<int32_t>("c_max_deg")
        .Attr<int32_t>("b_min_deg")
        .Attr<int32_t>("c_min_deg")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
        cpu_dense_st_mul,
        rpy::jax::cpu::cpu_dense_st_mul_impl,
        xla::ffi::Ffi::Bind()
                .Ret<xla::ffi::AnyBuffer>()
                .Arg<xla::ffi::AnyBuffer>()
                .Arg<xla::ffi::AnyBuffer>()
                .Attr<int32_t>("width")
                .Attr<int32_t>("depth")
                .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
                .Attr<int32_t>("out_max_deg")
                .Attr<int32_t>("lhs_max_deg")
                .Attr<int32_t>("rhs_max_deg")
                .Attr<int32_t>("lhs_min_deg")
                .Attr<int32_t>("rhs_min_deg")
);
