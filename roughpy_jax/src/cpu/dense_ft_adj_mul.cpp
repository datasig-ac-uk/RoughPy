#include "dense_ft_adj_mul.h"

#include <utility>
#include <vector>

#include <roughpy_compute/common/basis.hpp>
#include <roughpy_compute/dense/views.hpp>
#include <roughpy_compute/dense/basic/free_tensor_adjoint_left_mul.hpp>
#include <roughpy_compute/dense/basic/free_tensor_antipode.hpp>

#include "xla_common.hpp"
#include "batching_loop.hpp"

using namespace rpy::compute;

namespace rpy::jax::cpu {


struct DenseFTAdjLMulStaticArgs
{
    TensorBasis basis;
    int32_t op_max_deg;
    int32_t arg_max_deg;
};

template <ffi::DataType DType>
struct DenseFTAdjLMulFunctor : DenseFTAdjLMulStaticArgs
{
    using Scalar = ffi::NativeType<DType>;
    static constexpr size_t core_dims = 1;

    using StaticData = DenseFTAdjLMulStaticArgs;

    explicit DenseFTAdjLMulFunctor(DenseFTAdjLMulStaticArgs args)
        : DenseFTAdjLMulStaticArgs(std::move(args))
    {}

    ffi::Error operator()(Scalar* out_data,
         const Scalar* mul_data,
         const Scalar* arg_data) noexcept
    {
        DenseTensorView<Scalar*> out_view(out_data, basis, 0, basis.depth);
        DenseTensorView<const Scalar*>
                mul_view(mul_data, basis, 0, op_max_deg);
        DenseTensorView<const Scalar*>
                arg_view(arg_data, basis, 0, arg_max_deg);

        basic::ft_adj_lmul(out_view, mul_view, arg_view);

        return ffi::Error::Success();
    }
};


ffi::Error cpu_dense_ft_adj_lmul_impl(
    ffi::Result<ffi::AnyBuffer> out,
    ffi::AnyBuffer op,
    ffi::AnyBuffer arg,
    int32_t width,
    int32_t depth,
    ffi::Span<const int64_t> degree_begin,
    int32_t op_max_deg,
    int32_t arg_max_deg
    ) noexcept
{
    DenseFTAdjLMulStaticArgs static_args {
        TensorBasis { degree_begin.begin(), width, depth },
        op_max_deg,
        arg_max_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(out, static_args.basis, static_args.basis.depth)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(op, static_args.basis, op_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(arg, static_args.basis, arg_max_deg)
    );

    return select_implementation_and_go<DenseFTAdjLMulFunctor>(
            std::move(static_args),
            out->element_type(),
            out,
            op,
            arg
    );

}



template <ffi::DataType DType>
struct DenseFTAdjRMulFunctor : DenseFTAdjLMulFunctor<DType>
{
    using Base = DenseFTAdjLMulFunctor<DType>;
    using Scalar = typename Base::Scalar;

    std::vector<Scalar> op_workspace;
    std::vector<Scalar> out_workspace;

    explicit DenseFTAdjRMulFunctor(DenseFTAdjLMulStaticArgs args)
        : Base(std::move(args))
    {
        out_workspace.resize(data_size_to_degree(this->basis, this->basis.depth));
        op_workspace.resize(data_size_to_degree(this->basis, this->op_max_deg));
    }


    ffi::Error operator()(Scalar* out, const Scalar* op_data, const Scalar* arg_data) noexcept
    {
        DenseTensorView<Scalar*>
                out_view(out, this->basis, 0, this->basis.depth);
        DenseTensorView<const Scalar*>
                op_view(op_data, this->basis, 0, this->op_max_deg);
        DenseTensorView<const Scalar*>
                arg_view(arg_data, this->basis, 0, this->arg_max_deg);

        DenseTensorView<Scalar*> op_workspace_view(
                op_workspace.data(),
                this->basis,
                0,
                this->op_max_deg
        );

        DenseTensorView<Scalar*> out_workspace_view(
                out_workspace.data(),
                this->basis,
                0,
                this->basis.depth
        );

        basic::ft_antipode(
                op_workspace_view,
                op_view,
                basic::BasicAntipodeConfig{},
                basic::DefaultSigner{}
        );

        RPY_XLA_SUCCESS_OR_RETURN(
                Base::
                operator()(out_workspace.data(), op_workspace.data(), arg_data)
        );

        basic::ft_antipode(
                out_view,
                out_workspace_view,
                basic::BasicAntipodeConfig{},
                basic::DefaultSigner{}
        );

        return ffi::Error::Success();
    }

};



ffi::Error cpu_dense_ft_adj_rmul_impl(
    ffi::Result<ffi::AnyBuffer> out,
    ffi::AnyBuffer op,
    ffi::AnyBuffer arg,
    int32_t width,
    int32_t depth,
    ffi::Span<const int64_t> degree_begin,
    int32_t op_max_deg,
    int32_t arg_max_deg
    ) noexcept
{
    DenseFTAdjLMulStaticArgs static_args {
        TensorBasis { degree_begin.begin(), width, depth },
        op_max_deg,
        arg_max_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(out, static_args.basis, static_args.basis.depth)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(op, static_args.basis, op_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
            check_data_degree(arg, static_args.basis, arg_max_deg)
    );

    return select_implementation_and_go<DenseFTAdjRMulFunctor>(
            std::move(static_args),
            out->element_type(),
            out,
            op,
            arg
    );

}

} // namespace rpy::jax::cpu



XLA_FFI_DEFINE_HANDLER_SYMBOL(cpu_dense_ft_adj_lmul,
    rpy::jax::cpu::cpu_dense_ft_adj_lmul_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
        .Attr<int32_t>("op_max_deg")
        .Attr<int32_t>("arg_max_deg")
);


XLA_FFI_DEFINE_HANDLER_SYMBOL(cpu_dense_ft_adj_rmul,
    rpy::jax::cpu::cpu_dense_ft_adj_lmul_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
        .Attr<int32_t>("op_max_deg")
        .Attr<int32_t>("arg_max_deg")
);
