#include "dense_st_adj_mul.h"

#include <utility>

#include <roughpy_compute/common/basis.hpp>
#include <roughpy_compute/dense/basic/shuffle_tensor_adjoint_mul.hpp>
#include <roughpy_compute/dense/views.hpp>

#include "batching_loop.hpp"
#include "xla_common.hpp"

using namespace rpy::compute;

namespace rpy::jax::cpu {

struct DenseSTAdjMulStaticArgs {
    TensorBasis basis;
    int32_t op_max_deg;
    int32_t arg_max_deg;
};

template <ffi::DataType DType>
struct DenseSTAdjMulFunctor : DenseSTAdjMulStaticArgs {
    using Scalar = ffi::NativeType<DType>;
    static constexpr size_t core_dims = 1;

    using StaticData = DenseSTAdjMulStaticArgs;

    explicit DenseSTAdjMulFunctor(StaticData data)
        : DenseSTAdjMulStaticArgs(std::move(data))
    {}

    ffi::Error operator()(
            Scalar* out_data,
            const Scalar* op_data,
            const Scalar* arg_data
    ) noexcept
    {
        DenseTensorView<Scalar*> out_view(out_data, basis, 0, basis.depth);
        DenseTensorView<const Scalar*> op_view(op_data, basis, 0, op_max_deg);
        DenseTensorView<const Scalar*>
                arg_view(arg_data, basis, 0, arg_max_deg);

        std::fill_n(out_view.data(), out_view.size(), Scalar{});
        basic::st_adj_mul(out_view, op_view, arg_view);

        return ffi::Error::Success();
    }
};

ffi::Error cpu_dense_st_adj_mul(
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
    DenseSTAdjMulStaticArgs static_args{
            TensorBasis{degree_begin.begin(), width, depth},
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

    return select_implementation_and_go<DenseSTAdjMulFunctor>(
            std::move(static_args),
            out->element_type(),
            out,
            op,
            arg
    );
}

}// namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
        cpu_dense_st_adj_mul,
        rpy::jax::cpu::cpu_dense_st_adj_mul,
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