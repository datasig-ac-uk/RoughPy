#include "roughpy_compute/dense/basic/free_tensor_fma.hpp"

#include "xla_common.hpp"

namespace rpy::jax::cpu {

ffi::Error cpu_dense_ft_fma_impl(
    int width,
    int depth,
    int out_depth,
    int lhs_depth,
    int rhs_depth,
    ffi::Span<const int64_t> degree_begin,
    FloatBuffer out,
    FloatBuffer lhs,
    FloatBuffer rhs,
    ffi::ResultBuffer<XlaFloatType> result
) {
    using namespace rpy::compute;

    if (degree_begin.size() != static_cast<size_t>(depth + 2)) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin size must be depth + 2");
    }

    auto [out_size, out_dim] = get_buffer_dims(out);
    if (out_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma out must be an array");
    }

    auto [lhs_size, lhs_dim] = get_buffer_dims(lhs);
    if (lhs_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma lhs must be an array");
    }

    auto [rhs_size, rhs_dim] = get_buffer_dims(rhs);
    if (rhs_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma rhs must be an array");
    }

    auto [result_size, result_dim] = get_buffer_dims(*result);
    if (result_dim != out_dim) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma result dimension must match out array");
    }

    if (result_size != out_size) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma result size must match out array");
    }

    // Compute fma into result originally copied from out array
    copy_result_buffer(out, out_size, result);

    const int out_max_degree = default_max_degree(out_depth, depth);
    const int lhs_max_degree = default_max_degree(lhs_depth, depth);
    const int rhs_max_degree = default_max_degree(rhs_depth, depth);
    TensorBasis basis = { degree_begin.begin(), width, depth };
    DenseTensorView<RpyFloatType*> result_view(result->typed_data(), basis, 0, out_max_degree);
    DenseTensorView<const RpyFloatType*> lhs_view(lhs.typed_data(), basis, 0, lhs_max_degree);
    DenseTensorView<const RpyFloatType*> rhs_view(rhs.typed_data(), basis, 0, rhs_max_degree);
    basic::ft_fma(result_view, lhs_view, rhs_view);

    return ffi::Error::Success();
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_fma,
    rpy::jax::cpu::cpu_dense_ft_fma_impl,
    xla::ffi::Ffi::Bind()
        .Attr<int>("width")
        .Attr<int>("depth")
        .Attr<int>("out_depth")
        .Attr<int>("lhs_depth")
        .Attr<int>("rhs_depth")
        .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Ret<rpy::jax::cpu::FloatBuffer>()
);
