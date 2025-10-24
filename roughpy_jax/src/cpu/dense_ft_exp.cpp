#include "roughpy_compute/dense/intermediate/free_tensor_exp.hpp"

#include "xla_common.hpp"

namespace rpy::jax::cpu {

ffi::Error cpu_dense_ft_exp_impl(
    int width,
    int depth,
    int arg_depth,
    ffi::Span<const int64_t> degree_begin,
    FloatBuffer arg,
    ffi::ResultBuffer<XlaFloatType> result
) {
    using namespace rpy::compute;

    if (degree_begin.size() != static_cast<size_t>(depth + 2)) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_exp degree_begin size must be depth + 2");
    }

    auto [arg_size, arg_dim] = get_buffer_dims(arg);
    if (arg_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_exp arg must be an array");
    }

    auto [result_size, result_dim] = get_buffer_dims(*result);
    if (result_dim != arg_dim) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_exp result dimension must match out array");
    }

    if (result_size != arg_size) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_exp result size must match out array");
    }

    zero_result_buffer(arg_size, result);

    const int arg_max_degree = default_max_degree(arg_depth, depth);
    TensorBasis basis = { degree_begin.begin(), width, depth };
    DenseTensorView<RpyFloatType*> result_view(result->typed_data(), basis, 0, arg_max_degree);
    DenseTensorView<const RpyFloatType*> arg_view(arg.typed_data(), basis, 0, arg_max_degree);
    intermediate::ft_exp(result_view, arg_view);

    return ffi::Error::Success();
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_exp,
    rpy::jax::cpu::cpu_dense_ft_exp_impl,
    xla::ffi::Ffi::Bind()
        .Attr<int>("width")
        .Attr<int>("depth")
        .Attr<int>("arg_depth")
        .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Ret<rpy::jax::cpu::FloatBuffer>()
);
