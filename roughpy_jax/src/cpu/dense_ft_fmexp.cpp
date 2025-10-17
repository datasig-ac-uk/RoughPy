#include "roughpy_compute/dense/intermediate/free_tensor_fmexp.hpp"

#include "xla_common.hpp"

namespace rpy::jax::cpu {

ffi::Error cpu_dense_ft_fmexp_impl(
    int width,
    int depth,
    int out_depth,
    int mul_depth,
    int exp_depth,
    IndexBuffer degree_begin,
    FloatBuffer multiplier,
    FloatBuffer exponent,
    ffi::ResultBuffer<XlaFloatType> result
) {
    using namespace rpy::compute;

    auto [mul_size, mul_dim] = get_buffer_dims(multiplier);
    if (mul_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp multiplier must be an array");
    }

    auto [exp_size, exp_dim] = get_buffer_dims(exponent);
    if (exp_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp exp must be an array");
    }

    auto [degree_begin_size, degree_begin_dim] = get_buffer_dims(degree_begin);
    if (degree_begin_size != depth + 2) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp degree_begin size must be depth + 2");
    }

    // FIXME confirm if multiplier is correct size equivalent for result
    auto [result_size, result_dim] = get_buffer_dims(*result);
    if (result_dim != mul_dim) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp result dimension must match exponent array");
    }

    if (result_size != mul_size) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp result size must match exponent array");
    }

    // FIXME for review: narrowing conversion on width and depth, underlying types
    auto degree_begin_i64 = copy_degree_begin_i64(degree_begin, degree_begin_size);
    TensorBasis basis = { degree_begin_i64.data(), width, depth };

    int out_max_degree = default_max_degree(out_depth, depth);
    int mul_max_degree = default_max_degree(mul_depth, depth);
    int exp_max_degree = default_max_degree(exp_depth, depth);
    DenseTensorView<RpyFloatType*> result_view(result->typed_data(), basis, 0, out_max_degree);
    DenseTensorView<const RpyFloatType*> mul_view(multiplier.typed_data(), basis, 0, mul_max_degree);
    DenseTensorView<const RpyFloatType*> exp_view(exponent.typed_data(), basis, 0, exp_max_degree);
    intermediate::ft_fmexp(result_view, mul_view, exp_view);

    return ffi::Error::Success();
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_fmexp,
    rpy::jax::cpu::cpu_dense_ft_fmexp_impl,
    xla::ffi::Ffi::Bind()
        .Attr<int>("width")
        .Attr<int>("depth")
        .Attr<int>("out_depth")
        .Attr<int>("mul_depth")
        .Attr<int>("exp_depth")
        .Arg<rpy::jax::cpu::IndexBuffer>()
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Ret<rpy::jax::cpu::FloatBuffer>()
);
