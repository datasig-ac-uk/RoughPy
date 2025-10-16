#include "roughpy_compute/dense/intermediate/free_tensor_exp.hpp"

#include "xla_common.hpp"

namespace rpy::jax::cpu {

ffi::Error cpu_dense_ft_exp_impl(
    int width,
    int depth,
    int arg_depth,
    IndexBuffer degree_begin,
    FloatBuffer arg,
    ffi::ResultBuffer<XlaFloatType> result
) {
    using namespace rpy::compute;

    auto [arg_size, arg_dim] = get_buffer_dims(arg);
    if (arg_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_exp arg must be an array");
    }

    auto [degree_begin_size, degree_begin_dim] = get_buffer_dims(degree_begin);
    if (degree_begin_size != depth + 2) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin size must be depth + 2");
    }

    auto [result_size, result_dim] = get_buffer_dims(*result);
    if (result_dim != arg_dim) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_exp result array must match out array");
    }

    if (result_size != arg_size) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_exp result size must match out size");
    }

    auto [arg_min_degree, arg_max_degree] = default_min_max_degree(arg_depth, depth);
    if (arg_max_degree < arg_min_degree) {
        return ffi::Error::InvalidArgument("arg min degree must be less than max degree");
    }

    // FIXME for review: narrowing conversion on width and depth, underlying types
    auto degree_begin_i64 = copy_degree_begin_i64(degree_begin, degree_begin_size);
    TensorBasis basis = { degree_begin_i64.data(), width, depth };

    zero_result_buffer(arg_size, result);
    DenseTensorView<RpyFloatType*> result_view(result->typed_data(), basis, arg_min_degree, arg_max_degree);
    DenseTensorView<const RpyFloatType*> arg_view(arg.typed_data(), basis, arg_min_degree, arg_max_degree);
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
        .Arg<rpy::jax::cpu::IndexBuffer>()
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Ret<rpy::jax::cpu::FloatBuffer>()
);
