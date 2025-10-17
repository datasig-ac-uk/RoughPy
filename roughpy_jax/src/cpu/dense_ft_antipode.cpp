#include "roughpy_compute/dense/basic/free_tensor_antipode.hpp"

#include "xla_common.hpp"

namespace rpy::jax::cpu {

ffi::Error cpu_dense_ft_antipode_impl(
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
        return ffi::Error::InvalidArgument("cpu_dense_ft_antipode arg must be an array");
    }

    auto [degree_begin_size, degree_begin_dim] = get_buffer_dims(degree_begin);
    if (degree_begin_size != depth + 2) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_antipode degree_begin size must be depth + 2");
    }

    auto [result_size, result_dim] = get_buffer_dims(*result);
    if (result_dim != arg_dim) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_antipode result dimension must match out array");
    }

    if (result_size != arg_size) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_antipode result size must match out array");
    }

    // FIXME for review: narrowing conversion on width and depth, underlying types
    auto degree_begin_i64 = copy_degree_begin_i64(degree_begin, degree_begin_size);
    TensorBasis basis = { degree_begin_i64.data(), width, depth };

    int arg_max_degree = default_max_degree(arg_depth, depth);
    DenseTensorView<RpyFloatType*> result_view(result->typed_data(), basis, 0, arg_max_degree);
    DenseTensorView<const RpyFloatType*> arg_view(arg.typed_data(), basis, 0, arg_max_degree);

    basic::ft_antipode(
        result_view,
        arg_view,
        basic::BasicAntipodeConfig{},
        basic::DefaultSigner{}
    );

    return ffi::Error::Success();
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_antipode,
    rpy::jax::cpu::cpu_dense_ft_antipode_impl,
    xla::ffi::Ffi::Bind()
        .Attr<int>("width")
        .Attr<int>("depth")
        .Attr<int>("arg_depth")
        .Arg<rpy::jax::cpu::IndexBuffer>()
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Ret<rpy::jax::cpu::FloatBuffer>()
);
