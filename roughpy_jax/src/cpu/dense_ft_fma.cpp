#include "roughpy_compute/dense/basic/free_tensor_fma.hpp"

#include "xla_common.hpp"

namespace rpy::jax::cpu {

ffi::Error cpu_dense_ft_fma_impl(
    int width,
    int depth,
    int out_depth, // FIXME review out naming; not strictly correct in JAX
    int lhs_depth,
    int rhs_depth,
    IndexBuffer degree_begin,
    FloatBuffer out,
    FloatBuffer lhs,
    FloatBuffer rhs,
    ffi::ResultBuffer<XlaFloatType> result
) {
    using namespace rpy::compute;

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

    auto [degree_begin_size, degree_begin_dim] = get_buffer_dims(degree_begin);
    if (degree_begin_size != depth + 2) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma degree_begin size must be depth + 2");
    }

    auto [result_size, result_dim] = get_buffer_dims(*result);
    if (result_dim != out_dim) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma result array must match out array");
    }

    if (result_size != out_size) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma result size must match out size");
    }

    auto [out_min_degree, out_max_degree] = default_min_max_degree(out_depth, depth);
    if (out_max_degree < out_min_degree) {
        return ffi::Error::InvalidArgument("out min degree must be less than max degree");
    }

    auto [lhs_min_degree, lhs_max_degree] = default_min_max_degree(lhs_depth, depth);
    if (lhs_max_degree < lhs_min_degree) {
        return ffi::Error::InvalidArgument("lhs min degree must be less than max degree");
    }

    auto [rhs_min_degree, rhs_max_degree] = default_min_max_degree(rhs_depth, depth);
    if (rhs_max_degree < rhs_min_degree) {
        return ffi::Error::InvalidArgument("rhs min degree must be less than max degree");
    }

    // FIXME for review: narrowing conversion on width and depth, underlying types
    auto degree_begin_i64 = copy_degree_begin_i64(degree_begin, degree_begin_size);
    TensorBasis basis = { degree_begin_i64.data(), width, depth };

    // Compute fma into result originally copied from out array
    copy_result_buffer(out, out_size, result);
    DenseTensorView<RpyFloatType*> result_view(result->typed_data(), basis, out_min_degree, out_max_degree);
    DenseTensorView<const RpyFloatType*> lhs_view(lhs.typed_data(), basis, lhs_min_degree, lhs_max_degree);
    DenseTensorView<const RpyFloatType*> rhs_view(rhs.typed_data(), basis, rhs_min_degree, rhs_max_degree);
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
        .Arg<rpy::jax::cpu::IndexBuffer>()
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Arg<rpy::jax::cpu::FloatBuffer>()
        .Ret<rpy::jax::cpu::FloatBuffer>()
);
