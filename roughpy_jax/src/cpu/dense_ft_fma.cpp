#include "roughpy_compute/dense/basic/free_tensor_fma.hpp"

#include "xla_common.hpp"

namespace {

using namespace rpy::compute;

struct ComputeTypedFtFma {
    const int out_max_degree;
    const int lhs_max_degree;
    const int rhs_max_degree;
    TensorBasis basis;

    template <typename T>
    void compute(
        T* result_data,
        const T* lhs_data,
        const T* rhs_data
    ) {
        DenseTensorView<T*> result_view(result_data, basis, 0, out_max_degree);
        DenseTensorView<const T*> lhs_view(lhs_data, basis, 0, lhs_max_degree);
        DenseTensorView<const T*> rhs_view(rhs_data, basis, 0, rhs_max_degree);
        basic::ft_fma(result_view, lhs_view, rhs_view);
    }
};

} // namespace

namespace rpy::jax::cpu {

ffi::Error cpu_dense_ft_fma_impl(
    int width,
    int depth,
    int out_depth,
    int lhs_depth,
    int rhs_depth,
    ffi::Span<const int64_t> degree_begin,
    ffi::AnyBuffer out,
    ffi::AnyBuffer lhs,
    ffi::AnyBuffer rhs,
    ffi::Result<ffi::AnyBuffer> result
) {
    if (!all_buffers_valid_type(out, lhs, rhs, *result)) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fma all buffers types must match and be F32 or F64");
    }

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

    // Dispatch fma for appropriate type
    ComputeTypedFtFma ft_fma = {
        default_max_degree(out_depth, depth),
        default_max_degree(lhs_depth, depth),
        default_max_degree(rhs_depth, depth),
        { degree_begin.begin(), width, depth }
    };

    switch (result->element_type()) {
    case ffi::DataType::F32:
        ft_fma.compute(
            result->typed_data<float>(),
            lhs.typed_data<float>(),
            rhs.typed_data<float>()
        );
        break;
    case ffi::DataType::F64:
        ft_fma.compute(
            result->typed_data<double>(),
            lhs.typed_data<double>(),
            rhs.typed_data<double>()
        );
        break;
    default:
        assert(false); // Unsupported type; reject with all_buffers_valid_type
    }

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
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
);
