#include "roughpy_compute/dense/intermediate/free_tensor_exp.hpp"

#include "xla_common.hpp"

namespace {

using namespace rpy::compute;

struct ComputeTypedFtExp {
    const int arg_max_degree;
    TensorBasis basis;

    template <typename T>
    void compute(
        T* result_data,
        const T* arg_data
    ) {
        DenseTensorView<T*> result_view(result_data, basis, 0, arg_max_degree);
        DenseTensorView<const T*> arg_view(arg_data, basis, 0, arg_max_degree);
        intermediate::ft_exp(result_view, arg_view);
    }
};

} // namespace

namespace rpy::jax::cpu {

ffi::Error cpu_dense_ft_exp_impl(
    int width,
    int depth,
    int arg_depth,
    ffi::Span<const int64_t> degree_begin,
    ffi::AnyBuffer arg,
    ffi::Result<ffi::AnyBuffer> result
) {
    if (!all_buffers_valid_type(arg, *result)) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_exp all buffers types must match and be F32 or F64");
    }

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

    // Dispatch exp for appropriate type
    ComputeTypedFtExp ft_exp = {
        default_max_degree(arg_depth, depth),
        { degree_begin.begin(), width, depth }
    };

    switch (result->element_type()) {
    case ffi::DataType::F32:
        ft_exp.compute(
            result->typed_data<float>(),
            arg.typed_data<float>()
        );
        break;
    case ffi::DataType::F64:
        ft_exp.compute(
            result->typed_data<double>(),
            arg.typed_data<double>()
        );
        break;
    default:
        assert(false); // Unsupported type; reject with all_buffers_valid_type
    }

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
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
);
