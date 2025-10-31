#include "roughpy_compute/dense/intermediate/free_tensor_fmexp.hpp"

#include "xla_common.hpp"

namespace {

using namespace rpy::compute;

struct ComputeTypedFtFmExp {
    const int out_max_degree;
    const int mul_max_degree;
    const int exp_max_degree;
    TensorBasis basis;

    template <typename T>
    void compute(
        T* result_data,
        const T* multiplier_data,
        const T* exponent_data
    ) {
        DenseTensorView<T*> result_view(result_data, basis, 0, out_max_degree);
        DenseTensorView<const T*> mul_view(multiplier_data, basis, 0, mul_max_degree);
        DenseTensorView<const T*> exp_view(exponent_data, basis, 0, exp_max_degree);
        intermediate::ft_fmexp(result_view, mul_view, exp_view);
    }
};

} // namespace

namespace rpy::jax::cpu {

ffi::Error cpu_dense_ft_fmexp_impl(
    int width,
    int depth,
    int out_depth,
    int mul_depth,
    int exp_depth,
    ffi::Span<const int64_t> degree_begin,
    ffi::AnyBuffer multiplier,
    ffi::AnyBuffer exponent,
    ffi::Result<ffi::AnyBuffer> result
) {
    if (!all_buffers_valid_type(multiplier, exponent, *result)) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_exp all buffers types must match and be F32 or F64");
    }

    if (degree_begin.size() != static_cast<size_t>(depth + 2)) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp degree_begin size must be depth + 2");
    }

    auto [mul_size, mul_dim] = get_buffer_dims(multiplier);
    if (mul_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp multiplier must be an array");
    }

    auto [exp_size, exp_dim] = get_buffer_dims(exponent);
    if (exp_dim == 0) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp exp must be an array");
    }

    // FIXME confirm if multiplier is correct size equivalent for result
    auto [result_size, result_dim] = get_buffer_dims(*result);
    if (result_dim != mul_dim) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp result dimension must match exponent array");
    }

    if (result_size != mul_size) {
        return ffi::Error::InvalidArgument("cpu_dense_ft_fmexp result size must match exponent array");
    }

    // Dispatch fmexp for appropriate type
    ComputeTypedFtFmExp ft_fmexp = {
        default_max_degree(out_depth, depth),
        default_max_degree(mul_depth, depth),
        default_max_degree(exp_depth, depth),
        { degree_begin.begin(), width, depth }
    };

    switch (result->element_type()) {
    case ffi::DataType::F32:
        ft_fmexp.compute(
            result->typed_data<float>(),
            multiplier.typed_data<float>(),
            exponent.typed_data<float>()
        );
        break;
    case ffi::DataType::F64:
        ft_fmexp.compute(
            result->typed_data<double>(),
            multiplier.typed_data<double>(),
            exponent.typed_data<double>()
        );
        break;
    default:
        assert(false); // Unsupported type; reject with all_buffers_valid_type
    }

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
        .Attr<xla::ffi::Span<const int64_t>>("degree_begin")
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
);
