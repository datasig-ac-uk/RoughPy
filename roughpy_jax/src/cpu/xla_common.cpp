#include "xla_common.hpp"

namespace rpy::jax::cpu {

std::pair<int64_t, int64_t> get_buffer_dims(const ffi::AnyBuffer& buffer) {
    auto dims = buffer.dimensions();
    if (dims.size() == 0) {
        return std::make_pair(0, 0);
    }
    return std::make_pair(buffer.element_count(), dims.back());
}

int default_max_degree(int buffer_depth, int basis_depth) {
    int max_degree = buffer_depth;
    if (max_degree == -1 || max_degree >= basis_depth) {
        max_degree = basis_depth;
    }
    return max_degree;
}

void copy_result_buffer(
    ffi::AnyBuffer out,
    const int64_t out_element_count,
    ffi::Result<ffi::AnyBuffer> result
) {
    // The RoughPy JAX API for CPU uses the RoughPy compute internal which works
    // by mutating the output array. However, JAX arrays are immutable so out is
    // copied to a result buffer for the computation and return result.
    switch (result->element_type()) {
    case ffi::DataType::F32:
        std::copy_n(out.typed_data<float>(), out_element_count, result->typed_data<float>());
        break;
    case ffi::DataType::F64:
        std::copy_n(out.typed_data<double>(), out_element_count, result->typed_data<double>());
        break;
    default:
        assert(false); // Unsupported type; reject with all_buffers_valid_type
    }
}

void zero_result_buffer(
    const int64_t out_size,
    ffi::Result<ffi::AnyBuffer> result
) {
    switch (result->element_type()) {
    case ffi::DataType::F32:
        std::fill_n(result->typed_data<float>(), out_size, 0.0f);
        break;
    case ffi::DataType::F64:
        std::fill_n(result->typed_data<double>(), out_size, 0.0);
        break;
    default:
        assert(false); // Unsupported type; reject with all_buffers_valid_type
    }
}

} // namespace rpy::jax::cpu
