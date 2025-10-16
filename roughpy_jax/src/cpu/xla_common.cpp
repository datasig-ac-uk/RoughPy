#include "xla_common.hpp"

namespace rpy::jax::cpu {

int default_max_degree(int buffer_depth, int basis_depth) {
    int max_degree = buffer_depth;
    if (max_degree == -1 || max_degree >= basis_depth) {
        max_degree = basis_depth;
    }

    return max_degree;
}

std::vector<rpy::compute::TensorBasis::BasisBase::Index> copy_degree_begin_i64(
    const IndexBuffer& degree_begin,
    const int64_t degree_begin_size
) {
    std::vector<rpy::compute::TensorBasis::BasisBase::Index> degree_begin_i64(degree_begin_size);
    std::copy(
        degree_begin.typed_data(),
        degree_begin.typed_data() + degree_begin_size,
        degree_begin_i64.begin()
    );

    return degree_begin_i64;
}

void copy_result_buffer(
    FloatBuffer out,
    const int64_t out_size,
    ffi::ResultBuffer<XlaFloatType> result
) {
    // FIXME for review: JAX arrays are immutable but we are currently following
    // convention of RoughPy compute's function calls, which are ternary
    // mutating the first (out) argument. The short-term fix is to instead have
    // quaternary approach where out is copied into result and then reused.
    std::copy_n(out.typed_data(), out_size, result->typed_data());
}

void zero_result_buffer(
    const int64_t out_size,
    ffi::ResultBuffer<XlaFloatType> result
) {
    std::fill_n(result->typed_data(), out_size, 0.0f);
}

} // namespace rpy::jax::cpu
