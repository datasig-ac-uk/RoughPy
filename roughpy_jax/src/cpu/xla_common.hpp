#ifndef ROUGHPY_JAX_CPU_XLA_COMMON_HPP
#define ROUGHPY_JAX_CPU_XLA_COMMON_HPP

#include <type_traits>
#include <vector>

#include "roughpy_compute/common/basis.hpp"

#include "xla_includes.hpp"

namespace rpy::jax::cpu {

namespace ffi = xla::ffi;

// Data type for index buffers, e.g. degree_begin passed from python
inline constexpr ffi::DataType XlaIndexType = ffi::DataType::S32; // FIXME change to 64 bit pending TensorBasis implementation
using IndexBuffer = ffi::Buffer<XlaIndexType>;

// Data type for float buffers. Currently hard-coded to 32 bit because JAX
// prefers single-precision. See JAX_ENABLE_X64 in JAX gotcha docs for more info
using RpyFloatType = float;
inline constexpr ffi::DataType XlaFloatType = ffi::DataType::F32;
static_assert(
    std::is_same_v<
        RpyFloatType,
        ffi::NativeType<XlaFloatType>
    >,
    "XlaFloatType must match underlying float type"
);
using FloatBuffer = ffi::Buffer<XlaFloatType>;

// Convenience function getting XLA dims when validating python buffer size
template <ffi::DataType T>
std::pair<int64_t, int64_t> get_buffer_dims(const ffi::Buffer<T>& buffer)
{
    auto dims = buffer.dimensions();
    if (dims.size() == 0) {
        return std::make_pair(0, 0);
    }
    return std::make_pair(buffer.element_count(), dims.back());
}

// Generate a min/max degree based on python input degree, validating against basis
inline std::pair<int, int> default_min_max_degree(int buffer_depth, int basis_depth) {
    // FIXME review/remove before merge to main. This has been worked from
    // roughpy/compute/_src/call_config.cpp (still checking min < max after
    // setting min = 0) until JAX python interface is clarified in review.
    int max_degree = buffer_depth;
    if (max_degree == -1 || max_degree >= basis_depth) {
        max_degree = basis_depth;
    }

    int min_degree = 0;
    return std::make_pair(min_degree, max_degree);
}

// Convert python degree_begin buffer (i32) into a native RoughPy buffer (i64).
// Currently required because the internal TensorBasis degree_begin array is 64
// bit ptrdiffs and we are passing 32 bit ints in python.
// FIXME ideally this method can be removed with new python TensorBasis impl
inline std::vector<rpy::compute::TensorBasis::BasisBase::Index> copy_degree_begin_i64(
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

// Prepare result buffer based on output buffer. Currently necessary to allow
// JAX interface to play well with underlying compute calls.
inline void copy_result_buffer(
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

// Prepare result with zeros given size
inline void zero_result_buffer(
    const int64_t out_size,
    ffi::ResultBuffer<XlaFloatType> result
) {
    std::fill_n(result->typed_data(), out_size, 0.0f);
}

} // namespace rpy::jax::cpu

#endif // ROUGHPY_JAX_CPU_XLA_COMMON_HPP
