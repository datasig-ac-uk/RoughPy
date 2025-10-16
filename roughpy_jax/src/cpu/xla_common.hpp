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

// General pattern for determining max degree from python input degree
int default_max_degree(int buffer_depth, int basis_depth);

// Convert python degree_begin buffer (i32) into a native RoughPy buffer (i64).
// Currently required because the internal TensorBasis degree_begin array is 64
// bit ptrdiffs and we are passing 32 bit ints in python.
// FIXME ideally this method can be removed with new python TensorBasis impl
std::vector<rpy::compute::TensorBasis::BasisBase::Index> copy_degree_begin_i64(
    const IndexBuffer& degree_begin,
    const int64_t degree_begin_size
);

// Prepare result buffer based on output buffer. Currently necessary to allow
// JAX interface to play well with underlying compute calls.
void copy_result_buffer(
    FloatBuffer out,
    const int64_t out_size,
    ffi::ResultBuffer<XlaFloatType> result
);

// Prepare result with zeros given size
void zero_result_buffer(
    const int64_t out_size,
    ffi::ResultBuffer<XlaFloatType> result
);

} // namespace rpy::jax::cpu

#endif // ROUGHPY_JAX_CPU_XLA_COMMON_HPP
