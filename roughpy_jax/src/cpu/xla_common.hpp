#ifndef ROUGHPY_JAX_CPU_XLA_COMMON_HPP
#define ROUGHPY_JAX_CPU_XLA_COMMON_HPP

#include <type_traits>
#include <vector>

#include "roughpy_compute/common/basis.hpp"

#include "xla_includes.hpp"

#define RPY_XLA_SUCCESS_OR_RETURN(stat)                                        \
    do {                                                                       \
        const auto status = (stat);                                            \
        if (!status.success()) { return status; }                              \
    } while (0)

namespace rpy::jax::cpu {

namespace ffi = xla::ffi;

template <typename... BufferType>
bool all_buffers_match_type(xla::ffi::DataType expected_type, const BufferType&... buffer) {
    return ((buffer.element_type() == expected_type) && ...);
}

template <typename... BufferType>
bool all_buffers_valid_type(const BufferType&... buffer) {
    if (!all_buffers_match_type(ffi::DataType::F32, buffer...)) {
        if (!all_buffers_match_type(ffi::DataType::F64, buffer...)) {
            return false;
        }
    }
    return true;
}

namespace detail {

inline ffi::Span<const int64_t> shape(ffi::AnyBuffer const& arg) noexcept
{
    return arg.dimensions();
}
inline ffi::Span<const int64_t> shape(ffi::Result<ffi::AnyBuffer>& arg) noexcept
{
    return arg->dimensions();
}
}


template <typename Buffer, typename Basis>
ffi::Error check_data_degree(Buffer& buf, const Basis& basis, int32_t degree, int dim=-1)
{
    if (degree > basis.depth) {
        return ffi::Error::InvalidArgument("degree exceeds basis depth");
    }

    const auto buf_shape = detail::shape(buf);
    if (dim < 0) {
        dim = buf_shape.size() + dim;
    }
    if (dim < 0 || static_cast<size_t>(dim) >= buf_shape.size()) {
        return ffi::Error::InvalidArgument("invalid dimension for buffer");
    }

    const auto val = buf_shape[dim];
    const auto data_size = data_size_to_degree(basis, degree);

    if (val < data_size) {
        return ffi::Error::InvalidArgument(
                "data dimension is too small for specified degree"
        );
    }

    return ffi::Error::Success();
}


// Convenience function getting XLA dims when validating python buffer size
std::pair<int64_t, int64_t> get_buffer_dims(const ffi::AnyBuffer& buffer);

// General pattern for determining max degree from python input degree
int default_max_degree(int buffer_depth, int basis_depth);

// Prepare result buffer based on output buffer. Currently necessary to allow
// JAX interface to play well with underlying compute calls.
void copy_result_buffer(
    ffi::AnyBuffer out,
    const int64_t out_size,
    ffi::Result<ffi::AnyBuffer> result
);

// Prepare result with zeros given size
void zero_result_buffer(
    const int64_t out_size,
    ffi::Result<ffi::AnyBuffer> result
);

} // namespace rpy::jax::cpu

#endif // ROUGHPY_JAX_CPU_XLA_COMMON_HPP
