//
// Created by sam on 6/12/24.
//

#ifndef DENSE_FREE_TENSOR_MULTIPLICATION_H
#define DENSE_FREE_TENSOR_MULTIPLICATION_H

#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>
#include <roughpy/device_support/device_core.h>
#include <roughpy/device_support/operators.h>

namespace rpy {
namespace algebra {
namespace kernels {

RPY_DEVICE dimn_t thread_block_size(deg_t dimn = 0) noexcept;
RPY_DEVICE dimn_t thread_index(deg_t dim = 0) noexcept;
RPY_DEVICE dimn_t group_size(deg_t dim = 0) noexcept;
RPY_DEVICE dimn_t group_index(deg_t dim = 0) noexcept;

template <typename Scalar, typename Op>
RPY_KERNEL void dense_free_tensor_fma(
        Slice<Scalar> out,
        Slice<const Scalar> left,
        Slice<const Scalar> right,
        Slice<const dimn_t> levels,
        Op&& op
)
{}

}// namespace kernels
}// namespace algebra
}// namespace rpy

#endif// DENSE_FREE_TENSOR_MULTIPLICATION_H
