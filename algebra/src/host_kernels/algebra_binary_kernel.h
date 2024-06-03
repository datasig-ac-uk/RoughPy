//
// Created by sam on 03/06/24.
//

#ifndef ALGEBRA_BINARY_KERNEL_H
#define ALGEBRA_BINARY_KERNEL_H

#include "common.h"
#include <roughpy/core/ranges.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>
#include <roughpy/devices/device_handle.h>

namespace rpy {
namespace algebra {

namespace ops = devices::operators;

class MultiplicationKernelBase
{

protected:
    struct WorkPacket {
        dimn_t left_index;
        Slice<const dimn_t> right_indices;
        const void* coeffs;
    };

    devices::Buffer m_work;

    template <typename O, typename S, typename T, typename Op>
    static RPY_HOST_DEVICE void do_work_packets(
            Slice<const WorkPacket> packets,
            Slice<O> out,
            Slice<const S> left,
            Slice<const T> right,
            Op&& op
    )
    {
        auto fn = [&op, &left, &right](O& out_val, const WorkPacket& packet) {
            const auto& left_coeff = left[packet.left_index];
            for (const auto& right_coeff : right) {
                out_val += op(left_coeff * right_coeff);
            }
        };

        ranges::for_each(views::zip(out, packets), fn);
    }

    template <typename O, typename S, typename T>
    static RPY_HOST_DEVICE void fma(
            Slice<const WorkPacket> packets,
            Slice<O> out,
            Slice<const S> left,
            Slice<const T> right
        )
    {
        do_work_packets(packets, out, left, right, ops::Identity<O>());
    }

    template <typename O, typename S, typename T>
    static RPY_HOST_DEVICE void minus_fma(
            Slice<const WorkPacket> packets,
            Slice<O> out,
            Slice<const S> left,
            Slice<const T> right
        )
    {
        do_work_packets(packets, out, left, right, ops::Uminus<O>());
    }

    template <typename O, typename S, typename T>
    static RPY_HOST_DEVICE void post_multiply_fma(
            Slice<const WorkPacket> packets,
            Slice<O> out,
            Slice<const S> left,
            Slice<const T> right,
            const O& multiplier
        )
    {
        do_work_packets(packets, out, left, right, ops::RightScalarMultiply(multiplier));
    }



public:
};

class TriangularMultiplicationBase
{
};

}// namespace algebra
}// namespace rpy

#endif// ALGEBRA_BINARY_KERNEL_H
