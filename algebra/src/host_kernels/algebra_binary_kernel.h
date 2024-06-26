//
// Created by sam on 03/06/24.
//

#ifndef ALGEBRA_BINARY_KERNEL_H
#define ALGEBRA_BINARY_KERNEL_H

#include "common.h"
#include <roughpy/core/container/map.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <roughpy/algebra/vector.h>
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>
#include <roughpy/devices/device_handle.h>

namespace rpy {
namespace algebra {

namespace ops = devices::operators;

void ROUGHPY_ALGEBRA_EXPORT make_work_packages(
        devices::Buffer indices_buffer,
        devices::Buffer coeff_buffer,
        Vector& out,
        const Vector& left,
        const Vector& right
);

template <typename O, typename S, typename T>
class MultiplicationKernelBase
{

protected:
    struct WorkPacket {
        dimn_t left_index;
        Slice<const dimn_t> right_indices;
        const void* coeffs;
    };

    devices::Buffer m_iwork;
    devices::Buffer m_work;

    template <typename Op>
    static RPY_HOST_DEVICE void do_work_packets(
            Slice<const WorkPacket> packets,
            Slice<O> out,
            Slice<const S> left,
            Slice<const T> right,
            Op&& op
    )
    {
        for (dimn_t i = 0; i < packets.size(); ++i) {
            auto& out_val = out[i];
            const auto& packet = packets[i];
            const auto& left_coeff = left[packet.left_index];
            if (packet.coeffs != nullptr) {
                for (dimn_t j = 0; j < packet.right_indices.size(); ++j) {
                    const auto& right_coeff = right[j];
                    out_val += packet.coeffs[j] * op(left_coeff * right_coeff);
                }
            } else {
                for (dimn_t j = 0; j < packet.right_indices.size(); ++j) {
                    const auto& right_coeff = right[j];
                    out_val += op(left_coeff * right_coeff);
                }
            }
        }
    }

    static RPY_HOST_DEVICE void
    fma(Slice<const WorkPacket> packets,
        Slice<O> out,
        Slice<const S> left,
        Slice<const T> right)
    {
        do_work_packets(packets, out, left, right, ops::Identity<O>());
    }

    static RPY_HOST_DEVICE void minus_fma(
            Slice<const WorkPacket> packets,
            Slice<O> out,
            Slice<const S> left,
            Slice<const T> right
    )
    {
        do_work_packets(packets, out, left, right, ops::Uminus<O>());
    }

    static RPY_HOST_DEVICE void post_multiply_fma(
            Slice<const WorkPacket> packets,
            Slice<O> out,
            Slice<const S> left,
            Slice<const T> right,
            const O& multiplier
    )
    {
        do_work_packets(
                packets,
                out,
                left,
                right,
                ops::RightScalarMultiply(multiplier)
        );
    }

public:
};


struct MultiplicationFrame {
    dimn_t left_index;
    dimn_t right_index_begin;
    dimn_t right_index_end;
    dimn_t coeffs_index_begin = 0;
};

struct MultiplicationFrameHeader {
    dimn_t num_frames;
    MultiplicationFrame first_frame[1];
};

using KeyProductFunction
        = std::function<Vector(const BasisKey&, const BasisKey&)>;
}// namespace algebra
}// namespace rpy

#endif// ALGEBRA_BINARY_KERNEL_H
