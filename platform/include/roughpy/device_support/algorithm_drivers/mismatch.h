#ifndef ROUGHPY_DEVICE_SUPPORT_MISMATCH_H_
#define ROUGHPY_DEVICE_SUPPORT_MISMATCH_H_

#include "algorithm_base.h"

#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace devices {

namespace __mismatch {

template <typename S, typename T, bool Cond>
struct Func : dtl::AlgorithmFunctionBase<S, T, Cond> {
    optional<dimn_t>
    operator()(const Buffer& left_buffer, const Buffer& right_buffer) const
    {
        const auto left_view = left_buffer.map();
        const auto right_view = right_buffer.map();
        const auto left_slice = left_view.as_slice<S>();
        const auto right_slice = right_view.as_slice<S>();

        auto mismatch = ranges::mismatch(left_slice, right_slice);

        if (mismatch.in1 != left_slice.end()) {
            return static_cast<dimn_t>(mismatch.in1 - left_slice.begin());
        }

        if (mismatch.in2 != right_slice.end()) {
            return static_cast<dimn_t>(mismatch.in2 - right_slice.begin());
        }

        return {};
    }
};


template <typename S, typename T>
using func = Func<S, T, is_equal_comparable_v<S, T>>;


}// namespace __mismatch

template <typename S, typename T>
inline constexpr __mismatch::func<S, T> mismatch_func {};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_MISMATCH_H_
