#ifndef ROUGHPY_DEVICE_SUPPORT_SWAP_RANGES_H_
#define ROUGHPY_DEVICE_SUPPORT_SWAP_RANGES_H_

#include "algorithm_base.h"
#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace devices {

namespace __swap_ranges {

template <typename S, typename T>
struct Func : dtl::AlgorithmFunctionBase<S, T, false> {
    void operator()(Buffer& left, Buffer& right) const
    {
        RPY_THROW(
                std::runtime_error,
                "cannot swap ranges containing different types"
        );
    }
};

template <typename T>
struct Func<T, T> : dtl::AlgorithmFunctionBase<T, T, true> {
    void operator()(Buffer& left_buffer, Buffer& right_buffer) const
    {
        auto left_view = left_buffer.map();
        auto right_view = right_buffer.map();
        auto left_slice = left_view.as_mut_slice<T>();
        auto right_slice = right_view.as_mut_slice<T>();

        rpy::ranges::swap_ranges(left_slice, right_slice);
    }
};

}// namespace __swap_ranges

template <typename S, typename T>
constexpr __swap_ranges::Func<S, T> swap_ranges_func{};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_SWAP_RANGES_H_
