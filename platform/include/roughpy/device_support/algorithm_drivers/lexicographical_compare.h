#ifndef ROUGHPY_DEVICE_SUPPORT_LEXICOGRAPHICAL_COMPARE_H_
#define ROUGHPY_DEVICE_SUPPORT_LEXICOGRAPHICAL_COMPARE_H_

#include "algorithm_base.h"
#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace devices {

namespace __lexicographical_compare {

template <typename S, typename T, bool Cond>
struct Func : dtl::AlgorithmFunctionBase<S, T, Cond> {
    bool operator()(const Buffer& left_buffer, const Buffer& right_buffer) const
    {
        auto left_view = left_buffer.map();
        auto right_view = right_buffer.map();
        auto left_slice = left_view.as_slice<S>();
        auto right_slice = right_view.as_slice<S>();

        return ranges::lexicographical_compare(left_slice, right_slice);
    }
};

template <typename S, typename T>
struct Func<S, T, false> : dtl::AlgorithmFunctionBase<S, T, false> {
    bool operator()(const Buffer& left_buffer, const Buffer& right_buffer) const
    {
        RPY_THROW(std::runtime_error, "bad");
    }
};

template <typename S, typename T>
using func
        = Func<S,
               T,
               is_less_comparable_v<S, T> && is_less_equal_comparable_v<S, T>
                       && is_equal_comparable_v<S, T>>;

}// namespace __lexicographical_compare

template <typename S, typename T>
constexpr __lexicographical_compare::func<S, T> lexicographical_compare_func{};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_LEXICOGRAPHICAL_COMPARE_H_
