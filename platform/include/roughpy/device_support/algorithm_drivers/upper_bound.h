#ifndef ROUGHPY_DEVICE_SUPPORT_UPPER_BOUND_H_
#define ROUGHPY_DEVICE_SUPPORT_UPPER_BOUND_H_

#include "algorithm_base.h"
#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace devices {
namespace __upper_bound {

template <typename S, typename T, bool Cond>
struct Func : dtl::AlgorithmFunctionBase<S, T, Cond> {
    optional<dimn_t>
    operator()(const Buffer& buffer, ConstReference value) const
    {
        const auto& bound_val = value.value<T>();
        auto buffer_view = buffer.map();
        auto buffer_slice = buffer_view.as_slice<S>();

        auto bound = std::upper_bound(
                buffer_slice.begin(),
                buffer_slice.end(),
                bound_val
        );

        if (bound != buffer_slice.end()) {
            return rpy::ranges::distance(buffer_slice.begin(), bound);
        }

        return {};
    }
};

template <typename S, typename T>
struct Func<S, T, false> : dtl::AlgorithmFunctionBase<S, T, false> {
    optional<dimn_t>
    operator()(const Buffer& buffer, ConstReference value) const
    {
        RPY_THROW(std::runtime_error, "bad");
    }
};

template <typename S, typename T>
using func
        = Func<S,
               T,
               is_less_comparable_v<S, T> && is_greater_comparable_v<S, T>>;
}// namespace __upper_bound

template <typename S, typename T>
constexpr __upper_bound::func<S, T> upper_bound_func{};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_UPPER_BOUND_H_
