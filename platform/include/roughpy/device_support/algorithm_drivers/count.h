#ifndef ROUGHPY_DEVICE_SUPPORT_COUNT_H_
#define ROUGHPY_DEVICE_SUPPORT_COUNT_H_

#include "algorithm_base.h"

#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace devices {

namespace __count {
template <typename S, typename T, bool Cond = true>
struct Func : dtl::AlgorithmFunctionBase<S, T, Cond> {
    dimn_t operator()(const Buffer& buffer, ConstReference value) const
    {
        const auto view = buffer.map();
        const auto slice = view.as_slice<S>();
        return ranges::count(slice, value.value<T>());
    }
};

template <typename S, typename T>
struct Func<S, T, false> : dtl::AlgorithmFunctionBase<S, T, false> {
    RPY_NO_RETURN dimn_t operator()(const Buffer&, ConstReference) const
    {
        RPY_THROW(std::runtime_error, "");
    }
};

template <typename S, typename T>
using func = Func<S, T, is_equal_comparable_v<S, T>>;

}// namespace __count

template <typename S, typename T>
constexpr __count::func<S, T> count_func{};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_COUNT_H_
