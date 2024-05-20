#ifndef ROUGHPY_DEVICE_SUPPORT_REVERSE_H_
#define ROUGHPY_DEVICE_SUPPORT_REVERSE_H_

#include "algorithm_base.h"

#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace devices {

namespace __reverse {

template <typename S, typename T>
struct Func : dtl::AlgorithmFunctionBase<S, T, true> {
    void operator()(Buffer& buffer) const
    {
        auto buffer_view = buffer.map();
        auto buffer_slice = buffer_view.as_mut_slice<S>();
        ranges::reverse(buffer_slice);
    }
};

}// namespace __reverse

template <typename S, typename T>
constexpr __reverse::Func<S, T> reverse_func {};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_REVERSE_H_
