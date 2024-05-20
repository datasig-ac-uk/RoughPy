//
// Created by sam on 5/18/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_ROTATE_H
#define ROUGHPY_DEVICE_SUPPORT_ROTATE_H

#include "algorithm_base.h"

#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace devices {

namespace __rotate {

template <typename T>
struct Func : dtl::AlgorithmFunctionBase<T, T, true> {
    void operator()(Buffer& buffer, dimn_t middle) const
    {
        auto buffer_view = buffer.map();
        auto buffer_slice = buffer_view.as_slice<T>();
        ranges::rotate(buffer_slice, buffer_slice.begin() + middle);
    }
};

}// namespace __rotate

template <typename S, typename T>
constexpr __rotate::Func<S> rotate_func {};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_ROTATE_H
