//
// Created by sam on 09/08/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_DEFAULT_FILL_H
#define ROUGHPY_DEVICE_SUPPORT_DEFAULT_FILL_H

namespace rpy {
namespace devices {

namespace default_fill {

template <typename T, bool Cond>
struct Func : dtl::AlgorithmFunctionBase<T, T, Cond> {
    void operator()(Buffer& buffer) const
    {
        auto slice = buffer.as_mut_slice<T>();
        std::uninitialized_default_construct_n(slice.data(), slice.size());
    }
};

template <typename T>
using func = Func<T, is_default_constructible_v<T>>;

}// namespace default_fill

template <typename T>
inline constexpr default_fill::func<T> default_fill_func;

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_DEFAULT_FILL_H
