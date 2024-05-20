#ifndef ROUGHPY_DEVICE_SUPPORT_FILL_H_
#define ROUGHPY_DEVICE_SUPPORT_FILL_H_

namespace rpy {
namespace devices {

namespace __fill {

template <typename S, typename T, bool Cond>
struct Func : dtl::AlgorithmFunctionBase<S, T, Cond> {
    void operator()(Buffer& destination_buffer, ConstReference value) const
    {
        const auto& fill_val = value.value<T>();
        auto destination_view = destination_buffer.map();
        auto destination_slice = destination_view.as_mut_slice<S>();

        ranges::fill(destination_slice, fill_val);
    }
};

template <typename S, typename T>
struct Func<S, T, false> : dtl::AlgorithmFunctionBase<S, T, false> {
    void operator()(Buffer& destination_buffer, ConstReference value) const
    {
        RPY_THROW(std::runtime_error, "bad");
    }
};

template <typename S, typename T>
using func = Func<S, T, is_constructible_v<S, const T&>>;

}// namespace __fill

template <typename S, typename T>
constexpr __fill::func<S, T> fill_func{};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_FILL_H_
