#ifndef ROUGHPY_COMPUTE_COMMON_OPERATIONS_H
#define ROUGHPY_COMPUTE_COMMON_OPERATIONS_H


#include <functional>
#include <type_traits>

namespace rpy::compute::ops {


struct Identity {
    template <typename T>
    constexpr T operator()(T arg) const noexcept { return arg; }
};


using Negate = std::negate<>;


template <typename T>
struct LeftMultiplyBy {
    T factor_;

    constexpr LeftMultiplyBy(T factor)
            noexcept(std::is_nothrow_constructible_v<T, T>)
        : factor_(factor) {}

    constexpr T operator()(T arg) const noexcept(noexcept(factor_ * arg))
    { return factor_ * arg; }
};

template <typename T>
struct RightMultiplyBy {
    T factor_;

    constexpr RightMultiplyBy(T factor)
            noexcept(std::is_nothrow_constructible_v<T, T>)
        : factor_(factor) {}

    constexpr T operator()(T arg) const noexcept(noexcept(arg * factor_))
    { return arg * factor_; }
};

template <typename T>
using MultiplyBy = LeftMultiplyBy<T>;

template <typename T>
struct DivideBy : RightMultiplyBy<T> {
    template <typename S>
    constexpr DivideBy(S factor)
        noexcept(std::is_nothrow_constructible_v<RightMultiplyBy<T>, T>
            && noexcept(T{1} / factor))
        : RightMultiplyBy<T>(T{1} / factor) {}
};




} // namespace rpy::compute::ops

#endif //ROUGHPY_COMPUTE_COMMON_OPERATIONS_H
