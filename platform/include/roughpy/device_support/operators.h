//
// Created by sam on 01/05/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_OPERATORS_H
#define ROUGHPY_DEVICE_SUPPORT_OPERATORS_H

#include <roughpy/core/macros.h>

#ifndef RPY_HOST
#  define RPY_HOST
#endif

#ifndef RPY_DEVICE
#  define RPY_DEVICE
#endif

#ifndef RPY_HOST_DEVICE
#  define RPY_HOST_DEVICE RPY_HOST RPY_DEVICE
#endif

namespace rpy {
namespace devices {
namespace operators {


template <typename T>
struct Identity
{
    static constexpr string_view name = "identity";

    RPY_NO_DISCARD RPY_HOST_DEVICE T operator()(const T& arg) const
    {
        return arg;
    }
};

template <typename T>
struct Uminus {

    static constexpr string_view name = "uminus";

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(const T& arg
    ) noexcept(noexcept(-arg))
    {
        return -arg;
    }
};

template <typename T>
struct Add {
    static constexpr string_view name = "addition";
    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T
    operator()(const T& left, const T& right) noexcept(noexcept(left + right))
    {
        return left + right;
    }
};

template <typename T>
struct Sub {
    static constexpr string_view name = "subtraction";
    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T
    operator()(const T& left, const T& right) noexcept(noexcept(left - right))
    {
        return left - right;
    }
};

template <typename T>
struct LeftScalarMultiply {
    const T& data;

    static constexpr string_view name = "left_scalar_multiply";
    constexpr explicit LeftScalarMultiply(const T& d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(const T& arg
    ) noexcept(noexcept(data * arg))
    {
        return data * arg;
    }
};

template <typename T>
struct RightScalarMultiply {
    const T& data;

    static constexpr string_view name = "right_scalar_multiply";
    constexpr explicit RightScalarMultiply(const T& d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(const T& arg
    ) noexcept(noexcept(arg * data))
    {
        return arg * data;
    }
};

template <typename T, typename S = T>
struct RightScalarDivide {
    const S& data;

    static constexpr string_view name = "right_scalar_divide";
    constexpr explicit RightScalarDivide(const S& divisor) : data(divisor) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(const T& arg
    ) noexcept(noexcept(arg / data))
    {
        return arg / data;
    }
};

template <typename T>
struct FusedLeftScalarMultiplyAdd {
    const T& data;

    static constexpr string_view name = "fused_add_scalar_left_mul";
    constexpr explicit FusedLeftScalarMultiplyAdd(const T& d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(
            const T& left,
            const T& right
    ) noexcept(noexcept(left + data * right))
    {
        return left + data * right;
    }
};

template <typename T>
struct FusedRightScalarMultiplyAdd {
    const T& data;

    static constexpr string_view name = "fused_add_scalar_right_mul";
    constexpr explicit FusedRightScalarMultiplyAdd(const T& d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(
            const T& left,
            const T& right
    ) noexcept(noexcept(left + right * data))
    {
        return left + right * data;
    }
};

template <typename T>
struct FusedLeftScalarMultiplySub {
    const T& data;

    static constexpr string_view name = "fused_sub_scalar_left_mul";
    constexpr explicit FusedLeftScalarMultiplySub(const T& d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(
            const T& left,
            const T& right
    ) noexcept(noexcept(left - data * right))
    {
        return left - data * right;
    }
};

template <typename T>
struct FusedRightScalarMultiplySub {
    const T& data;

    static constexpr string_view name = "fused_sub_scalar_right_mul";
    constexpr explicit FusedRightScalarMultiplySub(const T& d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(
            const T& left,
            const T& right
    ) noexcept(noexcept(left - right * data))
    {
        return left - right * data;
    }
};

template <typename T, typename S = T>
struct FusedRightScalarDivideAdd {
    const S& data;

    static constexpr string_view name = "fused_add_scalar_right_divide";
    constexpr explicit FusedRightScalarDivideAdd(const S& d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(
            const T& left,
            const T& right
    ) noexcept(noexcept(left + right / data))
    {
        return left + right / data;
    }
};

template <typename T, typename S = T>
struct FusedRightScalarDivideSub {
    const S& data;

    static constexpr string_view name = "fused_sub_scalar_right_divide";
    constexpr explicit FusedRightScalarDivideSub(const S& d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr T operator()(
            const T& left,
            const T& right
    ) noexcept(noexcept(left - right / data))
    {
        return left - right / data;
    }
};

}// namespace operators
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_OPERATORS_H
