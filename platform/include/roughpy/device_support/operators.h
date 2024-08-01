//
// Created by sam on 01/05/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_OPERATORS_H
#define ROUGHPY_DEVICE_SUPPORT_OPERATORS_H

#include "macros.h"

namespace rpy {
namespace devices {

class Value;
class ConstReference;
class Reference;

namespace operators {

template <typename T>
struct ArgumentTraits {
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
};

template <>
struct ArgumentTraits<Value> {
    using value_type = Value;
    using reference = Reference;
    using const_reference = ConstReference;
};

template <typename T>
struct Identity {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    static constexpr string_view name = "identity";

    RPY_NO_DISCARD RPY_HOST_DEVICE value_type operator()(const_reference arg
    ) const
    {
        return arg;
    }
};

template <typename T>
struct Uminus {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    static constexpr string_view name = "uminus";

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type
    operator()(const_reference arg) noexcept(noexcept(-arg))
    {
        return -arg;
    }
};

template <typename T>
struct Add {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    static constexpr string_view name = "addition";

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left + right))
    {
        return left + right;
    }
};

template <typename T>
struct Sub {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    static constexpr string_view name = "subtraction";

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left - right))
    {
        return left - right;
    }
};

template <typename T>
struct LeftScalarMultiply {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    const_reference data;

    static constexpr string_view name = "left_scalar_multiply";
    constexpr explicit LeftScalarMultiply(const_reference d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type
    operator()(const_reference arg) noexcept(noexcept(data * arg))
    {
        return data * arg;
    }
};

template <typename T>
struct RightScalarMultiply {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    const_reference data;

    static constexpr string_view name = "right_scalar_multiply";
    constexpr explicit RightScalarMultiply(const_reference d) : data(d) {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type
    operator()(const_reference arg) noexcept(noexcept(arg * data))
    {
        return arg * data;
    }
};

template <typename T, typename S = T>
struct RightScalarDivide {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    typename ArgumentTraits<S>::const_reference data;

    static constexpr string_view name = "right_scalar_divide";
    constexpr explicit
    RightScalarDivide(typename ArgumentTraits<S>::const_reference divisor)
        : data(divisor)
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type
    operator()(const_reference arg) noexcept(noexcept(arg / data))
    {
        return arg / data;
    }
};

template <typename T>
struct FusedLeftScalarMultiplyAdd {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;
    const_reference data;

    static constexpr string_view name = "fused_add_scalar_left_mul";
    constexpr explicit FusedLeftScalarMultiplyAdd(const_reference d) : data(d)
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left + data * right))
    {
        return left + data * right;
    }
};

template <typename T>
struct FusedRightScalarMultiplyAdd {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;
    const_reference data;

    static constexpr string_view name = "fused_add_scalar_right_mul";
    constexpr explicit FusedRightScalarMultiplyAdd(const_reference d) : data(d)
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left + right * data))
    {
        return left + right * data;
    }
};

template <typename T>
struct FusedLeftScalarMultiplySub {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;
    const_reference data;

    static constexpr string_view name = "fused_sub_scalar_left_mul";
    constexpr explicit FusedLeftScalarMultiplySub(const_reference d) : data(d)
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left - data * right))
    {
        return left - data * right;
    }
};

template <typename T>
struct FusedRightScalarMultiplySub {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;
    const_reference data;

    static constexpr string_view name = "fused_sub_scalar_right_mul";
    constexpr explicit FusedRightScalarMultiplySub(const_reference d) : data(d)
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left - right * data))
    {
        return left - right * data;
    }
};

template <typename T, typename S = T>
struct FusedRightScalarDivideAdd {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    typename ArgumentTraits<S>::const_reference data;

    static constexpr string_view name = "fused_add_scalar_right_divide";
    constexpr explicit
    FusedRightScalarDivideAdd(typename ArgumentTraits<S>::const_reference d)
        : data(d)
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left + right / data))
    {
        return left + right / data;
    }
};

template <typename T, typename S = T>
struct FusedRightScalarDivideSub {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;
    typename ArgumentTraits<S>::const_reference data;

    static constexpr string_view name = "fused_sub_scalar_right_divide";
    constexpr explicit
    FusedRightScalarDivideSub(typename ArgumentTraits<S>::const_reference d)
        : data(d)
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left - right / data))
    {
        return left - right / data;
    }
};

}// namespace operators
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_OPERATORS_H
