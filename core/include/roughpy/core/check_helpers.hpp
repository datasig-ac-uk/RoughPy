//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_CORE_CHECK_HELPERS_H
#define ROUGHPY_CORE_CHECK_HELPERS_H

// Rather than using the traits header from RoughPy, we're using the type_traits
// standard header directly, because we don't want to have any cyclic
// dependencies
#include <type_traits>
#include <utility>

namespace rpy {

// C++20 adds helpful functions for comparing integers, but we don't necessarily
// have access to those, so reimplement them here. These are just the possible
// implementations lifted directly from cppreference.com.
// We don't use the boost safe_numerics because it depends on mp11, which we
// we absolutely want to avoid.

template <typename T, typename U>
constexpr
std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>, bool>
compare_equal(T t, U u) noexcept {
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {
        return t == u;
    } else if constexpr (std::is_signed_v<T>) {
        return t >= 0 && std::make_unsigned_t<T>(t) == u;
    } else {
        return u >= 0 && std::make_unsigned_t<U>(u) == t;
    }
}


template <typename T, typename U>
constexpr
std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>, bool>
compare_less(T t, U u) noexcept {
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {
        return t < u;
    } else if constexpr (std::is_signed_v<T>) {
        return t < 0 || std::make_unsigned_t<T>(t) < u;
    } else {
        return u >= 0 && t < std::make_unsigned_t<U>(u);
    }
}

template <typename T, typename U>
constexpr
std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>, bool>
compare_less_equal(T t, U u) noexcept {
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {
        return t <= u;
    } else if constexpr (std::is_signed_v<T>) {
        return t < 0 || std::make_unsigned_t<T>(t) <= u;
    } else {
        return u >= 0 && t <= std::make_unsigned_t<U>(u);
    }
}

template <typename T, typename U>
constexpr
std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>, bool>
compare_greater(T t, U u) noexcept {
    return !compare_less_equal(t, u);
}

template <typename T, typename U>
constexpr
std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>, bool>
compare_greater_equal(T t, U u) noexcept {
    return !compare_less(t, u);
}




// If we want to use these safe functions for integers, we need "safe" functions
// for other comparisons too

template <typename T, typename U>
constexpr
std::enable_if_t<!std::is_integral_v<T> || !std::is_integral_v<U>, bool>
compare_equal(const T& t, const U& u) noexcept(noexcept(t == u))
{
    return t == u;
}

// Compare less and compare greater might carry different semantic meaning for
// some non-trvial types
template <typename T, typename U>
constexpr
std::enable_if_t<!std::is_integral_v<T> || !std::is_integral_v<U>, bool>
compare_less(const T& t, const U& u) noexcept(noexcept(t < u))
{
    return t < u;
}

template <typename T, typename U>
constexpr
std::enable_if_t<!std::is_integral_v<T> || !std::is_integral_v<U>, bool>
compare_less_equal(const T& t, const U& u) noexcept(noexcept(t <= u))
{
    return t <= u;
}

template <typename T, typename U>
constexpr
std::enable_if_t<!std::is_integral_v<T> || !std::is_integral_v<U>, bool>
compare_greater(const T& t, const U& u) noexcept(noexcept(t > u))
{
    return t > u;
}

template <typename T, typename U>
constexpr
std::enable_if_t<!std::is_integral_v<T> || !std::is_integral_v<U>, bool>
compare_greater_equal(const T& t, const U& u) noexcept(noexcept(t >= u))
{
    return t >= u;
}

// It's always a safe assumption that "is not equal" is synonymous with not
// "is equal"
template <typename T, typename U>
constexpr bool
compare_not_equal(const T& t, const U& u) noexcept {
    return !compare_equal(t, u);
}

}


#endif //ROUGHPY_CORE_DETAIL_CHECK_HELPERS_H
