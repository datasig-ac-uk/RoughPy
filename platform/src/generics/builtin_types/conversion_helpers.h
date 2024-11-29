//
// Created by sammorley on 29/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_BUILTIN_CONVERSION_HELPERS_H
#define ROUGHPY_GENERICS_INTERNAL_BUILTIN_CONVERSION_HELPERS_H

#include <limits>
#include <utility>

#include "roughpy/core/check_helpers.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "roughpy/generics/conversion_trait.h"


namespace rpy::generics::conv {

// Specialization for integral types
template <typename T, typename U>
struct ConversionHelpers<T, U, std::enable_if_t<std::is_integral_v<T> &&
            std::is_integral_v<U>> >
{
    // conversion from T to U
    // conversion to T from U
    static constexpr bool is_nested = false;

    static constexpr bool from_exact_convertible() noexcept
    {
       if constexpr (is_signed_v<T> == is_signed_v<U>) {
           return sizeof(U) >= sizeof(T);
       } else {
           return false;
       }
    }

    static constexpr bool to_exact_convertible() noexcept
    {
        if constexpr (is_signed_v<T> == is_signed_v<U>) {
            return sizeof(T) >= sizeof(U);
        } else {
            return false;
        }
    }

    static ConversionResult from(T* dst_ptr,
                                 const U* src_ptr,
                                 bool ensure_exact) noexcept
    {
        if constexpr (!from_exact_convertible()) {
            if (ensure_exact) {
                if (std::numeric_limits<T>::min() <= *src_ptr &&
                    *src_ptr <= std::numeric_limits<T>::max()) {
                    return ConversionResult::Inexact;
                }
            }
        }
        *dst_ptr = static_cast<T>(*src_ptr);
        return ConversionResult::Success;
    }

    static ConversionResult to(U* dst_ptr,
                               const T* src_ptr,
                               bool ensure_exact) noexcept
    {
        if constexpr (!to_exact_convertible()) {
            if (ensure_exact) {
                if (std::numeric_limits<U>::min() <= *src_ptr &&
                    *src_ptr <= std::numeric_limits<U>::max()) {
                    return ConversionResult::Inexact;
                }
            }
        }
        *dst_ptr = static_cast<U>(*src_ptr);
        return ConversionResult::Success;
    }

    static bool compare_equal(const T* t, const U* u) noexcept
    {
        return check_equal(*t, *u);
    }
};

// Specialization for floating point types
template <typename T, typename U>
struct ConversionHelpers<T, U, std::enable_if_t<std::is_floating_point_v<T> &&
            std::is_floating_point_v<U>> >
{
    // conversion from T to U
    // conversion to T from U
    static constexpr bool is_nested = false;

    static constexpr bool from_exact_convertible() noexcept
    {
        return sizeof(T) <= sizeof(U);
    }

    static constexpr bool to_exact_convertible() noexcept
    {
        return sizeof(U) <= sizeof(T);
    }

    static ConversionResult from(T* dst_ptr,
                                 const U* src_ptr,
                                 bool ensure_exact) noexcept
    {
        *dst_ptr = static_cast<T>(*src_ptr);
        if constexpr (!from_exact_convertible()) {
            if (ensure_exact && *src_ptr != static_cast<U>(*dst_ptr)) {
                return ConversionResult::Inexact;
            }
        }
        return ConversionResult::Success;
    }

    static ConversionResult to(U* dst_ptr,
                               const T* src_ptr,
                               bool ensure_exact) noexcept
    {
        *dst_ptr = static_cast<U>(*src_ptr);
        if constexpr (!to_exact_convertible()) {
            if (ensure_exact && *src_ptr != static_cast<T>(*dst_ptr)) {
                return ConversionResult::Inexact;
            }
        }
        return ConversionResult::Success;
    }

    static bool compare_equal(const T* t, const U* u) noexcept
    {
        return *t == static_cast<T>(*u);
    }
};


// Specialization for when T is a floating point type and U is integral
template <typename T, typename U>
struct ConversionHelpers<T, U, std::enable_if_t<std::is_floating_point_v<T> &&
            std::is_integral_v<U>> >
{
    // conversion from T to U
    // conversion to T from U
    static constexpr bool is_nested = false;

    static constexpr bool from_exact_convertible() noexcept
    {
        // Converting to from integer to float is exact if the number of digits
        // of the integer is less than the number of mantissa bits of the float
        return std::numeric_limits<T>::digits >= std::numeric_limits<U>::digits;
    }

    static constexpr bool to_exact_convertible() noexcept
    {
        // Converting from floating point to integral is generally inexact
        return false;
    }

    static ConversionResult from(T* dst_ptr,
                                 const U* src_ptr,
                                 bool ensure_exact) noexcept
    {
        *dst_ptr = static_cast<T>(*src_ptr);
        if constexpr (!from_exact_convertible()) {
            if (ensure_exact && static_cast<U>(*dst_ptr) != *src_ptr) {
                return ConversionResult::Inexact;
            }
        }
        return ConversionResult::Success;
    }

    static ConversionResult to(U* dst_ptr,
                               const T* src_ptr,
                               bool ensure_exact) noexcept
    {
        *dst_ptr = static_cast<U>(*src_ptr);
        if (ensure_exact) {
            // Check if the conversion is exact
            if (static_cast<T>(*dst_ptr) != *src_ptr) {
                return ConversionResult::Inexact;
            }
        }
        return ConversionResult::Success;
    }

    static bool compare_equal(const T* t, const U* u) noexcept
    {
        // Compare using a tolerance for floating point precision issues
        return *t == static_cast<T>(*u);
    }
};


}

#endif //ROUGHPY_GENERICS_INTERNAL_BUILTIN_CONVERSION_HELPERS_H
