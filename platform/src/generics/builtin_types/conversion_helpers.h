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


template <typename From, typename To>
struct ConversionHelper<From, To, enable_if_t<is_integral_v<From>&&is_integral_v<To>>>
{
    using from_ptr = const From*;
    using to_ptr = To*;

    static constexpr bool is_possible = true;
    // Exact if both have the same
    static constexpr bool is_always_exact = is_signed_v<From> == is_signed_v<To>
            && sizeof(From) >= sizeof(To);

    static bool check_fits_in_to_type(const From& src) noexcept
    {
        return compare_less_equal(std::numeric_limits<To>::min(), src) &&
                compare_less_equal(src, std::numeric_limits<To>::max());
    }

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        if constexpr (!is_always_exact) {
            if (ensure_exact && !check_fits_in_to_type(*src)) {
                return ConversionResult::Inexact;
            }
        }
        *dst = static_cast<To>(*src);
        return ConversionResult::Success;
    }
};


template <typename From, typename To>
struct ConversionHelper<From, To, enable_if_t<is_floating_point_v<From> &&
            is_floating_point_v<To>> >
{
    using from_ptr = const From*;
    using to_ptr = To*;

    static constexpr bool is_possible = true;
    // Exact if the From type has less or equal precision compared to the To type
    static constexpr bool is_always_exact = sizeof(From) <= sizeof(To);

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        *dst = static_cast<To>(*src);
        if constexpr (!is_always_exact) {
            if (ensure_exact && static_cast<From>(*dst) != *src) {
                return ConversionResult::Inexact;
            }
        }
        return ConversionResult::Success;
    }
};

template <typename From, typename To>
struct ConversionHelper<From, To, enable_if_t<is_integral_v<From> &&
            is_floating_point_v<To>> >
{
    using from_ptr = const From*;
    using to_ptr = To*;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact =
        std::numeric_limits<From>::digits <= std::numeric_limits<To>::digits;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        if constexpr (std::numeric_limits<To>::digits < std::numeric_limits<From>::digits) {
            From to_max = static_cast<From>(1) << std::numeric_limits<To>::digits;
            if (ensure_exact && *src > to_max) {
                return ConversionResult::Inexact;
            }
            if constexpr (std::is_signed_v<From>) {
                if (ensure_exact && *src < -to_max) {
                    return ConversionResult::Inexact;
                }
            }
        }
        *dst = static_cast<To>(*src);
        return ConversionResult::Success;
    }
};


template <typename From, typename To>
struct ConversionHelper<From, To, enable_if_t<is_floating_point_v<From> &&
            is_integral_v<To>> >
{
    using from_ptr = const From*;
    using to_ptr = To*;

    static constexpr bool is_possible = true;
    // Conversion to integer from floating point is almost never exact
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        From integral_part;
        auto fractional_part = std::modf(*src, &integral_part);

        if constexpr (std::numeric_limits<To>::digits > std::numeric_limits<From>::digits) {
            const auto to_max = static_cast<To>(std::numeric_limits<From>::max());
            const auto to_min = static_cast<To>(std::numeric_limits<From>::min());

            if (to_min < integral_part || integral_part > to_max) {
                return ConversionResult::Failed;
            }

        }

        if (ensure_exact && fractional_part != 0.0) {
            return ConversionResult::Inexact;
        }

        *dst = static_cast<To>(*src);
        return ConversionResult::Success;
    }
};




}

#endif //ROUGHPY_GENERICS_INTERNAL_BUILTIN_CONVERSION_HELPERS_H
