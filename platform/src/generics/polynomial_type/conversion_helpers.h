//
// Created by sammorley on 02/12/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_CONVERSION_HELPERS_H
#define ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_CONVERSION_HELPERS_H

#include "polynomial.h"
#include "generics/builtin_types/conversion_helpers.h"
#include "generics/multiprecision_types/conversion_helpers.h"

namespace rpy::generics::conv {


template <typename From>
struct ConversionHelper<From, Polynomial, void>
{
    using from_ptr = const From*;
    using to_ptr = Polynomial*;

    using helper = ConversionHelper<From, MPRational>;

    static constexpr bool is_possible = helper::is_possible;
    static constexpr bool is_always_exact = helper::is_always_exact;


    static ConversionResult convert(to_ptr to, from_ptr from, bool ensure_exact) noexcept
    {
        dtl::RationalCoeff rational;
        auto result = helper::convert(rational.content, from, ensure_exact);
        if (result == ConversionResult::Failed) {
            return result;
        } if (ensure_exact && result == ConversionResult::Inexact) {
            return result;
        }

        try {
            *to = Polynomial(std::move(rational));
        } catch (std::exception&) {
            return ConversionResult::Failed;
        }
        return ConversionResult::Success;
    }
};

template <typename To>
struct ConversionHelper<Polynomial, To, void>
{
    using from_ptr = const Polynomial*;
    using to_ptr = To*;

    using helper = ConversionHelper<MPRational, To>;
    static constexpr bool is_possible = helper::is_possible;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr to, from_ptr from, bool ensure_exact) noexcept
    {
        if (from->empty()) {
            int8_t zero = 0;
            return ConversionHelper<int8_t, To>::convert(to, &zero, ensure_exact);
        }

        if (!from->is_constant()) {
            return ConversionResult::Failed;
        }

        const auto* constant = from->begin()->second.content;

        return helper::convert(to, constant, ensure_exact);
    }
};



}

#endif //ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_CONVERSION_HELPERS_H
