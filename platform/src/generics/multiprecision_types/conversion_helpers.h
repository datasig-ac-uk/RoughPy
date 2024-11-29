//
// Created by sammorley on 29/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_CONVERSION_HELPERS_H
#define ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_CONVERSION_HELPERS_H


#include <limits>
#include <utility>

#include <gmp.h>
#include <mpfr.h>

#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "generics/builtin_types/conversion_helpers.h"

#include "multiprecision_type_ids.h"

namespace rpy::generics::conv {

template <typename I, bool Nested>
struct ConversionHelpers<MPInt, I, Nested, enable_if_t<is_integral_v<I> > >
{

    static constexpr bool is_nested = Nested;

    static constexpr bool from_exact_convertible() noexcept { return true; }

    static constexpr bool to_exact_convertible() noexcept { return false; }


    static ConversionResult from(mpz_ptr dst_ptr,
                                 const I* src_ptr,
                                 bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);
        if constexpr (is_signed_v<I>) { mpz_set_si(dst_ptr, *src_ptr); } else {
            mpz_set_ui(dst_ptr, *src_ptr);
        }
        return ConversionResult::Success;
    }

    static ConversionResult to(I* dst_ptr,
                               mpz_ptr src_ptr,
                               bool ensure_exact) noexcept
    {
        if constexpr (is_signed_v<I>) {
            auto val = mpz_get_si(src_ptr);
            return ConversionHelpers<decltype(val),
                I>::to(dst_ptr, &val, ensure_exact);
        } else {
            auto val = mpz_get_ui(src_ptr);
            return ConversionHelpers<decltype(val), I>::to(
                dst_ptr,
                &val,
                ensure_exact);
        }
    }

};


template <typename F, bool Nested>
struct ConversionHelpers<MPInt, F, Nested, enable_if_t<is_floating_point_v<F> > >
{
    // conversion from T to U
    // conversion to T from U
    static constexpr bool is_nested = Nested;

    static constexpr bool from_exact_convertible() noexcept { return false; }

    static constexpr bool to_exact_convertible() noexcept { return false; }

    static ConversionResult from(mpz_ptr dst_ptr,
                                 const F* src_ptr,
                                 bool ensure_exact) noexcept
    {
        if (ensure_exact) {
            int64_t tmp;
            auto first_result = ConversionHelpers<int64_t, F>::from(
                &tmp,
                src_ptr,
                ensure_exact);
            if (first_result != ConversionResult::Success) {
                return first_result;
            }
            return ConversionHelpers<MPInt, int64_t>::from(
                dst_ptr,
                &tmp,
                ensure_exact);
        }

        mpz_set_d(dst_ptr, *src_ptr);
        return ConversionResult::Success;
    }

    static ConversionResult to(F* dst_ptr,
                               mpz_ptr src_ptr,
                               bool ensure_exact) noexcept
    {
        constexpr int64_t F_max = (static_cast<int64_t>(1) <<
            std::numeric_limits<F>::digits) - 1;

        if (ensure_exact) {
            if (mpz_cmp_si(src_ptr, F_max) > 0) {
                return ConversionResult::Inexact;
            }
        }
        auto val = mpz_get_d(src_ptr);
        return ConversionHelpers<F, decltype(val)>::to(
            dst_ptr,
            &val,
            ensure_exact);
    }

    static bool compare_equal(mpz_srcptr lhs, const F* rhs) noexcept
    {
        return mpz_cmp_d(lhs, static_cast<double>(*rhs)) == 0;
    }

};


template <typename I, bool Nested>
struct ConversionHelpers<MPRational, I, Nested, enable_if_t<is_integral_v<I> > >
{
    // conversion from T to U
    // conversion to T from U
    static constexpr bool is_nested = false;

    static constexpr bool from_exact_convertible() noexcept { return true; }

    static constexpr bool to_exact_convertible() noexcept { return false; }

    static ConversionResult from(mpq_ptr dst_ptr,
                                 const I* src_ptr,
                                 bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);
        mpq_set_si(dst_ptr, *src_ptr, 1);
        return ConversionResult::Success;
    }

    static ConversionResult to(I* dst_ptr,
                               mpq_ptr src_ptr,
                               bool ensure_exact) noexcept
    {
        if (mpz_cmp_ui(mpq_denref(src_ptr), 1UL)) {
            return ConversionResult::Failed;
        }

        constexpr std::make_unsigned_t<I> I_max = std::numeric_limits<I>::max();

        if (ensure_exact && mpz_cmpabs_ui(mpq_numref(src_ptr), I_max)) {
            return ConversionResult::Inexact;
        }

        if constexpr (is_signed_v<I>) {
            auto val = mpz_get_si(mpq_numref(src_ptr));
            return ConversionHelpers<decltype(val),
                I>::to(dst_ptr, &val, ensure_exact);
        } else {
            if (mpq_sgn(src_ptr) < 0) { return ConversionResult::Failed; }
            auto val = mpz_get_ui(mpq_numref(src_ptr));
            return ConversionHelpers<decltype(val), I>::to(
                dst_ptr,
                &val,
                ensure_exact);
        }
    }

    static bool compare_equal(mpq_srcptr lhs, const I* rhs) noexcept
    {
        if (mpz_cmp_ui(mpq_denref(lhs), 1UL) != 0) { return false; }
        if constexpr (is_signed_v<I>) {
            return mpz_cmp_si(mpq_numref(lhs), *rhs) == 0;
        } else { return mpz_cmp_ui(mpq_numref(lhs), *rhs) == 0; }
    }
};


template <typename F, bool Nested>
struct ConverisonHelpers<MPRational, F, Nested, enable_if_t<is_floating_point_v<F> > >
{
    // conversion from T to U
    // conversion to T from U
    static constexpr bool is_nested = Nested;

    static constexpr bool from_exact_convertible() noexcept { return true; }

    static constexpr bool to_exact_convertible() noexcept { return false; }

    static ConversionResult from(mpq_ptr dst_ptr,
                                 const F* src_ptr,
                                 bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);
        mpq_set_d(dst_ptr, *src_ptr);
        return ConversionResult::Success;
    }

    static ConversionResult to(F* dst_ptr,
                               mpq_ptr src_ptr,
                               bool ensure_exact) noexcept
    {
        if (ensure_exact) {
            // It's basically impossible to convert faithfully from mpq_t to
            // floating point value
            return ConversionResult::Inexact;
        }
        auto val = mpq_get_d(src_ptr);
        return ConversionHelpers<F, decltype(val)>::to(
            dst_ptr,
            &val,
            ensure_exact);
    }

    static bool compare_equal(mpq_srcptr lhs, const F* rhs) noexcept
    {
        return false;
    }
};


template <bool Nested>
struct ConversionHelpers<MPRational, MPInt, Nested, void>
{
    static constexpr bool is_nested = Nested;
    static constexpr bool from_exact_convertible() noexcept { return true; }
    static constexpr bool to_exact_convertible() noexcept { return true; }

    static ConversionResult from(mpq_ptr dst_ptr,
                                 mpz_srcptr src_ptr,
                                 bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);
        mpq_set_z(dst_ptr, src_ptr);
        return ConversionResult::Success;
    }

    static ConversionResult to(mpz_ptr dst_ptr,
                               mpq_srcptr src_ptr,
                               bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);
        if (mpz_cmp_ui(mpq_denref(src_ptr), 1UL)) {
            return ConversionResult::Failed;
        }
        mpz_set(dst_ptr, mpq_numref(src_ptr));

        return ConversionResult::Success;
    }

    static bool compare_equal(mpq_srcptr lhs, mpz_srcptr rhs) noexcept
    {
        if (mpz_cmp_ui(mpq_denref(lhs), 1UL) != 0) { return false; }
        return mpz_cmp(mpq_numref(lhs), rhs) == 0;
    }
};


template <typename I>
struct ConversionHelpers<MPFloat, I, enable_if_t<is_integral_v<I> > >
{
    static constexpr bool is_nested = false;
    static constexpr bool from_exact_convertible() noexcept { return true; }
    static constexpr bool to_exact_convertible() noexcept { return false; }

    static ConversionResult from(mpfr_ptr dst_ptr,
                                 const I* src_ptr,
                                 bool ensure_exact) noexcept
    {
        if (ensure_exact) {
            if (mpfr_fits_si_p(dst_ptr, *src_ptr, MPFR_RNDN) != 0) {
                return ConversionResult::Inexact;
            }
        }
        mpfr_set_si(dst_ptr, *src_ptr);
        return ConversionResult::Success;
    }

    static ConversionResult to(I* dst_ptr,
                               mpfr_ptr src_ptr,
                               bool ensure_exact) noexcept
    {
        auto val = mpfr_get_si(src_ptr, MPFR_RNDN);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~MPFR_FLAGS_INEXACT) { return ConversionResult::Failed; }
        if (ensure_exact && t & MPFR_FLAGS_INEXACT) {
            return ConversionResult::Inexact;
        }

        return ConversionHelpers<decltype(val),
            I>::to(dst_ptr, &val, ensure_exact);

    }

    static bool compare_equal(mpfr_srcptr lhs, const I* rhs) noexcept
    {
        return mpfr_cmp_si(lhs, *rhs) == 0;
    }
};


template <typename F>
struct ConversionHelpers<MPFloat, F, enable_if_t<is_floating_point_v<F> > >
{
    static constexpr bool is_nested = false;
    static constexpr bool from_exact_convertible() noexcept { return true; }
    static constexpr bool to_exact_convertible() noexcept { return false; }

    static ConversionResult from(mpfr_ptr dst_ptr,
                                 const F* src_ptr,
                                 bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);

        mpfr_set_d(dst_ptr, *src_ptr, MPFR_RNDN);
        return ConversionResult::Success;
    }

    static ConversionResult to(F* dst_ptr,
                               mpfr_ptr src_ptr,
                               bool ensure_exact) noexcept
    {
        if constexpr (is_same_v<F, float>) {
            *dst_ptr = mpfr_get_flt(src_ptr, MPFR_RNDN);
        } else { *dst_ptr = mpfr_get_d(src_ptr, MPFR_RNDN); }

        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }

    static bool compare_equal(mpfr_srcptr lhs, const F* rhs) noexcept
    {
        return mpfr_cmp_d(lhs, *rhs) == 0;
    }
};


template <>
struct ConversionHelpers<MPFloat, MPInt, void>
{


    static constexpr bool is_nested = false;

    static constexpr bool from_exact_convertible() noexcept { return false; }
    static constexpr bool to_exact_convertible() noexcept { return false; }


    static ConversionResult from(mpfr_ptr dst_ptr,
                                 mpz_srcptr src_ptr,
                                 bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);
        mpfr_set_z(dst_ptr, src_ptr, MPFR_RNDN);

        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }

    static ConversionResult to(mpz_ptr dst_ptr,
                               mpfr_srcptr src_ptr,
                               bool ensure_exact) noexcept
    {
        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        mpfr_get_z(dst_ptr, src_ptr, MPFR_RNDN);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }

    static bool compare_equal(mpfr_srcptr lhs, mpz_srcptr rhs) noexcept
    {
        return mpfr_cmp_z(lhs, rhs) == 0;
    }
};

template <>
struct ConversionHelpers<MPFloat, MPRational, void>
{
    static constexpr bool is_nested = false;

    static constexpr bool from_exact_convertible() noexcept { return false; }
    static constexpr bool to_exact_convertible() noexcept { return true; }


    static ConversionResult from(mpfr_ptr dst_ptr,
                                 mpq_srcptr src_ptr,
                                 bool ensure_exact) noexcept
    {
        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        mpfr_set_q(dst_ptr, src_ptr, MPFR_RNDN);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }

    static ConversionResult to(mpq_ptr dst_ptr,
                               mpfr_srcptr src_ptr,
                               bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);
        mpfr_get_q(dst_ptr, src_ptr);
        return ConversionResult::Success;
    }

    static bool compare_equal(mpfr_srcptr lhs, mpq_srcptr rhs) noexcept
    {
        return mpfr_cmp_q(lhs, rhs) == 0;
    }
};

}

#endif //ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_CONVERSION_HELPERS_H